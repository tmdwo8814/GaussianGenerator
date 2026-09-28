"""RoMaV2 supervises the student's actual queries and its fitted Sim(3).

One context pair per rank/step. No target images/poses or additional rendering.
RoMa is a plain Python-owned frozen module, absent from checkpoints/optimizers.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..encoder.common.support_alignment import gather, grid_indices, masked_mean
from ..encoder.common.weighted_sim3 import scene_radius


@dataclass
class DescriptorTeacherCfg:
    enabled: bool = False
    setting: str = 'fast'
    every_n_steps: int = 1
    warmup_steps: int = 1000
    match_weight: float = .05
    alignment_weight: float = .05
    min_confidence: float = .5
    correctness_pixels: float = 2.0
    log_every_n_steps: int = 50

    def __post_init__(self):
        if self.setting not in ('turbo', 'fast', 'base'):
            raise ValueError('Use a single-direction RoMa setting: turbo/fast/base')
        if min(self.every_n_steps, self.log_every_n_steps) < 1 or self.warmup_steps < 0:
            raise ValueError('Invalid teacher schedule')
        if min(self.match_weight, self.alignment_weight, self.correctness_pixels) <= 0:
            raise ValueError('Teacher loss weights and pixel tolerance must be positive')
        if not 0 <= self.min_confidence <= 1:
            raise ValueError('min_confidence must be in [0,1]')


@dataclass
class TeacherField:
    scene: int
    values: Tensor                 # [1,3,H_teacher,W_teacher]: target xy, confidence


@torch.no_grad()
def teacher_at(field: TeacherField, indices: Tensor, shape: tuple[int, int]):
    """The ONLY added bilinear lookup: 3-channel teacher field at query pixels."""
    h, w = shape
    xy = torch.stack((indices % w, indices // w), -1).float()
    size = xy.new_tensor([w, h])
    grid = 2 * (xy + .5) / size - 1
    sampled = F.grid_sample(field.values, grid[:, :, None], mode='bilinear',
                            align_corners=False, padding_mode='border')[:, :, :, 0].transpose(1, 2)
    target_grid, confidence = sampled[..., :2], sampled[..., 2]
    finite = sampled.isfinite().all(-1)
    inside = finite & (target_grid >= -1).all(-1) & (target_grid <= 1).all(-1)
    target_xy = (torch.nan_to_num(target_grid) + 1) * size / 2 - .5
    # Out-of-image predictions get ZERO weight, never fake boundary labels.
    target_index_xy = target_xy.round().long().clamp_min(0)
    target_index_xy = torch.minimum(target_index_xy, size.long() - 1)
    target_indices = target_index_xy[..., 1] * w + target_index_xy[..., 0]
    confidence = torch.where(inside, confidence.clamp(0, 1), 0)
    return target_xy, target_indices, confidence, inside


def weighted_average(value: Tensor, weight: Tensor):
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def teacher_objective(student, dump: dict, field: TeacherField,
                      cfg: DescriptorTeacherCfg):
    """Only student parameters have gradients; original features/XYZ are constants."""
    features, points = dump['features'].detach().float(), dump['points'].detach().float()
    shape = dump['shape']
    h, w = shape
    matches = student.match(features, shape, dump['queries'])
    transform, _, _, _ = student.estimate(points, matches)
    teacher_xy, target_index, confidence, inside = teacher_at(field, matches.queries, shape)
    usable = inside & (confidence >= cfg.min_confidence)
    weights = confidence * usable
    coarse_grid = student.cfg.target_grid
    # Label the actual integer grid centers (also correct for non-divisible H/W),
    # rather than treating their cells as continuous half-pixel bins.
    centers = torch.arange(coarse_grid, device=points.device) + .5
    coarse_x = (teacher_xy[..., 0, None] - (centers * w / coarse_grid).long()).abs().argmin(-1)
    coarse_y = (teacher_xy[..., 1, None] - (centers * h / coarse_grid).long()).abs().argmin(-1)
    coarse_label = coarse_y * coarse_grid + coarse_x
    coarse_loss = weighted_average(F.cross_entropy(
        matches.coarse_logits.transpose(1, 2), coarse_label, reduction='none'), weights)
    in_window = matches.fine_indices == target_index[..., None]
    fine_label = in_window.long().argmax(-1)
    fine_loss = weighted_average(F.cross_entropy(
        matches.fine_logits.transpose(1, 2), fine_label, reduction='none'),
        weights * in_window.any(-1))
    predicted_xy = torch.stack((matches.targets % w, matches.targets // w), -1).float()
    error_px = (predicted_xy - teacher_xy).norm(dim=-1)
    # RoMa overlap alone must not endorse the student's *wrong* selected match.
    correctness = confidence * torch.exp(-.5 * (error_px / cfg.correctness_pixels).square())
    correctness = correctness.detach()
    confidence_loss = F.binary_cross_entropy_with_logits(matches.confidence_logits, correctness)
    matching_loss = coarse_loss + fine_loss + confidence_loss
    source = gather(points[:, 1], matches.queries)
    teacher_target = gather(points[:, 0], target_index)
    radius = scene_radius(points[:, 0])
    residual = (transform.apply(source) - teacher_target).norm(dim=-1) / radius[:, None]
    robust_error = (residual.square() + 1e-6).sqrt() - .001
    alignment_loss = weighted_average(robust_error, weights * transform.valid[:, None])
    loss = cfg.match_weight * matching_loss + cfg.alignment_weight * alignment_loss
    # Keep every student parameter in the DDP graph, including all-invalid batches.
    loss = loss + sum(p.reshape(-1)[0] * 0 for p in student.parameters())
    metrics = {}
    if dump.get('log', False):
        with torch.no_grad():
            correct = error_px <= cfg.correctness_pixels
            metrics = {
                'loss_coarse': coarse_loss.detach(), 'loss_fine': fine_loss.detach(),
                'loss_confidence': confidence_loss.detach(), 'loss_alignment': alignment_loss.detach(),
                'valid_pairs': usable.float().sum(),
                'pixel_error': masked_mean(error_px, usable),
                'pixel_accuracy': masked_mean(correct.float(), usable),
                'coarse_accuracy': masked_mean((matches.coarse_logits.argmax(-1) == coarse_label).float(), usable),
                'fine_window_hit_fraction': masked_mean(in_window.any(-1).float(), usable),
                'confidence_correct': masked_mean(matches.confidence, usable & correct),
                'confidence_wrong': masked_mean(matches.confidence, usable & ~correct),
            }
            # Independent fixed audit queries: not selected for fitting or CE.
            audit = grid_indices(h, w, student.cfg.source_grid, points.device, offset=.125)[None]
            _, audit_target, audit_conf, audit_inside = teacher_at(field, audit, shape)
            audit_source = gather(points[:, 1], audit)
            audit_target = gather(points[:, 0], audit_target)
            valid = audit_inside & (audit_conf >= cfg.min_confidence)
            before = (audit_source - audit_target).norm(dim=-1) / radius[:, None]
            after = (transform.apply(audit_source) - audit_target).norm(dim=-1) / radius[:, None]
            measured = valid & transform.valid[:, None]
            before_mean, after_mean = masked_mean(before, measured), masked_mean(after, measured)
            metrics.update(
                audit_pairs=valid.float().sum(), audit_measured_pairs=measured.float().sum(),
                audit_residual_before=before_mean, audit_residual_after=after_mean,
                audit_after_before_ratio=torch.where(before_mean > 1e-6,
                                                     after_mean / before_mean.clamp_min(1e-6), float('nan')),
                audit_improved_fraction=masked_mean((after < before).float(), measured),
            )
    return loss, metrics


class DescriptorTeacher:
    def __init__(self, cfg: DescriptorTeacherCfg):
        self.cfg = cfg
        self.model = None

    def _model(self, device):
        if self.model is None:
            try:
                from romav2 import RoMaV2
            except ModuleNotFoundError as error:
                raise RuntimeError('RoMaV2 teacher requires: python -m pip install -e ./RoMaV2') from error
            self.model = RoMaV2(RoMaV2.Cfg(setting=self.cfg.setting, compile=False))
            self.model.to(device).eval().requires_grad_(False)
        return self.model

    @torch.no_grad()
    def prepare(self, rgb: Tensor, step: int):
        # Diagnostics must not systematically miss the teacher when intervals
        # differ (e.g. every 2 steps, with W&B flushing on step 49/99/etc.).
        if step % self.cfg.every_n_steps and (step + 1) % self.cfg.log_every_n_steps:
            return None
        if rgb.ndim != 5 or rgb.shape[1:3] != (2, 3):
            raise ValueError('RoMa teacher requires context RGB [B,2,3,H,W] in [0,1]')
        scene = (step // self.cfg.every_n_steps) % len(rgb)
        device_context = torch.cuda.device(rgb.device) if rgb.is_cuda else nullcontext()
        precision = torch.get_float32_matmul_precision()
        try:
            # Required by RoMa, scoped so the Gaussian model retains its setting.
            torch.set_float32_matmul_precision('highest')
            with device_context, torch.autocast(device_type=rgb.device.type, enabled=False):
                model = self._model(rgb.device)
                # RoMa AB means first argument -> second: source B -> reference A.
                prediction = model.match(rgb[scene, 1].float(), rgb[scene, 0].float())
                values = torch.cat((prediction['warp_AB'], prediction['overlap_AB']), -1)
                values = values.permute(0, 3, 1, 2).contiguous().float()
        finally:
            torch.set_float32_matmul_precision(precision)
        return TeacherField(scene, values)

    def loss(self, student, dump, field):
        if field is None:
            return sum(p.reshape(-1)[0] * 0 for p in student.parameters()), {}
        with torch.autocast(device_type=dump['features'].device.type, enabled=False):
            return teacher_objective(student, dump, field, self.cfg)
