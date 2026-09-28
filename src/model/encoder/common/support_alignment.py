"""Small image-support matcher -> one B-to-A Sim(3) -> ALL B supports.

The image-grid adapter is intentionally separate from weighted_sim3.py.
Only this adapter assumes pixel supports; fitting itself accepts arbitrary pairs.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .weighted_sim3 import robust_similarity, scene_radius


@dataclass
class SupportAlignmentCfg:
    enabled: bool = False
    descriptor_dim: int = 32
    hidden_dim: int = 64
    source_grid: int = 16
    target_grid: int = 32
    extra_cells: int = 64
    fine_window: int = 16
    temperature: float = 0.1

    def __post_init__(self):
        if min(self.descriptor_dim, self.hidden_dim, self.source_grid,
               self.target_grid, self.fine_window) < 2:
            raise ValueError('Alignment dimensions/grid/window must be >= 2')
        if not 0 <= self.extra_cells <= self.source_grid ** 2 or self.temperature <= 0:
            raise ValueError('Invalid extra_cells or temperature')


def gather(values: Tensor, indices: Tensor) -> Tensor:
    batch = torch.arange(len(values), device=values.device)
    return values[batch.reshape(-1, *([1] * (indices.ndim - 1))), indices]


def grid_indices(height: int, width: int, grid: int, device, offset: float = .5):
    y = ((torch.arange(grid, device=device) + offset) * height / grid).long()
    x = ((torch.arange(grid, device=device) + offset) * width / grid).long()
    return (y[:, None] * width + x[None]).flatten()


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    valid = valid & values.isfinite()
    total = torch.where(valid, values, 0).sum()
    count = valid.sum()
    return torch.where(count > 0, total / count.clamp_min(1), float('nan'))


@dataclass
class Matches:
    queries: Tensor
    targets: Tensor
    confidence_logits: Tensor
    coarse_logits: Tensor
    fine_logits: Tensor
    fine_indices: Tensor

    @property
    def confidence(self):
        return self.confidence_logits.sigmoid()


def unique_weights(matches: Matches, count: int) -> Tensor:
    """Best confidence per target pixel, deterministic first-query tie break."""
    confidence = matches.confidence
    with torch.no_grad():
        best = confidence.new_full((len(confidence), count), -1)
        best.scatter_reduce_(1, matches.targets, confidence, reduce='amax', include_self=True)
        order = torch.arange(confidence.shape[1], device=confidence.device).expand_as(matches.targets)
        candidates = torch.where(confidence == best.gather(1, matches.targets), order, order.shape[1])
        winner = matches.targets.new_full(best.shape, order.shape[1])
        winner.scatter_reduce_(1, matches.targets, candidates, reduce='amin', include_self=True)
        keep = order == winner.gather(1, matches.targets)
    return confidence * keep


class SupportAlignment(nn.Module):
    def __init__(self, cfg: SupportAlignmentCfg, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        # Normalize after projection: avoid a full-resolution C=256 LN buffer.
        self.descriptor = nn.Sequential(
            nn.Linear(feature_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.descriptor_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(2 * cfg.descriptor_dim + 4, cfg.hidden_dim),
            nn.GELU(), nn.Linear(cfg.hidden_dim, 1),
        )

    def describe(self, features):
        return F.normalize(self.descriptor(features), dim=-1, eps=1e-6)

    def _match_queries(self, reference, source_features, queries, coarse_indices, shape):
        h, w = shape
        query = self.describe(gather(source_features, queries))
        coarse = query @ reference[:, coarse_indices].transpose(-1, -2)
        coarse_top = coarse.topk(2, dim=-1).values
        anchors = coarse_indices[coarse.argmax(-1)]
        window = self.cfg.fine_window
        top = ((anchors // w) - window // 2).clamp(0, h - window)
        left = ((anchors % w) - window // 2).clamp(0, w - window)
        offsets = torch.arange(window, device=queries.device)
        offsets = (offsets[:, None] * w + offsets[None]).flatten()
        fine_indices = (top * w + left)[..., None] + offsets
        # Only requested windows are gathered; never unfold the full image.
        patches = gather(reference, fine_indices)
        fine = torch.einsum('bqd,bqkd->bqk', query, patches)
        best = fine.argmax(-1)
        targets = fine_indices.gather(-1, best[..., None]).squeeze(-1)
        matched = gather(reference, targets)
        fine_top = fine.topk(2, dim=-1).values
        evidence = torch.stack((coarse_top[..., 0], coarse_top[..., 0] - coarse_top[..., 1],
                                fine_top[..., 0], fine_top[..., 0] - fine_top[..., 1]), -1)
        confidence = self.confidence_head(torch.cat((query, matched, evidence), -1)).squeeze(-1)
        return Matches(queries, targets, confidence, coarse / self.cfg.temperature,
                       fine / self.cfg.temperature, fine_indices)

    def match(self, features: Tensor, shape: tuple[int, int], queries: Tensor | None = None):
        """[B,2,H*W,C]. Integer supports only; no feature/XYZ interpolation.

        Explicit queries replay the main pass for the one supervised scene.
        """
        h, w = shape
        if features.ndim != 4 or features.shape[1:3] != (2, h * w):
            raise ValueError('Alignment requires [B,2,H*W,C] features')
        if min(h, w) < max(self.cfg.source_grid * 4, self.cfg.target_grid, self.cfg.fine_window):
            raise ValueError('Image too small for alignment grid/window configuration')
        reference = self.describe(features[:, 0])
        coarse = grid_indices(h, w, self.cfg.target_grid, features.device)
        if queries is not None:
            return self._match_queries(reference, features[:, 1], queries, coarse, shape)
        base = grid_indices(h, w, self.cfg.source_grid, features.device).expand(len(features), -1)
        first = self._match_queries(reference, features[:, 1], base, coarse, shape)
        if not self.cfg.extra_cells:
            return first
        # Stable sort resolves equal-confidence cells by their original index.
        cells = first.confidence.detach().argsort(dim=-1, descending=True, stable=True)[:, :self.cfg.extra_cells]
        row, col = cells // self.cfg.source_grid, cells % self.cfg.source_grid
        quarters = features.new_tensor([[.25, .25], [.25, .75], [.75, .25], [.75, .75]])
        ys = ((row[..., None] + quarters[:, 0]) * h / self.cfg.source_grid).long()
        xs = ((col[..., None] + quarters[:, 1]) * w / self.cfg.source_grid).long()
        extra = (ys * w + xs).flatten(1)
        second = self._match_queries(reference, features[:, 1], extra, coarse, shape)
        return Matches(*(torch.cat((getattr(first, key), getattr(second, key)), 1)
                         for key in Matches.__dataclass_fields__))

    def estimate(self, points: Tensor, matches: Matches):
        source = gather(points[:, 1], matches.queries)
        target = gather(points[:, 0], matches.targets)
        weights = unique_weights(matches, points.shape[2])
        return robust_similarity(source, target, weights), source, target, weights

    def forward(self, points: Tensor, features: Tensor, shape: tuple[int, int],
                dump: dict | None = None):
        """Detached fitting, but the transform is applied to LIVE point tensors."""
        with torch.no_grad(), torch.autocast(device_type=features.device.type, enabled=False):
            matches = self.match(features.detach().float(), shape)
            transform, source, target, weights = self.estimate(points.detach().float(), matches)
            transform = transform.detached()
            if dump is not None:
                # Only this scene is replayed with gradients by the teacher loss.
                scene = dump.get('scene', 0)
                dump.update(points=points[scene:scene + 1].detach(),
                            features=features[scene:scene + 1].detach(),
                            queries=matches.queries[scene:scene + 1], shape=shape)
                if dump.get('log', False):
                    radius = scene_radius(points[:, 0].float())
                    before = (source - target).norm(dim=-1) / radius[:, None]
                    after = (transform.apply(source) - target).norm(dim=-1) / radius[:, None]
                    pair_valid = (weights > 0) & transform.valid[:, None]
                    trace = transform.rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
                    angle = ((trace - 1) / 2).clamp(-1, 1).acos().rad2deg()
                    effective = weights.sum(-1).square() / weights.square().sum(-1).clamp_min(1e-12)
                    dump['metrics'] = {
                        'fit_valid_fraction': transform.valid.float().mean(),
                        'applied_fraction': transform.valid.float().mean() * float(dump.get('apply', True)),
                        'identity_fallback_fraction': (~transform.valid).float().mean(),
                        'unique_pairs': (weights > 0).float().sum(-1).mean(),
                        'effective_pairs': effective.mean(),
                        'confidence': matches.confidence.mean(),
                        'student_residual_before': masked_mean(before, pair_valid),
                        'student_residual_after': masked_mean(after, pair_valid),
                        'scale': masked_mean(transform.scale, transform.valid),
                        'rotation_deg': masked_mean(angle, transform.valid),
                        'translation_relative': masked_mean(transform.translation.norm(dim=-1) / radius, transform.valid),
                    }
        # No per-point or per-scene switch to within-view kNN.
        # Fixed training warm-up keeps the original point frame while the matcher
        # learns. Gaussian training and unified kNN continue during these steps.
        if dump is not None and not dump.get('apply', True):
            return points
        return torch.stack((points[:, 0], transform.apply(points[:, 1])), 1)
