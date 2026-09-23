"""Training-only observed-color supervision; no target pose or attribute labels.

The matcher is intentionally owned by a plain Python object, not an nn.Module:
its frozen weights are excluded from DDP, the optimizer and scene checkpoints.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from ..types import Gaussians
from .matching_prior import MatchingPriorCfg, estimate_matching_prior, sample_image
from .roma_matching import Correspondences, RoMaMatcher, RoMaMatcherCfg


@dataclass
class ObservedColorAuxCfg:
    enabled: bool = False
    weight: float = 0.05
    warm_up_steps: int = 100
    ramp_steps: int = 200
    every_n_steps: int = 1
    max_pairs_per_batch: int = 1
    near_scale: float = 0.01
    far_scale: float = 100.0
    roma: RoMaMatcherCfg = field(default_factory=RoMaMatcherCfg)
    pose: MatchingPriorCfg = field(default_factory=MatchingPriorCfg)

    def __post_init__(self):
        if self.weight < 0 or min(self.warm_up_steps, self.ramp_steps) < 0:
            raise ValueError("Auxiliary weight and schedule must be nonnegative")
        if min(self.every_n_steps, self.max_pairs_per_batch) < 1:
            raise ValueError("Auxiliary sampling intervals must be positive")
        if not 0 < self.near_scale < self.far_scale:
            raise ValueError("Expected 0 < near_scale < far_scale")

    def weight_at(self, step: int) -> float:
        if not self.enabled or step < self.warm_up_steps:
            return 0.0
        ramp = min(1.0, (step - self.warm_up_steps) / self.ramp_steps) if self.ramp_steps else 1.0
        return self.weight * ramp


@dataclass
class PreparedPair:
    batch_index: int
    matches: Correspondences


@torch.no_grad()
def observed_colors(means: Tensor, camera: Tensor, intrinsics: Tensor,
                    image: Tensor) -> tuple[Tensor, Tensor]:
    """Sample at CURRENT projected centers, then detach the entire color path."""
    camera_points = (means.detach().float() - camera[:3, 3]) @ camera[:3, :3]
    projected = camera_points @ intrinsics.float().T
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    height, width = image.shape[-2:]
    border = uv.new_tensor([0.5 / width, 0.5 / height])
    valid = (torch.isfinite(uv).all(-1) & (camera_points[:, 2] > 1e-8)
             & (uv >= border).all(-1) & (uv <= 1 - border).all(-1))
    colors = sample_image(image.detach().float(), 2 * uv[valid] - 1)
    return colors.detach(), valid


class ObservedColorAuxiliary:
    def __init__(self, cfg: ObservedColorAuxCfg):
        self.cfg = cfg
        self.matcher = RoMaMatcher(cfg.roma)

    def initialize(self, device: torch.device):
        if self.cfg.enabled and self.cfg.weight > 0:
            self.matcher.initialize(device)

    def release(self):
        self.matcher.release()

    def prepare_pairs(self, images: Tensor, step: int) -> list[PreparedPair]:
        """Run before encoder forward to reduce peak activation memory.

        images are original RGB in [0,1], before the backbone normalization shim.
        Only context images are accepted; neither context nor target GT pose is.
        """
        if self.cfg.weight_at(step) == 0 or step % self.cfg.every_n_steps:
            return []
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError("The initial auxiliary integration requires two RGB context views")
        batch_size = images.shape[0]
        count = min(batch_size, self.cfg.max_pairs_per_batch)
        start = (step // self.cfg.every_n_steps * count) % batch_size
        return [PreparedPair(index, self.matcher.match(images[index].detach()))
                for index in ((start + offset) % batch_size for offset in range(count))]

    def compute(
        self, gaussians: Gaussians, images: Tensor, intrinsics: Tensor,
        support_points: Tensor | None, pairs: list[PreparedPair], step: int,
        background_color: Tensor, render_fn: Callable | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Symmetric source-only splats, evaluated at reliable target matches.

        Slot ownership follows the decoder's view-major output ordering. The
        target-view slots are excluded so copying target RGB cannot solve loss.
        Masks and near/far planes are detached, independent of GT camera scale.
        """
        zero = gaussians.means.sum() * 0.0
        stats: dict[str, Tensor | float] = {
            "weight": self.cfg.weight_at(step), "attempted_pairs": float(len(pairs)),
            "valid_pairs": 0.0, "directions": 0.0, "inliers": 0.0,
            "alignment_error": 0.0, "translation_scale": 0.0, "raw_loss": zero.detach(),
        }
        if not pairs or stats["weight"] == 0:
            return zero, stats
        batch_size, views, _, height, width = images.shape
        slots = height * width
        if views != 2 or gaussians.means.shape[:2] != (batch_size, views * slots):
            raise ValueError("Auxiliary rendering requires one view-major Gaussian slot per pixel")
        if support_points is None or support_points.shape != (batch_size, views, height, width, 3):
            raise ValueError("Auxiliary alignment requires raw point-head supports [B,2,H,W,3]")

        losses, accepted_priors = [], []
        # Disable training autocast: cameras, projected colors and CUDA splats use FP32.
        with torch.autocast(device_type=images.device.type, enabled=False):
            for pair in pairs:
                batch_index = pair.batch_index
                prior = estimate_matching_prior(pair.matches, intrinsics[batch_index],
                                                support_points[batch_index, 0], self.cfg.pose)
                if prior is None:
                    continue
                scene_scale = support_points[batch_index].detach().float().norm(dim=-1).median()
                if not bool(torch.isfinite(scene_scale)) or scene_scale <= 0:
                    continue
                scene_scale = scene_scale.clamp_min(1e-6)
                prior_used = False
                for source, target, target_grid in ((0, 1, prior.grid_b), (1, 0, prior.grid_a)):
                    selection = slice(source * slots, (source + 1) * slots)
                    means = gaussians.means[batch_index, selection].float()
                    colors, valid = observed_colors(means, prior.cameras[source],
                                                    intrinsics[batch_index, source], images[batch_index, source])
                    if not bool(valid.any()):
                        continue
                    if render_fn is None:
                        # Keep CPU checks and decoder-only evaluation independent of CUDA/RoMa.
                        from ..decoder.cuda_splatting import render_cuda
                        render_fn = render_cuda
                    rendered, _ = render_fn(
                        extrinsics=prior.cameras[target:target + 1],
                        intrinsics=intrinsics[batch_index, target:target + 1].detach().float(),
                        near=(scene_scale * self.cfg.near_scale).reshape(1),
                        far=(scene_scale * self.cfg.far_scale).reshape(1),
                        image_shape=(height, width),
                        background_color=background_color.detach().to(means).reshape(1, 3),
                        gaussian_means=means[valid].unsqueeze(0),
                        gaussian_covariances=gaussians.covariances[batch_index, selection][valid].float().unsqueeze(0),
                        gaussian_sh_coefficients=colors[None, :, :, None],
                        gaussian_opacities=gaussians.opacities[batch_index, selection][valid].float().unsqueeze(0),
                        use_sh=False,
                    )
                    prediction = sample_image(rendered[0], target_grid.float())
                    target_rgb = sample_image(images[batch_index, target].detach().float(), target_grid.float())
                    confidence = prior.confidence.detach().float()
                    losses.append(((prediction - target_rgb).abs() * confidence[:, None]).sum()
                                  / (3 * confidence.sum()).clamp_min(1e-8))
                    prior_used = True
                if prior_used:
                    accepted_priors.append(prior)

        if not losses:
            return zero, stats
        raw_loss = torch.stack(losses).mean()
        count = len(accepted_priors)
        stats.update({
            "valid_pairs": float(count), "directions": float(len(losses)),
            "inliers": sum(len(p.confidence) for p in accepted_priors) / count,
            "alignment_error": sum(p.alignment_error for p in accepted_priors) / count,
            "translation_scale": sum(p.scale for p in accepted_priors) / count,
            "raw_loss": raw_loss.detach(),
        })
        return self.cfg.weight_at(step) * raw_loss, stats
