"""Batched weighted similarity fitting. No cameras, features, or kNN here."""

from dataclasses import dataclass

import torch
from torch import Tensor


class _Rotation(torch.autograd.Function):
    """Proper polar factor with a stable derivative at repeated singular values.

    Differentiating SVD's U and V separately divides by singular-value
    *differences*, although the rotation only needs their signed *sums*.
    This matters for nearly isotropic point clouds and degenerate matches.
    """

    @staticmethod
    def forward(ctx, covariance):
        u, singular, vh = torch.linalg.svd(covariance)
        sign = torch.ones_like(singular)
        sign[:, -1] = torch.linalg.det(u @ vh).sign()
        corrected_vh = sign[..., None] * vh
        signed = singular * sign
        ctx.save_for_backward(u, corrected_vh, signed)
        ctx.mark_non_differentiable(singular)
        return u @ corrected_vh, singular

    @staticmethod
    def backward(ctx, gradient, _singular_gradient):
        u, vh, singular = ctx.saved_tensors
        local = u.transpose(-1, -2) @ gradient @ vh.transpose(-1, -2)
        denominator = singular[:, :, None] + singular[:, None, :]
        floor = singular.abs().amax(-1, keepdim=True)[..., None] * 1e-6
        safe = denominator > floor.clamp_min(1e-12)
        skew = torch.where(safe, (local - local.transpose(-1, -2)) /
                           denominator.clamp_min(1e-12), 0)
        return u @ skew @ vh


@dataclass
class Similarity:
    scale: Tensor                 # [B]
    rotation: Tensor              # [B,3,3], column-vector convention
    translation: Tensor           # [B,3]
    valid: Tensor                 # [B], numerical validity, not match accuracy

    def apply(self, points: Tensor) -> Tensor:
        return self.scale[:, None, None] * (
            points @ self.rotation.transpose(-1, -2)
        ) + self.translation[:, None]

    def detached(self):
        return Similarity(*(x.detach() for x in
                            (self.scale, self.rotation, self.translation, self.valid)))


def scene_radius(points: Tensor) -> Tensor:
    """Translation-invariant RMS radius for reporting dimensionless residuals."""
    centered = points - points.mean(1, keepdim=True)
    return centered.square().sum(-1).mean(-1).sqrt().clamp_min(1e-6)


def fit_similarity(source: Tensor, target: Tensor, weights: Tensor) -> Similarity:
    """Weighted Umeyama; collinear/empty inputs yield identity and valid=False.

    Planar, non-collinear inputs are valid. Reflections are never returned.
    Invalid pairs are masked before any arithmetic, including centroid fitting.
    """
    finite = (source.isfinite().all(-1) & target.isfinite().all(-1)
              & weights.isfinite() & (weights > 0))
    source = torch.where(finite[..., None], source, 0)
    target = torch.where(finite[..., None], target, 0)
    weights = torch.where(finite, weights, 0)
    normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    mean_s = (source * normalized[..., None]).sum(1)
    mean_t = (target * normalized[..., None]).sum(1)
    xs, xt = source - mean_s[:, None], target - mean_t[:, None]
    variance = (normalized * xs.square().sum(-1)).sum(-1)
    covariance = xs.transpose(-1, -2) @ (normalized[..., None] * xt)
    row_rotation, singular = _Rotation.apply(covariance)
    # The optimum trace is differentiable through the proper rotation, too.
    scale = (covariance * row_rotation).sum((-1, -2)) / variance.clamp_min(1e-12)
    translation = mean_t - scale[:, None] * (mean_s[:, None] @ row_rotation)[:, 0]
    with torch.no_grad():
        valid = ((finite.sum(-1) >= 3) & (variance > 1e-12)
                 & (singular[:, 1] > singular[:, 0].clamp_min(1e-12) * 1e-5)
                 & scale.isfinite() & (scale > 1e-6)
                 & translation.isfinite().all(-1))
    eye = torch.eye(3, device=source.device, dtype=source.dtype).expand_as(row_rotation)
    return Similarity(torch.where(valid, scale, 1),
                      torch.where(valid[:, None, None], row_rotation.transpose(-1, -2), eye),
                      torch.where(valid[:, None], translation, 0), valid)


def robust_similarity(source: Tensor, target: Tensor, weights: Tensor) -> Similarity:
    """One fit, one detached Cauchy reweight, one refit. No RANSAC loop."""
    with torch.no_grad():
        first = fit_similarity(source, target, weights)
        residual = (first.apply(source) - target).norm(dim=-1)
        usable = (weights > 0) & residual.isfinite()
        ordered = residual.masked_fill(~usable, float('inf')).sort(-1).values
        index = ((usable.sum(-1) - 1).clamp_min(0) // 2)[:, None]
        median = ordered.gather(1, index).squeeze(1)
        median = torch.where(median.isfinite(), median, 1).clamp_min(1e-6)
        factor = 1 / (1 + (residual / median[:, None]).square())
        factor = torch.where(usable, factor, 0)
    return fit_similarity(source, target, weights * factor)
