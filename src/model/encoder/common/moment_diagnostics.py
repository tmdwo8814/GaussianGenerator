"""Detached incoming-weight diagnostics; computed only on logging steps."""

import torch
from torch import Tensor


@torch.no_grad()
def incoming_statistics(geometry: Tensor, appearance: Tensor, neighbors: Tensor) -> Tensor:
    """[M,5]: TV, geometry/appearance entropy, geometry/appearance concentration."""
    values = torch.stack((
        .5 * (geometry - appearance).abs(),
        -geometry * geometry.clamp_min(torch.finfo(geometry.dtype).tiny).log(),
        -appearance * appearance.clamp_min(torch.finfo(appearance.dtype).tiny).log(),
        geometry.square(), appearance.square(),
    ), dim=-1)
    result = geometry.new_zeros((len(geometry), 5))
    return result.index_add_(0, neighbors.reshape(-1), values.reshape(-1, 5))


@torch.no_grad()
def summarize_statistics(statistics: Tensor, mass: Tensor, covariances: Tensor,
                         scene_scales: Tensor, mass_epsilon: float) -> dict[str, Tensor]:
    """Equal weight per active slot across the LOCAL batch; DDP averages ranks.

The radius is sqrt(trace(Sigma)/3) divided by scene scale. Pearson correlation
uses geometry concentration and log(radius). Constant inputs report zero plus
corr_valid=0, so an undefined coefficient is not mistaken for independence.
"""
    active = mass > mass_epsilon
    count = active.sum().clamp_min(1)

    def mean(value):
        return torch.where(active, value, 0).sum() / count

    tv, hg, ha, cg, ca = statistics.unbind(-1)
    radius = (covariances.diagonal(dim1=-2, dim2=-1).sum(-1) / 3).clamp_min(0).sqrt()
    radius = radius / scene_scales[:, None]
    log_radius = radius.clamp_min(torch.finfo(radius.dtype).tiny).log()
    x, y = cg - mean(cg), log_radius - mean(log_radius)
    vx, vy = mean(x.square()), mean(y.square())
    valid = (active.sum() > 1) & (vx > 1e-12) & (vy > 1e-12)
    correlation = mean(x * y) / (vx * vy).clamp_min(1e-24).sqrt()
    return {
        'weight_tv': mean(tv),
        'geometry_entropy': mean(hg),
        'appearance_entropy': mean(ha),
        'geometry_concentration': mean(cg),
        'appearance_concentration': mean(ca),
        'geometry_effective_supports': mean(cg.clamp_min(1e-12).reciprocal()),
        'appearance_effective_supports': mean(ca.clamp_min(1e-12).reciprocal()),
        'covariance_radius_normalized': mean(radius),
        'concentration_covariance_corr': torch.where(valid, correlation.clamp(-1, 1), 0),
        'concentration_covariance_corr_valid': valid.to(radius.dtype),
        'active_slot_fraction': active.to(radius.dtype).mean(),
    }
