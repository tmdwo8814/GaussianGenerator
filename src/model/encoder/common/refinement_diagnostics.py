"""Detached summaries of one-step allocation refinement, for sparse logging."""

import torch
from torch import Tensor


@torch.no_grad()
def summarize_refinement(statistics: Tensor, mass: Tensor, covariances: Tensor,
                         scene_scales: Tensor, epsilon: float) -> dict[str, Tensor]:
    # Incoming statistics use active final slots in the local batch. DDP logs
    # the mean of rank summaries. Outgoing TV includes every source support.
    active = mass > epsilon
    count = active.sum().clamp_min(1)

    def mean(value):
        return torch.where(active, value, 0).sum() / count

    tv, update2, c0, c1, h0, h1, d0, d1 = statistics.unbind(-1)
    radius = (covariances.diagonal(dim1=-2, dim2=-1).sum(-1) / 3).clamp_min(0).sqrt()
    radius = radius / scene_scales[:, None]
    log_radius = radius.clamp_min(torch.finfo(radius.dtype).tiny).log()
    x, y = c1 - mean(c1), log_radius - mean(log_radius)
    vx, vy = mean(x.square()), mean(y.square())
    valid = (active.sum() > 1) & (vx > 1e-12) & (vy > 1e-12)
    correlation = mean(x * y) / (vx * vy).clamp_min(1e-24).sqrt()
    return {
        'allocation_tv': tv.mean(),
        'logit_update_centered_rms': update2.mean().sqrt(),
        'concentration_before': mean(c0),
        'concentration_after': mean(c1),
        'entropy_before': mean(h0),
        'entropy_after': mean(h1),
        'appearance_variance_before': mean(d0),
        'appearance_variance_after': mean(d1),
        'covariance_radius_normalized': mean(radius),
        'concentration_covariance_corr': torch.where(valid, correlation.clamp(-1, 1), 0),
        'concentration_covariance_corr_valid': valid.to(radius.dtype),
        'active_slot_fraction': active.to(radius.dtype).mean(),
    }
