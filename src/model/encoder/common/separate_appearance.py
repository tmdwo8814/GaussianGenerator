"""Appearance weights on the original support -> slot graph.

wA[j,i] is proportional to wG[j,i] * exp(score[j,i]). Both weights
normalize INCOMING edges at slot i. Geometry budgets/moments are untouched.
"""

import torch
from torch import Tensor

from .appearance_neighborhood import incoming_softmax


def appearance_weights(geometry: Tensor, scores: Tensor, neighbors: Tensor) -> Tensor:
    positive = geometry > 0
    # Masked zero edges must not evaluate log(0), including in backward.
    safe_geometry = torch.where(positive, geometry, torch.ones_like(geometry))
    return incoming_softmax(safe_geometry.log() + scores, neighbors, positive)


@torch.no_grad()
def appearance_weight_statistics(geometry: Tensor, appearance: Tensor,
                                 neighbors: Tensor) -> dict[str, Tensor]:
    """Equal weight per slot; effective support count is 1 / sum_j w_ji**2."""
    tiny = torch.finfo(geometry.dtype).tiny
    values = torch.stack((
        .5 * (geometry - appearance).abs(),
        -geometry * geometry.clamp_min(tiny).log(),
        -appearance * appearance.clamp_min(tiny).log(),
        geometry.square(), appearance.square(),
    ), -1)
    incoming = geometry.new_zeros(len(geometry), 5).index_add(
        0, neighbors.flatten(), values.reshape(-1, 5),
    )
    tv, hg, ha, cg, ca = incoming.unbind(-1)
    return {
        'weight_tv': tv.mean(),
        'geometry_entropy': hg.mean(),
        'appearance_entropy': ha.mean(),
        'geometry_concentration': cg.mean(),
        'appearance_concentration': ca.mean(),
        'geometry_effective_supports': cg.clamp_min(tiny).reciprocal().mean(),
        'appearance_effective_supports': ca.clamp_min(tiny).reciprocal().mean(),
    }
