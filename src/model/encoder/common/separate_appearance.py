"""Appearance selection on the EXISTING support -> slot graph.

wA[j,k] is proportional to wG[j,k] * exp(score[j,k]), normalized over
all edges arriving at destination i=neighbors[j,k], NOT over outgoing k.
Geometry weights include the decoder's tiny self prior. No extra opacity
mass, candidates, cameras, or detached learning paths are introduced here.
"""

import torch
from torch import Tensor


def appearance_weights(geometry: Tensor, scores: Tensor, neighbors: Tensor) -> Tensor:
    """Stable incoming softmax(log(wG) + score), preserving zero-weight edges."""
    destinations = neighbors.reshape(-1)
    positive = geometry > 0
    # Avoid log(0) even in the masked branch (its backward would produce NaN).
    safe_geometry = torch.where(positive, geometry, torch.ones_like(geometry))
    logits = (safe_geometry.log() + scores).masked_fill(~positive, -torch.inf)
    with torch.no_grad():
        maximum = scores.new_full((len(scores),), -torch.inf)
        maximum.scatter_reduce_(0, destinations, logits.detach().reshape(-1),
                                reduce='amax', include_self=True)
    # Each slot has its positive self prior, including zero-opacity slots.
    numerator = (logits - maximum[neighbors]).exp()
    denominator = scores.new_zeros(len(scores)).index_add(
        0, destinations, numerator.reshape(-1)
    )
    return numerator / denominator[neighbors]
