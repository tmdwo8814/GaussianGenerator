"""Geometry-conditioned channel interaction for support -> slot allocation.

The original additive feature/geometry path stays in the decoder. This module
adds tanh(Linear(local_geometry)) * LN(slot_projection) * LN(source_projection).
LayerNorm has no affine parameters; the only new parameters are the 4 -> H gate.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GeometryConditionedRelation(nn.Module):
    def __init__(self, hidden_dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.gate = nn.Linear(4, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @staticmethod
    def normalize_projection(projection: Tensor) -> Tensor:
        # Call once per support, before gathering edges. No learnable LN affine.
        return F.layer_norm(projection, (projection.shape[-1],), eps=1e-5)

    def forward(self, geometry: Tensor, source: Tensor, destination: Tensor):
        """geometry [chunk,K,4], normalized source [chunk,H], destination [chunk,K,H].

        Geometry is [delta_xyz, squared_distance] in scene-normalized coordinates.
        The radius includes all K candidates, including self. It remains
        differentiable; clamp BEFORE sqrt/rsqrt keeps coincident points finite.
        """
        radius_squared = geometry[..., 3:].mean(dim=1, keepdim=True).clamp_min(
            self.epsilon ** 2
        )
        local_geometry = torch.cat((
            geometry[..., :3] * radius_squared.rsqrt(),
            geometry[..., 3:] / radius_squared,
        ), dim=-1)
        gate = self.gate(local_geometry).tanh()
        interaction = destination * source[:, None]
        return gate * interaction, gate
