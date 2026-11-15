"""Regress geometry from the SAME incoming evidence selected by frozen BEAM."""

from math import expm1, log

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class GeometryRegressionHead(nn.Module):
    def __init__(self, feature_dim: int, evidence_dim: int = 128,
                 hidden_dim: int = 256, init_scale: float = 0.01):
        super().__init__()
        self.feature_dim = feature_dim
        self.evidence_dim = evidence_dim
        self.init_scale = init_scale
        self.evidence_mlp = nn.Sequential(
            nn.Linear(feature_dim + 4, evidence_dim), nn.SiLU(),
            nn.Linear(evidence_dim, evidence_dim), nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * feature_dim + evidence_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 9),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        # Small RANDOM outputs, with a positive isotropic Cholesky bias.
        # No fitted mean/covariance or geometry distillation is used.
        nn.init.normal_(self.readout[-1].weight, std=1e-4)
        nn.init.zeros_(self.readout[-1].bias)
        with torch.no_grad():
            self.readout[-1].bias[[3, 5, 8]] = log(expm1(self.init_scale))

    def _messages(self, source_points, projection, points, neighbors, weights):
        delta = source_points[:, None] - points[neighbors]
        geometry = torch.cat((delta, delta.square().sum(-1, keepdim=True)), -1)
        first = self.evidence_mlp[0]
        hidden = projection[:, None] + F.linear(
            geometry, first.weight[:, self.feature_dim:], first.bias,
        )
        for layer in self.evidence_mlp[1:]:
            hidden = layer(hidden)
        return weights[..., None] * hidden

    def forward(self, points, features, pooled, mass, weights, neighbors, *,
                chunk_size: int, checkpoint_chunks: bool, mass_epsilon: float):
        """Coordinates are scene-normalized; neighbors[j,k] is destination i.

        Each slot reads ALL its incoming supports, not a new outgoing kNN.
        Checkpointing bounds trainable edge activations to one chunk.
        """
        def run(function, *args):
            if checkpoint_chunks and self.training and torch.is_grad_enabled():
                return checkpoint(function, *args, use_reentrant=False,
                                  preserve_rng_state=False)
            return function(*args)

        projection = F.linear(features, self.evidence_mlp[0].weight[:, :self.feature_dim])
        evidence = features.new_zeros(len(points), self.evidence_dim)
        for start in range(0, len(points), chunk_size):
            stop = start + chunk_size
            indices = neighbors[start:stop]
            messages = run(self._messages, points[start:stop], projection[start:stop],
                           points, indices, weights[start:stop])
            evidence.index_add_(0, indices.reshape(-1), messages.reshape(-1, self.evidence_dim))

        raw = []
        for start in range(0, len(points), chunk_size):
            stop = start + chunk_size
            inputs = torch.cat((features[start:stop], pooled[start:stop], evidence[start:stop],
                                (mass[start:stop] + mass_epsilon).log()[:, None]), -1)
            raw.append(run(self.readout, inputs))
        raw = torch.cat(raw)
        means = points + raw[:, :3]
        a, b, c, d, e, f = raw[:, 3:].unbind(-1)
        zero = torch.zeros_like(a)
        lower = torch.stack((F.softplus(a), zero, zero, b, F.softplus(c), zero,
                             d, e, F.softplus(f)), -1).reshape(-1, 3, 3)
        return means, lower @ lower.transpose(-1, -2)
