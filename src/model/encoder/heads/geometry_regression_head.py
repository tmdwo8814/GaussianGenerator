"""A single affine readout of frozen BEAM's existing pooled features."""

from math import expm1, log

import torch
import torch.nn.functional as F
from torch import nn


class GeometryRegressionHead(nn.Module):
    def __init__(self, feature_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.init_scale = init_scale
        # Center offset (3), lower-triangular covariance factor (6), opacity (1).
        # No feature projection, edge MLP, hidden layer or learned scale factor.
        self.readout = nn.Linear(feature_dim, 10)
        self.reset_parameters()

    def reset_parameters(self):
        # Small RANDOM outputs, with a positive isotropic Cholesky bias.
        # Opacity starts near sigmoid(0) = 0.5, independently of BEAM mass.
        nn.init.normal_(self.readout.weight, std=1e-4)
        nn.init.zeros_(self.readout.bias)
        with torch.no_grad():
            self.readout.bias[[3, 5, 8]] = log(expm1(self.init_scale))

    def forward(self, points, pooled):
        """Use h_i = sum_j w_ji f_j without adding another evidence encoder.

        Points provide only the center's fixed support anchor, in normalized
        coordinates. No fitted mean/covariance or mass-to-opacity formula is
        supplied to this head. Positive covariance and bounded opacity use
        parameter-free output transforms.
        """
        raw = self.readout(pooled)
        means = points + raw[:, :3]
        a, b, c, d, e, f = raw[:, 3:9].unbind(-1)
        zero = torch.zeros_like(a)
        lower = torch.stack((F.softplus(a), zero, zero, b, F.softplus(c), zero,
                             d, e, F.softplus(f)), -1).reshape(-1, 3, 3)
        return means, lower @ lower.transpose(-1, -2), raw[:, 9].sigmoid()
