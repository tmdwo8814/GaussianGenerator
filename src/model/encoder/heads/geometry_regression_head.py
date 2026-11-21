"""A single affine readout of frozen BEAM's existing pooled features."""

from math import expm1, log

import torch
import torch.nn.functional as F
from torch import nn


class GeometryRegressionHead(nn.Module):
    def __init__(self, feature_dim: int, init_scale: float = 0.01,
                 attributes: str = 'RRR'):
        super().__init__()
        if len(attributes) != 3 or any(mode not in 'RF' for mode in attributes) or 'R' not in attributes:
            raise ValueError('Regression attributes must be a three-letter F/R mask containing R')
        self.attributes = attributes
        self.output_indices = [index for mode, indices in zip(attributes, (range(3), range(3, 9), range(9, 10)))
                               if mode == 'R' for index in indices]
        self.init_scale = init_scale
        # Center offset (3), lower-triangular covariance factor (6), opacity (1).
        # No feature projection, edge MLP, hidden layer or learned scale factor.
        self.readout = nn.Linear(feature_dim, len(self.output_indices))
        self.reset_parameters()

    def reset_parameters(self):
        # Small RANDOM outputs, with a positive isotropic Cholesky bias.
        # Opacity starts near sigmoid(0) = 0.5, independently of BEAM mass.
        # Select rows from the same canonical initialization in every variant.
        # Leave-one-out heads allocate only the outputs that are actually used.
        weight = torch.empty(10, self.readout.in_features, device=self.readout.weight.device,
                             dtype=self.readout.weight.dtype)
        nn.init.normal_(weight, std=1e-4)
        bias = self.readout.bias.new_zeros(10)
        bias[[3, 5, 8]] = log(expm1(self.init_scale))
        with torch.no_grad():
            self.readout.weight.copy_(weight[self.output_indices])
            self.readout.bias.copy_(bias[self.output_indices])

    def forward(self, points, pooled):
        """Use h_i = sum_j w_ji f_j without adding another evidence encoder.

        Points provide only the center's fixed support anchor, in normalized
        coordinates. No fitted mean/covariance or mass-to-opacity formula is
        supplied to this head. Positive covariance and bounded opacity use
        parameter-free output transforms.
        """
        raw = self.readout(pooled)
        means = covariance = opacity = None
        offset = 0
        if self.attributes[0] == 'R':
            means = points + raw[..., offset:offset + 3]
            offset += 3
        if self.attributes[1] == 'R':
            a, b, c, d, e, f = raw[..., offset:offset + 6].unbind(-1)
            zero = torch.zeros_like(a)
            lower = torch.stack((F.softplus(a), zero, zero, b, F.softplus(c), zero,
                                 d, e, F.softplus(f)), -1).reshape(*raw.shape[:-1], 3, 3)
            covariance = lower @ lower.transpose(-1, -2)
            offset += 6
        if self.attributes[2] == 'R':
            opacity = raw[..., offset].sigmoid()
        return means, covariance, opacity
