"""Learn an identity-initialized congruence transform of covariance moments.

This changes only covariance. Allocation, centers, budgets and opacity are
computed by the original moment decoder. No inverse or eigendecomposition.
"""

from math import isfinite, log

import torch
from torch import Tensor, nn

from .tensor_kernels import run_tensor_kernel


class MomentShape(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128,
                 scale_limit: float = 4.0, shear_limit: float = 0.5,
                 epsilon: float = 1e-8, *, compile_kernels: bool = False):
        super().__init__()
        if min(feature_dim, hidden_dim) < 1:
            raise ValueError('Moment shape feature and hidden dimensions must be positive')
        if not isfinite(scale_limit) or scale_limit <= 1:
            raise ValueError('shape_scale_limit must be finite and greater than one')
        if not isfinite(shear_limit) or shear_limit < 0:
            raise ValueError('shape_shear_limit must be finite and nonnegative')
        if not isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Moment shape epsilon must be finite and positive')
        self.log_scale_limit = log(scale_limit)
        self.shear_limit = shear_limit
        self.epsilon = epsilon
        self.compile_kernels = compile_kernels
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        return run_tensor_kernel(MomentShape._forward, self, features, covariance,
                                 enabled=self.compile_kernels)

    def _forward(self, features, covariance):
        # Six unique entries, normalized by trace to describe shape rather
        # than scene/coverage scale. The clamp also handles zero-rank moments.
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        shape = covariance / trace.clamp_min(self.epsilon)[..., None, None]
        descriptor = torch.stack((shape[..., 0, 0], shape[..., 1, 1], shape[..., 2, 2],
                                  shape[..., 1, 0], shape[..., 2, 0], shape[..., 2, 1]), -1)
        raw = self.mlp(torch.cat((features, descriptor), -1))
        diagonal = (self.log_scale_limit * raw[..., :3].tanh()).exp()
        shear = self.shear_limit * raw[..., 3:].tanh()
        d0, d1, d2 = diagonal.unbind(-1)
        s10, s20, s21 = shear.unbind(-1)
        zero = torch.zeros_like(d0)
        transform = torch.stack((d0, zero, zero, s10, d1, zero, s20, s21, d2), -1)
        transform = transform.reshape(*features.shape[:-1], 3, 3)
        transformed = transform @ covariance @ transform.transpose(-1, -2)
        transformed = 0.5 * (transformed + transformed.transpose(-1, -2))
        # The decoder adds its isotropic floor AFTER this transformation.
        return transformed, transform


@torch.no_grad()
def shape_statistics(before: Tensor, after: Tensor, transform: Tensor) -> dict[str, Tensor]:
    """Chunk means; radii are 3D scene-normalized RMS radii, not pixel footprints."""
    diagonal = transform.diagonal(dim1=-2, dim2=-1)
    shear = torch.stack((transform[..., 1, 0], transform[..., 2, 0], transform[..., 2, 1]), -1)
    trace_before = before.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0)
    trace_after = after.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0)
    positive = trace_before > torch.finfo(before.dtype).tiny
    ratio = (trace_after / trace_before.clamp_min(torch.finfo(before.dtype).tiny)).sqrt()
    ratio = torch.where(positive, ratio, torch.ones_like(ratio))
    eye = torch.eye(3, dtype=transform.dtype, device=transform.device)
    return {
        'shape/diagonal_mean': diagonal.mean(),
        'shape/shear_abs_mean': shear.abs().mean(),
        'shape/identity_distance': (transform - eye).square().sum((-1, -2)).sqrt().mean(),
        'shape/radius_before': (trace_before / 3).sqrt().mean(),
        'shape/radius_after': (trace_after / 3).sqrt().mean(),
        'shape/radius_ratio': ratio.mean(),
        'shape/nonzero_moment_fraction': positive.float().mean(),
        'shape/transform_volume_ratio': diagonal.prod(-1).mean(),
    }
