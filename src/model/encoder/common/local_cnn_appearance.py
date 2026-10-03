"""Local nonlinear SH on the ORIGINAL view raster, before support allocation.

Weights are shared across views. No RGB backbone, slot-space convolution,
resampling, extra neighbor graph or camera input is used.
"""

import torch
from torch import Tensor, nn


class LocalCnnAppearance(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, sh_channels: int):
        super().__init__()
        if min(feature_dim, hidden_dim, sh_channels) < 1:
            raise ValueError('Local CNN channel dimensions must be positive')
        self.layers = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, sh_channels, kernel_size=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features: Tensor, image_size: tuple[int, int]) -> Tensor:
        """One view: [H*W,D] -> [H*W,SH], preserving its original pixel order.

        Called inside activation checkpointing by the decoder, so the dense
        hidden map is recomputed in backward rather than retained for all views.
        No spatial tiling: the convolution sees every 3x3 neighborhood intact.
        """
        h, w = image_size
        if features.ndim != 2 or features.shape != (h * w, self.layers[0].in_channels):
            raise ValueError('Local CNN requires one raster-ordered [H*W,feature_dim] view')
        image = features.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        output = self.layers(image)
        return output.permute(0, 2, 3, 1).reshape(h * w, -1)


@torch.no_grad()
def source_sh_mean_squares(base: Tensor, local: Tensor, sh_mask: Tensor) -> Tensor:
    """Masked mean squares before pooling; callers aggregate before sqrt."""
    values = []
    for coefficients in (base, local):
        masked = coefficients.reshape(-1, 3, len(sh_mask)) * sh_mask
        values.append(masked.square().mean())
    return torch.stack(values)
