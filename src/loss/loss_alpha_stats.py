"""
LossAlphaStats: MSE between predicted and GT alpha-blending statistics.

GT is computed from Gaussian_2nd after alpha blending (training only):
  - T_GT        : 1 - accumulated_alpha  (from rasterizer opacity output)
  - sigma_D2_GT : depth variance         (two extra rasterizer calls)
"""

from dataclasses import dataclass
from math import isqrt

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.decoder.cuda_splatting import get_projection_matrix
from ..model.types import Gaussians
from ..geometry.projection import get_fov
from .loss import Loss


@dataclass
class LossAlphaStatsCfg:
    weight: float


@dataclass
class LossAlphaStatsCfgWrapper:
    alpha_stats: LossAlphaStatsCfg


class LossAlphaStats(Loss[LossAlphaStatsCfg, LossAlphaStatsCfgWrapper]):
    """
    Computes L_stat = MSE(stats_pred, stats_GT) * weight.

    stats_pred: (B, V, 2, H, W) — from encoder visualization_dump
    stats_GT  : (B, V, 2, H, W) — computed here from Gaussian_2nd

    GT channel 0: T_GT        = 1 - accumulated_alpha  (free from existing render)
    GT channel 1: sigma_D2_GT = E[z²] - E[z]²          (2 extra rasterizer calls)
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        stats_pred = getattr(gaussians, "stats_pred", None)
        if stats_pred is None:
            return torch.tensor(0.0, device=prediction.color.device)

        stats_gt = getattr(gaussians, "stats_gt", None)
        if stats_gt is None:
            return torch.tensor(0.0, device=prediction.color.device)

        # stats_pred: (B, V_ctx, 2, H, W) → (B, 2, H, W)
        # stats_gt:   (B, V_tgt, 2, H, W) → (B, 2, H, W)
        # view 차원을 평균내어 shape 통일
        stats_pred_mean = stats_pred.mean(dim=1)   # (B, 2, H, W)
        stats_gt_mean   = stats_gt.mean(dim=1)     # (B, 2, H, W)

        loss = F.mse_loss(stats_pred_mean, stats_gt_mean.detach())
        return self.cfg.weight * loss