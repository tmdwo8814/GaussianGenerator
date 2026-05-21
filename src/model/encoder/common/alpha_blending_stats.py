"""
alpha_blending_stats.py

Stage 3: GaussianRefinementDecoder
  Input : Gaussian_1st attributes (B, 88, H, W)
          stats_pred               (B,  2, H, W)  ← from DPT StatsAuxHead
          RGB image                (B,  3, H, W)
  Output: delta (B, 76, H, W),  Tanh * 0.1
          76 = opacity(1) + harmonics(75)

NOTE:
  - AlphaBlendingPredictor is REMOVED.
    stats_pred now comes from StatsAuxHead inside the DPT gs_params head,
    so L_stat gradient propagates all the way back to the ViT backbone.
  - means delta is REMOVED (geometry stability, negligible effect observed).
  - covariance delta is REMOVED (PSD constraint).
  - GaussianRefinementDecoder head is zero-initialized to avoid
    the early-training performance dip seen in alpha1 experiment.
"""

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# Stage 3: GaussianRefinementDecoder
# ---------------------------------------------------------------------------

class GaussianRefinementDecoder(nn.Module):
    """
    Produces per-pixel Gaussian attribute offsets conditioned on:
      - 1st-stage Gaussian features   (B, 88, H, W)
      - alpha-blending stats_pred     (B,  2, H, W)  from DPT aux head
      - RGB context image             (B,  3, H, W)

    Output delta layout (76ch):
      ch 0      : Δopacity   (1)
      ch 1 - 75 : Δharmonics (75 = 3 * 25)

    means and covariances are NOT updated.
    """

    GAUSS_CH_IN = 88
    STATS_CH    = 2
    RGB_CH      = 3
    DELTA_OUT   = 76
    DELTA_SCALE = 0.1

    def __init__(self):
        super().__init__()

        self.gauss_branch = ConvBnRelu(self.GAUSS_CH_IN, 64)

        self.rgb_branch = nn.Sequential(
            ConvBnRelu(self.RGB_CH, 32, kernel=7, padding=3),
            ConvBnRelu(32, 64),
        )

        self.fusion = ConvBnRelu(128, 64)

        self.stats_cond = nn.Sequential(
            nn.Conv2d(64 + self.STATS_CH, 64, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            ConvBnRelu(64, 64),
            nn.Conv2d(64, self.DELTA_OUT, kernel_size=1, bias=True),
            nn.Tanh(),
        )

        # Zero-init: delta ≈ 0 at start → Gaussian_2nd ≈ Gaussian_1st early on
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(
        self,
        gauss_feat:  torch.Tensor,   # (B, 88, H, W)
        stats_pred:  torch.Tensor,   # (B,  2, H, W)
        rgb:         torch.Tensor,   # (B,  3, H, W)
    ) -> torch.Tensor:               # (B, 76, H, W)

        b1    = self.gauss_branch(gauss_feat)
        b2    = self.rgb_branch(rgb)
        fused = self.fusion(torch.cat([b1, b2], dim=1))
        cond  = self.stats_cond(torch.cat([fused, stats_pred], dim=1))
        return self.head(cond) * self.DELTA_SCALE


# ---------------------------------------------------------------------------
# Helper: build pixel-aligned feature map from Gaussians (per view)
# ---------------------------------------------------------------------------

def build_gauss_feat(
    means:       torch.Tensor,   # (B, G, 3)
    opacities:   torch.Tensor,   # (B, G)
    covariances: torch.Tensor,   # (B, G, 3, 3)
    harmonics:   torch.Tensor,   # (B, G, 3, d_sh)
    h: int,
    w: int,
) -> torch.Tensor:               # (B, 88, H, W)
    """
    88ch = mean(3) + opacity(1) + cov_flat(9) + harm_flat(75)
    Covariance is included as context even though delta does not update it.
    """
    cov_flat  = rearrange(covariances, "b g i j -> b g (i j)")
    harm_flat = rearrange(harmonics,   "b g c d -> b g (c d)")
    opa_exp   = opacities.unsqueeze(-1)

    feat = torch.cat([means, opa_exp, cov_flat, harm_flat], dim=-1)  # (B, G, 88)
    return rearrange(feat, "b (h w) c -> b c h w", h=h, w=w)        # (B, 88, H, W)


# ---------------------------------------------------------------------------
# Helper: apply delta
# ---------------------------------------------------------------------------

def apply_delta(
    means:        torch.Tensor,   # (B, G, 3)
    opacities:    torch.Tensor,   # (B, G)
    covariances:  torch.Tensor,   # (B, G, 3, 3)
    harmonics:    torch.Tensor,   # (B, G, 3, d_sh)
    delta:        torch.Tensor,   # (B, 76, H, W)
    h: int,
    w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply delta to opacity and harmonics only.
    Delta layout: ch0=Δopacity, ch1-75=Δharmonics.
    """
    d = rearrange(delta, "b c h w -> b (h w) c")    # (B, G, 76)

    d_opa  = d[..., 0]                               # (B, G)
    d_harm = d[..., 1:].reshape(*d.shape[:2], 3, -1) # (B, G, 3, 25)

    # Scale check (once per process)
    if not getattr(apply_delta, "_printed", False):
        apply_delta._printed = True
        print("\n===== Gaussian Attribute Scale Check =====")
        print(f"[means]     NOT refined (geometry stability)")
        print(f"[opacity]   val  | mean={opacities.abs().mean():.4f}, max={opacities.abs().max():.4f}")
        print(f"[opacity]   delta| mean={d_opa.abs().mean():.4f}, max={d_opa.abs().max():.4f}")
        print(f"[harmonics] val  | mean={harmonics.abs().mean():.4f}, max={harmonics.abs().max():.4f}")
        print(f"[harmonics] delta| mean={d_harm.abs().mean():.4f}, max={d_harm.abs().max():.4f}")
        print(f"[ratio opacity]   {(d_opa.abs().mean()/(opacities.abs().mean()+1e-8)):.4f}")
        print(f"[ratio harmonics] {(d_harm.abs().mean()/(harmonics.abs().mean()+1e-8)):.4f}")
        print("==========================================\n")

    return (
        means,
        (opacities + d_opa).clamp(0.0, 1.0),
        covariances,
        harmonics + d_harm,
    )