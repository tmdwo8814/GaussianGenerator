"""
Standalone analysis script for NoPoSplat alpha-blending behavior.

For each test sample (single target view):
  - Run the encoder → Gaussians (existing code).
  - Call the existing CUDA rasterizer 3 times for analysis:
      1) standard SH render → rendered image (for PSNR)
      2) colors_precomp = c_i (DC color of each Gaussian) → first moment image
      3) colors_precomp = c_i^2                              → second moment image
    Each call also returns `opacity` = per-pixel accumulated alpha (1 - T_final).
  - From these we derive per-pixel:
      * alpha(p) = accumulated alpha (proxy for participation strength)
      * V(p)     = contribution-weighted RGB variance of contributing Gaussians
      * PSNR(p)  = -10 * log10(MSE per pixel)
  - Aggregate to 32x32 patches, accumulate triplets across all samples.
  - Keep per-pixel maps for a few sanity-check samples.

Run:
    CUDA_VISIBLE_DEVICES=0 python -m src.scripts.analysis_alpha_blending \
        +experiment=re10k mode=test \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
        checkpointing.load=PATH/TO/CKPT \
        wandb.mode=disabled

Output:
    outputs/alpha_blending_analysis.npz
"""

from math import isqrt
from pathlib import Path

import hydra
import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import rearrange
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from tqdm import tqdm

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.geometry.projection import get_fov
    from src.misc.step_tracker import StepTracker
    from src.misc.weight_modify import checkpoint_filter_fn
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import get_projection_matrix
    from src.model.encoder import get_encoder


PATCH_SIZE = 32
SANITY_IDS = {0, 1000, 3000}
OUTPUT_PATH = Path("outputs/alpha_blending_analysis.npz")


# ----------------------------------------------------------------------------
# Per-sample analysis: 3 rasterizer calls reusing the *exact same* CUDA code.
# Logic is duplicated from render_cuda() but does NOT modify it.
# ----------------------------------------------------------------------------
@torch.no_grad()
def analyze_one_sample(
    extrinsics: torch.Tensor,    # (1, 4, 4)  — single target view, c2w
    intrinsics: torch.Tensor,    # (1, 3, 3)  — normalized intrinsics
    near: torch.Tensor,          # (1,)
    far: torch.Tensor,           # (1,)
    image_shape: tuple[int, int],
    background_color: torch.Tensor,  # (1, 3)
    means: torch.Tensor,         # (1, G, 3)
    covariances: torch.Tensor,   # (1, G, 3, 3)
    sh: torch.Tensor,            # (1, G, 3, d_sh)
    opacities: torch.Tensor,     # (1, G)
    scale_invariant: bool = True,
):
    """Returns rendered (3,H,W), alpha (H,W), V (H,W). All on device."""
    # --- exactly the same preprocessing as render_cuda() ---
    if scale_invariant:
        scale = 1.0 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        covariances = covariances * (scale[:, None, None, None] ** 2)
        means = means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n_sh = sh.shape
    degree = isqrt(n_sh) - 1
    shs_perm = rearrange(sh, "b g xyz n -> b g n xyz").contiguous()

    h, w = image_shape
    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    proj = get_projection_matrix(near, far, fov_x, fov_y)
    proj = rearrange(proj, "b i j -> b j i")
    view_mat = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_proj = view_mat @ proj

    # Settings exactly like render_cuda
    settings = GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=tan_fov_x[0].item(),
        tanfovy=tan_fov_y[0].item(),
        bg=background_color[0],
        scale_modifier=1.0,
        viewmatrix=view_mat[0],
        projmatrix=full_proj[0],
        projmatrix_raw=proj[0],
        sh_degree=degree,
        campos=extrinsics[0, :3, 3],
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(settings)

    row, col = torch.triu_indices(3, 3)
    cov_packed = covariances[0, :, row, col]
    means_g = means[0]
    opa = opacities[0, ..., None]
    means2D = torch.zeros_like(means_g, requires_grad=True)
    try:
        means2D.retain_grad()
    except Exception:
        pass

    # Pass 1 — standard SH render (used for PSNR + alpha)
    image, _, _, alpha_pix, _ = rasterizer(
        means3D=means_g,
        means2D=means2D,
        shs=shs_perm[0],
        colors_precomp=None,
        opacities=opa,
        cov3D_precomp=cov_packed,
        theta=None,
        rho=None,
    )

    # Black-bg settings for analysis passes (avoid bg leaking into moments)
    settings_black = GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=tan_fov_x[0].item(),
        tanfovy=tan_fov_y[0].item(),
        bg=torch.zeros_like(background_color[0]),
        scale_modifier=1.0,
        viewmatrix=view_mat[0],
        projmatrix=full_proj[0],
        projmatrix_raw=proj[0],
        sh_degree=degree,
        campos=extrinsics[0, :3, 3],
        prefiltered=False,
        debug=False,
    )
    rasterizer_black = GaussianRasterizer(settings_black)

    # Per-Gaussian DC color (same formula 3DGS uses internally)
    C0 = 0.28209479177387814
    rgb_dc = (0.5 + C0 * sh[0, :, :, 0]).clamp(0.0, 1.0)  # (G, 3)

    # Pass 2 — first moment: sum_i T_i a_i * c_i
    img_first, _, _, _, _ = rasterizer_black(
        means3D=means_g,
        means2D=means2D,
        shs=None,
        colors_precomp=rgb_dc,
        opacities=opa,
        cov3D_precomp=cov_packed,
        theta=None,
        rho=None,
    )

    # Pass 3 — second moment: sum_i T_i a_i * c_i^2
    img_second, _, _, _, _ = rasterizer_black(
        means3D=means_g,
        means2D=means2D,
        shs=None,
        colors_precomp=rgb_dc ** 2,
        opacities=opa,
        cov3D_precomp=cov_packed,
        theta=None,
        rho=None,
    )

    # Combine: V = E[c^2] - (E[c])^2, weighted by T*a (normalize by alpha_pix)
    A = alpha_pix.clamp(min=1e-6)            # (1, H, W)
    mean_c = img_first / A                   # (3, H, W)
    second_c = img_second / A                # (3, H, W)
    var_c = (second_c - mean_c ** 2).clamp(min=0.0)
    V_map = var_c.mean(dim=0)                # (H, W) — average over RGB

    return image, alpha_pix.squeeze(0), V_map  # (3,H,W), (H,W), (H,W)


# ----------------------------------------------------------------------------
def patches_mean(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(H, W) -> (Hp*Wp,) patch-mean."""
    H, W = x.shape
    Hp, Wp = H // patch, W // patch
    return rearrange(
        x[: Hp * patch, : Wp * patch].float(),
        "(hp ph) (wp pw) -> (hp wp) (ph pw)",
        ph=patch, pw=patch,
    ).mean(dim=-1)


# ----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    device = torch.device("cuda")

    # Build encoder + decoder
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    # Load checkpoint
    ckpt_path = cfg.checkpointing.load
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = {k[len("encoder."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
        encoder.load_state_dict(sd, strict=False)
    elif "model" in ckpt:
        sd = checkpoint_filter_fn(ckpt["model"], encoder)
        encoder.load_state_dict(sd, strict=False)
    print("Checkpoint loaded.")

    # Data
    step_tracker = StepTracker()
    dm = DataModule(cfg.dataset, cfg.data_loader, step_tracker, global_rank=0)
    dm.setup("test")
    test_loader = dm.test_dataloader()
    data_shim = encoder.get_data_shim()

    # Background color from decoder cfg (default white in NoPoSplat)
    bg = decoder.background_color.to(device)  # (3,)

    all_alpha = []
    all_V = []
    all_psnr = []
    sanity = {}

    pbar = tqdm(test_loader, desc="Analyzing")
    for sample_idx, batch in enumerate(pbar):
        # Move to device + shim
        def _to(x):
            if torch.is_tensor(x):
                return x.to(device)
            if isinstance(x, dict):
                return {k: _to(v) for k, v in x.items()}
            return x
        batch = _to(batch)
        batch = data_shim(batch)

        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Encoder
        gaussians = encoder(batch["context"], 0)

        # Take first target view only
        target_extr = batch["target"]["extrinsics"][:, 0]  # (1, 4, 4)
        target_intr = batch["target"]["intrinsics"][:, 0]  # (1, 3, 3)
        target_near = batch["target"]["near"][:, 0]        # (1,)
        target_far = batch["target"]["far"][:, 0]
        target_gt = batch["target"]["image"][0, 0]         # (3, H, W)

        rendered, alpha_map, V_map = analyze_one_sample(
            extrinsics=target_extr,
            intrinsics=target_intr,
            near=target_near,
            far=target_far,
            image_shape=(h, w),
            background_color=bg.unsqueeze(0),
            means=gaussians.means,
            covariances=gaussians.covariances,
            sh=gaussians.harmonics,
            opacities=gaussians.opacities,
            scale_invariant=decoder.make_scale_invariant,
        )

        # Per-pixel PSNR
        mse_map = ((rendered - target_gt) ** 2).mean(dim=0)
        psnr_map = -10.0 * torch.log10(mse_map.clamp(min=1e-10))

        # Aggregate to patches
        a_p = patches_mean(alpha_map, PATCH_SIZE).cpu().numpy()
        v_p = patches_mean(V_map, PATCH_SIZE).cpu().numpy()
        psnr_p = patches_mean(psnr_map, PATCH_SIZE).cpu().numpy()
        all_alpha.append(a_p)
        all_V.append(v_p)
        all_psnr.append(psnr_p)

        if sample_idx in SANITY_IDS:
            sanity[sample_idx] = {
                "gt": target_gt.cpu().numpy(),
                "rendered": rendered.cpu().numpy(),
                "alpha_map": alpha_map.cpu().numpy(),
                "var_map": V_map.cpu().numpy(),
                "psnr_map": psnr_map.cpu().numpy(),
            }

        pbar.set_postfix({"alpha": float(a_p.mean()), "V": float(v_p.mean()), "psnr": float(psnr_p.mean())})

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "alpha": np.concatenate(all_alpha, axis=0),
        "V": np.concatenate(all_V, axis=0),
        "psnr": np.concatenate(all_psnr, axis=0),
    }
    for sid, d in sanity.items():
        for k, v in d.items():
            payload[f"sanity_{sid}_{k}"] = v
    np.savez_compressed(OUTPUT_PATH, **payload)
    print(f"\nSaved {payload['alpha'].shape[0]} patches → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()