"""
Plot results from analysis_alpha_blending.py.

Reads:  outputs/alpha_blending_analysis.npz
Writes: outputs/alpha_blending_figures/{fig_A_main.png, fig_B_sanity.png}

Usage:
    python -m src.scripts.plot_alpha_blending
        [--npz outputs/alpha_blending_analysis.npz]
        [--out outputs/alpha_blending_figures]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def bin_and_average(x, y, n_bins=10):
    """Bin x into n_bins quantile bins; return centers, mean(y), std(y)."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    centers, means, stds = [], [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() < 5:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(y[m].mean())
        stds.append(y[m].std())
    return np.array(centers), np.array(means), np.array(stds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/alpha_blending_analysis.npz", type=Path)
    ap.add_argument("--out", default="outputs/alpha_blending_figures", type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz)

    alpha = data["alpha"]
    V = data["V"]
    psnr = data["psnr"]

    valid = np.isfinite(psnr) & np.isfinite(alpha) & np.isfinite(V)
    alpha, V, psnr = alpha[valid], V[valid], psnr[valid]
    print(f"Total valid patches: {len(alpha)}")
    print(f"  alpha:  mean={alpha.mean():.4f}, median={np.median(alpha):.4f}, max={alpha.max():.4f}")
    print(f"  V:      mean={V.mean():.6f}, median={np.median(V):.6f}, max={V.max():.6f}")
    print(f"  PSNR:   mean={psnr.mean():.2f}, median={np.median(psnr):.2f}")

    # ---------- Figure A: two binned curves ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cx, my, sy = bin_and_average(alpha, psnr, n_bins=10)
    axes[0].errorbar(cx, my, yerr=sy, marker="o", capsize=3, color="steelblue")
    axes[0].set_xlabel("alpha (patch-mean accumulated alpha)")
    axes[0].set_ylabel("PSNR (patch mean)")
    axes[0].set_title("Participation strength vs PSNR")
    axes[0].grid(alpha=0.3)

    cx, my, sy = bin_and_average(V, psnr, n_bins=10)
    axes[1].errorbar(cx, my, yerr=sy, marker="o", capsize=3, color="darkorange")
    axes[1].set_xlabel("V (patch-mean color variance)")
    axes[1].set_ylabel("PSNR (patch mean)")
    axes[1].set_title("Color disagreement vs PSNR")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out / "fig_A_main.png", dpi=150, bbox_inches="tight")
    print(f"Saved {args.out / 'fig_A_main.png'}")

    try:
        from scipy.stats import pearsonr, spearmanr
        print("\nCorrelations:")
        print(f"  alpha vs PSNR: pearson={pearsonr(alpha, psnr)[0]:.4f}, "
              f"spearman={spearmanr(alpha, psnr)[0]:.4f}")
        print(f"  V vs PSNR:     pearson={pearsonr(V, psnr)[0]:.4f}, "
              f"spearman={spearmanr(V, psnr)[0]:.4f}")
    except ImportError:
        print("(scipy missing — skip)")

    # ---------- Figure B: sanity samples ----------
    sanity_ids = sorted({
        int(k.split("_")[1])
        for k in data.files
        if k.startswith("sanity_") and k.endswith("_alpha_map")
    })
    if not sanity_ids:
        print("No sanity samples.")
        return

    fig, axes = plt.subplots(len(sanity_ids), 4, figsize=(16, 4 * len(sanity_ids)))
    if len(sanity_ids) == 1:
        axes = axes[None]

    for row, sid in enumerate(sanity_ids):
        gt = data[f"sanity_{sid}_gt"]
        rendered = data[f"sanity_{sid}_rendered"]
        alpha_map = data[f"sanity_{sid}_alpha_map"]
        var_map = data[f"sanity_{sid}_var_map"]

        axes[row, 0].imshow(np.transpose(gt.clip(0, 1), (1, 2, 0)))
        axes[row, 0].set_title(f"sample {sid}: GT")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(np.transpose(rendered.clip(0, 1), (1, 2, 0)))
        axes[row, 1].set_title("rendered")
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(alpha_map, cmap="viridis")
        axes[row, 2].set_title(f"alpha (mean={alpha_map.mean():.3f})")
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

        im = axes[row, 3].imshow(var_map, cmap="magma")
        axes[row, 3].set_title(f"V (mean={var_map.mean():.4f})")
        axes[row, 3].axis("off")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046)

    fig.tight_layout()
    fig.savefig(args.out / "fig_B_sanity.png", dpi=150, bbox_inches="tight")
    print(f"Saved {args.out / 'fig_B_sanity.png'}")


if __name__ == "__main__":
    main()