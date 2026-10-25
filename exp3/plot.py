"""Make the two-panel figure from cached, real Gaussian samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_figure(directory: Path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    samples = {}
    for name in ("original", "ours"):
        with np.load(directory / "samples" / f"{name}.npz", allow_pickle=False) as data:
            samples[name] = {key: data[key] for key in ("log_s_max", "opacity")}
        if not len(samples[name]["opacity"]):
            raise ValueError(f"No selected Gaussians for {name}; cannot plot")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    all_x = np.concatenate([data["log_s_max"] for data in samples.values()])
    lo, hi = float(all_x.min()), float(all_x.max())
    padding = max((hi - lo) * 0.04, 0.02)
    lo, hi = lo - padding, hi + padding
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.24, top=0.83, wspace=0.14)
    artists = []
    for ax, name, title in zip(axes, ("original", "ours"), ("NoPoSplat", "Ours")):
        data = samples[name]
        artist = ax.hexbin(data["log_s_max"], data["opacity"], gridsize=(70, 45),
                           extent=(lo, hi, 0, 1), mincnt=1, cmap="viridis",
                           linewidths=0, rasterized=True)
        artists.append(artist)
        total = metadata["counts"][name]["N_selected"]
        ax.set_title(f"{title}\n{total:,} Gaussians; {len(data['opacity']):,} shown")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\log_{10}(\max_k s_k)$")
        ax.spines[["top", "right"]].set_visible(False)
    norm = LogNorm(vmin=1, vmax=max(2, *(float(a.get_array().max()) for a in artists)))
    for artist in artists:
        artist.set_norm(norm)
    axes[0].set_ylabel(r"opacity $\alpha$")
    color_axis = fig.add_axes([0.89, 0.24, 0.018, 0.58])
    fig.colorbar(artists[0], cax=color_axis, label="sampled Gaussians / bin")
    (directory / "figs").mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / "figs" / f"figC_scale_opacity.{suffix}", dpi=300)
    plt.close(fig)
