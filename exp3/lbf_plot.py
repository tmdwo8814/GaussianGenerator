"""LBF joint plots with marginals, density contours and full-population labels."""

import json

import numpy as np
from scipy.stats import gaussian_kde

from exp3.lbf_metrics import FATES


def _density_contours(ax, x, y, limits, color, seed):
    if len(x) < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(x), min(8000, len(x)), replace=False)
    try:
        kde = gaussian_kde(np.vstack((x[chosen], y[chosen])))
        xx, yy = np.meshgrid(np.linspace(*limits, 100), np.linspace(0, 1, 80))
        density = kde(np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    except np.linalg.LinAlgError:
        return  # Constant/collinear fixtures cannot define a 2D KDE.
    descending = np.sort(density.ravel())[::-1]
    if not np.isfinite(descending).all() or descending.sum() <= 0:
        return
    mass = np.cumsum(descending) / descending.sum()
    levels = np.unique([descending[min(np.searchsorted(mass, p), len(descending) - 1)]
                        for p in (0.5, 0.8, 0.95)])
    ax.contour(xx, yy, density, levels=levels, colors=color, linewidths=0.85)


def make_lbf_figures(directory, thresholds, summary, transition_summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    from exp3.lbf_report import main_row

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    (directory / "figs").mkdir(exist_ok=True)
    seed = metadata["arguments"]["seed"]
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    for space in ("projected", "world"):
        samples = {}
        for name in ("original", "ours"):
            path = (directory / "lbf_samples" if space == "projected" else directory / "samples") / f"{name}.npz"
            with np.load(path, allow_pickle=False) as data:
                samples[name] = (data["log_s_px" if space == "projected" else "log_s_max"], data["opacity"])
            if not len(samples[name][0]):
                raise ValueError(f"No {space} plot population for {name}")
        x = np.concatenate([value[0] for value in samples.values()])
        cutoff = np.log10(thresholds[space]["P90"])
        low, high = min(float(x.min()), cutoff), max(float(x.max()), cutoff)
        padding = max(0.05 * (high - low), 0.04)
        low, high = low - padding, high + padding
        fig = plt.figure(figsize=(7.0, 3.45))
        outer = fig.add_gridspec(1, 2, left=0.09, right=0.98, bottom=0.18, top=0.86, wspace=0.3)
        artists = []
        marginal_axes = []
        for column, (name, title, cmap, color) in enumerate((
            ("original", "NoPoSplat (regressive)", "Greys", "#596579"),
            ("ours", "NoPoSplat + Ours", "Oranges", "#e65100"),
        )):
            grid = outer[column].subgridspec(2, 2, height_ratios=(1, 4), width_ratios=(4, 1),
                                           hspace=0.05, wspace=0.05)
            ax = fig.add_subplot(grid[1, 0])
            top = fig.add_subplot(grid[0, 0], sharex=ax)
            right = fig.add_subplot(grid[1, 1], sharey=ax)
            xx, yy = samples[name]
            artists.append(ax.hexbin(xx, yy, gridsize=(60, 40), extent=(low, high, 0, 1),
                                     mincnt=1, cmap=cmap, linewidths=0, rasterized=True))
            _density_contours(ax, xx, yy, (low, high), color, seed)
            top.hist(xx, bins=np.linspace(low, high, 70), density=True, histtype="stepfilled",
                     facecolor=color, alpha=0.3, edgecolor=color, linewidth=0.8)
            right.hist(yy, bins=np.linspace(0, 1, 60), density=True, orientation="horizontal",
                       histtype="stepfilled", facecolor=color, alpha=0.3, edgecolor=color, linewidth=0.8)
            top.axis("off")
            right.axis("off")
            marginal_axes.append((top, right))
            ax.set_xlim(low, high)
            ax.set_ylim(0, 1)
            if column == 0:
                ax.set_ylabel(r"opacity $\alpha$")
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlabel(r"projected size $\log_{10}(s_{\max} f_x / z)$ [px]" if space == "projected"
                          else r"world scale $\log_{10}(\max_k s_k)$")
            row = main_row(summary, space, name)
            eligible = int(row["denominator_total"])
            top.set_title(f"{title}\n{eligible:,} eligible; {len(xx):,} shown", pad=5)
            ax.add_patch(Rectangle((cutoff, 0), high - cutoff, 0.25,
                                   fill=False, edgecolor="#c62828", linestyle="--", linewidth=1.25))
            label = (f"LBF {100*row['estimate']:.2f}%\n"
                     f"95% CI [{100*row['ci_low']:.2f}, {100*row['ci_high']:.2f}]")
            ax.text(high - 0.025 * (high - low), 0.27, label, color="#c62828", ha="right", va="bottom",
                    fontsize=7.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1})
            ax.grid(alpha=0.15)
        norm = LogNorm(vmin=1, vmax=max(2, *(float(a.get_array().max()) for a in artists)))
        for artist in artists:
            artist.set_norm(norm)
        top_limit = max(top.get_ylim()[1] for top, _ in marginal_axes)
        right_limit = max(right.get_xlim()[1] for _, right in marginal_axes)
        for top, right in marginal_axes:
            top.set_ylim(0, top_limit)
            right.set_xlim(0, right_limit)
        fig.text(0.5, 0.025, "Pooled full-population LBF; paired scene bootstrap CI; common bins and log-count normalization",
                 ha="center", fontsize=6.5)
        for suffix in ("pdf", "png"):
            fig.savefig(directory / "figs" / f"figC_lbf_{space}.{suffix}", dpi=300)
        plt.close(fig)

    selected = {row["fate"]: row for row in transition_summary if row["overlap"] == "overall"}
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    labels = ("Inactive", "Invalid origin depth", "Smaller only", "More opaque only",
              "Smaller + more opaque", "Remains LBF")
    colors = ("#9e9e9e", "#b39ddb", "#64b5f6", "#ffb74d", "#81c784", "#e57373")
    total = selected[FATES[0]]["N_original_lbf"]
    left = 0.0
    if total:
        for key, label, color in zip(FATES, labels, colors):
            width = 100 * selected[key]["fraction"]
            ax.barh([0], [width], left=left, color=color, label=label, height=0.5)
            if width >= 5:
                ax.text(left + width / 2, 0, f"{width:.1f}%", ha="center", va="center", fontsize=8)
            left += width
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=7)
    else:
        ax.text(50, 0, "No baseline LBF slots at the main threshold", ha="center")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of original LBF source-pixel slots [%]")
    ax.set_title(f"Ours at the same origin slots; {total:,} baseline LBF slots")
    fig.subplots_adjust(bottom=0.42, left=0.06, right=0.98, top=0.82)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / "figs" / f"figC_lbf_transitions.{suffix}", dpi=300)
    plt.close(fig)
