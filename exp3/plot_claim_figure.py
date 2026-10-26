"""Draw the measured scale-opacity association and LBF prevalence panels.

Prefer original per_scene.csv when available. The existing vector scatter PDF
also retains all 500 pairs, with enough precision to reproduce the distributions.
Those coordinates are recovered as data, never as synthetic Gaussian samples.
"""

import argparse
import csv
import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np


MODELS = ("original", "ours")
NAMES = {"original": "NoPoSplat", "ours": "Ours"}
COLORS = {"original": "#626b80", "ours": "#ea6516"}


def from_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        selected = [r for r in csv.DictReader(file) if r["size_space"] == "projected"
                    and r["threshold"] == "P90" and float(r["tau_alpha"]) == .25]
    models = {m: {r["scene"]: float(r["spearman_rho"]) for r in selected if r["decoder"] == m}
              for m in MODELS}
    scenes = sorted(models["original"])
    if set(scenes) != set(models["ours"]):
        raise ValueError("Scene pairs do not match")
    return np.array([[models[m][s] for m in MODELS] for s in scenes])


def from_vector_scatter(path, n_scenes):
    """Recover paired rho from PDF marker translations in the [-1,1]^2 axes.

    PDF line plots may be simplified, so use scatter markers, not the CDF path.
    Matplotlib encodes marker positions as cumulative translation matrices.
    We require one unique n_scene-point square scatter and then validate it
    against independently saved means, sign counts and paired statistics.
    """
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    rectangle = re.compile(rf"(?:Q\s+)?q\s+({number})\s+({number})\s+({number})\s+({number})\s+re W n(.*?)\nQ", re.S)
    translation = re.compile(rf"1 0 0 1 ({number}) ({number}) cm /M\d+ Do")
    candidates = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", path.read_bytes(), re.S):
        try:
            stream = zlib.decompress(match.group(1)).decode("latin1")
        except zlib.error:
            continue
        for record in rectangle.finditer(stream):
            rect = np.array(record.groups()[:4], dtype=float)
            commands = translation.findall(record.group(5))
            if len(commands) != n_scenes or not np.isclose(rect[2], rect[3]):
                continue
            positions = np.cumsum(np.array(commands, dtype=float), axis=0)
            candidates.append(2 * (positions - rect[:2]) / rect[2:] - 1)
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely recover the complete paired scatter; supply --per-scene")
    return candidates[0]


def validate_pairs(pairs, block):
    if pairs.shape != (block["n_scenes"], 2) or not np.isfinite(pairs).all() or np.any(np.abs(pairs) > 1):
        raise ValueError("Invalid scene correlations")
    for j, model in enumerate(MODELS):
        np.testing.assert_allclose(pairs[:, j].mean(), block[model]["rho_summary"]["estimate"], atol=1e-8, rtol=0)
        if int((pairs[:, j] > 0).sum()) != block[model]["positive_rho_scenes"]:
            raise ValueError("Positive-correlation scene count differs")
    if int((pairs[:, 1] > pairs[:, 0]).sum()) != block["rho_increases"]:
        raise ValueError("Paired correlation changes differ")
    if int(((pairs[:, 0] < 0) & (pairs[:, 1] > 0)).sum()) != block["rho_negative_to_positive"]:
        raise ValueError("Negative-to-positive scene count differs")


def reflected_density(grid, values, bandwidth):
    def gaussian(x):
        distance = (x[:, None] - values) / bandwidth
        return np.exp(-.5 * distance**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return gaussian(grid) + gaussian(-2 - grid) + gaussian(2 - grid)


def draw_coupling(ax, pairs, block, bandwidth):
    grid = np.linspace(-1, 1, 1601)
    curves = {m: reflected_density(grid, pairs[:, j], bandwidth) for j, m in enumerate(MODELS)}
    peak = max(y.max() for y in curves.values())
    ax.axvline(0, color="#929292", linestyle=(0, (3, 3)), linewidth=.9, zorder=0)
    for j, model in enumerate(MODELS):
        color, density = COLORS[model], curves[model]
        ax.fill_between(grid, density, color=color, alpha=.24, linewidth=0)
        ax.plot(grid, density, color=color, linewidth=1.8)
        mean = block[model]["rho_summary"]["estimate"]
        y = float(np.interp(mean, grid, density))
        ax.plot([mean, mean], [0, y], color=color, linestyle=(0, (2, 2)), linewidth=.8, alpha=.9)
        # One mark per measured scene; no subsampling or distribution simulation.
        ax.plot(pairs[:, j], np.full(len(pairs), -.025 * peak), "|", markersize=3.5,
                markeredgewidth=.45, color=color, alpha=.25, clip_on=False)
        ax.text(mean, peak * 1.07, f"{NAMES[model]}\nmean $\\rho$ = {mean:+.2f}",
                color=color, ha="center", va="bottom", fontsize=10)
    ax.set(xlim=(-1, 1), ylim=(0, peak * 1.40), xlabel=r"Within-scene Spearman correlation $\rho$", ylabel="Density of scenes")
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_title("(a) Scale–opacity relation", loc="left", fontsize=12, pad=15)
    ax.text(.23, -.25, "Larger tends to be fainter", transform=ax.transAxes, ha="center", fontsize=8, color=COLORS["original"])
    ax.text(.77, -.25, "Larger tends to be more opaque", transform=ax.transAxes, ha="center", fontsize=8, color=COLORS["ours"])


def draw_prevalence(ax, block):
    main = block["main"]["lbf_count"]
    values = np.array([main[m]["estimate"] for m in MODELS]) * 100
    low = np.array([main[m]["ci_low"] for m in MODELS]) * 100
    high = np.array([main[m]["ci_high"] for m in MODELS]) * 100
    ax.bar([0, 1], values, width=.48, color=[COLORS[m] for m in MODELS], zorder=3)
    ax.errorbar([0, 1], values, yerr=[values - low, high - values], fmt="none", color="#252525",
                capsize=3, elinewidth=.9, capthick=.9, zorder=4)
    for x, y, top in zip([0, 1], values, high):
        ax.text(x, top + .20, f"{y:.2f}%", ha="center", va="bottom", fontsize=11)
    bracket = high.max() + 1.35
    ax.plot([0, 0, 1, 1], [bracket-.18, bracket, bracket, bracket-.18], color="#333", linewidth=.8)
    ax.text(.5, bracket+.33, f"{100 * block['relative_prevalence_reduction']:.1f}% lower", ha="center",
            va="bottom", fontsize=12, fontweight="bold", color=COLORS["ours"])
    ax.set(xlim=(-.6, 1.6), ylim=(0, bracket+1.7), ylabel="LBF prevalence (%)")
    ax.set_xticks([0, 1], [NAMES[m] for m in MODELS])
    ax.set_yticks(np.arange(0, 9, 2))
    ax.yaxis.grid(True, color="#e9e9e9", linewidth=.65, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(b) Large-but-faint prevalence", loc="left", fontsize=12, pad=15)
    ax.text(.5, -.25, f"Lower LBF in {block['improved_scenes']}/{block['n_scenes']} scenes", transform=ax.transAxes,
            ha="center", fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("exp3/output/lbf_review/review.json"))
    parser.add_argument("--per-scene", type=Path)
    parser.add_argument("--scatter-pdf", type=Path,
                        default=Path("exp3/output/lbf_review/fig_lbf_coupling_and_incidence.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp3/output/lbf_claim"))
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    block = review["spaces"]["projected"]
    source = args.per_scene if args.per_scene else args.scatter_pdf
    pairs = from_csv(source) if args.per_scene else from_vector_scatter(source, block["n_scenes"])
    validate_pairs(pairs, block)
    # Common absolute bandwidth: Scott's rule on the pooled measured correlations.
    bandwidth = float(np.std(pairs.ravel(), ddof=1) * pairs.size**(-1/5))
    if bandwidth <= 0:
        raise ValueError("Degenerate correlation distribution")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.labelsize": 10, "axes.linewidth": .85,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "path.simplify": False})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.65), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=.085, right=.98, bottom=.29, top=.82, wspace=.34)
    draw_coupling(axes[0], pairs, block, bandwidth)
    draw_prevalence(axes[1], block)
    threshold = block["main"]["lbf_count"]["original"]["tau_s"]
    footer = (f"{block['n_scenes']} test scenes; one correlation per scene. LBF: projected size > baseline P90 ({threshold:.2f} px), opacity < 0.25.\n"
              "Active Gaussians with positive source-camera depth. Error bars: 95% scene-bootstrap confidence intervals.")
    fig.text(.5, .015, footer, ha="center", va="bottom", fontsize=7.3, linespacing=1.4)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"fig_lbf_claim.{suffix}", dpi=400)
    plt.close(fig)
    for name, function in (("a_coupling", lambda ax: draw_coupling(ax, pairs, block, bandwidth)),
                           ("b_lbf_prevalence", lambda ax: draw_prevalence(ax, block))):
        single, ax = plt.subplots(figsize=(4.4, 3.4))
        single.subplots_adjust(left=.16, right=.97, bottom=.25, top=.82)
        function(ax)
        for suffix in ("png", "pdf"):
            single.savefig(args.output_dir / f"panel_{name}.{suffix}", dpi=400)
        plt.close(single)
    with (args.output_dir / "scene_correlations.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("plot_index", "original_rho", "ours_rho"))
        writer.writerows((i, *row) for i, row in enumerate(pairs))
    metadata = {"correlation_source": str(source.resolve()),
                "correlation_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "correlation_extraction": "original CSV" if args.per_scene else "all paired vector PDF scatter coordinates; inverse axis transform",
                "review_source": str(args.review.resolve()),
                "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
                "n_scenes": block["n_scenes"], "mean_rho": pairs.mean(axis=0).tolist(),
                "validation": "means within 1e-8; exact positive, increasing and sign-flip scene counts",
                "kde": {"kernel": "Gaussian", "bandwidth": bandwidth, "same_bandwidth_both_models": True,
                        "boundaries": "reflection at -1 and +1"},
                "prevalence": block["main"]["lbf_count"],
                "note": "plot_index is scatter order, not a scene identifier; no synthetic scene values"}
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
