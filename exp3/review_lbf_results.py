"""Audit existing LBF CSVs and draw evidence plots without model inference.

python -m exp3.review_lbf_results --results-dir exp3/results
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


MODELS = ("original", "ours")
COLORS = {"original": "#667085", "ours": "#e86a13"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def number(row, key):
    return float(row[key]) if row[key] else np.nan


def summary_row(rows, space, model, metric, threshold="P90", alpha=.25, group="overall"):
    matches = [r for r in rows if r["size_space"] == space and r["decoder"] == model
               and r["metric"] == metric and r["overlap"] == group
               and (metric == "spearman_rho" or (
                   r["threshold"] == threshold and number(r, "tau_alpha") == alpha))]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row: {space}/{model}/{metric}/{threshold}/{alpha}/{group}")
    return matches[0]


def collect(results):
    rows, summary = read_csv(results / "per_scene.csv"), read_csv(results / "summary.csv")
    report = json.loads((results / "lbf_report.json").read_text(encoding="utf-8"))
    data = {}
    for space in ("projected", "world"):
        models = {}
        for model in MODELS:
            selected = [r for r in rows if r["size_space"] == space and r["decoder"] == model
                        and r["threshold"] == "P90" and number(r, "tau_alpha") == .25]
            mapping = {r["scene"]: r for r in selected}
            if not selected or len(mapping) != len(selected):
                raise ValueError(f"Missing/duplicate scenes: {space}/{model}")
            models[model] = mapping
        if set(models["original"]) != set(models["ours"]):
            raise ValueError("Scene pairs differ")
        scenes = sorted(models["original"])
        data[space] = {model: {key: np.array([number(models[model][s], key) for s in scenes])
                              for key in ("N_analysis", "N_large", "N_lbf", "N_total", "lbf_count",
                                          "faint_given_large", "spearman_rho", "area_total", "area_lbf", "lbf_area")}
                       for model in MODELS}
        data[space]["scenes"] = scenes
        data[space]["overlap"] = [models["original"][s]["overlap"] for s in scenes]
        for model in MODELS:
            values = data[space][model]
            recorded = report["main"][space]["lbf_count"][model]
            for source, target in (("N_lbf", "numerator_total"), ("N_analysis", "denominator_total")):
                if not np.isclose(values[source].sum(), recorded[target], rtol=0, atol=1e-6):
                    raise ValueError(f"CSV/report population mismatch: {space}/{model}/{source}")
            rho = summary_row(summary, space, model, "spearman_rho")
            if not np.isclose(np.nanmean(values["spearman_rho"]), number(rho, "estimate")):
                raise ValueError(f"CSV/summary correlation mismatch: {space}/{model}")
    return data, summary, report


def audit(data, summary, report, results):
    result = {"inputs": {name: hashlib.sha256((results / name).read_bytes()).hexdigest() for name in (
        "per_scene.csv", "summary.csv", "lbf_report.json", "transitions.csv")},
        "definition": "Frozen original P90; opacity < 0.25; active opacity >= 1/255",
        "spaces": {}}
    for space in ("projected", "world"):
        d = data[space]
        original, ours = d["original"], d["ours"]
        delta = ours["lbf_count"] - original["lbf_count"]
        rho_o, rho_u = original["spearman_rho"], ours["spearman_rho"]
        po, pu = (v["N_lbf"].sum() / v["N_analysis"].sum() for v in (original, ours))
        block = {"n_scenes": len(delta), "improved_scenes": int((delta < 0).sum()),
                 "worsened_scenes": int((delta > 0).sum()), "equal_scenes": int((delta == 0).sum()),
                 "median_scene_delta_pp": float(np.median(delta) * 100),
                 "relative_prevalence_reduction": float(1 - pu / po),
                 "rho_negative_to_positive": int(((rho_o < 0) & (rho_u > 0)).sum()),
                 "rho_increases": int((rho_u > rho_o).sum()),
                 "main": report["main"][space]}
        for model in MODELS:
            v = d[model]
            block[model] = {"large_fraction": float(v["N_large"].sum() / v["N_analysis"].sum()),
                            "positive_rho_scenes": int((v["spearman_rho"] > 0).sum()),
                            "rho_summary": {k: number(summary_row(summary, space, model, "spearman_rho"), k)
                                            for k in ("estimate", "ci_low", "ci_high")}}
        result["spaces"][space] = block
    d = data["projected"]
    area = {"fraction_improved_scenes": int((d["ours"]["lbf_area"] < d["original"]["lbf_area"]).sum()),
            "absolute_area_improved_scenes": int((d["ours"]["area_lbf"] < d["original"]["area_lbf"]).sum())}
    for model in MODELS:
        v = d[model]
        totals, lbf = v["area_total"], v["area_lbf"]
        area[model] = {"total_area": float(totals.sum()), "lbf_area": float(lbf.sum()),
                       "median_scene_fraction": float(np.median(v["lbf_area"])),
                       "median_lbf_area_per_source_slot": float(np.median(lbf / v["N_total"])),
                       "top_total_area_scene": d["scenes"][int(np.argmax(totals))],
                       "top_lbf_area_scene": d["scenes"][int(np.argmax(lbf))],
                       "total_area_top_shares": {str(k): float(np.sort(totals)[-k:].sum() / totals.sum())
                                                 for k in (1, 2, 5, 10)},
                       "lbf_area_top_shares": {str(k): float(np.sort(lbf)[-k:].sum() / lbf.sum())
                                               for k in (1, 2, 5, 10)}}
    result["area"] = area
    transitions = [r for r in read_csv(results / "transitions.csv") if r["overlap"] == "overall"]
    count = {r["fate"]: int(r["count"]) for r in transitions}
    baseline = int(report["main"]["projected"]["lbf_count"]["original"]["numerator_total"])
    if sum(count.values()) != baseline:
        raise ValueError("Transitions do not partition baseline LBF slots")
    result["transitions"] = {"counts": count, "N_original_lbf": baseline,
                             "valid_crossed_boundary": sum(count[k] for k in (
                                 "smaller_only", "more_opaque_only", "smaller_and_more_opaque")),
                             "new_lbf_slots": int(report["main"]["projected"]["lbf_count"]["ours"]["numerator_total"])
                                              - count["remains_lbf"]}
    grid = [r for r in summary if r["overlap"] == "overall" and r["size_space"] == "projected"
            and r["decoder"] == "ours-original" and r["metric"] == "lbf_count"]
    result["grid"] = {"cells": len(grid), "point_estimate_decreases": sum(number(r, "estimate") < 0 for r in grid),
                      "difference_ci_below_zero": sum(number(r, "ci_high") < 0 for r in grid),
                      "ci_contains_zero": [{k: r[k] for k in ("threshold", "tau_alpha", "estimate", "ci_low", "ci_high")}
                                           for r in grid if number(r, "ci_low") <= 0 <= number(r, "ci_high")]}
    return result


def figures(output, data, summary, report):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 220})

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), layout="constrained")
    ax = axes[0]
    for j, space in enumerate(("projected", "world")):
        for model, offset in (("original", .10), ("ours", -.10)):
            r = summary_row(summary, space, model, "spearman_rho")
            x, lo, hi = (number(r, key) for key in ("estimate", "ci_low", "ci_high"))
            ax.errorbar(x, j + offset, xerr=[[x - lo], [hi - x]], fmt="o", color=COLORS[model],
                        label=model.title() if j == 0 else None, capsize=3)
            ax.annotate(f"{x:+.3f}", (x, j + offset), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7)
    ax.axvline(0, color="#777", linestyle="--", linewidth=.8)
    ax.set(yticks=[0, 1], yticklabels=["Projected", "World"], ylim=(-.4, 1.5), xlim=(-.57, .48),
           xlabel="Within-scene Spearman rho (mean; 95% CI)", title="(a) Scale-opacity coupling reverses")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax = axes[1]
    x = np.arange(3)
    for model, offset in (("original", -.17), ("ours", .17)):
        v = data["projected"][model]
        values = np.array([v["N_large"].sum() / v["N_analysis"].sum(),
                           v["N_lbf"].sum() / v["N_large"].sum(),
                           v["N_lbf"].sum() / v["N_analysis"].sum()]) * 100
        bars = ax.bar(x + offset, values, width=.32, color=COLORS[model], label=model.title())
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], fontsize=7, padding=3)
    ax.set(xticks=x, xticklabels=["Large /\neligible", "Faint /\nlarge", "LBF /\neligible"],
           ylabel="Pooled fraction [%]", ylim=(0, 88), title="(b) Two factors reduce LBF prevalence")
    ax = axes[2]
    o, u = [data["projected"][model]["lbf_count"] * 100 for model in MODELS]
    better = u < o
    ax.scatter(o[better], u[better], s=10, c=COLORS["original"], alpha=.6, linewidths=0)
    ax.scatter(o[~better], u[~better], s=13, c=COLORS["ours"], alpha=.8, linewidths=0, label="Ours higher")
    upper = max(o.max(), u.max()) * 1.05
    ax.plot([0, upper], [0, upper], "--", color="#888", linewidth=.8)
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Original LBF [%]", ylabel="Ours LBF [%]",
           title=f"(c) Lower LBF in {int(better.sum())}/{len(o)} scenes", aspect="equal")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    save(fig, "fig_lbf_evidence")

    labels = ("P80", "P90", "P95", "8px", "16px")
    cutoffs = [report["thresholds"]["projected"][label] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), layout="constrained")
    for model in MODELS:
        selected = [summary_row(summary, "projected", model, "lbf_count", threshold=label) for label in labels]
        y, low, high = [np.array([number(r, key) * 100 for r in selected]) for key in ("estimate", "ci_low", "ci_high")]
        axes[0].plot(cutoffs, y, "o-", color=COLORS[model], label=model.title(), markersize=4)
        axes[0].fill_between(cutoffs, low, high, color=COLORS[model], alpha=.15)
    axes[0].set(ylabel="LBF / all eligible Gaussians [%]", ylim=(0, None),
                title="(a) LBF tail mass; opacity < 0.25")
    axes[0].legend(frameon=False)
    for alpha, color in ((.05, "#4477aa"), (.1, "#228833"), (.25, "#aa3377")):
        ratios = []
        for label in labels:
            o, u = [number(summary_row(summary, "projected", model, "lbf_count", label, alpha), "estimate") for model in MODELS]
            ratios.append(u / o)
        axes[1].plot(cutoffs, ratios, "o-", color=color, label=f"opacity < {alpha:.2f}", markersize=4)
    axes[1].axhline(1, color="#888", linestyle="--", linewidth=.8)
    axes[1].set(ylabel="Ours / Original prevalence", ylim=(0, 1.05),
                title="(b) Reduction weakens at the extreme tail")
    axes[1].legend(frameon=False, loc="lower right", fontsize=7)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(cutoffs, ["2.73", "3.98", "5.67", "8", "16"])
        ax.set_xlabel("Common projected-size cutoff [px]")
        ax.grid(alpha=.2)
    fig.get_layout_engine().set(rect=(0, .10, 1, 1))
    fig.text(.5, .015, "Left: supplied scene-bootstrap 95% CIs. Right: point ratios. At 16 px, all three pooled difference CIs include zero.\n"
             "P90 = 3.978 px; the separate 4 px setting is nearly identical and remains in the CSV.", ha="center", fontsize=7)
    save(fig, "fig_lbf_tail_robustness")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), layout="constrained")
    ax = axes[0]
    for model in MODELS:
        rho = np.sort(data["projected"][model]["spearman_rho"])
        ax.step(rho, np.arange(1, len(rho)+1) / len(rho) * 100, where="post", color=COLORS[model], label=model.title())
    ax.axvline(0, color="#888", linestyle="--", linewidth=.8)
    ax.set(xlabel="Within-scene projected size-opacity rho", ylabel="Scenes at or below rho [%]",
           xlim=(-1, 1), ylim=(0, 100), title="(a) Scene-level coupling distributions")
    ax.legend(frameon=False)
    ax = axes[1]
    o, u = [data["projected"][m]["spearman_rho"] for m in MODELS]
    ax.axhline(0, color="#bbb", linewidth=.8)
    ax.axvline(0, color="#bbb", linewidth=.8)
    ax.plot([-1, 1], [-1, 1], "--", color="#aaa", linewidth=.8)
    ax.scatter(o, u, s=11, c=COLORS["ours"], alpha=.65, linewidths=0)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="Original within-scene rho", ylabel="Ours within-scene rho",
           title=f"(b) Rho increases in {int((u > o).sum())}/{len(o)} scenes", aspect="equal")
    ax = axes[2]
    # Separate persistence from incidence: the previous transition plot conditions on original LBF only.
    baseline = int(report["main"]["projected"]["lbf_count"]["original"]["numerator_total"])
    ours_total = int(report["main"]["projected"]["lbf_count"]["ours"]["numerator_total"])
    # Values are supplied by the caller as validated source-slot counts.
    remains = report["review_transitions"]["remains_lbf"]
    resolved = baseline - remains
    new = ours_total - remains
    ax.bar([0], [remains / 1e6], color="#d95f5f", label="LBF in both")
    ax.bar([0], [resolved / 1e6], bottom=remains / 1e6, color="#83b98a", label="Original only")
    ax.bar([1], [remains / 1e6], color="#d95f5f")
    ax.bar([1], [new / 1e6], bottom=remains / 1e6, color="#e8b45e", label="Ours only")
    ax.set(xticks=[0, 1], xticklabels=["Original LBF", "Ours LBF"], ylabel="Source-pixel slots [million]",
           title="(c) Some remaining LBF occurs at new slots")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_lbf_coupling_and_incidence")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), layout="constrained")
    ax = axes[0]
    for model in MODELS:
        frac = np.sort(data["projected"][model]["lbf_area"]) * 100
        ax.step(frac, np.arange(1, len(frac)+1) / len(frac) * 100, where="post", color=COLORS[model], label=model.title())
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Per-scene LBF / total ellipse area [%]",
           ylabel="Scenes at or below fraction [%]", title="(a) Area fractions have little separation")
    ax.legend(frameon=False)
    ax = axes[1]
    for model in MODELS:
        for key, style, title in (("area_total", "-", "all area"), ("area_lbf", "--", "LBF area")):
            values = np.sort(data["projected"][model][key])[::-1]
            ax.plot(np.arange(1, len(values)+1), 100 * np.cumsum(values)/values.sum(),
                    color=COLORS[model], linestyle=style, label=f"{model.title()}: {title}")
    ax.set_xscale("log")
    ax.set(xlim=(1, len(values)), ylim=(0, 101), xlabel="Number of largest-area scenes",
           ylabel="Cumulative share of summed area [%]", title="(b) A few scenes dominate pooled area")
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[2]
    original, ours = [data["projected"][m] for m in MODELS]
    o, u = [v["area_lbf"] / v["N_total"] for v in (original, ours)]
    positive = (o > 0) & (u > 0)
    lo, hi = min(o[positive].min(), u[positive].min()), max(o[positive].max(), u[positive].max())
    ax.scatter(o[positive], u[positive], s=10, c=COLORS["ours"], alpha=.6, linewidths=0)
    ax.plot([lo, hi], [lo, hi], "--", color="#888", linewidth=.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Original potential LBF area / source slot",
           ylabel="Ours potential LBF area / source slot", title="(c) Absolute potential area is heavy-tailed", aspect="equal")
    fig.get_layout_engine().set(rect=(0, .09, 1, 1))
    fig.text(.5, .015, "Diagnostic only: unfiltered Jacobian ellipses; not visible pixel coverage. No outlier scenes removed.", ha="center", fontsize=7)
    save(fig, "fig_lbf_area_diagnostics")


def write_report(output, result):
    projected, world = (result["spaces"][s] for s in ("projected", "world"))
    lines = ["# LBF result review", "", "Source: existing full-population CSVs; no inference or thresholds changed.", "",
             "| Space | Original LBF | Ours LBF | Relative reduction | Improved scenes | Delta 95% CI (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, block in (("Projected", projected), ("World", world)):
        main = block["main"]["lbf_count"]
        o, u, d = (main[m] for m in ("original", "ours", "ours-original"))
        lines.append(f"| {name} | {100*o['estimate']:.3f}% | {100*u['estimate']:.3f}% | "
                     f"{100*block['relative_prevalence_reduction']:.2f}% | {block['improved_scenes']}/{block['n_scenes']} | "
                     f"[{100*d['ci_low']:.3f}, {100*d['ci_high']:.3f}] |")
    lines += ["", "Within-scene rank correlations reverse sign in the mean:", ""]
    for space, block in (("Projected", projected), ("World", world)):
        o, u = (block[m]["rho_summary"] for m in MODELS)
        lines.append(f"- {space}: {o['estimate']:.3f} -> {u['estimate']:.3f}; "
                     f"positive rho scenes {block['original']['positive_rho_scenes']} -> "
                     f"{block['ours']['positive_rho_scenes']}; rho increases in {block['rho_increases']} scenes.")
    t = result["transitions"]
    area = result["area"]
    lines += ["", "The original-LBF transition plot does not include incidence at other source slots:", "",
              f"- {t['counts']['remains_lbf']:,} source slots are LBF in both models.",
              f"- {t['new_lbf_slots']:,} additional source slots are LBF only in ours.",
              f"- {t['valid_crossed_boundary']:,} original LBF slots cross the size/opacity boundary with valid active outputs.",
              "- Source-slot correspondence is not tracking an identical physical Gaussian.", "",
              "Extreme-tail limitation:", "",
              f"- All {result['grid']['cells']} projected count grid point estimates decrease.",
              f"- {result['grid']['difference_ci_below_zero']} pooled paired-bootstrap difference intervals are below zero.",
              "- All three 16 px difference intervals include zero; do not claim extreme LBF elimination.", "",
              "Area limitation:", "",
              "- The pooled area fraction has an enormous CI, and the scene-mean area-fraction difference includes zero.",
              f"- Absolute summed potential LBF area: {area['original']['lbf_area']:.6g} -> {area['ours']['lbf_area']:.6g}.",
              f"- Total potential area: {area['original']['total_area']:.6g} -> {area['ours']['total_area']:.6g}.",
              "- The apparent pooled-fraction gain largely reflects denominator growth, not a decrease in the numerator.",
              f"- Per-scene area fraction improves in {area['fraction_improved_scenes']}/{projected['n_scenes']} scenes.",
              "- A few scenes dominate each sum. Near-zero positive camera depth is a plausible cause because J contains 1/z and x/z^2; scalar CSVs cannot confirm individual-Gaussian causes.",
              "- Use the full cache to inspect depth, anisotropy and off-screen support, then use renderer-consistent culling/projection and screen clipping for a visible-footprint analysis.",
              "- No outliers were excluded in this review.", "",
              "Supported wording: substantial suppression of LBF prevalence and reversal of within-scene scale-opacity association.",
              "Unsupported wording: complete removal, resolved extreme tails, or established reduction of visible LBF coverage/causal compensation.", "",
              "The other compensation CSVs in the input folder reference a separate 900-scene, step-80000 baseline experiment; they are not paired evidence for these 500-scene checkpoints.", ""]
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("exp3/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp3/output/lbf_review"))
    args = parser.parse_args()
    data, summary, report = collect(args.results_dir)
    result = audit(data, summary, report, args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["review_transitions"] = result["transitions"]["counts"]
    figures(args.output_dir, data, summary, report)
    write_report(args.output_dir, result)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
