"""Scene-cluster bootstrap, paired tests, robustness grid, and origin-slot transitions."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from exp3.lbf_metrics import (
    ACTIVE, ALPHAS, FATES, cache_path, freeze_thresholds, scene_statistics,
    transition_counts, validate_cache,
)


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def intervals(values):
    """Column-wise percentile CI; undefined columns remain explicitly undefined."""
    result = np.full((values.shape[1], 2), np.nan)
    for column in range(values.shape[1]):
        finite = values[:, column][np.isfinite(values[:, column])]
        if len(finite):
            result[column] = np.quantile(finite, [0.025, 0.975])
    return result


def bootstrap_comparison(original_num, original_den, ours_num, ours_den, samples=10000, seed=0):
    """Resample SCENES jointly across models and metrics, then recompute ratios.

    Returns both pooled fractions and arithmetic means of scene fractions.
    Paired pooled-difference CIs use the SAME sampled scenes for the two models.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in (original_num, original_den, ours_num, ours_den)]
    if any(a.shape != arrays[0].shape for a in arrays) or arrays[0].ndim != 2 or samples < 1:
        raise ValueError("Expected paired (scene,metric) numerator/denominator arrays")
    on, od, un, ud = arrays
    n, columns = on.shape
    ro, ru = ratio(on, od), ratio(un, ud)
    paired = np.isfinite(ro) & np.isfinite(ru)
    numerators = np.concatenate((on, un, np.where(np.isfinite(ro), ro, 0),
                                np.where(np.isfinite(ru), ru, 0), np.where(paired, ru - ro, 0)), axis=1)
    denominators = np.concatenate((od, ud, np.isfinite(ro), np.isfinite(ru), paired), axis=1)
    estimate = ratio(numerators.sum(0), denominators.sum(0))
    boot = np.full((samples, 5 * columns), np.nan)
    if n:
        rng = np.random.default_rng(seed)
        for start in range(0, samples, 128):
            count = min(128, samples - start)
            weights = rng.multinomial(n, np.full(n, 1 / n), size=count).astype(np.float64)
            boot[start:start + count] = ratio(weights @ numerators, weights @ denominators)
    ci = intervals(boot)
    delta = boot[:, columns:2 * columns] - boot[:, :columns]
    mean_delta = boot[:, 4 * columns:]
    return {"estimate": estimate, "ci": ci, "delta_ci": intervals(delta),
            "scene_mean_delta_ci": intervals(mean_delta), "original_ratios": ro, "ours_ratios": ru}


def paired_pvalue(original, ours):
    valid = np.isfinite(original) & np.isfinite(ours)
    if not valid.any():
        return np.nan
    difference = ours[valid] - original[valid]
    if not np.any(difference):
        return 1.0
    return float(wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: clean(value) for key, value in row.items()} for row in rows)


def catalog(thresholds):
    result = []
    for space in ("projected", "world"):
        for label, threshold in thresholds[space].items():
            for alpha in ALPHAS:
                metrics = ("lbf_count", "faint_given_large", "lbf_area") if space == "projected" else (
                    "lbf_count", "faint_given_large")
                for metric in metrics:
                    result.append({"size_space": space, "threshold": label, "tau_s": threshold,
                                   "tau_alpha": alpha, "metric": metric})
        result.append({"size_space": space, "threshold": "", "tau_s": np.nan,
                       "tau_alpha": np.nan, "metric": "spearman_rho"})
    return result


def summarize(numerators, denominators, specifications, groups, samples, seed):
    rows = []
    for group in ("overall", "small", "medium", "large"):
        chosen = np.ones(len(groups), dtype=bool) if group == "overall" else np.asarray(groups) == group
        on, od = numerators["original"][chosen], denominators["original"][chosen]
        un, ud = numerators["ours"][chosen], denominators["ours"][chosen]
        stats = bootstrap_comparison(on, od, un, ud, samples, seed)
        k = len(specifications)
        for column, spec in enumerate(specifications):
            common = {"model": "noposplat", "overlap": group, **spec, "n_scenes": int(chosen.sum())}
            for name, offset, numerator, denominator in (
                ("original", 0, on, od), ("ours", k, un, ud),
            ):
                values = ratio(numerator[:, column], denominator[:, column])
                rows.append({**common, "decoder": name,
                             "estimate": stats["estimate"][offset + column],
                             "ci_low": stats["ci"][offset + column, 0],
                             "ci_high": stats["ci"][offset + column, 1],
                             "scene_mean": stats["estimate"][2 * k + offset + column],
                             "scene_mean_ci_low": stats["ci"][2 * k + offset + column, 0],
                             "scene_mean_ci_high": stats["ci"][2 * k + offset + column, 1],
                             "n_defined": int(np.isfinite(values).sum()),
                             "numerator_total": float(numerator[:, column].sum()),
                             "denominator_total": float(denominator[:, column].sum()),
                             "wilcoxon_p_unadjusted": np.nan})
            valid = np.isfinite(stats["original_ratios"][:, column]) & np.isfinite(stats["ours_ratios"][:, column])
            rows.append({**common, "decoder": "ours-original",
                         "estimate": stats["estimate"][k + column] - stats["estimate"][column],
                         "ci_low": stats["delta_ci"][column, 0],
                         "ci_high": stats["delta_ci"][column, 1],
                         "scene_mean": stats["estimate"][4 * k + column],
                         "scene_mean_ci_low": stats["scene_mean_delta_ci"][column, 0],
                         "scene_mean_ci_high": stats["scene_mean_delta_ci"][column, 1],
                         "n_defined": int(valid.sum()), "numerator_total": np.nan,
                         "denominator_total": np.nan,
                         "wilcoxon_p_unadjusted": paired_pvalue(
                             stats["original_ratios"][:, column], stats["ours_ratios"][:, column])})
    return rows


def main_row(rows, space, name, metric="lbf_count", group="overall"):
    return next(row for row in rows if row["size_space"] == space and row["decoder"] == name
                and row["metric"] == metric and row["overlap"] == group
                and row["threshold"] == "P90" and row["tau_alpha"] == 0.25)


def _report(directory, scenes, thresholds, samples, seed, group_fn):
    specifications = catalog(thresholds)
    columns = {(s["size_space"], s["threshold"], s["tau_alpha"], s["metric"]): j
               for j, s in enumerate(specifications) if s["metric"] != "spearman_rho"}
    rho_columns = {s["size_space"]: j for j, s in enumerate(specifications) if s["metric"] == "spearman_rho"}
    numerators = {name: np.zeros((len(scenes), len(specifications))) for name in ("original", "ours")}
    denominators = {name: np.zeros_like(value) for name, value in numerators.items()}
    population = {name: {} for name in numerators}
    rows, transition_rows, groups = [], [], []
    for index, (scene, entry) in enumerate(scenes.items()):
        group = group_fn(entry["overlap"])
        groups.append(group)
        with np.load(cache_path(directory, "original", index), allow_pickle=False) as original_file, np.load(
            cache_path(directory, "ours", index), allow_pickle=False,
        ) as ours_file:
            keys = ("scene", "opacity", "s_world", "s_px", "area_px", "valid_ref", "all_behind",
                    "camera_extrinsics", "camera_intrinsics", "image_shape")
            original = {key: original_file[key] for key in keys}
            ours = {key: ours_file[key] for key in keys}
            if str(original["scene"]) != scene or str(ours["scene"]) != scene:
                raise ValueError(f"Cache scene identity mismatch: {scene}")
            for key in ("camera_extrinsics", "camera_intrinsics", "image_shape"):
                if not np.allclose(original[key], ours[key], rtol=1e-6, atol=1e-8):
                    raise ValueError(f"Models used different context cameras/layout for {scene}: {key}")
            transition_rows.append({"scene": scene, "overlap": group,
                                    **transition_counts(original, ours, thresholds["projected"]["P90"])})
            for name, data in (("original", original), ("ours", ours)):
                active = data["opacity"] >= ACTIVE
                invalid = ~data["valid_ref"]
                diagnostics = {"N_total": len(active), "N_active": int(active.sum()),
                               "N_excl": int(invalid.sum()), "N_excl_active": int((invalid & active).sum()),
                               "N_excl_all_behind": int(data["all_behind"].sum()),
                               "N_excl_origin_only": int((invalid & ~data["all_behind"]).sum())}
                for key, value in diagnostics.items():
                    population[name][key] = population[name].get(key, 0) + value
                for space in ("projected", "world"):
                    distribution = None
                    for label, threshold in thresholds[space].items():
                        for alpha in ALPHAS:
                            stat = scene_statistics(data, space, threshold, alpha,
                                                    compute_diagnostics=distribution is None)
                            if distribution is None:
                                distribution = {key: stat[key] for key in ("spearman_rho", "median_size")}
                            stat.update(distribution)
                            values = {"lbf_count": (stat["N_lbf"], stat["N_analysis"]),
                                      "faint_given_large": (stat["N_lbf"], stat["N_large"])}
                            if space == "projected":
                                values["lbf_area"] = (stat["area_lbf"], stat["area_total"])
                            fractions = {}
                            for metric, (num, den) in values.items():
                                column = columns[(space, label, alpha, metric)]
                                numerators[name][index, column] = num
                                denominators[name][index, column] = den
                                fractions[metric] = float(ratio(num, den))
                            rows.append({"model": "noposplat", "decoder": name, "scene": scene,
                                         "overlap": group, "size_space": space, "threshold": label,
                                         "tau_s": threshold, "tau_alpha": alpha,
                                         **diagnostics, **stat, **fractions})
                    rho = distribution["spearman_rho"]
                    if np.isfinite(rho):
                        numerators[name][index, rho_columns[space]] = rho
                        denominators[name][index, rho_columns[space]] = 1
    summary = summarize(numerators, denominators, specifications, groups, samples, seed)
    result = directory / "results"
    write_csv(result / "per_scene.csv", rows)
    write_csv(result / "summary.csv", summary)
    write_csv(result / "transitions_per_scene.csv", transition_rows)
    transition_summary = []
    for group in ("overall", "small", "medium", "large"):
        selected = [row for row in transition_rows if group == "overall" or row["overlap"] == group]
        num = np.array([[row[key] for key in FATES] for row in selected], dtype=float).reshape(-1, len(FATES))
        den = np.repeat(np.array([row["N_original_lbf"] for row in selected])[:, None], len(FATES), axis=1)
        stats = bootstrap_comparison(num, den, num, den, samples, seed)
        for column, key in enumerate(FATES):
            transition_summary.append({"overlap": group, "fate": key,
                                       "count": int(num[:, column].sum()),
                                       "N_original_lbf": int(den[:, column].sum()),
                                       "fraction": stats["estimate"][column],
                                       "ci_low": stats["ci"][column, 0], "ci_high": stats["ci"][column, 1]})
    write_csv(result / "transitions.csv", transition_summary)
    main = {space: {metric: {name: main_row(summary, space, name, metric)
                            for name in ("original", "ours", "ours-original")}
                    for metric in (("lbf_count", "faint_given_large", "lbf_area") if space == "projected"
                                   else ("lbf_count", "faint_given_large"))}
            for space in ("projected", "world")}
    nondecrease = [row for row in summary if row["overlap"] == "overall"
                  and row["size_space"] == "projected" and row["decoder"] == "ours-original"
                  and row["metric"] in ("lbf_count", "lbf_area") and row["estimate"] >= 0]
    report = {"bootstrap_resamples": samples, "seed": seed,
              "estimand": "pooled ratio; scene-cluster paired percentile bootstrap",
              "world_population": "all active slots",
              "projected_population": "active slots with positive origin-camera depth",
              "population_diagnostics": population, "thresholds": thresholds,
              "area_caveat": "potential unfiltered ellipse area; no occlusion, overlap, or screen-boundary correction",
              "main": main, "projected_grid_nondecreases": nondecrease}
    (result / "lbf_report.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# LBF measurements", "", f"Scenes: {len(scenes)}; seed: {seed}; bootstrap: {samples}.", "",
             "Main cutoff: baseline full-population P90, opacity < 0.25, active opacity >= 1/255.",
             f"Frozen size cutoffs: projected {thresholds['projected']['P90']:.6g} px; world {thresholds['world']['P90']:.6g}.",
             "Projected reference: origin context view and dataset camera poses; no pose optimization.", "",
             "| Space / metric | Original | Ours | Ours - original (95% CI) | Paired Wilcoxon p |",
             "|---|---:|---:|---:|---:|"]
    for space, metrics in main.items():
        for metric, values in metrics.items():
            o, u, d = (values[name] for name in ("original", "ours", "ours-original"))
            lines.append(f"| {space} / {metric} | {100*o['estimate']:.3f}% | {100*u['estimate']:.3f}% | "
                         f"{100*d['estimate']:.3f} pp [{100*d['ci_low']:.3f}, {100*d['ci_high']:.3f}] | "
                         f"{d['wilcoxon_p_unadjusted']:.5g} |")
    lines += ["", f"Projected grid cells/metrics with non-decreasing point estimates: {len(nondecrease)}.",
              "All grid rows and exclusions are retained in per_scene.csv and summary.csv.",
              "Wilcoxon p-values are unadjusted; the P90 / alpha=0.25 comparison is the main setting.",
              "CIs condition on the frozen baseline size thresholds.",
              "Area is potential unfiltered 1-sigma ellipse area, without occlusion/overlap/screen-boundary correction.",
              "Transitions match source-pixel slots, not identical physical Gaussians."]
    lines += ["", "| Decoder | Total slots | Active | Nonpositive origin depth (active) | Behind all contexts | Origin only behind |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, diag in population.items():
        lines.append(f"| {name} | {diag['N_total']} | {diag['N_active']} | "
                     f"{diag['N_excl']} ({diag['N_excl_active']}) | {diag['N_excl_all_behind']} | {diag['N_excl_origin_only']} |")
    (result / "lbf_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = [r"\begin{tabular}{llrrr}", r"\hline", r"Size cutoff & Opacity & Original (\%) & Ours (\%) & $\Delta$ (pp) \\", r"\hline"]
    for label in thresholds["projected"]:
        for alpha in ALPHAS:
            selected = {row["decoder"]: row for row in summary if row["overlap"] == "overall"
                        and row["size_space"] == "projected" and row["metric"] == "lbf_count"
                        and row["threshold"] == label and row["tau_alpha"] == alpha}
            tex.append(f"{label} & {alpha:.2f} & {100*selected['original']['estimate']:.2f} & "
                       f"{100*selected['ours']['estimate']:.2f} & {100*selected['ours-original']['estimate']:.2f} " + r"\\")
    tex += [r"\hline", r"\end{tabular}"]
    (result / "table_robustness.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    from exp3.lbf_plot import make_lbf_figures
    make_lbf_figures(directory, thresholds, summary, transition_summary)
    return report


def run_report(directory, scenes, samples=10000, seed=0):
    from exp3.plot_scale_opacity import overlap_group
    directory = Path(directory)
    for name in ("original", "ours"):
        validate_cache(directory, name, scenes)
    thresholds = freeze_thresholds(directory, scenes)
    status = directory / "results" / "lbf_status.txt"
    status.write_text("RUNNING\n", encoding="utf-8")
    try:
        result = _report(directory, scenes, thresholds, samples, seed, overlap_group)
    except Exception:
        status.write_text("FAILED\n", encoding="utf-8")
        raise
    status.write_text("COMPLETE\n", encoding="utf-8")
    print(f"[LBF] report saved to {directory / 'results/lbf_report.md'}", flush=True)
    return result
