"""Geometry, full-population statistics, and the optional LBF workflow."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from exp3.lbf_metrics import (
    ACTIVE, FATES, LBFCollector, cache_path, eligible_mask, exact_percentiles,
    freeze_thresholds, projected_geometry, renderer_focals, scene_statistics,
    transition_counts, validate_cache,
)
from exp3.lbf_report import bootstrap_comparison, paired_pvalue, run_report
from exp3.plot_scale_opacity import FIELDS, main


def camera(views=1):
    extrinsics = np.repeat(np.eye(4)[None], views, axis=0)
    intrinsics = np.repeat(np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1.]])[None], views, axis=0)
    return extrinsics, intrinsics


def inputs(sizes, opacity=None):
    """One source row; desired projected radii at z=1, centered isotropic covariances."""
    sizes = np.asarray(sizes, dtype=float)
    count = len(sizes)
    opacity = np.full(count, 0.1, dtype=np.float32) if opacity is None else np.array(opacity, dtype=np.float32)
    extrinsics, intrinsics = camera()
    world = sizes / (count / 2)
    means = np.zeros((count, 3))
    means[:, 2] = 1
    covariance = world[:, None, None]**2 * np.eye(3)
    return (means, covariance, opacity, np.log10(world[opacity >= ACTIVE]),
            extrinsics, intrinsics, (1, count))


def cache_arrays(sizes, opacity, area=None, valid=None):
    sizes = np.array(sizes, dtype=float)
    count = len(sizes)
    return {"s_px": sizes, "s_world": sizes.copy(), "opacity": np.array(opacity, dtype=np.float32),
            "area_px": np.ones(count) if area is None else np.array(area, dtype=float),
            "valid_ref": np.ones(count, dtype=bool) if valid is None else np.array(valid),
            "all_behind": np.zeros(count, dtype=bool)}


class ProjectionTests(unittest.TestCase):
    def test_renderer_focal_and_off_center_fov(self):
        _, k = camera()
        np.testing.assert_allclose(renderer_focals(k, (100, 200)), [[100, 50]])
        k[0, 0, 2] = 0.3
        expected = 200 / (2 * np.tan((np.arctan(0.3 / 0.5) + np.arctan(0.7 / 0.5)) / 2))
        self.assertAlmostEqual(renderer_focals(k, (100, 200))[0, 0], expected)
        self.assertNotAlmostEqual(expected, k[0, 0, 0] * 200)

    def test_projected_size_area_and_scene_scale_invariance(self):
        args = list(inputs([1, 2, 3, 4]))
        data = projected_geometry(*args)
        np.testing.assert_allclose(data["s_px"], [1, 2, 3, 4])
        # Height=1, width=4 gives fy/fx=1/4, so the ellipse is not circular.
        np.testing.assert_allclose(data["area_px"], np.pi * np.arange(1, 5)**2 / 4)
        scale = 7
        args[0] *= scale
        args[1] *= scale**2
        args[3] += np.log10(scale)
        args[4][:, :3, 3] *= scale
        scaled = projected_geometry(*args)
        np.testing.assert_allclose(scaled["s_px"], data["s_px"])
        np.testing.assert_allclose(scaled["area_px"], data["area_px"])
        np.testing.assert_allclose(scaled["s_world"], scale * data["s_world"])

    def test_area_uses_anisotropy_and_perspective(self):
        args = list(inputs([1]))
        args[0][0] = [2, 0, 2]
        args[1][0] = np.diag([4., 9., 16.])
        args[3] = np.array([np.log10(4)])
        data = projected_geometry(*args)
        # fx=fy=.5; J=[[.25,0,-.25],[0,.25,0]].
        self.assertAlmostEqual(data["area_px"][0], np.pi * np.sqrt(1.25 * 0.5625))
        self.assertAlmostEqual(data["s_px"][0], 1)

    def test_source_view_is_preserved_and_behind_counts_are_explicit(self):
        extrinsics, intrinsics = camera(2)
        extrinsics[1, 2, 3] = -2
        means = np.array([[0, 0, -1.], [0, 0, -3.], [0, 0, 2.], [0, 0, -1.]])
        data = projected_geometry(means, np.repeat(np.eye(3)[None], 4, axis=0),
                                  np.full(4, .1), np.zeros(4), extrinsics, intrinsics, (1, 2))
        np.testing.assert_array_equal(data["ref_cam"], [0, 0, 1, 1])
        np.testing.assert_array_equal(data["valid_ref"], [False, False, True, True])
        np.testing.assert_array_equal(data["all_behind"], [False, True, False, False])
        np.testing.assert_allclose(data["s_px"][2:], [.25, 1])
        self.assertTrue(np.isnan(data["s_px"][:2]).all())
        self.assertEqual(eligible_mask(data, "projected").sum(), 2)
        self.assertEqual(eligible_mask(data, "world").sum(), 4)

    def test_active_boundary_and_invalid_inputs(self):
        args = list(inputs([1, 2, 3], [0, ACTIVE, .25]))
        data = projected_geometry(*args)
        np.testing.assert_array_equal(eligible_mask(data, "projected"), [False, True, True])
        self.assertTrue(np.isnan(data["s_world"][0]))
        args[0][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            projected_geometry(*args)


class StatisticsTests(unittest.TestCase):
    def test_count_conditional_and_area_have_distinct_denominators(self):
        data = cache_arrays([1, 3, 4, 5], [.1, .8, .25, .1], [1, 1, 1, 7])
        stat = scene_statistics(data, "projected", 2, .25)
        self.assertEqual((stat["N_analysis"], stat["N_large"], stat["N_lbf"]), (4, 3, 1))
        self.assertAlmostEqual(stat["area_lbf"] / stat["area_total"], .7)
        # Strict large/faint inequalities: alpha=.25 and size=threshold do not qualify.
        self.assertEqual(scene_statistics(data, "world", 5, .25)["N_lbf"], 0)

    def test_six_transition_categories_partition_the_baseline_lbf(self):
        original = cache_arrays([10]*6, [.1]*6)
        ours = cache_arrays([np.nan, np.nan, 2, 10, 2, 10],
                            [0, .1, .1, .5, .5, .1], valid=[True, False, True, True, True, True])
        counts = transition_counts(original, ours, 5)
        self.assertEqual(counts["N_original_lbf"], 6)
        self.assertEqual([counts[key] for key in FATES], [1]*6)
        self.assertEqual(transition_counts(original, ours, 20)["N_original_lbf"], 0)

    def test_bootstrap_distinguishes_pooled_from_scene_mean_and_pairs_draws(self):
        num, den = np.array([[1.], [0.]]), np.array([[1.], [9.]])
        stats = bootstrap_comparison(num, den, num, den, samples=128)
        self.assertAlmostEqual(stats["estimate"][0], .1)
        self.assertAlmostEqual(stats["estimate"][2], .5)
        np.testing.assert_array_equal(stats["delta_ci"], [[0, 0]])
        np.testing.assert_array_equal(stats["scene_mean_delta_ci"], [[0, 0]])
        self.assertEqual(paired_pvalue(num[:, 0], num[:, 0]), 1)

    def test_undefined_ratios_and_only_paired_scene_mean(self):
        stats = bootstrap_comparison([[1.], [0.]], [[1.], [0.]], [[0.], [9.]], [[1.], [9.]], samples=64)
        # Original only defines scene 0; difference of separate means would be -.5.
        self.assertEqual(stats["estimate"][4], -1)
        empty = np.empty((0, 1))
        stats = bootstrap_comparison(empty, empty, empty, empty, samples=2)
        self.assertTrue(np.isnan(stats["estimate"]).all())
        self.assertTrue(np.isnan(stats["delta_ci"]).all())
        self.assertTrue(np.isnan(paired_pvalue(np.array([np.nan]), np.array([0.]))))


class LBFWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.scenes = {f"scene{i}": {"context": [0, 10], "target": [5], "overlap": overlap}
                       for i, overlap in enumerate((.1, .4))}
        (self.directory / "samples").mkdir()
        (self.directory / "metadata.json").write_text(json.dumps({"arguments": {"seed": 0}}), encoding="utf-8")

    def collect(self, name):
        collector = LBFCollector(self.directory, name, self.scenes, seed=0, capacity=2)
        for i, scene in enumerate(self.scenes):
            opacity = np.full(100, .8, dtype=np.float32)
            opacity[90:] = .1
            if name == "ours":
                opacity[90:99] = .8
            args = inputs(np.arange(1, 101), opacity)
            collector.add(i, scene, *args)
        collector.finish()
        # Tiny display sample deliberately cannot encode full-population LBF statistics.
        np.savez_compressed(self.directory / "samples" / f"{name}.npz",
                            log_s_max=np.array([-1., 0.]), opacity=np.array([.8, .8]),
                            scene_index=[0, 1], gaussian_index=[0, 1])

    def test_exact_baseline_quantiles_and_freeze_do_not_read_ours(self):
        self.collect("original")
        thresholds = freeze_thresholds(self.directory, self.scenes)
        np.testing.assert_allclose([thresholds["projected"][f"P{p}"] for p in (80, 90, 95)],
                                   np.percentile(np.tile(np.arange(1, 101), 2), [80, 90, 95]))
        before = (self.directory / "results/lbf_thresholds.json").read_bytes()
        self.collect("ours")
        self.assertEqual(freeze_thresholds(self.directory, self.scenes), thresholds)
        self.assertEqual((self.directory / "results/lbf_thresholds.json").read_bytes(), before)
        self.assertFalse(any((self.directory / "results").glob("percentile_work*")))
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            exact_percentiles(self.directory, self.scenes, "projected", 199)

    def test_missing_or_changed_full_cache_is_rejected(self):
        self.collect("original")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_cache(self.directory, "original", {"other": self.scenes["scene0"]})
        cache_path(self.directory, "original", 0).unlink()
        with self.assertRaises(FileNotFoundError):
            validate_cache(self.directory, "original", self.scenes)

    def test_end_to_end_full_population_report_and_figures(self):
        self.collect("original")
        freeze_thresholds(self.directory, self.scenes)
        self.collect("ours")
        report = run_report(self.directory, self.scenes, samples=32)
        count = report["main"]["projected"]["lbf_count"]
        self.assertAlmostEqual(count["original"]["estimate"], .1)
        self.assertAlmostEqual(count["ours"]["estimate"], .01)
        self.assertEqual(count["original"]["denominator_total"], 200)
        self.assertAlmostEqual(count["ours-original"]["ci_low"], -.09)
        area = report["main"]["projected"]["lbf_area"]["original"]["estimate"]
        self.assertAlmostEqual(area, np.arange(91, 101, dtype=float).dot(np.arange(91, 101))
                               / np.arange(1, 101, dtype=float).dot(np.arange(1, 101)))
        with (self.directory / "results/transitions.csv").open(encoding="utf-8", newline="") as file:
            transitions = [row for row in csv.DictReader(file) if row["overlap"] == "overall"]
        self.assertEqual(sum(int(row["count"]) for row in transitions), 20)
        self.assertEqual(next(int(row["count"]) for row in transitions if row["fate"] == "more_opaque_only"), 18)
        with (self.directory / "results/summary.csv").open(encoding="utf-8", newline="") as file:
            summary = list(csv.DictReader(file))
        grid = [r for r in summary if r["overlap"] == "overall" and r["size_space"] == "projected"
                and r["decoder"] == "ours-original" and r["metric"] == "lbf_count"]
        self.assertEqual(len(grid), 18)
        empty_group = next(r for r in summary if r["overlap"] == "large")
        self.assertEqual(empty_group["estimate"], "")
        self.assertEqual((self.directory / "results/lbf_status.txt").read_text().strip(), "COMPLETE")
        for figure in ("projected", "world", "transitions"):
            self.assertTrue((self.directory / f"figs/figC_lbf_{figure}.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((self.directory / f"figs/figC_lbf_{figure}.pdf").read_bytes().startswith(b"%PDF"))

    def test_cli_freezes_thresholds_before_ours_and_plot_only_reuses_full_cache(self):
        checkpoint = self.directory / "model.ckpt"
        checkpoint.touch()
        index = self.directory / "index.json"
        index.write_text(json.dumps(self.scenes), encoding="utf-8")
        output = self.directory / "run"
        arguments = ["--original-checkpoint", str(checkpoint), "--ours-checkpoint", str(checkpoint),
                     "--data-root", str(self.directory), "--index", str(index),
                     "--output-dir", str(output), "--sample-size", "2", "--lbf", "--bootstrap-samples", "16"]

        def extract(args, name, scenes, directory):
            if name == "ours":
                self.assertTrue((directory / "results/lbf_thresholds.json").is_file())
            self.directory = directory
            self.collect(name)
            with (directory / "per_scene.csv").open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                for scene in scenes:
                    writer.writerow(dict(decoder=name, scene=scene, overlap="small", N_total=100,
                                         N_active=100, N_selected=100, median_log_s_max=-1))
            (directory / f"checkpoint_load_{name}.json").write_text(
                json.dumps({"strict": True, "all_encoder_weights": True}), encoding="utf-8")
            return dict(N_total=200, N_active=200, N_selected=200, checkpoint_step=1, scenes=2)

        with patch("exp3.plot_scale_opacity.extract", side_effect=extract):
            main(arguments)
        with patch("exp3.plot_scale_opacity.extract") as extraction, patch("exp3.lbf_report.run_report"):
            main([*arguments, "--resume"])
            extraction.assert_not_called()
        with patch("exp3.plot_scale_opacity.extract") as extraction, patch("exp3.lbf_report.run_report") as report:
            main(["--plot-only", "--output-dir", str(output), "--bootstrap-samples", "8"])
            extraction.assert_not_called()
            self.assertEqual(report.call_args.args[2:], (8, 0))


if __name__ == "__main__":
    unittest.main()
