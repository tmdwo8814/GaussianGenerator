import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from exp3.plot_scale_opacity import ROOT, compose_config, main, overlap_group, parser, select_scenes
from exp3.scale_opacity import PrioritySample, log_max_scale


class DistributionTests(unittest.TestCase):
    def test_final_covariance_scale_rotation_and_rescaling(self):
        angle = 0.73
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        covariance = (rotation @ np.diag(np.array([0.01, 0.02, 0.04]) ** 2) @ rotation.T)[None]
        np.testing.assert_allclose(log_max_scale(covariance), np.log10(0.04))
        np.testing.assert_allclose(log_max_scale(49 * covariance), np.log10(7 * 0.04))

    def test_invalid_covariance_is_reported(self):
        for value in (np.zeros((1, 3, 3)), np.full((1, 3, 3), np.nan),
                      np.diag([-0.1, 0.1, 0.2])[None]):
            with self.assertRaises(ValueError):
                log_max_scale(value)
        self.assertEqual(log_max_scale(np.empty((0, 3, 3))).shape, (0,))

    def test_pooled_sample_is_independent_of_chunk_order(self):
        priority = np.random.default_rng(42).random(10000)
        samples = [PrioritySample(200), PrioritySample(200)]
        chunks = [np.arange(100), np.arange(100, 10000)]
        for sample, order in zip(samples, (chunks, list(reversed(chunks)))):
            for indices in order:
                sample.add(priority[indices], log_s_max=indices * 0.01,
                           opacity=np.full(len(indices), 0.5), gaussian_index=indices,
                           scene_index=(indices >= 100).astype(int))
        for key in samples[0].arrays():
            np.testing.assert_array_equal(samples[0].arrays()[key], samples[1].arrays()[key])
        selected = samples[0].arrays()["gaussian_index"]
        self.assertEqual(len(np.unique(selected)), 200)
        self.assertGreater(np.count_nonzero(selected >= 100), 180)


class ConfigurationTests(unittest.TestCase):
    def test_current_training_architecture_is_preserved(self):
        args = parser().parse_args(["--num-workers", "0"])
        index = Path(tempfile.gettempdir()) / "selected.json"
        original, _ = compose_config(args, "original", index)
        ours, _ = compose_config(args, "ours", index)
        self.assertEqual(original.model.encoder.gs_params_head_type, "dpt_gs")
        self.assertEqual(ours.model.encoder.gs_params_head_type, "moment")
        cfg = ours.model.encoder.moment_decoder
        for flag in ("appearance_2d", "separate_appearance", "appearance_capacity", "appearance_moments"):
            self.assertTrue(cfg[flag])
        self.assertEqual(cfg.appearance_mlp_depth, 2)
        self.assertEqual(cfg.appearance_mlp_dim, 256)
        self.assertEqual(2 * cfg.feature_dim + cfg.appearance_heads * cfg.appearance_dim, 576)
        self.assertFalse(cfg.compile_kernels)
        for cfg in (original, ours):
            self.assertEqual(cfg.dataset.re10k.view_sampler.index_path, str(index))
            self.assertEqual(list(cfg.dataset.re10k.input_image_shape), [256, 256])
            self.assertEqual(cfg.model.encoder.pretrained_weights, "")

    def test_actual_index_and_overlap_tails(self):
        for value, expected in ((0, "small"), (0.04031, "small"), (0.3, "small"),
                                (0.31, "medium"), (0.55, "medium"), (0.95, "large"), (1, "large")):
            self.assertEqual(overlap_group(value), expected)
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                overlap_group(value)
        index = json.loads((ROOT / "assets/evaluation_index_re10k.json").read_text(encoding="utf-8"))
        valid = {key for key, value in index.items() if value is not None}
        self.assertEqual(set(select_scenes(index, 0, 0)), valid)
        subset = select_scenes(index, 500, 0)
        self.assertEqual(len(subset), 500)
        self.assertEqual(subset, select_scenes(dict(reversed(list(index.items()))), 500, 0))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        checkpoint = root / "model.ckpt"
        checkpoint.touch()
        index = root / "index.json"
        index.write_text(json.dumps({"scene": {
            "context": [0, 10], "target": [5], "overlap": 0.04031,
        }}), encoding="utf-8")
        self.output = root / "output"
        self.arguments = ["--original-checkpoint", str(checkpoint), "--ours-checkpoint", str(checkpoint),
                          "--data-root", str(root), "--index", str(index), "--output-dir", str(self.output)]

    def extract_fixture(self, args, name, scenes, directory):
        row = {"decoder": name, "scene": "scene", "overlap": "small", "N_total": 10,
               "N_active": 8, "N_selected": 8, "median_log_s_max": -2.0}
        with (directory / "per_scene.csv").open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=row).writerow(row)
        np.savez_compressed(directory / "samples" / f"{name}.npz",
                            log_s_max=np.linspace(-3, -1, 8), opacity=np.linspace(0.1, 0.9, 8),
                            scene_index=np.zeros(8, dtype=int), gaussian_index=np.arange(8))
        (directory / f"checkpoint_load_{name}.json").write_text(json.dumps({
            "strict": True, "all_encoder_weights": True,
        }), encoding="utf-8")
        return {"N_total": 10, "N_active": 8, "N_selected": 8, "checkpoint_step": 140000, "scenes": 1}

    def fail_after_original(self):
        def extract(args, name, scenes, directory):
            counts = self.extract_fixture(args, name, scenes, directory)
            if name == "ours":
                raise RuntimeError("ours interrupted")
            return counts
        with patch("exp3.plot_scale_opacity.extract", side_effect=extract):
            with self.assertRaisesRegex(RuntimeError, "ours interrupted"):
                main(self.arguments)
        self.assertEqual((self.output / "status.txt").read_text(encoding="utf-8").strip(), "FAILED")
        self.assertFalse((self.output / "figs").exists())

    def test_resume_legacy_original_then_generate_png_and_pdf(self):
        self.fail_after_original()
        metadata_path = self.output / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["arguments"].pop("resume")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        (self.output / "checkpoint_load_original.json").unlink()
        sample = self.output / "samples/original.npz"
        before = sample.read_bytes()
        with patch("exp3.plot_scale_opacity.extract", side_effect=self.extract_fixture) as extract:
            main([*self.arguments, "--resume"])
        self.assertEqual([call.args[1] for call in extract.call_args_list], ["ours"])
        self.assertEqual(sample.read_bytes(), before)
        with (self.output / "per_scene.csv").open(newline="", encoding="utf-8") as file:
            self.assertEqual([row["decoder"] for row in csv.DictReader(file)], ["original", "ours"])
        self.assertTrue((self.output / "figs/figC_scale_opacity.png").read_bytes().startswith(b"\x89PNG"))
        self.assertTrue((self.output / "figs/figC_scale_opacity.pdf").read_bytes().startswith(b"%PDF"))

    def test_resume_rejects_changed_seed_without_touching_cache(self):
        self.fail_after_original()
        before = (self.output / "per_scene.csv").read_bytes()
        with patch("exp3.plot_scale_opacity.extract") as extract:
            with self.assertRaisesRegex(ValueError, "changed seed"):
                main([*self.arguments, "--resume", "--seed", "1"])
        extract.assert_not_called()
        self.assertEqual((self.output / "per_scene.csv").read_bytes(), before)

    def test_geometry_only_ours_cache_is_not_reused(self):
        with patch("exp3.plot_scale_opacity.extract", side_effect=self.extract_fixture):
            main(self.arguments)
        (self.output / "checkpoint_load_ours.json").write_text(
            json.dumps({"strict": True, "geometry_only": True}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different restore policy"):
            main([*self.arguments, "--resume"])


if __name__ == "__main__":
    unittest.main()
