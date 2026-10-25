import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from exp3.plot import make_figure
from exp3.plot_scale_opacity import compose_config, main, parser, restore_encoder, select_scenes
from exp3.scale_opacity import PrioritySample, log_max_scale


class ScaleTests(unittest.TestCase):
    def test_final_covariance_scale_rotation_and_world_rescaling(self):
        # Known standard deviations, deliberately not the covariance eigenvalues.
        scales = np.array([[0.01, 0.02, 0.04], [0.2, 0.3, 0.1]])
        angle = 0.73
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        covariance = np.array([rotation @ np.diag(s ** 2) @ rotation.T for s in scales])
        np.testing.assert_allclose(log_max_scale(covariance), np.log10(scales.max(axis=1)))
        np.testing.assert_allclose(log_max_scale(49 * covariance),
                                   log_max_scale(covariance) + np.log10(7))

    def test_reject_invalid_active_gaussians(self):
        for covariance in (np.zeros((1, 3, 3)), np.diag([-0.1, 0.1, 0.2])[None],
                           np.full((1, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                log_max_scale(covariance)

    def test_empty_scene_has_no_log_values(self):
        self.assertEqual(log_max_scale(np.empty((0, 3, 3))).shape, (0,))


class SamplingTests(unittest.TestCase):
    def test_pooled_sample_does_not_depend_on_chunk_order(self):
        rng = np.random.default_rng(42)
        priority = rng.random(10000)
        samplers = [PrioritySample(200), PrioritySample(200)]
        chunks = [np.arange(0, 100), np.arange(100, 10000)]
        for sampler, order in zip(samplers, (chunks, list(reversed(chunks)))):
            for indices in order:
                sampler.add(priority[indices], log_s_max=indices * 0.01,
                            opacity=np.full(len(indices), 0.5),
                            scene_index=(indices >= 100).astype(int), gaussian_index=indices)
        for key in samplers[0].arrays():
            np.testing.assert_array_equal(samplers[0].arrays()[key], samplers[1].arrays()[key])
        chosen = samplers[0].arrays()["gaussian_index"]
        # A scene with 99% of the population should dominate a pooled sample.
        self.assertGreater(np.count_nonzero(chosen >= 100), 180)
        self.assertEqual(len(np.unique(chosen)), 200)

    def test_scene_selection_is_stratified_and_fixed(self):
        index = {f"{tag}_{i}": {"overlap": overlap, "context": [0, 10], "target": [5]}
                 for tag, overlap in (("small", 0.2), ("medium", 0.4), ("large", 0.7))
                 for i in range(5)}
        index["null_scene"] = None
        selected = select_scenes(index, 6, 0)
        self.assertEqual(selected, select_scenes(dict(reversed(list(index.items()))), 6, 0))
        self.assertEqual(len(selected), 6)
        for tag in ("small", "medium", "large"):
            self.assertEqual(sum(key.startswith(tag) for key in selected), 2)
        self.assertEqual(len(select_scenes(index, 0, 0)), 15)

    def test_invalid_index_is_reported(self):
        with self.assertRaises(ValueError):
            select_scenes({"bad": {"overlap": 0.4, "context": [0], "target": [1]}}, 0, 0)


class CheckpointTests(unittest.TestCase):
    def test_restore_all_encoder_weights_strictly_and_preserve_step(self):
        fake_torch = SimpleNamespace(load=Mock(return_value={
            "global_step": 140000,
            "state_dict": {"encoder.backbone.weight": "backbone",
                           "encoder.gaussian_decoder.weight": "head", "losses.0.weight": "unused"},
        }))
        encoder = Mock()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            step = restore_encoder(encoder, Path("checkpoint.ckpt"))
        self.assertEqual(step, 140000)
        encoder.load_state_dict.assert_called_once_with(
            {"backbone.weight": "backbone", "gaussian_decoder.weight": "head"}, strict=True)

    def test_missing_encoder_state_does_not_fall_back_to_initial_weights(self):
        fake_torch = SimpleNamespace(load=lambda *a, **kw: {"state_dict": {"other": 1}})
        with patch.dict("sys.modules", {"torch": fake_torch}), self.assertRaises(ValueError):
            restore_encoder(Mock(), Path("bad.ckpt"))


class ConfigTests(unittest.TestCase):
    def test_repository_experiments_compose_with_shared_scene_list(self):
        args = parser().parse_args(["--num-workers", "0"])
        index = Path(tempfile.gettempdir()) / "selected_scenes.json"
        original, _ = compose_config(args, "original", index)
        ours, _ = compose_config(args, "ours", index)
        self.assertEqual(original.model.encoder.gs_params_head_type, "dpt_gs")
        self.assertEqual(ours.model.encoder.gs_params_head_type, "moment")
        self.assertEqual(ours.model.encoder.moment_decoder.feature_dim, 256)
        self.assertEqual(ours.model.encoder.moment_decoder.scale_max, 3.0)
        self.assertFalse(ours.model.encoder.moment_decoder.compile_kernels)
        for cfg in (original, ours):
            self.assertEqual(cfg.model.encoder.pretrained_weights, "")
            self.assertEqual(cfg.dataset.re10k.view_sampler.index_path, str(index))
            self.assertEqual(list(cfg.dataset.re10k.input_image_shape), [256, 256])
            self.assertEqual(cfg.data_loader.test.batch_size, 1)

    def test_run_failure_is_recorded_without_making_a_comparison_figure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.touch()
            index = root / "index.json"
            index.write_text(json.dumps({"scene": {
                "context": [0, 10], "target": [5], "overlap": 0.4,
            }}), encoding="utf-8")
            output = root / "output"
            with patch("exp3.plot_scale_opacity.extract", side_effect=RuntimeError("missing scene")):
                with self.assertRaisesRegex(RuntimeError, "missing scene"):
                    main(["--original-checkpoint", str(checkpoint),
                          "--ours-checkpoint", str(checkpoint), "--data-root", str(root),
                          "--index", str(index), "--output-dir", str(output)])
            self.assertEqual((output / "status.txt").read_text(encoding="utf-8").strip(), "FAILED")
            self.assertIn("missing scene", (output / "failures.txt").read_text(encoding="utf-8"))
            self.assertFalse((output / "figs").exists())


class PlotTests(unittest.TestCase):
    def test_cached_real_value_format_produces_png_and_pdf(self):
        # Synthetic fixture lives only in a temporary directory and is never an experiment result.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "samples").mkdir()
            metadata = {"counts": {name: {"N_selected": 1000} for name in ("original", "ours")}}
            (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            rng = np.random.default_rng(0)
            for name in ("original", "ours"):
                np.savez_compressed(directory / "samples" / f"{name}.npz",
                                    log_s_max=rng.normal(-2, 0.3, 1000), opacity=rng.random(1000))
            make_figure(directory)
            self.assertTrue((directory / "figs/figC_scale_opacity.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((directory / "figs/figC_scale_opacity.pdf").read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
