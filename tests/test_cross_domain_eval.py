"""Cross-domain protocol, model composition, and orchestration without CUDA."""

from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import eval_re10k2dl3dv as evaluation


class CrossDomainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dl3dv"
        (self.dataset / "test").mkdir(parents=True)
        self.scene_index = self.dataset / "test/index.json"
        evaluation.write_json(self.scene_index, {"scene-a": "000000.torch"})
        self.index = self.root / "views.json"
        evaluation.write_json(self.index, {"scene-a": {"context": [0, 22], "target": [5, 11, 16]}})
        self.checkpoint = self.root / "model.ckpt"
        self.checkpoint.touch()
        self.output = self.root / "output"

    def args(self):
        return Namespace(seed=111123, num_workers=4, no_align_pose=False, save_image=False)

    def argv(self):
        return ["--dataset-root", str(self.dataset), "--ours-checkpoint", str(self.checkpoint),
                "--baseline-checkpoint", str(self.checkpoint), "--index", str(self.index),
                "--output-dir", str(self.output)]

    def test_protocol_uses_two_inputs_and_distinct_novel_targets(self):
        self.assertEqual(evaluation.fixed_views(100, 22, 3),
                         {"context": [0, 22], "target": [5, 11, 16]})
        self.assertIsNone(evaluation.fixed_views(22, 22, 3))
        for gap in range(2, 30):
            for targets in range(1, gap):
                entry = evaluation.fixed_views(gap + 1, gap, targets)
                self.assertEqual(len(set(entry["target"])), targets)
                self.assertTrue(all(0 < i < gap for i in entry["target"]))
        for gap, targets in ((1, 1), (22, 22), (22, 0)):
            with self.assertRaises(ValueError):
                evaluation.fixed_views(100, gap, targets)

    def test_rejects_chunk_index_unknown_scenes_and_target_leakage(self):
        for index in ({"scene-a": "000000.torch"},
                      {"unknown": {"context": [0, 22], "target": [5]}},
                      {"scene-a": {"context": [0, 22], "target": [0]}},
                      {"scene-a": {"context": [0, 22], "target": [-1]}},
                      {"scene-a": {"context": [0, 22], "target": [True]}},
                      {"scene-a": None}):
            evaluation.write_json(self.index, index)
            with self.assertRaises(ValueError):
                evaluation.read_view_index(self.index, self.dataset)

    def test_generation_reads_cpu_chunks_and_records_short_scenes(self):
        for number in range(2):
            (self.dataset / "test" / f"{number:06}.torch").touch()
        evaluation.write_json(self.scene_index, {"scene-a": "000000.torch", "scene-b": "000001.torch"})
        chunks = [[{"key": "scene-a", "cameras": [None] * 100, "images": [None] * 100}],
                  [{"key": "scene-b", "cameras": [None] * 10, "images": [None] * 10}]]
        torch = SimpleNamespace(load=Mock(side_effect=chunks))
        with patch.dict("sys.modules", {"torch": torch}), contextlib.redirect_stdout(io.StringIO()):
            index = evaluation.generate_view_index(self.dataset, self.index, 22, 3)
        self.assertIsNone(index["scene-b"])
        self.assertEqual(index["scene-a"]["target"], [5, 11, 16])
        for call in torch.load.call_args_list:
            self.assertEqual(call.kwargs, {"map_location": "cpu", "weights_only": False})
        protocol = json.loads(self.index.with_suffix(".protocol.json").read_text())
        self.assertEqual(protocol["too_short_scenes"], ["scene-b"])

    def test_config_preserves_each_trained_architecture_and_uses_only_dl3dv(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        for label, experiment in evaluation.EXPERIMENTS.items():
            with initialize_config_dir(version_base=None, config_dir=str(evaluation.REPO_ROOT / "config")):
                original = compose(config_name="main", overrides=[f"+experiment={experiment}"])
            cfg = evaluation.compose_config(label, self.checkpoint, self.dataset, self.index, self.output, self.args())
            # Only the unused backbone initialization path changes in the model.
            original.model.encoder.pretrained_weights = ""
            self.assertEqual(OmegaConf.to_container(cfg.model), OmegaConf.to_container(original.model))
            self.assertEqual(cfg.loss, original.loss)
            self.assertTrue(cfg.model.decoder.make_scale_invariant)
            self.assertEqual(list(cfg.dataset), ["dl3dv"])
            self.assertEqual(cfg.dataset.dl3dv.name, "dl3dv")
            self.assertEqual(cfg.dataset.dl3dv.roots, [str(self.dataset)])
            self.assertEqual(cfg.dataset.dl3dv.original_image_shape, [270, 480])
            self.assertEqual(cfg.dataset.dl3dv.input_image_shape, [256, 256])
            self.assertEqual(OmegaConf.to_container(cfg.dataset.dl3dv.view_sampler),
                             {"name": "evaluation", "num_context_views": 2, "index_path": str(self.index)})
            self.assertEqual(cfg.mode, "test")
            self.assertEqual(cfg.checkpointing.load, str(self.checkpoint))
            self.assertTrue(cfg.test.align_pose)
            self.assertEqual(cfg.test.pose_align_steps, 100)
            self.assertEqual(cfg.wandb.mode, "disabled")
            self.assertEqual(cfg.model.encoder.gs_params_head_type, "moment" if label == "ours" else "dpt_gs")
            if label == "ours":
                self.assertTrue(cfg.model.encoder.moment_decoder.appearance_moments)
                self.assertEqual(cfg.model.encoder.moment_decoder.appearance_mlp_depth, 2)

    def test_dry_run_has_no_gpu_import_process_or_output_writes(self):
        with patch.object(evaluation.subprocess, "Popen") as popen, \
             patch.object(evaluation, "generate_view_index") as generate, \
             contextlib.redirect_stdout(io.StringIO()):
            evaluation.main(self.argv() + ["--dry-run"])
        popen.assert_not_called()
        generate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_parallel_workers_have_separate_gpus_and_frozen_shared_views(self):
        from omegaconf import OmegaConf

        events = []

        def spawn(command, **kwargs):
            events.append("start")
            process = Mock(pid=len(events))
            process.wait.side_effect = lambda: events.append("wait") or 0
            process.poll.return_value = 0
            return process

        with patch.object(evaluation.subprocess, "Popen", side_effect=spawn) as popen, \
             patch.object(evaluation, "summarize") as summarize, \
             patch.dict(evaluation.os.environ, {"WANDB_RUN_ID": "old-training", "WANDB_RESUME": "must"}), \
             contextlib.redirect_stdout(io.StringIO()):
            evaluation.main(self.argv() + ["--gpus", "2", "1"])
        self.assertEqual(events[:3], ["start", "start", "wait"])
        summarize.assert_called_once_with(self.output, ["ours", "baseline"])
        for call, gpu in zip(popen.call_args_list, ("2", "1")):
            env = call.kwargs["env"]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], gpu)
            self.assertEqual(env["WANDB_MODE"], "disabled")
            self.assertNotIn("WANDB_RUN_ID", env)
            self.assertNotIn("WANDB_RESUME", env)
        for label in ("ours", "baseline"):
            cfg = OmegaConf.load(self.output / label / "config.yaml")
            self.assertEqual(cfg.dataset.dl3dv.view_sampler.index_path,
                             str(self.output / "evaluation_index_dl3dv.json"))
        self.assertEqual((self.output / "evaluation_index_dl3dv.json").read_bytes(), self.index.read_bytes())

    def test_missing_checkpoints_and_shared_gpu_fail_before_launch(self):
        with patch.object(evaluation.subprocess, "Popen") as popen, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluation.main(self.argv() + ["--gpus", "0", "0"])
            self.checkpoint.unlink()
            with self.assertRaises(FileNotFoundError):
                evaluation.main(self.argv())
        popen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_worker_produces_no_comparison(self):
        process = Mock(pid=1)
        process.wait.return_value = 1
        process.poll.return_value = 1
        with patch.object(evaluation.subprocess, "Popen", return_value=process), \
             patch.object(evaluation, "summarize") as summarize, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Evaluation failed"):
                evaluation.main(self.argv())
        summarize.assert_not_called()

    def test_summary_rejects_mismatched_views(self):
        for label, target in (("ours", [5, 11, 16]), ("baseline", [6, 11, 16])):
            evaluation.write_json(self.output / label / "metrics.json", {
                "num_scenes": 1, "num_target_views": 3,
                "metrics": {"psnr": 25, "ssim": .8, "lpips": .2},
            })
            evaluation.write_json(self.output / label / "evaluated_scenes.json", {
                "scene-a": {"context": [0, 22], "target": target},
            })
        with self.assertRaisesRegex(RuntimeError, "different scenes/views"):
            evaluation.summarize(self.output, ["ours", "baseline"])
        self.assertFalse((self.output / "summary.csv").exists())

    def run_worker(self, scenes):
        from omegaconf import OmegaConf

        cfg = evaluation.compose_config("ours", self.checkpoint, self.dataset, self.index,
                                        self.output, self.args())
        self.output.mkdir()
        config_path = self.output / "config.yaml"
        OmegaConf.save(cfg, config_path)
        model = SimpleNamespace(_test_overlap_tags=("small", "medium", "large"))

        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

            def __truediv__(self, value):
                return Scalar(self.value / value)

        class Indices:
            def __init__(self, indices):
                self.indices = indices

            def tolist(self):
                return self.indices

        def test(model, datamodule, ckpt_path):
            callback = factories.Trainer.call_args.kwargs["callbacks"][0]
            for scene in scenes:
                batch = {"scene": [scene], "context": {"index": [Indices([0, 22])]},
                         "target": {"index": [Indices([5, 11, 16])]}}
                callback.on_test_batch_end(None, model, None, batch, 0)
            model._test_metric_count = Scalar(len(scenes))
            model._test_metric_sums = {f"{name}_ours": Scalar(value * len(scenes))
                                       for name, value in (("psnr", 25), ("ssim", .8), ("lpips", .2))}

        trainer = SimpleNamespace(test=Mock(side_effect=test))
        factories = SimpleNamespace(
            DataModule=Mock(), ModelWrapper=Mock(return_value=model),
            Trainer=Mock(return_value=trainer), get_decoder=Mock(),
            get_encoder=Mock(return_value=("encoder", "visualizer")), get_losses=Mock(),
            load_typed_root_config=lambda config: config, set_cfg=Mock(),
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
                                  manual_seed=Mock()),
        )
        with patch.dict("sys.modules", {"src.main": factories, "lightning.pytorch": SimpleNamespace(Callback=object)}), \
             contextlib.redirect_stdout(io.StringIO()):
            evaluation.evaluate_worker(config_path)
        return factories, model, trainer

    def test_worker_restores_strict_checkpoint_and_saves_scores_without_wandb(self):
        factories, model, trainer = self.run_worker(["scene-a"])
        self.assertTrue(model.strict_loading)
        self.assertEqual(model._test_overlap_tags, ())
        self.assertEqual(factories.Trainer.call_args.kwargs["devices"], 1)
        self.assertFalse(factories.Trainer.call_args.kwargs["logger"])
        self.assertFalse(factories.Trainer.call_args.kwargs["inference_mode"])
        self.assertEqual(trainer.test.call_args.kwargs["ckpt_path"], str(self.checkpoint))
        result = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(result["num_scenes"], 1)
        self.assertEqual(result["num_target_views"], 3)
        self.assertEqual(result["metrics"], {"psnr": 25, "ssim": .8, "lpips": .2})
        self.assertEqual(len(result["evaluation_index_sha256"]), 64)

    def test_worker_rejects_silently_filtered_scenes(self):
        with self.assertRaisesRegex(RuntimeError, "requested scenes were skipped"):
            self.run_worker([])
        coverage = json.loads((self.output / "coverage.json").read_text())
        self.assertEqual(coverage["skipped_scenes"], ["scene-a"])
        self.assertFalse((self.output / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
