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
from scripts import eval_acid2dl3dv as acid_evaluation
from scripts import eval_re10k2acid as re10k_acid_evaluation


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

    def test_progress_eta_excludes_initialization_and_throttles_updates(self):
        path = self.root / "progress.json"
        progress = evaluation.EvaluationProgress("ours", 4, path)
        with patch.object(evaluation, "monotonic", side_effect=[100, 200, 201, 210, 220]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            progress.report(0, phase="initializing", force=True)
            self.assertIsNone(json.loads(path.read_text())["eta_seconds"])
            progress.report(0, force=True)
            progress.report(1)  # Avoid printing on every fast scene.
            self.assertEqual(json.loads(path.read_text())["completed"], 0)
            progress.report(2)
            snapshot = json.loads(path.read_text())
            self.assertEqual(snapshot["elapsed_seconds"], 10)
            self.assertEqual(snapshot["eta_seconds"], 10)
            progress.report(4)
        self.assertIn("2/4 scenes (50.0%)", output.getvalue())
        self.assertIn("elapsed=00:00:10 | ETA=00:00:10", output.getvalue())
        self.assertEqual(json.loads(path.read_text())["eta_seconds"], 0)

    def test_monitor_displays_each_model_and_keeps_waiting_after_one_fails(self):
        ours = Mock()
        ours.poll.side_effect = [None, 0]
        ours.wait.return_value = 0
        baseline = Mock()
        baseline.poll.return_value = 1
        baseline.wait.return_value = 1
        for label in ("ours", "baseline"):
            evaluation.write_json(self.output / label / "progress.json", {
                "phase": "evaluating", "completed": 1, "total": 4,
                "elapsed_seconds": 5, "eta_seconds": 15,
            })
        with patch.object(evaluation, "sleep") as sleep, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            failures = evaluation.monitor_jobs(
                [("ours", ours), ("baseline", baseline)], self.output
            )
        self.assertEqual(failures, [("baseline", 1)])
        sleep.assert_called_once_with(1)
        self.assertEqual(output.getvalue().count("[Progress][ours]"), 1)
        self.assertIn("[Progress][baseline]", output.getvalue())
        self.assertIn("ETA=00:00:15", output.getvalue())
        self.assertIn("baseline FAILED", output.getvalue())
        self.assertIn("ours finished", output.getvalue())

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

    def test_acid_preserves_trained_model_and_uses_identical_dl3dv_protocol(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        for label, experiment in evaluation.EXPERIMENTS_BY_SOURCE["acid"].items():
            with initialize_config_dir(
                version_base=None, config_dir=str(evaluation.REPO_ROOT / "config")
            ):
                original = compose(config_name="main", overrides=[f"+experiment={experiment}"])
            cfg = evaluation.compose_config(
                label, self.checkpoint, self.dataset, self.index, self.output, self.args(),
                training_dataset="acid",
            )
            re10k = evaluation.compose_config(
                label, self.checkpoint, self.dataset, self.index, self.output, self.args()
            )
            original.model.encoder.pretrained_weights = ""
            self.assertEqual(OmegaConf.to_container(cfg.model), OmegaConf.to_container(original.model))
            self.assertEqual(cfg.loss, original.loss)
            self.assertEqual(list(cfg.dataset), ["dl3dv"])
            self.assertEqual(cfg.dataset, re10k.dataset)
            self.assertEqual(cfg.test, re10k.test)
            self.assertEqual(cfg.data_loader.test, re10k.data_loader.test)
            self.assertEqual(cfg.seed, re10k.seed)
            self.assertIn("acid", cfg.wandb.tags)
            self.assertFalse(cfg.trainer.auto_eval)
            self.assertEqual(cfg.checkpointing.load, str(self.checkpoint))
            self.assertEqual(cfg.model.encoder.gs_params_head_type,
                             "moment" if label == "ours" else "dpt_gs")

    def test_acid_entry_point_defaults_and_dry_run_output(self):
        with contextlib.redirect_stdout(io.StringIO()) as help_output:
            with self.assertRaises(SystemExit) as exit_status:
                acid_evaluation.main(["--help"])
        self.assertEqual(exit_status.exception.code, 0)
        self.assertIn("Final++_acid.ckpt", help_output.getvalue())
        self.assertIn("noposplat_acid.ckpt", help_output.getvalue())
        self.assertNotIn("Final++_re10k.ckpt", help_output.getvalue())
        # Omit --output-dir to verify ACID results go into their own directory.
        with contextlib.redirect_stdout(io.StringIO()) as output, \
             patch.object(evaluation.subprocess, "Popen") as popen:
            acid_evaluation.main(self.argv()[:-2] + ["--dry-run"])
        self.assertIn("experiment=acid_moment", output.getvalue())
        self.assertIn("baseline: experiment=acid,", output.getvalue())
        self.assertIn("acid2dl3dv", output.getvalue())
        popen.assert_not_called()

    def test_acid_launches_both_models_with_correct_configs_and_shared_views(self):
        from omegaconf import OmegaConf

        process = Mock(pid=1)
        process.poll.return_value = 0
        process.wait.return_value = 0
        with patch.object(evaluation.subprocess, "Popen", return_value=process) as popen, \
             patch.object(evaluation, "summarize") as summarize, \
             contextlib.redirect_stdout(io.StringIO()):
            acid_evaluation.main(self.argv() + ["--gpus", "0", "1"])
        self.assertEqual(popen.call_count, 2)
        for label, call, gpu in zip(("ours", "baseline"), popen.call_args_list, ("0", "1")):
            cfg = OmegaConf.load(self.output / label / "config.yaml")
            self.assertIn("acid", cfg.wandb.tags)
            self.assertEqual(list(cfg.dataset), ["dl3dv"])
            self.assertEqual(cfg.dataset.dl3dv.view_sampler.index_path,
                             str(self.output / "evaluation_index_dl3dv.json"))
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], gpu)
        summarize.assert_called_once_with(self.output, ["ours", "baseline"])

    def test_re10k_to_acid_preserves_models_and_native_acid_preprocessing(self):
        for label in ("ours", "baseline"):
            cfg = evaluation.compose_config(
                label, self.checkpoint, self.dataset, self.index, self.output, self.args(),
                evaluation_dataset="acid",
            )
            source = evaluation.compose_config(
                label, self.checkpoint, self.dataset, self.index, self.output, self.args()
            )
            native_acid = evaluation.compose_config(
                label, self.checkpoint, self.dataset, self.index, self.output, self.args(),
                training_dataset="acid", evaluation_dataset="acid",
            )
            self.assertEqual(cfg.model, source.model)
            self.assertEqual(cfg.loss, source.loss)
            self.assertEqual(cfg.dataset, native_acid.dataset)
            self.assertEqual(list(cfg.dataset), ["re10k"])
            self.assertEqual(cfg.dataset.re10k.roots, [str(self.dataset)])
            self.assertEqual(cfg.dataset.re10k.original_image_shape, [360, 640])
            self.assertEqual(cfg.dataset.re10k.input_image_shape, [256, 256])
            self.assertEqual(cfg.dataset.re10k.view_sampler.name, "evaluation")
            self.assertEqual(cfg.test, source.test)
            self.assertEqual(cfg.cross_domain.training_dataset, "re10k")
            self.assertEqual(cfg.cross_domain.evaluation_dataset, "acid")
            self.assertEqual(cfg.cross_domain.experiment, evaluation.EXPERIMENTS[label])

    def test_acid_benchmark_index_keeps_context_endpoints_and_ignores_null_scenes(self):
        path = evaluation.REPO_ROOT / "assets/evaluation_index_acid.json"
        original = json.loads(path.read_text())
        evaluation.write_json(self.scene_index, {
            scene: "000000.torch" for scene, entry in original.items() if entry is not None
        })
        actual = evaluation.read_view_index(path, self.dataset, evaluation_dataset="acid")
        self.assertEqual(actual, original)
        self.assertTrue(any(entry is None for entry in actual.values()))
        self.assertTrue(any(set(entry["context"]) & set(entry["target"])
                            for entry in actual.values() if entry is not None))
        # A benchmark exception for ACID must not weaken the DL3DV protocol.
        with self.assertRaisesRegex(ValueError, "context and target"):
            evaluation.read_view_index(path, self.dataset)

    def test_re10k_to_acid_launcher_defaults_and_separate_output(self):
        with contextlib.redirect_stdout(io.StringIO()) as help_output:
            with self.assertRaises(SystemExit) as exit_status:
                re10k_acid_evaluation.main(["--help"])
        self.assertEqual(exit_status.exception.code, 0)
        self.assertIn("Final++_re10k.ckpt", help_output.getvalue())
        self.assertIn("evaluation_index_acid.json", help_output.getvalue())
        self.assertNotIn("Final++_acid.ckpt", help_output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output, \
             patch.object(evaluation.subprocess, "Popen") as popen:
            re10k_acid_evaluation.main(self.argv()[:-2] + ["--dry-run"])
        self.assertIn("re10k2acid", output.getvalue())
        self.assertIn("ACID=", output.getvalue())
        self.assertIn("experiment=re10k_moment", output.getvalue())
        popen.assert_not_called()

    def test_re10k_to_acid_launches_with_frozen_default_benchmark_and_one_gpu_each(self):
        from omegaconf import OmegaConf

        benchmark = evaluation.REPO_ROOT / "assets/evaluation_index_acid.json"
        entries = json.loads(benchmark.read_text())
        evaluation.write_json(self.scene_index, {
            scene: "000000.torch" for scene, entry in entries.items() if entry is not None
        })
        argv = self.argv()
        position = argv.index("--index")
        del argv[position:position + 2]
        process = Mock(pid=1)
        process.poll.return_value = 0
        process.wait.return_value = 0
        with patch.object(evaluation.subprocess, "Popen", return_value=process) as popen, \
             patch.object(evaluation, "generate_view_index") as generate, \
             patch.object(evaluation, "summarize") as summarize, \
             contextlib.redirect_stdout(io.StringIO()):
            re10k_acid_evaluation.main(argv + ["--gpus", "0", "1"])
        generate.assert_not_called()
        self.assertEqual(popen.call_count, 2)
        frozen = self.output / "evaluation_index_acid.json"
        self.assertEqual(frozen.read_bytes(), benchmark.read_bytes())
        self.assertFalse((self.output / "evaluation_index_dl3dv.json").exists())
        for label, call, gpu in zip(("ours", "baseline"), popen.call_args_list, ("0", "1")):
            cfg = OmegaConf.load(self.output / label / "config.yaml")
            self.assertEqual(list(cfg.dataset), ["re10k"])
            self.assertEqual(cfg.dataset.re10k.view_sampler.index_path, str(frozen))
            self.assertEqual(cfg.cross_domain.experiment, evaluation.EXPERIMENTS[label])
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], gpu)
        summarize.assert_called_once_with(self.output, ["ours", "baseline"])

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

    def run_worker(self, scenes, *, evaluation_dataset="dl3dv"):
        from omegaconf import OmegaConf

        cfg = evaluation.compose_config(
            "ours", self.checkpoint, self.dataset, self.index, self.output, self.args(),
            evaluation_dataset=evaluation_dataset,
        )
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
            callback.on_test_start(None, model)
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
        progress = json.loads((self.output / "progress.json").read_text())
        self.assertEqual(progress["phase"], "complete")
        self.assertEqual(progress["completed"], 1)
        self.assertEqual(progress["eta_seconds"], 0)

    def test_worker_rejects_silently_filtered_scenes(self):
        with self.assertRaisesRegex(RuntimeError, "requested scenes were skipped"):
            self.run_worker([])
        coverage = json.loads((self.output / "coverage.json").read_text())
        self.assertEqual(coverage["skipped_scenes"], ["scene-a"])
        self.assertFalse((self.output / "metrics.json").exists())

    def test_worker_evaluates_acid_loader_and_records_source_and_destination(self):
        factories, model, _ = self.run_worker(["scene-a"], evaluation_dataset="acid")
        self.assertTrue(model.strict_loading)
        self.assertEqual(list(factories.DataModule.call_args.args[0]), ["re10k"])
        result = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(result["training_dataset"], "re10k")
        self.assertEqual(result["evaluation_dataset"], "acid")
        self.assertEqual(result["experiment"], "re10k_moment")
        self.assertEqual(result["num_scenes"], 1)
        progress = json.loads((self.output / "progress.json").read_text())
        self.assertEqual(progress["phase"], "complete")


if __name__ == "__main__":
    unittest.main()
