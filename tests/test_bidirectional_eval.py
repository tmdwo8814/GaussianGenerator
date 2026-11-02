"""Four-GPU RE10K/ACID orchestration and results without CUDA/checkpoints."""

import contextlib
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from scripts import eval_re10k_acid as bidirectional
from scripts import eval_re10k2dl3dv as evaluation


class BidirectionalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.roots, self.indices, self.checkpoints = {}, {}, {}
        for dataset in ("re10k", "acid"):
            self.roots[dataset] = self.root / dataset
            self.indices[dataset] = self.root / f"{dataset}_views.json"
            scene = f"{dataset}-scene"
            evaluation.write_json(self.roots[dataset] / "test/index.json", {scene: "000000.torch"})
            evaluation.write_json(self.indices[dataset], {
                scene: {"context": [0, 22], "target": [5, 11, 16]},
            })
            for model in bidirectional.MODELS:
                checkpoint = self.root / evaluation.CHECKPOINT_NAMES[dataset][model]
                checkpoint.touch()
                self.checkpoints[dataset, model] = checkpoint

    def argv(self):
        argv = ["--output-dir", str(self.output)]
        for dataset in ("re10k", "acid"):
            argv += [f"--{dataset}-root", str(self.roots[dataset]),
                     f"--{dataset}-index", str(self.indices[dataset])]
            for model in bidirectional.MODELS:
                argv += [f"--{dataset}-{model}-checkpoint", str(self.checkpoints[dataset, model])]
        return argv

    def test_four_workers_start_before_wait_with_correct_weights_views_and_gpus(self):
        from omegaconf import OmegaConf

        events = []

        def spawn(command, **kwargs):
            events.append("start")
            config_path = Path(command[-1])
            evaluation.write_json(config_path.parent / "progress.json", {
                "phase": "evaluating", "completed": 1, "total": 1,
                "elapsed_seconds": 3, "eta_seconds": 0,
            })
            process = Mock(pid=len(events))
            process.poll.return_value = 0
            process.wait.side_effect = lambda: events.append("wait") or 0
            return process

        with patch.object(evaluation.subprocess, "Popen", side_effect=spawn) as popen, \
             patch.object(bidirectional, "summarize") as summarize, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            bidirectional.main(self.argv())
        self.assertEqual(events[:5], ["start"] * 4 + ["wait"])
        self.assertEqual(popen.call_count, 4)
        summarize.assert_called_once_with(self.output)
        jobs = [(source, target, model) for source, target in bidirectional.DIRECTIONS
                for model in bidirectional.MODELS]
        for gpu, ((source, target, model), call) in enumerate(zip(jobs, popen.call_args_list)):
            direction = f"{source}2{target}"
            config_path = self.output / direction / model / "config.yaml"
            cfg = OmegaConf.load(config_path)
            self.assertEqual(cfg.checkpointing.load, str(self.checkpoints[source, model]))
            self.assertEqual(cfg.cross_domain.training_dataset, source)
            self.assertEqual(cfg.cross_domain.evaluation_dataset, target)
            self.assertEqual(cfg.cross_domain.experiment, evaluation.EXPERIMENTS_BY_SOURCE[source][model])
            self.assertIn(source, cfg.wandb.tags)
            self.assertEqual(cfg.dataset.re10k.roots, [str(self.roots[target])])
            self.assertEqual(cfg.dataset.re10k.original_image_shape, [360, 640])
            self.assertEqual(cfg.model.encoder.gs_params_head_type,
                             "moment" if model == "ours" else "dpt_gs")
            frozen = self.output / direction / f"evaluation_index_{target}.json"
            self.assertEqual(cfg.dataset.re10k.view_sampler.index_path, str(frozen))
            self.assertEqual(frozen.read_bytes(), self.indices[target].read_bytes())
            self.assertEqual(cfg.test.output_path, str(self.output / direction / "renders"))
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(gpu))
            self.assertIn(f"[Progress][{direction}/{model}]", output.getvalue())

    def test_dry_run_checks_all_four_without_processes_or_writes(self):
        with patch.object(evaluation.subprocess, "Popen") as popen, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            bidirectional.main(self.argv() + ["--dry-run"])
        popen.assert_not_called()
        self.assertFalse(self.output.exists())
        for source, target in bidirectional.DIRECTIONS:
            for model in bidirectional.MODELS:
                self.assertIn(f"{source}2{target}/{model}", output.getvalue())

    def test_missing_fourth_checkpoint_prevents_every_launch(self):
        self.checkpoints["acid", "baseline"].unlink()
        with patch.object(evaluation.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(FileNotFoundError, "acid-baseline-checkpoint"):
                bidirectional.main(self.argv())
        popen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_duplicate_gpu_is_rejected_before_launch(self):
        with patch.object(evaluation.subprocess, "Popen") as popen, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bidirectional.main(self.argv() + ["--gpus", "0", "1", "1", "3"])
        popen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_one_failure_still_waits_for_other_workers_and_saves_no_summary(self):
        processes = []
        for code in (0, 1, 0, 0):
            process = Mock(pid=len(processes))
            process.poll.return_value = code
            process.wait.return_value = code
            processes.append(process)
        with patch.object(evaluation.subprocess, "Popen", side_effect=processes) as popen, \
             patch.object(bidirectional, "summarize") as summarize, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Evaluation failed"):
                bidirectional.main(self.argv())
        self.assertEqual(popen.call_count, 4)
        for process in processes:
            process.wait.assert_called()
        summarize.assert_not_called()
        self.assertFalse((self.output / "summary.csv").exists())

    def test_summary_has_four_rows_and_compares_coverage_within_each_direction(self):
        for source, target in bidirectional.DIRECTIONS:
            for model in bidirectional.MODELS:
                directory = self.output / f"{source}2{target}" / model
                evaluation.write_json(directory / "metrics.json", {
                    "training_dataset": source, "evaluation_dataset": target,
                    "num_scenes": 1, "num_target_views": 3,
                    "metrics": {"psnr": 25, "ssim": .8, "lpips": .2},
                })
                # The destinations intentionally have different scene keys.
                evaluation.write_json(directory / "evaluated_scenes.json", {
                    f"{target}-scene": {"context": [0, 22], "target": [5, 11, 16]},
                })
        with contextlib.redirect_stdout(io.StringIO()):
            bidirectional.summarize(self.output)
        result = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(len(result), 4)
        with (self.output / "summary.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["direction"] for row in rows}, {"re10k2acid", "acid2re10k"})
        self.assertEqual({row["model"] for row in rows}, {"ours", "baseline"})


if __name__ == "__main__":
    unittest.main()
