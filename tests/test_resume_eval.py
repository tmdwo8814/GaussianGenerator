"""CPU-only regression checks for evaluating checkpoints in an existing run.

Extract production definitions with AST, as in test_auto_eval_logging.py, to
exercise orchestration without CUDA, Lightning, credentials, or network access.
"""

import ast
import contextlib
import io
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class Config(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def load_definitions():
    path = Path(__file__).resolve().parents[1] / "src/eval_checkpoints.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    for node in definitions:
        node.decorator_list = []
    namespace = {
        "json": json, "os": os, "re": re, "time": time, "traceback": traceback,
        "datetime": datetime, "timezone": timezone, "Path": Path,
        "DictConfig": Config, "Callback": object, "WandbLogger": Mock(),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ResumeEvalTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_definitions()
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir()
        self.paths = {}
        for epoch, step in ((2, 20000), (5, 40000), (8, 60000), (11, 80000)):
            path = self.checkpoints / f"epoch_{epoch}-step_{step}.ckpt"
            path.touch()
            self.paths[step] = path
        (self.checkpoints / "last.ckpt").touch()
        self.dataset_root = self.root / "dataset"
        test_dir = self.dataset_root / "test"
        test_dir.mkdir(parents=True)
        (test_dir / "chunk.torch").touch()
        (test_dir / "index.json").write_text('{"acid-scene": "chunk.torch"}', encoding="utf-8")
        self.eval_index = self.root / "evaluation_index_acid.json"
        self.eval_index.write_text('{"acid-scene": {}, "excluded": null}', encoding="utf-8")

    def config(self):
        return Config(
            mode="train", checkpointing=Config(load=None),
            resume_eval=Config(checkpoint_dir=str(self.checkpoints),
                               steps=[20000, 40000, 60000, 80000]),
            wandb=Config(entity="tmdwo8814-", project="noposplat", id="p05s37ze",
                         mode="online", name="original training name", tags=["acid"]),
            dataset=Config(re10k=Config(
                roots=[str(self.dataset_root)],
                view_sampler=Config(name="bounded", num_context_views=2),
            )),
            trainer=Config(num_nodes=1, auto_eval_index_path=str(self.eval_index),
                           auto_eval_save_image=False, auto_eval_save_video=False,
                           auto_eval_save_compare=False),
            test=Config(align_pose=True, compute_scores=True, output_path=self.root / "test"),
            train=Config(distiller=""), model=Config(encoder="encoder", decoder="decoder"),
            optimizer="optimizer", loss="loss", data_loader="data_loader", seed=111123,
        )

    def setup_evaluation(self, cfg, num_samples=3):
        run = Mock(path="tmdwo8814-/noposplat/p05s37ze", url="https://example.test/run")
        logger = SimpleNamespace(experiment=run)
        trainer = Mock(global_rank=0, logger=logger)
        model = SimpleNamespace(test_cfg=cfg.test,
                                _test_metric_count=SimpleNamespace(item=lambda: num_samples))
        self.namespace.update(
            hydra=SimpleNamespace(core=SimpleNamespace(hydra_config=SimpleNamespace(
                HydraConfig=SimpleNamespace(get=lambda: SimpleNamespace(
                    runtime=SimpleNamespace(output_dir=str(self.root / "output"))
                ))
            ))),
            torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3), manual_seed=Mock()),
            wandb=SimpleNamespace(run=run, finish=Mock()),
            load_typed_root_config=lambda config: config,
            set_cfg=Mock(), get_encoder=Mock(return_value=("encoder", "visualizer")),
            get_decoder=Mock(), get_losses=Mock(), ModelWrapper=Mock(return_value=model),
            DataModule=Mock(), get_distiller=Mock(),
        )
        self.namespace["WandbLogger"].return_value = logger

        def create_trainer(**kwargs):
            trainer.callbacks = kwargs["callbacks"]
            return trainer

        self.namespace["Trainer"] = Mock(side_effect=create_trainer)

        def test(current_model, **kwargs):
            for callback in trainer.callbacks:
                callback.on_test_start(trainer, current_model)
                for batch_idx in range(2):
                    callback.on_test_batch_end(trainer, current_model, None, None, batch_idx)
                callback.on_test_end(trainer, current_model)

        trainer.test.side_effect = test
        return trainer, model, run

    def evaluate(self, cfg):
        with contextlib.redirect_stdout(io.StringIO()):
            self.namespace["evaluate"](cfg)

    def records(self, rank=0):
        path = self.root / "output" / f"progress-rank{rank}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_selects_requested_steps_and_ignores_last_checkpoint(self):
        actual = self.namespace["select_checkpoints"](self.checkpoints, [80000, 20000])
        self.assertEqual(actual, [(step, self.paths[step]) for step in (20000, 80000)])

    def test_missing_checkpoint_fails_before_opening_wandb(self):
        self.paths[80000].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "80000"):
            self.evaluate(self.config())
        self.namespace["WandbLogger"].assert_not_called()

    def test_duplicate_checkpoint_is_rejected(self):
        (self.checkpoints / "auto_eval_step_00020000.ckpt").touch()
        with self.assertRaisesRegex(ValueError, "Multiple checkpoints for step 20000"):
            self.namespace["select_checkpoints"](self.checkpoints, [20000])

    def test_invalid_step_lists_are_rejected(self):
        for steps in ([], [0], [-1], [True], [1.5], [20000, 20000]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.namespace["select_checkpoints"](self.checkpoints, steps)

    def test_preflight_uses_acid_index_and_post_eval_flags(self):
        cfg = self.config()
        self.namespace["prepare_eval_config"](cfg)
        self.assertEqual(cfg.mode, "test")
        self.assertEqual(cfg.dataset.re10k.view_sampler, {
            "name": "evaluation", "index_path": str(self.eval_index), "num_context_views": 2
        })
        self.assertFalse(cfg.test.save_image)
        self.assertTrue(cfg.test.align_pose)

    def test_index_mismatch_fails_before_opening_wandb(self):
        self.eval_index.write_text('{"re10k-scene": {}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shares no scenes"):
            self.evaluate(self.config())
        self.namespace["WandbLogger"].assert_not_called()

    def test_missing_test_chunks_are_rejected(self):
        (self.dataset_root / "test" / "chunk.torch").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "No test .torch chunks"):
            self.namespace["prepare_eval_config"](self.config())

    def test_logger_requires_online_run_id_and_preserves_metadata(self):
        cfg = self.config()
        self.namespace["create_resume_logger"](cfg.wandb, self.root)
        self.namespace["WandbLogger"].assert_called_once_with(
            entity="tmdwo8814-", project="noposplat", id="p05s37ze", resume="must",
            mode="online", log_model=False, save_dir=self.root,
        )
        for key, value in (("mode", "offline"), ("id", None), ("entity", None), ("project", None)):
            cfg = self.config()
            cfg.wandb[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.namespace["create_resume_logger"](cfg.wandb, self.root)

    def test_evaluates_four_checkpoints_without_training_and_flushes_logs(self):
        cfg = self.config()
        trainer, model, run = self.setup_evaluation(cfg)
        self.evaluate(cfg)
        trainer.fit.assert_not_called()
        self.assertEqual([call.kwargs["ckpt_path"] for call in trainer.test.call_args_list],
                         [str(self.paths[step]) for step in (20000, 40000, 60000, 80000)])
        self.assertEqual(model._auto_eval_log_prefix, "auto_eval")
        self.assertEqual(model._auto_eval_step, 80000)
        self.assertFalse(self.namespace["Trainer"].call_args.kwargs["inference_mode"])
        self.namespace["wandb"].finish.assert_called_once_with(exit_code=0)
        completed = [row for row in self.records() if row["event"] == "checkpoint_complete"]
        self.assertEqual([row["ckpt_step"] for row in completed], [20000, 40000, 60000, 80000])
        self.assertTrue(all(row["num_samples"] == 3 for row in completed))
        self.assertTrue(all(call.kwargs == {"commit": True} for call in run.log.call_args_list))
        self.assertTrue(all(key.startswith("resume_eval/") for call in run.log.call_args_list
                            for key in call.args[0]))

    def test_empty_metrics_stop_remaining_checkpoints_and_save_failure(self):
        cfg = self.config()
        trainer, _, _ = self.setup_evaluation(cfg, num_samples=0)
        with self.assertRaisesRegex(RuntimeError, "no test metrics accumulated"):
            self.evaluate(cfg)
        self.assertEqual(trainer.test.call_count, 1)
        failure = next(row for row in self.records() if row["event"] == "failed")
        self.assertIn("RuntimeError", failure["traceback"])
        self.namespace["wandb"].finish.assert_called_once_with(exit_code=1)

    def test_evaluation_exception_is_saved_before_finishing_wandb(self):
        cfg = self.config()
        trainer, _, _ = self.setup_evaluation(cfg)
        trainer.test.side_effect = RuntimeError("CUDA failure")
        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            self.evaluate(cfg)
        events = [row["event"] for row in self.records()]
        self.assertLess(events.index("failed"), events.index("finishing_wandb"))
        self.namespace["wandb"].finish.assert_called_once_with(exit_code=1)

    def test_wrong_run_identity_stops_before_progress_is_logged(self):
        cfg = self.config()
        _, _, run = self.setup_evaluation(cfg)
        run.path = "someone/another-project/another-id"
        with self.assertRaisesRegex(RuntimeError, "Unexpected W&B run"):
            self.evaluate(cfg)
        run.log.assert_not_called()

    def test_nonzero_rank_writes_progress_without_accessing_wandb(self):
        progress = self.namespace["ResumeEvalProgress"](self.root / "output", "expected")
        trainer = SimpleNamespace(global_rank=1)  # Deliberately has no logger.
        model = SimpleNamespace(_auto_eval_step=20000)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            progress.on_test_start(trainer, model)
            for batch_idx in range(11):
                progress.on_test_batch_end(trainer, model, None, None, batch_idx)
            progress.on_test_end(trainer, model)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual([row["completed_batches"] for row in self.records(rank=1)], [0, 1, 10, 11])


if __name__ == "__main__":
    unittest.main()
