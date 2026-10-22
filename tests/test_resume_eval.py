"""CPU-only checks for checkpoint selection and resumed evaluation orchestration.

Like test_auto_eval_logging.py, extract production functions with AST so these
checks do not need CUDA, Lightning, W&B credentials, or network access.
Run: python -m unittest discover -s tests -p 'test_*eval*.py'
"""

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shlex
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_auto_eval_logging import Scalar


class Config(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def load_functions():
    path = Path(__file__).resolve().parents[1] / "src/eval_checkpoints.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    namespace = {"Path": Path, "re": re, "DictConfig": Config, "WandbLogger": Mock(),
                 "json": json, "os": os, "shlex": shlex, "__file__": str(path)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ResumeEvalTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_functions()
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir()
        self.checkpoints = {}
        for epoch, step in ((0, 20000), (0, 40000), (0, 60000), (1, 80000),
                            (1, 100000), (2, 120000), (2, 140000)):
            checkpoint = self.checkpoint_dir / f"epoch_{epoch}-step_{step}.ckpt"
            checkpoint.touch()
            self.checkpoints[step] = checkpoint
        (self.checkpoint_dir / "last.ckpt").touch()
        self.index = self.root / "index.json"
        self.index.write_text("{}", encoding="utf-8")

    def config(self):
        return Config(
            mode="train",
            checkpointing=Config(load=None),
            resume_eval=Config(
                checkpoint_dir=str(self.checkpoint_dir), steps=[80000, 100000, 120000, 140000]
            ),
            wandb=Config(entity="tmdwo8814-", project="noposplat", id="qqvj055m",
                         mode="online", name="final_plus", tags=["final_plus"]),
            dataset=Config(re10k=Config(view_sampler=Config(
                name="bounded", num_context_views=2, warm_up_steps=75000
            ))),
            trainer=Config(num_nodes=1, auto_eval_index_path=str(self.index),
                           auto_eval_save_image=False, auto_eval_save_video=False,
                           auto_eval_save_compare=False),
            test=Config(align_pose=True, compute_scores=True, save_image=True,
                        save_video=True, save_compare=True,
                        output_path=self.root / "test"),
            train=Config(distiller=""), model=Config(encoder="encoder", decoder="decoder"),
            optimizer="optimizer", loss="loss", data_loader="data_loader", seed=111123,
        )

    def setup_evaluation(self, cfg, rank=0):
        trainer = Mock(global_rank=0)
        model = SimpleNamespace(test_cfg=cfg.test)

        class DataModule:
            def __init__(self, *args, global_rank=0, **kwargs):
                self.global_rank = global_rank

            def setup(self, stage=None):
                pass

        self.namespace.update(
            hydra=SimpleNamespace(core=SimpleNamespace(hydra_config=SimpleNamespace(
                HydraConfig=SimpleNamespace(get=lambda: SimpleNamespace(
                    runtime=SimpleNamespace(output_dir=str(self.root))
                ))
            ))),
            torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3),
                                  manual_seed=Mock()),
            wandb=SimpleNamespace(run=None, finish=Mock()),
            inherit_training_auth=Mock(return_value="SDK saved login"),
            load_typed_root_config=lambda config: config,
            set_cfg=Mock(), Trainer=Mock(return_value=trainer),
            get_encoder=Mock(return_value=("encoder", "visualizer")),
            get_decoder=Mock(return_value="decoder"), get_losses=Mock(return_value=[]),
            ModelWrapper=Mock(return_value=model), DataModule=DataModule,
        )

        def test(current_model, *, datamodule, ckpt_path):
            # Standalone DDP assigns ranks after DataModule construction.
            trainer.global_rank = rank
            datamodule.trainer = trainer
            datamodule.setup("test")
            self.assertEqual(datamodule.global_rank, rank)
            self.assertTrue(current_model.strict_loading)
            current_model._test_metric_count = Scalar(2)
            current_model._test_metric_sums = {
                "psnr_ours": Scalar(current_model._auto_eval_step / 10000),
                "ssim_ours": Scalar(1.6), "lpips_ours": Scalar(.2),
            }
            current_model._test_overlap_tags = ("small", "medium", "large")
            current_model._test_overlap_counts = {
                tag: Scalar(0 if tag == "medium" else 1)
                for tag in current_model._test_overlap_tags
            }
            current_model._test_overlap_sums = {
                tag: {name: Scalar(1) for name in current_model._test_metric_sums}
                for tag in current_model._test_overlap_tags
            }
            if rank == 0:
                self.namespace["wandb"].run = object()

        trainer.test.side_effect = test
        return trainer, model

    def evaluate(self, cfg):
        with contextlib.redirect_stdout(io.StringIO()):
            self.namespace["evaluate"](cfg)

    def test_selects_only_requested_checkpoints_in_step_order(self):
        result = self.namespace["select_checkpoints"](
            self.checkpoint_dir, [140000, 100000, 120000]
        )
        self.assertEqual(result, [(step, self.checkpoints[step])
                                  for step in (100000, 120000, 140000)])

    def test_missing_or_ambiguous_checkpoint_stops_before_wandb(self):
        cfg = self.config()
        self.checkpoints[140000].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "140000"):
            self.evaluate(cfg)
        self.namespace["WandbLogger"].assert_not_called()
        self.checkpoints[140000].touch()
        (self.checkpoint_dir / "auto_eval_step_00100000.ckpt").touch()
        with self.assertRaisesRegex(ValueError, "Multiple checkpoints for step 100000"):
            self.evaluate(cfg)
        self.namespace["WandbLogger"].assert_not_called()

    def test_invalid_step_lists_are_rejected(self):
        for steps in ([], [0], [-1], [True], [100000.5], [100000, 100000]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.namespace["select_checkpoints"](self.checkpoint_dir, steps)

    def test_uses_same_sampler_and_flags_as_post_training_evaluation(self):
        cfg = self.config()
        self.namespace["prepare_eval_config"](cfg)
        self.assertEqual(cfg.mode, "test")
        self.assertEqual(cfg.dataset.re10k.view_sampler, {
            "name": "evaluation", "index_path": str(self.index), "num_context_views": 2
        })
        self.assertTrue(cfg.test.align_pose)
        self.assertFalse(cfg.test.save_image)
        self.assertFalse(cfg.test.save_video)
        self.assertFalse(cfg.test.save_compare)
        self.index.unlink()
        with self.assertRaises(FileNotFoundError):
            self.namespace["prepare_eval_config"](self.config())

    def test_logger_requires_existing_run_without_replacing_metadata(self):
        cfg = self.config()
        self.namespace["create_resume_logger"](cfg.wandb, self.root)
        self.namespace["WandbLogger"].assert_called_once_with(
            entity="tmdwo8814-", project="noposplat", id="qqvj055m", resume="must",
            mode="online", log_model=False, save_dir=self.root,
        )
        for mode, key in (("offline", None), ("disabled", None), ("online", "id"),
                          ("online", "entity"), ("online", "project")):
            cfg = self.config()
            cfg.wandb.mode = mode
            if key is not None:
                del cfg.wandb[key]
            with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                self.namespace["create_resume_logger"](cfg.wandb, self.root)

    def test_evaluates_four_checkpoints_without_training_and_flushes_logs(self):
        cfg = self.config()
        trainer, model = self.setup_evaluation(cfg)
        observed = []
        populate_metrics = trainer.test.side_effect

        def test(current_model, *, datamodule, ckpt_path):
            populate_metrics(current_model, datamodule=datamodule, ckpt_path=ckpt_path)
            observed.append((current_model._auto_eval_step,
                             current_model._auto_eval_log_prefix, ckpt_path,
                             current_model.test_cfg.output_path))

        trainer.test.side_effect = test
        self.evaluate(cfg)
        trainer.fit.assert_not_called()
        self.assertEqual(observed, [
            (step, "auto_eval", str(self.checkpoints[step]),
             self.root / "test" / f"step_{step:08d}")
            for step in (80000, 100000, 120000, 140000)
        ])
        for step, _, checkpoint, output in observed:
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["checkpoint"], checkpoint)
            self.assertEqual(saved["metrics"]["auto_eval/ckpt_step"], step)
            self.assertEqual(saved["metrics"]["auto_eval/overall/psnr"], step / 20000)
            self.assertNotIn("auto_eval/medium/psnr", saved["metrics"])
        trainer_options = self.namespace["Trainer"].call_args.kwargs
        self.assertEqual(trainer_options["devices"], "auto")
        self.assertEqual(trainer_options["strategy"], "ddp_find_unused_parameters_true")
        self.assertFalse(trainer_options["inference_mode"])
        self.assertFalse(trainer_options["enable_checkpointing"])
        self.namespace["wandb"].finish.assert_called_once_with(exit_code=0)

    def test_failure_stops_later_checkpoints_and_flushes_logs_as_failed(self):
        cfg = self.config()
        trainer, _ = self.setup_evaluation(cfg)
        original = trainer.test.side_effect

        def fail_second(model, **kwargs):
            if model._auto_eval_step == 100000:
                raise RuntimeError("evaluation failed")
            original(model, **kwargs)

        trainer.test.side_effect = fail_second
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            self.evaluate(cfg)
        self.assertEqual(trainer.test.call_count, 2)
        self.namespace["wandb"].finish.assert_called_once_with(exit_code=1)
        self.assertTrue((self.root / "test/step_00080000/metrics.json").is_file())
        self.assertFalse((self.root / "test/step_00100000/metrics.json").is_file())

    def test_nonzero_rank_uses_own_shard_and_does_not_write_results(self):
        cfg = self.config()
        trainer, _ = self.setup_evaluation(cfg, rank=2)
        self.evaluate(cfg)
        self.assertEqual(trainer.test.call_count, 4)
        self.assertFalse(list(self.root.rglob("metrics.json")))
        self.namespace["wandb"].finish.assert_not_called()

    def test_dry_run_stops_before_auth_logger_or_trainer(self):
        cfg = self.config()
        cfg.resume_eval.dry_run = True
        self.setup_evaluation(cfg)
        self.evaluate(cfg)
        for name in ("inherit_training_auth", "WandbLogger", "Trainer"):
            self.namespace[name].assert_not_called()

    def test_empty_evaluation_is_not_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError, "no scored samples"):
            self.namespace["save_evaluation_result"](
                SimpleNamespace(_test_metric_count=Scalar(0)), 80000,
                self.checkpoints[80000], self.root / "empty",
            )
        self.assertFalse((self.root / "empty").exists())

    def test_auth_reuses_literal_only_and_preserves_exported_key(self):
        script = self.root / "train.sh"
        inherit = self.namespace["inherit_training_auth"]
        for assignment in ('export WANDB_API_KEY="fake-test-key"',
                           'export WANDB_API_KEY="${WANDB_API_KEY:-fake-test-key}"'):
            script.write_text(assignment + "\nexit 99\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(inherit(script), "training script")
                self.assertEqual(os.environ["WANDB_API_KEY"], "fake-test-key")
        with patch.dict(os.environ, {"WANDB_API_KEY": "already-exported"}, clear=True):
            self.assertEqual(inherit(script), "environment")
            self.assertEqual(os.environ["WANDB_API_KEY"], "already-exported")

    def test_auth_never_executes_shell_expressions(self):
        script = self.root / "train.sh"
        for assignment in ('export WANDB_API_KEY="$(touch should-not-exist)"',
                           'export WANDB_API_KEY="$ANOTHER_KEY"',
                           'export WANDB_API_KEY=`whoami`'):
            script.write_text(assignment, encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(self.namespace["inherit_training_auth"](script), "SDK saved login")
                self.assertNotIn("WANDB_API_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()
