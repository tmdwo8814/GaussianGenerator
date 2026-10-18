"""CPU-only checks for checkpoint selection and resumed evaluation orchestration.

Like test_auto_eval_logging.py, extract production functions with AST so these
checks do not need CUDA, Lightning, W&B credentials, or network access.
Run: python -m unittest discover -s tests -p 'test_*eval*.py'
"""

import ast
import contextlib
import io
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


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
    namespace = {"Path": Path, "re": re, "DictConfig": Config, "WandbLogger": Mock()}
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
                checkpoint_dir=str(self.checkpoint_dir), steps=[100000, 120000, 140000]
            ),
            wandb=Config(entity="tmdwo8814-", project="noposplat", id="ob49551z",
                         mode="online", name="final", tags=["final"]),
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

    def setup_evaluation(self, cfg):
        trainer = Mock(global_rank=0)
        model = SimpleNamespace(test_cfg=cfg.test)
        self.namespace.update(
            hydra=SimpleNamespace(core=SimpleNamespace(hydra_config=SimpleNamespace(
                HydraConfig=SimpleNamespace(get=lambda: SimpleNamespace(
                    runtime=SimpleNamespace(output_dir=str(self.root))
                ))
            ))),
            torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3),
                                  manual_seed=Mock()),
            wandb=SimpleNamespace(run=object(), finish=Mock()),
            load_typed_root_config=lambda config: config,
            set_cfg=Mock(), Trainer=Mock(return_value=trainer),
            get_encoder=Mock(return_value=("encoder", "visualizer")),
            get_decoder=Mock(return_value="decoder"), get_losses=Mock(return_value=[]),
            ModelWrapper=Mock(return_value=model), DataModule=Mock(),
        )
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
            entity="tmdwo8814-", project="noposplat", id="ob49551z", resume="must",
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

    def test_evaluates_three_checkpoints_without_training_and_flushes_logs(self):
        cfg = self.config()
        trainer, model = self.setup_evaluation(cfg)
        observed = []

        def test(current_model, *, datamodule, ckpt_path):
            observed.append((current_model._auto_eval_step,
                             current_model._auto_eval_log_prefix, ckpt_path,
                             current_model.test_cfg.output_path))

        trainer.test.side_effect = test
        self.evaluate(cfg)
        trainer.fit.assert_not_called()
        self.assertEqual(observed, [
            (step, "auto_eval", str(self.checkpoints[step]),
             self.root / "test" / f"step_{step:08d}")
            for step in (100000, 120000, 140000)
        ])
        trainer_options = self.namespace["Trainer"].call_args.kwargs
        self.assertEqual(trainer_options["devices"], "auto")
        self.assertEqual(trainer_options["strategy"], "ddp_find_unused_parameters_true")
        self.assertFalse(trainer_options["inference_mode"])
        self.assertFalse(trainer_options["enable_checkpointing"])
        self.namespace["wandb"].finish.assert_called_once_with(exit_code=0)

    def test_failure_stops_later_checkpoints_and_flushes_logs_as_failed(self):
        cfg = self.config()
        trainer, _ = self.setup_evaluation(cfg)
        trainer.test.side_effect = [None, RuntimeError("evaluation failed")]
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            self.evaluate(cfg)
        self.assertEqual(trainer.test.call_count, 2)
        self.namespace["wandb"].finish.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
