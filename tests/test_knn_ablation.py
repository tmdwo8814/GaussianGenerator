"""CPU checks for expensive-run safeguards without CUDA/model dependencies.

Production functions are extracted with AST as in test_auto_eval_logging.py.
Run: python -m unittest discover -s tests -p test_knn_ablation.py -v
"""

import ast
import contextlib
import csv
from copy import deepcopy
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.misc.inference_timing import summarize_inference
import test_auto_eval_logging


ROOT = Path(__file__).resolve().parents[1]


def extract(path, namespace, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        self[key] = value


class InferenceTimingTests(unittest.TestCase):
    def test_frame_weighted_fps_excludes_warmup_and_optimization(self):
        records = [dict(num_frames=3, encoder_seconds=100, render_seconds=100)]
        records += [dict(num_frames=2, encoder_seconds=0.4, render_seconds=0.1),
                    dict(num_frames=4, encoder_seconds=0.3, render_seconds=0.2)]
        result = summarize_inference(records, 1)
        self.assertAlmostEqual(result["fps"], 6)
        self.assertAlmostEqual(result["render_fps"], 20)
        self.assertEqual(result["num_timed_scenes"], 2)
        self.assertEqual(result["num_timed_frames"], 6)

    def test_empty_or_invalid_timing_fails(self):
        for records, warmup in (([], 0), ([{}], 1), ([{}], -1)):
            with self.assertRaises(ValueError):
                summarize_inference(records, warmup)
        for time in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                summarize_inference([dict(num_frames=1, encoder_seconds=time,
                                          render_seconds=1)], 0)

    def test_timer_synchronizes_before_and_after_the_gpu_work(self):
        events = []
        times = iter((1.0, 1.2))
        namespace = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                synchronize=lambda device: events.append(("sync", device)))),
            "perf_counter": lambda: next(times), "Path": Path,
            "defaultdict": __import__("collections").defaultdict,
            "contextmanager": contextlib.contextmanager,
        }
        timer_class = extract("src/misc/benchmarker.py", namespace, "Benchmarker")
        timer = timer_class(synchronize_device="cuda:0")
        with timer.time("decoder", num_calls=2):
            events.append("render")
        self.assertEqual(events, [("sync", "cuda:0"), "render", ("sync", "cuda:0")])
        self.assertAlmostEqual(sum(timer.execution_times["decoder"]), 0.2)


class FrozenSceneTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = {"state_dict": {
            "encoder.backbone.weight": SimpleNamespace(shape=(3, 3)),
            "encoder.downstream_head1.weight": SimpleNamespace(shape=(3,)),
            "encoder.downstream_head2.weight": SimpleNamespace(shape=(3,)),
            "encoder.gaussian_param_head.weight": SimpleNamespace(shape=(256,)),
            "encoder.gaussian_param_head2.weight": SimpleNamespace(shape=(256,)),
            "encoder.gaussian_decoder.weight": SimpleNamespace(shape=(99,)),
        }}
        expected = {name.removeprefix("encoder."): value
                    for name, value in self.checkpoint["state_dict"].items()}
        self.encoder = SimpleNamespace(state_dict=lambda: expected,
                                       load_state_dict=Mock(), parameters=lambda: [])
        self.namespace = {"Path": Path, "torch": SimpleNamespace(
            load=lambda *args, **kwargs: self.checkpoint)}
        self.load = extract("src/misc/frozen_scene.py", self.namespace, "load_scene_encoder")

    def test_loads_all_point_and_feature_providers_and_never_loads_decoder(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = self.load(self.encoder, "Final++.ckpt")
        state = self.encoder.load_state_dict.call_args.args[0]
        self.assertEqual(report["scene_tensors_loaded"], 5)
        self.assertNotIn("gaussian_decoder.weight", state)
        self.assertIn("gaussian_param_head2.weight", state)

    def test_partial_or_wrong_shape_scene_checkpoint_is_rejected_before_loading(self):
        for mutation in ("missing", "extra", "shape"):
            with self.subTest(mutation=mutation):
                checkpoint = deepcopy(self.checkpoint)
                if mutation == "missing":
                    checkpoint["state_dict"].pop("encoder.gaussian_param_head2.weight")
                elif mutation == "extra":
                    checkpoint["state_dict"]["encoder.extra.weight"] = SimpleNamespace(shape=(1,))
                else:
                    checkpoint["state_dict"]["encoder.backbone.weight"] = SimpleNamespace(shape=(2,))
                self.namespace["torch"].load = lambda *args, **kwargs: checkpoint
                with self.assertRaisesRegex(ValueError, "Incompatible scene checkpoint"):
                    self.load(self.encoder, "Final++.ckpt")
                self.encoder.load_state_dict.assert_not_called()

    def test_train_keeps_scene_heads_in_eval_mode(self):
        class Module:
            def __init__(self):
                self.training = True

            def train(self, mode=True):
                self.training = mode
                for child in getattr(self, "children", {}).values():
                    child.train(mode)
                return self

            def eval(self):
                return self.train(False)

        tree = ast.parse((ROOT / "src/model/encoder/encoder_noposplat.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if getattr(node, "name", None) == "EncoderNoPoSplat")
        cls.bases = [ast.Name(id="Module", ctx=ast.Load())]
        cls.body = [node for node in cls.body if getattr(node, "name", None) == "train"]
        namespace = {"Module": Module}
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        exec(compile(module, "encoder", "exec"), namespace)
        encoder = namespace["EncoderNoPoSplat"]()
        encoder.cfg = SimpleNamespace(freeze_scene_encoder=True)
        encoder.children = {name: Module() for name in (
            "backbone", "downstream_head1", "downstream_head2",
            "gaussian_param_head", "gaussian_param_head2", "gaussian_decoder")}
        encoder.named_children = lambda: encoder.children.items()
        for _ in range(2):
            encoder.eval()
            encoder.train()
            self.assertTrue(encoder.children["gaussian_decoder"].training)
            self.assertTrue(all(not child.training for name, child in encoder.children.items()
                                if name != "gaussian_decoder"))


class CallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = Config(model=Config(encoder=Config(moment_decoder=Config(num_neighbors=4))))
        self.numpy_state = object()
        self.namespace = {
            "Callback": object, "deepcopy": deepcopy, "Path": Path,
            "random": random, "csv": csv, "json": json,
            "np": SimpleNamespace(random=SimpleNamespace(
                get_state=lambda: self.numpy_state, set_state=Mock())),
            "torch": SimpleNamespace(
                random=SimpleNamespace(fork_rng=lambda **kw: contextlib.nullcontext()),
                no_grad=contextlib.nullcontext,
            ),
        }
        callback_class = extract("src/misc/knn_ablation.py", self.namespace,
                                 "DecoderAblationEvalCallback")
        self.callback = callback_class(self.cfg, self.root)
        self.callback.interval = 2000
        self.callback.eval_test_cfg = SimpleNamespace(output_path=self.root)
        self.callback.eval_loaders = [["scene A", "scene B"]]
        self.trainer = SimpleNamespace(global_step=2000, save_checkpoint=Mock())
        parameter = Mock(requires_grad=True)
        parameter.requires_grad_.side_effect = lambda state: setattr(parameter, "requires_grad", state)
        self.parameter = parameter
        self.training_test_cfg = SimpleNamespace(output_path=self.root / "training")
        self.model = Mock(training=True, device=SimpleNamespace(index=0),
                          test_cfg=self.training_test_cfg)
        self.model.modules.return_value = [self.model]
        self.model.parameters.return_value = [parameter]
        self.model.eval.side_effect = lambda: setattr(self.model, "training", False)
        self.model.requires_grad_.side_effect = parameter.requires_grad_
        self.model.transfer_batch_to_device.side_effect = lambda batch, *args: batch
        self.model._test_result_payload = {
            f"auto_eval/overall/{name}": value for name, value in
            dict(psnr=25.0, ssim=0.85, lpips=0.15, fps=20.0,
                 render_fps=100.0, num_samples=2).items()
        }

    def run_callback(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.callback.on_train_batch_end(self.trainer, self.model, None, None, 0)

    def assert_restored(self):
        self.assertTrue(self.model.training)
        self.assertTrue(self.parameter.requires_grad)
        self.assertIs(self.model.test_cfg, self.training_test_cfg)
        self.assertNotIn("_auto_eval_step", self.model.__dict__)
        self.namespace["np"].random.set_state.assert_called_with(self.numpy_state)

    def test_all_five_steps_and_accumulation_do_not_duplicate_evaluation(self):
        for step in range(0, 10001, 1000):
            self.trainer.global_step = step
            self.run_callback()
            self.run_callback()  # Repeated batch hook under gradient accumulation.
        self.assertEqual(self.trainer.save_checkpoint.call_count, 5)
        self.assertEqual(self.model.test_step.call_count, 10)
        self.assertEqual([row["step"] for row in self.callback.rows],
                         [2000, 4000, 6000, 8000, 10000])
        with (self.root / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[-1]["fps"], "20.0")
        self.assert_restored()

    def test_evaluation_failure_restores_flags_modes_and_rng(self):
        def fail(*args):
            random.random()
            raise RuntimeError("pose optimizer failure")

        self.model.test_step.side_effect = fail
        state = random.getstate()
        with self.assertRaisesRegex(RuntimeError, "pose optimizer failure"):
            self.run_callback()
        self.assert_restored()
        self.assertEqual(random.getstate(), state)
        self.assertFalse((self.root / "metrics.csv").exists())

    def test_fit_rejects_unfrozen_or_extra_trainable_parameters(self):
        self.callback.cfg_dict = Config(
            model=Config(encoder=Config(freeze_scene_encoder=True)),
            trainer=Config(auto_eval=True, eval_every_n_steps=2000, max_steps=10000),
            test=Config(compute_scores=True, measure_fps=True),
            train=Config(distiller=""),
        )
        trainer = SimpleNamespace(world_size=1)
        self.model.named_parameters.return_value = [("encoder.backbone.weight", self.parameter)]
        with self.assertRaisesRegex(ValueError, "Unexpected trainable parameters"):
            self.callback.on_fit_start(trainer, self.model)
        self.callback.cfg_dict.model.encoder.freeze_scene_encoder = False
        with self.assertRaisesRegex(ValueError, "frozen scene encoders"):
            self.callback.on_fit_start(trainer, self.model)


class ArtifactTests(unittest.TestCase):
    def test_test_end_saves_quality_fps_and_per_scene_timings_together(self):
        tree = ast.parse((ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8"))
        wrapper = next(node for node in tree.body if getattr(node, "name", None) == "ModelWrapper")
        method = next(node for node in wrapper.body if getattr(node, "name", None) == "on_test_end")
        namespace = {
            "get_cfg": lambda: {"wandb": {"name": "abl_K4"}},
            "summarize_inference": summarize_inference, "json": json,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "wrapper", "exec"), namespace)
        with TemporaryDirectory() as folder:
            model = SimpleNamespace(
                test_cfg=SimpleNamespace(output_path=Path(folder), measure_fps=True,
                                         fps_warmup_steps=1, align_pose=True, pose_align_steps=100),
                trainer=SimpleNamespace(world_size=1), global_rank=0,
                encoder=SimpleNamespace(cfg=SimpleNamespace(moment_decoder=SimpleNamespace(num_neighbors=4))),
                _inference_records=[dict(scene="warmup", num_frames=3,
                                         encoder_seconds=1.0, render_seconds=1.0),
                                    dict(scene="measured", num_frames=3,
                                         encoder_seconds=0.2, render_seconds=0.1)],
                _reduce_test_metric_buffers=Mock(),
                _print_final_test_metrics=Mock(return_value={"auto_eval/overall/psnr": 25.0}),
                _eval_step=lambda: 2000, benchmarker=Mock(),
            )
            namespace["on_test_end"](model)
            result = json.loads((Path(folder) / "abl_K4/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["step"], 2000)
            self.assertEqual(result["k"], 4)
            self.assertAlmostEqual(result["timing"]["fps"], 10)
            self.assertEqual(result["metrics"]["auto_eval/overall/psnr"], 25)
            self.assertEqual(len(result["per_scene_timing"]), 2)
            self.assertFalse((Path(folder) / "abl_K4/metrics.json.tmp").exists())


class FpsLoggingTests(unittest.TestCase):
    def test_fps_and_quality_use_the_same_wandb_checkpoint_axis(self):
        helper = test_auto_eval_logging.AutoEvalLoggingTests()
        helper.setUp()
        model = helper.model()
        model.test_cfg.measure_fps = True
        model._test_inference_summary = dict(fps=12.0, render_fps=80.0, num_timed_frames=30)
        helper.add(model, 25, 0.85, 0.15, "small")
        helper.finish(model)
        payload = helper.namespace["wandb"].log.call_args.args[0]
        self.assertEqual(payload["auto_eval/overall/fps"], 12)
        self.assertEqual(payload["auto_eval/overall/psnr"], 25)
        self.assertEqual(payload["auto_eval/ckpt_step"], 20000)


if __name__ == "__main__":
    unittest.main()
