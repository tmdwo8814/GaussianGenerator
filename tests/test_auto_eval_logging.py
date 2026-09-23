"""Dependency-free checks of the production metric methods.

Run with: python -m unittest discover -s tests -p test_auto_eval_logging.py
The methods are extracted with AST so CUDA/model imports are not required.
Tensor scalars, distributed collectives, and W&B are replaced with test doubles.
"""

import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class Scalar:
    def __init__(self, value=0):
        self.value = float(value)

    def item(self):
        return self.value

    def detach(self):
        return self

    def __iadd__(self, other):
        self.value += other.value if isinstance(other, Scalar) else other
        return self

    def __truediv__(self, other):
        return Scalar(self.value / other.value)


class FakeWandbLogger:
    def __init__(self):
        self.experiment = Mock()


def load_metric_class():
    path = Path(__file__).resolve().parents[1] / "src/model/model_wrapper.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelWrapper")
    names = {
        "_eval_step", "on_test_start", "_update_test_metric_buffers",
        "_reduce_test_metric_buffers", "_print_final_test_metrics",
    }
    wrapper.bases = []
    wrapper.body = [n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[wrapper], type_ignores=[])
    namespace = {
        "torch": SimpleNamespace(zeros=lambda *a, **kw: Scalar()),
        "dist": SimpleNamespace(is_available=lambda: False),
        "wandb": SimpleNamespace(run=SimpleNamespace(summary={}), log=Mock()),
        "WandbLogger": FakeWandbLogger,
        "Benchmarker": object,
        "tabulate": lambda *a, **kw: "metrics",
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["ModelWrapper"], namespace


class AutoEvalLoggingTests(unittest.TestCase):
    def setUp(self):
        self.wrapper_class, self.namespace = load_metric_class()

    def model(self, rank=0):
        model = self.wrapper_class()
        model.device = "cpu"
        model.global_rank = rank
        model.global_step = 80001
        model.logger = FakeWandbLogger()
        model.test_cfg = SimpleNamespace(compute_scores=True)
        model._auto_eval_step = 20000
        model._auto_eval_log_prefix = "auto_eval"
        model._test_metric_names = ("psnr_ours", "ssim_ours", "lpips_ours")
        model._test_overlap_tags = ("small", "medium", "large")
        model.on_test_start()
        return model

    def add(self, model, psnr, ssim, lpips, tag):
        model._update_test_metric_buffers(
            {"psnr_ours": Scalar(psnr), "ssim_ours": Scalar(ssim), "lpips_ours": Scalar(lpips)},
            tag,
        )

    def finish(self, model):
        with contextlib.redirect_stdout(io.StringIO()):
            model._print_final_test_metrics("auto_eval", model._auto_eval_step)

    def test_unequal_rank_counts_use_global_sample_average(self):
        local, remote = self.model(), self.model(rank=1)
        self.add(local, 10, 0.4, 0.3, "small")
        self.add(remote, 20, 0.7, 0.2, "large")
        self.add(remote, 30, 1.0, 0.1, "large")

        remote_tensors = [remote._test_metric_count]
        remote_tensors.extend(remote._test_metric_sums.values())
        for tag in remote._test_overlap_tags:
            remote_tensors.append(remote._test_overlap_counts[tag])
            remote_tensors.extend(remote._test_overlap_sums[tag].values())
        contributions = iter(remote_tensors)

        def all_reduce(tensor, op):
            tensor += next(contributions)

        self.namespace["dist"] = SimpleNamespace(
            is_available=lambda: True, is_initialized=lambda: True,
            ReduceOp=SimpleNamespace(SUM="sum"), all_reduce=all_reduce,
        )
        local._reduce_test_metric_buffers()
        self.finish(local)
        log = self.namespace["wandb"].log
        log.assert_called_once()
        payload = log.call_args.args[0]
        expected = {
            "auto_eval/overall/psnr": 20, "auto_eval/overall/ssim": 0.7,
            "auto_eval/overall/lpips": 0.2, "auto_eval/overall/num_samples": 3,
            "auto_eval/small/psnr": 10, "auto_eval/small/ssim": 0.4,
            "auto_eval/small/lpips": 0.3, "auto_eval/small/num_samples": 1,
            "auto_eval/large/psnr": 25, "auto_eval/large/ssim": 0.85,
            "auto_eval/large/lpips": 0.15, "auto_eval/large/num_samples": 2,
            "auto_eval/ckpt_step": 20000,
        }
        self.assertEqual(payload.keys(), expected.keys())
        for key, value in expected.items():
            self.assertAlmostEqual(payload[key], value)
        self.assertEqual(self.namespace["wandb"].run.summary, payload)

    def test_next_checkpoint_resets_metrics_and_uses_its_step(self):
        model = self.model()
        self.add(model, 10, 0.4, 0.3, "small")
        self.finish(model)
        old_benchmarker = model.benchmarker
        model._auto_eval_step = 40000
        model.on_test_start()
        self.assertIsNot(model.benchmarker, old_benchmarker)
        self.assertEqual(model._test_metric_count.item(), 0)
        self.assertEqual(model._eval_step(), 40000)
        self.add(model, 30, 0.9, 0.1, "medium")
        self.finish(model)
        payload = self.namespace["wandb"].log.call_args.args[0]
        self.assertEqual(payload["auto_eval/overall/psnr"], 30)
        self.assertEqual(payload["auto_eval/overall/num_samples"], 1)
        self.assertEqual(payload["auto_eval/ckpt_step"], 40000)
        self.assertNotIn("auto_eval/small/psnr", payload)

    def test_only_rank_zero_logs_and_defines_checkpoint_axis(self):
        model = self.model()
        model.logger.experiment.define_metric.assert_any_call(
            "auto_eval/*", step_metric="auto_eval/ckpt_step"
        )
        remote = self.model(rank=1)
        self.add(remote, 20, 0.8, 0.2, "large")
        self.finish(remote)
        remote.logger.experiment.define_metric.assert_not_called()
        self.namespace["wandb"].log.assert_not_called()

    def test_empty_disabled_and_no_wandb_run(self):
        model = self.model()
        self.finish(model)
        self.add(model, 20, 0.8, 0.2, "large")
        model.test_cfg.compute_scores = False
        self.finish(model)
        model.test_cfg.compute_scores = True
        self.namespace["wandb"].run = None
        self.finish(model)
        self.namespace["wandb"].log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
