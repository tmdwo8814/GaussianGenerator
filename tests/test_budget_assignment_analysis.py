"""Validate selection provenance, fail-closed execution and real decoder control.

Run on the server: python -m unittest discover -s tests \
    -p test_budget_assignment_analysis.py -v
CPU tests use the actual decoder; none generate example plots or model results.
"""

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.visualize_budget_assignment import (
    ROOT,
    encoder_weights,
    infer_budget_pair,
    incoming_evidence,
    main,
    resolve_selections,
    restore_budget_path,
)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "assets/budget_analysis_samples.json").read_text(encoding="utf-8")
        )
        self.index = json.loads(
            (ROOT / "assets/evaluation_index_re10k.json").read_text(encoding="utf-8")
        )
        self.manifest = {"image_size_wh": [256, 256], "samples": []}
        for sample in self.config["samples"]:
            path = Path(sample["image_path"])
            scene = path.parent.name.rsplit("_", 1)[-1]
            entry = self.index[scene]
            self.manifest["samples"].append({
                "scene": scene, "context": entry["context"], "target": entry["target"],
                "chunk_path": "/dataset/test/000000.torch",
                "images": [
                    {"path": f"context_{view}_frame_{frame:06d}.png"}
                    for view, frame in enumerate(entry["context"])
                ],
            })

    def test_five_user_pixels_and_context_order(self):
        selections = resolve_selections(self.config, self.index, self.manifest)
        self.assertEqual([s["slot"] for s in selections],
                         [34438, 42706, 126876, 43154, 102730])
        self.assertEqual([s["view"] for s in selections], [0, 0, 1, 0, 1])
        self.assertEqual(selections[2]["context"], [8, 63])
        changed = copy.deepcopy(self.config)
        changed["samples"][0]["pixel_xy"] = [256, 134]
        with self.assertRaisesRegex(ValueError, "Invalid pixel"):
            resolve_selections(changed, self.index, self.manifest)

    def test_lightning_and_plain_state_extraction(self):
        self.assertEqual(encoder_weights({"state_dict": {
            "encoder.a": 1, "encoder.b": 2, "losses.c": 3,
        }}), {"a": 1, "b": 2})
        self.assertEqual(encoder_weights({"model": {"a": 1}}), {"a": 1})
        self.assertEqual(encoder_weights({"a": 1}), {"a": 1})

    def test_missing_server_checkpoint_never_creates_a_plot_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.ckpt"
            config = dict(self.config, ours_checkpoint=str(missing),
                          baseline_checkpoint=str(missing))
            path = root / "samples.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as caught:
                    main(["--samples", str(path), "--output-dir", str(root / "output")])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("No plots were generated", errors.getvalue())
            self.assertFalse((root / "output").exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class ActualDecoderTests(unittest.TestCase):
    def test_restore_original_budget_weights_and_control_only_budget(self):
        import torch
        from torch import nn

        from test_moment_gaussian_decoder import Cfg, Decoder, build_knn

        torch.manual_seed(83)
        decoder = Decoder(Cfg(feature_dim=5, hidden_dim=8, num_neighbors=3,
                              chunk_size=4, checkpoint_chunks=False)).eval()
        points, features = torch.randn(1, 9, 3), torch.randn(1, 9, 5)
        graph = build_knn(points[0], 3)
        # Original A, before adding the checkpoint's budget path.
        with torch.no_grad():
            raw = decoder.predict_allocation(points[0], features[0], graph)
            allocation = raw / raw.sum(-1, keepdim=True)
        weight = torch.tensor([[2., -1., .5, .2, -.3]])
        bias = torch.tensor([.1])
        state = {f"gaussian_decoder.{k}": v.clone()
                 for k, v in decoder.state_dict().items()}
        state["gaussian_decoder.budget_head.weight"] = weight
        state["gaussian_decoder.budget_head.bias"] = bias
        restore_budget_path(decoder, state, 2.)
        decoder.load_state_dict({k[len("gaussian_decoder."):]: v
                                 for k, v in state.items()}, strict=True)
        with torch.no_grad():
            actual = decoder.predict_allocation(points[0], features[0], graph)
        expected_budget = 2 * (features[0] @ weight.T + bias).sigmoid()
        torch.testing.assert_close(actual, allocation * expected_budget)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.gaussian_decoder = decoder

            def forward(self, context, step):
                return self.gaussian_decoder(context["points"], context["features"])

        before = {k: v.clone() for k, v in decoder.state_dict().items()}
        learned, constant, lt, ct = infer_budget_pair(
            Encoder(), {"points": points, "features": features}, 0
        )
        torch.testing.assert_close(lt["allocation"], ct["allocation"], rtol=0, atol=0)
        torch.testing.assert_close(ct["budget"], lt["budget"].mean().expand(9))
        self.assertIs(lt["neighbors"], ct["neighbors"])
        self.assertIs(lt["args"][0], ct["args"][0])
        self.assertIs(lt["args"][1], ct["args"][1])
        self.assertGreater(float((lt["weights"] - ct["weights"]).abs().max()), .01)
        self.assertGreater(
            float((learned.opacities - constant.opacities).abs().max()), .01
        )
        for key, value in decoder.state_dict().items():
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)
        self.assertFalse(decoder._forward_pre_hooks)
        self.assertFalse(decoder.budget_head._forward_hooks)

    def test_no_budget_checkpoint_cannot_be_used_as_learned_budget(self):
        with self.assertRaisesRegex(ValueError, "no learned budget head"):
            restore_budget_path(None, {}, 2.)

    def test_selects_incoming_edges_across_views(self):
        import numpy as np
        import torch

        graph = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])
        q = torch.tensor([[.2, .8], [.3, .7], [.4, .6], [.5, .5]])
        trace = dict(neighbors=graph, q=q, budget=q.sum(-1), weights=q,
                     allocation=q, args=(torch.arange(12).reshape(1, 4, 3),))
        data = incoming_evidence(trace, trace, 0, (1, 2))
        np.testing.assert_array_equal(data["source_index"], [0, 1, 2, 3])
        np.testing.assert_array_equal(data["source_view"], [0, 0, 1, 1])
        np.testing.assert_allclose(data["q_learned"], [.2, .7, .6, .5])


if __name__ == "__main__":
    unittest.main()
