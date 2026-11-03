"""Selection checks and budget intervention invariants (no CUDA needed)."""

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
    budget_intervention,
    encoder_weights,
    incoming_evidence,
    load_encoder_checkpoint,
    main,
    selection_plan,
)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "assets/budget_analysis_samples.json").read_text(encoding="utf-8")
        )
        self.index = json.loads(
            (ROOT / "assets/evaluation_index_re10k.json").read_text(encoding="utf-8")
        )

    def test_user_selections_preserve_context_order_and_xy_slots(self):
        plan = selection_plan(self.config, self.index, None)
        self.assertEqual([s["slot"] for s in plan],
                         [34438, 42706, 126876, 43154, 102730])
        self.assertEqual([s["context_view"] for s in plan], [0, 0, 1, 0, 1])
        self.assertEqual(plan[2]["context"], [8, 63])
        self.assertEqual(plan[4]["frame_index"], 127)

    def test_wrong_frame_out_of_bounds_and_changed_manifest_are_rejected(self):
        config = copy.deepcopy(self.config)
        config["samples"][0]["image_path"] = config["samples"][0]["image_path"].replace(
            "000088", "000089"
        )
        with self.assertRaisesRegex(ValueError, "Frame differs"):
            selection_plan(config, self.index, None)
        config = copy.deepcopy(self.config)
        config["samples"][0]["pixel_xy"] = [256, 134]
        with self.assertRaisesRegex(ValueError, "outside"):
            selection_plan(config, self.index, None)
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            selection_plan(self.config, self.index, {"samples": []})

    def test_encoder_checkpoint_formats(self):
        self.assertEqual(encoder_weights({"state_dict": {
            "encoder.a": 1, "encoder.b": 2, "losses.c": 3,
        }}), {"a": 1, "b": 2})
        self.assertEqual(encoder_weights({"model": {"a": 1}}), {"a": 1})
        self.assertEqual(encoder_weights({"a": 1}), {"a": 1})
        with self.assertRaisesRegex(ValueError, "state dictionary"):
            encoder_weights({"state_dict": {}})

    def test_missing_server_checkpoint_stops_before_model_import_or_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.ckpt"
            config = dict(self.config, baseline_checkpoint=str(missing),
                          ours_checkpoint=str(missing))
            source = root / "samples.json"
            source.write_text(json.dumps(config), encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as caught:
                    main(["--samples", str(source),
                          "--output-dir", str(root / "output")])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("No plots were generated", errors.getvalue())
            self.assertFalse((root / "output").exists())

    def test_long_checkpoint_options_resolve_to_supplied_server_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "samples.json"
            source.write_text(json.dumps(self.config), encoding="utf-8")
            ours, baseline = root / "ours.ckpt", root / "baseline.ckpt"
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as caught:
                    main(["--samples", str(source),
                          "--ours-checkpoint", str(ours),
                          "--baseline-checkpoint", str(baseline)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn(str(baseline), errors.getvalue())


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class BudgetInterventionTests(unittest.TestCase):
    def test_checkpoint_loading_is_strict(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.ckpt"
            reference = torch.nn.Linear(3, 1)
            model = torch.nn.Linear(3, 1)
            state = {f"encoder.{k}": v for k, v in reference.state_dict().items()}
            torch.save({"state_dict": state, "global_step": 17}, path)
            step, count = load_encoder_checkpoint(model, path)
            self.assertEqual((step, count), (17, 2))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[key])
            with self.assertRaisesRegex(ValueError, "no learned budget head"):
                load_encoder_checkpoint(model, path, require_budget=True)
            del state["encoder.bias"]
            torch.save({"state_dict": state}, path)
            with self.assertRaises(RuntimeError):
                load_encoder_checkpoint(model, path)

    def test_budget_control_keeps_assignment_and_total_mass(self):
        import torch
        from torch import nn

        from test_moment_gaussian_decoder import Cfg, Decoder

        torch.manual_seed(83)
        decoder = Decoder(Cfg(
            feature_dim=5, hidden_dim=8, num_neighbors=3,
            chunk_size=4, checkpoint_chunks=False,
        )).eval()
        with torch.no_grad():
            decoder.budget_head.weight.copy_(torch.tensor([[2., -1., .5, .2, -.3]]))
        points = torch.randn(1, 9, 3)
        features = torch.randn(1, 9, 5)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.gaussian_decoder = decoder

            def forward(self, context, step):
                return self.gaussian_decoder(context["points"], context["features"])

        original_state = {k: v.clone() for k, v in decoder.state_dict().items()}
        learned, constant, lt, ct = budget_intervention(
            Encoder(), {"points": points, "features": features}, 0
        )
        self.assertGreater(float(lt["budget"].std()), .05)
        torch.testing.assert_close(ct["budget"], lt["budget"].mean().expand(9))
        torch.testing.assert_close(lt["allocation"], ct["allocation"], rtol=0, atol=0)
        self.assertIs(lt["neighbors"], ct["neighbors"])
        self.assertIs(lt["args"][0], ct["args"][0])
        self.assertIs(lt["args"][1], ct["args"][1])
        self.assertGreater(float((lt["weights"] - ct["weights"]).abs().max()), .01)
        self.assertGreater(
            float((learned.opacities - constant.opacities).abs().max()), .01
        )
        for key, value in decoder.state_dict().items():
            torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)
        self.assertFalse(decoder._forward_pre_hooks)
        self.assertFalse(decoder.budget_head._forward_hooks)
        self.assertFalse(decoder.allocation_head[2]._forward_hooks)
        with torch.no_grad():
            rerun = decoder(points, features)
        torch.testing.assert_close(rerun.means, learned.means, rtol=0, atol=0)

    def test_evidence_uses_incoming_edges_not_the_selected_knn_row(self):
        import numpy as np
        import torch

        # Slot 0 receives four sources despite having only two outgoing candidates.
        graph = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])
        q = torch.tensor([[.2, .8], [.3, .7], [.4, .6], [.5, .5]])
        trace = dict(
            neighbors=graph, q=q, budget=q.sum(-1),
            weights=q, allocation=q, args=(torch.arange(12).reshape(1, 4, 3),),
        )
        evidence = incoming_evidence(trace, trace, 0)
        np.testing.assert_array_equal(evidence["source_index"], [0, 1, 2, 3])
        np.testing.assert_array_equal(evidence["edge_index"], [0, 1, 1, 1])
        np.testing.assert_allclose(evidence["q_learned"], [.2, .7, .6, .5])


if __name__ == "__main__":
    unittest.main()
