"""Selection checks and budget intervention invariants (no CUDA needed)."""

import copy
import importlib.util
import json
import unittest

from scripts.visualize_budget_assignment import (
    ROOT,
    budget_intervention,
    incoming_evidence,
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


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class BudgetInterventionTests(unittest.TestCase):
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
