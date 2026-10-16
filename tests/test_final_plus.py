"""Final++: weighted appearance spread and expanded SH readout."""

import copy
import importlib
import unittest
from unittest.mock import patch

import torch

from test_appearance_capacity import activate, cfg, selection
from test_moment_gaussian_decoder import Decoder, attributes, moment

moments = importlib.import_module('_moment_test.encoder.common.appearance_moments')
dpt = importlib.import_module('_moment_test.encoder.heads.dpt_block')


class FinalPlusTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(419)
        torch.set_num_threads(1)

    def test_counts_for_baseline_final_a_b_and_combined(self):
        # Actual baseline module, two independently parameterized view heads.
        head = dpt.DPTOutputAdapter(num_channels=83, feature_dim=256, head_type='gs_params').head
        self.assertEqual(2 * sum(p.numel() for p in head.parameters()), 1222310)
        for name, width, depth, spread, expected in (
            ('Final', 128, 1, False, 154649), ('A', 256, 2, False, 271129),
            ('B', 256, 1, True, 270873), ('A+B', 256, 2, True, 336665),
        ):
            with self.subTest(variant=name):
                model = Decoder(cfg(feature_dim=256, hidden_dim=64, appearance_heads=2,
                                    appearance_dim=32, appearance_mlp_dim=width,
                                    appearance_mlp_depth=depth, appearance_moments=spread))
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
                self.assertEqual(model.appearance_sh_head[0].in_features, 576 if spread else 320)
                self.assertEqual(len(model.appearance_sh_head), 2 * depth + 1)
                self.assertEqual(model.appearance_sh_head[-1].weight.count_nonzero().item(), 0)
                self.assertEqual(model.appearance_sh_head[-1].bias.count_nonzero().item(), 0)

    def test_std_and_gradients_match_dense_central_variance(self):
        neighbors = torch.tensor([[0, 1, 2], [1, 3, 0], [2, 4, 0], [3, 0, 1], [4, 2, 3]])
        features = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
        scores = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
        weights = selection.appearance_weights(torch.ones_like(scores), scores, neighbors)
        mean = moment.pool_features(features, weights, neighbors, 3)
        std = moments.appearance_feature_std(features, weights, neighbors, mean, 3)
        reference = []
        for slot in range(5):
            source, edge = (neighbors == slot).nonzero(as_tuple=True)
            w, f = weights[source, edge], features[source]
            center = (w[:, None] * f).sum(0)
            reference.append(((w[:, None] * (f - center).square()).sum(0) + 1e-6).sqrt())
        reference = torch.stack(reference)
        torch.testing.assert_close(std, reference, rtol=1e-10, atol=1e-12)
        signal = torch.randn_like(std)
        for actual, expected in zip(
            torch.autograd.grad((std * signal).sum(), (features, scores), retain_graph=True),
            torch.autograd.grad((reference * signal).sum(), (features, scores)),
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_constant_features_single_support_and_chunk_boundaries(self):
        for count, chunk in ((1, 3), (7, 1), (7, 3)):
            f = torch.ones(count, 4, requires_grad=True)
            neighbors = torch.arange(count)[:, None]
            w = torch.ones(count, 1, requires_grad=True)
            mean = moment.pool_features(f, w, neighbors, chunk)
            std = moments.appearance_feature_std(f, w, neighbors, mean, chunk)
            torch.testing.assert_close(std, torch.full_like(f, .001))
            std.sum().backward()
            self.assertTrue(torch.isfinite(f.grad).all() and torch.isfinite(w.grad).all())

    def test_same_mean_different_spread_is_available_to_sh_mlp(self):
        neighbors = torch.zeros(2, 1, dtype=torch.long)
        w = torch.full((2, 1), .5)
        narrow = torch.ones(2, 5)
        wide = narrow + torch.tensor([[-1.], [1.]])
        mean = moment.pool_features(narrow, w, neighbors, 1)
        torch.testing.assert_close(moment.pool_features(wide, w, neighbors, 1), mean)
        std1 = moments.appearance_feature_std(narrow, w, neighbors, mean, 1)
        std2 = moments.appearance_feature_std(wide, w, neighbors, mean, 1)
        model = Decoder(cfg(appearance_moments=True, appearance_mlp_depth=2))
        # Isolate the std input path; no geometry or mean difference.
        with torch.no_grad():
            for parameter in model.appearance_sh_head.parameters():
                parameter.zero_()
            model.appearance_sh_head[0].weight[0, 5] = 1.
            model.appearance_sh_head[2].weight[0, 0] = 1.
            model.appearance_sh_head[4].weight[0, 0] = 1.
        context = torch.zeros(2, 12)
        a = model._capacity_harmonics(mean, context, None, std1)
        b = model._capacity_harmonics(mean, context, None, std2)
        self.assertGreater((a[0] - b[0]).abs().sum().item(), .1)

    def test_initial_function_and_geometry_match_final(self):
        torch.manual_seed(419)
        final = Decoder(cfg())
        torch.manual_seed(419)
        expanded = Decoder(cfg(appearance_moments=True, appearance_mlp_depth=2,
                               appearance_mlp_dim=12))
        inputs = [torch.randn(1, 12, d, requires_grad=True) for d in (3, 5, 3)]
        reference = final(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3))
        actual = expanded(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3))
        for a, b in zip(attributes(actual), attributes(reference)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        activate(expanded)
        report = {}
        changed = expanded(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3), diagnostics=report)
        for key in ('means', 'covariances', 'opacities'):
            torch.testing.assert_close(getattr(changed, key), getattr(reference, key), rtol=0, atol=0)
        self.assertGreater((changed.harmonics - reference.harmonics).abs().sum().item(), 0)
        self.assertIn('feature_std_mean', report)
        self.assertIn('feature_std_rms', report)
        self.assertTrue(all(not v.requires_grad and torch.isfinite(v) for v in report.values()))

    def compare_checkpointed(self, device, compiled=False):
        actual = Decoder(cfg(appearance_moments=True, appearance_mlp_depth=2,
                             appearance_mlp_dim=12, batch_appearance_projections=True)).to(device)
        activate(actual)
        reference = copy.deepcopy(actual)
        actual.cfg.compile_kernels = compiled
        reference.cfg.checkpoint_appearance = False
        reference.cfg.checkpoint_chunks = False
        inputs = [torch.randn(2, 12, d, device=device, requires_grad=True) for d in (3, 5, 3)]
        other = [x.detach().clone().requires_grad_() for x in inputs]
        report, expected_report = {}, {}
        with patch.object(moment, 'build_knn', wraps=moment.build_knn) as search:
            a = actual(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3), diagnostics=report)
            sum(x.square().mean() for x in attributes(a)).backward()
            self.assertEqual(search.call_count, 2)
        b = reference(other[0], other[1], rgb=other[2], image_shape=(2, 2, 3), diagnostics=expected_report)
        sum(x.square().mean() for x in attributes(b)).backward()
        for x, y in zip(attributes(a), attributes(b)):
            torch.testing.assert_close(x, y, rtol=5e-4, atol=5e-5)
        for key in report:
            torch.testing.assert_close(report[key], expected_report[key], rtol=5e-4, atol=5e-5)
        for x, y in [*zip(inputs, other), *zip(actual.parameters(), reference.parameters())]:
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-3, atol=2e-4)
        with torch.inference_mode():
            inference = actual(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3))
        for x, y in zip(attributes(a), attributes(inference)):
            torch.testing.assert_close(x, y, rtol=5e-4, atol=5e-5)

    def test_backward_checkpoint_and_inference_parity(self):
        self.compare_checkpointed('cpu')

    def test_invalid_config_and_missing_spread_fail_clearly(self):
        for options in (dict(appearance_mlp_depth=0), dict(appearance_mlp_depth=1.5),
                        dict(appearance_variance_epsilon=0.0), dict(appearance_variance_epsilon=float('nan')),
                        dict(appearance_moments=True, appearance_capacity=False)):
            with self.assertRaises(ValueError):
                invalid = cfg()
                for key, value in options.items():
                    setattr(invalid, key, value)
                Decoder(invalid)
        model = Decoder(cfg(appearance_moments=True))
        with self.assertRaisesRegex(ValueError, 'requires feature std'):
            model._capacity_harmonics(torch.zeros(2, 5), torch.zeros(2, 12), None)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for Inductor/checkpoint comparison')
    def test_cuda_compilation_and_nested_checkpoints(self):
        self.compare_checkpointed('cuda', compiled=True)


if __name__ == '__main__':
    unittest.main()
