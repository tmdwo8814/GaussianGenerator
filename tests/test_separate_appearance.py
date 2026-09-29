"""Separate appearance: independent dense references and real gradient paths."""

import ast
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from einops import rearrange

from test_moment_gaussian_decoder import Cfg, Decoder, attributes, build_knn


selection = importlib.import_module('_moment_test.encoder.common.separate_appearance')
diagnostics = importlib.import_module('_moment_test.encoder.common.moment_diagnostics')
ROOT = Path(__file__).resolve().parents[1]


class SeparateAppearanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)

    def test_incoming_normalization_matches_dense_values_and_gradients(self):
        # Slot zero has five incoming sources, although outgoing K is only two.
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0], [4, 0]])
        geometry = torch.rand(5, 2, dtype=torch.double, requires_grad=True)
        scores = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
        output = selection.appearance_weights(geometry, scores, neighbors)
        dense = geometry.new_full((5, 5), -torch.inf)  # destination, source
        for j in range(5):
            for k in range(2):
                dense[neighbors[j, k], j] = geometry[j, k].log() + scores[j, k]
        expected_dense = dense.softmax(dim=1)
        expected = expected_dense[neighbors, torch.arange(5)[:, None]]
        torch.testing.assert_close(output, expected)
        totals = torch.zeros(5, dtype=torch.double).index_add(0, neighbors.flatten(), output.flatten())
        torch.testing.assert_close(totals, torch.ones_like(totals))
        signal = torch.randn_like(output)
        actual_grad = torch.autograd.grad((output * signal).sum(), (geometry, scores))
        reference_grad = torch.autograd.grad((expected * signal).sum(), (geometry, scores))
        for actual, reference in zip(actual_grad, reference_grad):
            torch.testing.assert_close(actual, reference)
        self.assertTrue(torch.autograd.gradcheck(
            lambda g, a: selection.appearance_weights(g, a, neighbors), (geometry, scores)
        ))

    def test_zero_edges_and_extreme_scores_remain_finite(self):
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])
        geometry = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [1., 0.]], requires_grad=True)
        scores = torch.tensor([[-1000., 1000.], [1000., -1000.],
                               [-1000., 1000.], [1000., -1000.]], requires_grad=True)
        output = selection.appearance_weights(geometry, scores, neighbors)
        torch.testing.assert_close(output, geometry)
        (output * torch.randn_like(output)).sum().backward()
        self.assertTrue(torch.isfinite(geometry.grad).all())
        self.assertTrue(torch.isfinite(scores.grad).all())
        # Per-destination offsets must cancel, even with nonzero scores.
        weights = torch.rand(4, 2)
        scores = torch.randn(4, 2)
        offsets = torch.tensor([2000., -2000., 1000., -1000.])
        torch.testing.assert_close(selection.appearance_weights(weights, scores, neighbors),
                                   selection.appearance_weights(weights, scores + offsets[neighbors], neighbors),
                                   rtol=3e-4, atol=3e-5)

    def test_zero_initialized_head_preserves_existing_initialization_outputs_and_gradients(self):
        cfg = Cfg(feature_dim=7, hidden_dim=9, num_neighbors=4, chunk_size=3)
        torch.manual_seed(93)
        baseline = Decoder(copy.deepcopy(cfg), sh_degree=2)
        baseline_rng = torch.get_rng_state()
        cfg.separate_appearance = True
        torch.manual_seed(93)
        model = Decoder(cfg, sh_degree=2)
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(parameter, dict(model.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(model.appearance_head.weight.count_nonzero().item(), 0)
        points = torch.randn(2, 13, 3, requires_grad=True)
        features = torch.randn(2, 13, 7, requires_grad=True)
        bp, bf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
        report = {}
        result, expected = model(points, features, diagnostics=report), baseline(bp, bf)
        for actual, reference in zip(attributes(result), attributes(expected)):
            torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-6)
        signals = [torch.randn_like(value) for value in attributes(result)]
        sum((value * signal).sum() for value, signal in zip(attributes(result), signals)).backward()
        sum((value * signal).sum() for value, signal in zip(attributes(expected), signals)).backward()
        for actual, reference in ((points.grad, bp.grad), (features.grad, bf.grad)):
            torch.testing.assert_close(actual, reference, rtol=4e-4, atol=1e-5)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, parameter.grad,
                                       rtol=4e-4, atol=1e-5)
        self.assertGreater(model.appearance_head.weight.grad.abs().sum().item(), 0)
        self.assertLess(report['weight_tv'].item(), 1e-6)
        for value in report.values():
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value))

    def test_appearance_changes_only_harmonics_at_fixed_geometry_parameters(self):
        model = Decoder(Cfg(feature_dim=7, hidden_dim=9, num_neighbors=5, separate_appearance=True))
        points, features = torch.randn(2, 17, 3), torch.randn(2, 17, 7)
        before = model(points, features)
        torch.nn.init.normal_(model.appearance_head.weight, std=2.)
        report = {}
        after = model(points, features, diagnostics=report)
        for old, new in ((before.means, after.means), (before.covariances, after.covariances),
                         (before.opacities, after.opacities)):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertGreater((before.harmonics - after.harmonics).abs().max().item(), 1e-4)
        self.assertGreater(report['weight_tv'].item(), .01)

    def test_projected_sh_pooling_matches_dense_feature_pooling_and_backward(self):
        model = Decoder(Cfg(feature_dim=7, hidden_dim=9, num_neighbors=4,
                            chunk_size=3, separate_appearance=True))
        torch.nn.init.normal_(model.appearance_head.weight, std=.4)
        # Nonconstant coverage exercises its unchanged GEOMETRY feature path.
        torch.nn.init.normal_(model.scale_head.weight, std=.3)
        points = torch.randn(1, 11, 3, requires_grad=True)
        features = torch.randn(1, 11, 7, requires_grad=True)
        output = model(points, features)
        scale = points.detach().norm(dim=-1).median(dim=-1).values.clamp_min(model.cfg.scene_epsilon)
        normalized = points[0] / scale[0]
        neighbors = build_knn(points[0], 4)
        q, scores = model.predict_allocation(normalized, features[0], neighbors, return_appearance=True)
        mass, means, cov, pooled, wg = model.aggregate_moments(
            normalized, features[0], neighbors, q, return_weights=True
        )
        wa = selection.appearance_weights(wg, scores, neighbors)
        dense = torch.zeros(11, 11).index_put(
            (neighbors.flatten(), torch.arange(11).repeat_interleave(4)), wa.flatten(), accumulate=True
        )
        dense_harmonics = model.sh_head(dense @ features[0])
        expected = model.build_gaussians(mass, means, cov, pooled, scale[0],
                                         appearance_harmonics=dense_harmonics)
        targets = (points, features, *model.parameters())
        signals = [torch.randn_like(value) for value in attributes(output)]
        actual_loss = sum((value * signal).sum() for value, signal in zip(attributes(output), signals))
        expected_loss = sum((value * signal).sum() for value, signal in zip(attributes(expected), signals))
        for actual, reference in zip(attributes(output), attributes(expected)):
            torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-6)
        for actual, reference in zip(torch.autograd.grad(actual_loss, targets),
                                     torch.autograd.grad(expected_loss, targets)):
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, reference, rtol=5e-4, atol=1e-5)

    def test_checkpointing_and_logging_preserve_values_and_gradients(self):
        model = Decoder(Cfg(feature_dim=7, hidden_dim=9, num_neighbors=5, chunk_size=3,
                            separate_appearance=True))
        torch.nn.init.normal_(model.appearance_head.weight, std=.5)
        reference = copy.deepcopy(model)
        reference.cfg.checkpoint_chunks = False
        reference.cfg.chunk_size = 100
        points = torch.randn(2, 13, 3, requires_grad=True)
        features = torch.randn(2, 13, 7, requires_grad=True)
        rp, rf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
        actual, expected = model(points, features, diagnostics={}), reference(rp, rf)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [(points, rp), (features, rf), *zip(model.parameters(), reference.parameters())]:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=5e-6)

    def test_diagnostics_measure_incoming_slots_and_normalized_covariance(self):
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])
        wg = torch.tensor([[.25, .5], [.5, .25], [1., .25], [1., .25]])
        wa = torch.tensor([[1., .5], [.5, 0.], [1., 0.], [1., 0.]])
        stats = diagnostics.incoming_statistics(wg, wa, neighbors)
        torch.testing.assert_close(stats[0], torch.tensor([.75, 1.38629436, 0., .25, 1.]))
        mass = torch.ones(1, 4)
        # log(normalized radius) = -concentration gives Pearson -1 exactly.
        radius = torch.exp(-stats[:, 3])
        cov = torch.eye(3)[None, None] * (radius * 7).square()[None, :, None, None]
        metrics = diagnostics.summarize_statistics(stats[None], mass, cov, torch.tensor([7.]), 1e-6)
        torch.testing.assert_close(metrics['weight_tv'], torch.tensor(.75 / 4))
        torch.testing.assert_close(metrics['concentration_covariance_corr'], torch.tensor(-1.))
        torch.testing.assert_close(metrics['concentration_covariance_corr_valid'], torch.tensor(1.))
        empty = diagnostics.summarize_statistics(stats[None], mass * 0, cov, torch.tensor([7.]), 1e-6)
        self.assertTrue(all(torch.isfinite(value) for value in empty.values()))
        self.assertEqual(empty['active_slot_fraction'].item(), 0)
        self.assertEqual(empty['concentration_covariance_corr_valid'].item(), 0)

    def test_training_diagnostics_reach_logger_on_flush_steps_only(self):
        path = ROOT / 'src/model/model_wrapper.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
        namespace = {'torch': torch, 'rearrange': rearrange,
                     'compute_psnr': lambda target, image: image.flatten(1).mean(1)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        requests, logs = [], []

        class FakeEncoder:
            gaussian_decoder = SimpleNamespace(cfg=SimpleNamespace(log_every_n_steps=30))

            def __call__(self, context, step, visualization_dump=None, diagnostics_dump=None):
                requests.append(diagnostics_dump is not None)
                if diagnostics_dump is not None:
                    diagnostics_dump['weight_tv'] = torch.tensor(.25)
                return None

        batch = {'context': {'image': torch.zeros(1, 2, 3, 2, 2)},
                 'target': {'image': torch.zeros(1, 1, 3, 2, 2), 'extrinsics': None,
                            'intrinsics': None, 'near': None, 'far': None}}
        wrapper = SimpleNamespace(
            data_shim=lambda value: value, encoder=FakeEncoder(), distiller=None,
            decoder=SimpleNamespace(forward=lambda *a, **kw: SimpleNamespace(color=batch['target']['image'])),
            train_cfg=SimpleNamespace(depth_mode=None, print_log_every_n_steps=10), losses=[],
            global_rank=1, trainer=SimpleNamespace(log_every_n_steps=50), step_tracker=None,
            log=lambda *a, **kw: None, log_dict=lambda values, **kw: logs.append((values, kw)),
        )
        for step in (0, 29, 48, 49, 50):
            wrapper.global_step = step
            namespace['training_step'](wrapper, batch, 0)
        self.assertEqual(requests, [False, False, False, True, False])
        self.assertEqual(len(logs), 1)
        self.assertIn('moment/weight_tv', logs[0][0])
        self.assertTrue(logs[0][1]['sync_dist'])
        self.assertFalse(logs[0][1]['on_epoch'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the appearance device check')
    def test_cuda_incoming_selection_matches_cpu_values_and_gradients(self):
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])
        weights = torch.rand(4, 2, requires_grad=True)
        scores = torch.randn(4, 2, requires_grad=True)
        cuda_weights = weights.detach().cuda().requires_grad_()
        cuda_scores = scores.detach().cuda().requires_grad_()
        actual = selection.appearance_weights(cuda_weights, cuda_scores, neighbors.cuda())
        expected = selection.appearance_weights(weights, scores, neighbors)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
        signal = torch.randn_like(expected)
        ga = torch.autograd.grad((actual * signal.cuda()).sum(), (cuda_weights, cuda_scores))
        gb = torch.autograd.grad((expected * signal).sum(), (weights, scores))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a.cpu(), b, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
