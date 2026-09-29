"""Allocation refinement: dense references, initialization, gradients and logs."""

import ast
import copy
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from einops import rearrange

from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes, build_knn, moment

refinement = importlib.import_module('_moment_test.encoder.common.group_aware_refinement')
diagnostics = importlib.import_module('_moment_test.encoder.common.refinement_diagnostics')


def dense_groups(points, appearance, neighbors, q, epsilon):
    count, k = neighbors.shape
    dense = q.new_zeros(count, count).index_put(
        (neighbors.flatten(), torch.arange(count).repeat_interleave(k)),
        q.flatten(), accumulate=True,
    )
    mass = dense.sum(-1)
    incoming = dense + torch.eye(count, dtype=q.dtype) * epsilon
    incoming = incoming / incoming.sum(-1, keepdim=True)
    means, colors = incoming @ points, incoming @ appearance
    variance = (incoming[..., None] * (appearance[None] - colors[:, None]).square()).sum((1, 2))
    return mass, means, colors, variance


class GroupAwareRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        # Slot zero receives FOUR supports even though each source has K=2.
        self.neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])

    def test_incoming_groups_match_dense_values_and_gradients(self):
        points = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        appearance = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        q = torch.rand(4, 2, dtype=torch.double, requires_grad=True)
        args = (points, appearance, q)

        def gather(p, a, allocation):
            group = refinement.gather_slot_groups(p, a, self.neighbors, allocation, 1e-6, 2)
            return group.mass, group.means, group.appearance, group.disagreement

        actual = gather(*args)
        expected = dense_groups(points, appearance, self.neighbors, q, 1e-6)
        signals = [torch.randn_like(value) for value in actual]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        ga = torch.autograd.grad(sum((v * s).sum() for v, s in zip(actual, signals)), args)
        gb = torch.autograd.grad(sum((v * s).sum() for v, s in zip(expected, signals)), args)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b)
        self.assertTrue(torch.autograd.gradcheck(gather, args))

    def test_refinement_matches_dense_reference_and_conserves_budget(self):
        refiner = refinement.GroupAwareRefinement(9).double()
        torch.nn.init.normal_(refiner.score[-1].weight, std=.3)
        points = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        appearance = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        logits = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
        budget = torch.rand(4, 1, dtype=torch.double, requires_grad=True)
        q = logits.softmax(-1) * budget
        actual, stats = refiner(points, appearance, self.neighbors, q, logits, budget,
                               epsilon=1e-6, chunk_size=2, checkpoint_chunks=True,
                               collect_statistics=True)
        mass, means, colors, variance = dense_groups(points, appearance, self.neighbors, q, 1e-6)
        # Deliberately evaluate the formula edge by edge, independently of the
        # chunked production implementation and its sparse group pooling.
        update = []
        for j in range(4):
            row = []
            for k, i in enumerate(self.neighbors[j]):
                values = torch.cat((logits[j, k:k+1], points[j] - means[i],
                                    (appearance[j] - colors[i]).square().sum().reshape(1),
                                    variance[i:i+1], mass[i:i+1].log1p()))
                row.append(refiner.score(values).squeeze(-1))
            update.append(torch.stack(row))
        expected = budget * (logits + torch.stack(update)).softmax(-1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1, keepdim=True), budget)
        signal = torch.randn_like(actual)
        targets = (points, appearance, logits, budget, *refiner.parameters())
        ga = torch.autograd.grad((actual * signal).sum(), targets, retain_graph=True)
        gb = torch.autograd.grad((expected * signal).sum(), targets)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b)
            self.assertTrue(torch.isfinite(a).all())
            self.assertGreater(a.abs().sum().item(), 0)
        self.assertFalse(stats.requires_grad)

    def test_other_incoming_support_changes_a_fixed_pair_score(self):
        refiner = refinement.GroupAwareRefinement(1).double()
        with torch.no_grad():
            refiner.score[0].weight.zero_()
            refiner.score[0].bias.zero_()
            refiner.score[0].weight[0, 4] = 1  # SH distance to the gathered group.
            refiner.score[-1].weight.fill_(-1)
        points = torch.randn(4, 3, dtype=torch.double)
        appearance = torch.zeros(4, 3, dtype=torch.double, requires_grad=True)
        logits = torch.zeros(4, 2, dtype=torch.double)
        budget = torch.ones(4, 1, dtype=torch.double)
        q = logits.softmax(-1) * budget
        kwargs = dict(epsilon=1e-6, chunk_size=2, checkpoint_chunks=False)
        before, _ = refiner(points, appearance, self.neighbors, q, logits, budget, **kwargs)
        changed = appearance.detach().clone()
        changed[2] = 4  # Support 2 sends to slot 0, but is NOT a candidate of source 0.
        changed.requires_grad_()
        after, _ = refiner(points, changed, self.neighbors, q, logits, budget, **kwargs)
        self.assertGreater((before[0] - after[0]).abs().max().item(), .01)
        after[0, 0].backward()
        self.assertGreater(changed.grad[2].abs().sum().item(), 0)

    def test_zero_initialization_preserves_base_rng_outputs_and_gradients(self):
        cfg = Cfg(feature_dim=7, hidden_dim=9, num_neighbors=4, chunk_size=3)
        torch.manual_seed(93)
        baseline = Decoder(copy.deepcopy(cfg), sh_degree=2)
        baseline_rng = torch.get_rng_state()
        cfg.group_aware_refine = True
        torch.manual_seed(93)
        model = Decoder(cfg, sh_degree=2)
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(parameter, dict(model.named_parameters())[name], rtol=0, atol=0)
        points = torch.randn(2, 13, 3, requires_grad=True)
        features = torch.randn(2, 13, 7, requires_grad=True)
        bp, bf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
        report = {}
        actual, expected = model(points, features, diagnostics=report), baseline(bp, bf)
        signals = [torch.randn_like(value) for value in attributes(actual)]
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        sum((v * s).sum() for v, s in zip(attributes(actual), signals)).backward()
        sum((v * s).sum() for v, s in zip(attributes(expected), signals)).backward()
        for a, b in ((points.grad, bp.grad), (features.grad, bf.grad)):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=1e-5)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, parameter.grad,
                                       rtol=5e-4, atol=1e-5)
        self.assertGreater(model.group_refiner.score[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(report['allocation_tv'].item(), 0)
        self.assertEqual(report['logit_update_centered_rms'].item(), 0)
        torch.testing.assert_close(report['appearance_variance_before'], report['appearance_variance_after'])
        for value in report.values():
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value))

    def test_checkpoint_logging_and_one_knn_per_scene(self):
        model = Decoder(Cfg(feature_dim=7, hidden_dim=9, num_neighbors=5, chunk_size=3,
                            group_aware_refine=True))
        torch.nn.init.normal_(model.group_refiner.score[-1].weight, std=.3)
        reference = copy.deepcopy(model)
        reference.cfg.checkpoint_chunks = False
        reference.cfg.chunk_size = 100
        points = torch.randn(2, 13, 3, requires_grad=True)
        features = torch.randn(2, 13, 7, requires_grad=True)
        rp, rf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
        report = {}
        with patch.object(moment, 'build_knn', wraps=moment.build_knn) as search:
            actual = model(points, features, diagnostics=report)
            self.assertEqual(search.call_count, len(points))
        expected = reference(rp, rf)
        self.assertEqual(actual.means.shape, points.shape)
        self.assertGreater(report['allocation_tv'].item(), 0)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        sum(v.square().mean() for v in attributes(actual)).backward()
        sum(v.square().mean() for v in attributes(expected)).backward()
        for a, b in [(points, rp), (features, rf), *zip(model.parameters(), reference.parameters())]:
            self.assertTrue(torch.isfinite(a.grad).all())
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=5e-6)

    def test_final_attributes_all_use_the_same_refined_allocation(self):
        model = Decoder(Cfg(feature_dim=7, hidden_dim=9, num_neighbors=4, group_aware_refine=True))
        torch.nn.init.normal_(model.group_refiner.score[-1].weight, std=.5)
        points, features = torch.randn(1, 11, 3), torch.randn(1, 11, 7)
        result = model(points, features)
        scale = points.norm(dim=-1).median(dim=-1).values
        normalized = points[0] / scale[0]
        neighbors = build_knn(points[0], 4)
        q, logits, budget = model.predict_allocation(normalized, features[0], neighbors, return_logits=True)
        sh = (model.sh_head(features[0]).reshape(-1, 3, model.d_sh) * model.sh_mask).flatten(1)
        q, _ = model.group_refiner(normalized, sh, neighbors, q, logits, budget,
                                  epsilon=model.cfg.mass_epsilon, chunk_size=3, checkpoint_chunks=False)
        # Dense reference for ALL final moments, independent of aggregate_moments.
        mass, means, pooled, _ = dense_groups(normalized, features[0], neighbors, q, model.cfg.mass_epsilon)
        dense = q.new_zeros(11, 11).index_put(
            (neighbors.flatten(), torch.arange(11).repeat_interleave(4)), q.flatten(), accumulate=True,
        ) + model.cfg.mass_epsilon * torch.eye(11)
        weights = dense / dense.sum(-1, keepdim=True)
        delta = normalized[None] - means[:, None]
        covariance = torch.einsum('ij,ijc,ijd->icd', weights, delta, delta)
        expected = model.build_gaussians(mass, means, covariance, pooled, scale)
        for a, b in zip(attributes(result), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(mass.sum(), budget.sum())

    def test_empty_groups_and_single_support_are_finite(self):
        points, appearance = torch.randn(4, 3), torch.randn(4, 6)
        group = refinement.gather_slot_groups(points, appearance, self.neighbors,
                                             torch.zeros(4, 2), 1e-6, 2)
        torch.testing.assert_close(group.means, points)
        torch.testing.assert_close(group.appearance, appearance)
        torch.testing.assert_close(group.mass, torch.zeros(4))
        torch.testing.assert_close(group.disagreement, torch.zeros(4), atol=2e-6, rtol=0)
        refiner = refinement.GroupAwareRefinement(3)
        q, stats = refiner(points, appearance, self.neighbors, torch.zeros(4, 2),
                          torch.zeros(4, 2), torch.zeros(4, 1), epsilon=1e-6,
                          chunk_size=2, checkpoint_chunks=False, collect_statistics=True)
        self.assertEqual(q.count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(stats).all())
        model = Decoder(Cfg(feature_dim=7, group_aware_refine=True))
        report = {}
        points, features = torch.randn(1, 1, 3), torch.randn(1, 1, 7)
        result = model(points, features, diagnostics=report)
        torch.testing.assert_close(result.means, points)
        self.assertTrue(all(torch.isfinite(v).all() for v in attributes(result)))
        self.assertEqual(report['allocation_tv'].item(), 0)
        self.assertEqual(report['concentration_covariance_corr_valid'].item(), 0)

    def test_diagnostics_correlation_and_no_active_slots(self):
        stats = torch.zeros(1, 4, 8)
        stats[..., 0] = .25
        stats[..., 1] = .04
        stats[..., 3] = torch.tensor([.25, .5, .75, 1.])
        radius = torch.exp(-stats[..., 3])
        covariance = torch.eye(3) * (radius * 7).square()[..., None, None]
        args = (stats, torch.ones(1, 4), covariance, torch.tensor([7.]), 1e-6)
        report = diagnostics.summarize_refinement(*args)
        self.assertEqual(report['allocation_tv'].item(), .25)
        torch.testing.assert_close(report['logit_update_centered_rms'], torch.tensor(.2))
        torch.testing.assert_close(report['concentration_covariance_corr'], torch.tensor(-1.))
        self.assertEqual(report['concentration_covariance_corr_valid'].item(), 1)
        empty = diagnostics.summarize_refinement(stats, torch.zeros(1, 4), covariance, torch.tensor([7.]), 1e-6)
        self.assertTrue(all(torch.isfinite(v) for v in empty.values()))
        self.assertEqual(empty['active_slot_fraction'].item(), 0)
        self.assertEqual(empty['concentration_covariance_corr_valid'].item(), 0)

    def test_training_logs_on_flush_steps_and_can_disable_diagnostics(self):
        path = ROOT / 'src/model/model_wrapper.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
        namespace = {'torch': torch, 'rearrange': rearrange,
                     'compute_psnr': lambda target, image: image.flatten(1).mean(1)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        requests, logs = [], []

        class FakeEncoder:
            gaussian_decoder = SimpleNamespace(cfg=SimpleNamespace(group_aware_refine=True, log_every_n_steps=30))

            def __call__(self, context, step, visualization_dump=None, diagnostics_dump=None):
                requests.append(diagnostics_dump is not None)
                if diagnostics_dump is not None:
                    diagnostics_dump['allocation_tv'] = torch.tensor(.25)
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
        self.assertIn('refine/allocation_tv', logs[0][0])
        self.assertTrue(logs[0][1]['sync_dist'])
        self.assertFalse(logs[0][1]['on_epoch'])
        wrapper.global_step = 49
        wrapper.encoder.gaussian_decoder.cfg.log_every_n_steps = 0
        namespace['training_step'](wrapper, batch, 0)
        self.assertFalse(requests[-1])
        self.assertEqual(len(logs), 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the refinement device check')
    def test_cuda_matches_cpu_values_and_gradients(self):
        cpu = refinement.GroupAwareRefinement(9)
        torch.nn.init.normal_(cpu.score[-1].weight, std=.2)
        gpu = copy.deepcopy(cpu).cuda()
        p, a = torch.randn(4, 3, requires_grad=True), torch.randn(4, 6, requires_grad=True)
        logits, budget = torch.randn(4, 2, requires_grad=True), torch.rand(4, 1, requires_grad=True)
        gp, ga, gl, gb = [v.detach().cuda().requires_grad_() for v in (p, a, logits, budget)]
        kwargs = dict(epsilon=1e-6, chunk_size=2, checkpoint_chunks=True)
        actual, _ = gpu(gp, ga, self.neighbors.cuda(), gl.softmax(-1) * gb, gl, gb, **kwargs)
        expected, _ = cpu(p, a, self.neighbors, logits.softmax(-1) * budget, logits, budget, **kwargs)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
        signal = torch.randn_like(expected)
        x = torch.autograd.grad((actual * signal.cuda()).sum(), (gp, ga, gl, gb, *gpu.parameters()))
        y = torch.autograd.grad((expected * signal).sum(), (p, a, logits, budget, *cpu.parameters()))
        for v, w in zip(x, y):
            torch.testing.assert_close(v.cpu(), w, rtol=3e-4, atol=3e-6)


if __name__ == '__main__':
    unittest.main()
