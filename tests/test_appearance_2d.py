"""Image candidate expansion, incoming attention, gradients and integration."""

import ast
import copy
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from einops import rearrange

from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes, build_knn, moment

neighborhood = importlib.import_module('_moment_test.encoder.common.appearance_neighborhood')
appearance = importlib.import_module('_moment_test.encoder.common.image_neighborhood_appearance')


class Appearance2DTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def test_image_edges_equal_explicit_coordinates_without_view_or_border_wrapping(self):
        for views, height, width in ((2, 9, 10), (3, 1, 7), (2, 5, 1), (1, 1, 1)):
            image = neighborhood.make_image_neighborhood((views, height, width), [1, 4], torch.device('cpu'))
            self.assertEqual(image.neighbors.shape, (views * height * width, 16))
            for j in range(len(image.neighbors)):
                v, pixel = divmod(j, height * width)
                y, x = divmod(pixel, width)
                expected = set()
                for r in (1, 4):
                    for dy in (-r, 0, r):
                        for dx in (-r, 0, r):
                            if (dy or dx) and 0 <= y + dy < height and 0 <= x + dx < width:
                                expected.add(v * height * width + (y + dy) * width + x + dx)
                actual = image.neighbors[j, image.valid[j]].tolist()
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(expected))
                self.assertTrue(all(i // (height * width) == v for i in actual))

    def test_union_masks_duplicates_and_keeps_original_geometry_order(self):
        image = neighborhood.make_image_neighborhood((2, 5, 7), [1, 4], torch.device('cpu'))
        count = len(image.neighbors)
        geom = torch.stack((torch.arange(count), torch.arange(count).roll(1), torch.arange(count).roll(2)), -1)
        indices, valid = neighborhood.merge_appearance_neighbors(geom, image, 4)
        torch.testing.assert_close(indices[:, :3], geom)
        for j in range(count):
            expected = set(geom[j].tolist()) | set(image.neighbors[j, image.valid[j]].tolist())
            actual = indices[j, valid[j]].tolist()
            self.assertEqual(set(actual), expected)
            self.assertEqual(len(actual), len(expected))

    def test_incoming_softmax_matches_dense_values_gradients_and_second_derivatives(self):
        indices = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 0, 1], [3, 0, 0]])
        valid = torch.ones_like(indices, dtype=torch.bool)
        valid[-1, -1] = False
        scores = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        result = neighborhood.incoming_softmax(scores, indices, valid)
        # Independent normalization over all incoming sources for each slot.
        flat_expected = scores.new_zeros(12)
        for i in range(4):
            positions = ((indices == i) & valid).flatten().nonzero().flatten()
            flat_expected = flat_expected.index_copy(0, positions, scores.flatten()[positions].softmax(0))
        expected = flat_expected.reshape_as(scores)
        torch.testing.assert_close(result, expected)
        totals = scores.new_zeros(4).index_add(0, indices.flatten(), result.flatten())
        torch.testing.assert_close(totals, torch.ones_like(totals))
        signal = torch.randn_like(scores)
        ga = torch.autograd.grad((result * signal).sum(), scores)[0]
        gb = torch.autograd.grad((expected * signal).sum(), scores)[0]
        torch.testing.assert_close(ga, gb)
        self.assertEqual(ga[-1, -1].item(), 0)
        fn = lambda logits: neighborhood.incoming_softmax(logits, indices, valid)
        self.assertTrue(torch.autograd.gradcheck(fn, (scores,)))
        self.assertTrue(torch.autograd.gradgradcheck(fn, (scores,)))
        extreme = scores.detach() * 10000
        extreme[-1, -1] = 1e30
        output = fn(extreme)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output[-1, -1].item(), 0)

    def test_context_matches_independent_dense_edge_formula_and_gradients(self):
        reader = appearance.ImageNeighborhoodAppearance(5, 4).double()
        image = neighborhood.make_image_neighborhood((2, 2, 3), [1, 4], torch.device('cpu'))
        n = len(image.neighbors)
        geom = torch.stack((torch.arange(n), (torch.arange(n) + 6) % n), -1)
        p, f, rgb, means, pooled = [torch.randn(n, d, dtype=torch.double, requires_grad=True)
                                   for d in (3, 5, 3, 3, 5)]
        result, _ = reader(p, f, rgb, means, pooled, geom, image,
                           chunk_size=3, checkpoint_chunks=True)
        keys, values = reader.source(torch.cat((f, rgb), -1)).chunk(2, -1)
        queries = reader.query(pooled)
        # Assemble the union from Python sets, then work destination by
        # destination. This independently checks source/destination semantics.
        scores, messages = [[] for _ in range(n)], [[] for _ in range(n)]
        for j in range(n):
            base = set(geom[j].tolist())
            union = base | set(image.neighbors[j, image.valid[j]].tolist())
            for i in sorted(union):
                same = image.view_ids[j] == image.view_ids[i]
                delta = p[j] - means[i]
                geometry = torch.cat((delta, delta.square().sum().reshape(1),
                                      (image.xy[j] - image.xy[i]) * same,
                                      delta.new_tensor([float(same), float(i not in base)])))
                scores[i].append(reader.score(F.silu(keys[j] + queries[i] + reader.score_position(geometry))).squeeze())
                messages[i].append(F.silu(values[j] + reader.value_position(geometry)))
        expected = torch.stack([(torch.stack(s).softmax(0)[:, None] * torch.stack(v)).sum(0)
                                for s, v in zip(scores, messages)])
        torch.testing.assert_close(result, expected)
        targets = (p, f, rgb, means, pooled, *reader.parameters())
        signal = torch.randn_like(result)
        ga = torch.autograd.grad((result * signal).sum(), targets)
        gb = torch.autograd.grad((expected * signal).sum(), targets)
        for actual, reference in zip(ga, gb):
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-8)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertGreater(actual.abs().sum().item(), 0)

    def test_cross_view_pixel_deltas_are_masked(self):
        image = neighborhood.make_image_neighborhood((2, 2, 3), [1, 4], torch.device('cpu'))
        p = torch.zeros(12, 3)
        indices = torch.tensor([[5, 11]])  # same-view and cross-view slots with same image xy
        geometry = appearance.ImageNeighborhoodAppearance._edge_geometry(
            p[:1], p, image.xy[:1], image.xy, image.view_ids[:1], image.view_ids,
            indices, torch.zeros(2),
        )
        self.assertGreater(geometry[0, 0, 4:6].abs().sum().item(), 0)
        torch.testing.assert_close(geometry[0, 1, 4:6], torch.zeros(2))
        torch.testing.assert_close(geometry[0, :, 6], torch.tensor([1., 0.]))

    def test_zero_initialization_preserves_original_rng_outputs_and_gradients(self):
        cfg = Cfg(feature_dim=5, hidden_dim=9, num_neighbors=3, chunk_size=4)
        torch.manual_seed(93)
        baseline = Decoder(copy.deepcopy(cfg), sh_degree=2)
        original_rng = torch.get_rng_state()
        cfg.appearance_2d = True
        torch.manual_seed(93)
        model = Decoder(cfg, sh_degree=2)
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(parameter, dict(model.named_parameters())[name], rtol=0, atol=0)
        points, features = torch.randn(2, 12, 3, requires_grad=True), torch.randn(2, 12, 5, requires_grad=True)
        bp, bf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
        report = {}
        actual = model(points, features, image_shape=(2, 2, 3), rgb=torch.rand(2, 12, 3), diagnostics=report)
        expected = baseline(bp, bf)
        signals = [torch.randn_like(v) for v in attributes(actual)]
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum((v * s).sum() for v, s in zip(attributes(actual), signals)).backward()
        sum((v * s).sum() for v, s in zip(attributes(expected), signals)).backward()
        for a, b in ((points.grad, bp.grad), (features.grad, bf.grad)):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, parameter.grad,
                                       rtol=2e-5, atol=2e-6)
        self.assertGreater(model.appearance_sh_head.weight.grad.abs().sum().item(), 0)
        self.assertEqual(report['context_sh_rms'].item(), 0)
        self.assertGreater(report['image_weight'].item(), 0)
        self.assertTrue(all(not v.requires_grad and torch.isfinite(v) for v in report.values()))

    def test_new_image_edges_influence_sh_without_changing_geometry(self):
        # K=1 deliberately excludes EVERY other support from geometry. The
        # image-neighbor path must still deliver RGB/features to slot 4.
        model = Decoder(Cfg(feature_dim=5, hidden_dim=8, num_neighbors=1, appearance_2d=True, chunk_size=3))
        torch.nn.init.normal_(model.appearance_sh_head.weight, std=.1)
        points = torch.stack((torch.arange(9) * 100., torch.zeros(9), torch.ones(9)), -1)[None]
        features, rgb = torch.randn(1, 9, 5, requires_grad=True), torch.rand(1, 9, 3, requires_grad=True)
        result = model(points, features, image_shape=(1, 3, 3), rgb=rgb)
        gf, gr = torch.autograd.grad(result.harmonics[0, 4].square().sum(), (features, rgb))
        self.assertGreater(gf[0, 0].abs().sum().item(), 0)
        self.assertGreater(gr[0, 0].abs().sum().item(), 0)
        changed = rgb.detach().clone()
        changed[0, 0] += .5
        other = model(points, features, image_shape=(1, 3, 3), rgb=changed)
        for a, b in ((result.means, other.means), (result.covariances, other.covariances),
                     (result.opacities, other.opacities)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater((result.harmonics[0, 4] - other.harmonics[0, 4]).abs().max().item(), 1e-5)

    def test_checkpoint_logging_and_geometry_match_reference_with_nonzero_head(self):
        cfg = Cfg(feature_dim=5, hidden_dim=8, num_neighbors=3, appearance_2d=True, chunk_size=4)
        model = Decoder(cfg)
        torch.nn.init.normal_(model.appearance_sh_head.weight, std=.1)
        ref = copy.deepcopy(model)
        ref.cfg.checkpoint_chunks = False
        ref.cfg.chunk_size = 1000
        p, f, rgb = [torch.randn(2, 24, d, requires_grad=True) for d in (3, 5, 3)]
        rp, rf, rr = [t.detach().clone().requires_grad_() for t in (p, f, rgb)]
        with patch.object(moment, 'build_knn', wraps=moment.build_knn) as search, \
                patch.object(moment, 'make_image_neighborhood', wraps=moment.make_image_neighborhood) as image:
            actual = model(p, f, image_shape=(3, 2, 4), rgb=rgb, diagnostics={})
            self.assertEqual(search.call_count, 2)
            self.assertEqual(image.call_count, 1)
        expected = ref(rp, rf, image_shape=(3, 2, 4), rgb=rr)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        sum(v.square().mean() for v in attributes(actual)).backward()
        sum(v.square().mean() for v in attributes(expected)).backward()
        for a, b in [(p, rp), (f, rf), (rgb, rr), *zip(model.parameters(), ref.parameters())]:
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad).all())
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=5e-6)
        base_cfg = copy.deepcopy(cfg)
        base_cfg.appearance_2d = False
        baseline = Decoder(base_cfg)
        baseline.load_state_dict({k: v for k, v in model.state_dict().items()
                                  if not k.startswith(('appearance_reader.', 'appearance_sh_head.'))})
        original = baseline(p.detach(), f.detach())
        for a, b in ((actual.means, original.means), (actual.covariances, original.covariances),
                     (actual.opacities, original.opacities)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_empty_extra_graph_single_pixel_and_metadata_validation(self):
        for radii in ([], [1, 4]):
            model = Decoder(Cfg(feature_dim=5, appearance_2d=True, appearance_2d_radii=radii))
            report = {}
            p, f, rgb = torch.randn(1, 1, 3), torch.randn(1, 1, 5), torch.rand(1, 1, 3)
            result = model(p, f, image_shape=(1, 1, 1), rgb=rgb, diagnostics=report)
            self.assertTrue(all(torch.isfinite(t).all() for t in attributes(result)))
            self.assertEqual(report['added_candidates'].item(), 0)
            self.assertEqual(report['image_weight'].item(), 0)
            self.assertEqual(report['entropy'].item(), 0)
        with patch.object(moment, 'build_knn') as search:
            with self.assertRaisesRegex(ValueError, 'image_shape'):
                model(p, f)
            with self.assertRaisesRegex(ValueError, 'image_shape'):
                model(p, f, image_shape=(1, 2, 3), rgb=rgb)
            with self.assertRaisesRegex(ValueError, 'context RGB'):
                model(p, f, image_shape=(1, 1, 1))
            search.assert_not_called()
        for radii in ([1, 1], [0], [-1], [1.5]):
            cfg = Cfg()
            cfg.appearance_2d_radii = radii
            with self.assertRaisesRegex(ValueError, 'positive integers'):
                Decoder(cfg)

    def test_edge_metrics_match_known_step_boundary_and_no_edge_case(self):
        target = torch.zeros(1, 1, 3, 2, 4)
        target[..., 2:] = 1
        prediction = target.clone()
        prediction[..., 1] = .5  # The rightward difference labels column 1.
        metrics = appearance.image_error_statistics(prediction, target)
        self.assertEqual(metrics['edge_fraction'].item(), .25)
        self.assertEqual(metrics['edge_mse'].item(), .25)
        self.assertEqual(metrics['smooth_mse'].item(), 0)
        empty = appearance.image_error_statistics(torch.ones(1, 1, 3, 1, 1), torch.zeros(1, 1, 3, 1, 1))
        self.assertEqual(empty['edge_mse'].item(), 0)
        self.assertEqual(empty['smooth_mse'].item(), 1)

    def test_training_logs_on_flush_steps_without_changing_loss(self):
        path = ROOT / 'src/model/model_wrapper.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
        namespace = {'torch': torch, 'rearrange': rearrange,
                     'compute_psnr': lambda target, image: image.flatten(1).mean(1),
                     'image_error_statistics': appearance.image_error_statistics}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        requests, logs = [], []

        class FakeEncoder:
            gaussian_decoder = SimpleNamespace(cfg=SimpleNamespace(appearance_2d=True, log_every_n_steps=30))

            def __call__(self, context, step, visualization_dump=None, diagnostics_dump=None):
                self_context = set(context)
                if self_context != {'image'}:
                    raise AssertionError('Encoder received non-context data')
                requests.append(diagnostics_dump is not None)
                if diagnostics_dump is not None:
                    diagnostics_dump['image_weight'] = torch.tensor(.25)
                return None

        batch = {'context': {'image': torch.zeros(1, 2, 3, 2, 2)},
                 'target': {'image': torch.zeros(1, 1, 3, 2, 2), 'extrinsics': None,
                            'intrinsics': None, 'near': None, 'far': None}}
        wrapper = SimpleNamespace(
            data_shim=lambda value: value, encoder=FakeEncoder(), distiller=None,
            decoder=SimpleNamespace(forward=lambda *a, **kw: SimpleNamespace(color=batch['target']['image'])),
            train_cfg=SimpleNamespace(depth_mode=None, print_log_every_n_steps=10),
            losses=[SimpleNamespace(name='test', forward=lambda *a: torch.tensor(2.))],
            global_rank=1, trainer=SimpleNamespace(log_every_n_steps=50), step_tracker=None,
            log=lambda *a, **kw: None, log_dict=lambda values, **kw: logs.append((values, kw)),
        )
        for step in (0, 29, 48, 49, 50):
            wrapper.global_step = step
            loss = namespace['training_step'](wrapper, batch, 0)
            self.assertEqual(loss.item(), 2.)
        self.assertEqual(requests, [False, False, False, True, False])
        self.assertEqual(len(logs), 1)
        self.assertIn('appearance_2d/image_weight', logs[0][0])
        self.assertIn('appearance_2d/edge_mse', logs[0][0])
        self.assertTrue(logs[0][1]['sync_dist'])
        wrapper.global_step = 49
        wrapper.encoder.gaussian_decoder.cfg.log_every_n_steps = 0
        namespace['training_step'](wrapper, batch, 0)
        self.assertFalse(requests[-1])
        self.assertEqual(len(logs), 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the appearance device check')
    def test_cuda_reader_matches_cpu_values_and_gradients(self):
        reader = appearance.ImageNeighborhoodAppearance(5, 4)
        cuda_reader = copy.deepcopy(reader).cuda()
        inputs = [torch.randn(12, d, requires_grad=True) for d in (3, 5, 3, 3, 5)]
        cuda_inputs = [t.detach().cuda().requires_grad_() for t in inputs]
        geom = torch.stack((torch.arange(12), (torch.arange(12) + 6) % 12), -1)
        grid = neighborhood.make_image_neighborhood((2, 2, 3), [1, 4], torch.device('cpu'))
        cuda_grid = neighborhood.make_image_neighborhood((2, 2, 3), [1, 4], torch.device('cuda'))
        kwargs = dict(chunk_size=3, checkpoint_chunks=True)
        result, _ = cuda_reader(*cuda_inputs, geom.cuda(), cuda_grid, **kwargs)
        expected, _ = reader(*inputs, geom, grid, **kwargs)
        torch.testing.assert_close(result.cpu(), expected, rtol=3e-5, atol=3e-6)
        signal = torch.randn_like(expected)
        ga = torch.autograd.grad((result * signal.cuda()).sum(), (*cuda_inputs, *cuda_reader.parameters()))
        gb = torch.autograd.grad((expected * signal).sum(), (*inputs, *reader.parameters()))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a.cpu(), b, rtol=5e-4, atol=5e-6)


if __name__ == '__main__':
    unittest.main()
