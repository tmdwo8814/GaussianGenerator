"""CPU numerical checks and a seven-condition evaluation with a fake renderer."""

import ast
import copy
import contextlib
import importlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes

control = importlib.import_module('_moment_test.encoder.heads.gaussian_control')
neighborhood = importlib.import_module('_moment_test.encoder.common.appearance_neighborhood')
from src.evaluation.gaussian_control_report import CONDITIONS, GaussianControlReport


class GaussianControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119)
        torch.set_num_threads(1)
        self.model = Decoder(Cfg(
            feature_dim=5, hidden_dim=8, num_neighbors=3, chunk_size=4,
            appearance_2d=True, separate_appearance=True, appearance_capacity=True,
            appearance_heads=2, appearance_dim=3, appearance_mlp_dim=9,
        )).eval().requires_grad_(False)
        nn.init.normal_(self.model.appearance_head.weight, std=.3)
        nn.init.normal_(self.model.appearance_sh_head[-1].weight, std=.15)
        self.points = torch.randn(1, 12, 3)
        self.features = torch.randn(1, 12, 5)
        self.rgb = torch.rand(1, 12, 3)
        self.cache = {}
        with torch.no_grad():
            self.baseline = self.model(self.points, self.features, image_shape=(2, 2, 3),
                                       rgb=self.rgb, analysis_cache=self.cache)

    def test_export_preserves_original_forward_and_strict_checkpoint(self):
        with torch.no_grad():
            original = self.model(self.points, self.features, image_shape=(2, 2, 3), rgb=self.rgb)
        for expected, actual in zip(attributes(original), attributes(self.baseline)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        state = copy.deepcopy(self.model.state_dict())
        other = Decoder(copy.deepcopy(self.model.cfg))
        other.load_state_dict(state, strict=True)
        self.assertEqual(set(state), set(other.state_dict()))
        torch.testing.assert_close(self.cache['q'].sum(-1), self.cache['budget'])
        torch.testing.assert_close(self.cache['logits'].softmax(-1) * self.cache['budget'][:, None],
                                   self.cache['q'])
        with self.assertRaisesRegex(ValueError, 'eval'):
            self.model.train()(self.points, self.features, image_shape=(2, 2, 3),
                               rgb=self.rgb, analysis_cache={})

    def test_selection_uses_raw_mass_and_stable_ties_and_floor(self):
        mass = torch.tensor([2., 5., 5., 1., 5.])
        torch.testing.assert_close(control.select_slots(mass, .5), torch.tensor([1, 2]))
        torch.testing.assert_close(control.select_slots(mass, .01), torch.tensor([1]))
        torch.testing.assert_close(control.select_slots(mass, 1.), torch.arange(5))
        for fraction in (0, -1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                control.select_slots(mass, fraction)

    def test_pruning_gathers_all_attributes_verbatim(self):
        slots = control.select_slots(self.cache['mass'], .5)
        pruned = control.subset_gaussians(self.baseline, slots)
        for before, after in zip(attributes(self.baseline), attributes(pruned)):
            self.assertEqual(after.shape[1], 6)
            torch.testing.assert_close(after, before[:, slots], rtol=0, atol=0)

    def test_full_redistribution_matches_original_final(self):
        regenerated, mass, fallback = control.redistribute_gaussians(self.model, self.cache, torch.arange(12))
        self.assertEqual(fallback, 0.)
        torch.testing.assert_close(mass, self.cache['mass'], rtol=0, atol=0)
        for expected, actual in zip(attributes(self.baseline), attributes(regenerated)):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_all_ratios_conserve_budget_and_produce_exact_counts(self):
        for fraction in (.75, .5, .25):
            slots = control.select_slots(self.cache['mass'], fraction)
            result, mass, fallback = control.redistribute_gaussians(self.model, self.cache, slots)
            self.assertEqual(result.means.shape[1], int(fraction * 12))
            torch.testing.assert_close(mass.double().sum(), self.cache['budget'].double().sum())
            self.assertTrue(all(torch.isfinite(value).all() for value in attributes(result)))
            self.assertTrue(0 <= fallback <= 1)
            stats = control.gaussian_statistics(result, self.cache, slots, mass, fallback)
            self.assertLess(stats['budget_error_relative'], 1e-5)
            self.assertEqual(stats['retained_fraction'], fraction)

    def test_masked_softmax_recovers_survivor_even_after_original_underflow(self):
        cache = dict(self.cache)
        cache['neighbors'] = torch.stack((torch.arange(12), (torch.arange(12) + 1) % 12), -1)
        cache['logits'] = torch.tensor([[10000., -10000.]]).expand(12, -1).clone()
        neighbors, valid, q, _, active, _ = control.restricted_allocation(self.model, cache, torch.tensor([1]))
        self.assertEqual(neighbors[0, 1].item(), 1)
        self.assertEqual(q[0, 1].item(), cache['budget'][0].item())
        self.assertTrue((active[neighbors[valid]]).all())
        self.assertTrue((q[~valid] == 0).all())
        torch.testing.assert_close(q.sum(-1), cache['budget'])

    def test_orphans_route_to_nearest_open_slot_and_use_frozen_appearance_head(self):
        cache = dict(self.cache)
        cache['neighbors'] = torch.arange(12)[:, None]
        cache['logits'] = torch.zeros(12, 1)
        cache['scores'] = torch.zeros(12, 1)
        slots = torch.tensor([0, 7])
        neighbors, valid, q, scores, _, fallback = control.restricted_allocation(self.model, cache, slots)
        sources = torch.tensor([x for x in range(12) if x not in (0, 7)])
        distance = torch.cdist(cache['points'][sources], cache['points'][slots])
        expected_dst = slots[distance.argmin(-1)]
        torch.testing.assert_close(neighbors[sources, 0], expected_dst)
        self.assertAlmostEqual(float(fallback), 10 / 12, places=6)
        delta = cache['points'][sources] - cache['points'][expected_dst]
        inputs = torch.cat((cache['features'][expected_dst], cache['features'][sources],
                            delta, delta.square().sum(-1, keepdim=True)), -1)
        hidden = self.model.allocation_head[1](self.model.allocation_head[0](inputs))
        expected_scores = self.model.appearance_head(hidden).squeeze(-1)
        torch.testing.assert_close(scores[sources, 0], expected_scores, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(q.sum(-1), cache['budget'])
        self.assertTrue(valid.all())

    def test_prior_is_only_on_actual_open_self_and_closed_support_is_retained(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=8, num_neighbors=1, mass_epsilon=.25))
        points = torch.tensor([[0., 0, 0], [10., 0, 0]])
        features = torch.tensor([[1.] * 5, [3.] * 5])
        neighbors = torch.tensor([[1], [1]])  # row zero routes to a different slot
        mass, means, covariance, pooled, weights = model.aggregate_moments(
            points, features, neighbors, torch.tensor([[2.], [3.]]),
            return_weights=True, active_slots=torch.tensor([False, True]),
        )
        expected_weights = torch.tensor([2., 3.25]) / 5.25
        torch.testing.assert_close(weights[:, 0], expected_weights)
        torch.testing.assert_close(means[1], (points * expected_weights[:, None]).sum(0))
        torch.testing.assert_close(pooled[1], (features * expected_weights[:, None]).sum(0))
        self.assertEqual(mass[1].item(), 5.)  # epsilon does not change opacity mass
        delta = points - means[1]
        torch.testing.assert_close(covariance[1],
                                   (expected_weights[:, None, None] * delta[:, :, None] * delta[:, None, :]).sum(0))

    def test_appearance_filters_destinations_and_preserves_raster_metadata(self):
        image = self.cache['image']
        original_xy, original_views = image.xy.clone(), image.view_ids.clone()
        slots = torch.tensor([1, 7])
        neighbors, valid, _, _, active, _ = control.restricted_allocation(self.model, self.cache, slots)
        merged, mask = neighborhood.merge_appearance_neighbors(neighbors, image, 4, valid, active)
        self.assertTrue(active[merged[mask]].all())
        # A closed source still offers valid outgoing appearance information.
        self.assertTrue(mask[0].any())
        self.assertEqual(len(merged), 12)
        torch.testing.assert_close(image.xy, original_xy, rtol=0, atol=0)
        torch.testing.assert_close(image.view_ids, original_views, rtol=0, atol=0)

    def test_seven_conditions_encode_and_align_once_and_share_saved_cameras(self):
        from time import perf_counter
        from contextlib import contextmanager

        test = self

        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.gaussian_decoder = test.model
                self.calls = 0

            def forward(self, context, step, analysis_cache):
                self.calls += 1
                return self.gaussian_decoder(test.points, test.features, image_shape=(2, 2, 3),
                                             rgb=test.rgb, analysis_cache=analysis_cache)

        class FakeRenderer(nn.Module):
            def __init__(self):
                super().__init__()
                self.cameras, self.counts = [], []

            def forward(self, gaussians, extrinsics, intrinsics, near, far, shape):
                self.cameras.append(extrinsics.clone())
                self.counts.append(gaussians.means.shape[1])
                return SimpleNamespace(color=torch.zeros(1, 2, 3, *shape))

        class FakeWrapper(nn.Module):
            device = torch.device('cpu')

            def test_step_align(self, batch, gaussians, camera_dump):
                self.align_calls += 1
                camera_dump['extrinsics'] = batch['target']['extrinsics'] + .25

        tree = ast.parse((ROOT / 'src/eval_gaussian_control.py').read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'elapsed_ms'
                 or isinstance(node, ast.ClassDef) and node.name == 'GaussianControlModel']
        namespace = dict(torch=torch, Path=Path, contextmanager=contextmanager, perf_counter=perf_counter,
                         ModelWrapper=FakeWrapper, GaussianControlReport=GaussianControlReport,
                         FRACTIONS=(.75, .5, .25),
                         compute_psnr=lambda a, b: torch.ones(len(a)) * 25,
                         compute_ssim=lambda a, b: torch.ones(len(a)) * .8,
                         compute_lpips=lambda a, b: torch.ones(len(a)) * .2,
                         get_overlap_tag=lambda value: 'medium',
                         **{name: getattr(control, name) for name in (
                             'select_slots', 'subset_gaussians', 'redistribute_gaussians', 'gaussian_statistics')})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'control_eval', 'exec'), namespace)
        for align in (True, False):
            with self.subTest(align=align), TemporaryDirectory() as temporary:
                model = namespace['GaussianControlModel']()
                model.encoder, model.decoder = FakeEncoder(), FakeRenderer()
                model.test_cfg = SimpleNamespace(align_pose=align)
                model.trainer = SimpleNamespace(world_size=1)
                model.data_shim = lambda batch: batch
                model.analysis_directory = Path(temporary)
                model.analysis_metadata = {'camera_protocol': 'shared_final_pose' if align else 'dataset_pose',
                                           'checkpoint': 'Final.ckpt'}
                model.analysis_step, model.align_calls = 140000, 0
                extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
                batch = {'scene': ['test-scene'], 'context': {'index': torch.tensor([[4, 9]]),
                                                            'overlap': torch.tensor([.4])},
                         'target': {'image': torch.zeros(1, 2, 3, 12, 12), 'extrinsics': extrinsics,
                                    'intrinsics': torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
                                    'near': torch.ones(1, 2), 'far': torch.ones(1, 2) * 100,
                                    'index': torch.tensor([[5, 7]])}}
                with contextlib.redirect_stdout(io.StringIO()):
                    model.on_test_start()
                    model.test_step(batch, 0)
                    model.on_test_end()
                self.assertEqual(model.encoder.calls, 1)
                self.assertEqual(model.align_calls, int(align))
                self.assertEqual(model.decoder.counts, [12, 9, 9, 6, 6, 3, 3])
                expected = extrinsics + .25 if align else extrinsics
                for camera in model.decoder.cameras:
                    torch.testing.assert_close(camera, expected, rtol=0, atol=0)
                stored = torch.load(Path(temporary) / 'cameras/000000.pt', weights_only=True)
                torch.testing.assert_close(stored['extrinsics'], expected, rtol=0, atol=0)
                self.assertTrue(all((Path(temporary) / f'{name}.txt').is_file() for name in CONDITIONS))


if __name__ == '__main__':
    unittest.main()
