"""CPU tests of registration, gradient isolation, fallback, and teacher wiring."""

import ast
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from test_moment_gaussian_decoder import Cfg, Decoder, attributes, ROOT

package = ModuleType('_moment_test.auxiliary')
package.__path__ = [str(ROOT / 'src/model/auxiliary')]
sys.modules[package.__name__] = package
cv = importlib.import_module('_moment_test.encoder.common.cross_view_verifier')
teacher = importlib.import_module('_moment_test.auxiliary.descriptor_teacher')
roma = importlib.import_module('_moment_test.auxiliary.roma_matching')
knn = importlib.import_module('_moment_test.encoder.common.sparse_knn')


class CrossViewTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.cfg = cv.CrossViewVerifierCfg(enabled=True)
        self.source = torch.randn(120, 3)
        self.rotation = torch.tensor([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
        self.translation = torch.tensor([5., -7., 3.])
        self.target = self.source @ self.rotation.T + self.translation

    def verify(self, source=None, target=None):
        source = self.source if source is None else source
        target = self.target if target is None else target
        return cv.verify_rigid(source, target, torch.ones(len(source)), torch.tensor(.08), self.cfg)

    def test_large_rotation_translation_with_outliers(self):
        target = self.target.clone()
        target[:25] = torch.randn(25, 3) * 10
        rotation, translation, stats = self.verify(target=target)
        self.assertEqual(stats[0].item(), 1)
        torch.testing.assert_close(rotation, self.rotation, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(translation, self.translation)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.))

    def test_held_out_correspondences_can_reject_an_exact_training_fit(self):
        generator = torch.Generator().manual_seed(1729)
        validation = torch.randperm(len(self.source), generator=generator)[::4]
        target = self.target.clone()
        target[validation] += 50
        rotation, translation, stats = self.verify(target=target)
        self.assertEqual(stats[0].item(), 0)
        torch.testing.assert_close(rotation, torch.eye(3))
        torch.testing.assert_close(translation, torch.zeros(3))

    def test_planar_allowed_collinear_and_scale_mismatch_rejected(self):
        planar = self.source.clone()
        planar[:, 2] = 0
        self.assertEqual(self.verify(planar, planar @ self.rotation.T + self.translation)[2][0], 1)
        line = planar.clone()
        line[:, 1] = 0
        self.assertEqual(self.verify(line, line @ self.rotation.T + self.translation)[2][0], 0)
        self.assertEqual(self.verify(target=self.source * 3 + self.translation)[2][0], 0)
        self.assertEqual(self.verify(torch.zeros(120, 3), torch.zeros(120, 3))[2][0], 0)

    def test_empty_or_insufficient_matches_return_identity(self):
        for n in (0, 3, 23):
            rotation, translation, stats = self.verify(self.source[:n], self.target[:n])
            self.assertEqual(stats[0], 0)
            torch.testing.assert_close(rotation, torch.eye(3))
            torch.testing.assert_close(translation, torch.zeros(3))

    def test_fit_does_not_change_global_rng(self):
        before = torch.random.get_rng_state()
        self.verify()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_mutual_matches_find_permutation_and_reject_ambiguous_descriptors(self):
        a = torch.eye(32)
        order = torch.randperm(32)
        ia, ib, score = cv.mutual_matches(a, a[order], self.cfg)
        torch.testing.assert_close(ia, order[ib])
        self.assertEqual(len(ia), 32)
        self.assertTrue((score == 1).all())
        self.assertEqual(len(cv.mutual_matches(torch.ones(32, 4), torch.ones(32, 4), self.cfg)[0]), 0)

    def test_spacing_is_independent_of_origin_and_scales_with_scene(self):
        spacing = cv.representative_spacing(self.source)
        torch.testing.assert_close(cv.representative_spacing(self.source + 10), spacing, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(cv.representative_spacing(self.source * 3), spacing * 3)

    def test_pixel_sampling_spans_both_axes_without_duplicate_indices(self):
        for height, width, budget in ((256, 256, 1024), (32, 128, 512), (2, 2, 1024)):
            ids = cv.pixel_representatives(height, width, budget, 'cpu')
            self.assertEqual(len(ids), len(ids.unique()))
            self.assertLessEqual(len(ids), min(budget, height * width))
            self.assertTrue(((ids >= 0) & (ids < height * width)).all())
            self.assertGreater(len((ids // width).unique()), 1)

    def test_forward_aligns_only_second_view_and_keeps_point_gradient(self):
        n = len(self.source)
        student = cv.CrossViewVerifier(n, self.cfg)
        # Isolate geometry from training quality with exact descriptor IDs.
        student.describe = lambda f: F.normalize(f, dim=-1)
        points = torch.cat((self.target, self.source))[None].requires_grad_()
        features = torch.cat((torch.eye(n), torch.eye(n)))[None].requires_grad_()
        corrected, fused, stats = student(points, features, n, torch.arange(n), torch.arange(n))
        self.assertEqual(fused, [True])
        torch.testing.assert_close(corrected[0, :n], self.target)
        torch.testing.assert_close(corrected[0, n:], self.target, atol=2e-5, rtol=2e-5)
        corrected.square().mean().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(points.grad[:, n:].abs().sum(), 0)
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is None for p in student.parameters()))

    def test_failed_forward_retains_original_points_and_requests_partition(self):
        student = cv.CrossViewVerifier(5, self.cfg)
        points, features = torch.randn(2, 40, 3), torch.ones(2, 40, 5)
        result, fused, _ = student(points, features, 20, torch.arange(20), torch.arange(20))
        self.assertEqual(fused, [False, False])
        torch.testing.assert_close(result, points, rtol=0, atol=0)

    def test_partitioned_knn_never_crosses_views_and_self_stays_unique(self):
        points = torch.cat((self.source[:30], self.source[:30] + .0001))
        indices = knn.build_partitioned_knn(points, (30, 30), 16)
        torch.testing.assert_close(indices[:, 0], torch.arange(60))
        self.assertTrue((indices[:30] < 30).all())
        self.assertTrue((indices[30:] >= 30).all())
        for row in indices:
            self.assertEqual(len(row), len(row.unique()))
        tiny = knn.build_partitioned_knn(points[:5], (2, 3), 16)
        self.assertEqual(tiny.shape, (5, 2))

    def test_decoder_default_is_unchanged_and_fallback_gradients_are_finite(self):
        decoder = Decoder(Cfg(feature_dim=5, hidden_dim=8, num_neighbors=4, chunk_size=20))
        points = torch.randn(2, 20, 3, requires_grad=True)
        features = torch.randn(2, 20, 5, requires_grad=True)
        for a, b in zip(attributes(decoder(points, features)), attributes(decoder(points, features, [None, None]))):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        mixed = decoder(points, features, [None, (10, 10)])
        sum(x.square().mean() for x in attributes(mixed)).backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertTrue(torch.isfinite(features.grad).all())


class TeacherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)
        self.student = cv.CrossViewVerifier(8, cv.CrossViewVerifierCfg())
        self.cfg = teacher.DescriptorTeacherCfg(min_pairs=2, exclusion_pixels=.1)
        axis = (torch.arange(4) + .5) / 2 - 1
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack((xx, yy), -1).reshape(-1, 2)
        self.pairs = roma.Correspondences(grid, grid.clone(), torch.ones(16))

    def test_supervision_updates_only_descriptor_and_not_input_features(self):
        a, b = torch.randn(16, 8, requires_grad=True), torch.randn(16, 8, requires_grad=True)
        loss, metrics = teacher.descriptor_objective(self.student, a, b, self.pairs, (4, 4), self.cfg)
        self.assertEqual(metrics['pairs'], 16)
        loss.backward()
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.student.parameters()))
        self.assertGreater(sum(p.grad.abs().sum() for p in self.student.parameters()), 0)

    def test_missing_pairs_connect_all_student_parameters_with_zero_gradient(self):
        a = torch.randn(16, 8, requires_grad=True)
        loss, _ = teacher.descriptor_objective(self.student, a, a, None, (4, 4), self.cfg)
        self.assertEqual(loss, 0)
        loss.backward()
        self.assertIsNone(a.grad)
        self.assertTrue(all(p.grad is not None and (p.grad == 0).all() for p in self.student.parameters()))

    def test_duplicate_teacher_pairs_do_not_become_false_negatives(self):
        pairs = roma.Correspondences(torch.zeros(16, 2), torch.zeros(16, 2), torch.ones(16))
        a = torch.randn(16, 8)
        loss, metrics = teacher.descriptor_objective(self.student, a, a, pairs, (4, 4), self.cfg)
        self.assertEqual(loss, 0)
        self.assertEqual(metrics['pairs'], 0)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.student.parameters()))

    def test_feature_sampling_matches_pixel_centers(self):
        features = torch.arange(16 * 8).float().reshape(16, 8)
        result = teacher.sample_features(features, self.pairs.grid_a, 4, 4)
        torch.testing.assert_close(result, features)

    def test_teacher_rotates_one_scene_and_skips_scheduled_steps(self):
        owner = teacher.DescriptorTeacher(teacher.DescriptorTeacherCfg(every_n_steps=2))
        rgb = torch.rand(4, 2, 3, 4, 4)
        with patch.object(owner.matcher, 'match', return_value=self.pairs) as matcher:
            self.assertIsNone(owner.prepare(rgb, 1))
            matcher.assert_not_called()
            scene, pairs = owner.prepare(rgb, 4)
            self.assertEqual(scene, 2)
            matcher.assert_called_once()
            torch.testing.assert_close(matcher.call_args.args[0], rgb[2])

    def test_teacher_is_lazy_and_not_part_of_model_state(self):
        owner = teacher.DescriptorTeacher(self.cfg)
        self.assertIsNone(owner.matcher._model)
        self.assertNotIsInstance(owner, nn.Module)
        model = nn.Module()
        model.student, model.teacher = self.student, owner
        self.assertTrue(all(key.startswith('student.') for key in model.state_dict()))

    def test_precision_guard_restores_backbone_setting_even_after_failure(self):
        original = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('high')
            with self.assertRaises(RuntimeError), roma.matching_precision():
                self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                raise RuntimeError('test')
            self.assertEqual(torch.get_float32_matmul_precision(), 'high')
        finally:
            torch.set_float32_matmul_precision(original)

    def test_yaml_deserializes_to_actual_dataclasses(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            cfg = compose(config_name='main', overrides=['+experiment=re10k_moment'])
        self.assertTrue(from_dict(cv.CrossViewVerifierCfg, OmegaConf.to_container(cfg.model.encoder.cross_view_verifier)).enabled)
        self.assertTrue(from_dict(teacher.DescriptorTeacherCfg, OmegaConf.to_container(cfg.train.descriptor_teacher)).enabled)

    def test_real_training_step_uses_raw_context_and_isolates_teacher_gradient(self):
        from einops import rearrange
        path = ROOT / 'src/model/model_wrapper.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
        method = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
        namespace = {'torch': torch, 'rearrange': rearrange,
                     'compute_psnr': lambda a, b: torch.zeros(len(a))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)

        class FrontEnd(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.cross_view_verifier = self.student
                inner.features = nn.Parameter(torch.randn(2, 32, 8))

            def forward(inner, context, step, visualization_dump):
                torch.testing.assert_close(context['image'], raw * 2 - 1)
                visualization_dump['cv_features'] = inner.features.detach()
                visualization_dump['cv_stats'] = {'accepted': torch.tensor(0.)}
                return inner.features

        model = nn.Module()
        model.encoder = FrontEnd()
        model.descriptor_teacher = teacher.DescriptorTeacher(self.cfg)
        model.distiller = None
        model.global_step, model.global_rank = 1, 0
        model.train_cfg = SimpleNamespace(depth_mode=None, print_log_every_n_steps=100,
                                          descriptor_teacher=self.cfg)
        model.step_tracker = None
        model.log = lambda *a, **kw: None
        def shim(batch):
            batch['context']['image'] = batch['context']['image'] * 2 - 1
            return batch
        model.data_shim = shim
        model.decoder = SimpleNamespace(forward=lambda features, *args, **kwargs:
            SimpleNamespace(color=features.mean().expand(2, 1, 3, 4, 4)))
        # A differentiable zero photometric loss makes accidental teacher
        # gradients to the front end observable, while exercising the real loop.
        model.losses = [SimpleNamespace(name='photo', forward=lambda out, *args: out.color.sum() * 0)]
        raw = torch.rand(2, 2, 3, 4, 4)
        target = {'image': torch.rand(2, 1, 3, 4, 4),
                  'extrinsics': None, 'intrinsics': None, 'near': None, 'far': None}
        batch = {'context': {'image': raw.clone()}, 'target': target}
        with patch.object(model.descriptor_teacher.matcher, 'match', return_value=self.pairs) as matcher:
            loss = namespace['training_step'](model, batch, 0)
        torch.testing.assert_close(matcher.call_args.args[0], raw[1])
        loss.backward()
        self.assertEqual(model.encoder.features.grad.abs().sum(), 0)
        self.assertGreater(sum(p.grad.abs().sum() for p in self.student.parameters()), 0)

    def test_real_encoder_wires_rejected_registration_to_partitioned_knn(self):
        from einops import rearrange
        from typing import Optional
        moment = importlib.import_module('_moment_test.encoder.heads.moment_gaussian_decoder')
        path = ROOT / 'src/model/encoder/encoder_noposplat.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EncoderNoPoSplat')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        namespace = {'torch': torch, 'Optional': Optional, 'Gaussians': moment.Gaussians,
                     'rearrange': rearrange, 'pixel_representatives': cv.pixel_representatives}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)

        class FrontEnd(nn.Module):
            def forward(self, context, return_views):
                shape = torch.tensor([[4, 4]])
                return [], [], shape, shape, {'img': context['image'][:, 0]}, {'img': context['image'][:, 1]}

        class FeatureHead(nn.Module):
            def forward(self, *args):
                return torch.ones(1, 8, 4, 4)

        encoder = nn.Module()
        encoder.forward = MethodType(namespace['forward'], encoder)
        encoder.gs_params_head_type = 'moment'
        encoder.backbone = FrontEnd()
        encoder._downstream_head = lambda *a: {'pts3d': torch.randn(1, 4, 4, 3)}
        encoder.gaussian_param_head, encoder.gaussian_param_head2 = FeatureHead(), FeatureHead()
        encoder.cross_view_verifier = self.student
        encoder.gaussian_decoder = Decoder(Cfg(feature_dim=8, num_neighbors=4))
        dump = {}
        with patch.object(moment, 'build_partitioned_knn', wraps=knn.build_partitioned_knn) as grouped:
            result = encoder({'image': torch.rand(1, 2, 3, 4, 4)}, visualization_dump=dump)
        grouped.assert_called_once()
        self.assertEqual(grouped.call_args.args[1], (16, 16))
        self.assertEqual(dump['cv_stats']['accepted'], 0)
        self.assertFalse(dump['cv_features'].requires_grad)
        self.assertEqual(result.means.shape, (1, 32, 3))


if __name__ == '__main__':
    unittest.main()
