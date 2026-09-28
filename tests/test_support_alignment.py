"""CPU geometry, sampling, teacher, gradient and integration checks; no RoMa download."""

import ast
import importlib
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from einops import rearrange

from test_moment_gaussian_decoder import ROOT, feature_head, moment

alignment = importlib.import_module('_moment_test.encoder.common.support_alignment')
geometry = importlib.import_module('_moment_test.encoder.common.weighted_sim3')
package = ModuleType('_moment_test.auxiliary')
package.__path__ = [str(ROOT / 'src/model/auxiliary')]
sys.modules[package.__name__] = package
teacher = importlib.import_module('_moment_test.auxiliary.descriptor_teacher')


def identity_field(height=32, width=32, confidence=1.):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    xy = torch.stack((2 * (x + .5) / width - 1, 2 * (y + .5) / height - 1))
    return teacher.TeacherField(0, torch.cat((xy, torch.full((1, height, width), confidence)))[None])


def small_student():
    cfg = alignment.SupportAlignmentCfg(source_grid=2, target_grid=4, extra_cells=2,
                                        fine_window=8, descriptor_dim=8, hidden_dim=12)
    return alignment.SupportAlignment(cfg, 7)


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        torch.set_num_threads(1)

    def test_recovers_scale_rotation_translation_and_transforms_unmatched_points(self):
        source = torch.randn(2, 40, 3, dtype=torch.float64)
        rotation = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
        scale = torch.tensor([2.4, .3], dtype=torch.float64)
        translation = torch.tensor([[3., -2, 1], [-4, 1, .5]], dtype=torch.float64)
        target = scale[:, None, None] * (source @ rotation.T) + translation[:, None]
        fit = geometry.robust_similarity(source, target, torch.ones(2, 40, dtype=torch.float64))
        self.assertTrue(fit.valid.all())
        torch.testing.assert_close(fit.scale, scale)
        torch.testing.assert_close(fit.rotation, rotation.expand(2, -1, -1))
        torch.testing.assert_close(fit.translation, translation)
        unseen = torch.randn(2, 17, 3, dtype=torch.float64)
        torch.testing.assert_close(fit.apply(unseen), scale[:, None, None] * (unseen @ rotation.T) + translation[:, None])

    def test_planar_valid_collinear_empty_and_nan_pairs_safe(self):
        xy = torch.randn(1, 20, 3)
        xy[..., -1] = 0
        self.assertTrue(geometry.fit_similarity(xy, xy + 2, torch.ones(1, 20)).valid.all())
        line = torch.zeros_like(xy)
        line[..., 0] = torch.arange(20)
        for source, target, weights in [(line, line, torch.ones(1, 20)),
                                        (xy, xy, torch.zeros(1, 20)),
                                        (xy * 0, xy * 0, torch.ones(1, 20))]:
            weights.requires_grad_()
            fit = geometry.robust_similarity(source, target, weights)
            self.assertFalse(fit.valid.any())
            torch.testing.assert_close(fit.apply(source), source)
            fit.apply(source).sum().backward()
            self.assertTrue(weights.grad.isfinite().all())
        target = xy + 2
        xy[:, 0] = float('nan')
        fit = geometry.fit_similarity(xy, target, torch.ones(1, 20))
        self.assertTrue(fit.valid.all())
        torch.testing.assert_close(fit.translation, torch.full((1, 3), 2.))

    def test_no_reflection_and_rotation_gradient_matches_finite_differences(self):
        source = torch.randn(1, 11, 3, dtype=torch.float64)
        target = source * source.new_tensor([-1, 1, 1])
        weights = torch.rand(1, 11, dtype=torch.float64) + .4
        fit = geometry.fit_similarity(source, target, weights)
        torch.testing.assert_close(torch.linalg.det(fit.rotation), torch.ones(1, dtype=torch.float64))
        target = torch.randn_like(source)
        weights.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda weight: geometry.fit_similarity(source, target, weight).apply(source),
            (weights,), atol=2e-5, rtol=2e-4))

    def test_isotropic_repeated_singular_values_have_finite_gradients(self):
        source = torch.cat((torch.eye(3), -torch.eye(3)))[None]
        weights = torch.ones(1, 6, requires_grad=True)
        fit = geometry.fit_similarity(source, source, weights)
        (fit.apply(source) * torch.randn_like(source)).sum().backward()
        self.assertTrue(weights.grad.isfinite().all())

    def test_low_weight_outliers_do_not_dominate_fit(self):
        source = torch.randn(1, 50, 3)
        target = source * 1.8 + 3
        target[:, -5:] += 100
        weights = torch.ones(1, 50)
        weights[:, -5:] = 1e-4
        fit = geometry.robust_similarity(source, target, weights)
        self.assertLess((fit.apply(source[:, :-5]) - target[:, :-5]).abs().max().item(), .002)


class MatcherTeacherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_sampling_deterministic_unique_quarters_and_original_indices(self):
        model = small_student()
        features = torch.randn(2, 2, 256, 7)
        first = model.match(features, (16, 16))
        torch.rand(200)  # Global RNG does not affect selection.
        second = model.match(features, (16, 16))
        torch.testing.assert_close(first.queries, second.queries)
        torch.testing.assert_close(first.targets, second.targets)
        self.assertEqual(first.queries.shape, (2, 12))
        self.assertEqual(first.queries[0, :4].tolist(), [68, 76, 196, 204])
        for row in first.queries:
            self.assertEqual(row.unique().numel(), len(row))
        self.assertTrue(((first.targets >= 0) & (first.targets < 256)).all())
        self.assertTrue((first.fine_indices == first.targets[..., None]).any(-1).all())
        replay = model.match(features, (16, 16), first.queries)
        torch.testing.assert_close(replay.targets, first.targets)
        torch.testing.assert_close(replay.confidence, first.confidence)

    def test_target_dedup_keeps_strongest_and_first_tie(self):
        match = SimpleNamespace(targets=torch.tensor([[3, 3, 2, 2, 7]]),
                                confidence=torch.tensor([[.1, .9, .5, .5, .3]], requires_grad=True))
        result = alignment.unique_weights(match, 8)
        torch.testing.assert_close(result, torch.tensor([[0., .9, .5, 0., .3]]))
        result.sum().backward()
        torch.testing.assert_close(match.confidence.grad, torch.tensor([[0., 1, 1, 0, 1]]))

    def test_teacher_bilinear_half_pixel_mapping_and_invalid_labels(self):
        indices = torch.tensor([[0, 8, 35, 255]])
        xy, target, confidence, inside = teacher.teacher_at(identity_field(), indices, (16, 16))
        expected = torch.stack((indices % 16, indices // 16), -1).float()
        torch.testing.assert_close(xy, expected)
        torch.testing.assert_close(target, indices)
        self.assertTrue(inside.all())
        self.assertTrue((confidence == 1).all())
        field = identity_field()
        field.values[:, 0] = 4  # Outside image; no border pseudo-correspondence.
        _, _, confidence, inside = teacher.teacher_at(field, indices, (16, 16))
        self.assertFalse(inside.any())
        self.assertEqual(confidence.sum().item(), 0)

    def test_teacher_loss_trains_both_heads_but_never_frontend(self):
        model = small_student()
        features = torch.randn(1, 2, 256, 7, requires_grad=True)
        points = torch.randn(1, 2, 256, 3, requires_grad=True)
        dump = {'log': True}
        model(points, features, (16, 16), dump)
        loss, metrics = teacher.teacher_objective(model, dump, identity_field(), teacher.DescriptorTeacherCfg())
        self.assertTrue(loss.isfinite())
        loss.backward()
        self.assertIsNone(points.grad)
        self.assertIsNone(features.grad)
        for group in (model.descriptor, model.confidence_head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in group.parameters()), 0)
            self.assertTrue(all(p.grad.isfinite().all() for p in group.parameters()))
        self.assertGreater(metrics['valid_pairs'].item(), 0)
        self.assertIn('audit_residual_after', metrics)

    def test_no_teacher_pairs_is_finite_and_metrics_are_missing_not_zero(self):
        model = small_student()
        dump = {'log': True}
        model(torch.randn(1, 2, 256, 3), torch.randn(1, 2, 256, 7), (16, 16), dump)
        loss, metrics = teacher.teacher_objective(model, dump, identity_field(confidence=0.), teacher.DescriptorTeacherCfg())
        loss.backward()
        self.assertTrue(loss.isfinite())
        self.assertEqual(metrics['audit_measured_pairs'].item(), 0)
        self.assertTrue(metrics['audit_residual_after'].isnan())

    def test_main_path_detached_fit_live_geometry_and_warmup(self):
        model = small_student()
        points = torch.randn(1, 2, 256, 3, requires_grad=True)
        features = torch.randn(1, 2, 256, 7, requires_grad=True)
        fixed = geometry.Similarity(torch.tensor([2.]), torch.eye(3)[None], torch.tensor([[1., 2, 3]]), torch.tensor([True]))
        with patch.object(model, 'estimate', return_value=(fixed, points[:, 1, :12], points[:, 0, :12], torch.ones(1, 12))):
            output = model(points, features, (16, 16))
            torch.testing.assert_close(output[:, 0], points[:, 0])
            torch.testing.assert_close(output[:, 1], points[:, 1] * 2 + fixed.translation[:, None])
            output.sum().backward()
            torch.testing.assert_close(points.grad[:, 0], torch.ones_like(points.grad[:, 0]))
            torch.testing.assert_close(points.grad[:, 1], 2 * torch.ones_like(points.grad[:, 1]))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertIsNone(features.grad)
            torch.testing.assert_close(model(points, features, (16, 16), {'apply': False}), points)

    def test_teacher_direction_schedule_lazy_loading_and_precision_restore(self):
        owner = teacher.DescriptorTeacher(teacher.DescriptorTeacherCfg(every_n_steps=2))
        self.assertIsNone(owner.model)
        rgb = torch.rand(2, 2, 3, 16, 16)
        def match(source, reference):
            # RoMa checks dimension 1 for RGB channels before adding any batch
            # dimension. A permissive CHW mock used to hide the integration bug.
            self.assertEqual(source.shape, (1, 3, 16, 16))
            self.assertEqual(reference.shape, (1, 3, 16, 16))
            torch.testing.assert_close(source[0], rgb[1, 1])
            torch.testing.assert_close(reference[0], rgb[1, 0])
            field = identity_field().values.permute(0, 2, 3, 1)
            return {'warp_AB': field[..., :2], 'overlap_AB': field[..., 2:]}
        owner.model = SimpleNamespace(match=match)
        precision = torch.get_float32_matmul_precision()
        self.assertIsNone(owner.prepare(rgb, 1))
        field = owner.prepare(rgb, 2)
        self.assertEqual(field.scene, 1)
        self.assertEqual(torch.get_float32_matmul_precision(), precision)
        self.assertEqual(field.values.shape, (1, 3, 32, 32))

    def test_teacher_accepts_real_romav2_tensor_loader_without_model_weights(self):
        # Exercise the cloned API's real input checks without importing/loading
        # its heavy network. Optional when the third-party checkout is absent.
        path = ROOT / 'RoMaV2/src/romav2/romav2.py'
        if not path.exists():
            self.skipTest('RoMaV2 checkout is required for its input-loader check')
        import numpy as np
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RoMaV2')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_load_image')
        namespace = {'torch': torch, 'Path': Path, 'np': np, 'ImageLike': object,
                     'Image': SimpleNamespace(Image=type('UnusedPILImage', (), {})),
                     'device': torch.device('cpu')}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        loader = namespace['_load_image']
        for batch_size, scene in ((1, 0), (3, 2)):
            with self.subTest(batch_size=batch_size):
                owner = teacher.DescriptorTeacher(teacher.DescriptorTeacherCfg())
                rgb = torch.rand(batch_size, 2, 3, 16, 16)
                def match(source, reference):
                    source, reference = loader(None, source), loader(None, reference)
                    self.assertEqual(source.shape, (1, 3, 16, 16))
                    self.assertEqual(reference.shape, (1, 3, 16, 16))
                    torch.testing.assert_close(source[0], rgb[scene, 1])
                    torch.testing.assert_close(reference[0], rgb[scene, 0])
                    field = identity_field().values.permute(0, 2, 3, 1)
                    return {'warp_AB': field[..., :2], 'overlap_AB': field[..., 2:]}
                owner.model = SimpleNamespace(match=match)
                self.assertEqual(owner.prepare(rgb, scene).scene, scene)

    def test_default_256_channel_256_image_forward_and_support_count(self):
        model = alignment.SupportAlignment(alignment.SupportAlignmentCfg(), 256)
        features = torch.randn(1, 2, 256 * 256, 256)
        points = torch.randn(1, 2, 256 * 256, 3)
        dump = {'log': True}
        with torch.no_grad():
            output = model(points, features, (256, 256), dump)
        self.assertEqual(output.shape, points.shape)
        torch.testing.assert_close(output[:, 0], points[:, 0])
        self.assertEqual(dump['queries'].shape, (1, 512))
        self.assertTrue(output.isfinite().all())

    def test_full_width_dpt_is_identity_and_alignment_config_composes(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        head = feature_head.create_dpt_feature_head(SimpleNamespace(dec_depth=12, enc_embed_dim=16, dec_embed_dim=16), 256)
        self.assertIsInstance(head.dpt.head, nn.Identity)
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            cfg = compose(config_name='main', overrides=['+experiment=re10k_moment'])
        alignment_cfg = from_dict(alignment.SupportAlignmentCfg, OmegaConf.to_container(cfg.model.encoder.support_alignment))
        teacher_cfg = from_dict(teacher.DescriptorTeacherCfg, OmegaConf.to_container(cfg.train.descriptor_teacher))
        self.assertTrue(alignment_cfg.enabled and teacher_cfg.enabled)
        self.assertEqual(cfg.model.encoder.moment_decoder.feature_dim, 256)


class WrapperIntegrationTests(unittest.TestCase):
    def test_training_wires_raw_context_supervision_logging_and_live_frontend(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        path = ROOT / 'src/model/model_wrapper.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelWrapper')
        names = {'training_step', '_alignment_timer_start', '_alignment_timer_end', '_log_alignment_metrics', 'configure_optimizers'}
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {'torch': torch, 'dist': torch.distributed, 'rearrange': rearrange,
                     'compute_psnr': lambda actual, predicted: (actual - predicted).square().mean((1, 2, 3)),
                     'get_cfg': lambda: {'trainer': {'max_steps': 100}}}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 7)
                self.points = nn.Parameter(torch.randn(1, 2, 256, 3))
                self.support_alignment = small_student()

            def forward(self, context, step, visualization_dump=None, alignment_dump=None):
                image = context['image']
                feature = self.backbone(image.flatten(3).transpose(2, 3))
                points = self.points.expand(len(image), -1, -1, -1)
                aligned = self.support_alignment(points, feature, (16, 16), alignment_dump)
                color = (aligned.mean((1, 2, 3)) + feature.mean((1, 2, 3))).sigmoid()
                return SimpleNamespace(means=aligned.flatten(1, 2), color=color)

        class Teacher(teacher.DescriptorTeacher):
            def prepare(self, rgb, step):
                torch.testing.assert_close(rgb, raw_context)
                return identity_field()

        wrapper = nn.Module()
        for name in names:
            method = namespace[name]
            setattr(wrapper, name, method.__func__ if isinstance(method, staticmethod) else MethodType(method, wrapper))
        wrapper.encoder = Encoder()
        wrapper.descriptor_teacher = Teacher(teacher.DescriptorTeacherCfg(log_every_n_steps=1, warmup_steps=1))
        wrapper.train_cfg = SimpleNamespace(descriptor_teacher=wrapper.descriptor_teacher.cfg,
                                           depth_mode=None, print_log_every_n_steps=10)
        wrapper.data_shim = lambda batch: {**batch, 'context': {'image': batch['context']['image'] * 2 - 1}}
        wrapper.decoder = SimpleNamespace(forward=lambda gs, *args, **kwargs:
                                          SimpleNamespace(color=gs.color[:, None, None, None, None].expand(-1, 1, 3, 16, 16)))
        wrapper.losses = [SimpleNamespace(name='mse', forward=lambda output, batch, *args:
                                         (output.color - batch['target']['image']).square().mean())]
        wrapper.global_step, wrapper.global_rank = 0, 1
        wrapper.distiller, wrapper.step_tracker = None, None
        wrapper._alignment_timings, wrapper._alignment_completed_timings = {}, {}
        wrapper._alignment_wall_count, wrapper._alignment_wall_sum = 0, 0.
        logged = {}
        wrapper.log = lambda name, value, **kwargs: logged.update({name: value.detach() if isinstance(value, torch.Tensor) else value})
        wrapper.log_dict = lambda values, **kwargs: logged.update(values)
        raw_context = torch.rand(2, 2, 3, 16, 16)
        batch = {'context': {'image': raw_context},
                 'target': {'image': torch.rand(2, 1, 3, 16, 16), 'extrinsics': None,
                            'intrinsics': None, 'near': None, 'far': None}}
        for step in (0, 1):
            wrapper.zero_grad(set_to_none=True)
            wrapper.global_step = step
            loss = wrapper.training_step(batch, step)
            loss.backward()
            self.assertTrue(loss.isfinite())
            self.assertGreater(wrapper.encoder.backbone.weight.grad.abs().sum().item(), 0)
            self.assertGreater(wrapper.encoder.points.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all()
                                for p in wrapper.encoder.support_alignment.parameters()))
            self.assertEqual(logged['align/warmup'].item(), float(step == 0))
            self.assertIn('align_teacher/audit_residual_after', logged)
            self.assertIn('align/identity_fallback_fraction', logged)
        wrapper.optimizer_cfg = SimpleNamespace(lr=1e-4, backbone_lr_multiplier=.1, warm_up_steps=2)
        optimizer = wrapper.configure_optimizers()['optimizer']
        full_rate = {id(p) for p in optimizer.param_groups[0]['params']}
        self.assertTrue(all(id(p) in full_rate for p in wrapper.encoder.support_alignment.parameters()))

    def test_packed_logging_excludes_missing_rank_measurements(self):
        path = ROOT / 'src/model/model_wrapper.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelWrapper')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_log_alignment_metrics')
        calls = []
        def reduce(packed):
            calls.append(packed.clone())
            # Another rank has a measured residual=4, unlike this rank's NaN.
            packed[0, 0] += 4
            packed[1, 0] += 1
        namespace = {'torch': torch, 'dist': SimpleNamespace(is_available=lambda: True,
                      is_initialized=lambda: True, all_reduce=reduce)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        logged = {}
        wrapper = SimpleNamespace(_alignment_wall_count=0, _alignment_wall_sum=0.,
                                  log_dict=lambda values, **kwargs: logged.update(values))
        namespace['_log_alignment_metrics'](wrapper, {'align/residual': torch.tensor(float('nan'))})
        self.assertEqual(len(calls), 1)
        self.assertEqual(logged['align/residual'].item(), 4)


if __name__ == '__main__':
    unittest.main()
