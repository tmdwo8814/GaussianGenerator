"""CPU tests of production auxiliary logic; real RoMa/CUDA checks need a GPU.

Run: python -m unittest discover -s tests -p test_observed_color_aux.py -v
"""

import ast
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import torch
from einops import rearrange
from jaxtyping import install_import_hook


ROOT = Path(__file__).resolve().parents[1]
package = ModuleType('_aux_test')
package.__path__ = [str(ROOT / 'src/model')]
sys.modules[package.__name__] = package
with install_import_hook(('_aux_test',), ('beartype', 'beartype')):
    aux = importlib.import_module('_aux_test.auxiliary.observed_color_loss')
    prior_module = importlib.import_module('_aux_test.auxiliary.matching_prior')
    roma = importlib.import_module('_aux_test.auxiliary.roma_matching')


def scene():
    """Nonplanar points exactly at first-image pixel centers, known metric camera."""
    height, width = 32, 40
    y, x = torch.meshgrid((torch.arange(height) + .5) / height,
                          (torch.arange(width) + .5) / width, indexing='ij')
    depth = 2.5 + .4 * torch.sin(x * 17) + .5 * y
    points = torch.stack(((x - .5) / .9 * depth, (y - .5) / 1.1 * depth, depth), -1)
    k = torch.tensor([[.9, 0, .5], [0, 1.1, .5], [0, 0, 1.]]).repeat(2, 1, 1)
    angle = torch.tensor(.06)
    rotation = torch.tensor([[angle.cos(), 0, angle.sin()], [0, 1, 0],
                             [-angle.sin(), 0, angle.cos()]])
    center = torch.tensor([.23, .015, .01])
    camera_points = (points.reshape(-1, 3) - center) @ rotation
    uv_b = camera_points @ k[1].T
    uv_b = uv_b[:, :2] / uv_b[:, 2:]
    uv_a = torch.stack((x, y), -1).reshape(-1, 2)
    valid = ((uv_a > .1) & (uv_a < .9) & (uv_b > .1) & (uv_b < .9)).all(-1)
    ids = torch.nonzero(valid)[:, 0][::3]
    matches = roma.Correspondences(2 * uv_a[ids] - 1, 2 * uv_b[ids] - 1, torch.ones(len(ids)))
    cameras = torch.eye(4).repeat(2, 1, 1)
    cameras[1, :3, :3], cameras[1, :3, 3] = rotation, center
    return points, k, matches, cameras


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        cv2.setRNGSeed(11)
        torch.set_num_threads(1)

    def test_pose_recovers_rotation_and_canonical_scale_despite_outliers(self):
        points, k, matches, cameras = scene()
        noisy = copy.deepcopy(matches)
        noisy.grid_b[::5] = torch.rand_like(noisy.grid_b[::5]) * 1.5 - .75
        # At this tiny resolution, use a subpixel RANSAC threshold for precise
        # geometry checks (production defaults are for 256px context images).
        result = prior_module.estimate_matching_prior(noisy, k, points,
                    prior_module.MatchingPriorCfg(ransac_threshold_px=.05))
        self.assertIsNotNone(result)
        torch.testing.assert_close(result.cameras, cameras, atol=2e-3, rtol=2e-3)
        self.assertAlmostEqual(result.scale, cameras[1, :3, 3].norm().item(), places=3)
        self.assertLess(len(result.confidence), len(matches.confidence))
        self.assertLess(result.alignment_error, .002)

    def test_pose_scale_tracks_detached_raw_support_scale(self):
        points, k, matches, cameras = scene()
        points = (points * 3).requires_grad_()
        result = prior_module.estimate_matching_prior(matches, k, points,
                    prior_module.MatchingPriorCfg(ransac_threshold_px=.05))
        self.assertIsNotNone(result)
        torch.testing.assert_close(result.cameras[1, :3, 3], cameras[1, :3, 3] * 3, atol=2e-3, rtol=2e-3)
        self.assertFalse(result.cameras.requires_grad)

    def test_bad_pairs_are_rejected(self):
        points, k, matches, _ = scene()
        cfg = prior_module.MatchingPriorCfg()
        empty = roma.Correspondences(matches.grid_a[:4], matches.grid_b[:4], matches.confidence[:4])
        self.assertIsNone(prior_module.estimate_matching_prior(empty, k, points, cfg))
        self.assertIsNone(prior_module.estimate_matching_prior(matches, k, -points, cfg))
        stationary = roma.Correspondences(matches.grid_a, matches.grid_a.clone(), matches.confidence)
        self.assertIsNone(prior_module.estimate_matching_prior(stationary, k, points, cfg))

    def test_observed_colors_use_current_centers_and_have_no_gradient(self):
        image = torch.arange(3 * 4 * 6).reshape(3, 4, 6).float().requires_grad_()
        k = torch.eye(3)
        # Exact pixel centers; moving a slot changes its observed color next forward.
        means = torch.tensor([[1.5 / 6, 2.5 / 4, 1], [3.5 / 6, .5 / 4, 1],
                              [0, 0, -1.]]).requires_grad_()
        colors, valid = aux.observed_colors(means, torch.eye(4), k, image)
        torch.testing.assert_close(colors, torch.stack((image[:, 2, 1], image[:, 0, 3])))
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertFalse(colors.requires_grad)

    def test_aux_gradients_reach_geometry_and_opacity_but_not_color_or_prior(self):
        points, k, matches, _ = scene()
        height, width = points.shape[:2]
        means = points.reshape(1, -1, 3).repeat(1, 2, 1).requires_grad_()
        cov = (torch.eye(3).reshape(1, 1, 3, 3).repeat(1, means.shape[1], 1, 1) * .01).requires_grad_()
        opacity = torch.full(means.shape[:2], .6, requires_grad=True)
        harmonics = torch.randn(1, means.shape[1], 3, 4, requires_grad=True)
        gaussians = aux.Gaussians(means, cov, harmonics, opacity)
        images = torch.full((1, 2, 3, height, width), .8, requires_grad=True)
        supports = points[None, None].repeat(1, 2, 1, 1, 1).requires_grad_()
        renderer_calls = []

        def differentiable_renderer(**kwargs):
            renderer_calls.append(kwargs)
            self.assertFalse(kwargs['gaussian_sh_coefficients'].requires_grad)
            self.assertFalse(kwargs['extrinsics'].requires_grad)
            self.assertFalse(kwargs['use_sh'])
            self.assertLessEqual(kwargs['gaussian_means'].shape[1], height * width)
            color = (kwargs['gaussian_means'].mean() * .01
                     + kwargs['gaussian_covariances'].mean()
                     + kwargs['gaussian_opacities'].mean() * .1)
            return color.expand(1, 3, height, width), torch.zeros(1, height, width)

        helper = aux.ObservedColorAuxiliary(aux.ObservedColorAuxCfg(enabled=True, warm_up_steps=0, ramp_steps=0))
        loss, stats = helper.compute(gaussians, images, k[None], supports,
                                     [aux.PreparedPair(0, matches)], 1, torch.zeros(3), differentiable_renderer)
        self.assertEqual(stats['directions'], 2)
        self.assertEqual(stats['valid_pairs'], 1)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        for value in (means, cov, opacity):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for value in (harmonics, images, supports):
            self.assertIsNone(value.grad)
        self.assertEqual(len(renderer_calls), 2)

        # A rejected pair does not invoke rendering and contributes finite zero.
        rejected = roma.Correspondences(matches.grid_a[:1], matches.grid_b[:1], matches.confidence[:1])
        loss, stats = helper.compute(gaussians, images, k[None], supports,
                                     [aux.PreparedPair(0, rejected)], 1, torch.zeros(3), differentiable_renderer)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(stats['valid_pairs'], 0)
        self.assertEqual(len(renderer_calls), 2)

    def test_schedule_rotation_and_no_checkpoint_registration(self):
        helper = aux.ObservedColorAuxiliary(aux.ObservedColorAuxCfg(
            enabled=True, warm_up_steps=2, ramp_steps=2, every_n_steps=2))
        images = torch.rand(3, 2, 3, 8, 8)
        empty = roma.Correspondences(torch.empty(0, 2), torch.empty(0, 2), torch.empty(0))
        with patch.object(helper.matcher, 'match', return_value=empty) as match:
            for step in (0, 1, 2, 3):
                self.assertEqual(helper.prepare_pairs(images, step), [])
            match.assert_not_called()
            self.assertEqual(helper.prepare_pairs(images, 4)[0].batch_index, 2)
            self.assertEqual(helper.prepare_pairs(images, 6)[0].batch_index, 0)
        self.assertAlmostEqual(helper.cfg.weight_at(3), .025)
        self.assertAlmostEqual(helper.cfg.weight_at(4), .05)
        model = torch.nn.Module()
        model.auxiliary = helper
        helper.matcher._model = torch.nn.Linear(2, 3)
        self.assertEqual(len(model.state_dict()), 0)
        helper.release()
        self.assertIsNone(helper.matcher._model)

    def test_matcher_uses_bchw_handles_inference_tensors_and_zero_confidence(self):
        class FakeRoMa:
            def __init__(self):
                self.zero = False
                self.sample_count = None

            @torch.inference_mode()
            def match(self, a, b):
                assert a.shape == b.shape == (1, 3, 8, 8)
                return {'warp_AB': torch.zeros(1, 8, 8, 2),
                        'overlap_AB': torch.full((1, 8, 8, 1), 0. if self.zero else .9)}

            @torch.inference_mode()
            def sample(self, predictions, count):
                self.sample_count = count
                coordinates = torch.zeros(count, 4)
                confidence = torch.ones(count)
                coordinates[0] = 1  # Outside pixel-center bounds.
                confidence[1] = .1
                return coordinates, confidence, None, None

        matcher = roma.RoMaMatcher(roma.RoMaMatcherCfg(num_matches=2048))
        matcher._model, matcher._device = FakeRoMa(), torch.device('cpu')
        result = matcher.match(torch.rand(2, 3, 8, 8))
        self.assertEqual(matcher._model.sample_count, 16)  # 4*N <= positive candidates.
        self.assertEqual(len(result.confidence), 14)
        self.assertFalse(torch.is_inference(result.grid_a))
        parameter = torch.ones_like(result.grid_a, requires_grad=True)
        (parameter * result.grid_a).sum().backward()
        matcher._model.zero = True
        self.assertEqual(len(matcher.match(torch.rand(2, 3, 8, 8)).confidence), 0)

    def test_aux_config_inherits_decoder_and_baseline_loss(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            baseline = compose(config_name='main', overrides=['+experiment=re10k_moment'])
            config = compose(config_name='main', overrides=['+experiment=re10k_moment_aux'])
        cfg = from_dict(aux.ObservedColorAuxCfg, OmegaConf.to_container(config.train.auxiliary))
        self.assertTrue(cfg.enabled)
        self.assertIsInstance(cfg.roma, roma.RoMaMatcherCfg)
        for key in ('model', 'dataset', 'loss', 'optimizer', 'trainer', 'test', 'data_loader'):
            self.assertEqual(OmegaConf.to_container(config[key]), OmegaConf.to_container(baseline[key]))

    def test_training_step_matches_raw_context_before_encoder_and_uses_no_gt_pose(self):
        path = ROOT / 'src/model/model_wrapper.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelWrapper')
        method = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == 'training_step')
        namespace = {'torch': torch, 'rearrange': rearrange, 'BatchedExample': dict,
                     'compute_psnr': lambda *args: torch.ones(1)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        events = []
        raw = torch.full((1, 2, 3, 2, 2), .75)
        supports = torch.zeros(1, 2, 2, 2, 3)
        predicted = SimpleNamespace(means=torch.ones(1, 8, 3))

        class Helper:
            def prepare_pairs(self, images, step):
                torch.testing.assert_close(images, raw)
                events.append('match')
                return ['pair']

            def compute(self, gaussians, images, intrinsics, points, pairs, step, background):
                torch.testing.assert_close(images, raw)
                assert points is supports
                events.append('aux')
                return torch.tensor(.2), {'valid_pairs': 1.}

        def shim(batch):
            batch['context']['image'] = (batch['context']['image'] - .5) / .5
            return batch

        def encoder(context, step, visualization_dump):
            events.append('encoder')
            torch.testing.assert_close(context['image'], torch.full_like(raw, .5))
            visualization_dump['support_points'] = supports
            return predicted

        model = SimpleNamespace(
            auxiliary=Helper(), data_shim=shim, encoder=encoder, global_step=3,
            decoder=SimpleNamespace(background_color=torch.zeros(3),
                                    forward=lambda *args, **kwargs: SimpleNamespace(color=torch.ones(1, 1, 3, 2, 2))),
            train_cfg=SimpleNamespace(depth_mode=None, print_log_every_n_steps=100),
            losses=[SimpleNamespace(name='main', forward=lambda *args: torch.tensor(1.))],
            log=lambda *args: None, distiller=None, global_rank=1, step_tracker=None,
        )
        # Context GT extrinsics/near/far are deliberately absent.
        batch = {'context': {'image': raw.clone(), 'intrinsics': torch.eye(3).repeat(1, 2, 1, 1)},
                 'target': {'image': torch.ones(1, 1, 3, 2, 2), 'extrinsics': None,
                            'intrinsics': None, 'near': None, 'far': None}}
        result = namespace['training_step'](model, batch, 0)
        self.assertAlmostEqual(result.item(), 1.2)
        self.assertEqual(events, ['match', 'encoder', 'aux'])


if __name__ == '__main__':
    unittest.main()
