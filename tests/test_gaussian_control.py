"""CPU numerical checks; no checkpoint, dataset, CUDA or backbone import needed.

python -m unittest discover -s tests -p test_gaussian_control.py -v
"""

from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch

from test_moment_gaussian_decoder import Cfg, Decoder, attributes

# Use the real lightweight decoder modules, without encoder/__init__'s renderer
# and MASt3R imports. No production functions are replaced.
for name in ('encoder', 'encoder.common', 'encoder.heads'):
    sys.modules['src.model.' + name] = sys.modules['_moment_test.' + name]

from src.gaussian_control.common import (
    gaussian_dict, select_gaussians, target_count, uniform_pixel_indices,
)
from src.gaussian_control.redistribute import redistribute, redistribute_edges
from src.gaussian_control.nanogs import merge, to_nanogs
from src.eval_gaussian_control import (
    build_cache, cache_manifest, evaluate, load_encoder_weights, scene_key, save_json, save_torch,
    summarize, validate_scene, write_reports,
)
from src.third_party.nanogs.utils.merge import moment_matching
from src.third_party.nanogs.utils.splat_utils import sigma_from_scale_quat_batch


class GaussianControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(1)
        self.decoder = Decoder(Cfg(
            feature_dim=5, hidden_dim=7, num_neighbors=4, chunk_size=5,
            appearance_2d=True, separate_appearance=True, appearance_capacity=True,
            appearance_heads=2, appearance_dim=3, appearance_mlp_dim=9,
            appearance_mlp_depth=2, appearance_moments=True,
        ), sh_degree=2).eval()
        # Exercise trained/nonuniform scores, SH residuals and coverage.
        for parameter in self.decoder.parameters():
            torch.nn.init.normal_(parameter, std=.2)
        self.points = torch.randn(1, 32, 3)
        self.features = torch.randn(1, 32, 5)
        self.rgb = torch.rand(1, 32, 3)
        self.cache = {}
        with torch.no_grad():
            self.gaussians = self.decoder(
                self.points, self.features, image_shape=(2, 4, 4), rgb=self.rgb,
                control_cache=self.cache,
            )

    def test_cache_capture_preserves_the_entire_final_plus_prediction(self):
        with torch.no_grad():
            reference = self.decoder(self.points, self.features, image_shape=(2, 4, 4), rgb=self.rgb)
        for a, b in zip(attributes(self.gaussians), attributes(reference)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        q = self.cache['logits'].softmax(-1) * self.cache['budget']
        mass = torch.zeros(32).index_add(0, self.cache['neighbors'].flatten(), q.flatten())
        torch.testing.assert_close(mass, self.cache['mass'])

    def test_cache_is_detached_and_requires_eval_no_grad(self):
        self.assertTrue(all(not v.requires_grad for v in self.cache.values() if torch.is_tensor(v)))
        with self.assertRaises(ValueError):
            self.decoder(self.points, self.features, image_shape=(2, 4, 4), rgb=self.rgb, control_cache={})

    def test_full_open_redistribution_recovers_original_gaussians(self):
        actual, detail = redistribute(self.decoder, self.cache, 32)
        self.assertEqual(detail['fallback_supports'], 0)
        for a, b in zip(attributes(actual), attributes(self.gaussians)):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

    def test_all_budgets_produce_exact_finite_positive_covariances(self):
        for percent in (100, 75, 50, 25, 10, 5, 1):
            count = target_count(32, percent)
            value, _ = redistribute(self.decoder, self.cache, count)
            self.assertEqual(value.means.shape, (1, count, 3))
            self.assertTrue(all(torch.isfinite(v).all() for v in attributes(value)))
            self.assertTrue((torch.linalg.eigvalsh(value.covariances) > 0).all())
            self.assertTrue(((value.opacities >= 0) & (value.opacities <= 1)).all())

    def test_reallocation_matches_masked_original_logits_and_conserves_budget(self):
        keep, opened, neighbors, q, _, _ = redistribute_edges(self.decoder, self.cache, 5)
        torch.testing.assert_close(q.sum(-1, keepdim=True), self.cache['budget'])
        self.assertTrue((q[~opened[neighbors]] == 0).all())
        original_valid = opened[self.cache['neighbors']]
        connected = original_valid.any(-1)
        expected = self.cache['logits'][connected].masked_fill(~original_valid[connected], -torch.inf).softmax(-1)
        torch.testing.assert_close(q[connected] / self.cache['budget'][connected], expected)
        disconnected = ~connected
        distance = torch.cdist(self.cache['points'][disconnected], self.cache['points'][keep])
        torch.testing.assert_close(neighbors[disconnected, 0], keep[distance.argmin(-1)])
        torch.testing.assert_close(q[disconnected, 0], self.cache['budget'][disconnected, 0])

    def test_stable_mass_ties_and_no_fake_self_prior_on_closed_sources(self):
        cache = dict(self.cache, mass=torch.ones(32))
        keep, opened, neighbors, q, _, _ = redistribute_edges(self.decoder, cache, 1)
        self.assertEqual(keep.tolist(), [0])
        mass, means, covariance, pooled, weights = self.decoder.aggregate_moments(
            cache['points'], cache['features'], neighbors, q,
            return_weights=True, destination_mask=opened,
        )
        w = cache['budget'][:, 0].clone()
        w[0] += self.decoder.cfg.mass_epsilon
        w /= w.sum()
        expected_mean = (w[:, None] * cache['points']).sum(0)
        delta = cache['points'] - expected_mean
        expected_cov = torch.einsum('n,ni,nj->ij', w, delta, delta)
        torch.testing.assert_close(means[0], expected_mean)
        torch.testing.assert_close(covariance[0], expected_cov, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(pooled[0], (w[:, None] * cache['features']).sum(0))
        self.assertTrue((weights[~opened[neighbors]] == 0).all())

    def test_closed_slots_receive_no_3d_or_image_appearance_weights(self):
        keep, opened, neighbors, q, _, _ = redistribute_edges(self.decoder, self.cache, 3)
        _, means, _, pooled = self.decoder.aggregate_moments(
            self.cache['points'], self.cache['features'], neighbors, q, destination_mask=opened,
        )
        context, _ = self.decoder.appearance_reader(
            self.cache['points'], self.cache['features'], self.cache['rgb'], means, pooled, neighbors,
            self.decoder._image_neighbors((2, 4, 4), torch.device('cpu')),
            chunk_size=5, checkpoint_chunks=False, destination_mask=opened,
        )
        self.assertTrue((context[~opened] == 0).all())
        self.assertTrue(torch.isfinite(context).all())

    def test_uniform_grid_exact_unique_per_view_and_two_dimensional(self):
        for shape in ((2, 7, 11), (3, 1, 9), (2, 9, 1)):
            total = int(np.prod(shape))
            for count in range(1, total + 1):
                indices = uniform_pixel_indices(shape, count)
                self.assertEqual(len(indices.unique()), count)
                self.assertTrue(((indices >= 0) & (indices < total)).all())
                quotas = torch.bincount(indices // (shape[1] * shape[2]), minlength=shape[0])
                self.assertLessEqual(int(quotas.max() - quotas.min()), 1)
        ids = uniform_pixel_indices((1, 100, 100), 100)
        self.assertEqual(len((ids // 100).unique()), 10)
        self.assertEqual(len((ids % 100).unique()), 10)

    def test_prune_keeps_attributes_unchanged(self):
        ids = uniform_pixel_indices((2, 4, 4), 7)
        result = select_gaussians(self.gaussians, ids)
        for key, value in gaussian_dict(result).items():
            torch.testing.assert_close(value, gaussian_dict(self.gaussians)[key][:, ids], rtol=0, atol=0)

    def test_nanogs_parameter_adapter_preserves_full_covariance_and_all_sh(self):
        mu, scales, quat, op, sh = to_nanogs(self.gaussians)
        np.testing.assert_allclose(sigma_from_scale_quat_batch(scales, quat),
                                   self.gaussians.covariances[0].numpy(), rtol=3e-5, atol=3e-7)
        np.testing.assert_array_equal(sh.reshape(32, 3, 9), self.gaussians.harmonics[0].numpy())

    def test_native_nanogs_moments_match_independent_formula(self):
        mu, sc, quat, op, sh = to_nanogs(self.gaussians)
        result = moment_matching(mu[:1], sc[:1], quat[:1], op[:1], sh[:1],
                                 mu[1:2], sc[1:2], quat[1:2], op[1:2], sh[1:2])
        weights = (2 * np.pi) ** 1.5 * op[:2] * sc[:2].prod(-1) + 1e-12
        weights /= weights.sum()
        mean = (mu[:2] * weights[:, None]).sum(0)
        delta = mu[:2] - mean
        cov = sigma_from_scale_quat_batch(sc[:2], quat[:2])
        cov = (weights[:, None, None] * (cov + delta[:, :, None] * delta[:, None, :])).sum(0) + 1e-8 * np.eye(3)
        np.testing.assert_allclose(result[0][0], mean, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(sigma_from_scale_quat_batch(result[1], result[2])[0], cov, rtol=3e-5, atol=3e-7)
        np.testing.assert_allclose(result[4][0], (weights[:, None] * sh[:2]).sum(0), rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(result[3][0], 1 - (1 - op[0]) * (1 - op[1]), rtol=1e-6)

    def test_nanogs_exact_budgets_including_one_and_no_input_mutation(self):
        before = copy.deepcopy(self.gaussians)
        for count in (24, 16, 8, 3, 1):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result, _ = merge(self.gaussians, count)
            self.assertEqual(result.means.shape[1], count)
            self.assertTrue(all(torch.isfinite(v).all() for v in attributes(result)))
        for a, b in zip(attributes(before), attributes(self.gaussians)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_disk_cache_roundtrip_and_frame_identity_guard(self):
        entry = {'context': [3, 8], 'target': [4, 5, 6]}
        key = scene_key('test_scene', entry)
        manifest = {'signature': 'test', 'scenes': {'test_scene': {'key': key, 'entry': entry}}}
        value = {'signature': 'test', 'key': key, 'context_indices': [3, 8], 'target_indices': [4, 5, 6],
                 'gaussians': gaussian_dict(self.gaussians), 'support': self.cache}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.pt'
            save_torch(path, value)
            restored = torch.load(path, weights_only=True)
            validate_scene(restored, manifest, 'test_scene')
            self.assertEqual(restored['support']['image_shape'], (2, 4, 4))
        with self.assertRaises(RuntimeError):
            validate_scene(dict(value, target_indices=[6, 5, 4]), manifest, 'test_scene')

    def test_lightning_checkpoint_loads_strictly_and_rejects_partial_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Final++_RE10k.ckpt'
            state = {'encoder.' + name: value for name, value in self.decoder.state_dict().items()}
            state['losses.fake'] = torch.zeros(1)
            torch.save({'state_dict': state, 'global_step': 140000}, path)
            head = copy.deepcopy(self.decoder)
            self.assertEqual(load_encoder_weights(head, path), 140000)
            for value, expected in zip(head.parameters(), self.decoder.parameters()):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            del state[next(key for key in state if key.startswith('encoder.'))]
            torch.save({'state_dict': state}, path)
            with self.assertRaises(RuntimeError):
                load_encoder_weights(head, path)

    def test_cache_manifest_rejects_changed_alignment_loss_or_checkpoint(self):
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.ckpt'
            checkpoint.write_bytes(b'checkpoint identity')
            index = root / 'index.json'
            save_json(index, {'scene': {'context': [3, 8], 'target': [4, 5], 'overlap': .2}})
            cfg = OmegaConf.create({
                'dataset': {'re10k': {'view_sampler': {'name': 'evaluation', 'num_context_views': 2,
                                                       'index_path': str(index)}}},
                'checkpointing': {'load': str(checkpoint)}, 'model': {'name': 'Final++'},
                'test': {'align_pose': True}, 'loss': {'mse': {'weight': 1.0}},
                'seed': 123, 'control': {'max_scenes': None},
            })
            first, _ = cache_manifest(cfg, root / 'cache')
            second, _ = cache_manifest(cfg, root / 'cache')
            self.assertEqual(first['signature'], second['signature'])
            cfg.loss.mse.weight = 2.0
            with self.assertRaises(RuntimeError):
                cache_manifest(cfg, root / 'cache')
            cfg.loss.mse.weight = 1.0
            checkpoint.write_bytes(b'another checkpoint')
            with self.assertRaises(RuntimeError):
                cache_manifest(cfg, root / 'cache')

    def test_cache_builder_calls_encoder_once_then_reuses_scene_on_restart(self):
        from dataclasses import dataclass
        from omegaconf import OmegaConf
        @dataclass
        class RenderCfg:
            name: str = 'splatting_cuda'
        encoder = torch.nn.Module()
        encoder.gaussian_decoder = self.decoder
        calls = []
        def forward(context, step, control_dump):
            calls.append(1)
            return encoder.gaussian_decoder(
                self.points, self.features, image_shape=(2, 4, 4), rgb=self.rgb,
                control_cache=control_dump,
            )
        encoder.forward = forward
        encoder.get_data_shim = lambda: lambda batch: batch
        typed = SimpleNamespace(
            data_loader=SimpleNamespace(test=SimpleNamespace(batch_size=1)), dataset=[],
            model=SimpleNamespace(encoder=SimpleNamespace(gaussian_adapter=SimpleNamespace(sh_degree=2)),
                                  decoder=RenderCfg()), loss=[], test=SimpleNamespace(align_pose=False),
        )
        config_module = ModuleType('src.config')
        config_module.load_typed_root_config = lambda cfg: typed
        data_module = ModuleType('src.dataset.data_module')
        batch = {'scene': ['scene'], 'context': {'index': torch.tensor([[3, 8]]), 'overlap': torch.tensor([[.2]])},
                 'target': {'index': torch.tensor([[4, 5]]), 'image': torch.rand(1, 2, 3, 8, 8),
                            'extrinsics': torch.eye(4)[None, None].repeat(1, 2, 1, 1)}}
        data_module.DataModule = lambda *args, **kwargs: SimpleNamespace(test_dataloader=lambda: iter([copy.deepcopy(batch)]))
        decoder_module = ModuleType('src.model.decoder')
        decoder_module.get_decoder = lambda cfg: torch.nn.Identity()
        loss_module = ModuleType('src.loss')
        loss_module.get_losses = lambda cfg: []
        global_module = ModuleType('src.global_cfg')
        global_module.set_cfg = lambda cfg: None
        modules = {'src.config': config_module, 'src.dataset.data_module': data_module,
                   'src.model.decoder': decoder_module, 'src.loss': loss_module, 'src.global_cfg': global_module}
        entry = {'context': [3, 8], 'target': [4, 5]}
        manifest = {'signature': 'test', 'complete': False,
                    'scenes': {'scene': {'key': scene_key('scene', entry), 'entry': entry}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = OmegaConf.create({'dataset': {'re10k': {'view_sampler': {'index_path': 'unused'}}}})
            source_encoder = sys.modules['src.model.encoder']
            with patch.dict(sys.modules, modules), patch.object(source_encoder, 'get_encoder',
                    lambda cfg: (encoder, None), create=True), \
                    patch('src.eval_gaussian_control.cache_manifest', return_value=(manifest, Path('unused'))), \
                    patch('src.eval_gaussian_control.load_encoder_weights', return_value=140000), \
                    redirect_stdout(io.StringIO()):
                build_cache(cfg, root, torch.device('cpu'))
                self.assertEqual(len(calls), 1)
                build_cache(cfg, root, torch.device('cpu'))
                self.assertEqual(len(calls), 1)

    def test_report_averages_scenes_equally_and_saves_each_condition(self):
        rows = []
        for i, score in enumerate((10., 20.)):
            rows.append({'method': 'merge', 'percent': 5, 'scene': str(i), 'overlap_tag': 'small',
                         'gaussians': 3, 'actual_percent': 4.9, 'psnr': score, 'ssim': .6, 'lpips': .3,
                         'postprocess_ms': 2., 'render_ms_per_view': 3., 'target_views': 1 + i * 10})
        summary = summarize(rows, ['merge'], [5])
        self.assertEqual(summary[0]['psnr'], 15.)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_reports(output, rows, ['merge'], [5], 2)
            self.assertTrue((output / 'merge_005.txt').exists())
            self.assertEqual(len((output / 'per_scene.jsonl').read_text().splitlines()), 2)

    def test_cached_evaluation_shares_full_reference_and_resumes_without_rendering(self):
        from omegaconf import OmegaConf
        render_counts = []
        class Renderer(torch.nn.Module):
            def forward(self, gaussian, extrinsics, intrinsics, near, far, shape):
                render_counts.append(gaussian.means.shape[1])
                color = torch.full((1, extrinsics.shape[1], 3, *shape), gaussian.means.shape[1] / 64.)
                return SimpleNamespace(color=color)
        class RenderCfg:
            def __init__(self, **kwargs):
                pass
        renderer_module = ModuleType('src.model.decoder')
        renderer_module.get_decoder = lambda cfg: Renderer()
        cfg_module = ModuleType('src.model.decoder.decoder_splatting_cuda')
        cfg_module.DecoderSplattingCUDACfg = RenderCfg
        metrics = ModuleType('src.evaluation.metrics')
        for name in ('compute_lpips', 'compute_psnr', 'compute_ssim'):
            setattr(metrics, name, lambda gt, pred: (gt - pred).square().mean((1, 2, 3)))
        misc = ModuleType('src.misc.utils')
        misc.get_overlap_tag = lambda overlap: 'small'
        cfg = OmegaConf.create({'control': {
            'percentages': [100, 25, 1], 'methods': ['prune', 'merge', 'redistribute'],
            'timing_repeats': 2, 'warmup_runs': 0, 'render_chunk_size': 1, 'save_variants': False,
        }})
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'cache'
            output = Path(directory) / 'results'
            output.mkdir(parents=True)
            entry = {'context': [3, 8], 'target': [4, 5]}
            key = scene_key('scene', entry)
            manifest = {'signature': 'test', 'complete': True,
                        'scenes': {'scene': {'key': key, 'entry': entry}}}
            save_json(root / 'manifest.json', manifest)
            save_torch(root / 'decoder.pt', {
                'signature': 'test', 'cfg': asdict(self.decoder.cfg), 'sh_degree': 2,
                'state_dict': self.decoder.state_dict(), 'render_cfg': {},
            })
            save_torch(root / 'scenes' / f'{key}.pt', {
                'signature': 'test', 'key': key, 'scene': 'scene', 'overlap': .2,
                'context_indices': [3, 8], 'target_indices': [4, 5],
                'gaussians': gaussian_dict(self.gaussians), 'support': self.cache,
                'target': {'image': torch.rand(1, 2, 3, 8, 8),
                           'extrinsics': torch.eye(4)[None, None].repeat(1, 2, 1, 1),
                           'intrinsics': torch.eye(3)[None, None].repeat(1, 2, 1, 1),
                           'near': torch.ones(1, 2), 'far': torch.ones(1, 2) * 100},
            })
            modules = {'src.model.decoder': renderer_module,
                       'src.model.decoder.decoder_splatting_cuda': cfg_module,
                       'src.evaluation.metrics': metrics, 'src.misc.utils': misc}
            with patch.dict(sys.modules, modules), patch('torch.cuda.get_device_name', return_value='CPU test'), \
                    patch('torch.cuda.synchronize'), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                evaluate(cfg, root, output, torch.device('cpu'))
                self.assertEqual(render_counts.count(32), 4)  # 2 views x (warm + measured), only once
                rows = [json.loads(line) for line in (output / 'per_scene.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 9)
                full = [r for r in rows if r['percent'] == 100]
                self.assertEqual(len(full), 3)
                self.assertTrue(all(r['psnr'] == full[0]['psnr'] and r['postprocess_ms'] == 0 for r in full))
                for row in rows:
                    self.assertEqual(row['gaussians'], target_count(32, row['percent']))
                count = len(render_counts)
                evaluate(cfg, root, output, torch.device('cpu'))
                self.assertEqual(len(render_counts), count)


if __name__ == '__main__':
    unittest.main()
