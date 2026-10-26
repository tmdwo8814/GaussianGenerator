"""CPU numerical and resume checks. Optional native CUDA check is opt-in.

python -m unittest discover -s tests -p test_lightgaussian_control.py -v
LIGHTGAUSSIAN_CUDA_TEST=1 enables the real extension on a CUDA machine.
"""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src import eval_lightgaussian_control as runner
from src.gaussian_control.common import gaussian_dict, target_count
from src.gaussian_control.lightgaussian import (
    count_view, load_count_extension, prune, prune_with_scores, volume_weighted_score,
)
from src.gaussian_control.lightgaussian_cameras import (
    ContextCalibration, chunk_camera_metadata, crop_intrinsics, fit_context_cameras,
)
from src.model.types import Gaussians


def gaussian_fixture(n=20, device='cpu'):
    return Gaussians(
        means=torch.arange(n * 3, dtype=torch.float32, device=device).reshape(1, n, 3),
        covariances=torch.eye(3, device=device).expand(1, n, 3, 3).clone(),
        harmonics=torch.arange(n * 75, dtype=torch.float32, device=device).reshape(1, n, 3, 25),
        opacities=torch.linspace(.05, .95, n, device=device)[None],
    )


def camera_fixture():
    height, width, views = 20, 24, 2
    y, x = np.mgrid[:height, :width]
    k = np.array([[.8 * width, 0., .5 * width], [0., .9 * height, .5 * height], [0., 0., 1.]])
    points, poses = [], []
    for index in range(views):
        # Non-planar, with a known canonical coordinate frame and physical scale.
        z = 3. + .4 * np.sin(x / 3.) + .2 * np.cos(y / 4.)
        xyz = np.stack(((x - k[0, 2]) / k[0, 0] * z,
                        (y - k[1, 2]) / k[1, 1] * z, z), -1)
        angle = .1 * index
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        center = np.array([.7 * index, .1 * index, -.1])
        points.append((xyz @ rotation.T + center).reshape(-1, 3))
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, center
        poses.append(pose)
    intrinsic = torch.tensor(k, dtype=torch.float32)
    intrinsic[0] /= width
    intrinsic[1] /= height
    payload = {'scene': 'synthetic', 'support': {
        'points': torch.tensor(np.concatenate(points) / 2., dtype=torch.float64),
        'scene_scale': torch.tensor(2.), 'image_shape': (views, height, width),
    }}
    return payload, intrinsic.repeat(views, 1, 1), torch.tensor(np.stack(poses)).float()


class LightGaussianNumericalTests(unittest.TestCase):
    def test_released_volume_weight_uses_90_percent_reference_without_clipping(self):
        # sqrt(det(diag(r^2))) = r^3; reference is the smallest for N=10.
        radius = torch.arange(1, 11, dtype=torch.float64)
        covariance = torch.eye(3).repeat(10, 1, 1) * radius[:, None, None].square()
        importance = torch.linspace(1, 2, 10)
        expected = importance.double() * radius.pow(.3)
        actual = volume_weighted_score(covariance, importance)
        torch.testing.assert_close(actual, expected.float())
        self.assertGreater(actual[-1], importance[-1])

    def test_exact_budget_and_stable_ties_preserve_all_attributes(self):
        original = gaussian_fixture(20)
        scores = torch.tensor([1., 1., 2., 0., 3.] + [0.] * 15)
        for percent in runner.DEFAULT_PERCENTAGES:
            count = target_count(20, percent)
            result = prune_with_scores(original, scores, count)
            expected = sorted(sorted(range(20), key=lambda i: (-float(scores[i]), i))[:count])
            self.assertEqual(result.means.shape[1], count)
            for key, value in gaussian_dict(result).items():
                torch.testing.assert_close(value, gaussian_dict(original)[key][:, expected], rtol=0, atol=0)

    def test_pruning_scores_include_all_context_views_and_are_recomputed(self):
        original = gaussian_fixture(4)
        cameras = {name: torch.arange(2) for name in ('extrinsics', 'intrinsics', 'near', 'far')}
        calls = []
        def count_stub(extension, gaussians, extrinsics, *rest):
            calls.append(int(extrinsics))
            # View 0 sees ID0, view1 sees ID2; importance is accumulated.
            score = torch.tensor([1., 0., 0., 0.]) if int(extrinsics) == 0 else torch.tensor([0., 0., 2., 0.])
            return (score > 0).int(), score
        with patch('src.gaussian_control.lightgaussian.count_view', side_effect=count_stub):
            selected, detail = prune(None, original, cameras, (20, 24), 1)
            prune(None, original, cameras, (20, 24), 2)
        torch.testing.assert_close(selected.means, original.means[:, 2:3])
        self.assertEqual(calls, [0, 1, 0, 1])
        self.assertEqual(detail['zero_score_gaussians'], 2)

    def test_count_adapter_matches_existing_projection_and_covariance_convention(self):
        # Load only the existing projection function, without importing its CUDA
        # extension or the dataset stack on this CPU test machine.
        source = Path(runner.__file__).parent / 'model/decoder/cuda_splatting.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'get_projection_matrix')
        projection_module = ModuleType('src.model.decoder.cuda_splatting')
        namespace = {'torch': torch}
        code = 'from __future__ import annotations\n' + ast.unparse(node)
        exec(compile(code, str(source), 'exec'), namespace)
        projection_module.get_projection_matrix = namespace['get_projection_matrix']
        original = gaussian_fixture(2)
        original.covariances[0, 0] = torch.tensor([[2., .1, .2], [.1, 3., .3], [.2, .3, 4.]])
        pose = torch.eye(4)
        pose[:3, 3] = torch.tensor([1., 2., 3.])
        k = torch.tensor([[.5, 0., .5], [0., .5, .5], [0., 0., 1.]])
        received = []
        def count(*args):
            received.extend(args)
            return torch.ones(2, dtype=torch.int32), torch.ones(2), None
        with patch.dict('sys.modules', {'src.model.decoder.cuda_splatting': projection_module}):
            count_view(SimpleNamespace(count_gaussians=count), original, pose, k,
                       torch.tensor(.5), torch.tensor(10.), (24, 32), True)
        torch.testing.assert_close(received[1], original.means[0] * 2)
        indices = torch.triu_indices(3, 3)
        torch.testing.assert_close(received[7], original.covariances[0, :, indices[0], indices[1]] * 4)
        scaled_pose = pose.clone()
        scaled_pose[:3, 3] *= 2
        view = scaled_pose.inverse().T
        projection = namespace['get_projection_matrix'](torch.tensor([1.]), torch.tensor([20.]),
                                                       torch.tensor([np.pi / 2]), torch.tensor([np.pi / 2]))[0].T
        torch.testing.assert_close(received[8], view)
        torch.testing.assert_close(received[9], view @ projection)
        torch.testing.assert_close(received[16], scaled_pose[:3, 3])
        self.assertEqual(received[12:14], [24, 32])
        self.assertAlmostEqual(received[10], 1., places=6)

    def test_context_pnp_recovers_frame_and_scale_without_mutating_support(self):
        payload, intrinsic, expected = camera_fixture()
        before = payload['support']['points'].clone()
        cameras, reports = fit_context_cameras(payload, intrinsic, torch.ones(2), torch.full((2,), 100.))
        torch.testing.assert_close(cameras['extrinsics'], expected, rtol=1e-4, atol=2e-5)
        torch.testing.assert_close(before, payload['support']['points'], rtol=0, atol=0)
        self.assertTrue(all(row['median_reprojection_px'] < 1e-3 for row in reports))

    def test_context_metadata_matches_crop_and_baseline_and_reads_no_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test').mkdir()
            (root / 'test/index.json').write_text(json.dumps({'abc': 'chunk.torch'}))
            rows = torch.zeros(3, 18)
            rows[:, :4] = torch.tensor([.5, .8, .5, .5])
            poses = torch.eye(4).repeat(3, 1, 1)
            poses[2, 0, 3] = -2.
            rows[:, 6:] = poses[:, :3].reshape(3, 12)
            chunk = root / 'test/chunk.torch'
            torch.save([{'key': 'abc', 'cameras': rows,
                         'images': [torch.tensor([0, 255], dtype=torch.uint8)] * 3}], chunk)
            before = chunk.read_bytes()
            manifest = {'identity': {'dataset': {'re10k': {
                'roots': [str(root)], 'original_image_shape': [360, 640], 'make_baseline_1': True,
            }}}}
            calibration = ContextCalibration(manifest)
            intrinsic, near, far = calibration.intrinsics_and_bounds({
                'scene': 'abc', 'context_indices': [0, 2], 'support': {'image_shape': (2, 256, 256)},
            })
            self.assertAlmostEqual(float(intrinsic[0, 0, 0]), .5 * 455 / 256)
            torch.testing.assert_close(near, torch.full((2,), .05))
            torch.testing.assert_close(far, torch.full((2,), 50.))
            self.assertEqual(tuple(chunk_camera_metadata(str(chunk))['abc'].shape), (3, 18))
            self.assertEqual(chunk.read_bytes(), before)
            chunk_camera_metadata.cache_clear()


class LightGaussianPipelineTests(unittest.TestCase):
    def test_outputs_cannot_overlap_the_live_cache_or_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for output in (root / 'cache', root / 'cache/nested', root, root / 'final_plus_results'):
                with self.assertRaises(ValueError):
                    runner.validate_paths(root / 'cache', output)
            old = root / 'old_results'
            old.mkdir()
            (old / 'metadata.json').write_text(json.dumps({'methods': ['prune', 'merge', 'redistribute']}))
            with self.assertRaises(ValueError):
                runner.validate_paths(root / 'cache', old)

    def test_resume_reuses_shared_100_and_does_not_modify_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = runner.parser().parse_args([
                '--cache-dir', str(root / 'cache'), '--output-dir', str(root / 'new_results'),
                '--percentages', '100', '75', '50', '--warmup-runs', '0',
            ])
            args.baseline_results = root / 'original_results'
            args.output_dir.mkdir()
            manifest = {'signature': 'cached', 'identity': {'seed': 7}, 'scenes': {
                'synthetic': {'key': 'key1', 'entry': {'context': [0, 5], 'target': [2, 3]}}
            }}
            payload, intrinsic, _ = camera_fixture()
            payload.update(signature='cached', key='key1', context_indices=[0, 5], target_indices=[2, 3], overlap=.4)
            payload['gaussians'] = gaussian_dict(gaussian_fixture())
            payload['target'] = {
                'image': torch.zeros(1, 2, 3, 20, 24), 'extrinsics': torch.eye(4).repeat(1, 2, 1, 1),
                'intrinsics': intrinsic[None], 'near': torch.ones(1, 2), 'far': torch.full((1, 2), 100.),
            }
            path = args.cache_dir / 'scenes/key1.pt'
            runner.save_torch(path, payload)
            baseline = {
                'scene': 'synthetic', 'context_indices': [0, 5], 'target_indices': [2, 3], 'overlap': .4,
                'overlap_tag': 'medium', 'method': 'prune', 'percent': 100,
                'gaussians': 20, 'actual_percent': 100., 'psnr': 30., 'ssim': .9, 'lpips': .1,
                'postprocess_ms': 0., 'render_ms_per_view': 1.,
            }
            runner.save_json(args.baseline_results / 'metadata.json', {
                'signature': 'old', 'settings': {'cache_signature': 'cached'},
            })
            runner.save_json(args.baseline_results / 'scenes/key1.json', {'signature': 'old', 'rows': [baseline]})
            protected = [path, args.baseline_results / 'metadata.json', args.baseline_results / 'scenes/key1.json']
            before = {p: p.read_bytes() for p in protected}
            calibration = SimpleNamespace(intrinsics_and_bounds=lambda p: (intrinsic, torch.ones(2), torch.full((2,), 100.)))
            called = []
            def operation(extension, original, cameras, shape, count, invariant):
                called.append(count)
                return prune_with_scores(original, original.opacities[0], count), {}
            def render(gaussians, poses, *rest):
                return SimpleNamespace(color=torch.full((1, poses.shape[1], 3, 20, 24), .2))
            fail = [True]
            def metric(gt, pred):
                if called[-1] == 10 and fail[0]:
                    raise RuntimeError('simulate interruption')
                return torch.ones(len(gt)) * .5
            # Keep optional visualization dependencies outside this CPU test.
            utility = ModuleType('src.misc.utils')
            utility.get_overlap_tag = lambda overlap: 'medium'
            bundle = {'render_cfg': {'make_scale_invariant': True}}
            with patch.dict('sys.modules', {'src.misc.utils': utility}), patch.object(runner, 'prune', side_effect=operation), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'simulate interruption'):
                    runner.evaluate(args, manifest, bundle, 'new', torch.device('cpu'), None, render,
                                    {'psnr': metric, 'ssim': metric, 'lpips': metric}, calibration)
                self.assertEqual(called, [15, 10])
                partial = runner.read_json(args.output_dir / 'scenes/key1.json')['rows']
                self.assertEqual({r['percent'] for r in partial}, {100, 75})
                fail[0] = False
                runner.evaluate(args, manifest, bundle, 'new', torch.device('cpu'), None, render,
                                {'psnr': metric, 'ssim': metric, 'lpips': metric}, calibration)
                self.assertEqual(called, [15, 10, 10])
                # Fully completed resume performs no postprocess or render.
                runner.evaluate(args, manifest, bundle, 'new', torch.device('cpu'), None, render,
                                {'psnr': metric, 'ssim': metric, 'lpips': metric}, calibration)
                self.assertEqual(called, [15, 10, 10])
            completed = runner.read_json(args.output_dir / 'scenes/key1.json')['rows']
            identity = next(r for r in completed if r['percent'] == 100)
            self.assertEqual(identity['psnr'], 30.)
            self.assertTrue(identity['shared_final'])
            self.assertTrue((args.output_dir / 'context_cameras/key1.pt').is_file())
            self.assertIn('Timing mode: concurrent', (args.output_dir / 'summary.txt').read_text())
            self.assertEqual(len(runner.read_json(args.output_dir / 'summary.json')), 6)  # overall and medium
            for protected_path, content in before.items():
                self.assertEqual(protected_path.read_bytes(), content)


@unittest.skipUnless(os.getenv('LIGHTGAUSSIAN_CUDA_TEST') == '1' and torch.cuda.is_available(),
                     'native CUDA check is opt-in on a CUDA host')
class LightGaussianCudaTests(unittest.TestCase):
    def test_native_count_matches_one_ray_transmittance(self):
        torch.cuda.set_device(0)
        original = gaussian_fixture(2, 'cuda')
        original.means[:] = torch.tensor([[[0., 0., 2.], [0., 0., 3.]]], device='cuda')
        original.covariances *= .01
        original.opacities[:] = torch.tensor([[.4, .2]], device='cuda')
        with tempfile.TemporaryDirectory() as directory:
            extension = load_count_extension(directory)
            count, importance = count_view(
                extension, original, torch.eye(4, device='cuda'),
                torch.tensor([[.5, 0., .5], [0., .5, .5], [0., 0., 1.]], device='cuda'),
                torch.tensor(1., device='cuda'), torch.tensor(10., device='cuda'), (1, 1), False,
            )
            torch.testing.assert_close(count, torch.ones(2, dtype=torch.int32, device='cuda'))
            torch.testing.assert_close(importance, torch.tensor([.4, .12], device='cuda'), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
