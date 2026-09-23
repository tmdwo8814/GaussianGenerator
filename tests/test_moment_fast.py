"""Equivalence checks for optimized decoding and exact GPU neighborhoods.

GPU tests execute on the training server after requirements-fast.txt is installed.
"""

import copy
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial import cKDTree

from test_moment_gaussian_decoder import Cfg, Decoder, attributes, build_knn

cuda_knn = importlib.import_module('_moment_test.encoder.common.cuda_knn')
decoder_module = importlib.import_module(Decoder.__module__)


def original_allocation(model, points, features, neighbors):
    destination = features[neighbors]
    source = features[:, None].expand_as(destination)
    delta = points[:, None] - points[neighbors]
    pair = torch.cat((destination, source, delta, delta.square().sum(-1, keepdim=True)), -1)
    logits = model.allocation_head(pair).squeeze(-1)
    return logits.softmax(-1) * (model.cfg.budget_max * model.budget_head(features).sigmoid())


class OriginalDecoder(Decoder):
    def predict_allocation(self, points, features, neighbors):
        return original_allocation(self, points, features, neighbors)


class FastMomentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_factorized_allocation_matches_original_values_and_all_gradients(self):
        for checkpoint in (False, True):
            model = Decoder(Cfg(feature_dim=5, hidden_dim=7, chunk_size=3,
                                checkpoint_chunks=checkpoint)).double()
            # Nontrivial weights: equivalence must not rely on near-uniform initialization.
            for parameter in model.allocation_head.parameters():
                torch.nn.init.normal_(parameter, std=.3)
            reference = copy.deepcopy(model)
            points = torch.randn(19, 3, dtype=torch.float64, requires_grad=True)
            features = torch.randn(19, 5, dtype=torch.float64, requires_grad=True)
            rp, rf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
            neighbors = build_knn(points, 6)
            expected = original_allocation(reference, rp, rf, neighbors)
            actual = model.predict_allocation(points, features, neighbors)
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
            target = torch.randn_like(actual)
            (actual * target).sum().backward()
            (expected * target).sum().backward()
            for actual_gradient, expected_gradient in [(points.grad, rp.grad), (features.grad, rf.grad)]:
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-11)
            for (name, parameter), (rname, ref) in zip(model.named_parameters(), reference.named_parameters()):
                self.assertEqual(name, rname)
                if ref.grad is not None:
                    torch.testing.assert_close(parameter.grad, ref.grad, rtol=1e-10, atol=1e-11)

    def test_full_decoder_preserves_checkpoint_keys_attributes_and_gradients(self):
        cfg = Cfg(feature_dim=5, hidden_dim=7, num_neighbors=5, chunk_size=4)
        optimized = Decoder(cfg)
        reference = OriginalDecoder(copy.deepcopy(cfg))
        reference.load_state_dict(optimized.state_dict(), strict=True)
        points = torch.randn(2, 23, 3, requires_grad=True)
        features = torch.randn(2, 23, 5, requires_grad=True)
        rp, rf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
        actual, expected = optimized(points, features), reference(rp, rf)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
        sum(value.square().mean() for value in attributes(actual)).backward()
        sum(value.square().mean() for value in attributes(expected)).backward()
        for a, b in [(points.grad, rp.grad), (features.grad, rf.grad)]:
            torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6)
        for parameter, ref in zip(optimized.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, ref.grad, rtol=3e-4, atol=3e-6)

    def test_self_first_handles_self_missing_and_nonfirst_ties(self):
        candidates = torch.tensor([[2, 0, 3], [2, 3, 0], [0, 1, 2], [3, 1, 0]])
        expected = torch.tensor([[0, 2, 3], [1, 2, 3], [2, 0, 1], [3, 1, 0]])
        torch.testing.assert_close(cuda_knn.self_first(candidates), expected)
        self.assertEqual(cuda_knn.self_first(torch.zeros(4, 1, dtype=torch.long)).flatten().tolist(), [0, 1, 2, 3])

    def test_self_compaction_preserves_order_for_every_self_position_and_absence(self):
        for k in (1, 2, 16, 32):
            count = 40
            for position in range(k + 1):  # k means self absent.
                rows = []
                expected = []
                for own in range(count):
                    others = [i for i in torch.randperm(count).tolist() if i != own][:k]
                    candidates = others[:]
                    if position < k:
                        candidates.insert(position, own)
                        candidates = candidates[:k]
                    rows.append(candidates)
                    expected.append([own] + [i for i in candidates if i != own][:k - 1])
                torch.testing.assert_close(cuda_knn.self_first(torch.tensor(rows)), torch.tensor(expected))

    def test_decoder_validates_whole_batch_once_and_rejects_invalid_points_before_search(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=7, num_neighbors=4))
        points, features = torch.randn(3, 11, 3), torch.randn(3, 11, 5)
        with patch.object(decoder_module, 'validate_points', wraps=decoder_module.validate_points) as validate, \
             patch.object(decoder_module, 'build_knn', wraps=build_knn) as search:
            model(points, features)
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(validate.call_args.args[0].shape, (3, 11, 3))
            self.assertEqual(search.call_count, 3)
            self.assertTrue(all(call.kwargs['check_finite'] is False for call in search.call_args_list))
        for invalid in (float('nan'), float('inf')):
            bad = points.clone()
            bad[2, 0, 0] = invalid
            with patch.object(decoder_module, 'build_knn') as search:
                with self.assertRaisesRegex(ValueError, 'non-finite'):
                    model(bad, features)
                search.assert_not_called()

    def test_cpu_dispatch_stays_exact_and_invalid_backend_fails(self):
        points = torch.randn(37, 3)
        torch.testing.assert_close(build_knn(points, 16), build_knn(points, 16, backend='scipy'))
        with self.assertRaises(ValueError):
            build_knn(points, backend='approximate')
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            build_knn(points, backend='cupy')
        for k in (1, 16):
            with self.assertRaises(ValueError):
                build_knn(torch.full((20, 3), float('nan')), k)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for CuPy KD-tree checks')
    def test_gpu_neighbors_exact_for_random_planar_duplicate_and_small_clouds(self):
        generator = torch.Generator().manual_seed(43)
        random = torch.randn(257, 3, generator=generator)
        planar = random.clone()
        planar[:, 2] = 0
        clouds = [random, planar, torch.zeros(33, 3), torch.ones(1, 3),
                  torch.cat((random[:32], random[:32]))]
        for points in clouds:
            for k in (1, 16, min(33, len(points) + 1)):
                gpu_points = points.cuda()
                result = build_knn(gpu_points, k).cpu()
                clipped = min(k, len(points))
                self.assertEqual(result.shape, (len(points), clipped))
                torch.testing.assert_close(result[:, 0], torch.arange(len(points)))
                self.assertTrue(all(row.unique().numel() == clipped for row in result))
                xyz = points.double().numpy()
                expected, _ = cKDTree(xyz).query(xyz, k=clipped)
                distances = np.linalg.norm(xyz[result.numpy()] - xyz[:, None], axis=-1)
                np.testing.assert_allclose(np.sort(distances, axis=1), np.asarray(expected).reshape(-1, clipped),
                                           rtol=1e-8, atol=1e-10)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for stream/device checks')
    def test_gpu_knn_on_nondefault_stream_and_each_visible_device(self):
        for device_id in range(torch.cuda.device_count()):
            device = torch.device('cuda', device_id)
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.device(device), torch.cuda.stream(stream):
                points = torch.randn(199, 3, device=device)
                result = build_knn(points, 16)
                distances = (points.double()[result] - points.double()[:, None]).square().sum(-1).sqrt()
            stream.synchronize()
            self.assertEqual(result.device, device)
            xyz = points.cpu().double().numpy()
            expected, _ = cKDTree(xyz).query(xyz, k=16)
            np.testing.assert_allclose(np.sort(distances.cpu().numpy(), axis=1), expected, rtol=1e-8, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
