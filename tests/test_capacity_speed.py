"""Capacity speed changes preserve independent heads and nested checkpoints."""

from functools import lru_cache
from inspect import unwrap
import importlib
import unittest
from unittest.mock import patch

import torch

from test_appearance_capacity import cfg, activate, capacity, neighborhood
from test_capacity_memory import saved_storage_bytes
from test_moment_gaussian_decoder import Decoder, attributes, moment

pool = importlib.import_module('_moment_test.encoder.common.sparse_feature_pool')
reader = importlib.import_module('_moment_test.encoder.common.image_neighborhood_appearance')


def models(device='cpu', **options):
    old = Decoder(cfg(appearance_heads=2, cache_image_neighbors=False, **options)).to(device)
    new = Decoder(cfg(appearance_heads=2, compile_kernels=True,
                      batch_appearance_projections=True, **options)).to(device)
    activate(old)
    torch.nn.init.normal_(old.scale_head.weight, std=.1)
    new.load_state_dict(old.state_dict(), strict=True)
    return old, new


class CapacitySpeedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(509)
        torch.set_num_threads(1)

    def compare(self, device='cpu', **options):
        old, new = models(device, **options)
        self.assertEqual([(n, p.shape) for n, p in old.named_parameters()],
                         [(n, p.shape) for n, p in new.named_parameters()])
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        ids = torch.arange(12, device=device)
        neighbors = torch.stack((ids, (ids + 1) % 12, (ids + 4) % 12), -1)
        inputs = [torch.randn(2, 12, d, device=device, requires_grad=True) for d in (3, 5, 3)]
        copies = [x.detach().clone().requires_grad_() for x in inputs]
        reports = ({}, {})
        signals = None
        for index, (model, tensors) in enumerate(((old, inputs), (new, copies))):
            with patch.object(moment, 'build_knn', return_value=neighbors) as search:
                result = model(tensors[0], tensors[1], rgb=tensors[2], image_shape=(2, 2, 3),
                               diagnostics=reports[index])
                if index == 0:
                    expected = [x.detach().clone() for x in attributes(result)]
                    signals = [torch.randn_like(x) for x in expected]
                else:
                    for a, b in zip(attributes(result), expected):
                        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
                sum((x * g).sum() for x, g in zip(attributes(result), signals)).backward()
                self.assertEqual(search.call_count, 2)  # No kNN inside checkpoint recomputation.
        self.assertEqual(reports[0].keys(), reports[1].keys())
        for key in reports[0]:
            self.assertFalse(reports[1][key].requires_grad)
            torch.testing.assert_close(reports[0][key], reports[1][key], rtol=1e-4, atol=1e-5)
        for a, b in [*zip(inputs, copies), *zip(old.parameters(), new.parameters())]:
            self.assertIsNotNone(a.grad)
            self.assertIsNotNone(b.grad)
            self.assertTrue(torch.isfinite(b.grad).all())
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-3, atol=3e-5)

    def test_decoder_values_all_gradients_metrics_and_checkpoint_parameter_order(self):
        for scene_checkpoint, chunk_checkpoint in ((True, True), (True, False), (False, True)):
            with self.subTest(scene=scene_checkpoint, chunk=chunk_checkpoint):
                self.compare(checkpoint_appearance=scene_checkpoint, checkpoint_chunks=chunk_checkpoint)

    def test_aot_compilation_with_two_heads_and_nested_checkpoints(self):
        @lru_cache(maxsize=None)
        def compiled(function):
            return torch.compile(unwrap(function), backend='aot_eager', fullgraph=True, dynamic=False)

        def dispatch(function, *args, enabled=False):
            return (compiled(function) if enabled else function)(*args)

        with patch.object(pool, 'run_tensor_kernel', side_effect=dispatch), \
             patch.object(reader, 'run_tensor_kernel', side_effect=dispatch), \
             patch.object(capacity, 'run_tensor_kernel', side_effect=dispatch), \
             patch.object(moment, 'run_tensor_kernel', side_effect=dispatch):
            self.compare()

    def test_batched_projections_preserve_independent_heads_and_gradients(self):
        for heads in (1, 2, 4):
            old = capacity.MultiHeadAppearance(5, 3, heads).double()
            new = capacity.MultiHeadAppearance(5, 3, heads, batch_projections=True).double()
            new.load_state_dict(old.state_dict(), strict=True)
            inputs = [torch.randn(12, d, dtype=torch.double, requires_grad=True)
                      for d in (3, 5, 3, 3, 5)]
            copies = [x.detach().clone().requires_grad_() for x in inputs]
            ids = torch.arange(12)
            neighbors = torch.stack((ids, (ids + 3) % 12), -1)
            image = neighborhood.make_image_neighborhood((2, 2, 3), [1, 4], ids.device)
            options = dict(chunk_size=5, checkpoint_chunks=True, collect_statistics=True)
            expected, reference_report = old(*inputs, neighbors, image, **options)
            actual, report = new(*copies, neighbors, image, **options)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            for key in report:
                torch.testing.assert_close(report[key], reference_report[key], rtol=1e-12, atol=1e-12)
            signal = torch.randn_like(expected)
            (actual * signal).sum().backward()
            (expected * signal).sum().backward()
            for a, b in [*zip(inputs, copies), *zip(old.parameters(), new.parameters())]:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-11)

    def test_cache_inference_transition_and_layout_changes(self):
        _, model = models()
        with patch.object(moment, 'make_image_neighborhood', wraps=moment.make_image_neighborhood) as make:
            with torch.inference_mode():
                first = model._image_neighbors((2, 2, 3), torch.device('cpu'))
            self.assertFalse(first.xy.is_inference())
            self.assertIs(first, model._image_neighbors((2, 2, 3), torch.device('cpu')))
            self.assertEqual(make.call_count, 1)
            model._image_neighbors((2, 3, 2), torch.device('cpu'))
            model.cfg.appearance_2d_radii = [1]
            model._image_neighbors((2, 3, 2), torch.device('cpu'))
            self.assertEqual(make.call_count, 3)
            self.assertNotIn('_image_cache', dict(model.named_buffers()))
            inputs = [torch.randn(1, 12, d, requires_grad=True) for d in (3, 5, 3)]
            output = model(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 3, 2))
            sum(x.square().sum() for x in attributes(output)).backward()
            self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs))
            model.to(dtype=torch.float64)
            self.assertIsNone(model._image_cache)

    def test_actual_widths_preserve_parameters_and_forward_retention(self):
        old, new = models(feature_dim=256, hidden_dim=64, appearance_dim=32,
                          appearance_mlp_dim=128, num_neighbors=16, chunk_size=64)
        self.assertEqual(sum(p.numel() for p in new.parameters()), 154649)
        inputs = [torch.randn(2, 512, d, requires_grad=True) for d in (3, 256, 3)]
        sizes = []
        for model in (old, new):
            size, output = saved_storage_bytes(model, inputs, lambda: model(
                inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 16, 16),
            ))
            sizes.append(size)
            del output
        # CPU forward-retention guard, not CUDA peak/workspace/recompute memory.
        self.assertLessEqual(sizes[1], sizes[0] * 1.05,
                             f'Unexpected retention growth: eager={sizes[0]}, optimized={sizes[1]}')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for Inductor kernels')
    def test_cuda_compilation_values_gradients_metrics_and_nested_checkpoints(self):
        self.compare('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for full-width pooling')
    def test_cuda_compiled_pool_full_width_and_partial_chunk(self):
        for width in (75, 256):
            f = torch.randn(width, 43, device='cuda').t().requires_grad_()
            w = torch.rand(43, 16, device='cuda', requires_grad=True)
            neighbors = torch.randint(43, (43, 16), device='cuda')
            old = pool.pool_features(f, w, neighbors, 16)
            new = pool.pool_features(f, w, neighbors, 16, compile_kernels=True)
            torch.testing.assert_close(new, old, rtol=3e-5, atol=3e-5)
            signal = torch.randn_like(old)
            for a, b in zip(torch.autograd.grad((new * signal).sum(), (f, w)),
                            torch.autograd.grad((old * signal).sum(), (f, w))):
                torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)


if __name__ == '__main__':
    unittest.main()
