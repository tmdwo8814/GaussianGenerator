"""Same-model speed path: outputs, all gradients, caching and CUDA fusion."""

from functools import lru_cache
from inspect import unwrap
import importlib
import unittest
from unittest.mock import patch

import torch

from test_moment_gaussian_decoder import Cfg, Decoder, attributes, moment

pool = importlib.import_module('_moment_test.encoder.common.sparse_feature_pool')
reader = importlib.import_module('_moment_test.encoder.common.image_neighborhood_appearance')
shape = importlib.import_module('_moment_test.encoder.common.moment_shape')


def models(device='cpu'):
    options = dict(feature_dim=5, hidden_dim=8, num_neighbors=3, chunk_size=4,
                   separate_appearance=True, appearance_2d=True, appearance_dim=3,
                   moment_shape=True, shape_hidden_dim=9)
    old = Decoder(Cfg(**options, cache_image_neighbors=False)).to(device)
    new = Decoder(Cfg(**options, compile_kernels=True, shape_chunk_size=9)).to(device)
    # Zero-initialized branches would hide broken parameter/input gradients.
    for name in ('shape_head.mlp.2.weight', 'appearance_head.weight', 'appearance_sh_head.weight'):
        torch.nn.init.normal_(dict(old.named_parameters())[name], std=.15)
    new.load_state_dict(old.state_dict(), strict=True)
    return old, new


class MomentShapeSpeedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(191)
        torch.set_num_threads(1)

    def compare(self, device='cpu'):
        old, new = models(device)
        self.assertEqual([(n, p.shape) for n, p in old.named_parameters()],
                         [(n, p.shape) for n, p in new.named_parameters()])
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        # Fixed candidates isolate CUDA tensor kernels from the CuPy dependency.
        count = 12
        ids = torch.arange(count, device=device)
        neighbors = torch.stack((ids, (ids + 1) % count, (ids + 4) % count), -1)
        inputs = [torch.randn(2, count, d, device=device, requires_grad=True) for d in (3, 5, 3)]
        copies = [x.detach().clone().requires_grad_() for x in inputs]
        reports = ({}, {})
        with patch.object(moment, 'build_knn', return_value=neighbors):
            expected = old(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3),
                           diagnostics=reports[0])
            actual = new(copies[0], copies[1], rgb=copies[2], image_shape=(2, 2, 3),
                         diagnostics=reports[1])
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-6)
        self.assertEqual(reports[0].keys(), reports[1].keys())
        for key in reports[0]:
            torch.testing.assert_close(reports[0][key], reports[1][key], rtol=1e-4, atol=1e-5)
        signals = [torch.randn_like(x) for x in attributes(expected)]
        sum((x * g).sum() for x, g in zip(attributes(expected), signals)).backward()
        sum((x * g).sum() for x, g in zip(attributes(actual), signals)).backward()
        for a, b in [*zip(inputs, copies), *zip(old.parameters(), new.parameters())]:
            self.assertIsNotNone(a.grad)
            self.assertIsNotNone(b.grad)
            self.assertTrue(torch.isfinite(b.grad).all())
            torch.testing.assert_close(a.grad, b.grad, rtol=8e-4, atol=2e-5)

    def test_full_decoder_outputs_gradients_statistics_and_checkpoint_keys(self):
        self.compare()

    def test_aot_autograd_and_checkpoint_work_together(self):
        # Exercise Dynamo/AOTAutograd on this CPU-only host. This is NOT a
        # CUDA/Inductor performance test; the CUDA test below uses real kernels.
        @lru_cache(maxsize=None)
        def compiled(function):
            return torch.compile(unwrap(function), backend='aot_eager', fullgraph=True, dynamic=False)

        def dispatch(function, *args, enabled=False):
            return (compiled(function) if enabled else function)(*args)

        with patch.object(pool, 'run_tensor_kernel', side_effect=dispatch), \
             patch.object(reader, 'run_tensor_kernel', side_effect=dispatch), \
             patch.object(shape, 'run_tensor_kernel', side_effect=dispatch):
            self.compare()

    def test_grid_cache_reuses_only_static_layout_and_invalidates_on_changes(self):
        _, model = models()
        with patch.object(moment, 'make_image_neighborhood', wraps=moment.make_image_neighborhood) as make:
            first = model._image_neighbors((2, 2, 3), torch.device('cpu'))
            self.assertIs(first, model._image_neighbors((2, 2, 3), torch.device('cpu')))
            self.assertEqual(make.call_count, 1)
            model._image_neighbors((2, 3, 2), torch.device('cpu'))
            model.cfg.appearance_2d_radii = [1]
            model._image_neighbors((2, 3, 2), torch.device('cpu'))
            self.assertEqual(make.call_count, 3)
            self.assertNotIn('_image_cache', dict(model.named_buffers()))
            self.assertFalse(any('cache' in key for key in model.state_dict()))
            model.to(dtype=torch.float64)
            self.assertIsNone(model._image_cache)
            model._image_neighbors((2, 3, 2), torch.device('cpu'))
            self.assertEqual(make.call_count, 4)

    def test_inference_cache_can_be_reused_for_backward(self):
        _, model = models()
        with torch.inference_mode():
            image = model._image_neighbors((2, 2, 3), torch.device('cpu'))
        self.assertFalse(image.xy.is_inference())
        inputs = [torch.randn(1, 12, d, requires_grad=True) for d in (3, 5, 3)]
        output = model(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3))
        sum(x.square().sum() for x in attributes(output)).backward()
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs))

    def test_optimized_pool_keeps_zero_weight_derivatives_and_double_backward(self):
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0]])
        features = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        weights = torch.zeros(3, 2, dtype=torch.double, requires_grad=True)
        operation = lambda f, w: pool.pool_features(f, w, neighbors, 2, compile_kernels=True)
        self.assertTrue(torch.autograd.gradcheck(operation, (features, weights)))
        self.assertTrue(torch.autograd.gradgradcheck(operation, (features, weights)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for Inductor kernels')
    def test_cuda_compiled_decoder_values_gradients_and_statistics(self):
        self.compare('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for full-width pooling')
    def test_cuda_compiled_pool_full_width_noncontiguous_and_partial_chunk(self):
        for width in (75, 256):
            f = torch.randn(width, 43, device='cuda').t().requires_grad_()
            w = torch.rand(43, 16, device='cuda', requires_grad=True)
            neighbors = torch.randint(43, (43, 16), device='cuda')
            reference = pool.pool_features(f, w, neighbors, 16)
            actual = pool.pool_features(f, w, neighbors, 16, compile_kernels=True)
            torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-5)
            signal = torch.randn_like(reference)
            ga = torch.autograd.grad((actual * signal).sum(), (f, w))
            gb = torch.autograd.grad((reference * signal).sum(), (f, w))
            for a, b in zip(ga, gb):
                torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)


if __name__ == '__main__':
    unittest.main()
