"""Independent dense and finite-difference checks of sparse feature pooling."""

import importlib
import unittest

import torch

from test_moment_gaussian_decoder import Decoder, Cfg, attributes


pool_features = importlib.import_module('_moment_test.encoder.common.sparse_feature_pool').pool_features


class SparseFeaturePoolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        torch.set_num_threads(1)

    def test_values_and_gradients_match_dense_with_asymmetric_incoming_degrees(self):
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0], [4, 0]])
        for chunk in (1, 3, 20):
            for need_features, need_weights in ((True, True), (False, True), (True, False)):
                f = torch.randn(5, 4, dtype=torch.double, requires_grad=need_features)
                w = torch.rand(5, 2, dtype=torch.double, requires_grad=need_weights)
                rf = f.detach().clone().requires_grad_(need_features)
                rw = w.detach().clone().requires_grad_(need_weights)
                dense = torch.zeros(5, 5, dtype=torch.double).index_put(
                    (neighbors.flatten(), torch.arange(5).repeat_interleave(2)), rw.flatten(),
                    accumulate=True,
                )
                actual, expected = pool_features(f, w, neighbors, chunk), dense @ rf
                torch.testing.assert_close(actual, expected)
                signal = torch.randn_like(actual)
                (actual * signal).sum().backward()
                (expected * signal).sum().backward()
                if need_features:
                    torch.testing.assert_close(f.grad, rf.grad)
                if need_weights:
                    torch.testing.assert_close(w.grad, rw.grad)

    def test_first_and_second_derivatives_pass_finite_differences(self):
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 2]])
        features = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        weights = torch.rand(4, 2, dtype=torch.double, requires_grad=True)
        operation = lambda f, w: pool_features(f, w, neighbors, 3)
        self.assertTrue(torch.autograd.gradcheck(operation, (features, weights)))
        self.assertTrue(torch.autograd.gradgradcheck(operation, (features, weights)))

    def test_zero_weights_keep_gradients_and_noncontiguous_inputs_work(self):
        features = torch.randn(3, 5, dtype=torch.double).t().requires_grad_()
        weights = torch.zeros(5, 1, dtype=torch.double, requires_grad=True)
        neighbors = torch.arange(5)[:, None]
        result = pool_features(features, weights, neighbors, 2)
        torch.testing.assert_close(result, torch.zeros_like(features))
        result.sum().backward()
        torch.testing.assert_close(features.grad, torch.zeros_like(features))
        torch.testing.assert_close(weights.grad, features.detach().sum(-1, keepdim=True))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the pooling device check')
    def test_cuda_pool_matches_dense_values_and_gradients(self):
        count, dim, k = 97, 64, 16
        neighbors = torch.rand(count, count, device='cuda').argsort(dim=-1)[:, :k]
        features = torch.randn(count, dim, device='cuda', requires_grad=True)
        weights = torch.rand(count, k, device='cuda', requires_grad=True)
        dense = torch.zeros(count, count, device='cuda').index_put(
            (neighbors.flatten(), torch.arange(count, device='cuda').repeat_interleave(k)),
            weights.flatten(), accumulate=True,
        )
        actual, expected = pool_features(features, weights, neighbors, 11), dense @ features
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        signal = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * signal).sum(), (features, weights))
        gb = torch.autograd.grad((expected * signal).sum(), (features, weights))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)

    def test_batched_attributes_match_independent_per_scene_formula_and_gradients(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=7)).double()
        # Exercise nonconstant scale prediction as well as the default initializer.
        torch.nn.init.normal_(model.scale_head.weight, std=.2)
        inputs = [torch.rand(3, 7, dtype=torch.double, requires_grad=True),
                  torch.randn(3, 7, 3, dtype=torch.double, requires_grad=True),
                  torch.randn(3, 7, 3, 3, dtype=torch.double).square().requires_grad_(),
                  torch.randn(3, 7, 5, dtype=torch.double, requires_grad=True)]
        scale = torch.tensor([.01, 1., 123.], dtype=torch.double)
        actual = model.build_gaussians(*inputs, scale)
        expected = [[], [], [], []]
        for mass, means, cov, pooled, s in zip(*inputs, scale):
            coverage = model.cfg.scale_min + (model.cfg.scale_max - model.cfg.scale_min) * model.scale_head(pooled).sigmoid()
            cov = coverage[..., None].square() * cov
            guard = 8 * torch.finfo(cov.dtype).eps * cov.diagonal(dim1=-2, dim2=-1).sum(-1).detach()
            cov = (cov + (model.cfg.covariance_floor**2 + guard)[:, None, None] * torch.eye(3, dtype=cov.dtype)) * s.square()
            for bucket, value in zip(expected, (
                    means * s, cov, model.sh_head(pooled).reshape(-1, 3, model.d_sh) * model.sh_mask,
                    -torch.expm1(-mass))):
                bucket.append(value)
        expected = [torch.stack(values) for values in expected]
        signals = [torch.randn_like(x) for x in expected]
        for a, b in zip(attributes(actual), expected):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        targets = inputs + list(model.scale_head.parameters()) + list(model.sh_head.parameters())
        ga = torch.autograd.grad(sum((a * g).sum() for a, g in zip(attributes(actual), signals)), targets)
        gb = torch.autograd.grad(sum((b * g).sum() for b, g in zip(expected, signals)), targets)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
