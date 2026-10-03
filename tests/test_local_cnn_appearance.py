"""Local appearance before allocation: layout, gradients and geometry controls."""

import ast
import copy
import importlib
from types import MethodType, SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes, build_knn, moment

selection = importlib.import_module('_moment_test.encoder.common.separate_appearance')


def cfg(**overrides):
    values = dict(feature_dim=5, hidden_dim=8, num_neighbors=3, chunk_size=4,
                  separate_appearance=True, local_cnn=True, local_cnn_hidden_dim=7)
    values.update(overrides)
    return Cfg(**values)


def activate(model):
    nn.init.normal_(model.appearance_head.weight, std=.3)
    nn.init.normal_(model.local_appearance.layers[-1].weight, std=.2)
    nn.init.normal_(model.local_appearance.layers[-1].bias, std=.1)


def extract_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    return compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec')


class LocalCnnAppearanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(151)
        torch.set_num_threads(1)

    def test_configuration_and_parameter_count(self):
        from hydra import compose, initialize_config_dir
        from dacite import from_dict
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            config = compose(config_name='main', overrides=['+experiment=re10k_moment'])
        options = from_dict(Cfg, OmegaConf.to_container(config.model.encoder.moment_decoder))
        self.assertTrue(options.local_cnn and options.separate_appearance)
        self.assertEqual((options.feature_dim, options.local_cnn_hidden_dim, options.num_neighbors), (256, 256, 16))
        self.assertEqual(config.wandb.name, 'moment-local-cnn-v1')
        model = Decoder(options)
        self.assertEqual(sum(p.numel() for p in model.local_appearance.parameters()), 609099)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 662105)
        self.assertEqual(model.local_appearance.layers[0].kernel_size, (3, 3))
        self.assertEqual(model.local_appearance.layers[-1].kernel_size, (1, 1))
        self.assertIsNone(model.local_appearance.layers[0].bias)
        self.assertEqual(model.local_appearance.layers[-1].weight.count_nonzero().item(), 0)
        self.assertEqual(model.local_appearance.layers[-1].bias.count_nonzero().item(), 0)

    def test_zero_init_preserves_rng_values_and_original_gradients(self):
        torch.manual_seed(53)
        baseline = Decoder(cfg(local_cnn=False))
        rng = torch.get_rng_state()
        torch.manual_seed(53)
        model = Decoder(cfg())
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name], parameter, rtol=0, atol=0)
        points, features = [torch.randn(2, 24, d, requires_grad=True) for d in (3, 5)]
        bp, bf = [x.detach().clone().requires_grad_() for x in (points, features)]
        report = {}
        actual = model(points, features, image_shape=(2, 3, 4), diagnostics=report)
        expected = baseline(bp, bf)
        signals = [torch.randn_like(x) for x in attributes(actual)]
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        sum((a * s).sum() for a, s in zip(attributes(actual), signals)).backward()
        sum((b * s).sum() for b, s in zip(attributes(expected), signals)).backward()
        for a, b in ((points.grad, bp.grad), (features.grad, bf.grad)):
            torch.testing.assert_close(a, b, rtol=3e-4, atol=5e-6)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, parameter.grad,
                                       rtol=3e-4, atol=5e-6)
        self.assertGreater(model.local_appearance.layers[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.local_appearance.layers[-1].bias.grad.abs().sum().item(), 0)
        self.assertEqual(model.local_appearance.layers[0].weight.grad.count_nonzero().item(), 0)
        self.assertEqual(report['local_cnn/source_extra_sh_rms'].item(), 0)
        self.assertTrue(all(torch.isfinite(v) and not v.requires_grad for v in report.values()))

    def test_local_neighborhood_preserves_views_rows_and_columns(self):
        model = Decoder(cfg(feature_dim=1, local_cnn_hidden_dim=1), sh_degree=0)
        with torch.no_grad():
            model.sh_head.weight.zero_()
            model.sh_head.bias.zero_()
            model.local_appearance.layers[0].weight.fill_(1)
            model.local_appearance.layers[-1].weight.fill_(1)
        image = torch.zeros(2, 3, 4, 1)
        image[0, 1, 3, 0] = 2
        expected = torch.zeros(2, 3, 4, 3)
        expected[0, :, 2:, :] = 2
        actual, _ = model.predict_support_harmonics(image.reshape(-1, 1), (2, 3, 4))
        torch.testing.assert_close(actual.reshape_as(expected), expected, rtol=0, atol=0)
        # A direction-specific filter must preserve orientation, not average
        # the whole neighborhood or convolve the concatenated view boundary.
        with torch.no_grad():
            model.local_appearance.layers[0].weight.zero_()
            model.local_appearance.layers[0].weight[0, 0, 1, 2] = 1
        expected.zero_()
        expected[0, 1, 2] = 2
        actual, _ = model.predict_support_harmonics(image.reshape(-1, 1), (2, 3, 4))
        torch.testing.assert_close(actual.reshape_as(expected), expected, rtol=0, atol=0)

    def test_combined_source_sh_matches_dense_pooling_and_gradients(self):
        model = Decoder(cfg())
        activate(model)
        points, features = [torch.randn(1, 24, d, requires_grad=True) for d in (3, 5)]
        actual = model(points, features, image_shape=(2, 3, 4))
        scale = points.detach().norm(dim=-1).median(dim=-1).values.clamp_min(model.cfg.scene_epsilon)
        normalized = points[0] / scale[0]
        neighbors = build_knn(points[0], 3)
        q, scores = model.predict_allocation(normalized, features[0], neighbors, return_appearance=True)
        mass, means, cov, pooled, wg = model.aggregate_moments(
            normalized, features[0], neighbors, q, return_weights=True,
        )
        wa = selection.appearance_weights(wg, scores, neighbors)
        dense = points.new_zeros(24, 24).index_put(
            (neighbors.flatten(), torch.arange(24).repeat_interleave(3)), wa.flatten(), accumulate=True,
        )
        # Reference: convolve both original views together, add per-source
        # affine SH, then perform a dense incoming matrix multiplication.
        image = features[0].reshape(2, 3, 4, 5).permute(0, 3, 1, 2)
        first, _, last = model.local_appearance.layers
        local = F.conv2d(F.relu(F.conv2d(image, first.weight, padding=1)), last.weight, last.bias)
        local = local.permute(0, 2, 3, 1).reshape(24, 75)
        combined = model.sh_head(features[0]) + local
        expected = model.build_gaussians(mass, means, cov, pooled, scale[0],
                                         appearance_harmonics=dense @ combined)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        signal = torch.randn_like(actual.harmonics)
        # Coverage only participates in geometry; include all attributes in
        # the objective to test every trainable parameter's gradient.
        loss_a = (actual.harmonics * signal).sum() + sum(x.square().mean() for x in attributes(actual))
        loss_b = (expected.harmonics * signal).sum() + sum(x.square().mean() for x in attributes(expected))
        targets = (points, features, *model.parameters())
        for a, b in zip(torch.autograd.grad(loss_a, targets), torch.autograd.grad(loss_b, targets)):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=2e-5)

    def test_checkpoint_and_logging_parity_with_unchanged_geometry(self):
        model = Decoder(cfg())
        activate(model)
        reference = copy.deepcopy(model)
        reference.cfg.checkpoint_chunks = False
        reference.cfg.chunk_size = 1000
        points, features = [torch.randn(2, 24, d, requires_grad=True) for d in (3, 5)]
        rp, rf = [x.detach().clone().requires_grad_() for x in (points, features)]
        report = {}
        with patch.object(moment, 'build_knn', wraps=moment.build_knn) as search:
            actual = model(points, features, image_shape=(2, 3, 4), diagnostics=report)
            self.assertEqual(search.call_count, 2)
        expected = reference(rp, rf, image_shape=(2, 3, 4))
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [(points, rp), (features, rf), *zip(model.parameters(), reference.parameters())]:
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertGreater(a.grad.abs().sum().item(), 0)
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=5e-6)
        reference.local_appearance = None
        plain = reference(points.detach(), features.detach())
        for a, b in ((actual.means, plain.means), (actual.covariances, plain.covariances),
                     (actual.opacities, plain.opacities)):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        self.assertGreater((actual.harmonics - plain.harmonics).abs().sum().item(), 0)
        self.assertGreater(report['local_cnn/source_extra_sh_rms'].item(), 0)

    def test_image_neighbors_outside_3d_graph_can_affect_sh(self):
        model = Decoder(cfg(feature_dim=1, num_neighbors=1, local_cnn_hidden_dim=1), sh_degree=0)
        with torch.no_grad():
            model.sh_head.weight.zero_()
            model.sh_head.bias.zero_()
            model.local_appearance.layers[0].weight.fill_(1)
            model.local_appearance.layers[-1].weight.fill_(1)
        points = torch.randn(1, 18, 3)
        features = torch.ones(1, 18, 1, requires_grad=True)
        result = model(points, features, image_shape=(2, 3, 3))
        grad, = torch.autograd.grad(result.harmonics[0, 4].sum(), (features,))
        # K=1 gives slot 4 only its own support. Its local SH still reads all
        # nine original image pixels; none from the other view.
        self.assertTrue((grad[0, :9] > 0).all())
        self.assertEqual(grad[0, 9:].count_nonzero().item(), 0)
        torch.testing.assert_close(result.means, points)

    def test_invalid_layout_fails_before_search_and_tiny_views_are_valid(self):
        with self.assertRaises(ValueError):
            Decoder(cfg(separate_appearance=False))
        with self.assertRaises(ValueError):
            Decoder(cfg(local_cnn_hidden_dim=0))
        model = Decoder(cfg())
        points, features = torch.randn(1, 12, 3), torch.randn(1, 12, 5)
        for shape in (None, (2, 2, 2), (0, 3, 4)):
            with patch.object(moment, 'build_knn') as search, self.assertRaises(ValueError):
                model(points, features, image_shape=shape)
            search.assert_not_called()
        for shape in ((3, 1, 1), (2, 1, 4), (1, 4, 1)):
            count = shape[0] * shape[1] * shape[2]
            result = model(torch.randn(1, count, 3), torch.randn(1, count, 5), image_shape=shape)
            self.assertTrue(all(torch.isfinite(x).all() for x in attributes(result)))

    def test_encoder_passes_original_feature_maps_and_local_cnn_trains(self):
        namespace = {'torch': torch, 'Optional': Optional, 'Gaussians': moment.Gaussians, 'rearrange': rearrange}
        exec(extract_method(ROOT / 'src/model/encoder/encoder_noposplat.py', 'EncoderNoPoSplat', 'forward'), namespace)

        class FrontEnd(nn.Module):
            def forward(self, context, return_views):
                if set(context) != {'image'}:
                    raise AssertionError('Only context data belongs in the encoder')
                image = context['image']
                shape = torch.tensor([[3, 4]])
                return [], [], shape, shape, {'img': image[:, 0]}, {'img': image[:, 1]}

        class FeatureHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Conv2d(3, 5, 1)

            def forward(self, tokens, points, image, shape):
                return self.projection(image)

        encoder = nn.Module()
        encoder.forward = MethodType(namespace['forward'], encoder)
        encoder.gs_params_head_type = 'moment'
        encoder.backbone = FrontEnd()
        encoder.supports = nn.Parameter(torch.randn(2, 1, 3, 4, 3))
        encoder._downstream_head = lambda index, tokens, shape: {'pts3d': encoder.supports[index - 1]}
        encoder.gaussian_param_head, encoder.gaussian_param_head2 = FeatureHead(), FeatureHead()
        encoder.gaussian_decoder = Decoder(cfg(num_neighbors=1))
        activate(encoder.gaussian_decoder)
        image = torch.randn(1, 2, 3, 3, 4)
        expected_features = torch.cat([
            head.projection(image[:, v]).permute(0, 2, 3, 1).reshape(12, 5)
            for v, head in enumerate((encoder.gaussian_param_head, encoder.gaussian_param_head2))
        ])
        report = {}
        with patch.object(encoder.gaussian_decoder, 'predict_support_harmonics',
                          wraps=encoder.gaussian_decoder.predict_support_harmonics) as local:
            result = encoder({'image': image}, diagnostics_dump=report)
            torch.testing.assert_close(local.call_args.args[0], expected_features)
            self.assertEqual(local.call_args.args[1], (2, 3, 4))
        torch.testing.assert_close(result.means, encoder.supports.reshape(1, 24, 3))
        sum(x.square().mean() for x in attributes(result)).backward()
        for head in (encoder.gaussian_param_head, encoder.gaussian_param_head2):
            self.assertGreater(head.projection.weight.grad.abs().sum().item(), 0)
        self.assertIn('local_cnn/source_extra_sh_rms', report)
        self.assertGreater(encoder.gaussian_decoder.local_appearance.layers[0].weight.grad.abs().sum().item(), 0)

    def test_optimizer_includes_cnn_at_decoder_learning_rate(self):
        namespace = {'torch': torch, 'get_cfg': lambda: {'trainer': {'max_steps': 100}}}
        exec(extract_method(ROOT / 'src/model/model_wrapper.py', 'ModelWrapper', 'configure_optimizers'), namespace)
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.gaussian_decoder = Decoder(cfg())
        model.encoder.backbone = nn.Linear(2, 2)
        model.optimizer_cfg = SimpleNamespace(lr=1e-4, backbone_lr_multiplier=.1, warm_up_steps=2)
        optimizer = namespace['configure_optimizers'](model)['optimizer']
        decoder_parameters = {id(p) for p in optimizer.param_groups[0]['params']}
        self.assertTrue(all(id(p) in decoder_parameters for p in model.encoder.gaussian_decoder.local_appearance.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for device parity')
    def test_cuda_values_and_gradients_match_cpu(self):
        cpu = Decoder(cfg())
        activate(cpu)
        gpu = copy.deepcopy(cpu).cuda()
        points, features = [torch.randn(1, 24, d, requires_grad=True) for d in (3, 5)]
        gp, gf = [x.detach().cuda().requires_grad_() for x in (points, features)]
        actual = gpu(gp, gf, image_shape=(2, 3, 4))
        expected = cpu(points, features, image_shape=(2, 3, 4))
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a.cpu(), b, rtol=5e-4, atol=5e-5)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [(gp, points), (gf, features), *zip(gpu.parameters(), cpu.parameters())]:
            torch.testing.assert_close(a.grad.cpu(), b.grad, rtol=2e-3, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
