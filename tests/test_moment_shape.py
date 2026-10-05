"""Experiment 2: covariance shape freedom on separate-weight + original 2DKNN."""

import ast
import copy
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from einops import rearrange

from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes, moment

shape_module = importlib.import_module('_moment_test.encoder.common.moment_shape')
selection = importlib.import_module('_moment_test.encoder.common.separate_appearance')
reader_module = importlib.import_module('_moment_test.encoder.common.image_neighborhood_appearance')


def cfg(**overrides):
    values = dict(feature_dim=5, hidden_dim=8, num_neighbors=3, chunk_size=4,
                  appearance_2d=True, separate_appearance=True, moment_shape=True,
                  appearance_dim=3, shape_hidden_dim=9)
    values.update(overrides)
    return Cfg(**values)


def activate(model):
    torch.nn.init.normal_(model.shape_head.mlp[-1].weight, std=.1)
    torch.nn.init.normal_(model.appearance_head.weight, std=.2)
    torch.nn.init.normal_(model.appearance_sh_head.weight, std=.15)


def wrapper_method(name):
    path = ROOT / 'src/model/model_wrapper.py'
    cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
               if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec')


class MomentShapeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(151)
        torch.set_num_threads(1)

    def test_parameter_count_identity_and_invalid_settings(self):
        model = Decoder(cfg(feature_dim=256, hidden_dim=64, appearance_dim=32, shape_hidden_dim=128))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 115284)
        self.assertEqual(sum(p.numel() for p in model.shape_head.parameters()), 34438)
        self.assertEqual(sum(p.numel() for p in model.appearance_reader.parameters()), 25440)
        features = torch.randn(5, 256)
        basis = torch.randn(5, 3, 3)
        covariance = basis @ basis.transpose(-1, -2)
        covariance[0].zero_()
        after, transform = model.shape_head(features, covariance)
        torch.testing.assert_close(after, covariance, rtol=0, atol=0)
        torch.testing.assert_close(transform, torch.eye(3).expand(5, 3, 3), rtol=0, atol=0)
        for options in (dict(shape_hidden_dim=0), dict(shape_scale_limit=1.0),
                        dict(shape_scale_limit=float('inf')), dict(shape_shear_limit=-1.0)):
            with self.assertRaises(ValueError):
                Decoder(cfg(**options))

    def test_selection_matches_dense_incoming_reference_and_gradients(self):
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])
        geometry = torch.tensor([[.2, .3], [.7, .3], [1., .5], [1., 0.]],
                                dtype=torch.double, requires_grad=True)
        scores = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
        selected = selection.appearance_weights(geometry, scores, neighbors)
        expected = scores.new_zeros(8)
        for i in range(4):
            edge = ((neighbors == i) & (geometry > 0)).flatten().nonzero().flatten()
            logits = geometry.flatten()[edge].log() + scores.flatten()[edge]
            expected = expected.index_copy(0, edge, logits.softmax(0))
        expected = expected.reshape_as(scores)
        torch.testing.assert_close(selected, expected)
        self.assertEqual(selected[-1, -1].item(), 0)
        signal = torch.randn_like(scores)
        for a, b in zip(torch.autograd.grad((selected * signal).sum(), (geometry, scores)),
                        torch.autograd.grad((expected * signal).sum(), (geometry, scores))):
            torch.testing.assert_close(a, b)

    def test_initial_values_rng_and_original_gradients_match_2dknn(self):
        torch.manual_seed(53)
        baseline = Decoder(cfg(moment_shape=False, separate_appearance=False))
        rng = torch.get_rng_state()
        torch.manual_seed(53)
        model = Decoder(cfg())
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name], parameter, rtol=0, atol=0)
        p, f = [torch.randn(1, 12, d, requires_grad=True) for d in (3, 5)]
        bp, bf = [x.detach().clone().requires_grad_() for x in (p, f)]
        rgb = torch.rand(1, 12, 3)
        report = {}
        actual = model(p, f, image_shape=(2, 2, 3), rgb=rgb, diagnostics=report)
        expected = baseline(bp, bf, image_shape=(2, 2, 3), rgb=rgb)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        signals = [torch.randn_like(x) for x in attributes(actual)]
        sum((a * s).sum() for a, s in zip(attributes(actual), signals)).backward()
        sum((b * s).sum() for b, s in zip(attributes(expected), signals)).backward()
        for a, b in ((p.grad, bp.grad), (f.grad, bf.grad)):
            torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-6)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, parameter.grad,
                                       rtol=5e-5, atol=5e-6)
        self.assertGreater(model.shape_head.mlp[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.appearance_head.weight.grad.abs().sum().item(), 0)
        self.assertEqual(report['shape/identity_distance'].item(), 0)
        self.assertAlmostEqual(report['shape/radius_ratio'].item(), 1)
        self.assertLess(report['weight_tv'].item(), 1e-6)

    def test_shape_formula_bounds_spd_and_only_covariance_changes(self):
        model = Decoder(cfg())
        activate(model)
        with torch.no_grad():
            model.shape_head.mlp[-1].bias.copy_(torch.tensor([10., -10., .2, 10., -10., .3]))
        count = 7
        mass, means, pooled = torch.rand(count), torch.randn(count, 3), torch.randn(count, 5)
        basis = torch.randn(count, 3, 3)
        covariance = basis @ basis.transpose(-1, -2)
        covariance[0].zero_()
        covariance[1] = torch.diag(torch.tensor([1., 0., 0.]))
        coverage = 1 + 2 * model.scale_head(pooled).sigmoid()
        before = covariance * coverage[..., None].square()
        after, transform = model.shape_head(pooled, before)
        diagonal = transform.diagonal(dim1=-2, dim2=-1)
        self.assertTrue(((diagonal >= .25) & (diagonal <= 4)).all())
        self.assertTrue((transform.tril(-1).abs() <= .5).all())
        floor = model.cfg.covariance_floor**2 + 8 * torch.finfo(after.dtype).eps * after.diagonal(
            dim1=-2, dim2=-1).sum(-1)
        expected_covariance = (after + floor[:, None, None] * torch.eye(3)) * 9
        result = model.build_gaussians(mass, means, covariance, pooled, torch.tensor(3.))
        torch.testing.assert_close(result.covariances[0], expected_covariance)
        self.assertTrue((torch.linalg.eigvalsh(result.covariances) > 0).all())
        ref = copy.deepcopy(model)
        ref.shape_head = None
        plain = ref.build_gaussians(mass, means, covariance, pooled, torch.tensor(3.))
        for a, b in ((result.means, plain.means), (result.opacities, plain.opacities),
                     (result.harmonics, plain.harmonics)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater((result.covariances - plain.covariances).abs().max().item(), 0)

    def test_active_checkpointed_forward_gradients_and_scale_equivariance(self):
        model = Decoder(cfg())
        activate(model)
        ref = copy.deepcopy(model)
        ref.cfg.checkpoint_chunks = False
        ref.cfg.chunk_size = 1000
        p, f, rgb = [torch.randn(2, 12, d, requires_grad=True) for d in (3, 5, 3)]
        rp, rf, rr = [x.detach().clone().requires_grad_() for x in (p, f, rgb)]
        report, reference_report = {}, {}
        with patch.object(moment, 'build_knn', wraps=moment.build_knn) as search:
            actual = model(p, f, image_shape=(2, 2, 3), rgb=rgb, diagnostics=report)
            self.assertEqual(search.call_count, 2)
        expected = ref(rp, rf, image_shape=(2, 2, 3), rgb=rr, diagnostics=reference_report)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        for key, value in report.items():
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value))
            torch.testing.assert_close(value, reference_report[key], rtol=3e-5, atol=3e-6)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [(p, rp), (f, rf), (rgb, rr), *zip(model.parameters(), ref.parameters())]:
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertGreater(a.grad.abs().sum().item(), 0)
            torch.testing.assert_close(a.grad, b.grad, rtol=8e-4, atol=8e-6)
        with torch.no_grad():
            scaled = model(p * 3, f, image_shape=(2, 2, 3), rgb=rgb)
        for a, b in ((scaled.means, actual.means * 3),
                     (scaled.covariances, actual.covariances * 9),
                     (scaled.harmonics, actual.harmonics), (scaled.opacities, actual.opacities)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)

    def test_zero_moments_remain_finite_with_trainable_shape(self):
        model = Decoder(cfg(num_neighbors=1))
        activate(model)
        points = torch.ones(1, 1, 3, requires_grad=True)
        features = torch.randn(1, 1, 5, requires_grad=True)
        report = {}
        result = model(points, features, image_shape=(1, 1, 1), rgb=torch.rand(1, 1, 3), diagnostics=report)
        self.assertTrue(all(torch.isfinite(x).all() for x in attributes(result)))
        self.assertTrue((torch.linalg.eigvalsh(result.covariances) > 0).all())
        self.assertEqual(report['shape/nonzero_moment_fraction'].item(), 0)
        self.assertEqual(report['shape/radius_ratio'].item(), 1)
        sum(x.square().mean() for x in attributes(result)).backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertTrue(torch.isfinite(features.grad).all())
        for parameter in model.shape_head.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_experiment_config_and_optimizer_include_shape_and_selection(self):
        from hydra import compose, initialize_config_dir
        from dacite import from_dict
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            config = compose(config_name='main', overrides=['+experiment=re10k_moment'])
        options = from_dict(Cfg, OmegaConf.to_container(config.model.encoder.moment_decoder))
        self.assertTrue(options.moment_shape and options.separate_appearance and options.appearance_2d)
        self.assertEqual((options.feature_dim, options.appearance_dim, options.shape_hidden_dim), (256, 32, 128))
        self.assertEqual(config.wandb.name, 'moment-shape')
        model = torch.nn.Module()
        model.encoder = torch.nn.Module()
        model.encoder.gaussian_decoder = Decoder(cfg())
        model.encoder.backbone = torch.nn.Linear(2, 2)
        model.optimizer_cfg = SimpleNamespace(lr=1e-4, backbone_lr_multiplier=.1, warm_up_steps=2)
        namespace = {'torch': torch, 'get_cfg': lambda: {'trainer': {'max_steps': 100}}}
        exec(wrapper_method('configure_optimizers'), namespace)
        optimizer = namespace['configure_optimizers'](model)['optimizer']
        new_params = {id(p) for p in optimizer.param_groups[0]['params']}
        self.assertTrue(all(id(p) in new_params for p in model.encoder.gaussian_decoder.parameters()))

    def test_training_logs_shape_prefix_without_changing_loss(self):
        namespace = {'torch': torch, 'rearrange': rearrange,
                     'compute_psnr': lambda target, image: image.flatten(1).mean(1),
                     'image_error_statistics': reader_module.image_error_statistics}
        exec(wrapper_method('training_step'), namespace)
        logs, requests = [], []

        class Encoder:
            gaussian_decoder = SimpleNamespace(cfg=cfg(log_every_n_steps=50))

            def __call__(self, context, step, visualization_dump=None, diagnostics_dump=None):
                if set(context) != {'image'}:
                    raise AssertionError('Only context images belong in the encoder')
                requests.append(diagnostics_dump is not None)
                if diagnostics_dump is not None:
                    diagnostics_dump['shape/radius_ratio'] = torch.tensor(1.2)

        batch = {'context': {'image': torch.zeros(1, 2, 3, 2, 2)},
                 'target': {'image': torch.zeros(1, 1, 3, 2, 2), 'extrinsics': None,
                            'intrinsics': None, 'near': None, 'far': None}}
        wrapper = SimpleNamespace(
            data_shim=lambda x: x, encoder=Encoder(), distiller=None,
            decoder=SimpleNamespace(forward=lambda *a, **kw: SimpleNamespace(color=batch['target']['image'])),
            train_cfg=SimpleNamespace(depth_mode=None, print_log_every_n_steps=10),
            losses=[SimpleNamespace(name='test', forward=lambda *a: torch.tensor(2.))],
            global_rank=1, trainer=SimpleNamespace(log_every_n_steps=50), step_tracker=None,
            log=lambda *a, **kw: None, log_dict=lambda value, **kw: logs.append((value, kw)),
        )
        for step in (0, 48, 49, 50):
            wrapper.global_step = step
            self.assertEqual(namespace['training_step'](wrapper, batch, 0).item(), 2.)
        self.assertEqual(requests, [False, False, True, False])
        self.assertIn('moment_shape/shape/radius_ratio', logs[0][0])
        self.assertIn('moment_shape/edge_mse', logs[0][0])
        self.assertTrue(logs[0][1]['sync_dist'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for device parity')
    def test_cuda_matches_cpu_values_and_gradients(self):
        cpu = Decoder(cfg())
        activate(cpu)
        gpu = copy.deepcopy(cpu).cuda()
        inputs = [torch.randn(1, 12, d, requires_grad=True) for d in (3, 5, 3)]
        cuda_inputs = [x.detach().cuda().requires_grad_() for x in inputs]
        a = cpu(inputs[0], inputs[1], image_shape=(2, 2, 3), rgb=inputs[2])
        b = gpu(cuda_inputs[0], cuda_inputs[1], image_shape=(2, 2, 3), rgb=cuda_inputs[2])
        for x, y in zip(attributes(a), attributes(b)):
            torch.testing.assert_close(x, y.cpu(), rtol=5e-4, atol=5e-5)
        sum(x.square().mean() for x in attributes(a)).backward()
        sum(x.square().mean() for x in attributes(b)).backward()
        for x, y in [*zip(inputs, cuda_inputs), *zip(cpu.parameters(), gpu.parameters())]:
            torch.testing.assert_close(x.grad, y.grad.cpu(), rtol=2e-3, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
