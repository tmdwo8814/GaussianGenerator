"""Experiment 1: independent selection, multi-head readout and nonlinear SH.

Run: python -m unittest discover -s tests -p test_appearance_capacity.py -v
"""

import ast
import copy
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from einops import rearrange

from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes, moment

capacity = importlib.import_module('_moment_test.encoder.common.appearance_capacity')
selection = importlib.import_module('_moment_test.encoder.common.separate_appearance')
neighborhood = importlib.import_module('_moment_test.encoder.common.appearance_neighborhood')
old_reader = importlib.import_module('_moment_test.encoder.common.image_neighborhood_appearance')


def cfg(**overrides):
    values = dict(feature_dim=5, hidden_dim=8, num_neighbors=3, chunk_size=4,
                  appearance_2d=True, separate_appearance=True, appearance_capacity=True,
                  appearance_heads=4, appearance_dim=3, appearance_mlp_dim=9)
    values.update(overrides)
    return Cfg(**values)


def activate(model):
    torch.nn.init.normal_(model.appearance_head.weight, std=.2)
    torch.nn.init.normal_(model.appearance_sh_head[-1].weight, std=.15)


class AppearanceCapacityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(151)
        torch.set_num_threads(1)

    def test_parameter_count_matches_agreed_design_and_zero_outputs(self):
        model = Decoder(cfg(feature_dim=256, hidden_dim=64, appearance_dim=32,
                            appearance_mlp_dim=256))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 272601)
        self.assertEqual(sum(p.numel() for p in model.appearance_reader.parameters()), 101760)
        self.assertEqual(sum(p.numel() for p in model.appearance_sh_head.parameters()), 117835)
        self.assertEqual(model.appearance_head.weight.count_nonzero().item(), 0)
        self.assertEqual(model.appearance_sh_head[-1].weight.count_nonzero().item(), 0)
        self.assertEqual(model.appearance_sh_head[-1].bias.count_nonzero().item(), 0)
        for flags in (dict(appearance_2d=False), dict(separate_appearance=False),
                      dict(appearance_heads=0), dict(appearance_mlp_dim=0)):
            with self.assertRaises(ValueError):
                Decoder(cfg(**flags))

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
        actual_grad = torch.autograd.grad((selected * signal).sum(), (geometry, scores))
        reference_grad = torch.autograd.grad((expected * signal).sum(), (geometry, scores))
        for a, b in zip(actual_grad, reference_grad):
            torch.testing.assert_close(a, b)

    def test_heads_match_independent_single_head_readers_and_gradients(self):
        reader = capacity.MultiHeadAppearance(5, 3, 4).double()
        inputs = [torch.randn(12, d, dtype=torch.double, requires_grad=True)
                  for d in (3, 5, 3, 3, 5)]
        geometry = torch.stack((torch.arange(12), (torch.arange(12) + 6) % 12), -1)
        image = neighborhood.make_image_neighborhood((2, 2, 3), [1, 4], torch.device('cpu'))
        kwargs = dict(chunk_size=3, checkpoint_chunks=True)
        with patch.object(capacity, 'merge_appearance_neighbors', wraps=capacity.merge_appearance_neighbors) as merge:
            actual, report = reader(*inputs, geometry, image, collect_statistics=True, **kwargs)
            self.assertEqual(merge.call_count, 1)
        # Single-head correctness is independently checked against a dense
        # per-slot formula by test_appearance_2d; here verify head independence.
        expected = torch.cat([head(*inputs, geometry, image, **kwargs)[0]
                              for head in reader.heads], -1)
        torch.testing.assert_close(actual, expected)
        signal = torch.randn_like(actual)
        targets = (*inputs, *reader.parameters())
        for a, b in zip(torch.autograd.grad((actual * signal).sum(), targets),
                        torch.autograd.grad((expected * signal).sum(), targets)):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
        self.assertGreater(report['head_attention_tv'].item(), 0)
        for index in range(4):
            self.assertIn(f'head_{index}/image_weight', report)
        self.assertTrue(all(not x.requires_grad and torch.isfinite(x) for x in report.values()))

    def test_initial_values_and_original_gradients_match_moment(self):
        options = cfg()
        baseline_cfg = copy.deepcopy(options)
        baseline_cfg.appearance_2d = baseline_cfg.separate_appearance = baseline_cfg.appearance_capacity = False
        torch.manual_seed(53)
        baseline = Decoder(baseline_cfg)
        rng = torch.get_rng_state()
        torch.manual_seed(53)
        model = Decoder(options)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name], parameter, rtol=0, atol=0)
        p, f = [torch.randn(1, 12, d, requires_grad=True) for d in (3, 5)]
        bp, bf = [x.detach().clone().requires_grad_() for x in (p, f)]
        report = {}
        actual = model(p, f, image_shape=(2, 2, 3), rgb=torch.rand(1, 12, 3), diagnostics=report)
        expected = baseline(bp, bf)
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
        self.assertGreater(model.appearance_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.appearance_sh_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(report['extra_sh_rms'].item(), 0)
        self.assertLess(report['weight_tv'].item(), 1e-6)

    def test_active_model_keeps_geometry_and_checkpointed_gradients(self):
        model = Decoder(cfg())
        activate(model)
        ref = copy.deepcopy(model)
        ref.cfg.checkpoint_chunks = False
        ref.cfg.chunk_size = 1000
        p, f, rgb = [torch.randn(2, 12, d, requires_grad=True) for d in (3, 5, 3)]
        rp, rf, rr = [x.detach().clone().requires_grad_() for x in (p, f, rgb)]
        report = {}
        with patch.object(moment, 'build_knn', wraps=moment.build_knn) as search:
            actual = model(p, f, image_shape=(2, 2, 3), rgb=rgb, diagnostics=report)
            self.assertEqual(search.call_count, 2)
        expected = ref(rp, rf, image_shape=(2, 2, 3), rgb=rr)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [(p, rp), (f, rf), (rgb, rr), *zip(model.parameters(), ref.parameters())]:
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertGreater(a.grad.abs().sum().item(), 0)
            torch.testing.assert_close(a.grad, b.grad, rtol=8e-4, atol=8e-6)
        self.assertGreater(report['weight_tv'].item(), 0)
        self.assertGreater(report['extra_sh_rms'].item(), 0)
        plain_cfg = copy.deepcopy(model.cfg)
        plain_cfg.appearance_2d = plain_cfg.appearance_capacity = plain_cfg.separate_appearance = False
        plain = Decoder(plain_cfg)
        plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('appearance_')})
        base = plain(p.detach(), f.detach())
        for a, b in ((actual.means, base.means), (actual.covariances, base.covariances),
                     (actual.opacities, base.opacities)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_capacity_uses_image_only_supports_and_handles_empty_graph(self):
        model = Decoder(cfg(num_neighbors=1))
        activate(model)
        p = torch.stack((torch.arange(9) * 100., torch.zeros(9), torch.ones(9)), -1)[None]
        f, rgb = torch.randn(1, 9, 5, requires_grad=True), torch.rand(1, 9, 3, requires_grad=True)
        result = model(p, f, image_shape=(1, 3, 3), rgb=rgb)
        gf, gr = torch.autograd.grad(result.harmonics[0, 4].square().sum(), (f, rgb))
        self.assertGreater(gf[0, 0].abs().sum().item(), 0)
        self.assertGreater(gr[0, 0].abs().sum().item(), 0)
        for radii in ([], [1, 4]):
            one = Decoder(cfg(appearance_2d_radii=radii))
            report = {}
            result = one(torch.randn(1, 1, 3), torch.randn(1, 1, 5),
                         image_shape=(1, 1, 1), rgb=torch.rand(1, 1, 3), diagnostics=report)
            self.assertTrue(all(torch.isfinite(x).all() for x in attributes(result)))
            self.assertEqual(report['image_weight'].item(), 0)
            self.assertEqual(report['head_attention_tv'].item(), 0)

    def test_experiment_config_and_optimizer_include_new_heads(self):
        from hydra import compose, initialize_config_dir
        from dacite import from_dict
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            config = compose(config_name='main', overrides=['+experiment=re10k_moment'])
        options = from_dict(Cfg, OmegaConf.to_container(config.model.encoder.moment_decoder))
        self.assertTrue(options.appearance_capacity and options.separate_appearance)
        self.assertEqual((options.feature_dim, options.appearance_heads, options.appearance_dim,
                          options.appearance_mlp_dim), (256, 2, 32, 128))
        self.assertTrue(options.checkpoint_appearance)
        self.assertTrue(options.slot_control.enabled)
        self.assertEqual(sum(p.numel() for p in Decoder(options).parameters()), 167354)
        model = torch.nn.Module()
        model.encoder = torch.nn.Module()
        model.encoder.gaussian_decoder = Decoder(cfg())
        model.encoder.backbone = torch.nn.Linear(2, 2)
        model.optimizer_cfg = SimpleNamespace(lr=1e-4, backbone_lr_multiplier=.1, warm_up_steps=2)
        path = ROOT / 'src/model/model_wrapper.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_optimizers')
        namespace = {'torch': torch, 'get_cfg': lambda: {'trainer': {'max_steps': 100}}}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        optimizer = namespace['configure_optimizers'](model)['optimizer']
        new_params = {id(p) for p in optimizer.param_groups[0]['params']}
        self.assertTrue(all(id(p) in new_params for p in model.encoder.gaussian_decoder.parameters()))

    def test_training_logs_capacity_prefix_without_changing_loss(self):
        path = ROOT / 'src/model/model_wrapper.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
        namespace = {'torch': torch, 'rearrange': rearrange,
                     'compute_psnr': lambda target, image: image.flatten(1).mean(1),
                     'image_error_statistics': old_reader.image_error_statistics}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        logs, requests = [], []

        class Encoder:
            gaussian_decoder = SimpleNamespace(cfg=cfg(log_every_n_steps=50))

            def __call__(self, context, step, visualization_dump=None, diagnostics_dump=None):
                if set(context) != {'image'}:
                    raise AssertionError('Only context images belong in the encoder')
                requests.append(diagnostics_dump is not None)
                if diagnostics_dump is not None:
                    diagnostics_dump['head_attention_tv'] = torch.tensor(.1)
                    diagnostics_dump['extra_sh_rms'] = torch.tensor(.02)

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
        self.assertIn('appearance_capacity/head_attention_tv', logs[0][0])
        self.assertIn('appearance_capacity/edge_mse', logs[0][0])
        self.assertTrue(logs[0][1]['sync_dist'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for device parity')
    def test_cuda_capacity_matches_cpu_values_and_gradients(self):
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
