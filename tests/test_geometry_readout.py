"""Frozen Final++ evidence, geometry gradients, strict init and saved-run replay.

Run: python -m unittest discover -s tests -p test_geometry_readout.py -v
Uses the same CPU-only production-module loader as the existing decoder tests.
"""

import ast
import contextlib
import copy
import importlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from test_appearance_capacity import activate, cfg
from test_moment_gaussian_decoder import ROOT, Decoder, attributes

initialization = importlib.import_module('src.misc.geometry_readout')


def tiny_encoder(decoder_cfg):
    # Exercise the production train() override, including Lightning's repeated
    # train/eval transitions, without importing the real backbone or renderer.
    path = ROOT / 'src/model/encoder/encoder_noposplat.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'EncoderNoPoSplat')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
    cls.body = [method]
    cls.bases = [ast.Attribute(value=ast.Name(id='nn', ctx=ast.Load()), attr='Module', ctx=ast.Load())]
    namespace = {'nn': nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    encoder = namespace['EncoderNoPoSplat']()
    encoder.gs_params_head_type = 'moment'
    encoder.cfg = SimpleNamespace(moment_decoder=decoder_cfg)
    encoder.backbone = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Dropout(.8))
    encoder.gaussian_param_head = nn.Linear(3, 5)
    encoder.gaussian_param_head2 = nn.Linear(3, 5)
    encoder.gaussian_decoder = Decoder(decoder_cfg)
    return encoder


class GeometryReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / 'final_plus.ckpt'
        self.source = tiny_encoder(cfg(appearance_moments=True, appearance_mlp_depth=2))
        activate(self.source.gaussian_decoder)
        self.source.eval()
        self.state = {f'encoder.{key}': value.clone() for key, value in self.source.state_dict().items()}
        torch.save({'state_dict': self.state, 'global_step': 140000}, self.checkpoint)
        self.inputs = dict(points=torch.randn(2, 12, 3), features=torch.randn(2, 12, 5),
                           rgb=torch.rand(2, 12, 3), image_shape=(2, 2, 3))

    def initialized(self, mode='regression', **overrides):
        decoder_cfg = cfg(appearance_moments=True, appearance_mlp_depth=2,
                          geometry_readout_only=True, geometry_mode=mode, **overrides)
        encoder = tiny_encoder(decoder_cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            initialization.initialize_geometry_readout(encoder, str(self.checkpoint), 111123)
        return encoder

    def test_shared_weights_loaded_exactly_and_only_selected_readout_resets(self):
        regression = self.initialized()
        egf = self.initialized('egf')
        for encoder in (regression, egf):
            for key, value in self.source.state_dict().items():
                if encoder is regression or not key.startswith('gaussian_decoder.scale_head.'):
                    torch.testing.assert_close(encoder.state_dict()[key], value, rtol=0, atol=0)
        self.assertFalse(torch.equal(egf.gaussian_decoder.scale_head.weight,
                                     self.source.gaussian_decoder.scale_head.weight))
        trainable = {name for name, p in regression.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertEqual(trainable, {'gaussian_decoder.regression_head.readout.weight',
                                     'gaussian_decoder.regression_head.readout.bias'})
        self.assertEqual(sum(p.numel() for p in egf.parameters() if p.requires_grad), 6)

    def test_production_width_uses_one_linear_and_exactly_2570_trainable_parameters(self):
        model = Decoder(cfg(feature_dim=256, geometry_readout_only=True, geometry_mode='regression'))
        linears = [module for module in model.regression_head.modules() if isinstance(module, nn.Linear)]
        self.assertEqual(len(linears), 1)
        self.assertEqual((linears[0].in_features, linears[0].out_features), (256, 10))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 2570)
        self.assertFalse(any(p.requires_grad for p in model.scale_head.parameters()))
        self.assertFalse(hasattr(model.regression_head, 'evidence_mlp'))

    def test_train_eval_transitions_keep_frozen_modules_in_eval(self):
        model = self.initialized()
        for _ in range(2):
            model.eval()
            self.assertFalse(any(module.training for module in model.modules()))
            model.train()
            self.assertFalse(any(module.training for module in model.backbone.modules()))
            self.assertFalse(model.gaussian_param_head.training)
            self.assertFalse(model.gaussian_decoder.appearance_reader.training)
            self.assertFalse(model.gaussian_decoder.allocation_head.training)
            self.assertFalse(model.gaussian_decoder.scale_head.training)
            self.assertTrue(model.gaussian_decoder.regression_head.training)

    def test_regression_changes_geometry_and_opacity_but_preserves_color(self):
        reg = self.initialized().gaussian_decoder
        egf = self.initialized('egf').gaussian_decoder
        with torch.no_grad():
            source = self.source.gaussian_decoder(**self.inputs)
            a, b = reg(**self.inputs), egf(**self.inputs)
            torch.testing.assert_close(b.means, source.means, rtol=0, atol=0)
            torch.testing.assert_close(a.harmonics, source.harmonics, rtol=0, atol=0)
            torch.testing.assert_close(b.harmonics, source.harmonics, rtol=0, atol=0)
            torch.testing.assert_close(b.opacities, source.opacities, rtol=0, atol=0)
            self.assertGreater((a.means - b.means).abs().max().item(), 1e-4)
            self.assertGreater((a.covariances - b.covariances).abs().max().item(), 1e-4)
            self.assertGreater((a.opacities - b.opacities).abs().max().item(), 1e-4)
            reg.regression_head.readout.bias[0].add_(1.)
            changed = reg(**self.inputs)
            torch.testing.assert_close(changed.harmonics, a.harmonics, rtol=0, atol=0)
            torch.testing.assert_close(changed.opacities, a.opacities, rtol=0, atol=0)
            for value in attributes(changed):
                self.assertTrue(torch.isfinite(value).all())
            self.assertTrue((torch.linalg.eigvalsh(changed.covariances) > 0).all())
            self.assertTrue(((changed.opacities > 0) & (changed.opacities < 1)).all())

    def test_opacity_has_its_own_learned_output_and_scale_head_is_unused(self):
        model = self.initialized().gaussian_decoder
        with torch.no_grad():
            original = model(**self.inputs)
            model.regression_head.readout.bias[9].add_(1.)
            changed = model(**self.inputs)
            self.assertGreater((changed.opacities - original.opacities).abs().max().item(), .1)
            for name in ('means', 'covariances', 'harmonics'):
                torch.testing.assert_close(getattr(changed, name), getattr(original, name), rtol=0, atol=0)
            model.scale_head.weight.add_(100.)
            model.scale_head.bias.add_(100.)
            same = model(**self.inputs)
            for a, b in zip(attributes(changed), attributes(same)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_only_geometry_parameters_update_and_inputs_are_detached(self):
        for mode in ('regression', 'egf'):
            with self.subTest(mode=mode):
                encoder = self.initialized(mode)
                model = encoder.gaussian_decoder
                before = {name: value.clone() for name, value in encoder.state_dict().items()}
                inputs = {key: value.clone().requires_grad_() if isinstance(value, torch.Tensor) else value
                          for key, value in self.inputs.items()}
                result = model(**inputs)
                loss = (result.means.square().mean() + 1000 * result.covariances.square().mean()
                        + (result.opacities - .2).square().mean())
                loss.backward()
                for name, parameter in encoder.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                self.assertTrue(all(inputs[key].grad is None for key in ('points', 'features', 'rgb')))
                optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=1e-3)
                optimizer.step()
                for name, value in encoder.state_dict().items():
                    train_prefix = ('gaussian_decoder.regression_head.' if mode == 'regression'
                                    else 'gaussian_decoder.scale_head.')
                    if not name.startswith(train_prefix):
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                changed_name = train_prefix + ('readout.weight' if mode == 'regression' else 'weight')
                self.assertFalse(torch.equal(encoder.state_dict()[changed_name], before[changed_name]))
                if mode == 'regression':
                    self.assertGreater(model.regression_head.readout.weight.grad[9].abs().sum().item(), 0)

    def test_single_linear_reads_existing_incoming_pool_without_another_feature_encoder(self):
        model = self.initialized().gaussian_decoder
        head = copy.deepcopy(model.regression_head).double()
        points, features = torch.randn(4, 3, dtype=torch.double), torch.randn(4, 5, dtype=torch.double)
        neighbors = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])
        q = torch.tensor([[.2, .8], [.3, .7], [.4, .6], [.5, .5]], dtype=torch.double)
        _, _, _, pooled, weights = model.aggregate_moments(points, features, neighbors, q,
                                                          return_weights=True)
        observed = []
        handle = head.readout.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach()))
        head(points, pooled)
        handle.remove()
        expected = []
        for slot in range(4):
            source, edge = (neighbors == slot).nonzero(as_tuple=True)
            expected.append((weights[source, edge, None] * features[source]).sum(0))
        actual = torch.cat(observed)
        torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-10, atol=1e-12)

    def test_checkpointed_chunks_match_eager_outputs_and_gradients(self):
        a = self.initialized(checkpoint_chunks=False).gaussian_decoder
        b = copy.deepcopy(a)
        b.cfg.checkpoint_chunks = True
        b.cfg.chunk_size = 5
        outputs = []
        for model in (a, b):
            value = model(**self.inputs)
            (value.means.square().mean() + value.covariances.square().mean()
             + value.opacities.square().mean()).backward()
            outputs.append(value)
        for x, y in zip(attributes(outputs[0]), attributes(outputs[1])):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-7, msg=name)

    def test_initialization_rejects_incomplete_or_wrong_checkpoint(self):
        encoder = tiny_encoder(cfg(geometry_readout_only=True))
        broken = dict(self.state)
        del broken['encoder.gaussian_decoder.budget_head.weight']
        torch.save({'state_dict': broken}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'budget_head.weight'):
            initialization.initialize_geometry_readout(encoder, str(self.checkpoint), 7)
        torch.save({'model': {}}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, r'Final\+\+'):
            initialization.initialize_geometry_readout(encoder, str(self.checkpoint), 7)

    def test_saved_regression_checkpoint_replays_without_reset(self):
        model = self.initialized().gaussian_decoder
        with torch.no_grad():
            model.regression_head.readout.bias.add_(.2)
            expected = model(**self.inputs)
        restored = Decoder(copy.deepcopy(model.cfg))
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.eval()
        with torch.no_grad():
            actual = restored(**self.inputs)
        for x, y in zip(attributes(expected), attributes(actual)):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_experiment_configs_keep_existing_splits_and_evaluation(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from test_moment_gaussian_decoder import Cfg

        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            original = compose(config_name='main', overrides=['+experiment=re10k_moment'])
            configs = [compose(config_name='main', overrides=[f'+experiment={name}'])
                       for name in ('re10k_regression', 're10k_egf_readout')]
        for config in configs:
            decoder = from_dict(Cfg, OmegaConf.to_container(config.model.encoder.moment_decoder))
            self.assertTrue(decoder.geometry_readout_only)
            self.assertEqual(config.dataset.re10k.roots, original.dataset.re10k.roots)
            self.assertEqual(config.data_loader, original.data_loader)
            self.assertEqual(config.test, original.test)
            self.assertEqual(config.loss, original.loss)
            regression = decoder.geometry_mode == 'regression'
            self.assertEqual(config.trainer.max_steps, 10000 if regression else 30000)
            self.assertEqual(config.trainer.eval_every_n_steps, 5000 if regression else 6000)
            self.assertEqual(config.checkpointing.every_n_train_steps, 5000 if regression else 6000)
            if regression:
                self.assertEqual(config.trainer.devices, 4)
                steps = list(range(config.trainer.eval_every_n_steps, config.trainer.max_steps + 1,
                                   config.trainer.eval_every_n_steps))
                self.assertEqual(steps, [5000, 10000])
            self.assertIsNone(config.checkpointing.load)
            self.assertEqual(config.trainer.auto_eval_index_path, original.trainer.auto_eval_index_path)
            self.assertEqual(config.dataset.re10k.view_sampler.warm_up_steps, 0)
        self.assertEqual(configs[0].model.encoder.moment_decoder.geometry_mode, 'regression')
        self.assertEqual(configs[1].model.encoder.moment_decoder.geometry_mode, 'egf')


if __name__ == '__main__':
    unittest.main()
