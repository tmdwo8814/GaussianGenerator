"""Joint BEAM/decoder ablations with a strictly frozen scene encoder."""

import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from test_appearance_capacity import activate, cfg
from test_geometry_readout import tiny_encoder
from test_moment_gaussian_decoder import Decoder, attributes

initialization = importlib.import_module('src.misc.supplementary_ablation')

VARIANTS = {
    'M1': ('uniform', 'RRR'), 'M2': ('uniform', 'FFF'),
    'M3': ('learned', 'RRR'), 'M4': ('learned', 'FFF'),
    'M5': ('learned', 'RFF'), 'M6': ('learned', 'FRF'), 'M7': ('learned', 'FFR'),
}


class SupplementaryAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(509)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'encoder.ckpt'
        self.source = tiny_encoder(cfg())
        with torch.no_grad():
            for p in self.source.gaussian_decoder.parameters():
                p.fill_(.777)
        self.weights = {f'encoder.{name}': value.clone()
                        for name, value in self.source.state_dict().items()}
        torch.save({'state_dict': self.weights}, self.path)
        self.inputs = dict(points=torch.randn(1, 12, 3), features=torch.randn(1, 12, 5),
                           rgb=torch.rand(1, 12, 3), image_shape=(2, 2, 3))

    def encoder(self, variant):
        assignment, mode = VARIANTS[variant]
        torch.manual_seed(211)
        result = tiny_encoder(cfg(frozen_encoder_decoder_only=True, assignment_mode=assignment,
                                  attribute_modes=mode, covariance_coverage=False))
        with contextlib.redirect_stdout(io.StringIO()):
            initialization.initialize_supplementary_ablation(result, str(self.path), 111123)
        return result

    def test_checkpoint_loads_all_frozen_features_but_never_loads_pretrained_decoder(self):
        encoder = self.encoder('M3')
        for name, value in self.source.state_dict().items():
            actual = encoder.state_dict()[name]
            if not name.startswith('gaussian_decoder.'):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(actual, value), name)
        for name, parameter in encoder.named_parameters():
            if not name.startswith('gaussian_decoder.'):
                self.assertFalse(parameter.requires_grad, name)
        encoder.eval().train()
        self.assertFalse(any(module.training for module in encoder.backbone.modules()))
        self.assertFalse(encoder.gaussian_param_head.training)
        self.assertTrue(encoder.gaussian_decoder.appearance_reader.training)

    def test_original_noposplat_output_convolutions_are_discarded_and_missing_frozen_weights_fail(self):
        weights = dict(self.weights)
        weights['encoder.gaussian_param_head.dpt.head.0.weight'] = torch.randn(256, 256, 3, 3)
        weights['encoder.gaussian_param_head2.dpt.head.3.bias'] = torch.randn(83)
        torch.save({'state_dict': weights}, self.path)
        self.encoder('M4')
        del weights['encoder.gaussian_param_head.weight']
        torch.save({'state_dict': weights}, self.path)
        with self.assertRaisesRegex(ValueError, 'gaussian_param_head.weight'):
            self.encoder('M4')

    def test_shared_decoder_initialization_matches_and_partial_heads_reuse_corresponding_rows(self):
        full = self.encoder('M3').gaussian_decoder
        for name in VARIANTS:
            candidate = self.encoder(name).gaussian_decoder
            for key, value in full.state_dict().items():
                if not key.startswith('regression_head.'):
                    torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)
            if candidate.regression_head is not None:
                head = candidate.regression_head
                for key, value in head.readout.state_dict().items():
                    torch.testing.assert_close(value, full.regression_head.readout.state_dict()[key][head.output_indices],
                                               rtol=0, atol=0)

    def test_uniform_changes_only_proportions_and_retains_learned_budget_and_appearance(self):
        encoder = self.encoder('M1')
        model = encoder.gaussian_decoder
        activate(model)
        points, features = self.inputs['points'][0], self.inputs['features'][0]
        neighbors = torch.stack((torch.arange(12), torch.arange(12).roll(1), torch.arange(12).roll(3)), -1)
        q, scores = model.predict_allocation(points, features, neighbors, return_appearance=True)
        budget = model.cfg.budget_max * model.budget_head(features).sigmoid()
        torch.testing.assert_close(q, budget.expand_as(q) / 3)
        torch.testing.assert_close(q.sum(-1), budget.squeeze(-1))
        self.assertTrue(scores.requires_grad)
        (q.square().sum() + scores.square().sum()).backward()
        self.assertGreater(model.budget_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.appearance_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.allocation_head[0].weight.grad.abs().sum().item(), 0)
        self.assertFalse(any(p.requires_grad for p in model.allocation_head[-1].parameters()))

    def test_all_seven_variants_train_beam_budget_color_and_required_regression_outputs(self):
        for name in VARIANTS:
            with self.subTest(variant=name):
                encoder = self.encoder(name)
                model = encoder.gaussian_decoder
                activate(model)
                before = {key: value.clone() for key, value in encoder.state_dict().items()
                          if not key.startswith('gaussian_decoder.')}
                output = model(**self.inputs)
                for value in attributes(output):
                    self.assertTrue(torch.isfinite(value).all())
                self.assertTrue((torch.linalg.eigvalsh(output.covariances) > 0).all())
                loss = (output.means.square().mean() + output.covariances.square().mean()
                        + output.opacities.square().mean() + output.harmonics.square().mean())
                loss.backward()
                for key, parameter in encoder.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, key)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                    else:
                        self.assertIsNone(parameter.grad, key)
                self.assertGreater(model.budget_head.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.appearance_head.weight.grad.abs().sum().item(), 0)
                if VARIANTS[name][0] == 'learned':
                    self.assertGreater(model.allocation_head[-1].weight.grad.abs().sum().item(), 0)
                optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=1e-4)
                optimizer.step()
                for key, value in before.items():
                    torch.testing.assert_close(encoder.state_dict()[key], value, rtol=0, atol=0)

    def test_leave_one_out_selects_attributes_independently_of_regressed_center(self):
        fitted_mean = torch.randn(7, 3)
        fitted_covariance = torch.eye(3).expand(7, 3, 3) * .7
        points, pooled, mass = torch.randn(7, 3), torch.randn(7, 5), torch.rand(7)
        for name in ('M3', 'M4', 'M5', 'M6', 'M7'):
            with self.subTest(variant=name):
                model = self.encoder(name).gaussian_decoder
                actual = model.select_supplementary_attributes(points, mass, fitted_mean, fitted_covariance, pooled)
                reference = (fitted_mean, fitted_covariance, -torch.expm1(-mass))
                for index, mode in enumerate(model.cfg.attribute_modes):
                    if mode == 'F':
                        torch.testing.assert_close(actual[index], reference[index], rtol=0, atol=0)
                if name == 'M5':
                    with torch.no_grad():
                        model.regression_head.readout.bias.add_(20.)
                    changed = model.select_supplementary_attributes(points, mass, fitted_mean, fitted_covariance, pooled)
                    self.assertGreater((changed[0] - actual[0]).abs().max().item(), 10)
                    torch.testing.assert_close(changed[1], actual[1], rtol=0, atol=0)
                    torch.testing.assert_close(changed[2], actual[2], rtol=0, atol=0)

    def test_partial_heads_allocate_only_required_outputs_and_paper_covariance_has_no_learned_scale(self):
        for name, outputs in (('M3', 10), ('M5', 3), ('M6', 6), ('M7', 1)):
            model = self.encoder(name).gaussian_decoder
            self.assertEqual(model.regression_head.readout.out_features, outputs)
            self.assertEqual(sum(p.numel() for p in model.regression_head.parameters()), (5 + 1) * outputs)
            self.assertFalse(any(p.requires_grad for p in model.scale_head.parameters()))
        model = self.encoder('M4').gaussian_decoder
        with torch.no_grad():
            first = model(**self.inputs)
            model.scale_head.weight.add_(100.)
            model.scale_head.bias.add_(100.)
            second = model(**self.inputs)
        for a, b in zip(attributes(first), attributes(second)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_seven_server_configs_share_protocol_and_match_the_experiment_matrix(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from test_moment_gaussian_decoder import ROOT, Cfg

        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            reference = compose(config_name='main', overrides=['+experiment=re10k_moment'])
            configs = {name: compose(config_name='main', overrides=[f'+experiment=re10k_supp_{name.lower()}'])
                       for name in VARIANTS}
        for name, config in configs.items():
            with self.subTest(variant=name):
                decoder = from_dict(Cfg, OmegaConf.to_container(config.model.encoder.moment_decoder))
                self.assertEqual((decoder.assignment_mode, decoder.attribute_modes), VARIANTS[name])
                self.assertTrue(decoder.frozen_encoder_decoder_only)
                self.assertFalse(decoder.geometry_readout_only)
                self.assertFalse(decoder.covariance_coverage)
                self.assertEqual(config.model.encoder.pretrained_weights,
                                 '/workspace/research/GaussianGenerator/pretrained_weights/re10k.ckpt')
                self.assertEqual(config.trainer.devices, 1)
                self.assertEqual(config.data_loader.train.batch_size, 16)
                self.assertEqual(config.trainer.max_steps, 10000)
                self.assertEqual(config.trainer.eval_every_n_steps, 5000)
                self.assertTrue(config.trainer.auto_eval)
                self.assertEqual(config.checkpointing.every_n_train_steps, 5000)
                self.assertEqual(config.checkpointing.save_top_k, -1)
                self.assertFalse(config.checkpointing.save_weights_only)
                self.assertIsNone(config.checkpointing.load)
                self.assertEqual(config.data_loader, reference.data_loader)
                self.assertEqual(config.dataset.re10k.roots, reference.dataset.re10k.roots)
                self.assertEqual(config.test, reference.test)
                self.assertEqual(config.loss, reference.loss)
                self.assertEqual(config.trainer.auto_eval_index_path, reference.trainer.auto_eval_index_path)


if __name__ == '__main__':
    unittest.main()
