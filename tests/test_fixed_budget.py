"""Fixed-budget ablation: constant outgoing mass, learned allocation, full gradients."""

import ast
import copy
from math import log
from types import SimpleNamespace
import unittest

import torch
from dacite import from_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from test_moment_gaussian_decoder import Cfg, Decoder, ROOT, attributes, build_knn


def capacity_cfg(**overrides):
    options = dict(
        feature_dim=5, hidden_dim=7, num_neighbors=4, chunk_size=7,
        budget_mode='fixed', budget_max=1.0,
        separate_appearance=True, appearance_2d=True, appearance_capacity=True,
        appearance_heads=2, appearance_dim=3, appearance_mlp_dim=12,
        appearance_mlp_depth=2, appearance_moments=True,
        appearance_2d_radii=[1], batch_appearance_projections=True,
    )
    return Cfg(**(options | overrides))


class FixedBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_constant_budget_keeps_learned_nonuniform_allocation_and_gradients(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=7, budget_mode='fixed',
                            budget_max=1.0, chunk_size=3))
        with torch.no_grad():
            model.allocation_head[-1].weight.normal_()
            model.budget_head.weight.normal_()
            model.budget_head.bias.fill_(-10)
        points = torch.randn(9, 3, requires_grad=True)
        features = torch.randn(9, 5, requires_grad=True)
        neighbors = build_knn(points.detach(), 4)
        q = model.predict_allocation(points, features, neighbors)
        torch.testing.assert_close(q.sum(-1), torch.full((9,), log(2.0)))
        self.assertGreater((q - log(2.0) / 4).abs().max().item(), .01)
        with torch.no_grad():
            model.budget_head.weight.fill_(100)
            model.budget_head.bias.fill_(100)
        torch.testing.assert_close(model.predict_allocation(points, features, neighbors), q,
                                   rtol=0, atol=0)
        mass, *_ = model.aggregate_moments(points, features, neighbors, q)
        torch.testing.assert_close(mass.sum(), torch.tensor(9 * log(2.0)))
        (q * torch.randn_like(q)).sum().backward()
        for tensor in (points, features, model.allocation_head[0].weight,
                       model.allocation_head[-1].weight):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for p in model.budget_head.parameters()))

    def test_initialization_checkpoint_keys_and_initial_gaussians_match_final_plus(self):
        torch.manual_seed(123)
        learned = Decoder(capacity_cfg(budget_mode='support'))
        learned_rng = torch.get_rng_state()
        torch.manual_seed(123)
        fixed = Decoder(capacity_cfg())
        self.assertTrue(torch.equal(torch.get_rng_state(), learned_rng))
        self.assertEqual(list(learned.state_dict()), list(fixed.state_dict()))
        for key, value in learned.state_dict().items():
            torch.testing.assert_close(fixed.state_dict()[key], value, rtol=0, atol=0)
        fixed.load_state_dict(learned.state_dict(), strict=True)
        points, features = torch.randn(2, 12, 3), torch.randn(2, 12, 5)
        kwargs = dict(image_shape=(2, 2, 3), rgb=torch.rand(2, 12, 3))
        with torch.no_grad():
            actual, expected = fixed(points, features, **kwargs), learned(points, features, **kwargs)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        self.assertEqual(sum(p.numel() for p in learned.parameters() if p.requires_grad)
                         - sum(p.numel() for p in fixed.parameters() if p.requires_grad), 6)

    def test_capacity_backward_updates_all_trainable_heads_but_budget_stays_fixed(self):
        model = Decoder(capacity_cfg())
        with torch.no_grad():
            model.appearance_sh_head[-1].weight.normal_(std=.02)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        points = torch.randn(2, 12, 3, requires_grad=True)
        features = torch.randn(2, 12, 5, requires_grad=True)
        rgb = torch.rand(2, 12, 3)
        initial_head = copy.deepcopy(model.budget_head.state_dict())
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            points.grad = features.grad = None
            result = model(points, features, image_shape=(2, 2, 3), rgb=rgb)
            sum(value.square().mean() for value in attributes(result)).backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertTrue(name.startswith('budget_head.'), name)
                    self.assertIsNone(parameter.grad, name)
            for tensor in (points, features):
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(tensor.grad.abs().sum().item(), 0)
            optimizer.step()
            for scene in range(2):
                neighbors = build_knn(points[scene].detach(), 4)
                q = model.predict_allocation(points[scene], features[scene], neighbors)
                torch.testing.assert_close(q.sum(-1), torch.full((12,), log(2.0)))
        for key, value in initial_head.items():
            torch.testing.assert_close(model.budget_head.state_dict()[key], value, rtol=0, atol=0)

    def test_chunking_and_checkpointing_preserve_fixed_outputs_and_gradients(self):
        model = Decoder(capacity_cfg())
        reference = copy.deepcopy(model)
        reference.cfg.chunk_size = 100
        reference.cfg.checkpoint_chunks = False
        reference.cfg.checkpoint_appearance = False
        points = torch.randn(1, 12, 3, requires_grad=True)
        features = torch.randn(1, 12, 5, requires_grad=True)
        rp, rf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
        kwargs = dict(image_shape=(2, 2, 3), rgb=torch.rand(1, 12, 3))
        actual, expected = model(points, features, **kwargs), reference(rp, rf, **kwargs)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [(points, rp), (features, rf), *zip(model.parameters(), reference.parameters())]:
            if not a.requires_grad:
                self.assertIsNone(a.grad)
                self.assertIsNone(b.grad)
            else:
                self.assertIsNotNone(a.grad)
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)

    def test_re10k_and_acid_enable_fixed_without_changing_training(self):
        for dataset in ('re10k', 'acid'):
            with self.subTest(dataset=dataset), initialize_config_dir(
                    config_dir=str(ROOT / 'config'), version_base=None):
                base = compose(config_name='main', overrides=[f'+experiment={dataset}'])
                config = compose(config_name='main', overrides=[f'+experiment={dataset}_moment'])
                options = from_dict(Cfg, OmegaConf.to_container(config.model.encoder.moment_decoder))
                self.assertEqual(options.budget_mode, 'fixed')
                self.assertEqual(options.budget_init, log(2.0))
                self.assertEqual((options.feature_dim, options.num_neighbors), (256, 16))
                self.assertTrue(options.separate_appearance and options.appearance_2d
                                and options.appearance_capacity and options.appearance_moments)
                for key in ('optimizer', 'loss', 'data_loader', 'checkpointing'):
                    self.assertEqual(OmegaConf.to_container(config[key]), OmegaConf.to_container(base[key]))
                self.assertEqual(config.trainer.max_steps, base.trainer.max_steps)
                self.assertIn('fixed', config.wandb.name)
                # Removing the one option restores the original model's default.
                support_config = compose(config_name='main', overrides=[
                    f'+experiment={dataset}_moment', 'model.encoder.moment_decoder.budget_mode=support'])
                self.assertEqual(support_config.model.encoder.moment_decoder.budget_mode, 'support')

    def test_production_optimizer_excludes_frozen_budget_head(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Module()
        model.encoder.gaussian_decoder = Decoder(capacity_cfg())
        model.encoder.backbone = torch.nn.Linear(2, 2)
        model.optimizer_cfg = SimpleNamespace(lr=1e-4, backbone_lr_multiplier=.1, warm_up_steps=2)
        path = ROOT / 'src/model/model_wrapper.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_optimizers')
        namespace = {'torch': torch, 'get_cfg': lambda: {'trainer': {'max_steps': 100}}}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        optimizer = namespace['configure_optimizers'](model)['optimizer']
        decoder_ids = {id(p) for p in optimizer.param_groups[0]['params']}
        for parameter in model.encoder.gaussian_decoder.parameters():
            self.assertEqual(id(parameter) in decoder_ids, parameter.requires_grad)

    def test_invalid_budget_mode_rejected_and_default_remains_support(self):
        self.assertEqual(Cfg().budget_mode, 'support')
        with self.assertRaisesRegex(ValueError, 'budget_mode'):
            Decoder(Cfg(budget_mode='typo'))


if __name__ == '__main__':
    unittest.main()
