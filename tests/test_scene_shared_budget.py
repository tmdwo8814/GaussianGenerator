"""Scene-local budget averaging with learned allocation and full gradients."""

import copy
import unittest
from unittest.mock import patch

import torch

from test_appearance_capacity import activate, cfg
from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes, build_knn


class SceneSharedBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)
        torch.set_num_threads(1)

    def test_initialization_parameters_and_checkpoint_keys_are_unchanged(self):
        models, draws = [], []
        for mode in ('support', 'scene_shared'):
            torch.manual_seed(29)
            models.append(Decoder(cfg(budget_mode=mode, appearance_moments=True,
                                      appearance_mlp_depth=2)))
            draws.append(torch.rand(5))
        support, shared = models
        self.assertEqual(support.state_dict().keys(), shared.state_dict().keys())
        for name, value in support.state_dict().items():
            torch.testing.assert_close(value, shared.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in shared.parameters()))
        shared.load_state_dict(support.state_dict(), strict=True)
        p, f = torch.randn(7, 3), torch.randn(7, 5)
        neighbors = build_knn(p, 3)
        torch.testing.assert_close(support.predict_allocation(p, f, neighbors),
                                   shared.predict_allocation(p, f, neighbors))
        with self.assertRaisesRegex(ValueError, 'budget_mode'):
            Decoder(Cfg(budget_mode='typo'))

    def test_dense_values_and_gradients_use_mean_after_sigmoid(self):
        model = Decoder(cfg(budget_mode='scene_shared', chunk_size=2)).double()
        activate(model)
        with torch.no_grad():
            model.budget_head.weight.fill_(.8)
            model.budget_head.bias.fill_(.3)
            model.allocation_head[-1].weight.normal_(std=.4)
        reference = copy.deepcopy(model)
        p = torch.randn(7, 3, dtype=torch.double).requires_grad_()
        f = (torch.randn(7, 5, dtype=torch.double) + .5).requires_grad_()
        rp, rf = p.detach().clone().requires_grad_(), f.detach().clone().requires_grad_()
        neighbors = build_knn(p.detach(), 3)
        q, scores = model.predict_allocation(p, f, neighbors, return_appearance=True)
        delta = rp[:, None] - rp[neighbors]
        pair = torch.cat((rf[neighbors], rf[:, None].expand(-1, 3, -1),
                          delta, delta.square().sum(-1, keepdim=True)), dim=-1)
        hidden = reference.allocation_head[:2](pair)
        allocation = reference.allocation_head[-1](hidden).squeeze(-1).softmax(-1)
        raw_logits = reference.budget_head(rf)
        raw_budget = reference.cfg.budget_max * raw_logits.sigmoid()
        expected = allocation * raw_budget.mean()
        expected_scores = reference.appearance_head(hidden).squeeze(-1)
        torch.testing.assert_close(q, expected, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(scores, expected_scores, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(q.sum(-1), raw_budget.mean().expand(len(p)))
        torch.testing.assert_close(q.sum(), raw_budget.sum())
        self.assertGreater((allocation - 1 / 3).abs().max().item(), .01)
        self.assertGreater((raw_budget.mean() - model.cfg.budget_max * raw_logits.mean().sigmoid()).abs().item(), .001)
        signals = torch.randn_like(q), torch.randn_like(scores)
        (q.mul(signals[0]).sum() + scores.mul(signals[1]).sum()).backward()
        (expected.mul(signals[0]).sum() + expected_scores.mul(signals[1]).sum()).backward()
        for actual, target in [(p, rp), (f, rf), *zip(model.parameters(), reference.parameters())]:
            if target.grad is None:
                self.assertIsNone(actual.grad)
            else:
                torch.testing.assert_close(actual.grad, target.grad, rtol=1e-9, atol=1e-10)
        for parameter in (model.budget_head.weight, model.budget_head.bias,
                          model.allocation_head[-1].weight, model.appearance_head.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_one_support_mass_backpropagates_through_every_scene_support(self):
        # This distinguishes a differentiable full-scene mean from detached,
        # per-view or per-chunk means. K=1 removes the allocation dependence.
        model = Decoder(Cfg(feature_dim=2, hidden_dim=4, num_neighbors=1,
                            chunk_size=2, budget_mode='scene_shared')).double()
        with torch.no_grad():
            model.budget_head.weight.copy_(torch.tensor([[.4, -.2]]))
            model.budget_head.bias.fill_(.1)
        p = torch.randn(6, 3, dtype=torch.double)
        f = torch.linspace(-2, 3, 12, dtype=torch.double).reshape(6, 2).requires_grad_()
        q = model.predict_allocation(p, f, torch.arange(6)[:, None])
        q[0, 0].backward()
        sigmoid = model.budget_head(f.detach()).sigmoid()
        expected = model.cfg.budget_max / len(f) * sigmoid * (1 - sigmoid) * model.budget_head.weight
        torch.testing.assert_close(f.grad, expected, rtol=1e-10, atol=1e-11)
        self.assertTrue((f.grad.abs().sum(-1) > 0).all())

    def test_full_decoder_is_scene_local_and_checkpoint_gradients_match(self):
        model = Decoder(cfg(budget_mode='scene_shared', appearance_heads=2,
                            appearance_moments=True, appearance_mlp_depth=2,
                            batch_appearance_projections=True, chunk_size=4))
        activate(model)
        with torch.no_grad():
            model.budget_head.weight.fill_(.2)
        reference = copy.deepcopy(model)
        reference.cfg.checkpoint_chunks = False
        reference.cfg.checkpoint_appearance = False
        reference.cfg.chunk_size = 99
        f = torch.randn(2, 12, 5)
        f[0] -= 1
        f[1] += 1
        inputs = [torch.randn(2, 12, 3).requires_grad_(), f.requires_grad_(),
                  torch.rand(2, 12, 3).requires_grad_()]
        copies = [x.detach().clone().requires_grad_() for x in inputs]
        observed = []
        predict = model.predict_allocation

        def capture(*args, **kwargs):
            result = predict(*args, **kwargs)
            observed.append(result[0].detach().sum(-1))
            return result

        with patch.object(model, 'predict_allocation', side_effect=capture):
            actual = model(*inputs[:2], rgb=inputs[2], image_shape=(2, 2, 3))
        self.assertEqual(len(observed), 2)
        expected_means = (model.cfg.budget_max * model.budget_head(f.detach()).sigmoid()).mean(dim=1)
        for scene, outgoing in enumerate(observed):
            torch.testing.assert_close(outgoing, expected_means[scene].expand(12))
        self.assertGreater((observed[0][0] - observed[1][0]).abs().item(), .1)
        expected = reference(*copies[:2], rgb=copies[2], image_shape=(2, 2, 3))
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [*zip(inputs, copies), *zip(model.parameters(), reference.parameters())]:
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad).all())
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-3, atol=2e-4)
        self.assertGreater(model.budget_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.allocation_head[-1].weight.grad.abs().sum().item(), 0)
        with torch.inference_mode():
            # Evaluate each scene independently: no sharing across batch items.
            separate = [model(inputs[0][s:s+1], inputs[1][s:s+1],
                              rgb=inputs[2][s:s+1], image_shape=(2, 2, 3)) for s in range(2)]
        for index, a in enumerate(attributes(actual)):
            b = torch.cat([attributes(scene)[index] for scene in separate])
            torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)

    def test_configs_enable_scene_shared_with_learned_allocation(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        for dataset in ('re10k', 'acid'):
            with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
                baseline = compose(config_name='main', overrides=[f'+experiment={dataset}'])
                experiment = compose(config_name='main', overrides=[f'+experiment={dataset}_moment'])
            options = from_dict(Cfg, OmegaConf.to_container(experiment.model.encoder.moment_decoder))
            self.assertEqual(options.budget_mode, 'scene_shared')
            self.assertEqual(options.allocation_mode, 'learned')
            self.assertEqual(options.feature_dim, 256)
            self.assertTrue(options.appearance_capacity and options.appearance_moments)
            self.assertTrue(options.separate_appearance and options.appearance_2d)
            self.assertEqual(options.appearance_mlp_depth, 2)
            self.assertGreater(options.budget_max, options.budget_init)
            self.assertAlmostEqual(options.budget_init, .6931471805599453)
            self.assertIn('scene-shared', experiment.wandb.name)
            self.assertIn('scene_shared_budget', experiment.wandb.tags)
            self.assertNotIn('uniform_allocation', experiment.wandb.tags)
            for key in ('dataset', 'data_loader', 'optimizer', 'loss', 'train', 'checkpointing'):
                self.assertEqual(OmegaConf.to_container(experiment[key]),
                                 OmegaConf.to_container(baseline[key]))
            self.assertEqual(experiment.trainer.max_steps, baseline.trainer.max_steps)


if __name__ == '__main__':
    unittest.main()
