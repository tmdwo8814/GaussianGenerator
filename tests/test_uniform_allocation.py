"""Uniform outgoing allocation with learned budgets and unchanged appearance.

Run: python -m unittest discover -s tests -p test_uniform_allocation.py -v
"""

import copy
import unittest

import torch

from test_appearance_capacity import activate, cfg
from test_moment_gaussian_decoder import ROOT, Cfg, Decoder, attributes, build_knn


class UniformAllocationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(531)
        torch.set_num_threads(1)

    def test_initialization_and_checkpoint_keys_match_learned_mode(self):
        models, draws = [], []
        for mode in ('learned', 'uniform'):
            torch.manual_seed(29)
            models.append(Decoder(cfg(allocation_mode=mode, appearance_moments=True,
                                      appearance_mlp_depth=2)))
            draws.append(torch.rand(5))
        learned, uniform = models
        self.assertEqual(learned.state_dict().keys(), uniform.state_dict().keys())
        for key, value in learned.state_dict().items():
            torch.testing.assert_close(value, uniform.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)
        frozen = {name for name, p in uniform.named_parameters() if not p.requires_grad}
        self.assertEqual(frozen, {'allocation_head.2.weight', 'allocation_head.2.bias'})
        uniform.load_state_dict(learned.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, 'allocation_mode'):
            Decoder(Cfg(allocation_mode='typo'))

    def test_equal_outgoing_mass_and_learned_budget_with_actual_neighbor_count(self):
        points, features = torch.randn(5, 3), torch.randn(5, 5)
        for budget_max in (1., 2., 4.):
            for requested_k in (1, 3, 16):
                with self.subTest(budget_max=budget_max, requested_k=requested_k):
                    model = Decoder(Cfg(feature_dim=5, hidden_dim=8, chunk_size=2,
                                        allocation_mode='uniform', budget_max=budget_max))
                    with torch.no_grad():
                        model.budget_head.weight.normal_(std=.4)
                    neighbors = build_knn(points, requested_k)
                    q = model.predict_allocation(points, features, neighbors)
                    budget = budget_max * model.budget_head(features).sigmoid()
                    self.assertGreater(budget.std().item(), 0)
                    self.assertEqual(q.shape, neighbors.shape)
                    for column in range(neighbors.shape[-1]):
                        torch.testing.assert_close(q[:, column], budget[:, 0] / neighbors.shape[-1])
                    torch.testing.assert_close(q.sum(-1), budget[:, 0])
                    mass, *_ = model.aggregate_moments(points, features, neighbors, q)
                    torch.testing.assert_close(mass.sum(), budget.sum())
                    q.square().sum().backward()
                    self.assertGreater(model.budget_head.weight.grad.abs().sum().item(), 0)
                    for parameter in model.allocation_head.parameters():
                        self.assertIsNone(parameter.grad)

    def test_relation_scores_and_gradients_match_dense_reference(self):
        model = Decoder(cfg(allocation_mode='uniform', chunk_size=2)).double()
        activate(model)
        reference = copy.deepcopy(model)
        p, f = torch.randn(5, 3, dtype=torch.double), torch.randn(5, 5, dtype=torch.double)
        p.requires_grad_()
        f.requires_grad_()
        rp, rf = p.detach().clone().requires_grad_(), f.detach().clone().requires_grad_()
        neighbors = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 0, 3], [3, 0, 4], [4, 0, 1]])
        q, scores = model.predict_allocation(p, f, neighbors, return_appearance=True)
        delta = rp[:, None] - rp[neighbors]
        pair = torch.cat((rf[neighbors], rf[:, None].expand(-1, 3, -1),
                          delta, delta.square().sum(-1, keepdim=True)), dim=-1)
        expected_scores = reference.appearance_head(reference.allocation_head[:2](pair)).squeeze(-1)
        expected_q = (reference.cfg.budget_max * reference.budget_head(rf).sigmoid()).expand(-1, 3) / 3
        torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(scores, expected_scores)
        signals = torch.randn_like(q), torch.randn_like(scores)
        (q.mul(signals[0]).sum() + scores.mul(signals[1]).sum()).backward()
        (expected_q.mul(signals[0]).sum() + expected_scores.mul(signals[1]).sum()).backward()
        pairs = [(p, rp), (f, rf)] + list(zip(model.parameters(), reference.parameters()))
        for actual, expected in pairs:
            if expected.grad is None:
                self.assertIsNone(actual.grad)
            else:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-9, atol=1e-10)
        for parameter in (model.budget_head.weight, model.allocation_head[0].weight,
                          model.appearance_head.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # Changing only the scalar allocation scorer must have no effect.
        with torch.no_grad():
            model.allocation_head[-1].weight.fill_(1000)
            model.allocation_head[-1].bias.fill_(-1000)
        changed_q, changed_scores = model.predict_allocation(p, f, neighbors, return_appearance=True)
        torch.testing.assert_close(changed_q, q, rtol=0, atol=0)
        torch.testing.assert_close(changed_scores, scores, rtol=0, atol=0)

    def test_final_plus_matches_zero_logit_reference_with_checkpointing(self):
        model = Decoder(cfg(allocation_mode='uniform', appearance_heads=2,
                            appearance_moments=True, appearance_mlp_depth=2,
                            batch_appearance_projections=True, chunk_size=4))
        activate(model)
        reference = copy.deepcopy(model)
        reference.cfg.allocation_mode = 'learned'
        reference.cfg.checkpoint_chunks = False
        reference.cfg.checkpoint_appearance = False
        reference.cfg.chunk_size = 99
        # A zero scalar scorer makes the original softmax exactly uniform.
        with torch.no_grad():
            reference.allocation_head[-1].weight.zero_()
            reference.allocation_head[-1].bias.zero_()
        inputs = [torch.randn(2, 12, 3).requires_grad_(),
                  torch.randn(2, 12, 5).requires_grad_(),
                  torch.rand(2, 12, 3).requires_grad_()]
        copies = [x.detach().clone().requires_grad_() for x in inputs]
        stats, ref_stats = {}, {}
        actual = model(*inputs[:2], rgb=inputs[2], image_shape=(2, 2, 3), diagnostics=stats)
        expected = reference(*copies[:2], rgb=copies[2], image_shape=(2, 2, 3), diagnostics=ref_stats)
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
        self.assertEqual(actual.means.shape, (2, 12, 3))
        self.assertEqual(stats.keys(), ref_stats.keys())
        for name in stats:
            torch.testing.assert_close(stats[name], ref_stats[name], rtol=5e-4, atol=5e-5)
        sum(x.square().mean() for x in attributes(actual)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for a, b in [*zip(inputs, copies), *zip(model.parameters(), reference.parameters())]:
            if not a.requires_grad:
                self.assertIsNone(a.grad)
                continue
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad).all())
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-3, atol=2e-4)
        for parameter in (model.budget_head.weight, model.allocation_head[0].weight,
                          model.appearance_head.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        with torch.inference_mode():
            inference = model(*inputs[:2], rgb=inputs[2], image_shape=(2, 2, 3))
        for a, b in zip(attributes(actual), attributes(inference)):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)

    def test_uniform_overrides_preserve_training(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        for dataset in ('re10k', 'acid'):
            with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
                baseline = compose(config_name='main', overrides=[f'+experiment={dataset}'])
                experiment = compose(config_name='main', overrides=[f'+experiment={dataset}_moment',
                                     'model.encoder.moment_decoder.allocation_mode=uniform',
                                     'model.encoder.moment_decoder.budget_mode=support'])
                learned = compose(config_name='main', overrides=[f'+experiment={dataset}_moment',
                                  'model.encoder.moment_decoder.allocation_mode=learned'])
            options = from_dict(Cfg, OmegaConf.to_container(experiment.model.encoder.moment_decoder))
            self.assertEqual(options.allocation_mode, 'uniform')
            self.assertEqual(options.feature_dim, 256)
            self.assertTrue(options.appearance_capacity and options.separate_appearance)
            self.assertTrue(options.appearance_moments)
            self.assertGreater(options.budget_max, options.budget_init)
            self.assertAlmostEqual(options.budget_init, .6931471805599453)
            self.assertEqual(options.budget_mode, 'support')
            self.assertEqual(learned.model.encoder.moment_decoder.allocation_mode, 'learned')
            for key in ('dataset', 'data_loader', 'optimizer', 'loss', 'train', 'checkpointing'):
                self.assertEqual(OmegaConf.to_container(experiment[key]),
                                 OmegaConf.to_container(baseline[key]))
            self.assertEqual(experiment.trainer.max_steps, baseline.trainer.max_steps)


if __name__ == '__main__':
    unittest.main()
