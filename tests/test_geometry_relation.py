"""Final++rel: zero-start parity, explicit relation math and stable gradients."""

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_appearance_capacity import activate, cfg
from test_moment_gaussian_decoder import ROOT, Decoder, attributes, build_knn, moment


def relation_cfg(**overrides):
    values = dict(geometry_conditioned_relation=True, appearance_moments=True,
                  appearance_mlp_depth=2, appearance_mlp_dim=12, appearance_heads=2,
                  batch_appearance_projections=True)
    values.update(overrides)
    return cfg(**values)


def dense_allocation(model, points, features, neighbors):
    """Independent full-edge formula, without production projection caching."""
    first = model.allocation_head[0]
    dim, hidden = model.cfg.feature_dim, model.cfg.hidden_dim
    fi = features[neighbors]
    fj = features[:, None].expand_as(fi)
    delta = points[:, None] - points[neighbors]
    distance_squared = delta.square().sum(-1, keepdim=True)
    geometry = torch.cat((delta, distance_squared), -1)
    rho2 = distance_squared.mean(1, keepdim=True).clamp_min(model.cfg.scene_epsilon ** 2)
    local = torch.cat((delta / rho2.sqrt(), distance_squared / rho2), -1)
    u = F.layer_norm(F.linear(fi, first.weight[:, :dim]), (hidden,), eps=1e-5)
    v = F.layer_norm(F.linear(fj, first.weight[:, dim:2 * dim]), (hidden,), eps=1e-5)
    gate = F.linear(local, model.geometry_relation.gate.weight,
                    model.geometry_relation.gate.bias).tanh()
    base = first(torch.cat((fi, fj, geometry), -1))
    h = F.silu(base + gate * u * v)
    q = model.allocation_head[2](h).squeeze(-1).softmax(-1)
    q = q * model.cfg.budget_max * model.budget_head(features).sigmoid()
    return q, model.appearance_head(h).squeeze(-1)


class GeometryRelationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(541)
        torch.set_num_threads(1)

    def test_zero_gate_preserves_final_plus_initialization_outputs_and_base_gradients(self):
        torch.manual_seed(541)
        final = Decoder(relation_cfg(geometry_conditioned_relation=False))
        rng = torch.random.get_rng_state()
        torch.manual_seed(541)
        relation = Decoder(relation_cfg())
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in final.state_dict().items():
            torch.testing.assert_close(value, relation.state_dict()[name], rtol=0, atol=0)
        # Also verify parity with nonzero trained appearance heads.
        activate(final)
        loaded = relation.load_state_dict(final.state_dict(), strict=False)
        self.assertEqual(set(loaded.missing_keys),
                         {'geometry_relation.gate.weight', 'geometry_relation.gate.bias'})
        self.assertFalse(loaded.unexpected_keys)
        inputs = [torch.randn(1, 12, d, requires_grad=True) for d in (3, 5, 3)]
        other = [value.detach().clone().requires_grad_() for value in inputs]
        report = {}
        actual = relation(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3),
                          diagnostics=report)
        expected = final(other[0], other[1], rgb=other[2], image_shape=(2, 2, 3))
        signals = [torch.randn_like(value) for value in attributes(actual)]
        for a, b in zip(attributes(actual), attributes(expected)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in ('relation_gate_rms', 'relation_update_rms', 'relation_relative_update'):
            self.assertEqual(report[key].item(), 0)
        sum((value * signal).sum() for value, signal in zip(attributes(actual), signals)).backward()
        sum((value * signal).sum() for value, signal in zip(attributes(expected), signals)).backward()
        for a, b in zip(inputs, other):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)
        for name, parameter in final.named_parameters():
            torch.testing.assert_close(dict(relation.named_parameters())[name].grad, parameter.grad,
                                       rtol=1e-6, atol=1e-7)
        for parameter in relation.geometry_relation.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_nonzero_relation_matches_dense_values_gradients_and_budget(self):
        for checkpoint in (False, True):
            model = Decoder(relation_cfg(checkpoint_chunks=checkpoint, chunk_size=4)).double()
            for head in (model.allocation_head, model.appearance_head, model.geometry_relation):
                for parameter in head.parameters():
                    torch.nn.init.normal_(parameter, std=.15)
            reference = copy.deepcopy(model)
            points = torch.randn(13, 3, dtype=torch.double, requires_grad=True)
            features = torch.randn(13, 5, dtype=torch.double, requires_grad=True)
            rp, rf = points.detach().clone().requires_grad_(), features.detach().clone().requires_grad_()
            neighbors = build_knn(points, 4)
            report = {}
            actual = model.predict_allocation(points, features, neighbors, True, report)
            expected = dense_allocation(reference, rp, rf, neighbors)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
            budget = model.cfg.budget_max * model.budget_head(features).sigmoid().squeeze(-1)
            torch.testing.assert_close(actual[0].sum(-1), budget)
            signals = [torch.randn_like(value) for value in actual]
            sum((a * s).sum() for a, s in zip(actual, signals)).backward()
            sum((a * s).sum() for a, s in zip(expected, signals)).backward()
            for a, b in [*zip((points, features), (rp, rf)),
                         *zip(model.parameters(), reference.parameters())]:
                if b.grad is None:
                    self.assertIsNone(a.grad)
                else:
                    self.assertTrue(torch.isfinite(a.grad).all())
                    torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-11)
            self.assertGreater(report['relation_update_rms'].item(), 0)
            self.assertTrue(all(not value.requires_grad and torch.isfinite(value)
                                for value in report.values()))

    def compare_full_checkpointing(self, device, compiled=False):
        model = Decoder(relation_cfg()).to(device)
        activate(model)
        torch.nn.init.normal_(model.geometry_relation.gate.weight, std=.15)
        torch.nn.init.normal_(model.geometry_relation.gate.bias, std=.05)
        reference = copy.deepcopy(model)
        model.cfg.compile_kernels = compiled
        reference.cfg.checkpoint_chunks = False
        reference.cfg.checkpoint_appearance = False
        reference.cfg.chunk_size = 30
        inputs = [torch.randn(2, 12, d, device=device, requires_grad=True) for d in (3, 5, 3)]
        other = [value.detach().clone().requires_grad_() for value in inputs]
        report, expected_report = {}, {}
        with patch.object(moment, 'build_knn', wraps=moment.build_knn) as search:
            a = model(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3), diagnostics=report)
            sum(value.square().mean() for value in attributes(a)).backward()
            self.assertEqual(search.call_count, 2)
        b = reference(other[0], other[1], rgb=other[2], image_shape=(2, 2, 3),
                      diagnostics=expected_report)
        sum(value.square().mean() for value in attributes(b)).backward()
        for x, y in zip(attributes(a), attributes(b)):
            torch.testing.assert_close(x, y, rtol=5e-4, atol=5e-5)
        self.assertEqual(set(report), set(expected_report))
        for key in report:
            self.assertFalse(report[key].requires_grad)
            torch.testing.assert_close(report[key], expected_report[key], rtol=5e-4, atol=5e-5)
        for x, y in [*zip(inputs, other), *zip(model.parameters(), reference.parameters())]:
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-3, atol=2e-4)
        model.eval()
        with torch.inference_mode():
            inferred = model(inputs[0], inputs[1], rgb=inputs[2], image_shape=(2, 2, 3))
        for x, y in zip(attributes(a), attributes(inferred)):
            torch.testing.assert_close(x, y, rtol=5e-4, atol=5e-5)

    def test_full_checkpointing_and_inference(self):
        self.compare_full_checkpointing('cpu')

    def test_coincident_single_and_nearly_coincident_points_have_finite_gradients(self):
        for count, scale in ((1, 0), (7, 0), (7, 1e-10)):
            model = Decoder(relation_cfg())
            torch.nn.init.normal_(model.geometry_relation.gate.weight, std=.15)
            torch.nn.init.normal_(model.geometry_relation.gate.bias, std=.05)
            points = (torch.randn(1, count, 3) * scale).requires_grad_()
            features = torch.randn(1, count, 5, requires_grad=True)
            rgb = torch.rand(1, count, 3, requires_grad=True)
            result = model(points, features, rgb=rgb, image_shape=(1, 1, count))
            self.assertTrue(all(torch.isfinite(value).all() for value in attributes(result)))
            sum(value.square().mean() for value in attributes(result)).backward()
            for value in (points, features, rgb, *model.parameters()):
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_non_appearance_path_and_optional_statistics(self):
        model = Decoder(relation_cfg(appearance_moments=False, appearance_capacity=False,
                                     appearance_2d=False, separate_appearance=False))
        torch.nn.init.normal_(model.geometry_relation.gate.weight, std=.15)
        points, features = torch.randn(1, 7, 3), torch.randn(1, 7, 5)
        report = {}
        a = model(points, features, diagnostics=report)
        b = model(points, features)
        for x, y in zip(attributes(a), attributes(b)):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertGreater(report['relation_update_rms'].item(), 0)

    def test_composed_configs_preserve_final_plus_and_add_only_320_parameters(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        for experiment in ('re10k_moment', 'acid_moment'):
            with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
                config = compose(config_name='main', overrides=[f'+experiment={experiment}'])
            options = from_dict(moment.MomentDecoderCfg,
                                OmegaConf.to_container(config.model.encoder.moment_decoder))
            self.assertTrue(options.geometry_conditioned_relation and options.appearance_moments)
            self.assertEqual((options.appearance_mlp_dim, options.appearance_mlp_depth), (256, 2))
            model = Decoder(options)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 336985)
            self.assertEqual(sum(p.numel() for p in model.geometry_relation.parameters()), 320)
            options.geometry_conditioned_relation = False
            base = Decoder(options)
            self.assertEqual(sum(p.numel() for p in base.parameters()), 336665)
            self.assertFalse(any('geometry_relation' in name for name in base.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for compiled appearance checks')
    def test_cuda_existing_compilation_with_new_relation(self):
        self.compare_full_checkpointing('cuda', compiled=True)


if __name__ == '__main__':
    unittest.main()
