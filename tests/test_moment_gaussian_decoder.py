"""CPU checks of the actual decoder, without importing the CUDA renderer.

Run: python -m unittest discover -s tests -p test_moment_gaussian_decoder.py -v
Requires torch, scipy, jaxtyping, beartype, einops, hydra-core and dacite.
"""

import ast
import copy
import importlib
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
from typing import Optional
import unittest

import torch
from einops import rearrange
from jaxtyping import install_import_hook
from torch import nn


ROOT = Path(__file__).resolve().parents[1]

# Import production modules under an isolated package name so encoder/__init__
# does not eagerly import the backbone, Lightning, or CUDA rasterizer.
for name, path in {
    "_moment_test": ROOT / "src/model",
    "_moment_test.encoder": ROOT / "src/model/encoder",
    "_moment_test.encoder.heads": ROOT / "src/model/encoder/heads",
    "_moment_test.encoder.common": ROOT / "src/model/encoder/common",
}.items():
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

with install_import_hook(("_moment_test",), ("beartype", "beartype")):
    moment = importlib.import_module("_moment_test.encoder.heads.moment_gaussian_decoder")
    feature_head = importlib.import_module("_moment_test.encoder.heads.dpt_feature_head")
build_knn = importlib.import_module("_moment_test.encoder.common.sparse_knn").build_knn
Cfg = moment.MomentDecoderCfg
Decoder = moment.MomentGaussianDecoder


def attributes(gaussians):
    return (gaussians.means, gaussians.covariances,
            gaussians.harmonics, gaussians.opacities)


class MomentDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_knn_matches_brute_force_and_connects_views(self):
        # Adjacent rows across the view boundary must remain valid candidates.
        points = torch.tensor([[0., 0, 0], [10., 0, 0],
                               [0.01, 0, 0], [10.02, 0, 0]])
        indices = build_knn(points, 2)
        torch.testing.assert_close(indices[:, 0], torch.arange(4))
        torch.testing.assert_close(indices[:, 1], torch.tensor([2, 3, 0, 1]))
        distances = torch.cdist(points, points)
        distances.fill_diagonal_(float("inf"))
        torch.testing.assert_close(indices[:, 1], distances.argmin(-1))

    def test_knn_self_is_unique_even_for_coincident_points(self):
        points = torch.zeros(20, 3)
        for k in (1, 5, 30):
            indices = build_knn(points, k)
            self.assertEqual(indices.shape, (20, min(k, 20)))
            torch.testing.assert_close(indices[:, 0], torch.arange(20))
            for row in indices:
                self.assertEqual(row.unique().numel(), row.numel())
        self.assertEqual(build_knn(torch.ones(1, 3), 16).tolist(), [[0]])

    def test_allocation_conserves_each_support_budget(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=8, chunk_size=3))
        points, features = torch.randn(9, 3), torch.randn(9, 5)
        indices = build_knn(points, 4)
        q = model.predict_allocation(points, features, indices)
        budget = model.cfg.budget_max * model.budget_head(features).sigmoid().squeeze(-1)
        torch.testing.assert_close(q.sum(-1), budget)
        mass, *_ = model.aggregate_moments(points, features, indices, q)
        torch.testing.assert_close(mass.sum(), budget.sum())

    def test_incoming_moments_match_independent_dense_reference(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=8, chunk_size=2))
        points, features = torch.randn(4, 3), torch.randn(4, 5)
        # Directed neighborhoods: slot zero receives FOUR supports, despite K=2.
        indices = torch.tensor([[0, 1], [1, 0], [2, 0], [3, 0]])
        q = torch.tensor([[.2, .8], [.3, .7], [.4, .6], [.5, .5]])
        mass, means, covariance, pooled = model.aggregate_moments(points, features, indices, q)
        dense = torch.zeros(4, 4)  # [destination i, source j]
        for j in range(4):
            for k in range(2):
                dense[indices[j, k], j] = q[j, k]
        torch.testing.assert_close(mass, dense.sum(1))
        dense += model.cfg.mass_epsilon * torch.eye(4)
        weights = dense / dense.sum(1, keepdim=True)
        expected_means = weights @ points
        delta = points[None] - expected_means[:, None]
        expected_cov = torch.einsum("ij,ijc,ijd->icd", weights, delta, delta)
        torch.testing.assert_close(means, expected_means)
        torch.testing.assert_close(covariance, expected_cov)
        torch.testing.assert_close(pooled, weights @ features)

    def test_empty_slots_keep_self_without_artificial_opacity(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=8, chunk_size=2))
        points, features = torch.randn(4, 3), torch.randn(4, 5)
        indices = build_knn(points, 3)
        moments = model.aggregate_moments(points, features, indices, torch.zeros(4, 3))
        mass, means, covariance, pooled = moments
        torch.testing.assert_close(means, points)
        torch.testing.assert_close(pooled, features)
        torch.testing.assert_close(covariance, torch.zeros_like(covariance))
        gaussians = model.build_gaussians(*moments, torch.tensor(1.0))
        self.assertEqual(torch.count_nonzero(gaussians.opacities).item(), 0)
        self.assertTrue((torch.linalg.eigvalsh(gaussians.covariances) > 0).all())

    def test_checkpoint_chunks_preserve_outputs_and_all_gradients(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=8, num_neighbors=4, chunk_size=3))
        reference = copy.deepcopy(model)
        reference.cfg.checkpoint_chunks = False
        reference.cfg.chunk_size = 100
        points = torch.randn(2, 9, 3, requires_grad=True)
        features = torch.randn(2, 9, 5, requires_grad=True)
        ref_points = points.detach().clone().requires_grad_()
        ref_features = features.detach().clone().requires_grad_()
        result = model(points, features)
        expected = reference(ref_points, ref_features)
        for actual, target in zip(attributes(result), attributes(expected)):
            torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
        sum(x.square().mean() for x in attributes(result)).backward()
        sum(x.square().mean() for x in attributes(expected)).backward()
        for actual, target in [(points, ref_points), (features, ref_features),
                               *zip(model.parameters(), reference.parameters())]:
            self.assertIsNotNone(actual.grad)
            self.assertTrue(torch.isfinite(actual.grad).all())
            torch.testing.assert_close(actual.grad, target.grad, rtol=1e-4, atol=2e-6)
        for tensor in (points, features, model.allocation_head[0].weight,
                       model.budget_head.weight, model.scale_head.weight,
                       model.sh_head.weight):
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_forward_shapes_positive_covariance_and_scale_equivariance(self):
        model = Decoder(Cfg(feature_dim=5, hidden_dim=8, num_neighbors=4)).eval()
        points, features = torch.randn(2, 11, 3), torch.randn(2, 11, 5)
        with torch.no_grad():
            result = model(points, features)
            scaled = model(points * 3, features)
        self.assertEqual(result.means.shape, (2, 11, 3))
        self.assertEqual(result.covariances.shape, (2, 11, 3, 3))
        self.assertEqual(result.harmonics.shape, (2, 11, 3, 25))
        self.assertEqual(result.opacities.shape, (2, 11))
        self.assertTrue((torch.linalg.eigvalsh(result.covariances) > 0).all())
        self.assertTrue(((result.opacities >= 0) & (result.opacities <= 1)).all())
        torch.testing.assert_close(scaled.means, result.means * 3)
        torch.testing.assert_close(scaled.covariances, result.covariances * 9)
        torch.testing.assert_close(scaled.opacities, result.opacities)
        torch.testing.assert_close(scaled.harmonics, result.harmonics)

    def test_dpt_projection_and_decoder_backpropagate_to_tokens(self):
        backbone = SimpleNamespace(dec_depth=12, enc_embed_dim=16, dec_embed_dim=16)
        head = feature_head.create_dpt_feature_head(backbone, 5)
        self.assertIsInstance(head.dpt.head, nn.Conv2d)
        self.assertEqual(head.dpt.head.kernel_size, (1, 1))
        tokens = [torch.randn(1, 4, 16, requires_grad=True) for _ in range(13)]
        image = torch.randn(1, 3, 32, 32)
        projected = head(tokens, None, image, (32, 32))
        self.assertEqual(projected.shape, (1, 5, 32, 32))
        points = torch.randn(1, 1024, 3, requires_grad=True)
        model = Decoder(Cfg(feature_dim=5, hidden_dim=8, num_neighbors=4, chunk_size=256))
        result = model(points, projected.flatten(2).transpose(1, 2))
        sum(x.square().mean() for x in attributes(result)).backward()
        self.assertGreater(points.grad.abs().sum().item(), 0)
        self.assertGreater(head.dpt.head.weight.grad.abs().sum().item(), 0)
        for index in (0, 6, 9, 12):
            self.assertGreater(tokens[index].grad.abs().sum().item(), 0)

    def test_experiment_composition_preserves_baseline_training(self):
        from dacite import from_dict
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            baseline = compose(config_name="main", overrides=["+experiment=re10k"])
            cfg = compose(config_name="main", overrides=["+experiment=re10k_moment"])
        self.assertEqual(cfg.model.encoder.gs_params_head_type, "moment")
        decoder_cfg = from_dict(Cfg, OmegaConf.to_container(cfg.model.encoder.moment_decoder))
        self.assertEqual(decoder_cfg.feature_dim, 64)
        self.assertEqual(decoder_cfg.num_neighbors, 16)
        for key in ("dataset", "data_loader", "optimizer", "trainer", "test", "loss", "train"):
            self.assertEqual(OmegaConf.to_container(cfg[key]), OmegaConf.to_container(baseline[key]))

    def test_encoder_integration_uses_only_context_and_preserves_slot_order(self):
        # Run the production encoder forward with a tiny front end. DPT itself
        # is exercised separately above; this isolates the new connection code.
        path = ROOT / "src/model/encoder/encoder_noposplat.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EncoderNoPoSplat")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        namespace = {"torch": torch, "Optional": Optional,
                     "Gaussians": moment.Gaussians, "rearrange": rearrange}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

        class FrontEnd(nn.Module):
            def forward(self, context, return_views):
                image = context["image"]
                shape = torch.tensor([[2, 2]])
                return [], [], shape, shape, {"img": image[:, 0]}, {"img": image[:, 1]}

        class FeatureHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Conv2d(3, 5, 1)

            def forward(self, tokens, points, image, shape):
                return self.projection(image)

        encoder = nn.Module()
        encoder.forward = MethodType(namespace["forward"], encoder)
        encoder.gs_params_head_type = "moment"
        encoder.backbone = FrontEnd()
        encoder.supports = nn.Parameter(torch.randn(2, 1, 2, 2, 3))
        encoder._downstream_head = lambda index, tokens, shape: {"pts3d": encoder.supports[index - 1]}
        encoder.gaussian_param_head = FeatureHead()
        encoder.gaussian_param_head2 = FeatureHead()
        # K=1 makes the expected ordering unambiguous: means must equal supports.
        encoder.gaussian_decoder = Decoder(Cfg(feature_dim=5, num_neighbors=1))
        dump = {}
        result = encoder({"image": torch.randn(1, 2, 3, 2, 2)}, visualization_dump=dump)
        torch.testing.assert_close(result.means, encoder.supports.reshape(1, 8, 3))
        self.assertEqual(dump["means"].shape, (1, 2, 2, 2, 1, 3))
        self.assertEqual(dump["depth"].shape, (1, 2, 2, 2, 1, 1))
        sum(x.square().mean() for x in attributes(result)).backward()
        self.assertGreater(encoder.supports.grad.abs().sum().item(), 0)
        for head in (encoder.gaussian_param_head, encoder.gaussian_param_head2):
            self.assertGreater(head.projection.weight.grad.abs().sum().item(), 0)

    def test_optimizer_includes_decoder_in_full_learning_rate_group(self):
        # Exercise the real method with small modules instead of importing Lightning.
        path = ROOT / "src/model/model_wrapper.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelWrapper")
        method = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == "configure_optimizers")
        namespace = {"torch": torch, "optim": torch.optim,
                     "get_cfg": lambda: {"trainer": {"max_steps": 100}}}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.backbone = nn.Linear(3, 3)
        model.encoder.gaussian_param_head = nn.Linear(3, 5)
        model.encoder.gaussian_decoder = Decoder(Cfg(feature_dim=5, hidden_dim=8))
        model.optimizer_cfg = SimpleNamespace(lr=1e-4, backbone_lr_multiplier=.1, warm_up_steps=2)
        optimizer = namespace["configure_optimizers"](model)["optimizer"]
        new_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
        for p in model.encoder.gaussian_decoder.parameters():
            self.assertIn(id(p), new_ids)
        for p in model.encoder.backbone.parameters():
            self.assertNotIn(id(p), new_ids)


if __name__ == "__main__":
    unittest.main()
