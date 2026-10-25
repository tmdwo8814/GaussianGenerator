"""Check the real repository moment decoder without importing the CUDA renderer."""

import copy
import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from exp3.plot_scale_opacity import ROOT, restore_encoder


class GeometryCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = importlib.import_module("torch")
            importlib.import_module("scipy")
            importlib.import_module("jaxtyping")
        except ImportError as error:
            raise unittest.SkipTest(f"Actual decoder CPU checks require model dependencies: {error}")
        for name, path in {
            "_exp3_geometry": ROOT / "src/model",
            "_exp3_geometry.encoder": ROOT / "src/model/encoder",
            "_exp3_geometry.encoder.heads": ROOT / "src/model/encoder/heads",
            "_exp3_geometry.encoder.common": ROOT / "src/model/encoder/common",
        }.items():
            package = ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        cls.moment = importlib.import_module("_exp3_geometry.encoder.heads.moment_gaussian_decoder")

    def models(self):
        torch = self.torch
        torch.manual_seed(71)
        torch.set_num_threads(1)
        cfg = self.moment.MomentDecoderCfg(
            feature_dim=5, hidden_dim=8, num_neighbors=4, chunk_size=4,
            knn_backend="scipy", appearance_2d=True, separate_appearance=True,
            appearance_capacity=True, appearance_dim=3, appearance_heads=2,
            appearance_mlp_dim=7, checkpoint_appearance=False,
        )

        class Encoder(torch.nn.Module):
            def __init__(self, options):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 3)
                self.gaussian_decoder = self_moment.MomentGaussianDecoder(options)

        self_moment = self.moment
        full = Encoder(cfg).eval()
        # Exercise a nonzero, three-layer SH reader, unlike today's two-layer config.
        full.gaussian_decoder.appearance_sh_head = torch.nn.Sequential(
            torch.nn.Linear(11, 12), torch.nn.SiLU(), torch.nn.Linear(12, 12),
            torch.nn.SiLU(), torch.nn.Linear(12, 75),
        )
        with torch.no_grad():
            for parameter in full.parameters():
                parameter.add_(0.1 * torch.randn_like(parameter))
        geometry_cfg = copy.deepcopy(cfg)
        geometry_cfg.appearance_2d = False
        geometry_cfg.separate_appearance = False
        geometry_cfg.appearance_capacity = False
        geometry = Encoder(geometry_cfg).eval()
        state = {f"encoder.{key}": value for key, value in full.state_dict().items()}
        # Exact incompatible SH shapes from the reported server checkpoint.
        prefix = "encoder.gaussian_decoder.appearance_sh_head."
        for key, shape in {"0.weight": (256, 576), "0.bias": (256,),
                           "2.weight": (256, 256), "2.bias": (256,),
                           "4.weight": (75, 256), "4.bias": (75,)}.items():
            state[prefix + key] = torch.randn(shape)
        return full, geometry, {"state_dict": state, "global_step": 140000}

    def test_reported_sh_mismatch_does_not_change_geometry(self):
        torch = self.torch
        full, geometry, checkpoint = self.models()
        info = {}
        with patch.object(torch, "load", return_value=checkpoint):
            step = restore_encoder(geometry, Path("ours.ckpt"), geometry_only=True, load_info=info)
        self.assertEqual(step, 140000)
        self.assertTrue(info["strict"])
        self.assertIn("gaussian_decoder.appearance_sh_head.4.weight", info["omitted_appearance_keys"])
        torch.testing.assert_close(full.backbone.weight, geometry.backbone.weight)
        points, features = torch.randn(1, 12, 3), torch.randn(1, 12, 5)
        with torch.no_grad():
            reference = full.gaussian_decoder(points, features, image_shape=(2, 2, 3),
                                               rgb=torch.rand(1, 12, 3))
            actual = geometry.gaussian_decoder(points, features)
        for key in ("means", "covariances", "opacities"):
            torch.testing.assert_close(getattr(actual, key), getattr(reference, key),
                                       rtol=1e-6, atol=1e-7)

    def test_missing_geometry_weight_still_fails_strict_restore(self):
        _, geometry, checkpoint = self.models()
        del checkpoint["state_dict"]["encoder.gaussian_decoder.budget_head.weight"]
        with patch.object(self.torch, "load", return_value=checkpoint):
            with self.assertRaisesRegex(RuntimeError, "budget_head.weight"):
                restore_encoder(geometry, Path("ours.ckpt"), geometry_only=True)

    def test_geometry_restore_cannot_skip_an_enabled_appearance_branch(self):
        full, _, checkpoint = self.models()
        with patch.object(self.torch, "load", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "Disable appearance"):
                restore_encoder(full, Path("ours.ckpt"), geometry_only=True)


if __name__ == "__main__":
    unittest.main()
