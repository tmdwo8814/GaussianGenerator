"""Exercise the current full Final++ decoder and real strict checkpoint restore."""

import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import torch
from omegaconf import OmegaConf

from exp3.plot_scale_opacity import ROOT, compose_config, parser, restore_encoder


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, path in {
            "_exp3_test": ROOT / "src/model",
            "_exp3_test.encoder": ROOT / "src/model/encoder",
            "_exp3_test.encoder.heads": ROOT / "src/model/encoder/heads",
            "_exp3_test.encoder.common": ROOT / "src/model/encoder/common",
        }.items():
            package = ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        cls.moment = importlib.import_module("_exp3_test.encoder.heads.moment_gaussian_decoder")

    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "ours.ckpt"
        args = parser().parse_args([])
        cfg, _ = compose_config(args, "ours", Path(self.temporary.name) / "index.json")
        self.cfg = self.moment.MomentDecoderCfg(**OmegaConf.to_container(cfg.model.encoder.moment_decoder))
        self.cfg.knn_backend = "scipy"

    def model(self):
        decoder = self.moment.MomentGaussianDecoder(copy.deepcopy(self.cfg), sh_degree=4)
        encoder = torch.nn.Module()
        encoder.add_module("backbone", torch.nn.Linear(3, 3))
        encoder.add_module("gaussian_decoder", decoder)
        return encoder.eval()

    def checkpoint(self, model):
        return {"global_step": 140000,
                "state_dict": {f"encoder.{key}": value for key, value in model.state_dict().items()}}

    def test_current_sh_shapes_strict_restore_and_complete_forward(self):
        reference = self.model()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.add_(0.01 * torch.randn_like(parameter))
        head = reference.gaussian_decoder.appearance_sh_head
        self.assertEqual(tuple(head[0].weight.shape), (256, 576))
        self.assertEqual(tuple(head[2].weight.shape), (256, 256))
        self.assertEqual(tuple(head[4].weight.shape), (75, 256))
        torch.save(self.checkpoint(reference), self.path)
        checkpoint_bytes = self.path.read_bytes()
        restored = self.model()
        self.assertEqual(restore_encoder(restored, self.path), 140000)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        points, features, rgb = torch.randn(1, 12, 3), torch.randn(1, 12, 256), torch.rand(1, 12, 3)
        with torch.no_grad():
            expected = reference.gaussian_decoder(points, features, image_shape=(2, 2, 3), rgb=rgb)
            actual = restored.gaussian_decoder(points, features, image_shape=(2, 2, 3), rgb=rgb)
        for key in ("means", "covariances", "opacities", "harmonics"):
            torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(getattr(actual, key)).all())
        self.assertEqual(tuple(actual.harmonics.shape), (1, 12, 3, 25))
        self.assertEqual(self.path.read_bytes(), checkpoint_bytes)

    def test_missing_appearance_weight_is_an_error(self):
        checkpoint = self.checkpoint(self.model())
        del checkpoint["state_dict"]["encoder.gaussian_decoder.appearance_sh_head.4.weight"]
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(RuntimeError, "appearance_sh_head.4.weight"):
            restore_encoder(self.model(), self.path)

    def test_wrong_appearance_shape_is_an_error(self):
        checkpoint = self.checkpoint(self.model())
        checkpoint["state_dict"]["encoder.gaussian_decoder.appearance_sh_head.0.weight"] = torch.zeros(256, 320)
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            restore_encoder(self.model(), self.path)


if __name__ == "__main__":
    unittest.main()
