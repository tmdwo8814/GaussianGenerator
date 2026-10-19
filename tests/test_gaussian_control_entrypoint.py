"""Hydra composition and full-checkpoint restore orchestration without CUDA."""

import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]


class GaussianControlEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.checkpoint = self.directory / 'Final.ckpt'
        self.checkpoint.touch()
        self.index = self.directory / 'index.json'
        self.index.write_text('{}', encoding='utf-8')
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            self.cfg = compose(config_name='main', overrides=[
                '+experiment=re10k_moment',
                f'+gaussian_control.checkpoint={self.checkpoint.as_posix()}',
                f'+gaussian_control.index_path={self.index.as_posix()}',
                '+gaussian_control.max_scenes=2',
                f'dataset.re10k.roots=["{self.directory.as_posix()}"]',
                'test.align_pose=true', 'wandb.mode=disabled',
                'model.encoder.moment_decoder.compile_kernels=false',
            ])
        tree = ast.parse((ROOT / 'src/eval_gaussian_control.py').read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('prepare_config', 'evaluate')]
        for node in nodes:
            node.decorator_list = []
        self.namespace = {'Path': Path, 'DictConfig': DictConfig, 'OmegaConf': OmegaConf}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'control_entrypoint', 'exec'), self.namespace)

    def test_compose_final_and_replace_only_test_sampler(self):
        checkpoint = self.namespace['prepare_config'](self.cfg)
        self.assertEqual(checkpoint, self.checkpoint)
        self.assertEqual(self.cfg.mode, 'test')
        self.assertEqual(self.cfg.model.encoder.moment_decoder.appearance_heads, 2)
        self.assertEqual(self.cfg.model.encoder.moment_decoder.appearance_mlp_dim, 128)
        self.assertEqual(self.cfg.model.encoder.pretrained_weights, '')
        self.assertEqual(self.cfg.dataset.re10k.view_sampler.name, 'evaluation')
        self.assertEqual(self.cfg.dataset.re10k.view_sampler.num_context_views, 2)
        self.assertEqual(Path(self.cfg.dataset.re10k.view_sampler.index_path), self.index)

    def test_rejects_wrong_architecture_or_missing_checkpoint(self):
        self.cfg.model.encoder.gs_params_head_type = 'dpt_gs'
        with self.assertRaisesRegex(ValueError, 'Final moment encoder'):
            self.namespace['prepare_config'](self.cfg)
        self.cfg.model.encoder.gs_params_head_type = 'moment'
        self.checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.namespace['prepare_config'](self.cfg)

    def test_strict_full_restore_freeze_and_test_only(self):
        trainer, model, torch = Mock(), Mock(), Mock()
        state = {'encoder.backbone.weight': 'trained weight', 'encoder.gaussian_decoder.sh_head.weight': 'trained SH'}
        torch.load.return_value = {'state_dict': state, 'global_step': 140000}
        self.namespace.update(
            hydra=SimpleNamespace(core=SimpleNamespace(hydra_config=SimpleNamespace(
                HydraConfig=SimpleNamespace(get=lambda: SimpleNamespace(
                    runtime=SimpleNamespace(output_dir=str(self.directory))
                ))))),
            torch=torch, seed_everything=Mock(), set_cfg=Mock(),
            load_typed_root_config=lambda cfg: cfg,
            get_encoder=Mock(return_value=('encoder', None)), get_decoder=Mock(), get_losses=Mock(),
            GaussianControlModel=Mock(return_value=model), Trainer=Mock(return_value=trainer),
            DataModule=Mock(),
        )
        with redirect_stdout(StringIO()):
            self.namespace['evaluate'](self.cfg)
        torch.load.assert_called_once_with(self.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict.assert_called_once_with(state, strict=True)
        model.freeze.assert_called_once_with()
        self.assertEqual(model.analysis_step, 140000)
        self.assertEqual(model.analysis_metadata['camera_protocol'], 'shared_final_pose')
        trainer.test.assert_called_once()
        trainer.fit.assert_not_called()
        self.assertEqual(self.namespace['Trainer'].call_args.kwargs['devices'], 1)
        self.assertFalse(self.namespace['Trainer'].call_args.kwargs['inference_mode'])


if __name__ == '__main__':
    unittest.main()
