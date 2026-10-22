"""Selected-checkpoint evaluation without CUDA, Lightning or W&B network writes."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from scripts import eval_checkpoints as evaluation


class EvalCheckpointsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / 'run'
        (self.run_dir / '.hydra').mkdir(parents=True)
        (self.run_dir / 'checkpoints').mkdir()
        (self.root / 'dataset/test').mkdir(parents=True)
        self.index = self.root / 'index.json'
        self.index.write_text('{}', encoding='utf-8')
        self.config_path = self.run_dir / '.hydra/config.yaml'
        self.config = OmegaConf.create(dict(
            mode='train', seed=23, wandb=dict(name='original_final_plus', project='noposplat'),
            dataset=dict(re10k=dict(roots=[str(self.root / 'dataset')],
                                   view_sampler=dict(name='bounded', num_context_views=2, warm_up_steps=75000))),
            trainer=dict(num_nodes=1, max_steps=140001, auto_eval_index_path=str(self.index),
                         auto_eval_save_image=False, auto_eval_save_video=False, auto_eval_save_compare=False),
            test=dict(compute_scores=True, align_pose=True, pose_align_steps=100,
                      rot_opt_lr=.005, trans_opt_lr=.005, output_path='outputs/test',
                      save_image=True, save_video=True, save_compare=True),
            data_loader=dict(test=dict(batch_size=1)),
            model=dict(encoder=dict(moment_decoder=dict(feature_dim=256, appearance_moments=True,
                                                       appearance_mlp_dim=256, appearance_mlp_depth=2)),
                       decoder=dict(name='splatting_cuda')),
            loss=dict(mse=dict(weight=1.), lpips=dict(weight=.05)),
            train=dict(distiller=''), optimizer=dict(lr=.0001),
        ))
        OmegaConf.save(self.config, self.config_path)
        for epoch, step in ((0, 20000), (0, 40000), (0, 60000), (1, 80000),
                            (1, 100000), (2, 120000), (2, 140000)):
            (self.run_dir / 'checkpoints' / f'epoch_{epoch}-step_{step}.ckpt').touch()
        self.argv = ['--run-dir', str(self.run_dir), '--wandb-run', 'tmdwo8814-/noposplat/qqvj055m']

    def test_run_identifiers_and_forge_url(self):
        for value in ('tmdwo8814-/noposplat/qqvj055m',
                      'https://wandb.ai/tmdwo8814-/noposplat/runs/qqvj055m',
                      'forge.coreweave.com/wandb/tmdwo8814-/noposplat/runs/qqvj055m?nw=nwusertmdwo8814'):
            self.assertEqual(evaluation.parse_run_path(value), ('tmdwo8814-', 'noposplat', 'qqvj055m'))
        for value in ('qqvj055m', 'wrong/path', 'https://example.com/a/b/runs/c', 'a/../c'):
            with self.assertRaises(ValueError):
                evaluation.parse_run_path(value)

    def test_selection_skips_finished_steps_and_rejects_missing_or_ambiguous(self):
        directory = self.run_dir / 'checkpoints'
        actual = evaluation.select_checkpoints(directory, [140000, 80000, 120000, 100000])
        self.assertEqual([step for step, _ in actual], [80000, 100000, 120000, 140000])
        for steps in ([], [80000, 80000], [-1], [160000]):
            with self.assertRaises(ValueError):
                evaluation.select_checkpoints(directory, steps)
        (directory / 'copy-step_80000.ckpt').touch()
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            evaluation.select_checkpoints(directory, [80000])

    def test_saved_config_matches_existing_post_eval_transform(self):
        original = copy.deepcopy(self.config)
        loaded = evaluation.load_eval_config(self.config_path)
        self.assertEqual(loaded.mode, 'test')
        self.assertEqual(OmegaConf.to_container(loaded.dataset.re10k.view_sampler),
                         dict(name='evaluation', index_path=str(self.index), num_context_views=2))
        for key in ('model', 'optimizer', 'loss', 'data_loader', 'wandb', 'trainer'):
            self.assertEqual(loaded[key], original[key])
        for key in ('align_pose', 'pose_align_steps', 'rot_opt_lr', 'trans_opt_lr'):
            self.assertEqual(loaded.test[key], original.test[key])
        for key in ('save_image', 'save_video', 'save_compare'):
            self.assertFalse(loaded.test[key])
        self.assertEqual(OmegaConf.load(self.config_path), original)
        self.index.unlink()
        with self.assertRaises(FileNotFoundError):
            evaluation.load_eval_config(self.config_path)
        with self.assertRaises(FileNotFoundError):
            evaluation.load_eval_config(self.root / 'missing.yaml')

    def test_dry_run_never_loads_runtime_or_creates_outputs(self):
        with patch.dict(sys.modules, {'src.main': None}), contextlib.redirect_stdout(io.StringIO()) as out:
            evaluation.main(self.argv + ['--dry-run'])
        self.assertIn('Dry run OK', out.getvalue())
        self.assertFalse((self.run_dir / 'remaining_eval').exists())

    def runtime(self, rank=0):
        # Trainer.setup() changes rank after construction, as DDP child launch does.
        from test_auto_eval_logging import Scalar
        runtime = ModuleType('src.main')
        runtime.wandb = SimpleNamespace(run=None, finish=Mock())
        runtime.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 3),
                                        manual_seed=Mock())
        runtime.set_cfg = Mock()
        runtime.load_typed_root_config = lambda config: SimpleNamespace(**dict(config))
        runtime.get_encoder = lambda config: ('encoder', 'visualizer')
        runtime.get_decoder = lambda config: 'decoder'
        runtime.get_losses = lambda config: 'losses'
        runtime.get_distiller = Mock()
        runtime.WandbLogger = Mock(return_value='logger')
        model = SimpleNamespace(test_cfg=SimpleNamespace(), _test_metric_names=('psnr_ours', 'ssim_ours', 'lpips_ours'),
                                _test_overlap_tags=('small', 'medium', 'large'))
        runtime.ModelWrapper = Mock(return_value=model)

        class DataModule:
            def __init__(self, *args, global_rank=0, **kwargs):
                self.global_rank = global_rank

            def setup(self, stage=None):
                pass

        runtime.DataModule = DataModule
        trainer = SimpleNamespace(global_rank=0, is_global_zero=rank == 0, fit=Mock())
        calls = []

        def test(model, datamodule, ckpt_path):
            trainer.global_rank = rank
            datamodule.trainer = trainer
            datamodule.setup('test')
            self.assertEqual(datamodule.global_rank, rank)
            self.assertTrue(model.strict_loading)
            step = model._auto_eval_step
            calls.append((step, model._auto_eval_log_prefix, ckpt_path, model.test_cfg.output_path))
            model._test_metric_count = Scalar(2)
            model._test_metric_sums = {key: Scalar(step / 10000) for key in model._test_metric_names}
            model._test_overlap_counts = {tag: Scalar(1 if tag != 'medium' else 0) for tag in model._test_overlap_tags}
            model._test_overlap_sums = {tag: {key: Scalar(1) for key in model._test_metric_names}
                                         for tag in model._test_overlap_tags}
            if rank == 0:
                runtime.wandb.run = object()

        trainer.test = Mock(side_effect=test)
        runtime.Trainer = Mock(return_value=trainer)
        return runtime, trainer, calls

    def test_execution_calls_only_test_with_exact_steps_and_preserves_wandb_metadata(self):
        runtime, trainer, calls = self.runtime()
        with patch.dict(sys.modules, {'src.main': runtime}), contextlib.redirect_stdout(io.StringIO()):
            evaluation.main(self.argv)
        trainer.fit.assert_not_called()
        self.assertEqual([step for step, *_ in calls], [80000, 100000, 120000, 140000])
        for step, prefix, checkpoint, path in calls:
            self.assertEqual(prefix, 'auto_eval')
            self.assertIn(f'step_{step}', checkpoint)
            saved = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['metrics']['auto_eval/ckpt_step'], step)
            self.assertEqual(saved['metrics']['auto_eval/overall/psnr'], step / 20000)
            self.assertNotIn('auto_eval/medium/psnr', saved['metrics'])
        options = runtime.WandbLogger.call_args.kwargs
        self.assertEqual((options['id'], options['project'], options['entity'], options['resume']),
                         ('qqvj055m', 'noposplat', 'tmdwo8814-', 'must'))
        self.assertFalse({'name', 'config', 'tags', 'notes', 'resume_from'} & options.keys())
        self.assertFalse(runtime.Trainer.call_args.kwargs['inference_mode'])
        self.assertFalse(runtime.Trainer.call_args.kwargs['enable_checkpointing'])
        runtime.wandb.finish.assert_called_once_with()

    def test_nonzero_rank_uses_its_own_shard_and_does_not_write_results(self):
        runtime, trainer, calls = self.runtime(rank=2)
        with patch.dict(sys.modules, {'src.main': runtime}), contextlib.redirect_stdout(io.StringIO()):
            evaluation.main(self.argv + ['--steps', '120000', '140000'])
        self.assertEqual(len(calls), 2)
        self.assertFalse(list((self.run_dir / 'remaining_eval').glob('step_*/metrics.json')))
        runtime.wandb.finish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
