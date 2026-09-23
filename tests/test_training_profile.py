"""CPU checks of instrumentation; server GPU measurements remain necessary."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_training_profile', ROOT / 'scripts/profile_training.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class TrainingProfileTests(unittest.TestCase):
    def test_wrapper_preserves_forward_backward_and_restores_method(self):
        model = torch.nn.Linear(3, 2)
        values = torch.randn(4, 3)
        expected = model(values)
        expected.sum().backward()
        expected_gradient = model.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        timer = profile.StageTimer(lambda: None)
        timer.wrap(model, 'forward', 'linear_s')
        # Disabled phase records no stage timings.
        torch.testing.assert_close(model(values), expected)
        self.assertEqual(dict(timer.calls), {})
        timer.enabled = True
        actual = model(values)
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model.weight.grad, expected_gradient)
        self.assertEqual(timer.calls['linear_s'], 1)
        self.assertGreaterEqual(timer.values['linear_s'], 0)
        timer.restore()
        self.assertNotIn('forward', vars(model))
        torch.testing.assert_close(model(values), expected)

    def test_wrap_existing_function_restores_it_after_exception(self):
        def broken():
            raise RuntimeError('expected')
        owner = SimpleNamespace(run=broken)
        timer = profile.StageTimer(lambda: None)
        timer.wrap(owner, 'run', 'broken_s')
        timer.enabled = True
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            owner.run()
        self.assertEqual(timer.calls['broken_s'], 1)
        timer.restore()
        self.assertIs(owner.run, broken)

    def test_schedule_records_only_measured_phases_and_writes_rank_report(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(3, 3)
                self.encoder.cfg = SimpleNamespace()
                self.encoder.gs_params_head_type = 'dpt_gs'
                self.decoder = torch.nn.Linear(3, 1)
                self.device = torch.device('cpu')

            def training_step(self, values):
                return self.decoder(self.encoder(values)).sum()

        model = Model()
        trainer = SimpleNamespace(global_rank=1, world_size=2, is_global_zero=False)
        batch = {'context': {'image': torch.ones(1, 2, 3, 4, 4)}}
        with TemporaryDirectory() as tmp, \
             patch.object(torch.cuda, 'synchronize'), \
             patch.object(torch.cuda, 'reset_peak_memory_stats') as reset, \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=2**30), \
             patch.object(torch.cuda, 'get_device_name', return_value='test GPU'):
            args = SimpleNamespace(warmup=1, steps=2, output=Path(tmp) / 'timing.json')
            callback = profile.make_callback(object, args)
            callback.on_fit_start(trainer, model)
            for index in range(5):
                callback.on_train_batch_start(trainer, model, batch, index)
                loss = model.training_step(torch.ones(2, 3))
                callback.on_before_backward(trainer, model, loss)
                loss.backward()
                callback.on_after_backward(trainer, model)
                callback.on_train_batch_end(trainer, model, None, batch, index)
            self.assertEqual(reset.call_count, 3)
            self.assertEqual([len(callback.rows[key]) for key in ('throughput', 'stages')], [2, 2])
            self.assertNotIn('encoder_s', callback.rows['throughput'][0])
            self.assertEqual(callback.rows['stages'][0]['encoder_calls'], 1)
            self.assertEqual(callback.rows['stages'][0]['renderer_calls'], 1)
            self.assertEqual(callback.rows['stages'][0]['backward_calls'], 1)
            self.assertGreater(callback.rows['throughput'][0]['step_wall_s'], 0)
            callback.on_train_end(trainer, model)
            report = json.loads((Path(tmp) / 'timing.rank1.json').read_text())
            self.assertEqual(report['samples_per_phase'], {'throughput': 2, 'stages': 2})
            self.assertEqual(report['peak_allocated_gib']['throughput'], 1)
            self.assertNotIn('forward', vars(model.encoder))
            self.assertNotIn('training_step', vars(model))

    def test_cli_preserves_hydra_arguments(self):
        args, overrides = profile.parse_options([
            '+experiment=re10k', '--warmup', '3', '--steps', '4',
            'data_loader.train.batch_size=16', '--output', 'outputs/base.json'])
        self.assertEqual((args.warmup, args.steps), (3, 4))
        self.assertEqual(overrides, ['+experiment=re10k', 'data_loader.train.batch_size=16'])
        self.assertEqual(args.output, Path('outputs/base.json'))

    def test_ddp_child_keeps_parent_limits_and_output_after_hydra_argv_rewrite(self):
        inherited = json.dumps({'warmup': 5, 'steps': 7, 'output': 'outputs/ddp.json'})
        with patch.dict(os.environ, {'GAUSSIAN_SPEED_PROBE_OPTIONS': inherited}):
            args, overrides = profile.parse_options(['+experiment=re10k_moment'])
        self.assertEqual((args.warmup, args.steps), (5, 7))
        self.assertEqual(args.output, Path('outputs/ddp.json'))
        self.assertEqual(overrides, ['+experiment=re10k_moment'])

    @unittest.skipIf(sys.version_info >= (3, 14), 'Hydra 1.3 CLI help is incompatible with Python 3.14 argparse')
    def test_imported_entry_uses_repository_config_directory(self):
        seen = []
        def task(cfg):
            seen.append(cfg.value)
        task.__module__ = 'src.main'
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config').mkdir()
            (root / 'config/main.yaml').write_text('value: 3\n')
            entry = SimpleNamespace(__file__=str(root / 'src/main.py'),
                                    train=SimpleNamespace(__wrapped__=task))
            argv = ['profile_training', 'value=7', 'hydra.output_subdir=null',
                    f'hydra.run.dir={root.as_posix()}/run']
            with patch.object(sys, 'argv', argv):
                profile.launch_training(entry)
        self.assertEqual(seen, [7])


if __name__ == '__main__':
    unittest.main()
