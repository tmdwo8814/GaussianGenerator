"""Budget 1 resumed evaluation: config isolation, run routing and exact steps."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_resume_eval as existing


ROOT = Path(__file__).resolve().parents[1]


class Budget1EvalTests(unittest.TestCase):
    setUp = existing.ResumeEvalTests.setUp
    setup_evaluation = existing.ResumeEvalTests.setup_evaluation
    evaluate = existing.ResumeEvalTests.evaluate

    def config(self):
        cfg = existing.ResumeEvalTests.config(self)
        cfg.resume_eval.steps = [100000, 120000, 140000]
        cfg.resume_eval.expected_decoder = existing.Config(
            budget_max=1.0, allocation_mode='learned', budget_mode='support')
        cfg.wandb.id = 'un2ifuj6'
        cfg.model.encoder = existing.Config(gs_params_head_type='moment',
                                            moment_decoder=existing.Config(budget_max=1.0))
        return cfg

    def test_historical_flags_and_wrong_ablation_detection_before_wandb(self):
        self.namespace['validate_expected_decoder'](self.config())
        for key, value in (('budget_max', 2.), ('allocation_mode', 'uniform'),
                           ('budget_mode', 'scene_shared')):
            cfg = self.config()
            cfg.model.encoder.moment_decoder[key] = value
            self.setup_evaluation(cfg)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.evaluate(cfg)
            self.namespace['WandbLogger'].assert_not_called()
            self.namespace['Trainer'].assert_not_called()

    def test_exact_three_checkpoints_append_to_requested_run_without_training(self):
        cfg = self.config()
        trainer, _ = self.setup_evaluation(cfg)
        self.evaluate(cfg)
        trainer.fit.assert_not_called()
        self.assertEqual([Path(c.kwargs['ckpt_path']).name for c in trainer.test.call_args_list],
                         ['epoch_1-step_100000.ckpt', 'epoch_2-step_120000.ckpt',
                          'epoch_2-step_140000.ckpt'])
        options = self.namespace['WandbLogger'].call_args.kwargs
        self.assertEqual((options['entity'], options['project'], options['id'], options['resume']),
                         ('tmdwo8814-', 'noposplat', 'un2ifuj6', 'must'))
        self.assertFalse({'name', 'config', 'tags', 'notes'} & options.keys())
        for step in cfg.resume_eval.steps:
            path = self.root / 'test' / f'step_{step:08d}' / 'metrics.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(data['metrics']['auto_eval/ckpt_step'], step)
            self.assertEqual(data['metrics']['auto_eval/overall/psnr'], step / 20000)
        self.assertFalse((self.root / 'test/step_00080000').exists())
        self.namespace['wandb'].finish.assert_called_once_with(exit_code=0)

    def test_budget1_dry_run_performs_validation_without_external_writes(self):
        cfg = self.config()
        cfg.resume_eval.dry_run = True
        self.setup_evaluation(cfg)
        self.evaluate(cfg)
        for name in ('inherit_training_auth', 'WandbLogger', 'Trainer'):
            self.namespace[name].assert_not_called()
        self.assertFalse(list(self.root.rglob('metrics.json')))


class SavedConfigTests(unittest.TestCase):
    def test_saved_snapshot_is_composed_independently_with_runtime_id_override(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            trained = compose(config_name='main', overrides=['+experiment=re10k_moment'])
        # Distinct values ensure current branch defaults cannot silently win.
        trained.trainer.max_steps = 140001
        trained.test.pose_align_steps = 83
        trained.model.encoder.moment_decoder.budget_max = 1.0
        OmegaConf.update(trained, 'wandb.id', 'historical_id', force_add=True)
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / 'config.yaml'
            OmegaConf.save(trained, snapshot)
            original = snapshot.read_bytes()
            with initialize_config_dir(config_dir=temporary, version_base=None):
                loaded = compose(config_name='config', overrides=[
                    '++wandb.id=un2ifuj6', 'wandb.entity=tmdwo8814-',
                    'wandb.project=noposplat', 'wandb.mode=online',
                    '+resume_eval.steps=[100000,120000,140000]',
                    '+resume_eval.expected_decoder.budget_max=1.0',
                    '+resume_eval.expected_decoder.allocation_mode=learned',
                    '+resume_eval.expected_decoder.budget_mode=support',
                ])
            self.assertEqual(loaded.trainer.max_steps, 140001)
            self.assertEqual(loaded.test.pose_align_steps, 83)
            self.assertEqual(loaded.model, trained.model)
            self.assertEqual(loaded.dataset, trained.dataset)
            self.assertEqual(loaded.wandb.id, 'un2ifuj6')
            self.assertEqual(loaded.wandb.name, trained.wandb.name)
            self.assertEqual(list(loaded.resume_eval.steps), [100000, 120000, 140000])
            self.assertEqual(snapshot.read_bytes(), original)
            with contextlib.redirect_stdout(io.StringIO()):
                existing.load_functions()['validate_expected_decoder'](loaded)


def find_bash():
    if os.name == 'nt':
        candidate = Path('C:/Program Files/Git/bin/bash.exe')
        return str(candidate) if candidate.is_file() else None
    return shutil.which('bash')


def bash_path(path):
    value = Path(path).resolve().as_posix()
    if os.name == 'nt':
        value = '/' + value[0].lower() + value[2:]
    return value


@unittest.skipUnless(find_bash(), 'Bash required for launcher wiring checks')
class LauncherTests(unittest.TestCase):
    def launch(self, *, extra=(), missing_config=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run = root / 'training run'
        (run / 'checkpoints').mkdir(parents=True)
        (run / '.hydra').mkdir()
        if not missing_config:
            (run / '.hydra/config.yaml').write_text('mode: train\n', encoding='utf-8')
        bin_dir = root / 'bin'
        bin_dir.mkdir()
        fake_python = bin_dir / 'python'
        fake_python.write_text(
            '#!/usr/bin/env bash\n'
            'if [[ "$1" == "--version" ]]; then echo "Python (test stub)"; exit 0; fi\n'
            'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n', encoding='utf-8', newline='\n')
        fake_python.chmod(0o755)
        capture = root / 'args.txt'
        output = root / 'output'
        if os.name == 'nt':
            # MSYS mkdir -p attempts ancestor creation outside the Windows
            # workspace sandbox. Prepare known test dirs natively; the shim
            # still fails on any unexpected/missing launcher output path.
            for directory in (output, root / 'wandb', root / 'cache'):
                directory.mkdir()
            (bin_dir / 'mkdir').write_text(
                '#!/usr/bin/env bash\n'
                'for dir in "$@"; do [[ "$dir" == "-p" ]] && continue; '
                '[[ -d "$dir" ]] || exit 1; done\n', encoding='utf-8', newline='\n')
        env = dict(os.environ)
        env.update(SLURM_SUBMIT_DIR=bash_path(ROOT), BUDGET1_TRAIN_RUN_DIR=bash_path(run),
                   BUDGET1_EVAL_OUTPUT_DIR=bash_path(output), CAPTURE_FILE=bash_path(capture),
                   WANDB_RUN_ID='wrong_inherited_run', EVAL_STEPS='[20000]',
                   WANDB_DIR=bash_path(root / 'wandb'), WANDB_CACHE_DIR=bash_path(root / 'cache'))
        env.pop('BUDGET1_CONFIG', None)
        command = ['export PATH="$1:$PATH"; shift; exec bash scripts/eval_remaining_budget1.sh --dry-run "$@"']
        result = subprocess.run([find_bash(), '-c', *command, '_', bash_path(bin_dir), *extra],
                                cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=30)
        return result, capture, output

    def test_launcher_routes_snapshot_steps_run_and_durable_console_log(self):
        result, capture, output = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = capture.read_text(encoding='utf-8').splitlines()
        self.assertIn('src.eval_checkpoints', args)
        self.assertIn('++wandb.id=un2ifuj6', args)
        self.assertIn('+resume_eval.steps=[100000,120000,140000]', args)
        self.assertIn('+resume_eval.expected_decoder.budget_max=1.0', args)
        self.assertIn('+resume_eval.dry_run=true', args)
        self.assertIn('--config-name=config', args)
        self.assertTrue(any(a.startswith('--config-path=') and a.endswith('/.hydra') for a in args))
        self.assertFalse(any(a.startswith('+experiment=') for a in args))
        console = (output / 'console.log').read_text(encoding='utf-8')
        self.assertIn('tmdwo8814-/noposplat/un2ifuj6', console)
        self.assertIn('Saved training config:', console)

    def test_missing_snapshot_fails_with_local_log_before_python(self):
        result, capture, output = self.launch(missing_config=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(capture.exists())
        self.assertIn('Saved evaluation config not found',
                      (output / 'console.log').read_text(encoding='utf-8'))

    def test_remaining_step_override_is_forwarded(self):
        result, capture, _ = self.launch(extra=('--steps', '120000', '140000'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('+resume_eval.steps=[120000,140000]',
                      capture.read_text(encoding='utf-8').splitlines())


if __name__ == '__main__':
    unittest.main()
