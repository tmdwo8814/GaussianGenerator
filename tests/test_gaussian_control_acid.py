"""Cross-domain configuration, staging, cache guards and shared-cohort reports.

CPU checks; fake child processes never import a backbone or invoke CUDA.
"""

from contextlib import redirect_stdout
import copy
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('acid_control', ROOT / 'scripts/run_gaussian_control_acid.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class AcidConfigurationTests(unittest.TestCase):
    def test_acid_inherits_the_re10k_checkpoint_model_loss_and_alignment(self):
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            original = compose(config_name='gaussian_control')
            acid = compose(config_name='gaussian_control_acid')
        for key in ('model', 'loss', 'test'):
            self.assertEqual(OmegaConf.to_container(original[key], resolve=True),
                             OmegaConf.to_container(acid[key], resolve=True))
        self.assertEqual(acid.checkpointing.load, original.checkpointing.load)
        self.assertIn('Final++_RE10k.ckpt', acid.checkpointing.load)
        self.assertEqual(list(acid.dataset), ['re10k'])  # ACID's existing loader alias
        self.assertEqual(list(acid.dataset.re10k.roots), ['/workspace/datasets/acid'])
        self.assertEqual(acid.dataset.re10k.view_sampler.index_path, 'assets/evaluation_index_acid.json')
        self.assertEqual(acid.dataset.re10k.view_sampler.num_context_views, 2)
        self.assertEqual(len([v for v in runner.read_json(ROOT / 'assets/evaluation_index_acid.json').values() if v]), 1341)


class AcidPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.weight = self.root / 'weights/Final++ RE10k.ckpt'
        self.weight.parent.mkdir()
        self.weight.write_bytes(b'frozen RE10K checkpoint')
        self.data = self.root / 'acid'
        (self.data / 'test').mkdir(parents=True)
        (self.data / 'test/a.torch').write_bytes(b'metadata fixture')
        self.index = self.root / 'acid_index.json'
        self.selected = {scene: {'context': [0, 4], 'target': [1, 2, 3], 'overlap': .2} for scene in ('a', 'b')}
        runner.save_json(self.index, self.selected)
        runner.save_json(self.data / 'test/index.json', {'a': 'a.torch', 'b': 'a.torch'})
        self.output = self.root / 're10k_to_acid'
        self.argv = ['--checkpoint', str(self.weight), '--dataset-root', str(self.data),
                     '--index-path', str(self.index), '--output-root', str(self.output), '--percentages', '100', '75']
        self.args = runner.normalize(runner.parser().parse_args(self.argv))

    def tearDown(self):
        self.temporary.cleanup()

    def cache(self):
        manifest = {
            'signature': 'acid-cache', 'complete': True, 'checkpoint': str(self.weight),
            'identity': {'dataset': {'re10k': {'roots': [str(self.data)]}}, 'index': self.selected},
            'scenes': {scene: {'key': runner.scene_key(scene, entry), 'entry': entry}
                       for scene, entry in self.selected.items()},
        }
        runner.save_json(self.args.cache_dir / 'manifest.json', manifest)
        for item in manifest['scenes'].values():
            path = self.args.cache_dir / 'scenes' / f"{item['key']}.pt"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'cached once')
        return manifest

    def results(self, phase, scenes=('a', 'b'), mismatch=False):
        manifest = runner.read_json(self.args.cache_dir / 'manifest.json')
        methods = ['lightgaussian'] if phase == 'lightgaussian' else ['prune', 'merge', 'redistribute']
        directory = self.args.light_dir if phase == 'lightgaussian' else self.args.controls_dir
        runner.save_json(directory / 'metadata.json', {
            'signature': phase, 'settings': {'cache_signature': manifest['signature']},
            'methods': methods, 'percentages': self.args.percentages,
        })
        for scene in scenes:
            baseline = 20. if scene == 'a' else 40.
            rows = []
            for method in methods:
                for p in self.args.percentages:
                    rows.append({'scene': scene, 'context_indices': [0, 4], 'target_indices': [1, 2, 3],
                                 'overlap_tag': 'small', 'method': method, 'percent': p,
                                 'gaussians': 32 * p // 100, 'actual_percent': p,
                                 'psnr': baseline if p == 100 else baseline - 4., 'ssim': .8, 'lpips': .2,
                                 'postprocess_ms': 0. if p == 100 else 4., 'render_ms_per_view': 1.})
            if mismatch:
                rows[0]['psnr'] += .001
            runner.save_json(directory / 'scenes' / f"{manifest['scenes'][scene]['key']}.json",
                             {'signature': phase, 'rows': rows})

    def test_commands_reuse_original_algorithms_and_pass_acid_to_both_stages(self):
        commands = runner.commands(self.args)
        self.assertIn('src.eval_gaussian_control', commands['cache'])
        self.assertIn('--config-name=gaussian_control_acid', commands['controls'])
        self.assertIn('control.methods=[prune,merge,redistribute]', commands['controls'])
        self.assertIn('control.stage=cache', commands['cache'])
        self.assertIn('control.stage=evaluate', commands['controls'])
        self.assertTrue(any('Final++ RE10k.ckpt' in arg for arg in commands['cache']))
        light = commands['lightgaussian']
        self.assertEqual(light[light.index('--baseline-results') + 1], str(self.args.controls_dir))
        self.assertEqual(light[light.index('--dataset-root') + 1], str(self.data))
        self.assertNotIn('--max-scenes', light)  # selected cache already has the exact subset
        # Exercise Hydra's actual override grammar, including Final++ and spaces.
        from hydra import initialize_config_dir, compose
        overrides = [arg for arg in commands['cache'] if '=' in arg and not arg.startswith('--')]
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            cfg = compose(config_name='gaussian_control_acid', overrides=overrides)
        self.assertEqual(cfg.checkpointing.load, self.weight.as_posix())
        self.assertEqual(cfg.dataset.re10k.roots[0], self.data.as_posix())
        self.assertEqual(cfg.control.output_dir, self.args.controls_dir.as_posix())
        self.assertEqual(cfg.control.stage, 'cache')

    def test_check_only_requires_no_gpu_and_creates_no_outputs(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), redirect_stdout(io.StringIO()), \
                patch.object(runner, 'run_child') as child:
            runner.main(self.argv + ['--check-only'])
        child.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_space_and_dataset_guards_fail_before_any_inference(self):
        with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError, 'cache needs'):
                runner.preflight(self.args)
        self.assertFalse(self.output.exists())
        manifest = self.cache()
        wrong = copy.deepcopy(manifest)
        wrong['identity']['dataset']['re10k']['roots'] = [str(self.root / 're10k')]
        runner.save_json(self.args.cache_dir / 'manifest.json', wrong)
        with self.assertRaisesRegex(ValueError, 'another dataset'):
            runner.preflight(self.args)

    def test_combined_report_uses_one_cohort_and_one_identical_100_baseline(self):
        self.cache()
        self.results('controls')
        self.results('lightgaussian', scenes=('a',))
        self.assertFalse(runner.collect(self.args))
        partial = runner.read_json(self.output / 'summary.json')
        self.assertTrue(all(row['scenes'] == 1 for row in partial))
        baseline = [r for r in partial if r['percent'] == 100 and r['overlap'] == 'overall']
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]['psnr'], 20.)
        self.results('lightgaussian')
        self.assertTrue(runner.collect(self.args))
        complete = runner.read_json(self.output / 'summary_status.json')
        self.assertEqual(complete['common_scenes'], 2)
        self.assertTrue((self.output / 'lightgaussian_075.txt').exists())
        self.assertTrue((self.output / 'merge_075.txt').exists())
        self.results('lightgaussian', mismatch=True)
        with self.assertRaisesRegex(ValueError, '100% baseline differs'):
            runner.collect(self.args)

    def test_pipeline_runs_cache_controls_light_then_resumes_without_cache_stage(self):
        phases = []
        fail = [True]
        def fake_child(phase, command, output):
            phases.append(phase)
            if phase == 'cache':
                self.cache()
            elif phase == 'controls':
                self.results(phase)
            else:
                if fail[0]:
                    raise subprocess.CalledProcessError(1, command)
                self.results(phase)
        with patch.object(runner, 'run_child', side_effect=fake_child), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.main(self.argv)
            self.assertEqual(phases, ['cache', 'controls', 'lightgaussian'])
            self.assertIn('phase=lightgaussian', (self.output / 'run_status.txt').read_text())
            cached = {p: p.read_bytes() for p in self.args.cache_dir.rglob('*') if p.is_file()}
            fail[0] = False
            self.weight.unlink()  # evaluation must not require/reload the backbone checkpoint
            runner.main(self.argv + ['--stage', 'evaluate'])
            self.assertEqual(phases[-2:], ['controls', 'lightgaussian'])
            self.assertEqual((self.output / 'run_status.txt').read_text(), 'complete\n')
            self.assertTrue(all(path.read_bytes() == value for path, value in cached.items()))

    def test_lightgaussian_requires_existing_shared_baselines(self):
        self.cache()
        with self.assertRaisesRegex(ValueError, 'controls stage first'):
            runner.require_shared_baseline(self.args)
        self.results('controls')
        runner.require_shared_baseline(self.args)


if __name__ == '__main__':
    unittest.main()
