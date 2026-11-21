"""Local scores, progress and CSV output without GPU or W&B dependencies."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.misc.supplementary_results import batch_status, export_results, read_metrics, save_checkpoint_metrics
from test_auto_eval_logging import Scalar


class SupplementaryResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'variants.txt').write_text('M1\nM2\n', encoding='utf-8')
        sums = {'psnr_ours': Scalar(75), 'ssim_ours': Scalar(2.4), 'lpips_ours': Scalar(.45)}
        self.model = SimpleNamespace(_test_metric_count=Scalar(3), _test_metric_sums=sums,
                                     _test_overlap_tags=('medium',),
                                     _test_overlap_sums={'medium': sums},
                                     _test_overlap_counts={'medium': Scalar(3)})

    def save(self, variant='M1', step=5000):
        with contextlib.redirect_stdout(io.StringIO()):
            return save_checkpoint_metrics(self.model, step, Path(f'epoch_0-step_{step}.ckpt'),
                                           self.root / variant)

    def test_scores_are_persisted_per_checkpoint_with_global_means(self):
        first, second = self.save(), self.save(step=10000)
        self.assertNotEqual(first, second)
        self.assertEqual(first, self.root / 'M1/evaluation/step_00005000/metrics.json')
        data = json.loads(first.read_text(encoding='utf-8'))
        self.assertEqual(data['checkpoint'], 'epoch_0-step_5000.ckpt')
        self.assertEqual(data['metrics']['auto_eval/ckpt_step'], 5000)
        self.assertEqual(data['metrics']['auto_eval/overall/psnr'], 25)
        self.assertAlmostEqual(data['metrics']['auto_eval/overall/ssim'], .8)
        self.assertEqual(data['metrics']['auto_eval/medium/num_samples'], 3)
        self.assertFalse(first.with_suffix('.json.tmp').exists())

    def test_empty_and_nonfinite_results_fail_without_creating_a_success_file(self):
        self.model._test_metric_count = Scalar(0)
        with self.assertRaisesRegex(RuntimeError, 'no scored samples'):
            self.save()
        self.model._test_metric_count = Scalar(3)
        self.model._test_metric_sums['psnr_ours'] = Scalar(float('nan'))
        with self.assertRaises(ValueError):
            self.save()
        self.assertFalse((self.root / 'M1/evaluation/step_00005000/metrics.json').exists())

    def test_status_tracks_pending_training_evaluation_completion_and_failure(self):
        self.assertEqual([row['state'] for row in batch_status(self.root)], ['pending', 'pending'])
        run = self.root / 'M1'
        run.mkdir()
        (run / 'RUNNING').touch()
        (run / 'console.log').write_text('train step 120; loss=0.1\n', encoding='utf-8')
        self.assertEqual(batch_status(self.root)[0]['state'], 'training')
        self.assertEqual(batch_status(self.root)[0]['train_step'], 120)
        (run / 'checkpoints').mkdir()
        (run / 'checkpoints/epoch_0-step_10000.ckpt').touch()
        self.assertEqual(batch_status(self.root)[0]['state'], 'evaluating')
        self.save()
        (run / 'COMPLETED').touch()
        self.assertEqual(batch_status(self.root)[0]['state'], 'incomplete')
        self.save(step=10000)
        self.assertEqual(batch_status(self.root)[0]['state'], 'complete')
        (run / 'FAILED').touch()
        self.assertEqual(batch_status(self.root)[0]['state'], 'failed')

    def test_csv_contains_actual_variant_attributes_and_each_evaluated_step(self):
        self.save()
        self.save(step=10000)
        self.save(variant='M2')
        path = export_results(self.root)
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(row['variant'], row['step']) for row in rows],
                         [('M1', '5000'), ('M1', '10000'), ('M2', '5000')])
        self.assertEqual([row['center'] for row in rows], ['R', 'R', 'F'])
        self.assertTrue(all(row['assignment'] == 'uniform' for row in rows))
        self.assertTrue(all(float(row['psnr']) == 25 for row in rows))

    def test_status_rejects_mislabeled_or_empty_score_files(self):
        path = self.save()
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['metrics']['auto_eval/ckpt_step'] = 10000
        path.write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Invalid evaluation result'):
            read_metrics(self.root / 'M1', 5000)
        payload['metrics']['auto_eval/ckpt_step'] = 5000
        payload['metrics']['auto_eval/overall/num_samples'] = 0
        path.write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Invalid evaluation result'):
            read_metrics(self.root / 'M1', 5000)

    def test_status_command_requires_both_evaluations_and_exports_csv(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/supplementary_status.py'
        spec = importlib.util.spec_from_file_location('supplementary_status_test', path)
        command = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(command)
        self.save()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                command.main([str(self.root), '--require-complete', 'M1'])
        self.assertEqual(failure.exception.code, 2)
        self.save(step=10000)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            command.main([str(self.root), '--write-csv', '--require-complete', 'M1'])
        self.assertIn('5000, 10000', output.getvalue())
        self.assertTrue((self.root / 'results.csv').exists())


if __name__ == '__main__':
    unittest.main()
