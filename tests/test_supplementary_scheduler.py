"""Exercise the GPU queue with real, short CPU child processes."""

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.misc.supplementary_results import batch_status


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('supplementary_scheduler_test',
                                            ROOT / 'scripts/train_re10k_supplementary.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)

WORKER = '''\
import json
import os
from pathlib import Path
import sys
import time

run = Path(sys.argv[1])
mode = sys.argv[2]
model = run.name
started = time.time()
(run / 'started.json').write_text(json.dumps(dict(time=started, gpu=os.environ['CUDA_VISIBLE_DEVICES'])))
if mode == 'queue':
    deadline = time.monotonic() + 10
    if model == 'M2':
        while not all((run.parent / f'M{i}' / 'started.json').exists() for i in range(1, 5)):
            if time.monotonic() > deadline:
                sys.exit(3)
            time.sleep(.01)
    elif model in ('M1', 'M3', 'M4'):
        while not (run.parent / 'M5' / 'started.json').exists():
            if time.monotonic() > deadline:
                sys.exit(4)
            time.sleep(.01)
    else:
        time.sleep(.05)
elif mode == 'long':
    time.sleep(30)
elif mode == 'failed' and model == 'M1':
    sys.exit(7)
for step in (5000, 10000):
    if mode == 'missing' and model == 'M1' and step == 10000:
        continue
    metrics = {'auto_eval/ckpt_step': step, 'auto_eval/overall/num_samples': 3,
               'auto_eval/overall/psnr': 25, 'auto_eval/overall/ssim': .8, 'auto_eval/overall/lpips': .15}
    path = run / 'evaluation' / f'step_{step:08d}' / 'metrics.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}' if mode == 'invalid' and model == 'M1' else json.dumps({'metrics': metrics}))
    time.sleep(.02)
(run / 'ended.json').write_text(json.dumps(dict(time=time.time())))
print('train step 10000; fake test worker completed')
'''


class SupplementarySchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = self.root / 'batch'
        self.checkpoint = self.root / 'encoder.ckpt'
        self.checkpoint.touch()
        self.worker = self.root / 'worker.py'
        self.worker.write_text(WORKER, encoding='utf-8')

    def run_jobs(self, models, gpus, mode):
        scheduler.prepare_batch(self.batch, models, gpus, self.checkpoint)
        factory = lambda model, gpu, run_dir: [sys.executable, '-u', str(self.worker), str(run_dir), mode]
        with contextlib.redirect_stdout(io.StringIO()):
            return scheduler.run_queue(models, gpus, self.batch, self.checkpoint,
                                       command_factory=factory, poll_interval=.01)

    def test_four_independent_gpus_reuse_first_free_gpu_after_both_evaluations(self):
        models, gpus = [f'M{i}' for i in range(1, 8)], ['0', '1', '2', '3']
        self.assertEqual(self.run_jobs(models, gpus, 'queue'), 0)
        records = {}
        for model in models:
            run = self.batch / model
            start = json.loads((run / 'started.json').read_text())
            end = json.loads((run / 'ended.json').read_text())
            records[model] = dict(gpu=start['gpu'], start=start['time'], end=end['time'])
            self.assertEqual((run / 'gpu.txt').read_text().strip(), start['gpu'])
            self.assertTrue((run / 'COMPLETED').exists())
            self.assertFalse((run / 'RUNNING').exists())
            for step in (5000, 10000):
                self.assertTrue((run / 'evaluation' / f'step_{step:08d}' / 'metrics.json').is_file())
        self.assertEqual([records[model]['gpu'] for model in models[:4]], gpus)
        # M2 is deliberately the first to finish. Other initial jobs wait for M5.
        self.assertEqual(records['M5']['gpu'], '1')
        self.assertGreaterEqual(records['M5']['start'], records['M2']['end'])
        self.assertLess(records['M5']['start'], records['M1']['end'])
        for gpu in gpus:
            jobs = sorted((row for row in records.values() if row['gpu'] == gpu), key=lambda row: row['start'])
            for before, after in zip(jobs, jobs[1:]):
                self.assertGreaterEqual(after['start'], before['end'])
        rows = batch_status(self.batch)
        self.assertTrue(all(row['state'] == 'complete' for row in rows))
        with (self.batch / 'results.csv').open(newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 14)

    def test_failed_missing_or_invalid_evaluation_does_not_block_other_models(self):
        for mode in ('failed', 'missing', 'invalid'):
            with self.subTest(mode=mode):
                self.batch = self.root / mode
                self.assertEqual(self.run_jobs(['M1', 'M2', 'M3'], ['2'], mode), 1)
                self.assertTrue((self.batch / 'M1/FAILED').exists())
                self.assertFalse((self.batch / 'M1/COMPLETED').exists())
                self.assertTrue((self.batch / 'M2/COMPLETED').exists())
                self.assertTrue((self.batch / 'M3/COMPLETED').exists())
                self.assertEqual([row['state'] for row in batch_status(self.batch)],
                                 ['failed', 'complete', 'complete'])

    def test_interrupt_stops_active_children_and_leaves_queued_model_unstarted(self):
        scheduler.prepare_batch(self.batch, ['M1', 'M2', 'M3'], ['0', '1'], self.checkpoint)
        factory = lambda model, gpu, run_dir: [sys.executable, '-u', str(self.worker), str(run_dir), 'long']
        processes = []
        real_popen = subprocess.Popen

        def capture_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with contextlib.redirect_stdout(io.StringIO()), patch.object(scheduler.subprocess, 'Popen', capture_process):
            with patch.object(scheduler.time, 'sleep', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    scheduler.run_queue(['M1', 'M2', 'M3'], ['0', '1'], self.batch, self.checkpoint,
                                        command_factory=factory)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertTrue((self.batch / 'M1/FAILED').exists())
        self.assertTrue((self.batch / 'M2/FAILED').exists())
        self.assertFalse((self.batch / 'M3').exists())

    def test_command_pins_one_gpu_batch16_and_uses_separate_model_output(self):
        command = scheduler.training_command('M5', '3', self.batch / 'M5', self.checkpoint,
                                             ['trainer.devices=4', 'data_loader.train.batch_size=2', 'wandb.mode=offline'])
        self.assertEqual(command[:4], [sys.executable, '-u', '-m', 'src.main'])
        self.assertIn('+experiment=re10k_supp_m5', command)
        self.assertIn(f'hydra.run.dir={self.batch / "M5"}', command)
        self.assertIn('wandb.mode=offline', command)
        self.assertEqual(command[-2:], ['trainer.devices=1', 'data_loader.train.batch_size=16'])

    def test_invalid_plan_or_existing_batch_is_rejected_before_launch(self):
        for models, gpus in [([], ['0']), (['M1', 'M1'], ['0']), (['M8'], ['0']),
                             (['M1'], []), (['M1'], ['0', '0']), (['M1'], ['-1'])]:
            with self.subTest(models=models, gpus=gpus), self.assertRaises(ValueError):
                scheduler.prepare_batch(self.batch, models, gpus, self.checkpoint)
        self.assertFalse(self.batch.exists())
        scheduler.prepare_batch(self.batch, ['M1'], ['0'], self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'fresh RUN_ROOT'):
            scheduler.prepare_batch(self.batch, ['M1'], ['0'], self.checkpoint)


if __name__ == '__main__':
    unittest.main()
