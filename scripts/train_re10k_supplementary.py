"""Schedule independent RE10K ablations on the first available single GPU."""

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.misc.supplementary_results import EVAL_STEPS, VARIANTS, export_results, read_metrics


DEFAULT_CHECKPOINT = '/workspace/research/GaussianGenerator/pretrained_weights/re10k.ckpt'


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def prepare_batch(batch_dir, models, gpus, checkpoint):
    if not checkpoint.is_file():
        raise ValueError(f'NoPoSplat checkpoint not found: {checkpoint}')
    if not models or len(set(models)) != len(models) or any(model not in VARIANTS for model in models):
        raise ValueError('MODELS must contain distinct variants from M1-M7')
    if not gpus or any(not gpu or gpu == '-1' for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('CUDA_VISIBLE_DEVICES must contain distinct GPU IDs, for example 0,1,2,3')
    for model in models:
        if (batch_dir / model).exists():
            raise ValueError(f'Existing model directory: {batch_dir / model}. Use a fresh RUN_ROOT.')
    if (batch_dir / 'variants.txt').exists():
        raise ValueError(f'Existing batch plan: {batch_dir}. Use a fresh RUN_ROOT.')
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / 'variants.txt').write_text(''.join(f'{model}\n' for model in models), encoding='utf-8')
    (batch_dir / 'encoder_checkpoint.txt').write_text(f'{checkpoint}\n', encoding='utf-8')
    (batch_dir / 'gpus.txt').write_text(''.join(f'{gpu}\n' for gpu in gpus), encoding='utf-8')


def training_command(model, gpu, run_dir, checkpoint, overrides):
    return [sys.executable, '-u', '-m', 'src.main', f'+experiment=re10k_supp_{model.lower()}',
            f'model.encoder.pretrained_weights={checkpoint}', f'hydra.run.dir={run_dir}',
            f'test.output_path={run_dir / "test"}', 'wandb.mode=online', *overrides,
            'trainer.devices=1', 'data_loader.train.batch_size=16']


def log_event(batch_dir, message):
    line = f'[{timestamp()}] {message}'
    print(line, flush=True)
    with (batch_dir / 'scheduler.log').open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')


def launch_job(model, gpu, batch_dir, command_factory):
    run_dir = batch_dir / model
    run_dir.mkdir()
    (run_dir / 'gpu.txt').write_text(f'{gpu}\n', encoding='utf-8')
    (run_dir / 'RUNNING').write_text(timestamp() + '\n', encoding='utf-8')
    command = command_factory(model, gpu, run_dir)
    environment = os.environ.copy()
    # Each child sees exactly one physical GPU, mapped to its local cuda:0.
    environment['CUDA_VISIBLE_DEVICES'] = gpu
    environment['PYTHONUNBUFFERED'] = '1'
    environment.setdefault('OMP_NUM_THREADS', '1')
    stream = (run_dir / 'console.log').open('w', encoding='utf-8')
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=(os.name != 'nt'))
    except BaseException:
        stream.close()
        (run_dir / 'FAILED').write_text('Failed to start training process\n', encoding='utf-8')
        (run_dir / 'RUNNING').unlink(missing_ok=True)
        raise
    job = dict(model=model, gpu=gpu, run_dir=run_dir, process=process, stream=stream)
    try:
        (run_dir / 'launch.json').write_text(
            json.dumps(dict(gpu=gpu, pid=process.pid, command=command, started_at=timestamp()), indent=2),
            encoding='utf-8')
        log_event(batch_dir, f'START {model} GPU={gpu} PID={process.pid}; log={run_dir / "console.log"}')
    except BaseException:
        stop_jobs([job])
        raise
    return job


def finish_job(job, batch_dir):
    process, run_dir = job['process'], job['run_dir']
    job['stream'].close()
    reason = f'exit_code={process.returncode}' if process.returncode else ''
    if not reason:
        try:
            if any(read_metrics(run_dir, step) is None for step in EVAL_STEPS):
                reason = 'Both 5k and 10k evaluation score files are required'
        except (OSError, ValueError, KeyError, TypeError) as error:
            reason = f'Invalid evaluation results: {error}'
    marker = 'FAILED' if reason else 'COMPLETED'
    (run_dir / marker).write_text((reason or timestamp()) + '\n', encoding='utf-8')
    (run_dir / 'RUNNING').unlink(missing_ok=True)
    log_event(batch_dir, f'{marker} {job["model"]} GPU={job["gpu"]}' + (f': {reason}' if reason else ''))
    return not reason


def stop_jobs(jobs, timeout=10):
    """Stop entire Linux process groups, including data-loader subprocesses."""
    jobs = list(jobs)
    for job in jobs:
        process = job['process']
        if process.poll() is None:
            try:
                if os.name == 'nt':
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    for job in jobs:
        process = job['process']
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                if os.name == 'nt':
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        job['stream'].close()
        (job['run_dir'] / 'FAILED').write_text('Scheduler interrupted; job stopped\n', encoding='utf-8')
        (job['run_dir'] / 'RUNNING').unlink(missing_ok=True)


def run_queue(models, gpus, batch_dir, checkpoint, overrides=(), *, command_factory=None, poll_interval=.2):
    if command_factory is None:
        command_factory = lambda model, gpu, run_dir: training_command(model, gpu, run_dir, checkpoint, overrides)
    pending, active = deque(models), {}
    failures = []
    try:
        export_results(batch_dir)
        while pending or active:
            # Fill every free GPU before polling completions: M1/M2/M3/M4
            # always start on GPU 0/1/2/3 respectively with the default plan.
            for gpu in gpus:
                if pending and gpu not in active:
                    model = pending.popleft()
                    active[gpu] = launch_job(model, gpu, batch_dir, command_factory)
            completed = [(gpu, job) for gpu, job in active.items() if job['process'].poll() is not None]
            for gpu, job in completed:
                if not finish_job(job, batch_dir):
                    failures.append(job['model'])
                del active[gpu]
            if completed:
                # Only the scheduler writes the shared CSV automatically.
                export_results(batch_dir)
                continue
            if active:
                time.sleep(poll_interval)
    except BaseException:
        stop_jobs(active.values())
        raise
    log_event(batch_dir, f'FINISHED {len(models) - len(failures)}/{len(models)} succeeded'
              + (f'; failed={",".join(failures)}' if failures else ''))
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('overrides', nargs=argparse.REMAINDER, help='Additional Hydra overrides for every model')
    args = parser.parse_args(argv)
    models = os.environ.get('MODELS', 'M1 M2 M3 M4 M5 M6 M7').split()
    gpus = [gpu.strip() for gpu in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')]
    checkpoint = Path(os.environ.get('ENCODER_CHECKPOINT', DEFAULT_CHECKPOINT)).expanduser().resolve()
    default_root = ROOT / 'outputs' / 'supplementary' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    batch_dir = Path(os.environ.get('RUN_ROOT', str(default_root))).expanduser().resolve()
    try:
        prepare_batch(batch_dir, models, gpus, checkpoint)
    except ValueError as error:
        parser.error(str(error))
    for variable, default in [('WANDB_DIR', ROOT / 'wandb'), ('WANDB_CACHE_DIR', ROOT / '.cache' / 'wandb')]:
        Path(os.environ.setdefault(variable, str(default))).mkdir(parents=True, exist_ok=True)
    log_event(batch_dir, f'Supplementary batch: {batch_dir}; GPUs={",".join(gpus)}; batch=16/model; devices=1/model')
    try:
        return run_queue(models, gpus, batch_dir, checkpoint, args.overrides)
    except KeyboardInterrupt:
        log_event(batch_dir, 'INTERRUPTED; active jobs stopped; queued jobs were not started')
        return 130


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
