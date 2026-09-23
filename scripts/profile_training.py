"""Short throughput + stage diagnosis using the existing training entry point.

Example: python -m scripts.profile_training +experiment=re10k_moment
No model equations are modified. Warm-up, then N throughput steps, then N
synchronized stage steps. JSON is written independently by each DDP rank.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch


class StageTimer:
    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize, self.clock = synchronize, clock
        self.enabled = False
        self.values = defaultdict(float)
        self.calls = defaultdict(int)
        self.originals = []

    def start(self):
        self.synchronize()
        return self.clock()

    def finish(self, name, started):
        self.synchronize()
        self.values[name] += self.clock() - started
        self.calls[name] += 1

    def wrap(self, owner, attribute, name):
        original = getattr(owner, attribute)
        was_local = attribute in vars(owner)

        @wraps(original)
        def timed(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            started = self.start()
            try:
                return original(*args, **kwargs)
            finally:
                self.finish(name, started)

        self.originals.append((owner, attribute, original, was_local))
        setattr(owner, attribute, timed)

    def restore(self):
        self.enabled = False
        for owner, attribute, original, was_local in reversed(self.originals):
            if was_local:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self.originals.clear()


def summarize(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: {'mean': statistics.mean(row.get(key, 0.) for row in rows),
                  'median': statistics.median(row.get(key, 0.) for row in rows),
                  'max': max(row.get(key, 0.) for row in rows)} for key in keys}


def make_callback(callback_base, args):
    class SpeedCallback(callback_base):
        def __init__(self):
            super().__init__()
            self.completed = 0
            self.previous_end = None
            self.rows = {'throughput': [], 'stages': []}
            self.peak_gib = {'throughput': 0., 'stages': 0.}
            self.last_phase = None
            self.timer = None
            self.context_shape = None

        def on_fit_start(self, trainer, model):
            if getattr(model, 'auxiliary', None) is not None:
                raise ValueError('Profile decoder-only first: use +experiment=re10k_moment')
            sync = lambda: torch.cuda.synchronize(model.device)
            self.timer = StageTimer(sync)
            self.timer.wrap(model, 'training_step', 'training_step_s')
            self.timer.wrap(model.encoder, 'forward', 'encoder_s')
            self.timer.wrap(model.decoder, 'forward', 'renderer_s')
            moment = getattr(model.encoder, 'gaussian_decoder', None)
            if moment is not None:
                self.timer.wrap(moment, 'forward', 'moment_total_s')
                self.timer.wrap(moment, 'predict_allocation', 'allocation_s')
                self.timer.wrap(moment, 'aggregate_moments', 'aggregation_s')
                self.timer.wrap(moment, 'build_gaussians', 'attributes_s')
                module = importlib.import_module(type(moment).__module__)
                self.timer.wrap(module, 'build_knn', 'knn_s')

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            phase = ('warmup' if self.completed < args.warmup else
                     'throughput' if self.completed < args.warmup + args.steps else 'stages')
            self.timer.synchronize()
            now = self.timer.clock()
            self.gap = None if self.previous_end is None else now - self.previous_end
            if phase != self.last_phase:
                torch.cuda.reset_peak_memory_stats(model.device)
                if trainer.is_global_zero:
                    print(f'[SpeedProbe] {phase}', flush=True)
            self.last_phase = phase
            self.timer.enabled = phase == 'stages'
            self.timer.values.clear()
            self.timer.calls.clear()
            if isinstance(batch, dict):
                self.context_shape = list(batch['context']['image'].shape)
            self.started = self.timer.clock()

        def on_before_backward(self, trainer, model, loss):
            if self.timer.enabled:
                self.backward_started = self.timer.start()

        def on_after_backward(self, trainer, model):
            if self.timer.enabled:
                self.timer.finish('backward_s', self.backward_started)

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            self.timer.synchronize()
            now = self.timer.clock()
            phase = self.last_phase
            if phase != 'warmup':
                row = {'compute_step_s': now - self.started}
                if self.gap is not None:
                    row['between_batches_s'] = self.gap
                    row['step_wall_s'] = self.gap + row['compute_step_s']
                if phase == 'stages':
                    row.update(self.timer.values)
                    row.update({key.removesuffix('_s') + '_calls': value
                                for key, value in self.timer.calls.items()})
                self.rows[phase].append(row)
                self.peak_gib[phase] = max(self.peak_gib[phase],
                    torch.cuda.max_memory_allocated(model.device) / 2**30)
            self.completed += 1
            self.previous_end = self.timer.clock()

        def on_train_end(self, trainer, model):
            self.timer.restore()
            moment_cfg = getattr(model.encoder.cfg, 'moment_decoder', None)
            report = {
                'rank': trainer.global_rank, 'world_size': trainer.world_size,
                'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(model.device),
                'head': getattr(model.encoder, 'gs_params_head_type', None),
                'context_shape_per_rank': self.context_shape,
                'moment_config': asdict(moment_cfg) if is_dataclass(moment_cfg) else None,
                'warmup_steps': args.warmup, 'requested_samples_per_phase': args.steps,
                'samples_per_phase': {key: len(value) for key, value in self.rows.items()},
                'peak_allocated_gib': self.peak_gib,
                'throughput': summarize(self.rows['throughput']),
                'stages': summarize(self.rows['stages']),
                'per_step': self.rows,
                'notes': ['Stage timings synchronize CUDA and include nested scopes; do not sum them.',
                          'Use throughput.step_wall_s for comparison, not the stage-profile total.',
                          'between_batches_s includes loader wait, transfer, and framework overhead.',
                          'DDP backward includes communication/wait; inspect every rank.'],
            }
            path = args.output.with_name(f'{args.output.stem}.rank{trainer.global_rank}.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'[SpeedProbe rank {trainer.global_rank}] saved {path.resolve()}', flush=True)
            if trainer.is_global_zero:
                print(json.dumps({key: report[key] for key in
                      ['samples_per_phase', 'peak_allocated_gib', 'throughput', 'stages']}, indent=2), flush=True)

        def on_exception(self, trainer, model, exception):
            if self.timer is not None:
                self.timer.restore()

    return SpeedCallback()


def parse_options(argv):
    # Lightning's DDP launcher re-executes this module after sys.argv has been
    # reduced to Hydra overrides. Propagate identical limits/output to children.
    inherited = json.loads(os.environ.get('GAUSSIAN_SPEED_PROBE_OPTIONS', '{}'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup', type=int, default=inherited.get('warmup', 2))
    parser.add_argument('--steps', type=int, default=inherited.get('steps', 3), help='Samples in EACH of throughput and stage phases')
    parser.add_argument('--output', type=Path, default=Path(inherited.get('output', 'outputs/speed_profile.json')))
    args, overrides = parser.parse_known_args(argv)
    if args.warmup < 1 or args.steps < 1:
        parser.error('--warmup and --steps must be positive')
    return args, overrides


def launch_training(entry):
    import hydra
    # Importing src.main changes Hydra's relative config lookup to pkg://config.
    # Use the SAME task/config with an absolute path so no config package is needed.
    config_dir = str(Path(entry.__file__).resolve().parents[1] / 'config')
    hydra.main(version_base=None, config_path=config_dir, config_name='main')(entry.train.__wrapped__)()


def main():
    args, overrides = parse_options(sys.argv[1:])
    if not torch.cuda.is_available():
        raise RuntimeError('Run this diagnostic in the CUDA training environment')
    args.output = args.output.resolve()
    previous_options = os.environ.get('GAUSSIAN_SPEED_PROBE_OPTIONS')
    os.environ['GAUSSIAN_SPEED_PROBE_OPTIONS'] = json.dumps({
        'warmup': args.warmup, 'steps': args.steps, 'output': str(args.output)})

    # Import the normal entry point before patching; its model/config setup stays intact.
    from lightning.pytorch import Callback
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    import src.main as entry

    original_trainer = entry.Trainer

    class ProfileTrainer(original_trainer):
        def __init__(self, *positional, **kwargs):
            kwargs['callbacks'] = [cb for cb in kwargs.get('callbacks', [])
                                   if not isinstance(cb, (ModelCheckpoint, LearningRateMonitor))]
            kwargs['callbacks'].append(make_callback(Callback, args))
            kwargs.update(enable_checkpointing=False, num_sanity_val_steps=0,
                          limit_val_batches=0, max_steps=args.warmup + 2 * args.steps)
            super().__init__(*positional, **kwargs)

    # Force diagnostic-only controls, including no resumed optimizer/step state.
    controls = {'mode': 'train', 'wandb.mode': 'disabled', 'trainer.auto_eval': 'false',
                'trainer.val_check_interval': 'null', 'checkpointing.load': 'null'}
    overrides = [item for item in overrides if item.split('=', 1)[0].lstrip('+') not in controls]
    if not any(item.split('=', 1)[0].lstrip('+') == 'experiment' for item in overrides):
        overrides.insert(0, '+experiment=re10k_moment')
    sys.argv = [sys.argv[0], *overrides, *(f'{key}={value}' for key, value in controls.items())]
    entry.Trainer = ProfileTrainer
    try:
        launch_training(entry)
    finally:
        entry.Trainer = original_trainer
        if previous_options is None:
            os.environ.pop('GAUSSIAN_SPEED_PROBE_OPTIONS', None)
        else:
            os.environ['GAUSSIAN_SPEED_PROBE_OPTIONS'] = previous_options


if __name__ == '__main__':
    main()
