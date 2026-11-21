"""Local supplementary scores and batch status, without W&B/network imports."""

import csv
import json
import math
from pathlib import Path
import re


VARIANTS = {
    'M1': ('uniform', 'RRR'), 'M2': ('uniform', 'FFF'),
    'M3': ('learned', 'RRR'), 'M4': ('learned', 'FFF'),
    'M5': ('learned', 'RFF'), 'M6': ('learned', 'FRF'), 'M7': ('learned', 'FFR'),
}
EVAL_STEPS = (5000, 10000)


def save_checkpoint_metrics(model, step: int, checkpoint: Path, run_dir: Path) -> Path:
    """Called by rank zero AFTER on_test_end has globally reduced the scores."""
    if model._test_metric_count.item() == 0:
        raise RuntimeError(f'Supplementary step {step}: evaluation produced no scored samples')
    metrics = {'auto_eval/ckpt_step': step}
    groups = [('overall', model._test_metric_sums, model._test_metric_count)]
    groups += [(tag, model._test_overlap_sums[tag], model._test_overlap_counts[tag])
               for tag in model._test_overlap_tags]
    for tag, sums, count in groups:
        if count.item() == 0:
            continue
        metrics[f'auto_eval/{tag}/num_samples'] = int(count.item())
        for name, total in sums.items():
            metrics[f'auto_eval/{tag}/{name.removesuffix("_ours")}'] = (total / count).item()
    path = run_dir / 'evaluation' / f'step_{step:08d}' / 'metrics.json'
    payload = json.dumps({'checkpoint': str(checkpoint), 'metrics': metrics}, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(path)
    print(f'[Supplementary] Saved evaluation step {step}: {path}', flush=True)
    return path


def read_metrics(run_dir: Path, step: int):
    path = run_dir / 'evaluation' / f'step_{step:08d}' / 'metrics.json'
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text(encoding='utf-8'))['metrics']
    if metrics.get('auto_eval/ckpt_step') != step or metrics.get('auto_eval/overall/num_samples', 0) <= 0:
        raise ValueError(f'Invalid evaluation result: {path}')
    for name in ('psnr', 'ssim', 'lpips'):
        value = metrics.get(f'auto_eval/overall/{name}')
        if value is None or not math.isfinite(value):
            raise ValueError(f'Missing or non-finite {name}: {path}')
    return metrics


def batch_status(batch_dir: Path):
    plan = batch_dir / 'variants.txt'
    selected = plan.read_text(encoding='utf-8').split() if plan.is_file() else list(VARIANTS)
    rows = []
    for variant in selected:
        if variant not in VARIANTS:
            raise ValueError(f'Unknown supplementary variant: {variant}')
        run_dir = batch_dir / variant
        evaluated = [step for step in EVAL_STEPS if read_metrics(run_dir, step) is not None]
        checkpoint_steps = []
        for path in (run_dir / 'checkpoints').glob('*.ckpt'):
            match = re.search(r'step[_=](\d+)', path.name)
            if match:
                checkpoint_steps.append(int(match[1]))
        log = run_dir / 'console.log'
        tail = ''
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 262144))
                tail = stream.read().decode('utf-8', errors='replace')
        steps = [int(step) for step in re.findall(r'train step (\d+)', tail)] + checkpoint_steps
        progress = max(steps, default=0)
        if (run_dir / 'FAILED').exists():
            state = 'failed'
        elif (run_dir / 'COMPLETED').exists():
            state = 'complete' if evaluated == list(EVAL_STEPS) else 'incomplete'
        elif (run_dir / 'RUNNING').exists():
            state = 'evaluating' if progress >= 10000 or '[PostEval]' in tail else 'training'
        else:
            state = 'pending'
        rows.append({'variant': variant, 'state': state, 'train_step': progress,
                     'evaluated_steps': evaluated, 'run_dir': run_dir})
    return rows


def export_results(batch_dir: Path) -> Path:
    path = batch_dir / 'results.csv'
    fields = ['variant', 'assignment', 'center', 'covariance', 'opacity', 'step',
              'psnr', 'ssim', 'lpips', 'num_samples', 'run_dir']
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in batch_status(batch_dir):
            variant = row['variant']
            assignment, modes = VARIANTS[variant]
            for step in row['evaluated_steps']:
                metrics = read_metrics(row['run_dir'], step)
                writer.writerow(dict(variant=variant, assignment=assignment,
                                     center=modes[0], covariance=modes[1], opacity=modes[2], step=step,
                                     psnr=metrics['auto_eval/overall/psnr'],
                                     ssim=metrics['auto_eval/overall/ssim'],
                                     lpips=metrics['auto_eval/overall/lpips'],
                                     num_samples=metrics['auto_eval/overall/num_samples'],
                                     run_dir=str(row['run_dir'])))
    temporary.replace(path)
    return path
