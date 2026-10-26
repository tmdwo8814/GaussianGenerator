#!/usr/bin/env python3
"""Run the unchanged RE10K control evaluators on RE10K -> ACID predictions.

Stages use separate processes to release the backbone before timed evaluation.
No core evaluator/encoder sources are changed, so existing RE10K jobs can resume.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('prune', 'lightgaussian', 'merge', 'redistribute')
PERCENTAGES = (100, 75, 50, 25, 10, 5, 1)
KST = timezone(timedelta(hours=9), 'KST')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checkpoint', type=Path, default=Path('/workspace/research/GaussianGenerator/pretrained_weights/Final++_RE10k.ckpt'))
    cli.add_argument('--dataset-root', type=Path, default=Path('/workspace/datasets/acid'))
    cli.add_argument('--index-path', type=Path, default=ROOT / 'assets/evaluation_index_acid.json')
    cli.add_argument('--output-root', type=Path, default=ROOT / 'outputs/gaussian_control/re10k_to_acid')
    cli.add_argument('--cache-dir', type=Path, help='Optional separate disk/NAS for the large ACID cache')
    cli.add_argument('--stage', choices=['all', 'cache', 'evaluate', 'controls', 'lightgaussian', 'report'], default='all')
    cli.add_argument('--percentages', type=int, nargs='+', default=list(PERCENTAGES))
    cli.add_argument('--max-scenes', type=int)
    cli.add_argument('--timing-repeats', type=int, default=1)
    cli.add_argument('--warmup-runs', type=int, default=1)
    cli.add_argument('--timing-mode', choices=['concurrent', 'isolated'], default='concurrent')
    cli.add_argument('--check-only', action='store_true', help='Check paths/index/cache space and print commands, without writing outputs')
    return cli


def normalize(args):
    for name in ('checkpoint', 'dataset_root', 'index_path', 'output_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.cache_dir = (args.cache_dir or args.output_root / 'cache').expanduser().resolve()
    args.controls_dir = args.output_root / 'controls'
    args.light_dir = args.output_root / 'lightgaussian'
    if (not args.percentages or len(set(args.percentages)) != len(args.percentages)
            or 100 not in args.percentages or any(not 1 <= p <= 100 for p in args.percentages)):
        raise ValueError('Use distinct integer percentages in [1,100], including shared 100%')
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError('--max-scenes must be positive')
    if args.timing_repeats < 1 or args.warmup_runs < 0:
        raise ValueError('Invalid timing settings')
    for marker in ('manifest.json', 'decoder.pt', 'metadata.json', 'analysis_config.yaml'):
        if (args.output_root / marker).exists():
            raise ValueError('Output root is an existing cache/results directory; use a separate cross-domain root')
    if args.cache_dir == args.output_root or args.cache_dir in args.output_root.parents:
        raise ValueError('The cache must not contain the pipeline root')
    for directory in (args.controls_dir, args.light_dir, args.dataset_root):
        if args.cache_dir == directory or args.cache_dir in directory.parents or directory in args.cache_dir.parents:
            raise ValueError('Cache must be separate from dataset and result directories')
    if (args.output_root == args.dataset_root or args.dataset_root in args.output_root.parents
            or args.output_root in args.dataset_root.parents):
        raise ValueError('Do not write experiment outputs inside the dataset')
    return args


def selected_scenes(args):
    selected = {scene: entry for scene, entry in sorted(read_json(args.index_path).items()) if entry is not None}
    if args.max_scenes is not None:
        selected = dict(list(selected.items())[:args.max_scenes])
    if not selected:
        raise ValueError('ACID index contains no selected scenes')
    for scene, entry in selected.items():
        if len(entry['context']) != 2 or not entry['target']:
            raise ValueError(f'Expected two context views and nonempty targets: {scene}')
        if any(type(index) is not int or index < 0 for index in entry['context'] + entry['target']):
            raise ValueError(f'Invalid frame indices: {scene}')
    return selected


def scene_key(scene, entry):
    value = {'scene': scene, 'context': list(entry['context']), 'target': list(entry['target'])}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_manifest(args, selected):
    path = args.cache_dir / 'manifest.json'
    if not path.is_file():
        return None
    manifest = read_json(path)
    cfg = manifest['identity']['dataset']['re10k']
    if ([Path(root).resolve() for root in cfg['roots']] != [args.dataset_root]
            or manifest['identity']['index'] != selected):
        raise ValueError('Existing cache uses another dataset/index/subset; choose a separate ACID cache')
    if Path(manifest['checkpoint']).resolve() != args.checkpoint:
        raise ValueError('Existing cache uses another checkpoint path')
    return manifest


def preflight(args):
    selected = selected_scenes(args)
    manifest = validate_manifest(args, selected)
    if args.stage in ('all', 'cache') and not args.checkpoint.is_file():
        raise FileNotFoundError(f'RE10K-trained Final++ checkpoint missing: {args.checkpoint}')
    if args.stage in ('all', 'cache', 'evaluate', 'lightgaussian'):
        index = args.dataset_root / 'test/index.json'
        if not index.is_file():
            raise FileNotFoundError(f'ACID metadata missing: {index}')
        dataset_index = read_json(index)
        absent = set(selected) - dataset_index.keys()
        if absent:
            raise ValueError(f'{len(absent)} ACID evaluation scenes absent from dataset index: {sorted(absent)[:3]}')
        missing_chunks = [scene for scene in selected
                          if not (args.dataset_root / 'test' / dataset_index[scene]).is_file()]
        if missing_chunks:
            raise FileNotFoundError(f'ACID chunks missing, e.g. {missing_chunks[:3]}')
    missing = []
    sizes = []
    for scene, entry in selected.items():
        path = args.cache_dir / 'scenes' / f'{scene_key(scene, entry)}.pt'
        if path.is_file():
            sizes.append(path.stat().st_size)
        else:
            missing.append(scene)
    if args.stage in ('evaluate', 'controls', 'lightgaussian') and (not manifest or not manifest.get('complete') or missing):
        raise ValueError('ACID cache incomplete; run --stage cache or all first')
    ancestor = args.cache_dir
    while not ancestor.exists():
        ancestor = ancestor.parent
    # Same V=2,H=W=256,feature_dim=256 and 3 targets as the RE10K cache.
    scene_bytes = max([int(210.3 * 1024**2)] + sizes)
    required = math.ceil(len(missing) * scene_bytes * 1.05) + 2 * 1024**3
    free = shutil.disk_usage(ancestor).free
    if args.stage in ('all', 'cache') and free < required:
        raise RuntimeError(f'ACID cache needs about {required / 1024**3:.1f} GiB more; '
                           f'only {free / 1024**3:.1f} GiB free. Set CACHE_DIR to another disk.')
    print(f'[ACID] {len(selected)} scenes; {len(missing)} not cached; '
          f'estimated cache total {len(selected) * scene_bytes / 1024**3:.1f} GiB; '
          f'free {free / 1024**3:.1f} GiB', flush=True)
    return selected


def commands(args):
    quote = lambda path: json.dumps(path.as_posix(), ensure_ascii=False)
    shared = [sys.executable, '-u', '-m', 'src.eval_gaussian_control', '--config-name=gaussian_control_acid',
              f'checkpointing.load={quote(args.checkpoint)}', f'dataset.re10k.roots=[{quote(args.dataset_root)}]',
              f'dataset.re10k.view_sampler.index_path={quote(args.index_path)}',
              f'control.cache_dir={quote(args.cache_dir)}', f'control.output_dir={quote(args.controls_dir)}',
              'control.methods=[prune,merge,redistribute]', f'control.percentages={json.dumps(args.percentages)}',
              f'control.max_scenes={args.max_scenes if args.max_scenes is not None else "null"}',
              f'control.timing_repeats={args.timing_repeats}', f'control.warmup_runs={args.warmup_runs}',
              'control.render_chunk_size=1', 'control.save_variants=false']
    light = [sys.executable, '-u', '-m', 'src.eval_lightgaussian_control',
             '--cache-dir', str(args.cache_dir), '--output-dir', str(args.light_dir),
             '--baseline-results', str(args.controls_dir), '--dataset-root', str(args.dataset_root),
             '--threads', '1', '--timing-mode', args.timing_mode,
             '--timing-repeats', str(args.timing_repeats), '--warmup-runs', str(args.warmup_runs),
             '--percentages', *map(str, args.percentages)]
    # The cache manifest already contains the chosen subset; do not truncate twice.
    return {'cache': shared + ['control.stage=cache'], 'controls': shared + ['control.stage=evaluate'],
            'lightgaussian': light}


def run_child(phase, command, output):
    log = output / 'logs' / f'{phase}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'[ACID] Starting {phase}; log={log}', flush=True)
    with log.open('a', encoding='utf-8') as handle:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            child.stdout.close()
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def read_phase_rows(directory, manifest, methods, percentages):
    path = directory / 'metadata.json'
    if not path.exists():
        return {}
    metadata = read_json(path)
    if (metadata['settings']['cache_signature'] != manifest['signature']
            or metadata['methods'] != list(methods) or metadata['percentages'] != list(percentages)):
        raise ValueError(f'Phase results have different cache/methods/budgets: {directory}')
    result = {}
    for scene, item in manifest['scenes'].items():
        path = directory / 'scenes' / f"{item['key']}.json"
        if not path.exists():
            continue
        value = read_json(path)
        if value['signature'] != metadata['signature']:
            raise ValueError(f'Stale phase result: {path}')
        for row in value['rows']:
            if (row['scene'] != scene or row['context_indices'] != list(item['entry']['context'])
                    or row['target_indices'] != list(item['entry']['target'])):
                raise ValueError(f'Frame identity differs: {path}')
            result[scene, row['method'], row['percent']] = row
    return result


def require_shared_baseline(args):
    manifest = read_json(args.cache_dir / 'manifest.json')
    rows = read_phase_rows(args.controls_dir, manifest, ('prune', 'merge', 'redistribute'), args.percentages)
    if any((scene, 'prune', 100) not in rows for scene in manifest['scenes']):
        raise ValueError('Complete the controls stage first: LightGaussian must reuse every ACID 100% baseline')


def collect(args):
    manifest = read_json(args.cache_dir / 'manifest.json')
    rows = read_phase_rows(args.controls_dir, manifest, ('prune', 'merge', 'redistribute'), args.percentages)
    rows.update(read_phase_rows(args.light_dir, manifest, ('lightgaussian',), args.percentages))
    common = [scene for scene in manifest['scenes'] if all(
        (scene, method, percent) in rows for method in METHODS for percent in args.percentages)]
    selected = []
    for scene in common:
        baseline = rows[scene, 'prune', 100]
        for method in METHODS:
            if any(rows[scene, method, 100][metric] != baseline[metric] for metric in ('psnr', 'ssim', 'lpips')):
                raise ValueError(f'ACID shared 100% baseline differs: {scene}, {method}')
        selected.append({**baseline, 'method': 'final'})
        selected.extend(rows[scene, method, p] for p in args.percentages if p != 100 for method in METHODS)
    summary = []
    for method, p in [('final', 100)] + [(m, p) for p in args.percentages if p != 100 for m in METHODS]:
        for overlap in ('overall', 'small', 'medium', 'large'):
            subset = [r for r in selected if r['method'] == method and r['percent'] == p
                      and (overlap == 'overall' or r['overlap_tag'] == overlap)]
            if not subset:
                continue
            item = {'method': method, 'percent': p, 'overlap': overlap, 'scenes': len(subset)}
            for key in ('gaussians', 'actual_percent', 'psnr', 'ssim', 'lpips', 'postprocess_ms', 'render_ms_per_view'):
                item[key] = statistics.fmean(r[key] for r in subset)
            times = sorted(r['postprocess_ms'] for r in subset)
            item['postprocess_ms_p50'] = statistics.median(times)
            # Same linear percentile interpolation used by np.percentile(...,95).
            position = .95 * (len(times) - 1)
            lower, upper = math.floor(position), math.ceil(position)
            item['postprocess_ms_p95'] = times[lower] + (position - lower) * (times[upper] - times[lower])
            summary.append(item)
    preamble = (
        'RE10K-trained Final++ -> ACID; no model training.\n'
        f'Common completed scenes (all four methods/all budgets): {len(common)}/{len(manifest["scenes"])}\n'
        'Partial summaries use ONLY the same completed scenes for every method.\n'
        '100% is one shared cached/aligned Final++ baseline.\n'
        f'Timing mode: {args.timing_mode}; concurrent CPU/disk workloads can affect wall time.\n'
        'Postprocess excludes inference/cache I/O/camera preparation/NVS rendering/metrics.\n'
        'Includes LightGaussian scoring renders and Redistribute allocation/re-decoding.\n'
    )
    header = 'method percent overlap scenes Gaussians actual_percent PSNR SSIM LPIPS post_ms post_p50_ms post_p95_ms render_ms_per_view\n'
    def line(r):
        return (f"{r['method']:14s} {r['percent']:3d} {r['overlap']:7s} {r['scenes']:5d} {r['gaussians']:.1f} "
                f"{r['actual_percent']:.4f} {r['psnr']:.6f} {r['ssim']:.6f} {r['lpips']:.6f} "
                f"{r['postprocess_ms']:.4f} {r['postprocess_ms_p50']:.4f} {r['postprocess_ms_p95']:.4f} "
                f"{r['render_ms_per_view']:.4f}\n")
    reports = {'summary.txt': summary}
    for method, p in [('final', 100)] + [(m, p) for p in args.percentages if p != 100 for m in METHODS]:
        reports[f'{method}_{p:03d}.txt'] = [r for r in summary if r['method'] == method and r['percent'] == p]
    for name, subset in reports.items():
        path = args.output_root / name
        temp = path.with_suffix('.txt.tmp')
        temp.write_text(preamble + header + ''.join(map(line, subset)), encoding='utf-8')
        temp.replace(path)
    save_json(args.output_root / 'summary.json', summary)
    save_json(args.output_root / 'summary_status.json', {
        'common_scenes': len(common), 'expected_scenes': len(manifest['scenes']),
        'complete': len(common) == len(manifest['scenes']), 'scene_ids': common,
        'updated_at_kst': datetime.now(KST).isoformat(timespec='seconds'),
    })
    return len(common) == len(manifest['scenes'])


def main(argv=None):
    args = normalize(parser().parse_args(argv))
    selected = preflight(args)
    actions = commands(args)
    phases = {'all': ['cache', 'controls', 'lightgaussian'], 'cache': ['cache'],
              'evaluate': ['controls', 'lightgaussian'], 'controls': ['controls'],
              'lightgaussian': ['lightgaussian'], 'report': []}[args.stage]
    if args.check_only:
        for phase in phases:
            print(f'[ACID] {phase}: {json.dumps(actions[phase], ensure_ascii=False)}')
        print('[ACID] Check only: no inference, GPU use, output writes or package changes.')
        return
    if phases and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Select the unused GPU: GPU_ID=1 bash scripts/eval_gaussian_control_acid.sh (sbatch uses its allocated GPU)')
    from filelock import FileLock
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        with FileLock(str(args.output_root / '.summary.lock'), timeout=30):
            collect(args)
        print(f'[ACID] Summary updated: {args.output_root / "summary.txt"}')
        return
    settings = {key: str(getattr(args, key)) for key in ('checkpoint', 'dataset_root', 'index_path', 'cache_dir')}
    settings.update(percentages=args.percentages, max_scenes=args.max_scenes,
                    timing_repeats=args.timing_repeats, warmup_runs=args.warmup_runs,
                    timing_mode=args.timing_mode, selected_index=selected)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    with FileLock(str(args.output_root / '.pipeline.lock'), timeout=0):
        metadata_path = args.output_root / 'experiment.json'
        if metadata_path.exists() and read_json(metadata_path)['signature'] != signature:
            raise ValueError('ACID experiment settings changed; use a separate output root')
        save_json(metadata_path, {'signature': signature, 'training_domain': 'RE10K', 'evaluation_domain': 'ACID',
                                  'settings': settings, 'commands': actions})
        status = args.output_root / 'run_status.txt'
        phase = 'preparation'
        try:
            for phase in phases:
                status.write_text(f'running\nphase={phase}\n', encoding='utf-8')
                if phase == 'lightgaussian':
                    require_shared_baseline(args)
                run_child(phase, actions[phase], args.output_root)
                if phase != 'cache':
                    with FileLock(str(args.output_root / '.summary.lock'), timeout=30):
                        complete = collect(args)
            if args.stage in ('all', 'evaluate') and not complete:
                raise RuntimeError('Some ACID scene/condition results are missing')
        except BaseException as error:
            status.write_text(f'failed\nphase={phase}\n{type(error).__name__}: {error}\n', encoding='utf-8')
            if (args.cache_dir / 'manifest.json').exists():
                try:
                    with FileLock(str(args.output_root / '.summary.lock'), timeout=30):
                        collect(args)
                except Exception as report_error:
                    print(f'[ACID] Could not refresh combined summary: {report_error}', file=sys.stderr)
            raise
        status.write_text('complete\n' if args.stage in ('all', 'evaluate') else f'stage_complete\nphase={phase}\n', encoding='utf-8')
    print(f'[ACID] Finished {args.stage}; outputs={args.output_root}', flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
