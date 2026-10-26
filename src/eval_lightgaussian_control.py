"""Evaluate LightGaussian pruning only from a read-only Final++ scene cache.

This entry point intentionally does not modify/import the encoder or rebuild
the cache. Its extension, camera sidecars, locks and reports have their own paths.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np
import torch
from filelock import FileLock

from .eval_gaussian_control import (
    digest, file_digest, freeze, read_json, save_json, save_torch, summarize, validate_scene,
)
from .gaussian_control.common import gaussian_dict, synchronized_time, target_count, to_device
from .gaussian_control.lightgaussian import RASTERIZER_ROOT, load_count_extension, prune
from .gaussian_control.lightgaussian_cameras import ContextCalibration, fit_context_cameras
from .model.types import Gaussians


METHOD = 'lightgaussian'
DEFAULT_PERCENTAGES = (100, 75, 50, 25, 10, 5, 1)
SOURCE_ROOT = Path(__file__).resolve().parent


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cache-dir', type=Path, default=Path('outputs/gaussian_control/final_plus_cache'))
    cli.add_argument('--output-dir', type=Path, default=Path('outputs/gaussian_control/final_plus_lightgaussian_results'))
    cli.add_argument('--dataset-root', type=Path, action='append', help='Override RE10K root (repeat for multiple roots)')
    cli.add_argument('--baseline-results', type=Path, help='Read shared 100%% rows here; defaults to cache sibling final_plus_results')
    cli.add_argument('--percentages', type=int, nargs='+', default=list(DEFAULT_PERCENTAGES))
    cli.add_argument('--max-scenes', type=int)
    cli.add_argument('--threads', type=int, default=1)
    cli.add_argument('--timing-mode', choices=['concurrent', 'isolated'], default='concurrent')
    cli.add_argument('--timing-repeats', type=int, default=1)
    cli.add_argument('--warmup-runs', type=int, default=1)
    cli.add_argument('--report-every', type=int, default=25)
    cli.add_argument('--check-only', action='store_true', help='Check inputs, CUDA and build dependencies; do not evaluate')
    return cli


def validate_paths(cache, output):
    # Reject ancestor/descendant overlap as well as the currently running result
    # directory. All checks precede mkdir, log creation, status and lock writes.
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError('LightGaussian output and source cache must be separate directories')
    if output.name == 'final_plus_results':
        raise ValueError('Use a separate LightGaussian output, not final_plus_results')
    metadata = output / 'metadata.json'
    if metadata.exists() and read_json(metadata).get('methods') != [METHOD]:
        raise ValueError('Output contains another experiment; choose a separate LightGaussian output')


def load_cpu(path):
    # Scenes contain ~210 MiB of features; mmap avoids reading/copying these
    # unused tensors. Only Gaussian attributes, points and target tensors are used.
    try:
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error).lower():
            raise
        return torch.load(path, map_location='cpu', weights_only=True)


def available_memory():
    info = {}
    path = Path('/proc/meminfo')
    if path.exists():
        info = {key: int(value.split()[0]) * 1024 for key, value in
                (line.split(':', 1) for line in path.read_text().splitlines())}
    return info.get('MemAvailable')


def preflight(args, manifest):
    from torch.utils.cpp_extension import CUDA_HOME, is_ninja_available
    if not manifest.get('complete'):
        raise RuntimeError('Source cache is incomplete; finish the existing cache stage first')
    missing = [scene for scene, item in manifest['scenes'].items()
               if not (args.cache_dir / 'scenes' / f"{item['key']}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} cached scenes missing, e.g. {missing[:3]}')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; run in the original conda environment with GPU_ID=1')
    torch.cuda.set_device(0)
    if CUDA_HOME is None or not (Path(CUDA_HOME) / 'bin/nvcc').exists():
        raise RuntimeError('nvcc/CUDA_HOME is required to build the separate LightGaussian extension')
    if not is_ninja_available():
        raise RuntimeError('ninja is required for the JIT extension (install ninja in this environment)')
    import cv2  # existing model environment dependency; no images are decoded
    del cv2
    # Check the *existing* renderer/metrics without replacing their installation.
    from .evaluation.metrics import compute_lpips
    from .model.decoder import get_decoder
    del compute_lpips, get_decoder
    ContextCalibration(manifest, args.dataset_root)
    memory = available_memory()
    free_gpu, total_gpu = torch.cuda.mem_get_info(0)
    if memory is not None and memory < 4 * 1024**3:
        raise RuntimeError('Less than 4 GiB available host RAM; defer the concurrent run')
    if free_gpu < 4 * 1024**3:
        raise RuntimeError('Selected GPU has less than 4 GiB free; select the unused GPU')
    destination = args.output_dir
    while not destination.exists():
        destination = destination.parent
    if shutil.disk_usage(destination).free < 1024**3:
        raise RuntimeError('Less than 1 GiB output space available')
    resources = {
        'gpu': torch.cuda.get_device_name(0), 'visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'gpu_free_bytes': free_gpu, 'gpu_total_bytes': total_gpu,
        'host_available_bytes': memory, 'cpu_threads': args.threads,
        'max_jobs': os.getenv('MAX_JOBS', '1'), 'cuda_home': str(CUDA_HOME),
        'disk_free_bytes': shutil.disk_usage(destination).free,
        'pid': os.getpid(),
        'nice': os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, 'getpriority') else None,
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
    }
    print('[LightGaussian] Preflight OK: ' + json.dumps(resources), flush=True)
    return resources


def settings_for(args, manifest):
    sources = [Path(__file__), SOURCE_ROOT / 'gaussian_control/lightgaussian.py',
               SOURCE_ROOT / 'gaussian_control/lightgaussian_cameras.py',
               SOURCE_ROOT / 'gaussian_control/common.py', SOURCE_ROOT / 'eval_gaussian_control.py']
    sources += sorted(path for path in RASTERIZER_ROOT.rglob('*') if path.is_file())
    return {
        'format_version': 1, 'cache_signature': manifest['signature'],
        'method': 'LightGaussian pruning only', 'percentages': args.percentages,
        'max_scenes': args.max_scenes, 'timing_mode': args.timing_mode,
        'dataset_roots_override': list(map(str, args.dataset_root)) if args.dataset_root else None,
        'baseline_results': str(args.baseline_results),
        'timing_repeats': args.timing_repeats, 'warmup_runs': args.warmup_runs,
        'cpu_threads': args.threads, 'selection_views': 'two context views, no target inputs',
        'context_cameras': 'PnP from cached support points and dataset context intrinsics',
        'camera_seed': int(manifest['identity']['seed']), 'camera_max_points': 8192,
        'v_pow': .1, 'volume_reference': 'descending_volume[int(0.9*N)]', 'volume_clamp': False,
        'budget': 'exact stable top-k; original attributes unchanged',
        'recovery': False, 'sh_distillation': False, 'quantization': False,
        'sources': {path.relative_to(SOURCE_ROOT).as_posix(): file_digest(path) for path in sources},
    }


def shared_final_row(baseline, manifest, item):
    """Read an already completed original 100% result; never wait/write there."""
    metadata_path = baseline / 'metadata.json'
    scene_path = baseline / 'scenes' / f"{item['key']}.json"
    if not metadata_path.exists() or not scene_path.exists():
        return None
    metadata, scene = read_json(metadata_path), read_json(scene_path)
    if (metadata['settings']['cache_signature'] != manifest['signature']
            or scene['signature'] != metadata['signature']):
        raise RuntimeError('Shared 100% results do not match the source cache')
    for original in scene['rows']:
        if original['percent'] == 100:
            if (original['context_indices'] != list(item['entry']['context'])
                    or original['target_indices'] != list(item['entry']['target'])):
                raise RuntimeError('Shared 100% frame indices do not match')
            row = copy.deepcopy(original)
            row.update(method=METHOD, postprocess_ms=0., postprocess_ms_samples=[0.],
                       detail={'shared_final_source': str(scene_path), 'source_signature': metadata['signature']})
            row['shared_final'] = True
            # Its render timing retains its own original run provenance.
            row['timing_mode'] = 'shared original baseline; concurrency unknown'
            return row
    return None


def write_reports(output, rows, percentages, expected, timing_mode):
    summary = summarize(rows, [METHOD], percentages)
    save_json(output / 'summary.json', summary)
    # All report replacements are atomic; readers see complete snapshots.
    per_scene = output / 'per_scene.jsonl'
    temporary = per_scene.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    temporary.replace(per_scene)
    preamble = (
        f'LightGaussian pruning only; expected scenes per condition: {expected}\n'
        f'Timing mode: {timing_mode}. Concurrent CPU/disk contention can affect BOTH running experiments.\n'
        'Scores: context-view opacity * transmittance sums, volume weight v_pow=0.1 (released code).\n'
        'Postprocess includes scoring renders, volume weighting, ranking and selection for EACH ratio.\n'
        'Excludes cache/dataset I/O, camera preparation, NVS rendering and metrics; no encoder inference.\n'
        '100% is the unchanged cached Gaussian set; postprocess_ms=0.\n'
        'Existing 100% metrics are reused when available; their render timing comes from the original run.\n'
        'Metrics average target views, then scenes; cached target cameras stay fixed.\n'
    )
    header = 'method percent overlap scenes Gaussians actual_percent PSNR SSIM LPIPS post_ms post_p50_ms post_p95_ms render_ms_per_view\n'
    def line(row):
        return (f"{METHOD} {row['percent']:3d} {row['overlap']:7s} {row['scenes']:5d} "
                f"{row['gaussians']:.1f} {row['actual_percent']:.4f} "
                f"{row['psnr']:.6f} {row['ssim']:.6f} {row['lpips']:.6f} "
                f"{row['postprocess_ms']:.4f} {row['postprocess_ms_p50']:.4f} "
                f"{row['postprocess_ms_p95']:.4f} {row['render_ms_per_view']:.4f}\n")
    reports = {'summary.txt': summary}
    reports.update({f'{METHOD}_{p:03d}.txt': [r for r in summary if r['percent'] == p] for p in percentages})
    for name, subset in reports.items():
        path = output / name
        temp = path.with_suffix('.txt.tmp')
        temp.write_text(preamble + header + ''.join(map(line, subset)), encoding='utf-8')
        temp.replace(path)


def context_cameras(args, manifest, payload, calibration):
    path = args.output_dir / 'context_cameras' / f"{payload['key']}.pt"
    if path.exists():
        value = load_cpu(path)
        if value['cache_signature'] != manifest['signature'] or value['key'] != payload['key']:
            raise RuntimeError('Stale LightGaussian context-camera sidecar')
        return value['cameras'], value['reports'], value['prepare_ms']
    start = perf_counter()
    intrinsic, near, far = calibration.intrinsics_and_bounds(payload)
    cameras, reports = fit_context_cameras(payload, intrinsic, near, far, manifest['identity']['seed'])
    prepare_ms = 1000 * (perf_counter() - start)
    save_torch(path, {'cache_signature': manifest['signature'], 'key': payload['key'],
                      'cameras': cameras, 'reports': reports, 'prepare_ms': prepare_ms})
    return cameras, reports, prepare_ms


@torch.no_grad()
def evaluate(args, manifest, bundle, signature, device, extension, renderer, metrics, calibration):
    from .misc.utils import get_overlap_tag
    items = list(manifest['scenes'].items())[:args.max_scenes]
    requested = args.percentages
    all_rows = []
    warmed = set()
    completed = 0
    active = {}
    try:
        for scene, item in items:
            result_path = args.output_dir / 'scenes' / f"{item['key']}.json"
            previous = read_json(result_path) if result_path.exists() else {'signature': signature, 'rows': []}
            if previous['signature'] != signature:
                raise RuntimeError(f'Stale LightGaussian scene results: {scene}')
            active = {r['percent']: r for r in previous['rows']}
            if 100 in requested and 100 not in active:
                shared = shared_final_row(args.baseline_results, manifest, item)
                if shared is not None:
                    active[100] = shared
                    save_json(result_path, {'signature': signature, 'rows': list(active.values())})
            if not all(p in active for p in requested):
                payload = load_cpu(args.cache_dir / 'scenes' / f"{item['key']}.pt")
                validate_scene(payload, manifest, scene)
                # The support feature matrix and redistribution graph never go to GPU.
                original = Gaussians(**to_device(payload['gaussians'], device))
                target = to_device(payload['target'], device)
                total = original.means.shape[1]
                cameras, camera_report, prepare_ms = ({}, [], 0.)
                if any(p != 100 and p not in active for p in requested):
                    cameras, camera_report, prepare_ms = context_cameras(args, manifest, payload, calibration)
                    cameras = to_device(cameras, device)
                def render(gaussians):
                    images = []
                    for i in range(target['image'].shape[1]):
                        images.append(renderer(
                            gaussians, target['extrinsics'][:, i:i+1], target['intrinsics'][:, i:i+1],
                            target['near'][:, i:i+1], target['far'][:, i:i+1], target['image'].shape[-2:],
                        ).color)
                    return torch.cat(images, 1)[0]
                for percent in requested:
                    if percent in active:
                        continue
                    count = target_count(total, percent)
                    samples, detail = [0.], {}
                    if percent == 100:
                        variant = original
                    else:
                        def operation():
                            return prune(extension, original, cameras, payload['support']['image_shape'][1:],
                                         count, bundle['render_cfg']['make_scale_invariant'])
                        if percent not in warmed:
                            for _ in range(args.warmup_runs):
                                warm = operation()
                                del warm
                            if device.type == 'cuda':
                                torch.cuda.synchronize(device)
                            warmed.add(percent)
                        samples = []
                        for repeat in range(args.timing_repeats):
                            if repeat:
                                del variant
                            (variant, detail), seconds = synchronized_time(operation, device)
                            samples.append(seconds * 1000)
                    if variant.means.shape[1] != count:
                        raise AssertionError('LightGaussian returned an incorrect budget')
                    if not all(bool(torch.isfinite(x).all()) for x in gaussian_dict(variant).values()):
                        raise FloatingPointError('Non-finite LightGaussian output')
                    render(variant)  # renderer warm-up is outside the measured postprocess
                    prediction, render_seconds = synchronized_time(lambda: render(variant), device)
                    gt = target['image'][0]
                    values = {name: fn(gt, prediction) for name, fn in metrics.items()}
                    row = {
                        'scene': scene, 'context_indices': payload['context_indices'],
                        'target_indices': payload['target_indices'], 'overlap': payload['overlap'],
                        'overlap_tag': get_overlap_tag(payload['overlap']), 'method': METHOD,
                        'percent': percent, 'gaussians': count, 'original_gaussians': total,
                        'actual_percent': 100 * count / total, 'target_views': len(gt),
                        'postprocess_ms': float(np.mean(samples)), 'postprocess_ms_samples': samples,
                        'render_ms_per_view': 1000 * render_seconds / len(gt),
                        'timing_mode': args.timing_mode, 'camera_prepare_ms': prepare_ms,
                        'context_camera_fit': camera_report, 'detail': detail,
                        **{name: float(value.mean()) for name, value in values.items()},
                        'per_view': {name: value.tolist() for name, value in values.items()},
                    }
                    active[percent] = row
                    save_json(result_path, {'signature': signature, 'rows': list(active.values())})
                    print(f'[LightGaussian] {completed+1}/{len(items)} {scene} {percent}% N={count} '
                          f"PSNR={row['psnr']:.3f} post={row['postprocess_ms']:.3f}ms", flush=True)
                    del variant, prediction
                del payload, original, target, cameras
            all_rows.extend(active[p] for p in requested)
            active = {}
            completed += 1
            if completed == 1 or completed % args.report_every == 0:
                write_reports(args.output_dir, all_rows, requested, len(items), args.timing_mode)
                print(f'[LightGaussian] {completed}/{len(items)} scenes complete; summary updated', flush=True)
    finally:
        # Include condition rows saved before an interruption in the current scene.
        write_reports(args.output_dir, all_rows + list(active.values()), requested, len(items), args.timing_mode)
    if completed != len(items):
        raise RuntimeError('Incomplete LightGaussian evaluation')


def main(argv=None):
    args = parser().parse_args(argv)
    args.cache_dir, args.output_dir = args.cache_dir.resolve(), args.output_dir.resolve()
    args.baseline_results = (args.baseline_results or args.cache_dir.parent / 'final_plus_results').resolve()
    if args.output_dir == args.baseline_results or args.baseline_results in args.output_dir.parents:
        raise ValueError('LightGaussian output must be separate from the shared baseline results')
    validate_paths(args.cache_dir, args.output_dir)
    if (not args.percentages or len(set(args.percentages)) != len(args.percentages)
            or any(p < 1 or p > 100 for p in args.percentages)):
        raise ValueError('Use distinct integer percentages in [1,100]')
    if (min(args.threads, args.timing_repeats, args.report_every) < 1 or args.warmup_runs < 0
            or (args.max_scenes is not None and args.max_scenes < 1)):
        raise ValueError('Invalid scene/thread/timing settings')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = read_json(args.cache_dir / 'manifest.json')
    bundle = load_cpu(args.cache_dir / 'decoder.pt')
    if bundle['signature'] != manifest['signature']:
        raise RuntimeError('Cached renderer and scene cache signatures differ')
    resources = preflight(args, manifest)
    if args.check_only:
        print('[LightGaussian] Check only: no cache, output or model changes. CUDA compilation occurs on first run.', flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Another process cannot overwrite this runner's progress, even on resume.
    with FileLock(str(args.output_dir / '.lightgaussian.lock'), timeout=0):
        settings = settings_for(args, manifest)
        signature = digest(settings)
        metadata_path = args.output_dir / 'metadata.json'
        if metadata_path.exists() and read_json(metadata_path)['signature'] != signature:
            raise RuntimeError('LightGaussian settings/source changed; choose a new output directory')
        save_json(metadata_path, {
            'signature': signature, 'settings': settings, 'methods': [METHOD],
            'percentages': args.percentages,
            'cache_dir': str(args.cache_dir), 'resources': resources,
            'torch': str(torch.__version__), 'python': platform.python_version(),
            'cuda': torch.version.cuda, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        })
        status = args.output_dir / 'run_status.txt'
        status.write_text('running\n', encoding='utf-8')
        try:
            from .evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
            from .model.decoder import get_decoder
            from .model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
            device = torch.device('cuda', 0)
            torch.manual_seed(manifest['identity']['seed'])
            np.random.seed(manifest['identity']['seed'])
            extension = load_count_extension(str(args.output_dir / 'cuda_build'))
            renderer = freeze(get_decoder(DecoderSplattingCUDACfg(**bundle['render_cfg'])).to(device))
            evaluate(args, manifest, bundle, signature, device, extension, renderer,
                     {'psnr': compute_psnr, 'ssim': compute_ssim, 'lpips': compute_lpips},
                     ContextCalibration(manifest, args.dataset_root))
        except BaseException as error:
            status.write_text(f'failed\n{type(error).__name__}: {error}\n', encoding='utf-8')
            raise
        status.write_text('complete\n', encoding='utf-8')


if __name__ == '__main__':
    main()
