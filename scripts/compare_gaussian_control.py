#!/usr/bin/env python3
"""Compare saved Gaussian controls on exactly the same completed scenes.

Standard library only. Reads scene JSON and metadata, never Gaussian tensors,
CUDA, model weights or dataset chunks. Safe while the original evaluation runs.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9), 'KST')
METRICS = ('psnr', 'ssim', 'lpips')
TOLERANCES = {'psnr': 1e-4, 'ssim': 1e-6, 'lpips': 1e-5}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_rows(directory, manifest):
    metadata = read_json(directory / 'metadata.json')
    if metadata['settings']['cache_signature'] != manifest['signature']:
        raise ValueError(f'Results and cache have different identities: {directory}')
    result = {}
    for scene, item in manifest['scenes'].items():
        path = directory / 'scenes' / f"{item['key']}.json"
        if not path.is_file():
            continue
        value = read_json(path)
        if value['signature'] != metadata['signature']:
            raise ValueError(f'Scene result identity differs: {path}')
        rows = {}
        for row in value['rows']:
            if row['scene'] != scene:
                raise ValueError(f'Scene name differs: {path}')
            for key, index_key in (('context_indices', 'context'), ('target_indices', 'target')):
                if list(row[key]) != list(item['entry'][index_key]):
                    raise ValueError(f'Frame indices differ: {path}, {key}')
            key = row['method'], int(row['percent'])
            if key in rows:
                raise ValueError(f'Duplicate condition: {path}, {key}')
            if not all(math.isfinite(row[metric]) for metric in METRICS):
                raise ValueError(f'Non-finite metric: {path}, {key}')
            total = row.get('original_gaussians')
            if total is not None and row['gaussians'] != max(1, math.floor(total * key[1] / 100)):
                raise ValueError(f'Incorrect Gaussian count: {path}, {key}')
            rows[key] = row
        result[scene] = rows
    return metadata, result


def mean_row(rows, method, percent, cohort):
    return {'cohort': cohort, 'method': method, 'percent': percent, 'scenes': len(rows),
            **{metric: statistics.fmean(r[metric] for r in rows) for metric in
               (*METRICS, 'postprocess_ms', 'gaussians')}}


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'count': len(values), 'min': ordered[0], 'median': statistics.median(ordered),
            'p95': ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)],
            'max': ordered[-1], 'mean': statistics.fmean(ordered)}


def compare(cache, original_dir, light_dir):
    manifest = read_json(cache / 'manifest.json')
    original_meta, original = load_rows(original_dir, manifest)
    light_meta, light = load_rows(light_dir, manifest)
    methods = list(original_meta['methods'])
    if light_meta['methods'] != ['lightgaussian']:
        raise ValueError('Expected a LightGaussian result directory')
    if set(original_meta['percentages']) != set(light_meta['percentages']):
        raise ValueError('Experiments request different percentages; compare matching configurations')
    percentages = sorted(map(int, original_meta['percentages']), reverse=True)
    sources = {method: original for method in methods}
    sources['lightgaussian'] = light
    all_methods = methods + ['lightgaussian']
    conditions = [(m, p) for p in percentages for m in all_methods]
    common = sorted(scene for scene in manifest['scenes'] if all(
        (method, percent) in sources[method].get(scene, {}) for method, percent in conditions))
    tables = []
    for method, percent in conditions:
        saved = [rows[(method, percent)] for rows in sources[method].values() if (method, percent) in rows]
        if saved:
            tables.append(mean_row(saved, method, percent, 'all_saved'))
        paired = [sources[method][scene][method, percent] for scene in common]
        if paired:
            tables.append(mean_row(paired, method, percent, 'same_scenes'))

    baseline_pairs, baseline_details, copied = [], [], []
    for scene in sorted(set(original) & set(light)):
        candidates = [original[scene][m, 100] for m in methods if (m, 100) in original[scene]]
        new = light[scene].get(('lightgaussian', 100))
        if not candidates or new is None:
            continue
        old = candidates[0]
        for alternative in candidates[1:]:
            if any(alternative[key] != old[key] for key in METRICS):
                raise ValueError(f'Original methods have different 100% identity metrics: {scene}')
        delta = {key: new[key] - old[key] for key in METRICS}
        detail = {'scene': scene, 'shared_final': bool(new.get('shared_final')),
                  'original': {key: old[key] for key in METRICS},
                  'lightgaussian': {key: new[key] for key in METRICS}, 'delta': delta}
        baseline_pairs.append(detail)
        if detail['shared_final']:
            copied.append(detail)
        if any(abs(delta[key]) > TOLERANCES[key] for key in METRICS):
            baseline_details.append(detail)

    baseline = {
        'paired_scenes': len(baseline_pairs), 'copied_scenes': len(copied),
        'rerendered_scenes': len(baseline_pairs) - len(copied),
        'copied_nonidentical_scenes': sum(any(r['delta'][key] != 0 for key in METRICS) for r in copied),
        'exceeds_tolerance_scenes': len(baseline_details), 'tolerances': TOLERANCES,
        'absolute_differences': {key: distribution([abs(r['delta'][key]) for r in baseline_pairs]) for key in METRICS},
        'largest_differences': sorted(baseline_details, key=lambda r: abs(r['delta']['psnr']), reverse=True)[:20],
    }

    zero_fraction, inlier_fraction, reprojection = [], [], []
    identical_5_10 = 0
    pairs_5_10 = 0
    for rows in light.values():
        row = next((row for (m, p), row in rows.items() if p != 100), None)
        if row is not None:
            zero = row.get('detail', {}).get('zero_score_gaussians')
            total = row.get('original_gaussians')
            if zero is not None and total:
                zero_fraction.append(zero / total)
            for camera in row.get('context_camera_fit', []):
                inlier_fraction.append(camera['inliers'] / camera['sampled_points'])
                reprojection.append(camera['median_reprojection_px'])
        low, high = rows.get(('lightgaussian', 5)), rows.get(('lightgaussian', 10))
        if low and high:
            pairs_5_10 += 1
            identical_5_10 += all(low[key] == high[key] for key in METRICS)
    return {
        'read_at_kst': datetime.now(KST).isoformat(timespec='seconds'),
        'cache_signature': manifest['signature'], 'expected_scenes': len(manifest['scenes']),
        'original_dir': str(original_dir), 'lightgaussian_dir': str(light_dir),
        'same_scenes': common, 'same_scene_count': len(common), 'tables': tables,
        'baseline_100': baseline,
        'lightgaussian_diagnostics': {
            'zero_score_fraction': distribution(zero_fraction),
            'context_pnp_inlier_fraction': distribution(inlier_fraction),
            'context_pnp_inlier_median_reprojection_px': distribution(reprojection),
            'scenes_with_both_5_and_10': pairs_5_10,
            'scenes_with_exactly_identical_5_and_10_metrics': identical_5_10,
        },
    }


def format_report(result):
    lines = [
        f"Snapshot (KST): {result['read_at_kst']}",
        f"Expected scenes: {result['expected_scenes']}; SAME scenes for ALL conditions: {result['same_scene_count']}",
        'all_saved cohorts have different scene counts and are NOT a fair method comparison.',
        'same_scenes is a temporary matched subset, not a replacement for the complete benchmark.',
        'Timings retain their original run/concurrency conditions.',
        '',
    ]
    for cohort in ('all_saved', 'same_scenes'):
        lines += [cohort, 'method percent scenes gaussians PSNR SSIM LPIPS post_ms']
        for row in result['tables']:
            if row['cohort'] == cohort:
                lines.append(f"{row['method']:14s} {row['percent']:3d} {row['scenes']:5d} {row['gaussians']:9.1f} "
                             f"{row['psnr']:.6f} {row['ssim']:.6f} {row['lpips']:.6f} {row['postprocess_ms']:.4f}")
        lines.append('')
    baseline = result['baseline_100']
    lines += [
        f"100% paired scenes: {baseline['paired_scenes']}; copied: {baseline['copied_scenes']}; "
        f"rerendered: {baseline['rerendered_scenes']}",
        f"Copied 100% rows with ANY metric difference (must be 0): {baseline['copied_nonidentical_scenes']}",
        f"100% rows exceeding tolerance: {baseline['exceeds_tolerance_scenes']}",
        'Tolerance is a diagnostic threshold, not a guarantee of equivalence.',
        json.dumps(baseline['absolute_differences'], indent=2),
        '', 'LightGaussian saved diagnostics (no new scoring/rendering):',
        json.dumps(result['lightgaussian_diagnostics'], indent=2),
    ]
    return '\n'.join(lines) + '\n'


def validate_destination(output, inputs):
    for directory in inputs:
        if output == directory or directory in output.parents or output in directory.parents:
            raise ValueError('Comparison output must be separate from source cache/results')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    base = ROOT / 'outputs/gaussian_control'
    cli.add_argument('--cache-dir', type=Path, default=base / 'final_plus_cache')
    cli.add_argument('--original-dir', type=Path, default=base / 'final_plus_results')
    cli.add_argument('--lightgaussian-dir', type=Path, default=base / 'final_plus_lightgaussian_results')
    cli.add_argument('--output-dir', type=Path, help='Default: a new timestamped comparison directory')
    args = cli.parse_args()
    cache, original, light = [path.resolve() for path in (args.cache_dir, args.original_dir, args.lightgaussian_dir)]
    output = (args.output_dir or base / ('comparison_' + datetime.now(KST).strftime('%Y-%m-%d_%H-%M-%S_%f'))).resolve()
    validate_destination(output, [cache, original, light])
    result = compare(cache, original, light)
    report = format_report(result)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in [('comparison.txt', report), ('comparison.json', json.dumps(result, indent=2, allow_nan=False))]:
        temporary = output / (name + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(output / name)
    print(report, end='')
    print(f'Comparison saved to {output}')


if __name__ == '__main__':
    main()
