"""Summarize all ranks from one profile_training output prefix (stdlib only)."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def load_reports(prefix):
    prefix = Path(prefix)
    if prefix.suffix == '.json':
        prefix = prefix.with_suffix('')
    paths = sorted(prefix.parent.glob(prefix.name + '.rank*.json'))
    if not paths:
        raise ValueError(f'No completed rank reports found for {prefix}')
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    first = reports[0]
    expected = first['world_size']
    if len(reports) != expected or sorted(r['rank'] for r in reports) != list(range(expected)):
        raise ValueError(f'Need all {expected} ranks; found {[r["rank"] for r in reports]}')
    for report in reports:
        for key in ('world_size', 'head', 'context_shape_per_rank', 'moment_config',
                    'warmup_steps', 'requested_samples_per_phase', 'torch', 'gpu'):
            if report.get(key) != first.get(key):
                raise ValueError(f'Rank reports disagree on {key}; do not mix profiling runs')
        if any(report['samples_per_phase'].get(phase, 0) != first['requested_samples_per_phase']
               for phase in ('throughput', 'stages')):
            raise ValueError('A rank has incomplete profiling samples')
    return reports


def average(reports, phase, metric):
    return mean(report[phase][metric]['mean'] for report in reports)


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Reference seconds must be positive and finite')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', help='Example: outputs/profile_knn (without .rank0.json)')
    parser.add_argument('--baseline-seconds', type=positive_seconds)
    parser.add_argument('--previous-seconds', type=positive_seconds)
    args = parser.parse_args()
    try:
        reports = load_reports(args.prefix)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    current = average(reports, 'throughput', 'step_wall_s')
    print(f'Ranks: {len(reports)}; context per rank: {reports[0]["context_shape_per_rank"]}')
    print(f'Throughput step: {current:.3f} s (mean of rank means)')
    if all('throughput' in r.get('peak_allocated_gib', {}) for r in reports):
        peak = max(r['peak_allocated_gib']['throughput'] for r in reports)
        print(f'Peak torch allocated: {peak:.3f} GiB (throughput phase, max across ranks; excludes CuPy/NCCL)')
    if args.baseline_seconds is not None:
        ratio = current / args.baseline_seconds
        print(f'Historical baseline: {args.baseline_seconds:.3f} s; {ratio:.2f}x time ({(ratio - 1) * 100:+.1f}%)')
    if args.previous_seconds is not None:
        change = (current / args.previous_seconds - 1) * 100
        print(f'Previous moment: {args.previous_seconds:.3f} s; step time change {change:+.1f}%')
    print('Synchronized stages: seconds per local batch, mean across ranks')
    for key in ('knn_validate_s', 'knn_s', 'knn_prepare_s', 'knn_build_s',
                'knn_query_s', 'knn_export_s', 'knn_self_s', 'allocation_s',
                'aggregation_s', 'attributes_s', 'backward_s'):
        if all(key in r['stages'] for r in reports):
            print(f'  {key:<22} {average(reports, "stages", key):.6f}')
        else:
            print(f'  {key:<22} n/a')
    print('knn_s includes prepare/build/query/export/self; do not sum nested timings.')
    print('Use throughput step for speed comparison; stage timings add synchronization.')


if __name__ == '__main__':
    main()
