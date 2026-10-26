#!/usr/bin/env python3
"""Read-only Gaussian-control progress/ETA; Python stdlib only.

Uses the exact scene/condition identities in the saved manifests. ETA uses
recent completed-scene file timestamps, including rendering, metrics and I/O.
No torch imports, large tensor reads, or changes to the evaluation are needed.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
import time


KST = timezone(timedelta(hours=9), 'KST')
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def duration(seconds):
    if seconds is None:
        return 'unavailable (need at least 3 completed scenes)'
    minutes = round(seconds / 60)
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    return f'{days}d {hours}h {minutes}m'


def snapshot(cache, output, window, now=None):
    now = time.time() if now is None else now
    manifest = read_json(cache / 'manifest.json')
    metadata_path = output / 'metadata.json'
    metadata = read_json(metadata_path) if metadata_path.is_file() else None
    if metadata and metadata['settings']['cache_signature'] != manifest['signature']:
        raise ValueError('Cache and results belong to different runs')
    requested = {(m, int(p)) for m in metadata['methods'] for p in metadata['percentages']} if metadata else set()
    cache_times, completed_times, merge_times = [], [], []
    partial, invalid = 0, 0
    for item in manifest['scenes'].values():
        cache_path = cache / 'scenes' / f"{item['key']}.pt"
        if cache_path.is_file():
            cache_times.append(cache_path.stat().st_mtime)
        result_path = output / 'scenes' / f"{item['key']}.json"
        if not metadata or not result_path.is_file():
            continue
        try:
            result = read_json(result_path)
            if result['signature'] != metadata['signature']:
                raise ValueError('Different scene-result signature')
            rows = {(r['method'], int(r['percent'])): r for r in result['rows']}
            if requested.issubset(rows):
                completed_times.append(result_path.stat().st_mtime)
                merging = [rows[key]['postprocess_ms'] for key in requested if key[0] == 'merge' and key[1] < 100]
                if merging:
                    merge_times.append(sum(merging) / 1000)
            elif requested.intersection(rows):
                partial += 1
        except (OSError, ValueError, KeyError, TypeError):
            invalid += 1
    total = len(manifest['scenes'])
    cache_complete = bool(manifest.get('complete')) and len(cache_times) == total
    phase = 'evaluate' if cache_complete else 'cache'
    times = sorted(completed_times if cache_complete else cache_times)
    recent = times[-window:]
    mean_seconds = (recent[-1] - recent[0]) / (len(recent) - 1) if len(recent) >= 3 else None
    if mean_seconds is not None and mean_seconds <= 0:
        mean_seconds = None
    remaining = total - len(times)
    eta = remaining * mean_seconds if mean_seconds is not None else None
    if remaining == 0:
        eta = 0.
    status_path = output / 'run_status.txt'
    status_lines = status_path.read_text(encoding='utf-8').splitlines() if status_path.is_file() else []
    status = status_lines[0] if status_lines else 'unknown'
    return {
        'recorded_status': status, 'phase': phase, 'total_scenes': total,
        'cached_scenes': len(cache_times), 'evaluated_scenes': len(completed_times),
        'partial_scenes': partial, 'invalid_scene_files': invalid,
        'remaining_scenes_in_phase': remaining,
        'progress_percent': 100 * len(times) / total if total else 0.,
        'recent_completion_count': len(recent), 'recent_seconds_per_scene': mean_seconds,
        'seconds_since_last_completion': max(0., now - times[-1]) if times else None,
        'remaining_seconds_estimate': eta,
        'projected_finish_kst': datetime.fromtimestamp(now + eta, KST).isoformat(timespec='seconds') if eta is not None else None,
        'mean_merging_seconds_per_scene': statistics.fmean(merge_times) if merge_times else None,
        'remaining_merging_seconds_estimate': statistics.fmean(merge_times) * (total - len(completed_times)) if merge_times else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'outputs/gaussian_control/final_plus_cache')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/gaussian_control/final_plus_results')
    parser.add_argument('--window', type=int, default=50, help='Most recent completed scenes used for ETA (default 50)')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON')
    args = parser.parse_args()
    if args.window < 3:
        parser.error('--window must be at least 3')
    try:
        data = snapshot(args.cache_dir, args.output_dir, args.window)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Cannot read control progress: {error}\n')
    if args.json:
        print(json.dumps(data, indent=2, allow_nan=False))
        return
    print(f"Recorded status: {data['recorded_status']} (not a process-liveness check)")
    print(f"Cache: {data['cached_scenes']}/{data['total_scenes']} scenes")
    print(f"Evaluation: {data['evaluated_scenes']}/{data['total_scenes']} complete, "
          f"{data['partial_scenes']} partial, {data['invalid_scene_files']} unreadable/mismatched")
    print(f"Current phase: {data['phase']}; progress {data['progress_percent']:.2f}%")
    rate = data['recent_seconds_per_scene']
    if rate is not None:
        print(f"Recent wall time: {rate:.2f} s/scene ({data['recent_completion_count']} completion timestamps)")
    age = data['seconds_since_last_completion']
    if age is not None:
        print(f'Last scene completed {age:.0f} seconds ago')
    print(f"Remaining in this phase: {duration(data['remaining_seconds_estimate'])}")
    if data['projected_finish_kst']:
        print(f"Projected phase completion (Asia/Seoul): {data['projected_finish_kst']}")
    if data['mean_merging_seconds_per_scene'] is not None:
        print(f"Merging alone: {data['mean_merging_seconds_per_scene']:.2f} s/scene on completed scenes; "
              f"remaining estimate {duration(data['remaining_merging_seconds_estimate'])}")
    if data['phase'] == 'cache':
        print('Cache ETA excludes the evaluation phase that follows.')
    print('ETA assumes this process continues at the recent rate; restarts, pauses, '
          'parallel writers and storage stalls can distort timestamp-based estimates.')


if __name__ == '__main__':
    main()
