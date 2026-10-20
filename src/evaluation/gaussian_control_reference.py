"""Read completed evaluation metrics and locate cameras by scene/frame identity."""

import json
from pathlib import Path


def scene_key(scene, context_indices, target_indices):
    return str(scene), tuple(context_indices), tuple(target_indices)


class GaussianControlReference:
    def __init__(self, directory: Path, current_metadata: dict):
        self.directory = Path(directory).expanduser().resolve()
        if (self.directory / 'run_status.txt').read_text(encoding='utf-8').strip() != 'COMPLETE':
            raise ValueError(f'Reference evaluation is incomplete: {self.directory}')
        self.metadata = json.loads((self.directory / 'metadata.json').read_text(encoding='utf-8'))
        for key in ('checkpoint', 'checkpoint_step', 'dataset', 'dataset_roots',
                    'evaluation_index', 'input_image_shape', 'camera_protocol', 'seed'):
            if self.metadata.get(key) != current_metadata.get(key):
                raise ValueError(f'Reference evaluation mismatch for {key}: '
                                 f'{self.metadata.get(key)!r} != {current_metadata.get(key)!r}')
        self.entries = {}
        with (self.directory / 'per_scene.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                entry = json.loads(line)
                key = scene_key(entry['scene'], entry['context_indices'], entry['target_indices'])
                if key in self.entries:
                    raise ValueError(f'Duplicate reference scene/frame identity: {key}')
                baseline = entry['conditions'].get('final_100') or entry.get('reference_final_100')
                if baseline is None or any(metric not in baseline for metric in ('psnr', 'ssim', 'lpips')):
                    raise ValueError(f'Missing per-scene Final metrics: {key}')
                entry['reference_final_100'] = baseline
                self.camera_path(entry)  # Fail before testing if any camera is missing.
                self.entries[key] = entry
        if not self.entries:
            raise ValueError('Reference evaluation has no scenes')

    def camera_path(self, entry: dict) -> Path:
        path = (self.directory / entry['camera_file']).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Reference camera must be inside its evaluation directory')
        if not path.is_file():
            raise FileNotFoundError(f'Reference camera not found: {path}')
        return path

    def lookup(self, scene, context_indices, target_indices):
        key = scene_key(scene, context_indices, target_indices)
        if key not in self.entries:
            raise ValueError(f'No reference camera/Final metrics for scene and frames: {key}')
        return key, self.entries[key]

    def require_complete_coverage(self, seen):
        missing = set(self.entries) - seen
        if missing:
            raise ValueError(f'New evaluation did not process {len(missing)} reference scenes; '
                             'check dataset/index or use max_scenes for an intentional limited run')
