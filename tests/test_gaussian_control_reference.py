"""Reference cameras are selected by scene/frames, never by new loader order."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.evaluation.gaussian_control_reference import GaussianControlReference


class GaussianControlReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = {'checkpoint': '/checkpoints/Final.ckpt', 'checkpoint_step': 140000,
                         'dataset': 're10k', 'dataset_roots': ['/data/re10k'],
                         'evaluation_index': '/assets/index.json', 'input_image_shape': [256, 256],
                         'camera_protocol': 'shared_final_pose', 'seed': 111123}
        (self.root / 'metadata.json').write_text(json.dumps(self.metadata))
        (self.root / 'run_status.txt').write_text('COMPLETE\n')
        (self.root / 'cameras').mkdir()
        self.rows = []
        for index in range(2):
            row = {'scene': f'scene-{index}', 'context_indices': [index, 9], 'target_indices': [3, 7],
                   'camera_file': f'cameras/{index:06d}.pt', 'overlap_group': 'medium',
                   'conditions': {'final_100': {'psnr': 20. + index, 'ssim': .8, 'lpips': .1}}}
            (self.root / row['camera_file']).touch()
            self.rows.append(row)
        self.write_rows()

    def write_rows(self):
        (self.root / 'per_scene.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def test_matches_reordered_scenes_and_requires_all_frame_indices(self):
        reference = GaussianControlReference(self.root, self.metadata)
        seen = set()
        for index in (1, 0):
            key, row = reference.lookup(f'scene-{index}', [index, 9], [3, 7])
            self.assertEqual(row['reference_final_100']['psnr'], 20. + index)
            self.assertEqual(reference.camera_path(row).name, f'{index:06d}.pt')
            seen.add(key)
        reference.require_complete_coverage(seen)
        with self.assertRaisesRegex(ValueError, 'No reference camera'):
            reference.lookup('scene-1', [1, 9], [7, 3])
        with self.assertRaisesRegex(ValueError, 'did not process'):
            reference.require_complete_coverage(set())

    def test_rejects_incomplete_or_incompatible_reference_and_missing_camera(self):
        for key, value in (('camera_protocol', 'dataset_pose'), ('checkpoint_step', 120000),
                           ('input_image_shape', [512, 512]), ('dataset_roots', ['/other/data'])):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                GaussianControlReference(self.root, {**self.metadata, key: value})
        (self.root / 'run_status.txt').write_text('RUNNING\n')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            GaussianControlReference(self.root, self.metadata)
        (self.root / 'run_status.txt').write_text('COMPLETE\n')
        (self.root / self.rows[1]['camera_file']).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'camera not found'):
            GaussianControlReference(self.root, self.metadata)

    def test_rejects_duplicate_scene_frames_and_supports_reused_baseline_records(self):
        self.rows.append(self.rows[0])
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            GaussianControlReference(self.root, self.metadata)
        self.rows.pop()
        for row in self.rows:
            row['reference_final_100'] = row['conditions'].pop('final_100')
            row['conditions'] = {'prune_005': {'psnr': 10.}}
        self.write_rows()
        reference = GaussianControlReference(self.root, self.metadata)
        self.assertEqual(reference.lookup('scene-0', [0, 9], [3, 7])[1]['reference_final_100']['psnr'], 20.)
