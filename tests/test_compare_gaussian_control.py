"""Read-only cohort/baseline comparison; standard library only."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/compare_gaussian_control.py'
SPEC = importlib.util.spec_from_file_location('control_compare', SOURCE)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def row(scene, method, percent, psnr):
    return {
        'scene': scene, 'method': method, 'percent': percent,
        'context_indices': [0, 4], 'target_indices': [1, 3],
        'original_gaussians': 20, 'gaussians': 20 * percent // 100,
        'psnr': psnr, 'ssim': .8, 'lpips': .2, 'postprocess_ms': 0. if percent == 100 else 4.,
    }


class CompareControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache, self.original, self.light = [self.root / name for name in ('cache', 'old', 'new')]
        self.manifest = {'signature': 'cache', 'scenes': {
            scene: {'key': scene, 'entry': {'context': [0, 4], 'target': [1, 3]}} for scene in ('a', 'b')
        }}
        save(self.cache / 'manifest.json', self.manifest)
        self.methods = ['prune', 'merge', 'redistribute']
        for directory, methods, signature in [(self.original, self.methods, 'old'), (self.light, ['lightgaussian'], 'new')]:
            save(directory / 'metadata.json', {'signature': signature, 'methods': methods, 'percentages': [100, 75],
                                             'settings': {'cache_signature': 'cache'}})
        self.old_rows = [row('a', method, p, 20. if p == 100 else 15.) for method in self.methods for p in (100, 75)]
        save(self.original / 'scenes/a.json', {'signature': 'old', 'rows': self.old_rows})
        for scene, score in [('a', 20.), ('b', 40.)]:
            rows = [row(scene, 'lightgaussian', p, score if p == 100 else score - 2) for p in (100, 75)]
            rows[0]['shared_final'] = scene == 'a'
            rows[1].update(context_camera_fit=[{'inliers': 80, 'sampled_points': 100, 'median_reprojection_px': 1.}],
                           detail={'zero_score_gaussians': 2})
            save(self.light / f'scenes/{scene}.json', {'signature': 'new', 'rows': rows})

    def tearDown(self):
        self.temporary.cleanup()

    def test_different_full_averages_match_exactly_on_the_same_cohort(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*.json')}
        result = comparison.compare(self.cache, self.original, self.light)
        self.assertEqual(result['same_scenes'], ['a'])
        table = {(r['cohort'], r['method'], r['percent']): r for r in result['tables']}
        self.assertEqual(table['all_saved', 'prune', 100]['psnr'], 20.)
        self.assertEqual(table['all_saved', 'lightgaussian', 100]['psnr'], 30.)
        self.assertEqual(table['same_scenes', 'prune', 100]['psnr'], 20.)
        self.assertEqual(table['same_scenes', 'lightgaussian', 100]['psnr'], 20.)
        self.assertEqual(result['baseline_100']['copied_nonidentical_scenes'], 0)
        self.assertEqual(result['baseline_100']['exceeds_tolerance_scenes'], 0)
        self.assertEqual(result['lightgaussian_diagnostics']['context_pnp_inlier_fraction']['mean'], .8)
        self.assertIn('SAME scenes for ALL conditions: 1', comparison.format_report(result))
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))

    def test_partial_scene_is_not_in_matched_all_condition_table(self):
        save(self.original / 'scenes/b.json', {'signature': 'old', 'rows': [row('b', 'prune', 100, 40.)]})
        result = comparison.compare(self.cache, self.original, self.light)
        self.assertEqual(result['same_scenes'], ['a'])
        self.assertEqual(result['baseline_100']['paired_scenes'], 2)
        self.assertEqual(result['baseline_100']['rerendered_scenes'], 1)

    def test_copied_baseline_mismatch_is_reported_even_below_tolerance(self):
        path = self.light / 'scenes/a.json'
        value = comparison.read_json(path)
        value['rows'][0]['psnr'] += 1e-7
        save(path, value)
        baseline = comparison.compare(self.cache, self.original, self.light)['baseline_100']
        self.assertEqual(baseline['copied_nonidentical_scenes'], 1)
        self.assertEqual(baseline['exceeds_tolerance_scenes'], 0)
        value['rows'][0]['psnr'] += .1
        save(path, value)
        baseline = comparison.compare(self.cache, self.original, self.light)['baseline_100']
        self.assertEqual(baseline['exceeds_tolerance_scenes'], 1)

    def test_cache_identity_or_frame_mismatch_fails(self):
        path = self.light / 'metadata.json'
        metadata = comparison.read_json(path)
        wrong = copy.deepcopy(metadata)
        wrong['settings']['cache_signature'] = 'other'
        save(path, wrong)
        with self.assertRaisesRegex(ValueError, 'different identities'):
            comparison.compare(self.cache, self.original, self.light)
        save(path, metadata)
        path = self.light / 'scenes/a.json'
        value = comparison.read_json(path)
        value['rows'][0]['target_indices'] = [1, 2]
        save(path, value)
        with self.assertRaisesRegex(ValueError, 'Frame indices'):
            comparison.compare(self.cache, self.original, self.light)

    def test_wrong_budget_is_detected(self):
        path = self.light / 'scenes/a.json'
        value = comparison.read_json(path)
        value['rows'][1]['gaussians'] = 20
        save(path, value)
        with self.assertRaisesRegex(ValueError, 'Incorrect Gaussian count'):
            comparison.compare(self.cache, self.original, self.light)

    def test_output_path_cannot_overlap_inputs(self):
        for destination in (self.cache, self.cache / 'analysis', self.root):
            with self.assertRaises(ValueError):
                comparison.validate_destination(destination, [self.cache, self.original, self.light])


if __name__ == '__main__':
    unittest.main()
