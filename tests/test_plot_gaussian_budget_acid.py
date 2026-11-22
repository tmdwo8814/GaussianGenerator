"""Plotting from a saved combined ACID summary; no checkpoints or GPU needed."""

import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('acid_budget_plot', ROOT / 'scripts/plot_gaussian_budget_acid.py')
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def fixture():
    # Synthetic values exercise signs/selection only; never exported as real data.
    rows = [dict(method='final', percent=100, overlap='overall', scenes=7,
                 gaussians=131072., psnr=26., ssim=.8, lpips=.15, postprocess_ms=0.)]
    for index, percent in enumerate((10, 5, 1)):
        for method in plot.METHODS:
            if method == 'lightgaussian':
                psnr, ssim, lpips, ms = 7.-index, .13-.02*index, .7+.01*index, 4.
            elif method == 'merge':
                psnr, ssim, lpips, ms = 20.-index, .5-.1*index, .4+.05*index, 5000.+index*1000
            else:
                psnr, ssim, lpips, ms = 20.-index+(-1.38, -.01, .08)[index], .5-.1*index+(-.001, .08, .148)[index], .4+.05*index+(.074, .059, .036)[index], 100.
            rows.append(dict(method=method, percent=percent, overlap='overall', scenes=7,
                             gaussians=float(131072 * percent // 100), psnr=psnr, ssim=ssim,
                             lpips=lpips, postprocess_ms=ms))
    # Ensure subgroup means are not mixed into the requested overall rows.
    rows += [{**r, 'overlap': 'small', 'psnr': 99.} for r in rows]
    return rows


class AcidBudgetPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'summary.json'
        self.rows = fixture()
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.source.write_text(json.dumps(self.rows), encoding='utf-8')

    def test_default_reads_repository_summary_not_downloads_or_pdf(self):
        self.assertEqual(plot.DEFAULT_INPUT, ROOT / 'outputs/gaussian_control/re10k_to_acid/summary.json')
        _, rows, baseline, status = plot.load_results(self.root, [10, 5, 1])
        self.assertEqual(baseline['method'], 'final')
        self.assertEqual(rows['merge', 10]['psnr'], 20.)
        self.assertIsNone(status)

    def test_improvement_colors_include_lpips_and_small_nonzero_values(self):
        for metric in ('psnr', 'ssim'):
            self.assertEqual(plot.offset_color(metric, Decimal('.00000001')), plot.GREEN)
            self.assertEqual(plot.offset_color(metric, Decimal('-.00000001')), plot.RED)
        self.assertEqual(plot.offset_color('lpips', Decimal('.074')), plot.RED)
        self.assertEqual(plot.offset_color('lpips', Decimal('-.001')), plot.GREEN)
        self.assertEqual(plot.offset_color('psnr', Decimal(0)), plot.NEUTRAL)
        self.assertEqual(plot.offset_label('psnr', Decimal('-.01')), '-0.01 dB')
        self.assertIn('E-', plot.offset_label('ssim', Decimal('-.00000001')))

    def test_different_scene_counts_or_gaussian_budgets_are_rejected(self):
        self.rows[1]['scenes'] = 6
        self.write()
        with self.assertRaisesRegex(ValueError, 'Scene counts differ'):
            plot.load_results(self.source, [10, 5, 1])
        self.rows = fixture()
        self.rows[1]['gaussians'] -= 1
        self.write()
        with self.assertRaisesRegex(ValueError, 'Gaussian counts differ'):
            plot.load_results(self.source, [10, 5, 1])

    def test_controls_only_summary_and_zero_latency_have_clear_errors(self):
        self.rows = [r for r in fixture() if r['method'] != 'lightgaussian']
        self.write()
        with self.assertRaisesRegex(ValueError, 'combined re10k_to_acid'):
            plot.load_results(self.source, [10, 5, 1])
        self.rows = fixture()
        self.rows[1]['postprocess_ms'] = 0
        self.write()
        with self.assertRaisesRegex(ValueError, 'latency must be positive'):
            plot.load_results(self.source, [10, 5, 1])

    def test_real_plot_pipeline_writes_three_formats_and_numeric_provenance(self):
        source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        outputs, metadata = plot.plot_results(self.source, self.root / 'synthetic_test_figure', dpi=60)
        self.assertEqual({path.suffix for path in outputs}, {'.pdf', '.png', '.svg'})
        self.assertTrue(all(path.is_file() and path.stat().st_size > 100 for path in outputs))
        self.assertEqual(metadata['source_sha256'], source_hash)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), source_hash)
        self.assertEqual(metadata['scenes'], 7)
        self.assertEqual(len(metadata['annotations']), 9)
        self.assertEqual(metadata['speed_label'], '50\N{EN DASH}70\N{MULTIPLICATION SIGN} faster')
        colors = {(a['metric'], a['percent']): a['color'] for a in metadata['annotations']}
        self.assertEqual(colors['psnr', 1], plot.GREEN)
        self.assertEqual(colors['psnr', 5], plot.RED)
        self.assertEqual(colors['ssim', 10], plot.RED)
        self.assertTrue(all(colors['lpips', p] == plot.RED for p in (10, 5, 1)))
        self.assertTrue((outputs[0].parent / 'figure_metadata.json').is_file())

    def test_summary_status_and_shared_reference_must_match(self):
        (self.root / 'summary_status.json').write_text(json.dumps({'common_scenes': 6}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Summary/status'):
            plot.load_results(self.source, [10, 5, 1])
        (self.root / 'summary_status.json').write_text(json.dumps({'common_scenes': 7, 'complete': True}), encoding='utf-8')
        self.rows.append({**copy.deepcopy(self.rows[0]), 'method': 'merge', 'psnr': 25.})
        self.write()
        with self.assertRaisesRegex(ValueError, '100% reference values differ'):
            plot.load_results(self.source, [10, 5, 1])


if __name__ == '__main__':
    unittest.main()
