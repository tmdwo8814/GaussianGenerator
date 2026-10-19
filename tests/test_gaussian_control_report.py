import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.evaluation.gaussian_control_report import CONDITIONS, GaussianControlReport


class GaussianControlReportTests(unittest.TestCase):
    def test_scene_averages_overlap_baseline_deltas_and_matched_pair_differences(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = GaussianControlReport(root, {'camera_protocol': 'shared_final_pose', 'checkpoint': 'Final.ckpt'})
            for index, (overlap, psnr) in enumerate((('small', 20.), ('ignore', 40.))):
                values = {name: {'psnr': psnr, 'ssim': .8, 'lpips': .2} for name in CONDITIONS}
                values['prune_050']['psnr'] -= 4
                values['redistribute_050']['psnr'] -= 1
                values['redistribute_050']['lpips'] -= .05
                report.add_scene({'scene': str(index)}, overlap, values)
            self.assertEqual(report.average('final_100')['psnr'], 30.)
            self.assertEqual(report.average('final_100', 'small')['psnr'], 20.)
            self.assertEqual(report.counts['final_100']['overall'], 2)
            self.assertEqual(report.counts['final_100']['small'], 1)
            self.assertEqual(report.average('final_100', 'large'), {})
            summary = report.finish()
            self.assertIn('3.000000', summary)
            self.assertIn('-0.050000', summary)
            self.assertIn('COMPLETE', (root / 'run_status.txt').read_text())
            for name in CONDITIONS:
                self.assertTrue((root / f'{name}.txt').is_file())
            rows = [json.loads(line) for line in (root / 'per_scene.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(set(rows[0]['conditions']), set(CONDITIONS))
            with self.assertRaises(FileExistsError):
                GaussianControlReport(root, {})

    def test_empty_evaluation_does_not_produce_final_results(self):
        with TemporaryDirectory() as temporary:
            report = GaussianControlReport(Path(temporary), {})
            with self.assertRaisesRegex(RuntimeError, 'No evaluation scenes'):
                report.finish()
            self.assertEqual((Path(temporary) / 'run_status.txt').read_text(), 'RUNNING\n')
            self.assertFalse((Path(temporary) / 'summary.txt').exists())
            with self.assertRaises(ValueError):
                report.add_scene({}, 'small', {'final_100': {}})


if __name__ == '__main__':
    unittest.main()
