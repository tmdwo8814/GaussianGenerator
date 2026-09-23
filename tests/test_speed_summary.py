"""A partial or mixed DDP report set must not become a misleading comparison."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


spec = importlib.util.spec_from_file_location(
    '_speed_summary', Path(__file__).resolve().parents[1] / 'scripts/summarize_speed.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SpeedSummaryTests(unittest.TestCase):
    @staticmethod
    def report(rank):
        return {'rank': rank, 'world_size': 2, 'head': 'moment',
                'context_shape_per_rank': [16, 2, 3, 256, 256],
                'requested_samples_per_phase': 10,
                'samples_per_phase': {'throughput': 10, 'stages': 10},
                'throughput': {'step_wall_s': {'mean': 3. + rank}}}

    def test_complete_rank_set_averages_times(self):
        with TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'profile'
            for rank in range(2):
                (Path(tmp) / f'profile.rank{rank}.json').write_text(json.dumps(self.report(rank)))
            reports = summary.load_reports(prefix.with_suffix('.json'))
            self.assertEqual(summary.average(reports, 'throughput', 'step_wall_s'), 3.5)

    def test_missing_mixed_and_incomplete_rank_reports_fail(self):
        with TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'profile'
            with self.assertRaisesRegex(ValueError, 'No completed'):
                summary.load_reports(prefix)
            (Path(tmp) / 'profile.rank0.json').write_text(json.dumps(self.report(0)))
            with self.assertRaisesRegex(ValueError, 'Need all 2'):
                summary.load_reports(prefix)
            report = self.report(1)
            report['context_shape_per_rank'][0] = 8
            other = Path(tmp) / 'profile.rank1.json'
            other.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                summary.load_reports(prefix)
            report = self.report(1)
            report['samples_per_phase']['stages'] = 9
            other.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                summary.load_reports(prefix)


if __name__ == '__main__':
    unittest.main()
