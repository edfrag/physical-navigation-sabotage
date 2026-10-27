import tempfile
import unittest
from pathlib import Path

from scripts.check_results import load, planner_outcome, trial_summary


class ResultChecks(unittest.TestCase):
    def test_missing_planner_result_is_unresolved(self):
        for note in ('no result', 'goal rejected', 'goal not accepted', 'exception: timeout', ''):
            self.assertEqual(planner_outcome({'navfn_success': False, 'notes': note}), 'unresolved')

    def test_explicit_planner_denial(self):
        self.assertEqual(planner_outcome({'success': False, 'notes': 'empty path - DENIAL (status=6)'}),
                         'explicit_denial')
        self.assertEqual(planner_outcome({'navfn_success': True}), 'path')
        with self.assertRaises(ValueError):
            planner_outcome({'notes': 'missing outcome'})

    def test_infrastructure_errors_do_not_change_failure_rate(self):
        def trial(i, outcome):
            return {'experiment_id': 'test', 'sub_label': 'arm', 'trial_id': i,
                    'placement_method': 'random', 'k': 3, 'outcome': outcome}
        rows = [trial(0, 'success'), trial(1, 'timeout'), trial(2, 'error')]
        result = trial_summary(rows)['random_k3']
        self.assertEqual(result['total'], 3)
        self.assertEqual(result['valid_trials'], 2)
        self.assertEqual(result['failure_rate'], 0.5)
        with self.assertRaises(ValueError):
            trial_summary(rows + [rows[0]])
        with self.assertRaises(ValueError):
            trial_summary([trial(0, 'unrecognized')])
        self.assertIsNone(trial_summary([trial(0, 'error')])['random_k3']['failure_rate'])

    def test_malformed_trials_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trials.jsonl'
            path.write_text('{"trial": 1}\nnot-json\n')
            with self.assertRaises(ValueError):
                load(path)


if __name__ == '__main__':
    unittest.main()
