"""Check archived file integrity and summarize recorded outcomes."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def planner_outcome(row):
    success = row.get('navfn_success', row.get('success'))
    if type(success) is not bool:
        raise ValueError('Missing boolean planner outcome')
    if success:
        return 'path'
    notes = (row.get('notes') or '').lower()
    if 'empty path' in notes and 'denial' in notes:
        return 'explicit_denial'
    return 'unresolved'


def planner_summary(rows):
    counts = Counter(planner_outcome(row) for row in rows)
    return {
        'total': len(rows),
        **{key: counts[key] for key in ('path', 'explicit_denial', 'unresolved')},
        'unresolved_records': [
            {key: row.get(key) for key in ('env', 'route', 'method', 'condition', 'notes')
             if key in row}
            for row in rows if planner_outcome(row) == 'unresolved'
        ],
    }


def trial_summary(rows):
    groups = {}
    seen = set()
    for row in rows:
        key = (row['experiment_id'], row['sub_label'], row['trial_id'])
        if key in seen:
            raise ValueError(f'Duplicate trial: {key}')
        seen.add(key)
        outcome = row['outcome']
        if outcome not in {'success', 'failure_stuck', 'timeout', 'error'}:
            raise ValueError(f'Unknown trial outcome: {outcome}')
        group = f'{row["placement_method"]}_k{row["k"]}'
        groups.setdefault(group, Counter())[outcome] += 1
    result = {}
    for group, counts in sorted(groups.items()):
        failures = counts['failure_stuck'] + counts['timeout']
        valid = counts['success'] + failures
        result[group] = {
            'total': sum(counts.values()), 'success': counts['success'],
            'navigation_failure': failures, 'infrastructure_error': counts['error'],
            'valid_trials': valid, 'failure_rate': failures / valid if valid else None,
        }
    return result


def check_integrity(root):
    manifest = json.loads((root / 'artifact-checksums.json').read_text())
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Archived file missing or changed: {name}')
        if path.suffix in {'.json', '.jsonl'}:
            load(path)
    return len(manifest)


def summarize(root=ROOT):
    checked = check_integrity(root)
    results = root / 'results'
    gazebo = trial_summary(load(results / 'additional_experiments/exp1/exp1_trials.jsonl'))
    mppi = trial_summary(load(results / 'additional_experiments/exp4/exp4_trials.jsonl'))
    planner_files = {
        'hospital_a_named': 'additional_experiments/exp2b/planner_only_results.json',
        'hospital_a_sweep': 'additional_experiments/route_sweep/navfn_30routes.json',
        'hospital_b_named': 'northern_private/navfn_named.json',
        'hospital_b_sweep': 'northern_private/navfn_stratified.json',
        'smac': 'additional_experiments/exp3/cross_planner_results.json',
    }
    planners = {}
    for name, file in planner_files.items():
        rows = load(results / file)
        if name in {'hospital_a_named', 'smac'}:
            rows = [row for row in rows if row['condition'] == 'greedy_k5']
        planners[name] = planner_summary(rows)
    from scipy.stats import fisher_exact
    greedy, random = gazebo['greedy_k3'], gazebo['random_k3']
    p = fisher_exact([[greedy['navigation_failure'], greedy['success']],
                      [random['navigation_failure'], random['success']]]).pvalue
    return {'archived_files_verified': checked, 'gazebo_dwb': gazebo,
            'gazebo_fisher_two_sided': float(p), 'gazebo_mppi': mppi,
            'planners': planners,
            'hardware_per_trial_records': 'not included in this working copy'}


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2, allow_nan=False))
