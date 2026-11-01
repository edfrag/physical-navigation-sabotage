"""Aggregate hardware Phase 4 trials into a paper-ready summary.

Reads JSONL with one trial per line:
  {"trial": 1, "condition": "no_attack", "outcome": "success",
   "duration_s": 24.3, "path_length_m": 7.45}

Outcomes:
  - success: Nav2 reported SUCCEEDED
  - denied: Nav2 timed out (600s default) or returned no plan
  - infrastructure_error: controller crash, robot stuck, lidar dropout
    (excluded from denominator)

Computes per-condition success rate, mean duration on success, denial
rate, and Fisher's exact two-sided p-values for pairs:
  - no_attack vs greedy_k1
  - no_attack vs greedy_k2

Usage:
  python3 -m experiments.analyze_hw_trials \
      --trials results/hw/phase4_trials.jsonl \
      --out results/hw/phase5_summary.md
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import fisher_exact
except ImportError:
    fisher_exact = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    trials = [json.loads(l) for l in open(args.trials) if l.strip()]
    print(f'Loaded {len(trials)} trials')

    by_cond = defaultdict(list)
    for t in trials:
        by_cond[t['condition']].append(t)

    summary = {'config': vars(args), 'n_total': len(trials),
               'per_condition': {}, 'fisher_tests': []}

    for cond, ts in by_cond.items():
        valid = [t for t in ts if t.get('outcome') != 'infrastructure_error']
        n_valid = len(valid)
        n_success = sum(1 for t in valid if t.get('outcome') == 'success')
        n_denied = sum(1 for t in valid if t.get('outcome') == 'denied')
        durations = [t['duration_s'] for t in valid
                     if t.get('outcome') == 'success' and 'duration_s' in t]
        path_lens = [t['path_length_m'] for t in valid
                     if t.get('outcome') == 'success' and 'path_length_m' in t]
        summary['per_condition'][cond] = {
            'n_total': len(ts),
            'n_infrastructure_error': len(ts) - n_valid,
            'n_valid': n_valid,
            'n_success': n_success,
            'n_denied': n_denied,
            'success_rate': n_success / n_valid if n_valid else None,
            'denial_rate': n_denied / n_valid if n_valid else None,
            'mean_duration_s_success': statistics.mean(durations) if durations else None,
            'std_duration_s_success': statistics.stdev(durations) if len(durations) > 1 else 0,
            'mean_path_length_m_success': statistics.mean(path_lens) if path_lens else None,
        }

    # Fisher's exact for pairs of interest
    if fisher_exact and 'no_attack' in by_cond:
        base = summary['per_condition']['no_attack']
        for other_cond in [c for c in by_cond if c != 'no_attack']:
            other = summary['per_condition'][other_cond]
            # 2x2: [[base_success, base_denied], [other_success, other_denied]]
            table = [[base['n_success'], base['n_denied']],
                     [other['n_success'], other['n_denied']]]
            try:
                odds_ratio, p = fisher_exact(table, alternative='two-sided')
                summary['fisher_tests'].append({
                    'comparison': f'no_attack vs {other_cond}',
                    'contingency': table,
                    'odds_ratio': float(odds_ratio),
                    'p_value_two_sided': float(p),
                })
            except Exception as e:
                summary['fisher_tests'].append({
                    'comparison': f'no_attack vs {other_cond}',
                    'error': str(e),
                })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.out).with_suffix('.json')
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    # Human-readable
    md = [f'# Hardware Phase 5 — aggregated results\n',
          f'Total trials: {len(trials)} from {args.trials}\n',
          '## Per-condition\n',
          '| Condition | N | Errors | Success | Denied | Success rate | Denial rate | Mean dur. (s) success | Mean path (m) |',
          '|---|---|---|---|---|---|---|---|---|']
    for cond, info in summary['per_condition'].items():
        md.append(f'| {cond} | {info["n_total"]} | '
                  f'{info["n_infrastructure_error"]} | '
                  f'{info["n_success"]} | '
                  f'{info["n_denied"]} | '
                  f'{(info["success_rate"] or 0):.0%} | '
                  f'{(info["denial_rate"] or 0):.0%} | '
                  f'{info["mean_duration_s_success"]:.1f} ± {info["std_duration_s_success"]:.1f}'
                  if info["mean_duration_s_success"] else 'n/a' '|'
                  f'{info["mean_path_length_m_success"]:.2f}'
                  if info["mean_path_length_m_success"] else 'n/a' '|')
    md.append('\n## Fisher\'s exact (two-sided)\n')
    md.append('| Comparison | 2x2 (success, denied) | Odds ratio | p-value |')
    md.append('|---|---|---|---|')
    for t in summary['fisher_tests']:
        if 'error' in t:
            md.append(f'| {t["comparison"]} | ERR: {t["error"]} | — | — |')
        else:
            md.append(f'| {t["comparison"]} | {t["contingency"]} | '
                      f'{t["odds_ratio"]:.3g} | {t["p_value_two_sided"]:.3g} |')
    Path(args.out).write_text('\n'.join(md))
    print(f'Saved {json_path}')
    print(f'Saved {args.out}')
    for t in summary['fisher_tests']:
        if 'p_value_two_sided' in t:
            print(f'  {t["comparison"]}: p = {t["p_value_two_sided"]:.3g}')


if __name__ == '__main__':
    main()
