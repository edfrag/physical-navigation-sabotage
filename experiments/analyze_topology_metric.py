"""Analyze topology_metric.json: summary stats, bimodality check, plot.

Outputs:
  results/route_topology_metric/summary.json
  results/route_topology_metric/histogram.png

Reports:
  - 5-number summary (min, Q1, median, Q3, max) of m_route across hospital routes
  - Bimodality assessment (Hartigan dip statistic if scipy_stats available;
    otherwise gap-test + visual)
  - Specific m_route for corridor_endtoend and auto_route_20
  - Per-environment summary (warehouse/depot from open_envs are m_route=0
    by definition since they don't deny at any k tested)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = Path('results/route_topology_metric')
METRIC_PATH = OUT_DIR / 'topology_metric.json'
SUMMARY_PATH = OUT_DIR / 'summary.json'
PLOT_PATH = OUT_DIR / 'histogram.png'


def five_num(vals):
    a = np.array(sorted(vals), dtype=float)
    if len(a) == 0:
        return {}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {'n': int(len(a)), 'min': float(a.min()),
            'q1': float(q1), 'median': float(med),
            'q3': float(q3), 'max': float(a.max()),
            'iqr': float(q3 - q1)}


def gap_bimodality(vals, grid):
    """Crude bimodality flag: largest gap between adjacent populated grid
    bins. If the largest gap is wider than the second-largest by 2x AND
    splits the data into roughly balanced halves, call it 'likely bimodal'.
    Otherwise unimodal."""
    if len(vals) < 4:
        return 'too few points'
    counts = {g: sum(1 for v in vals if v == g) for g in grid}
    populated = [g for g in grid if counts[g] > 0]
    if len(populated) <= 1:
        return 'single mode'
    gaps = [(populated[i+1] - populated[i], i) for i in range(len(populated)-1)]
    gaps.sort(reverse=True)
    if len(gaps) == 1:
        return 'two-mode candidate (single gap)'
    largest, sec = gaps[0][0], gaps[1][0]
    split_idx = gaps[0][1]
    left_n = sum(counts[populated[j]] for j in range(split_idx + 1))
    right_n = sum(counts[populated[j]] for j in range(split_idx + 1, len(populated)))
    balance = min(left_n, right_n) / max(left_n, right_n)
    if largest >= 2 * sec and balance >= 0.4:
        return f'likely bimodal (gap={largest}, balance={balance:.2f})'
    return f'unimodal (max_gap={largest}, sec_gap={sec})'


def main():
    data = json.load(open(METRIC_PATH))
    cfg = data.get('config', {})
    routes = data['results']
    grid = cfg.get('min_dist_grid') or [15, 30, 50, 100]
    print(f"Loaded {len(routes)} routes; grid={grid}")

    by_id = {r['route_id']: r for r in routes}
    ms = [r['m_route_cells'] for r in routes]

    summary = {
        'config': cfg,
        'n_routes': len(routes),
        'hospital_5_num_cells': five_num(ms),
        'hospital_5_num_m': {k: round(v * 0.05, 3) if isinstance(v, (int, float)) else v
                              for k, v in five_num(ms).items()},
        'distribution_per_grid_value': {
            int(g): sum(1 for v in ms if v == g) for g in [0] + list(grid)
        },
        'distribution_shape': gap_bimodality(ms, [0] + list(grid)),
        'corridor_endtoend_m_route_cells': by_id.get('corridor_endtoend', {}).get('m_route_cells'),
        'auto_route_20_m_route_cells': by_id.get('auto_route_20', {}).get('m_route_cells'),
        'auto_route_2_m_route_cells': by_id.get('auto_route_2', {}).get('m_route_cells'),
        'nurse_to_far_room_m_route_cells': by_id.get('nurse_to_far_room', {}).get('m_route_cells'),
        'per_env_note': {
            'hospital': '34 routes, distribution above',
            'warehouse': 'm_route = 0 (offline_open_envs_inflation6.json shows '
                         'no denial at k=1..7 with default MIN_DIST=15)',
            'depot':     'm_route = 0 (same source: never denies at default MIN_DIST)',
        },
        'denial_method': cfg.get('denial_method', 'binary_surrogate_astar'),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {SUMMARY_PATH}")

    # Print key numbers
    fn = summary['hospital_5_num_cells']
    print("\n=== Hospital m_route 5-number (cells / metres) ===")
    print(f"  n        : {fn.get('n')}")
    print(f"  min      : {fn.get('min'):.0f}c / {fn.get('min')*0.05:.2f}m")
    print(f"  Q1       : {fn.get('q1'):.0f}c / {fn.get('q1')*0.05:.2f}m")
    print(f"  median   : {fn.get('median'):.0f}c / {fn.get('median')*0.05:.2f}m")
    print(f"  Q3       : {fn.get('q3'):.0f}c / {fn.get('q3')*0.05:.2f}m")
    print(f"  max      : {fn.get('max'):.0f}c / {fn.get('max')*0.05:.2f}m")
    print(f"  IQR      : {fn.get('iqr'):.0f}c / {fn.get('iqr')*0.05:.2f}m")
    print(f"\nDistribution shape: {summary['distribution_shape']}")
    print(f"Per-bin counts: {summary['distribution_per_grid_value']}")
    print(f"\nNamed paper routes:")
    for k in ['corridor_endtoend', 'nurse_to_far_room', 'auto_route_2', 'auto_route_20']:
        v = summary.get(f'{k}_m_route_cells')
        print(f"  {k:25s}: m_route = {v} cells / {(v or 0)*0.05:.2f} m")

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    bins_grid = [0] + list(grid)
    counts = [summary['distribution_per_grid_value'][g] for g in bins_grid]
    bar_x = np.arange(len(bins_grid))
    bars = ax.bar(bar_x, counts, color='#4C72B0',
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f'{g}\n({g*0.05:.2f}m)' for g in bins_grid])
    ax.set_xlabel('m_route (cells / metres) — largest MIN_DIST that still denies')
    ax.set_ylabel('Hospital routes (count)')
    ax.set_title(f'Topology deniability margin across {len(routes)} hospital routes\n'
                 f'(binary-surrogate A*, k={cfg.get("k", "?")}, '
                 f'stride={cfg.get("candidate_stride", "?")})')
    for b, c in zip(bars, counts):
        if c > 0:
            ax.text(b.get_x() + b.get_width()/2, c + 0.1, str(c),
                    ha='center', va='bottom', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved {PLOT_PATH}")


if __name__ == '__main__':
    main()
