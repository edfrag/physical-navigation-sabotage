"""Exp 2B: Compute greedy k=5 placements for 5 hospital routes.

Routes (5 total):
  - corridor_endtoend (already in multi_route_hospital.json k=5)
  - nurse_to_far_room (already in multi_route_hospital.json k=5)
  - 3 deterministically-selected from exp2a candidates (longest path
    per distinct region pair, top 3 unique region pairs by length)

Output: results/additional_experiments/exp2b/placements.json

Runs greedy AOPP solver in host venv (no Nav2 / no container).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import astar, path_cost

INFLATION = 6
OBJ_CELLS = 6  # 0.3 m at 0.05 m/cell
K = 5

OUT_DIR = Path('results/additional_experiments/exp2b')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'placements.json'

EXISTING = json.load(open('results/multi_route_hospital.json'))
CANDIDATES = json.load(open('results/additional_experiments/exp2a/candidate_routes.json'))


def auto_select_3():
    """Top 3 longest baseline paths from distinct region pairs."""
    from collections import defaultdict
    pairs = defaultdict(list)
    for c in CANDIDATES:
        key = tuple(sorted([c['start_region'], c['goal_region']]))
        pairs[key].append(c)
    bests = []
    for k, items in pairs.items():
        items.sort(key=lambda x: -x['baseline_length_m'])
        bests.append((k, items[0]))
    bests.sort(key=lambda x: -x[1]['baseline_length_m'])
    return [b[1] for b in bests[:3]]


def build_routes():
    routes = []
    for r in EXISTING:
        if r.get('k') == 5 and r.get('placements'):
            routes.append({
                'name': r['route'],
                'source': 'pre_existing',
                'start_world': tuple(r['start_world']),
                'goal_world': tuple(r['goal_world']),
                'cached_placements': r['placements'],
                'cached_delay': r.get('greedy_delay'),
                'cached_baseline_cost': r.get('baseline_cost'),
                'is_denial': r.get('is_denial', r.get('greedy_delay', 0) >= 1e6),
            })

    for c in auto_select_3():
        rid = c['candidate_id']
        routes.append({
            'name': f'auto_route_{rid}',
            'source': 'auto_exp2a',
            'candidate_id': rid,
            'start_region': c['start_region'],
            'goal_region': c['goal_region'],
            'start_world': tuple(c['start_world']),
            'goal_world': tuple(c['goal_world']),
            'baseline_length_m': c['baseline_length_m'],
        })
    return routes


def main():
    omap = OccupancyMap('maps/hospital.yaml')
    routes = build_routes()
    print(f'Routes to handle: {len(routes)}')
    for r in routes:
        print(f'  {r["name"]:25s} src={r["source"]:13s} start={r["start_world"]} goal={r["goal_world"]}')

    results = []
    if OUT.exists():
        results = json.load(open(OUT))
        done = {r['route'] for r in results}
        print(f'Resuming: {len(done)} done')
    else:
        done = set()

    for r in routes:
        if r['name'] in done:
            print(f"\n--- {r['name']} — SKIP (cached)")
            continue

        if r['source'] == 'pre_existing':
            results.append({
                'route': r['name'],
                'source': r['source'],
                'k': K,
                'start_world': list(r['start_world']),
                'goal_world': list(r['goal_world']),
                'baseline_cost': r['cached_baseline_cost'],
                'greedy_delay': r['cached_delay'],
                'is_denial': r['is_denial'],
                'placements': r['cached_placements'],
            })
            with open(OUT, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n--- {r['name']} — copied from cache (denial={r['is_denial']})")
            continue

        print(f"\n--- {r['name']} (k={K}) — running greedy")
        t0 = time.time()
        try:
            solver = AOPPSolver(
                omap, r['start_world'], r['goal_world'],
                k=K, obj_w=OBJ_CELLS, obj_h=OBJ_CELLS,
                corridor_d=10, inflation_r=INFLATION,
                candidate_stride=2, min_dist_to_endpoints=15,
            )
            place, gains, _ = solver.solve(verbose=False)
            delay = sum(gains)
            elapsed = time.time() - t0
            is_denial = delay >= 1e6
            pct = (delay / solver.baseline_cost * 100) if not is_denial else None
            status = 'DENIAL' if is_denial else f'{delay:.1f} ({pct:.1f}%)'
            print(f"  result: {status}  ({elapsed:.0f}s elapsed)")
            results.append({
                'route': r['name'],
                'source': r['source'],
                'candidate_id': r.get('candidate_id'),
                'start_region': r.get('start_region'),
                'goal_region': r.get('goal_region'),
                'k': K,
                'start_world': list(r['start_world']),
                'goal_world': list(r['goal_world']),
                'baseline_cost': solver.baseline_cost,
                'greedy_delay': delay,
                'greedy_delay_pct': pct,
                'is_denial': is_denial,
                'placements': [(int(rr), int(cc)) for rr, cc in place],
                'compute_time_s': elapsed,
            })
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results.append({
                'route': r['name'],
                'source': r['source'],
                'k': K,
                'error': str(e),
            })

        with open(OUT, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [checkpoint saved -> {OUT}]")

    print(f"\nDONE. Saved -> {OUT}")
    n_denial = sum(1 for r in results if r.get('is_denial'))
    print(f"Summary: {len(results)} routes, {n_denial} denial")


if __name__ == '__main__':
    main()
