"""Topology metric m_route — wider grid for cross-hospital comparison.

Identical methodology to exp_topology_metric.py but uses the wider grid
{50, 75, 100, 125, 150, 200} cells. The previous {15, 30, 50, 100} grid
right-censored 47% of hospital routes at the ceiling; this grid resolves
the upper tail.

Routes: 30 hospital sweep + 4 named paper routes (corridor_endtoend,
nurse_to_far_room, auto_route_2, auto_route_20).

Output: results/route_topology_metric_wide/topology_metric_wide.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import astar, inflate

# Reuse helpers from exp_topology_metric to stay consistent
from experiments.exp_topology_metric import (
    INFLATION, K, OBJ_CELLS, N_WORKERS, CANDIDATE_STRIDE,
    HOSPITAL_YAML, SWEEP_PATH, E2B_PATH,
    find_valid, run_eval, collect_routes,
)

MIN_DIST_GRID = [50, 75, 100, 125, 150, 200]

OUT_DIR = Path('results/route_topology_metric_wide')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'topology_metric_wide.json'


def find_m_route_wide(omap, sw, gw, route_id, log):
    """Binary search over the wide MIN_DIST grid. Returns
    (m_route_cells, sweep_results, at_max).
    m_route = 0 if MIN_DIST_GRID[0] doesn't deny.
    m_route >= MIN_DIST_GRID[-1] if even the largest denies."""
    g = MIN_DIST_GRID
    sweep = {}

    def eval_at(i):
        if g[i] in sweep:
            return sweep[g[i]]['is_denial']
        d, total, t, n = run_eval(omap, sw, gw, g[i])
        sweep[g[i]] = {'is_denial': d, 'total_delay': total,
                       'time_s': round(t, 1), 'n_candidates': n}
        log(f"    MIN_DIST={g[i]:3d} cand={n:5d} denial={d} delay={total:.0f} ({t:.0f}s)")
        return d

    # Test largest first
    if eval_at(len(g) - 1):
        return g[-1], sweep, True
    # Test smallest
    if not eval_at(0):
        return 0, sweep, False
    lo, hi = 0, len(g) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if eval_at(mid):
            lo = mid
        else:
            hi = mid
    return g[lo], sweep, False


def main():
    omap, routes = collect_routes()
    print(f"Total routes: {len(routes)} ({sum(1 for r in routes if r['source']=='sweep')} sweep + "
          f"{sum(1 for r in routes if r['source']=='exp2b')} named)")
    print(f"MIN_DIST grid: {MIN_DIST_GRID}")

    results = []
    done = set()
    if OUT.exists():
        prev = json.load(open(OUT))
        results = prev.get('results', [])
        done = {r['route_id'] for r in results}
        print(f"Resuming with {len(done)} routes done")

    log_lines = []
    def log(s):
        print(s)
        log_lines.append(s)

    t_start = time.time()
    for i, route in enumerate(routes):
        rid = route['route_id']
        if rid in done:
            print(f"\n[{i+1}/{len(routes)}] {rid}: SKIP (done)")
            continue
        sw = tuple(route['start_world'])
        gw = tuple(route['goal_world'])
        baseline_cost = route['baseline_cost']
        print(f"\n[{i+1}/{len(routes)}] {rid} (source={route['source']}, baseline={baseline_cost} cells):")
        m_cells, sweep, at_max = find_m_route_wide(omap, sw, gw, rid, log)
        m_m = round(m_cells * 0.05, 3) if m_cells > 0 else 0.0
        denial_at_default = sweep.get(50, {}).get('is_denial', None)
        if denial_at_default is None:
            denial_at_default = m_cells >= 50
        entry = {
            'route_id': rid,
            'source': route['source'],
            'candidate_id': route.get('candidate_id'),
            'baseline_path_len_cells': baseline_cost,
            'baseline_path_len_m': round((baseline_cost or 0) * 0.05, 2),
            'm_route_cells': int(m_cells),
            'm_route_m': float(m_m),
            'm_route_at_grid_max': bool(at_max),
            'denial_at_min_dist_50': bool(denial_at_default),
            'min_dist_grid': MIN_DIST_GRID,
            'sweep': sweep,
        }
        results.append(entry)
        with open(OUT, 'w') as f:
            json.dump({
                'config': {
                    'inflation': INFLATION, 'k': K, 'obj_cells': OBJ_CELLS,
                    'candidate_stride': CANDIDATE_STRIDE,
                    'denial_method': 'binary_surrogate_astar',
                    'min_dist_grid': MIN_DIST_GRID,
                    'note': 'Wider grid for cross-hospital comparison; '
                            'addresses right-censoring at MIN_DIST=100.',
                },
                'results': results,
            }, f, indent=2, default=str)
        elapsed = time.time() - t_start
        n_done = len(results) - len(done)
        if n_done > 0:
            avg = elapsed / n_done
            remaining = (len(routes) - i - 1) * avg
            print(f"  -> m_route = {m_cells} cells ({m_m} m). "
                  f"Elapsed {elapsed/60:.1f} min, ~{remaining/60:.0f} min left")

    print("\n" + "=" * 60)
    print(f"Done. {len(results)} routes evaluated.")


if __name__ == '__main__':
    main()
