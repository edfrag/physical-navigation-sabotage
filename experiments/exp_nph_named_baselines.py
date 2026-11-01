"""Run greedy k=5 + 3 topology baselines (min-cut, chokepoint, edge
betweenness) on the named routes for one Northern Private floor.

For each named route on the given floor:
  - Run greedy k=5 (AOPPSolver) -> placements + delay + denial
  - Run min-cut, chokepoint, betweenness baselines at k=5
  - Compute Jaccard placement-overlap with greedy

Resumable via checkpoint after each (route, method).

Output:
  results/northern_private/{floor}_named_baselines.json

Usage:
  python3 -m experiments.exp_nph_named_baselines level1
  python3 -m experiments.exp_nph_named_baselines level3
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import astar, inflate
from src.topology_baselines import (
    baseline_min_cut, baseline_chokepoint, baseline_betweenness,
    compute_min_cut_score, compute_betweenness_score,
    evaluate_placements, placement_jaccard,
)

INFLATION = 6
K = 5
OBJ_CELLS = 6
N_WORKERS = 24
CANDIDATE_STRIDE = 6   # matches stride=footprint rule
MIN_DIST = 15


def find_valid(omap, pt, inf_r):
    inf_grid = inflate(omap.grid, inf_r)
    if inf_grid[pt[0], pt[1]] < 99:
        return pt
    for d in range(1, 50):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = pt[0] + dr, pt[1] + dc
                if (0 <= nr < omap.height and 0 <= nc < omap.width
                        and inf_grid[nr, nc] < 99):
                    return (nr, nc)
    return pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('floor', choices=['level1', 'level3'])
    args = ap.parse_args()
    floor = args.floor

    yaml_path = f'maps/{floor}.yaml'
    named_path = Path(f'results/northern_private/{floor}_named_routes.json')
    out_path = Path(f'results/northern_private/{floor}_named_baselines.json')
    if not named_path.exists():
        print(f'ERROR: {named_path} not found. Run pin_nph_named_routes first.')
        return

    named = json.load(open(named_path))
    routes = named['routes']
    print(f'Floor: {floor} | {len(routes)} named routes')

    omap = OccupancyMap(yaml_path)

    # Resume if partial output exists
    results = []
    done = set()
    if out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get('results', [])
        done = {(r['route_id'], r['method']) for r in results}
        print(f'Resuming with {len(done)} (route, method) entries done')

    # Cache per-floor scores (betweenness is global)
    betw_score = None
    # min_cut score is per-route (depends on s, g)

    t_floor_start = time.time()
    for route in routes:
        rid = route['route_id']
        s_world = tuple(route['start_world'])
        g_world = tuple(route['goal_world'])
        s_cell = tuple(route['start_cell'])
        g_cell = tuple(route['goal_cell'])
        baseline_cost = route['baseline_cost_cells']
        print(f"\n=== {rid} ({route['baseline_length_m']:.1f}m baseline) ===")

        # --- Greedy k=5 ---
        if (rid, 'greedy') not in done:
            t0 = time.time()
            try:
                solver = AOPPSolver(
                    omap, s_world, g_world,
                    k=K, obj_w=OBJ_CELLS, obj_h=OBJ_CELLS,
                    corridor_d=10, inflation_r=INFLATION,
                    candidate_stride=CANDIDATE_STRIDE,
                    min_dist_to_endpoints=MIN_DIST,
                    n_workers=N_WORKERS,
                )
                place, gains, _ = solver.solve(verbose=False)
                delay = sum(gains)
                is_denial = delay >= 1e6
                elapsed = time.time() - t0
                placements = [(int(r), int(c)) for r, c in place]
                tag = 'DENIAL' if is_denial else f'{delay:.1f}'
                print(f'  greedy:      delay={tag}  placements={placements}  ({elapsed:.0f}s)')
                results.append({
                    'route_id': rid, 'method': 'greedy', 'k': K,
                    'placements': placements,
                    'delay': float(delay), 'is_denial': bool(is_denial),
                    'baseline_cost': float(solver.baseline_cost),
                    'compute_time_s': float(elapsed),
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f'  greedy: ERROR {type(e).__name__}: {e}')
                results.append({
                    'route_id': rid, 'method': 'greedy', 'k': K,
                    'error': str(e), 'compute_time_s': float(elapsed),
                })
            with open(out_path, 'w') as f:
                json.dump({'config': {'floor': floor, 'inflation': INFLATION,
                                      'k': K, 'obj_cells': OBJ_CELLS,
                                      'candidate_stride': CANDIDATE_STRIDE,
                                      'min_dist': MIN_DIST},
                          'results': results}, f, indent=2, default=str)

        # Get greedy placements to compute Jaccard with
        greedy_rec = next((r for r in results
                           if r['route_id'] == rid and r['method'] == 'greedy'), None)
        greedy_pl = (greedy_rec or {}).get('placements', [])

        # --- Compute betweenness once per floor ---
        if betw_score is None and any((rid, m) not in done
                                       for m in ['betweenness']):
            print(f'  Computing betweenness (sampled paths)...', flush=True)
            t0 = time.time()
            betw_score = compute_betweenness_score(omap, INFLATION)
            print(f'    max={betw_score.max()}, time={time.time()-t0:.0f}s')

        # Min-cut needs per-route compute (depends on s, g)
        # Other baselines also per-route
        for method in ['min_cut', 'chokepoint', 'betweenness']:
            if (rid, method) in done:
                continue
            t0 = time.time()
            try:
                if method == 'min_cut':
                    pl, cs, _ = baseline_min_cut(
                        omap, s_cell, g_cell, K,
                        OBJ_CELLS, OBJ_CELLS, INFLATION, MIN_DIST)
                    extra = {'cut_size': cs}
                elif method == 'chokepoint':
                    pl = baseline_chokepoint(
                        omap, s_cell, g_cell, K,
                        OBJ_CELLS, OBJ_CELLS, INFLATION, MIN_DIST)
                    extra = {}
                elif method == 'betweenness':
                    pl, _ = baseline_betweenness(
                        omap, s_cell, g_cell, K,
                        OBJ_CELLS, OBJ_CELLS, INFLATION, MIN_DIST,
                        score_cache=betw_score)
                    extra = {}
                delay, denial, base = evaluate_placements(
                    omap, s_cell, g_cell, pl, OBJ_CELLS, OBJ_CELLS, INFLATION)
                jacc = placement_jaccard(
                    [tuple(p) for p in pl], [tuple(p) for p in greedy_pl],
                    OBJ_CELLS, OBJ_CELLS)
                elapsed = time.time() - t0
                tag = 'DENIAL' if denial else f'{delay:.1f}'
                print(f'  {method:12s}: delay={tag}  jaccard_w_greedy={jacc:.2f}  ({elapsed:.0f}s)')
                results.append({
                    'route_id': rid, 'method': method, 'k': K,
                    'placements': [(int(r), int(c)) for r, c in pl],
                    'delay': float(delay), 'is_denial': bool(denial),
                    'jaccard_with_greedy': float(jacc),
                    'compute_time_s': float(elapsed),
                    **extra,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f'  {method:12s}: ERROR {type(e).__name__}: {e}')
                results.append({
                    'route_id': rid, 'method': method, 'k': K,
                    'error': str(e), 'compute_time_s': float(elapsed),
                })
            with open(out_path, 'w') as f:
                json.dump({'config': {'floor': floor, 'inflation': INFLATION,
                                      'k': K, 'obj_cells': OBJ_CELLS,
                                      'candidate_stride': CANDIDATE_STRIDE,
                                      'min_dist': MIN_DIST},
                          'results': results}, f, indent=2, default=str)

    floor_elapsed = time.time() - t_floor_start
    greedy_times = [r['compute_time_s'] for r in results
                    if r['method'] == 'greedy' and 'compute_time_s' in r]
    avg_greedy = sum(greedy_times) / len(greedy_times) if greedy_times else 0
    print(f"\n=== {floor} named-routes DONE: total {floor_elapsed/3600:.2f}h, "
          f"avg greedy per route {avg_greedy/60:.1f} min ===")


if __name__ == '__main__':
    main()
