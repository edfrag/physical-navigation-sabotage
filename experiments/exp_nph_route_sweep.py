"""Route sweep for Northern Private Hospital floors.

Mirrors experiments/exp_route_sweep.py but parameterized over floor name.
Selects 30 stratified routes from the candidate pool (top-2 longest per
region pair; fall back to longest remaining), runs greedy AOPP at k=5,
and records per-route topology features.

Usage:
  python3 -m experiments.exp_nph_route_sweep level1
  python3 -m experiments.exp_nph_route_sweep level3
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import astar, inflate

INFLATION = 6
K = 5
OBJ_CELLS = 6
N_WORKERS = 24
# stride = object footprint, matching the "stride ≤ footprint" rule used on
# warehouse (stride=8 / footprint=10) and depot (stride=8 / footprint=8).
# Adjacent candidate centres have exactly non-overlapping footprints.
CANDIDATE_STRIDE = 6
# N_TARGET_ROUTES can be overridden by env var NPH_N_TARGET (set by the
# adaptive launcher based on observed named-route timing + budget halt).
N_TARGET_ROUTES = int(os.environ.get('NPH_N_TARGET', '30'))
MIN_DIST = 15


def select_stratified(candidates, n_target):
    by_pair = defaultdict(list)
    for c in candidates:
        key = tuple(sorted([c['start_region'], c['goal_region']]))
        by_pair[key].append(c)
    selected = []
    for key, items in by_pair.items():
        items.sort(key=lambda x: -x['baseline_length_m'])
        selected.extend(items[:2])
    if len(selected) < n_target:
        remaining = [c for c in candidates if c not in selected]
        remaining.sort(key=lambda x: -x['baseline_length_m'])
        selected.extend(remaining[:n_target - len(selected)])
    selected = selected[:n_target]
    selected.sort(key=lambda x: -x['baseline_length_m'])
    return selected


def topology_features(omap, start_cell, goal_cell, baseline_path):
    grid = omap.grid
    H, W = grid.shape
    free = (grid < 99).astype(np.uint8)
    occ_mask = (grid >= 100)
    wall_dist = distance_transform_edt(free)
    inf_grid = inflate(grid, INFLATION)
    inf_free = (inf_grid < 99)
    feats = {}
    feats['baseline_path_cells'] = float(len(baseline_path))
    feats['baseline_path_length_m'] = float(len(baseline_path) * omap.resolution)
    if baseline_path:
        clearances = [wall_dist[r, c] for (r, c) in baseline_path]
        feats['path_min_clearance_cells'] = float(np.min(clearances))
        feats['path_mean_clearance_cells'] = float(np.mean(clearances))
        feats['path_min_clearance_m'] = float(np.min(clearances) * omap.resolution * 2)
    else:
        feats.update({'path_min_clearance_cells': 0.0,
                      'path_mean_clearance_cells': 0.0,
                      'path_min_clearance_m': 0.0})
    window = 10
    densities = []
    for (r, c) in baseline_path[::5]:
        r0, r1 = max(0, r - window), min(H, r + window + 1)
        c0, c1 = max(0, c - window), min(W, c + window + 1)
        densities.append(occ_mask[r0:r1, c0:c1].mean())
    feats['path_local_obstacle_density'] = float(np.mean(densities)) if densities else 0.0
    feats['path_bottleneck_cells'] = int(sum(
        1 for (r, c) in baseline_path if wall_dist[r, c] <= INFLATION + 1))
    cc, n_components = label(inf_free)
    sr, sc = start_cell; gr, gc = goal_cell
    sc_label = cc[sr, sc] if inf_free[sr, sc] else 0
    gc_label = cc[gr, gc] if inf_free[gr, gc] else 0
    feats['inflated_component_id_match'] = int(sc_label == gc_label and sc_label > 0)
    feats['inflated_component_size_cells'] = (
        int((cc == sc_label).sum()) if sc_label > 0 else 0)
    in_bottleneck = False
    n_segments = 0
    for (r, c) in baseline_path:
        is_b = wall_dist[r, c] <= INFLATION + 1
        if is_b and not in_bottleneck:
            n_segments += 1
            in_bottleneck = True
        elif not is_b and in_bottleneck:
            in_bottleneck = False
    feats['n_bottleneck_segments'] = n_segments
    return feats


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


def run_route(omap, route, idx, floor):
    name = f'{floor}_route_{route["candidate_id"]:02d}'
    s_world = tuple(route['start_world'])
    g_world = tuple(route['goal_world'])
    s_grid_raw = omap.world_to_grid(*s_world)
    g_grid_raw = omap.world_to_grid(*g_world)
    s_cell = find_valid(omap, s_grid_raw, INFLATION)
    g_cell = find_valid(omap, g_grid_raw, INFLATION)
    sw = omap.grid_to_world(*s_cell)
    gw = omap.grid_to_world(*g_cell)
    print(f'\n--- Route {idx} ({name}): start_region={route["start_region"]} '
          f'goal_region={route["goal_region"]} '
          f'baseline_length_m={route["baseline_length_m"]:.1f}')

    baseline = astar(omap.grid, s_cell, g_cell, INFLATION)
    if baseline is None:
        print('  ERROR: no baseline path under inflation=6, skip')
        return {'route': name, 'error': 'no baseline path'}

    feats = topology_features(omap, s_cell, g_cell, baseline)
    print(f"  baseline {len(baseline)} cells, "
          f"min clearance {feats['path_min_clearance_m']:.2f} m, "
          f"bottleneck cells {feats['path_bottleneck_cells']}")

    t0 = time.time()
    try:
        solver = AOPPSolver(
            omap, sw, gw,
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
        pct = (delay / solver.baseline_cost * 100) if not is_denial else None
        status = 'DENIAL' if is_denial else f'{delay:.1f} ({pct:.1f}%)'
        print(f'  greedy: {status}  ({elapsed:.0f}s)')
    except Exception as e:
        elapsed = time.time() - t0
        return {
            'route': name, 'candidate_id': route['candidate_id'],
            'start_region': route['start_region'],
            'goal_region': route['goal_region'],
            'features': feats, 'error': str(e),
            'compute_time_s': elapsed,
        }
    return {
        'route': name, 'candidate_id': route['candidate_id'],
        'start_region': route['start_region'],
        'goal_region': route['goal_region'],
        'start_world': list(sw), 'goal_world': list(gw),
        'k': K, 'baseline_cost': solver.baseline_cost,
        'greedy_delay': delay, 'greedy_delay_pct': pct,
        'is_denial': bool(is_denial),
        'placements': [(int(rr), int(cc)) for rr, cc in place],
        'greedy_marginal_gains': [float(g) for g in gains],
        'features': feats, 'compute_time_s': elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('floor', choices=['level1', 'level3'])
    args = ap.parse_args()
    floor = args.floor
    yaml_path = f'maps/{floor}.yaml'
    candidates_path = Path(f'results/northern_private/{floor}_routes.json')
    out_dir = Path(f'results/northern_private/{floor}_route_sweep')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / 'route_sweep.json'

    print(f'Loading candidates from {candidates_path}')
    candidates = json.load(open(candidates_path))
    print(f'  total candidates: {len(candidates)}')

    selected = select_stratified(candidates, N_TARGET_ROUTES)
    print(f'\nSelected {len(selected)} routes (stratified by region pair):')
    pair_counts = Counter(
        (c['start_region'], c['goal_region']) for c in selected)
    for pair, n in sorted(pair_counts.items()):
        print(f'  {pair[0]}--{pair[1]}: {n}')
    avg_len = np.mean([c['baseline_length_m'] for c in selected])
    print(f'  avg baseline length: {avg_len:.1f} m, '
          f"range [{min(c['baseline_length_m'] for c in selected):.1f}, "
          f"{max(c['baseline_length_m'] for c in selected):.1f}]")

    omap = OccupancyMap(yaml_path)
    results = []
    if out_json.exists():
        existing = json.load(open(out_json))
        results = existing.get('results', [])
        done_ids = {r.get('candidate_id') for r in results}
        print(f'\nResuming with {len(done_ids)} already-done routes')
    else:
        done_ids = set()

    for idx, route in enumerate(selected):
        if route['candidate_id'] in done_ids:
            print(f"\n--- Route {idx} (cand {route['candidate_id']}): SKIP (done)")
            continue
        entry = run_route(omap, route, idx, floor)
        results.append(entry)
        with open(out_json, 'w') as f:
            json.dump({
                'config': {
                    'floor': floor, 'inflation': INFLATION, 'k': K,
                    'obj_cells': OBJ_CELLS,
                    'candidate_stride': CANDIDATE_STRIDE,
                    'n_workers': N_WORKERS, 'min_dist': MIN_DIST,
                },
                'n_total_candidates': len(candidates),
                'n_selected': len(selected),
                'results': results,
            }, f, indent=2, default=str)
        print(f'  [checkpoint -> {out_json}]')

    print('\n' + '=' * 60)
    n_denial = sum(1 for r in results if r.get('is_denial'))
    n_delay = sum(1 for r in results
                  if not r.get('is_denial') and 'greedy_delay' in r)
    n_err = sum(1 for r in results if 'error' in r and 'placements' not in r)
    print(f'DONE {floor}. {len(results)} routes; '
          f'{n_denial} denial, {n_delay} delay, {n_err} error')


if __name__ == '__main__':
    main()
