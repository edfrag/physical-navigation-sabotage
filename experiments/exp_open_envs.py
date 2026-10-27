"""Warehouse + Depot offline AOPP at inflation=6 (paper's headline config).

Adds the missing rows for Table 1's open-environment claim. Output schema
matches results/offline_results_inflation6.json so it can be merged.

Output:
  results/additional_experiments/open_envs/offline_open_envs_inflation6.json

Usage (host venv):
  .venv/bin/python3 -m experiments.exp_open_envs
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import inflate

INFLATION = 6
N_RANDOM_SEEDS = 10

OUT_DIR = Path('results/additional_experiments/open_envs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'offline_open_envs_inflation6.json'


def find_valid(omap, pt, inf_r):
    inf_grid = inflate(omap.grid, inf_r)
    if inf_grid[pt[0], pt[1]] < 99: return pt
    for d in range(1, 200):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = pt[0] + dr, pt[1] + dc
                if (0 <= nr < omap.height and 0 <= nc < omap.width
                        and inf_grid[nr, nc] < 99):
                    return (nr, nc)
    return None


# Endpoints copied from experiments/run_all_remaining.py (legacy inflation=3)
# Re-snapped to inflation=6 below to ensure validity.
MAPS = {
    'warehouse': {
        'yaml': 'maps/warehouse.yaml',
        'start_world': (-7.35, -12.05), 'goal_world': (8.28, 12.84),
        'min_dist': 50, 'obj_cells': 10,
    },
    'depot': {
        'yaml': 'maps/depot.yaml',
        'start_world': (-9.12, -3.72), 'goal_world': (13.28, 3.80),
        'min_dist': 30, 'obj_cells': 8,
    },
}


def make_solver(name, cfg, k):
    omap = OccupancyMap(cfg['yaml'])
    s_grid = omap.world_to_grid(*cfg['start_world'])
    g_grid = omap.world_to_grid(*cfg['goal_world'])
    s_valid = find_valid(omap, s_grid, INFLATION)
    g_valid = find_valid(omap, g_grid, INFLATION)
    s_world = omap.grid_to_world(*s_valid)
    g_world = omap.grid_to_world(*g_valid)
    # candidate_stride=4 (vs 2 in main sweep) — adjacent strided placements
    # produce overlapping object footprints because stride < obj_size = 8-10 cells
    # on these maps; warehouse and depot are open environments where we are
    # demonstrating the *absence* of effective placements, so the worst-case
    # delay we report is robust to stride choice in this range.
    return AOPPSolver(
        omap, s_world, g_world,
        k=k, obj_w=cfg['obj_cells'], obj_h=cfg['obj_cells'],
        corridor_d=10, inflation_r=INFLATION,
        candidate_stride=8,
        min_dist_to_endpoints=cfg['min_dist'],
    ), omap


def main():
    results = []
    if OUT.exists():
        results = json.load(open(OUT))['results']
        done = {(r['env'], r['k']) for r in results}
        print(f"Resuming: {len(done)} (env, k) pairs already done")
    else:
        done = set()

    for name, cfg in MAPS.items():
        for k in [1, 2, 3, 5, 7]:
            if (name, k) in done:
                print(f"\n--- {name} k={k} — SKIP (cached)")
                continue
            print(f"\n--- {name} k={k} (inflation={INFLATION}) ---")
            t0 = time.time()
            solver, omap = make_solver(name, cfg, k)
            print(f"  baseline_cost={solver.baseline_cost:.1f}, "
                  f"candidates={len(solver.candidates)}, "
                  f"workers={solver.n_workers}")
            g_place, g_gains, _ = solver.solve(verbose=False)
            g_delay = sum(g_gains)
            pb_place, pb_delay, _ = solver.pathblock_placement()
            r_delays = [solver.random_placement(seed=s)[1] for s in range(N_RANDOM_SEEDS)]
            pct = (g_delay / solver.baseline_cost * 100) if g_delay < 1e6 else None
            elapsed = time.time() - t0

            entry = {
                'env': name, 'k': k, 'inflation_radius': INFLATION,
                'baseline_cost': solver.baseline_cost,
                'greedy_delay': g_delay,
                'greedy_delay_pct': pct,
                'is_denial': g_delay >= 1e6,
                'greedy_placements': [(int(r), int(c)) for r, c in g_place],
                'greedy_marginal_gains': [float(g) for g in g_gains],
                'pathblock_delay': float(pb_delay),
                'random_delay_mean': float(np.mean(r_delays)),
                'random_delay_std': float(np.std(r_delays)),
                'compute_time_s': elapsed,
            }
            results.append(entry)

            status = "DENIAL" if g_delay >= 1e6 else f"{g_delay:.1f} ({pct:.1f}%)"
            print(f"  -> greedy={status}  pathblock={pb_delay:.1f}  "
                  f"random={np.mean(r_delays):.1f}+/-{np.std(r_delays):.1f}  "
                  f"({elapsed:.0f}s)")

            with open(OUT, 'w') as f:
                json.dump({'results': results, 'inflation_radius': INFLATION}, f,
                          indent=2, default=str)
            print(f"  [checkpoint -> {OUT}]")

    print(f"\nDONE. Saved -> {OUT}")
    n = len(results)
    n_denial = sum(1 for r in results if r['is_denial'])
    print(f"Summary: {n} (env, k) entries, {n_denial} denial outcomes")


if __name__ == '__main__':
    main()
