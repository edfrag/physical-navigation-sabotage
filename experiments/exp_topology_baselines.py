"""Topology-aware baselines on four maps.

Runs three topology-aware baselines (min-cut, chokepoint, edge
betweenness) at k ∈ {1, 2, 3, 5, 7} on maze, hospital, warehouse, depot.
Compares each baseline's placements + delay/denial against the existing
greedy AOPP results, reporting Jaccard placement-overlap.

Output:
  results/topology_baselines/baselines_main.json
  results/topology_baselines/comparison_table.tex

Usage:
  .venv/bin/python3 -m experiments.exp_topology_baselines
"""
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from src.map_utils import OccupancyMap
from src.planner import astar
from src.topology_baselines import (
    baseline_min_cut, baseline_chokepoint, baseline_betweenness,
    compute_min_cut_score, compute_betweenness_score,
    evaluate_placements, placement_jaccard,
)

# Canonical Table-1 routes & per-env config (inflation_r = 6).
# Sources: experiments/run_inflation6.py, experiments/exp_open_envs.py.
ENVS = [
    {
        'env': 'maze',
        'yaml': 'maps/maze.yaml',
        'start_world': (-7.535, -7.855),
        'goal_world': (6.775, 7.085),
        'obj_w': 10, 'obj_h': 10,
        'min_dist': 30,
    },
    {
        'env': 'hospital',
        'yaml': 'maps/hospital.yaml',
        'start_world': (1.525, 12.125),
        'goal_world': (53.525, 12.125),
        'obj_w': 6, 'obj_h': 6,
        'min_dist': 20,
    },
    {
        'env': 'warehouse',
        'yaml': 'maps/warehouse.yaml',
        'start_world': (-7.35, -12.05),
        'goal_world': (8.28, 12.84),
        'obj_w': 10, 'obj_h': 10,
        'min_dist': 50,
    },
    {
        'env': 'depot',
        'yaml': 'maps/depot.yaml',
        'start_world': (-9.12, -3.72),
        'goal_world': (13.28, 3.80),
        'obj_w': 8, 'obj_h': 8,
        'min_dist': 30,
    },
]
INFLATION = 6
K_VALUES = [1, 2, 3, 5, 7]

OUT_DIR = Path('results/topology_baselines')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / 'baselines_main.json'
OUT_TEX = OUT_DIR / 'comparison_table.tex'


def load_greedy_reference():
    """Greedy data lives in two files. Build {env: {k: entry}}."""
    ref = {}
    p1 = Path('results/offline_results_inflation6.json')
    p2 = Path('results/additional_experiments/open_envs/offline_open_envs_inflation6.json')
    if p1.exists():
        d = json.load(open(p1))
        results = d if isinstance(d, list) else d.get('results', [])
        for r in results:
            env = r.get('env')
            k = r.get('k')
            if env and k is not None:
                ref.setdefault(env, {})[k] = r
    if p2.exists():
        d = json.load(open(p2))
        for r in d.get('results', []):
            env = r.get('env')
            k = r.get('k')
            if env and k is not None:
                ref.setdefault(env, {})[k] = r
    return ref


def find_valid(omap, pt, inf_r):
    from src.planner import inflate
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
    greedy_ref = load_greedy_reference()
    print(f"Loaded greedy reference: { {e: list(d.keys()) for e, d in greedy_ref.items()} }")

    results = []
    if OUT_JSON.exists():
        results = json.load(open(OUT_JSON)).get('results', [])
        done = {(r['env'], r['method'], r['k']) for r in results}
        print(f"Resuming with {len(done)} entries done")
    else:
        done = set()

    for cfg in ENVS:
        env = cfg['env']
        print(f"\n{'='*60}\n=== {env} ===\n{'='*60}")
        omap = OccupancyMap(cfg['yaml'])
        s_grid_raw = omap.world_to_grid(*cfg['start_world'])
        g_grid_raw = omap.world_to_grid(*cfg['goal_world'])
        s_cell = find_valid(omap, s_grid_raw, INFLATION)
        g_cell = find_valid(omap, g_grid_raw, INFLATION)
        sw = omap.grid_to_world(*s_cell)
        gw = omap.grid_to_world(*g_cell)
        print(f"  start_cell={s_cell} goal_cell={g_cell} obj={cfg['obj_w']}x{cfg['obj_h']} "
              f"min_dist={cfg['min_dist']}")

        # Pre-compute scores once per env (don't depend on k)
        # min-cut (stealth-aware: cut forced outside both endpoint zones)
        if any(('min_cut', k) not in {(r['method'], r['k']) for r in results
                                       if r['env'] == env} for k in K_VALUES):
            print(f"  Computing min-cut (min_dist={cfg['min_dist']})...", flush=True)
            t0 = time.time()
            mincut_score, cut_size = compute_min_cut_score(
                omap, s_cell, g_cell, INFLATION, min_dist=cfg['min_dist'])
            print(f"    cut_size={cut_size}, time={time.time()-t0:.1f}s")
        else:
            mincut_score, cut_size = None, None

        # betweenness
        if any(('betweenness', k) not in {(r['method'], r['k']) for r in results
                                           if r['env'] == env} for k in K_VALUES):
            print(f"  Computing betweenness (200 sampled paths)...", flush=True)
            t0 = time.time()
            betw_score = compute_betweenness_score(omap, INFLATION)
            print(f"    max_score={betw_score.max()}, time={time.time()-t0:.1f}s")
        else:
            betw_score = None

        for k in K_VALUES:
            for method in ['min_cut', 'chokepoint', 'betweenness']:
                if (env, method, k) in done:
                    print(f"  SKIP {env} {method} k={k}")
                    continue
                t0 = time.time()
                if method == 'min_cut':
                    pl, cs, _ = baseline_min_cut(
                        omap, s_cell, g_cell, k,
                        cfg['obj_w'], cfg['obj_h'], INFLATION,
                        cfg['min_dist'], score_cache=mincut_score)
                    extra = {'cut_size': cs}
                elif method == 'chokepoint':
                    pl = baseline_chokepoint(
                        omap, s_cell, g_cell, k,
                        cfg['obj_w'], cfg['obj_h'], INFLATION,
                        cfg['min_dist'])
                    extra = {}
                elif method == 'betweenness':
                    pl, _ = baseline_betweenness(
                        omap, s_cell, g_cell, k,
                        cfg['obj_w'], cfg['obj_h'], INFLATION,
                        cfg['min_dist'], score_cache=betw_score)
                    extra = {}
                # Evaluate on the original grid
                delay, is_denial, base_cost = evaluate_placements(
                    omap, s_cell, g_cell, pl, cfg['obj_w'], cfg['obj_h'],
                    INFLATION)
                # Compare to greedy
                greedy_entry = greedy_ref.get(env, {}).get(k, {})
                greedy_pl = greedy_entry.get('greedy_placements', [])
                # normalize greedy placements to tuples
                greedy_pl_t = [tuple(p) for p in greedy_pl]
                jaccard = placement_jaccard(
                    [tuple(p) for p in pl], greedy_pl_t,
                    cfg['obj_w'], cfg['obj_h'])
                elapsed = time.time() - t0
                tag = 'DENIAL' if is_denial else f'{delay:.1f}'
                print(f"  {method:12s} k={k}: delay={tag:>10s}  "
                      f"|placements|={len(pl)}  jaccard_w_greedy={jaccard:.2f}  "
                      f"({elapsed:.1f}s)")
                entry = {
                    'env': env,
                    'method': method,
                    'k': k,
                    'baseline_cost': base_cost,
                    'placements': [(int(r), int(c)) for r, c in pl],
                    'delay': delay,
                    'is_denial': bool(is_denial),
                    'jaccard_with_greedy': jaccard,
                    'compute_time_s': elapsed,
                    'greedy_delay': greedy_entry.get('greedy_delay'),
                    'greedy_is_denial': greedy_entry.get('is_denial'),
                    **extra,
                }
                results.append(entry)
                with open(OUT_JSON, 'w') as f:
                    json.dump({
                        'config': {
                            'inflation': INFLATION,
                            'k_values': K_VALUES,
                            'envs': [cfg['env'] for cfg in ENVS],
                        },
                        'results': results,
                    }, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("DONE — see baselines_main.json. Run gen_comparison_table next.")


if __name__ == '__main__':
    main()
