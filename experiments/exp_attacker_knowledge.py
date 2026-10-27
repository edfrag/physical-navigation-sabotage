"""Attacker-knowledge sensitivity in the binary surrogate.

Tests whether AOPP attacks survive realistic uncertainty in the adversary's
information. The attacker computes greedy placements on a *degraded* map,
then we evaluate those placements on the unmodified map with binary
inflation. Metrics: denial rate, delay, and path-cost increase relative to
the no-degradation baseline.

Three degradation modes:
  shifted      - map translated by N cells (attacker thinks the building is
                 a bit displaced; e.g., Δ ∈ {-5, +5} cells ≈ ±25 cm)
  eroded       - walls dilated by 2 cells in the attacker's map (attacker's
                 floor plan over-estimates wall thickness)
  wrong_infl   - attacker uses inflation_radius = 4 instead of true 6
                 (mismatch between attacker's assumed Nav2 params and true)

For each (route, mode), run greedy on the degraded map, then evaluate the
resulting placements on the TRUE map under TRUE inflation. Compare to the
baseline (greedy on true map; we already have these in
results/additional_experiments/exp2b/placements.json).

Output:
  results/additional_experiments/attacker_knowledge/results.json

Usage:
  .venv/bin/python3 -m experiments.exp_attacker_knowledge
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.ndimage import binary_dilation

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import astar, inflate

TRUE_INFLATION = 6
K = 5
OBJ_CELLS = 6
N_WORKERS = 24
CANDIDATE_STRIDE = 8   # faster; stride=8 was validated for open envs
MIN_DIST = 15

OUT_DIR = Path('results/additional_experiments/attacker_knowledge')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'results.json'

HOSPITAL_YAML = 'maps/hospital.yaml'


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


def degrade_shifted(omap, dr, dc):
    """Return a new OccupancyMap with the grid translated by (dr, dc).
    Cells that scroll off are wrapped to free space (treats as unknown
    edge regions)."""
    g = omap.grid
    H, W = g.shape
    out = np.zeros_like(g)  # start fully free
    r0_src = max(0, -dr); r0_dst = max(0, dr)
    c0_src = max(0, -dc); c0_dst = max(0, dc)
    h = H - abs(dr); w = W - abs(dc)
    out[r0_dst:r0_dst + h, c0_dst:c0_dst + w] = g[r0_src:r0_src + h, c0_src:c0_src + w]

    # Make a shallow copy of omap with the new grid
    fake = type(omap).__new__(type(omap))
    fake.__dict__.update(omap.__dict__)
    fake.grid = out
    return fake


def degrade_eroded(omap, dilation_iters=2):
    """Walls dilated (made thicker) in attacker's map."""
    occ = (omap.grid >= 100)
    occ_thick = binary_dilation(occ, iterations=dilation_iters)
    out = omap.grid.copy()
    out[occ_thick] = 100
    fake = type(omap).__new__(type(omap))
    fake.__dict__.update(omap.__dict__)
    fake.grid = out
    return fake


def evaluate_on_true_map(omap_true, s_cell, g_cell, placements, baseline_cost):
    """Return (delay_cells, is_denial). Stamp footprints (centered on each
    placement coordinate, matching exp2b convention which uses corner
    placement; here we keep the corner convention to be comparable to
    the existing 2B numbers)."""
    g = omap_true.grid.copy()
    for (r, c) in placements:
        r0 = max(0, r); r1 = min(g.shape[0], r + OBJ_CELLS)
        c0 = max(0, c); c1 = min(g.shape[1], c + OBJ_CELLS)
        g[r0:r1, c0:c1] = 100
    path = astar(g, s_cell, g_cell, TRUE_INFLATION)
    if path is None:
        return float('inf'), True
    return float(len(path) - baseline_cost), False


def run_greedy(omap_attacker, sw, gw, attacker_inflation):
    """Solve greedy on attacker's (possibly degraded) map and inflation."""
    solver = AOPPSolver(
        omap_attacker, sw, gw,
        k=K, obj_w=OBJ_CELLS, obj_h=OBJ_CELLS,
        corridor_d=10, inflation_r=attacker_inflation,
        candidate_stride=CANDIDATE_STRIDE,
        min_dist_to_endpoints=MIN_DIST,
        n_workers=N_WORKERS,
    )
    place, gains, _ = solver.solve(verbose=False)
    return [(int(r), int(c)) for r, c in place], sum(gains), solver.baseline_cost


def main():
    omap_true = OccupancyMap(HOSPITAL_YAML)

    # 4 routes from 2B (the ones we have known greedy placements for)
    e2b = json.load(open('results/additional_experiments/exp2b/placements.json'))
    routes = [r for r in e2b
              if not r.get('error') and r.get('placements')]
    print(f"Routes: {[r['route'] for r in routes]}")

    modes = [
        ('baseline_no_degradation', None),
        ('shifted_+5_cells_x',     ('shifted', 0, 5)),
        ('shifted_-5_cells_x',     ('shifted', 0, -5)),
        ('shifted_+5_cells_y',     ('shifted', 5, 0)),
        ('eroded_walls_2_cells',   ('eroded', 2)),
        ('wrong_inflation_4',      ('wrong_infl', 4)),
        ('wrong_inflation_8',      ('wrong_infl', 8)),
    ]

    results = []
    if OUT.exists():
        prev = json.load(open(OUT))
        results = prev.get('results', [])
        done = {(r['route'], r['mode']) for r in results}
        print(f"Resuming with {len(done)} (route, mode) entries done")
    else:
        done = set()

    for route in routes:
        rname = route['route']
        s_world = route['start_world']; g_world = route['goal_world']
        s_grid = omap_true.world_to_grid(*s_world)
        g_grid = omap_true.world_to_grid(*g_world)
        s_cell = find_valid(omap_true, s_grid, TRUE_INFLATION)
        g_cell = find_valid(omap_true, g_grid, TRUE_INFLATION)
        baseline = astar(omap_true.grid, s_cell, g_cell, TRUE_INFLATION)
        baseline_cost = len(baseline) if baseline else None
        sw = omap_true.grid_to_world(*s_cell)
        gw = omap_true.grid_to_world(*g_cell)
        print(f"\n=== {rname}: baseline {baseline_cost} cells ===")

        for mode_name, mode_spec in modes:
            if (rname, mode_name) in done:
                continue
            t0 = time.time()
            attacker_inflation = TRUE_INFLATION
            if mode_spec is None:
                omap_attacker = omap_true
            elif mode_spec[0] == 'shifted':
                _, dr, dc = mode_spec
                omap_attacker = degrade_shifted(omap_true, dr, dc)
            elif mode_spec[0] == 'eroded':
                _, dilation = mode_spec
                omap_attacker = degrade_eroded(omap_true, dilation)
            elif mode_spec[0] == 'wrong_infl':
                _, attacker_inflation = mode_spec
                omap_attacker = omap_true
            else:
                continue

            try:
                placements, attacker_predicted_delay, _ = run_greedy(
                    omap_attacker, sw, gw, attacker_inflation)
                # Evaluate on TRUE map
                true_delay, true_denial = evaluate_on_true_map(
                    omap_true, s_cell, g_cell, placements, baseline_cost)
                elapsed = time.time() - t0
                attacker_predicted = (
                    'DENIAL' if attacker_predicted_delay >= 1e6
                    else f'{attacker_predicted_delay:.0f}')
                true_str = 'DENIAL' if true_denial else f'{true_delay:.0f} cells'
                print(f"  {mode_name:30s}: attacker thinks={attacker_predicted:>10s}, "
                      f"true Nav2={true_str:>15s}  ({elapsed:.0f}s)")
                results.append({
                    'route': rname,
                    'mode': mode_name,
                    'placements': placements,
                    'attacker_predicted_delay': attacker_predicted_delay,
                    'attacker_predicted_denial': attacker_predicted_delay >= 1e6,
                    'true_delay': true_delay if not true_denial else None,
                    'true_denial': true_denial,
                    'compute_time_s': elapsed,
                })
                with open(OUT, 'w') as f:
                    json.dump({'config': {
                        'true_inflation': TRUE_INFLATION,
                        'k': K, 'obj_cells': OBJ_CELLS,
                        'candidate_stride': CANDIDATE_STRIDE,
                    }, 'results': results}, f, indent=2, default=str)
            except Exception as e:
                print(f"  {mode_name:30s}: ERROR {type(e).__name__}: {e}")
                results.append({'route': rname, 'mode': mode_name, 'error': str(e)})

    # Quick summary
    print("\n" + "=" * 60)
    by_mode = {}
    for r in results:
        by_mode.setdefault(r['mode'], []).append(r)
    print(f"{'Mode':30s} {'Denial':>8} {'Mean delay (cells)':>22}")
    for mode, entries in by_mode.items():
        n = len(entries)
        n_denial = sum(1 for e in entries if e.get('true_denial'))
        delays = [e['true_delay'] for e in entries
                  if e.get('true_delay') is not None]
        avg = f'{np.mean(delays):.1f}' if delays else 'all denial'
        print(f"{mode:30s} {n_denial}/{n:>2} {avg:>22}")


if __name__ == '__main__':
    main()
