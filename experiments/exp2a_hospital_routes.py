"""Exp 2A: Enumerate 50 hospital candidate start/goal pairs.

Stratified spatial sampling: split the hospital map into 6 regions (left wing
top/bot, centre top/bot, right wing top/bot) and sample candidates such that
start and goal are in different regions.

A candidate is valid iff:
  - both poses in free space (under inflation=6)
  - at least 2.0 m from any wall in raw grid
  - Euclidean start-goal distance >= 5.0 m
  - baseline path exists under inflation=6 A*

Output:
  results/additional_experiments/exp2a/candidate_routes.json   (all 50)
  results/additional_experiments/exp2a/candidate_routes_overview.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from src.map_utils import OccupancyMap
from src.planner import astar, path_cost, inflate

INFLATION = 6
MIN_WALL_DIST_M = 2.0
MIN_SEPARATION_M = 5.0
N_CANDIDATES = 50
RNG_SEED = 1

OUT_DIR = Path('results/additional_experiments/exp2a')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    omap = OccupancyMap('maps/hospital.yaml')
    res = omap.resolution
    H, W = omap.grid.shape
    print(f"Hospital: {W} x {H}, res={res} m/cell, dimensions {W*res:.1f}m × {H*res:.1f}m")

    # Wall distance map (m)
    occ_mask = (omap.grid >= 100)
    wall_dist_cells = distance_transform_edt(~occ_mask)
    wall_dist_m = wall_dist_cells * res
    inf_grid = inflate(omap.grid, INFLATION)
    valid = (inf_grid < 99) & (wall_dist_m >= MIN_WALL_DIST_M)
    valid_idx = np.argwhere(valid)
    print(f"Valid cells (free + ≥{MIN_WALL_DIST_M}m from wall): {len(valid_idx)}")

    # Region partition: split horizontally into 3 zones (left/centre/right)
    # and vertically into 2 (top/bot of corridor at row ~155)
    # Hospital corridor mid-row ≈ 155 in grid coords (we used 155 in placements)
    corridor_row = 155
    cols_per_zone = W // 3

    def region(r, c):
        zone = "L" if c < cols_per_zone else ("C" if c < 2 * cols_per_zone else "R")
        side = "T" if r < corridor_row else "B"
        return f"{zone}_{side}"

    # Group valid cells by region
    regions = {}
    for r, c in valid_idx:
        regions.setdefault(region(r, c), []).append((r, c))
    print(f"Regions: { {k: len(v) for k, v in regions.items()} }")

    # Sample candidates with start and goal in DIFFERENT regions
    rng = np.random.default_rng(RNG_SEED)
    candidates = []
    attempts = 0
    region_keys = list(regions.keys())
    while len(candidates) < N_CANDIDATES and attempts < N_CANDIDATES * 200:
        attempts += 1
        rk_s, rk_g = rng.choice(region_keys, 2, replace=False)
        s_pool = regions[rk_s]
        g_pool = regions[rk_g]
        s_cell = s_pool[rng.integers(len(s_pool))]
        g_cell = g_pool[rng.integers(len(g_pool))]
        s_world = omap.grid_to_world(*s_cell)
        g_world = omap.grid_to_world(*g_cell)
        dist_m = np.hypot(s_world[0] - g_world[0], s_world[1] - g_world[1])
        if dist_m < MIN_SEPARATION_M:
            continue
        # Baseline path
        path = astar(omap.grid, tuple(s_cell), tuple(g_cell), INFLATION)
        if path is None:
            continue
        path_cells = path_cost(path)
        candidates.append({
            'candidate_id': len(candidates),
            'start_world': [round(s_world[0], 3), round(s_world[1], 3)],
            'goal_world':  [round(g_world[0], 3), round(g_world[1], 3)],
            'start_cell':  [int(s_cell[0]), int(s_cell[1])],
            'goal_cell':   [int(g_cell[0]), int(g_cell[1])],
            'start_region': rk_s,
            'goal_region':  rk_g,
            'baseline_cost_cells': float(path_cells),
            'baseline_length_m': float(path_cells * res),
            'euclidean_dist_m': float(dist_m),
            'route_name': None,
            'selected': False,
        })

    print(f"\nGenerated {len(candidates)} candidates after {attempts} attempts")

    out = OUT_DIR / 'candidate_routes.json'
    with open(out, 'w') as f:
        json.dump(candidates, f, indent=2)
    print(f"Saved: {out}")

    # Render overview
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(omap.grid, cmap='gray_r', vmin=-1, vmax=100, aspect='equal')
    cmap = plt.cm.tab20
    for i, c in enumerate(candidates):
        sr, sc = c['start_cell']; gr, gc = c['goal_cell']
        col = cmap(i % 20)
        ax.plot([sc, gc], [sr, gr], '-', color=col, alpha=0.7, lw=1.0)
        ax.plot(sc, sr, 'o', color=col, markersize=6)
        ax.plot(gc, gr, '^', color=col, markersize=6)
        ax.text((sc + gc) / 2, (sr + gr) / 2, str(i),
                fontsize=7, color='black',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6))
    ax.set_title(f'Hospital — {len(candidates)} candidate routes (start ●, goal ▲)')
    plt.tight_layout()
    fig_path = OUT_DIR / 'candidate_routes_overview.png'
    plt.savefig(fig_path, dpi=120)
    print(f"Saved: {fig_path}")

    # Markdown summary
    md = OUT_DIR / 'candidate_routes_overview.md'
    with open(md, 'w') as f:
        f.write("# Hospital candidate routes\n\n")
        f.write(f"Generated {len(candidates)} candidates, "
                f"min Euclidean separation {MIN_SEPARATION_M} m, "
                f"min wall clearance {MIN_WALL_DIST_M} m, inflation={INFLATION}.\n\n")
        f.write("Operator: review the overview PNG, then for the routes you wish to attack, "
                "open `candidate_routes.json` and set `selected: true` plus `route_name`.\n\n")
        f.write("| ID | start (m) | goal (m) | start region | goal region | baseline (m) |\n")
        f.write("|----|-----------|----------|--------------|-------------|--------------|\n")
        # Group by start region for readability
        for c in sorted(candidates, key=lambda x: (x['start_region'], x['goal_region'])):
            f.write(f"| {c['candidate_id']} | "
                    f"({c['start_world'][0]:.1f}, {c['start_world'][1]:.1f}) | "
                    f"({c['goal_world'][0]:.1f}, {c['goal_world'][1]:.1f}) | "
                    f"{c['start_region']} | {c['goal_region']} | "
                    f"{c['baseline_length_m']:.1f} |\n")
    print(f"Saved: {md}")
    print(f"\nDONE. {len(candidates)} candidates, "
          f"baseline length range "
          f"[{min(c['baseline_length_m'] for c in candidates):.1f}, "
          f"{max(c['baseline_length_m'] for c in candidates):.1f}] m")


if __name__ == '__main__':
    main()
