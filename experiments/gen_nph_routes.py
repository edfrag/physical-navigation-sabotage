"""Generate stratified candidate route catalogs for Northern Private floors.

For each of Level 1 and Level 3, sample 50 candidate (start, goal) pairs
under the same constraints used for the existing hospital map (Exp 2A):

  - both endpoints in free space under inflation_r=6
  - both endpoints at least 2.0 m from any wall (clearance for the robot)
  - Euclidean start-goal distance >= 5.0 m
  - baseline A* path exists at inflation_r=6
  - start and goal in DIFFERENT spatial regions (3 col-zones x 2 row-bands)

Output:
  results/northern_private/level1_routes.json
  results/northern_private/level3_routes.json

Usage:
  python3 -m experiments.gen_nph_routes
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.map_utils import OccupancyMap
from src.planner import astar, path_cost, inflate

INFLATION = 6
MIN_WALL_DIST_M = 2.0
MIN_SEPARATION_M = 5.0
N_CANDIDATES = 50
RNG_SEED = 1

OUT_DIR = Path('results/northern_private')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def regionalize(r, c, corridor_row, cols_per_zone):
    zone = 'L' if c < cols_per_zone else ('C' if c < 2 * cols_per_zone else 'R')
    side = 'T' if r < corridor_row else 'B'
    return f'{zone}_{side}'


def generate_for_map(yaml_path: str, out_path: Path):
    omap = OccupancyMap(yaml_path)
    res = omap.resolution
    H, W = omap.grid.shape
    print(f'\n=== {yaml_path}: {W}x{H} cells, '
          f'{W*res:.1f}m × {H*res:.1f}m ===')

    occ_mask = (omap.grid >= 100)
    wall_dist_cells = distance_transform_edt(~occ_mask)
    wall_dist_m = wall_dist_cells * res
    inf_grid = inflate(omap.grid, INFLATION)
    valid = (inf_grid < 99) & (wall_dist_m >= MIN_WALL_DIST_M)
    valid_idx = np.argwhere(valid)
    print(f'  valid cells (free + ≥{MIN_WALL_DIST_M}m from wall): {len(valid_idx)}')

    corridor_row = H // 2  # middle row split
    cols_per_zone = W // 3

    regions = {}
    for r, c in valid_idx:
        key = regionalize(r, c, corridor_row, cols_per_zone)
        regions.setdefault(key, []).append((int(r), int(c)))
    print(f'  regions: { {k: len(v) for k, v in regions.items()} }')

    rng = np.random.default_rng(RNG_SEED)
    candidates = []
    attempts = 0
    region_keys = list(regions.keys())
    while len(candidates) < N_CANDIDATES and attempts < N_CANDIDATES * 300:
        attempts += 1
        k1, k2 = rng.choice(region_keys, size=2, replace=False)
        if not regions[k1] or not regions[k2]:
            continue
        s = regions[k1][rng.integers(len(regions[k1]))]
        g = regions[k2][rng.integers(len(regions[k2]))]
        sep_m = np.hypot(s[0] - g[0], s[1] - g[1]) * res
        if sep_m < MIN_SEPARATION_M:
            continue
        path = astar(omap.grid, s, g, INFLATION)
        if path is None:
            continue
        sw = omap.grid_to_world(*s)
        gw = omap.grid_to_world(*g)
        candidates.append({
            'candidate_id': len(candidates),
            'start_world': list(sw),
            'goal_world': list(gw),
            'start_cell': list(s),
            'goal_cell': list(g),
            'start_region': k1,
            'goal_region': k2,
            'baseline_length_cells': len(path),
            'baseline_length_m': float(len(path) * res),
            'separation_m': float(sep_m),
        })
    print(f'  generated {len(candidates)} candidates after {attempts} attempts')

    out_path.write_text(json.dumps(candidates, indent=2, default=str))
    print(f'  saved {out_path}')

    if candidates:
        lens = [c['baseline_length_m'] for c in candidates]
        print(f'  baseline length: min={min(lens):.1f}m, '
              f'median={sorted(lens)[len(lens)//2]:.1f}m, max={max(lens):.1f}m')


def main():
    generate_for_map('maps/level1.yaml', OUT_DIR / 'level1_routes.json')
    generate_for_map('maps/level3.yaml', OUT_DIR / 'level3_routes.json')


if __name__ == '__main__':
    main()
