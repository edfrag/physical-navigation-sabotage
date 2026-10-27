"""Generate new placements from explicitly supplied map and endpoint poses."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import astar, inflate, path_cost


def parse_xy(s):
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 2:
        raise ValueError(f'Expected x,y, got {s}')
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True)
    ap.add_argument('--start-world', required=True, help='x,y in metres')
    ap.add_argument('--goal-world', required=True, help='x,y in metres')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--inflation-r', type=int, default=6)
    ap.add_argument('--obj-cells', type=int, default=6)
    ap.add_argument('--min-dist', type=int, default=20,
                    help='Stealth: min cells from endpoints (20 cells = 1 m)')
    ap.add_argument('--candidate-stride', type=int, default=2)
    ap.add_argument('--n-workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    omap = OccupancyMap(args.map)
    sw = parse_xy(args.start_world)
    gw = parse_xy(args.goal_world)
    s_cell = omap.world_to_grid(*sw)
    g_cell = omap.world_to_grid(*gw)
    print(f'Map: {args.map} ({omap.width}x{omap.height} cells, '
          f'{omap.resolution} m/cell)')
    print(f'Start: world={sw} -> cell={s_cell}')
    print(f'Goal:  world={gw} -> cell={g_cell}')

    base = astar(omap.grid, s_cell, g_cell, args.inflation_r)
    if base is None:
        print('ERROR: No baseline path. Check start/goal poses.')
        return
    base_cost = path_cost(base)
    print(f'Baseline path: {len(base)} cells = '
          f'{len(base) * omap.resolution:.2f} m')

    out = {'config': vars(args),
           'start_world': sw, 'goal_world': gw,
           'start_cell': list(s_cell), 'goal_cell': list(g_cell),
           'baseline_cost_cells': float(base_cost),
           'baseline_length_m': float(base_cost * omap.resolution),
           'placements_per_k': {}}

    for k in range(1, args.k + 1):
        print(f'\n-- Solving greedy k={k} --')
        solver = AOPPSolver(
            omap, sw, gw,
            k=k, obj_w=args.obj_cells, obj_h=args.obj_cells,
            corridor_d=10, inflation_r=args.inflation_r,
            candidate_stride=args.candidate_stride,
            min_dist_to_endpoints=args.min_dist,
            n_workers=args.n_workers,
        )
        place, gains, _ = solver.solve(verbose=False)
        delay = sum(gains)
        is_denial = delay >= 1e6
        place_cells = [(int(r), int(c)) for r, c in place]
        place_world = [list(omap.grid_to_world(r, c)) for r, c in place_cells]
        # Physical centre of each box (cell topleft + obj_cells/2)
        centre_world = []
        for r, c in place_cells:
            cr = r + args.obj_cells / 2
            cc = c + args.obj_cells / 2
            centre_world.append(list(omap.grid_to_world(int(cr), int(cc))))
        tag = 'DENIAL' if is_denial else f'{delay:.1f} cells'
        print(f'  delay={tag}')
        for i, (cell, w, c) in enumerate(zip(place_cells, place_world, centre_world)):
            print(f'  Box {i+1}: cell topleft={cell}, '
                  f'world topleft=({w[0]:.3f},{w[1]:.3f}), '
                  f'world centre=({c[0]:.3f},{c[1]:.3f})')
        out['placements_per_k'][str(k)] = {
            'placement_cells_topleft': place_cells,
            'placement_world_topleft': place_world,
            'placement_world_centre': centre_world,
            'delay_cells': float(delay),
            'is_denial': bool(is_denial),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f'\nSaved {args.out}')

    # Tape positions markdown (relative to room corners would need user
    # input; here we report map-frame coordinates)
    md = ['# Tape positions for hardware Phase 4',
          '',
          f'Map: `{args.map}` ({omap.resolution} m/cell)',
          f'Start (tape A): ({sw[0]:.3f}, {sw[1]:.3f}) m',
          f'Goal  (tape B): ({gw[0]:.3f}, {gw[1]:.3f}) m',
          '',
          'Box positions (centre, in map frame):', '']
    for k_str, info in out['placements_per_k'].items():
        md.append(f'## k={k_str}')
        for i, c in enumerate(info['placement_world_centre']):
            md.append(f'  - Box {i+1}: ({c[0]:.3f}, {c[1]:.3f}) m')
        md.append('')
    md.append('Use a tape measure and a known fiducial (room corner, doorway')
    md.append("frame) to convert these to floor-tape positions. Tolerance ±5 cm.")
    Path(args.out).parent.joinpath('tape_positions.md').write_text('\n'.join(md))
    print(f'Saved {Path(args.out).parent / "tape_positions.md"}')


if __name__ == '__main__':
    main()
