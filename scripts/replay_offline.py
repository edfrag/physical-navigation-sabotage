"""Re-evaluate saved placements without changing archived results."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.map_utils import OccupancyMap
from src.planner import astar, inflate, path_cost

CONFIG = {
    'maze': ((-7.535, -7.855), (6.775, 7.085), 10),
    'hospital': ((1.525, 12.125), (53.525, 12.125), 6),
    'warehouse': ((-7.35, -12.05), (8.28, 12.84), 10),
    'depot': ((-9.12, -3.72), (13.28, 3.80), 8),
}


def snap(omap, point, inflated):
    if inflated[point] < 99:
        return point
    for distance in range(1, 100):
        for dr in range(-distance, distance + 1):
            for dc in range(-distance, distance + 1):
                r, c = point[0] + dr, point[1] + dc
                if 0 <= r < omap.height and 0 <= c < omap.width and inflated[r, c] < 99:
                    return r, c
    raise ValueError('No valid endpoint')


def main():
    count = 0
    for name in ['offline_results_inflation6.json',
                 'additional_experiments/open_envs/offline_open_envs_inflation6.json']:
        data = json.loads((ROOT / 'results' / name).read_text())
        for record in data['results']:
            env = record['env']
            start, goal, footprint = CONFIG[env]
            omap = OccupancyMap(ROOT / f'maps/{env}.yaml')
            inf = inflate(omap.grid, 6)
            start = snap(omap, omap.world_to_grid(*start), inf)
            goal = snap(omap, omap.world_to_grid(*goal), inf)
            baseline = path_cost(astar(omap.grid, start, goal, inflated_grid=inf))
            grid = omap.grid.copy()
            for r, c in record['greedy_placements']:
                if not omap.is_free_rect(r, c, footprint, footprint, grid):
                    raise ValueError(f'Invalid saved placement: {env}, k={record["k"]}')
                grid = omap.place_obstacle(r, c, footprint, footprint, grid)
            cost = path_cost(astar(grid, start, goal, inflation_radius=6))
            delay = 1e6 if math.isinf(cost) else cost - baseline
            if not (math.isclose(baseline, record['baseline_cost'], abs_tol=1e-7)
                    and math.isclose(delay, record['greedy_delay'], abs_tol=1e-7)):
                raise ValueError(f'Replay differs: {env}, k={record["k"]}')
            if math.isinf(cost) != record['is_denial']:
                raise ValueError(f'Denial label differs: {env}, k={record["k"]}')
            count += 1
            print(f'{env} k={record["k"]}: matches archived cost and denial', flush=True)
    print(f'{count} saved placement cases verified.')


if __name__ == '__main__':
    main()
