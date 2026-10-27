"""Exercise the recovered solver on a temporary synthetic corridor."""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aopp_solver import AOPPSolver
from src.map_utils import OccupancyMap
from src.planner import astar, inflate, path_cost


def main():
    with tempfile.TemporaryDirectory(prefix='aopp-smoke-') as directory:
        root = Path(directory)
        pixels = np.zeros((24, 40), dtype=np.uint8)
        pixels[8:16, 1:39] = 254
        Image.fromarray(pixels).save(root / 'map.pgm')
        (root / 'map.yaml').write_text(yaml.safe_dump({
            'image': 'map.pgm', 'resolution': 0.05, 'origin': [0, 0, 0],
            'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196,
        }))
        omap = OccupancyMap(root / 'map.yaml')
        start, goal = (11, 3), (11, 36)
        assert omap.world_to_grid(*omap.grid_to_world(*start)) == start
        assert path_cost(astar(omap.grid, start, goal, inflation_radius=1)) == 33
        solver = AOPPSolver(
            omap, omap.grid_to_world(*start), omap.grid_to_world(*goal),
            k=1, obj_w=6, obj_h=6, corridor_d=3, inflation_r=1,
            min_dist_to_endpoints=6, candidate_stride=3, n_workers=1,
        )
        placements, gains, modified = solver.solve(verbose=False)
        assert len(placements) == 1 and gains == [solver.DENIAL_COST]
        assert astar(modified, start, goal, inflation_radius=1) is None
        r, c = placements[0]
        local = solver._inflate_obstacle_local(inflate(omap.grid, 1), r, c, 6, 6)
        assert np.array_equal(local, inflate(modified, 1))
        assert solver.random_placement(seed=0)[:2] == solver.random_placement(seed=0)[:2]
    print('Map loading, baseline planning, greedy denial, inflation, and seeded random checks passed.')


if __name__ == '__main__':
    main()
