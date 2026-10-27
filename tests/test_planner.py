import math
import numpy as np

from aopp.planning import shortest_path
from aopp.greedy import greedy_attack


def test_astar_open_grid():
    occ = np.zeros((20, 20), dtype=bool)
    cost, path = shortest_path(occ, (1, 1), (18, 18))
    assert path
    assert math.isclose(cost, 17 * math.sqrt(2), rel_tol=1e-9)


def test_astar_denial_wall():
    occ = np.zeros((20, 20), dtype=bool)
    occ[:, 10] = True
    cost, path = shortest_path(occ, (5, 5), (5, 15))
    assert math.isinf(cost)
    assert path == []


def test_greedy_can_block_narrow_corridor():
    occ = np.ones((50, 50), dtype=bool)
    occ[20:30, 1:49] = False
    res = greedy_attack(
        occ, start=(25, 5), goal=(25, 44), footprint_size=4,
        inflation_radius=2, stealth_distance=6, k=1,
        candidate_stride=1, candidate_buffer_cells=1,
    )
    assert res.steps
    assert res.steps[0].denial
