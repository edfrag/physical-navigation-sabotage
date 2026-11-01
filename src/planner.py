"""A* planner on a binary-inflation surrogate grid."""

import numpy as np
import heapq
from typing import List, Optional, Tuple
from scipy.ndimage import binary_dilation


def inflate(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """Inflate obstacles by a given radius in cells.

    Uses square binary dilation, without Nav2's graded costs.

    Parameters
    ----------
    grid : np.ndarray
        Occupancy grid (0 = free, 100 = occupied, -1 = unknown).
    radius_cells : int
        Inflation radius in grid cells.

    Returns
    -------
    np.ndarray
        Inflated grid with inscribed costs set to 99.
    """
    occ_mask = grid >= 100
    struct = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1))
    inflated = binary_dilation(occ_mask, structure=struct)
    result = grid.copy()
    # Mark newly inflated cells as inscribed (cost 99)
    result[inflated & (~occ_mask)] = 99
    return result


def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
          inflation_radius: int = 3,
          inflated_grid: Optional[np.ndarray] = None) -> Optional[List[Tuple[int, int]]]:
    """A* search on an inflated occupancy grid.

    Uses 8-connected movement with Euclidean heuristic.

    Parameters
    ----------
    grid : np.ndarray
        Raw occupancy grid.
    start : tuple of int
        (row, col) start position.
    goal : tuple of int
        (row, col) goal position.
    inflation_radius : int
        Costmap inflation radius in cells.
    inflated_grid : np.ndarray, optional
        Pre-computed inflated grid to skip the inflation step.
        When provided, the grid parameter is ignored for inflation.

    Returns
    -------
    list of (row, col) or None
        Path from start to goal, or None if unreachable.
    """
    inf_grid = inflated_grid if inflated_grid is not None else inflate(grid, inflation_radius)
    rows, cols = inf_grid.shape

    # Ensure start and goal are traversable
    if inf_grid[start[0], start[1]] >= 99:
        return None
    if inf_grid[goal[0], goal[1]] >= 99:
        return None

    # 8-connected neighbourhood
    MOVES = [
        (-1,  0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
    ]

    open_set = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    def heuristic(a):
        return ((a[0] - goal[0]) ** 2 + (a[1] - goal[1]) ** 2) ** 0.5

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc, cost in MOVES:
            nb = (current[0] + dr, current[1] + dc)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            if inf_grid[nb[0], nb[1]] >= 99:
                continue
            if nb in visited:
                continue

            tent_g = g_score[current] + cost
            if tent_g < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb] = tent_g
                heapq.heappush(open_set, (tent_g + heuristic(nb), nb))

    return None


def path_cost(path: Optional[List[Tuple[int, int]]]) -> float:
    """Compute Euclidean path length in grid cells.

    Returns inf if path is None (no path found = navigation denial).
    """
    if path is None:
        return float('inf')
    cost = 0.0
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i-1][0])
        dc = abs(path[i][1] - path[i-1][1])
        cost += 1.414 if (dr + dc) == 2 else 1.0
    return cost
