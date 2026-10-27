"""Greedy solver for the Adversarial Object Placement Problem (AOPP).

Implements Algorithm 1 from the paper: corridor-pruned greedy search
with marginal gain tracking for submodularity analysis.
"""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from .map_utils import OccupancyMap
from .planner import astar, path_cost, inflate


def _eval_candidate(args):
    """Evaluate a single candidate placement (for multiprocessing).

    Runs in a worker process with its own copy of the data.
    """
    r, c, obj_w, obj_h, grid, inflated_grid, start, goal, inflation_r = args
    # Place obstacle on raw grid
    trial = grid.copy()
    r_end = min(r + obj_h, trial.shape[0])
    c_end = min(c + obj_w, trial.shape[1])
    trial[r:r_end, c:c_end] = 100
    # Locally inflate the new obstacle on pre-inflated grid
    trial_inf = inflated_grid.copy()
    trial_inf[r:r_end, c:c_end] = 100
    rad = inflation_r
    rows, cols = trial_inf.shape
    ir0 = max(0, r - rad)
    ir1 = min(rows, r + obj_h + rad)
    ic0 = max(0, c - rad)
    ic1 = min(cols, c + obj_w + rad)
    for ir in range(ir0, ir1):
        for ic in range(ic0, ic1):
            if trial_inf[ir, ic] < 99:
                dr = max(r - ir, 0, ir - (r_end - 1))
                dc = max(c - ic, 0, ic - (c_end - 1))
                if max(dr, dc) <= rad:
                    trial_inf[ir, ic] = 99
    p = astar(trial, start, goal, inflation_r, inflated_grid=trial_inf)
    cost = path_cost(p)
    return (r, c, cost)


class AOPPSolver:
    """Greedy solver for the Adversarial Object Placement Problem.

    Given an occupancy grid, a navigation task (start, goal), and an object
    budget k, finds placements that maximise navigation delay under A*.

    Parameters
    ----------
    occ_map : OccupancyMap
        The environment map.
    start_world : tuple of float
        Start position in world coordinates (x, y) in metres.
    goal_world : tuple of float
        Goal position in world coordinates (x, y) in metres.
    k : int
        Maximum number of objects to place.
    obj_w, obj_h : int
        Object footprint in grid cells (width, height).
    corridor_d : int
        Search corridor half-width around baseline path (cells).
    inflation_r : int
        Binary-surrogate inflation radius in cells.
    min_dist_to_endpoints : int
        Minimum distance from start/goal in cells (physical constraint).
    """

    DENIAL_COST = 1e6  # cost assigned when planner fails

    def __init__(
        self,
        occ_map: OccupancyMap,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
        k: int = 3,
        obj_w: int = 3,
        obj_h: int = 3,
        corridor_d: int = 10,
        inflation_r: int = 3,
        min_dist_to_endpoints: int = 5,
        candidate_stride: int = 1,
        n_workers: int = 0,
    ):
        self.omap = occ_map
        self.start = occ_map.world_to_grid(*start_world)
        self.goal = occ_map.world_to_grid(*goal_world)
        self.k = k
        self.obj_w = obj_w
        self.obj_h = obj_h
        self.inflation_r = inflation_r
        self.n_workers = n_workers if n_workers > 0 else max(1, cpu_count() - 2)

        # Compute baseline path
        self.baseline_path = astar(
            occ_map.grid, self.start, self.goal, inflation_r
        )
        if self.baseline_path is None:
            raise ValueError(
                f"No baseline path from {self.start} to {self.goal}. "
                f"Check start/goal coordinates and map."
            )
        self.baseline_cost = path_cost(self.baseline_path)

        # Generate candidate positions: free rectangles near baseline path
        self.candidates = self._gen_candidates(
            corridor_d, min_dist_to_endpoints, candidate_stride
        )
        print(
            f"AOPPSolver initialised: baseline cost={self.baseline_cost:.1f} "
            f"cells, {len(self.candidates)} candidate positions, "
            f"{self.n_workers} workers"
        )

    def _gen_candidates(
        self, d: int, min_ep_dist: int, stride: int = 1
    ) -> List[Tuple[int, int]]:
        """Generate candidate obstacle positions near the baseline path.

        Parameters
        ----------
        stride : int
            Grid stride for candidate generation. stride=2 checks every
            2nd position, reducing candidates by ~4x. Since objects span
            multiple cells, adjacent positions produce nearly identical
            placements, so stride <= obj_size preserves solution quality.
        """
        cands = set()

        for (r, c) in self.baseline_path:
            for dr in range(-d, d + 1, stride):
                for dc in range(-d, d + 1, stride):
                    nr, nc = r + dr, c + dc
                    # Must fit a full object footprint in free space
                    if not self.omap.is_free_rect(
                        nr, nc, self.obj_w, self.obj_h
                    ):
                        continue
                    # Respect minimum distance to start and goal
                    dist_start = max(
                        abs(nr - self.start[0]), abs(nc - self.start[1])
                    )
                    dist_goal = max(
                        abs(nr - self.goal[0]), abs(nc - self.goal[1])
                    )
                    if dist_start < min_ep_dist or dist_goal < min_ep_dist:
                        continue
                    cands.add((nr, nc))

        return list(cands)

    def _inflate_obstacle_local(self, inf_grid: np.ndarray,
                                r: int, c: int, w: int, h: int) -> np.ndarray:
        """Locally inflate a newly placed obstacle onto a pre-inflated grid.

        Produces identical results to full re-inflation but only updates
        the region around the new obstacle (much faster for large grids).
        """
        result = inf_grid.copy()
        rad = self.inflation_r
        rows, cols = result.shape
        # The obstacle occupies [r:r+h, c:c+w]. Mark those as occupied.
        r_end = min(r + h, rows)
        c_end = min(c + w, cols)
        result[r:r_end, c:c_end] = 100
        # Inflate: mark cells within inflation_r of the obstacle as inscribed
        r0 = max(0, r - rad)
        r1 = min(rows, r + h + rad)
        c0 = max(0, c - rad)
        c1 = min(cols, c + w + rad)
        for ir in range(r0, r1):
            for ic in range(c0, c1):
                if result[ir, ic] < 99:
                    # Check if within inflation radius of any obstacle cell
                    # (Chebyshev distance to the rectangle)
                    dr = max(r - ir, 0, ir - (r_end - 1))
                    dc = max(c - ic, 0, ic - (c_end - 1))
                    if max(dr, dc) <= rad:
                        result[ir, ic] = 99
        return result

    def _compute_delay(self, grid: np.ndarray,
                       inflated_grid: np.ndarray = None) -> float:
        """Compute delay (cost increase) relative to baseline."""
        p = astar(grid, self.start, self.goal, self.inflation_r,
                  inflated_grid=inflated_grid)
        c = path_cost(p)
        if c == float('inf'):
            return self.DENIAL_COST
        return c - self.baseline_cost

    def solve(self, verbose: bool = True):
        """Run greedy AOPP (Algorithm 1).

        Uses multiprocessing to evaluate candidates in parallel across
        CPU cores. The greedy selection logic is unchanged — all candidates
        are evaluated at each step, and the best marginal gain is chosen.

        Returns
        -------
        placements : list of (row, col)
            Greedy-optimal object positions.
        marginal_gains : list of float
            Marginal gain at each step (for submodularity analysis).
        final_grid : np.ndarray
            Map with all objects placed.
        """
        grid = self.omap.grid.copy()
        base_inflated = inflate(grid, self.inflation_r)
        placements = []
        gains = []
        current_delay = 0.0

        for i in range(self.k):
            best_gain = -np.inf
            best_pos = None

            # Filter candidates still valid on current grid
            active = [
                c for c in self.candidates
                if self.omap.is_free_rect(
                    c[0], c[1], self.obj_w, self.obj_h, grid
                )
            ]

            if verbose:
                print(f"  Object {i+1}/{self.k}: evaluating {len(active)} candidates "
                      f"with {self.n_workers} workers...")

            # Build argument list for parallel evaluation
            args_list = [
                (r, c, self.obj_w, self.obj_h, grid, base_inflated,
                 self.start, self.goal, self.inflation_r)
                for (r, c) in active
            ]

            # Evaluate all candidates in parallel
            with Pool(processes=self.n_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_eval_candidate, args_list,
                                        chunksize=20),
                    total=len(args_list),
                    desc=f"Object {i+1}/{self.k}",
                    disable=not verbose,
                ))

            # Find best placement from results
            for (r, c, cost) in results:
                if cost == float('inf'):
                    delay = self.DENIAL_COST
                else:
                    delay = cost - self.baseline_cost
                mg = delay - current_delay
                if mg > best_gain:
                    best_gain = mg
                    best_pos = (r, c)

            if best_pos is None:
                print(f"  No valid placement found at step {i+1}")
                break

            grid = self.omap.place_obstacle(
                best_pos[0], best_pos[1], self.obj_w, self.obj_h, grid
            )
            base_inflated = self._inflate_obstacle_local(
                base_inflated, best_pos[0], best_pos[1],
                self.obj_w, self.obj_h
            )
            current_delay += best_gain
            placements.append(best_pos)
            gains.append(best_gain)

            world_pos = self.omap.grid_to_world(*best_pos)
            if verbose:
                print(
                    f"  Placed object {i+1} at grid=({best_pos[0]},{best_pos[1]}) "
                    f"world=({world_pos[0]:.2f},{world_pos[1]:.2f}) "
                    f"marginal_gain={best_gain:.1f} total_delay={current_delay:.1f}"
                )

        return placements, gains, grid

    def solve_multigoal(
        self,
        goals_world: List[Tuple[float, float]],
        verbose: bool = True,
    ):
        """Greedy AOPP with goal uncertainty (Section III extension).

        Optimises average delay across multiple possible goals.

        Parameters
        ----------
        goals_world : list of (x, y)
            Sampled goal positions in world coordinates.

        Returns
        -------
        placements, marginal_gains, final_grid
        """
        goals_grid = [self.omap.world_to_grid(*g) for g in goals_world]

        # Compute baseline cost per goal
        baseline_costs = {}
        for g in goals_grid:
            p = astar(self.omap.grid, self.start, g, self.inflation_r)
            baseline_costs[g] = path_cost(p)

        grid = self.omap.grid.copy()
        placements = []
        gains = []
        current_avg_delay = 0.0

        for i in range(self.k):
            best_gain = -np.inf
            best_pos = None

            active = [
                c for c in self.candidates
                if self.omap.is_free_rect(
                    c[0], c[1], self.obj_w, self.obj_h, grid
                )
            ]

            iterator = tqdm(
                active, desc=f"MultiGoal Obj {i+1}/{self.k}",
                disable=not verbose,
            )
            for (r, c) in iterator:
                trial = self.omap.place_obstacle(
                    r, c, self.obj_w, self.obj_h, grid
                )
                avg_delay = 0.0
                for g in goals_grid:
                    p = astar(trial, self.start, g, self.inflation_r)
                    cost = path_cost(p)
                    delay = min(cost - baseline_costs[g], self.DENIAL_COST)
                    avg_delay += delay
                avg_delay /= len(goals_grid)

                mg = avg_delay - current_avg_delay
                if mg > best_gain:
                    best_gain = mg
                    best_pos = (r, c)

            if best_pos is None:
                break

            grid = self.omap.place_obstacle(
                best_pos[0], best_pos[1], self.obj_w, self.obj_h, grid
            )
            current_avg_delay += best_gain
            placements.append(best_pos)
            gains.append(best_gain)

        return placements, gains, grid

    # ---- Baselines ----

    def random_placement(self, seed: int = 42):
        """Baseline: place k objects at random valid positions."""
        rng = np.random.RandomState(seed)
        grid = self.omap.grid.copy()
        placed = []

        for _ in range(self.k):
            valid = [
                c for c in self.candidates
                if self.omap.is_free_rect(
                    c[0], c[1], self.obj_w, self.obj_h, grid
                )
            ]
            if not valid:
                break
            pos = valid[rng.randint(len(valid))]
            grid = self.omap.place_obstacle(
                pos[0], pos[1], self.obj_w, self.obj_h, grid
            )
            placed.append(pos)

        delay = self._compute_delay(grid)
        return placed, delay, grid

    def pathblock_placement(self):
        """Baseline: place objects evenly spaced along the baseline path."""
        grid = self.omap.grid.copy()
        placed = []
        step = max(1, len(self.baseline_path) // (self.k + 1))

        for i in range(1, self.k + 1):
            idx = min(i * step, len(self.baseline_path) - 1)
            r, c = self.baseline_path[idx]
            if self.omap.is_free_rect(r, c, self.obj_w, self.obj_h, grid):
                grid = self.omap.place_obstacle(
                    r, c, self.obj_w, self.obj_h, grid
                )
                placed.append((r, c))

        delay = self._compute_delay(grid)
        return placed, delay, grid
