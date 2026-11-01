"""Empirical submodularity diagnostics; no approximation guarantee is inferred."""

import numpy as np
import itertools
from typing import List, Tuple, Optional
from .map_utils import OccupancyMap
from .planner import astar, path_cost


def compute_submodularity_ratio(
    omap: OccupancyMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    placements: List[Tuple[int, int]],
    candidates_sample: List[Tuple[int, int]],
    inflation_r: int = 3,
    obj_w: int = 3,
    obj_h: int = 3,
    max_subset_size: int = 4,
    max_test_elements: int = 15,
) -> Tuple[float, int, List[dict]]:
    """Estimate the submodularity ratio gamma empirically.

    Tests the diminishing returns property across subset pairs
    (A subset B) and singleton elements e not in B.

    Parameters
    ----------
    omap : OccupancyMap
    start, goal : tuple
        Grid coordinates.
    placements : list of tuple
        Greedy solution placements to form subsets from.
    candidates_sample : list of tuple
        Additional candidate positions to test as singletons.
    inflation_r, obj_w, obj_h : int
        Planner and object parameters.
    max_subset_size : int
        Maximum subset size to enumerate (controls computation).
    max_test_elements : int
        Maximum singleton elements to test per subset pair.

    Returns
    -------
    gamma : float
        Minimum ratio of marginal gains. >= 1 means submodular.
    n_comparisons : int
        Number of valid comparisons made.
    details : list of dict
        Per-comparison records for analysis.
    """
    baseline_cost = path_cost(astar(omap.grid, start, goal, inflation_r))

    def delay_with_set(S):
        g = omap.grid.copy()
        for (r, c) in S:
            g = omap.place_obstacle(r, c, obj_w, obj_h, g)
        p = astar(g, start, goal, inflation_r)
        c = path_cost(p)
        if c == float('inf'):
            return 1e6
        return c - baseline_cost

    gamma = float('inf')
    n_comparisons = 0
    details = []

    # Build the pool of elements to test
    test_elements = list(set(candidates_sample[:max_test_elements]))

    # Enumerate subset pairs A subset B from the greedy solution
    n_place = min(len(placements), max_subset_size)
    for size_a in range(n_place):
        for A_tuple in itertools.combinations(placements[:n_place], size_a):
            A = set(A_tuple)
            for size_b in range(size_a + 1, n_place + 1):
                for B_tuple in itertools.combinations(
                    placements[:n_place], size_b
                ):
                    B = set(B_tuple)
                    if not A.issubset(B):
                        continue

                    for e in test_elements:
                        if e in B:
                            continue

                        gain_A = (
                            delay_with_set(A | {e}) - delay_with_set(A)
                        )
                        gain_B = (
                            delay_with_set(B | {e}) - delay_with_set(B)
                        )

                        if gain_B > 1e-6:
                            ratio = gain_A / gain_B
                            gamma = min(gamma, ratio)
                            n_comparisons += 1
                            details.append({
                                '|A|': len(A),
                                '|B|': len(B),
                                'gain_A': gain_A,
                                'gain_B': gain_B,
                                'ratio': ratio,
                            })

    if n_comparisons == 0:
        gamma = float('nan')

    return gamma, n_comparisons, details


def marginal_gain_analysis(gains: List[float]) -> dict:
    """Analyse the marginal gains sequence from greedy.

    Returns
    -------
    dict with:
        - monotone_decreasing: bool (True if gains are non-increasing)
        - max_gain, min_gain, total_delay
        - gain_ratios: consecutive ratios g[i+1]/g[i]
    """
    if not gains:
        return {'monotone_decreasing': True, 'gains': []}

    ratios = []
    for i in range(1, len(gains)):
        if gains[i-1] > 1e-6:
            ratios.append(gains[i] / gains[i-1])

    return {
        'monotone_decreasing': all(
            gains[i] <= gains[i-1] + 1e-6 for i in range(1, len(gains))
        ),
        'max_gain': max(gains),
        'min_gain': min(gains),
        'total_delay': sum(gains),
        'gains': gains,
        'gain_ratios': ratios,
    }
