from __future__ import annotations

import math
from typing import Tuple
import numpy as np

from .inflation import inflate
from .objective import cost_for_placements
from .placement import apply_placement, valid_placement
from .planning import shortest_path

Cell = Tuple[int, int]


def random_attack(
    base_occupied: np.ndarray,
    start: Cell,
    goal: Cell,
    footprint_size: int,
    inflation_radius: int,
    stealth_distance: float,
    k: int,
    seed: int = 0,
) -> tuple[list[Cell], float]:
    rng = np.random.RandomState(seed)
    placed = np.zeros_like(base_occupied, dtype=bool)
    placements: list[Cell] = []
    h, w = base_occupied.shape
    pool = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(pool)
    for p in pool:
        if valid_placement(base_occupied, placed, p, footprint_size, start, goal, stealth_distance):
            placements.append(p)
            placed = apply_placement(placed, p, footprint_size)
            if len(placements) == k:
                break
    cost, _ = cost_for_placements(base_occupied, placements, footprint_size, inflation_radius, start, goal)
    return placements, cost


def pathblock_attack(
    base_occupied: np.ndarray,
    start: Cell,
    goal: Cell,
    footprint_size: int,
    inflation_radius: int,
    stealth_distance: float,
    k: int,
) -> tuple[list[Cell], float]:
    baseline_cost, path = shortest_path(inflate(base_occupied, inflation_radius), start, goal)
    if math.isinf(baseline_cost):
        raise ValueError("baseline route is infeasible")
    placed = np.zeros_like(base_occupied, dtype=bool)
    placements: list[Cell] = []
    if not path or k <= 0:
        return placements, baseline_cost
    idx = np.linspace(0, len(path) - 1, num=k + 2, dtype=int)[1:-1]
    for i in idx:
        for j in sorted(range(len(path)), key=lambda j: abs(j - i)):
            p = path[j]
            if valid_placement(base_occupied, placed, p, footprint_size, start, goal, stealth_distance):
                placements.append(p)
                placed = apply_placement(placed, p, footprint_size)
                break
    cost, _ = cost_for_placements(base_occupied, placements, footprint_size, inflation_radius, start, goal)
    return placements, cost
