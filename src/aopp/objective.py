from __future__ import annotations

import math
from typing import Iterable, Tuple
import numpy as np

from .inflation import inflate
from .placement import apply_placement
from .planning import shortest_path

Cell = Tuple[int, int]


def cost_for_placements(
    base_occupied: np.ndarray,
    placements: Iterable[Cell],
    footprint_size: int,
    inflation_radius: int,
    start: Cell,
    goal: Cell,
) -> tuple[float, list[Cell]]:
    occ = base_occupied.copy()
    for p in placements:
        occ = apply_placement(occ, p, footprint_size)
    blocked = inflate(occ, inflation_radius)
    return shortest_path(blocked, start, goal)


def delay(baseline_cost: float, attacked_cost: float) -> float:
    if math.isinf(attacked_cost):
        return math.inf
    return attacked_cost - baseline_cost
