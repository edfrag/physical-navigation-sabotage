from __future__ import annotations

import math
from typing import Tuple
import numpy as np

Cell = Tuple[int, int]


def footprint_cells(center: Cell, size: int) -> tuple[slice, slice] | None:
    if size <= 0:
        raise ValueError("footprint size must be positive")
    r, c = center
    lo = size // 2
    hi = size - lo
    return slice(r - lo, r + hi), slice(c - lo, c + hi)


def euclidean(a: Cell, b: Cell) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def valid_placement(
    base_occupied: np.ndarray,
    placed_occupied: np.ndarray,
    center: Cell,
    footprint_size: int,
    start: Cell,
    goal: Cell,
    stealth_distance: float,
) -> bool:
    rs, cs = footprint_cells(center, footprint_size)
    if rs.start < 0 or cs.start < 0 or rs.stop > base_occupied.shape[0] or cs.stop > base_occupied.shape[1]:
        return False
    if euclidean(center, start) < stealth_distance or euclidean(center, goal) < stealth_distance:
        return False
    if np.any(base_occupied[rs, cs]):
        return False
    if np.any(placed_occupied[rs, cs]):
        return False
    return True


def apply_placement(occupied: np.ndarray, center: Cell, footprint_size: int) -> np.ndarray:
    out = occupied.copy()
    rs, cs = footprint_cells(center, footprint_size)
    out[rs, cs] = True
    return out
