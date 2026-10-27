from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.ndimage import binary_dilation

from .inflation import disk, inflate
from .objective import cost_for_placements
from .placement import apply_placement, valid_placement
from .planning import shortest_path

Cell = Tuple[int, int]


@dataclass
class GreedyStep:
    iteration: int
    placement: Cell
    total_cost: float
    total_delay: float
    marginal_gain: float
    denial: bool
    candidates_evaluated: int


@dataclass
class GreedyResult:
    baseline_cost: float
    placements: list[Cell]
    steps: list[GreedyStep]
    final_cost: float
    final_path: list[Cell]


def _path_margin(shape: tuple[int, int], path: list[Cell], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not path:
        return mask
    rr, cc = zip(*path)
    mask[np.asarray(rr), np.asarray(cc)] = True
    if radius > 0:
        mask = binary_dilation(mask, structure=disk(radius))
    return mask


def _candidate_centers(
    base_occupied: np.ndarray,
    placed_occupied: np.ndarray,
    current_path: list[Cell],
    footprint_size: int,
    inflation_radius: int,
    buffer_cells: int,
    stride: int,
    start: Cell,
    goal: Cell,
    stealth_distance: float,
) -> list[Cell]:
    margin = _path_margin(base_occupied.shape, current_path, inflation_radius + buffer_cells)
    coords = np.argwhere(margin)
    result: list[Cell] = []
    stride = max(1, stride)
    for r, c in coords:
        p = (int(r), int(c))
        if stride > 1 and ((r % stride) != 0 or (c % stride) != 0):
            continue
        if valid_placement(base_occupied, placed_occupied, p, footprint_size, start, goal, stealth_distance):
            result.append(p)
    return result


def _score_candidate(
    base_occupied: np.ndarray,
    placements: list[Cell],
    p: Cell,
    footprint_size: int,
    inflation_radius: int,
    start: Cell,
    goal: Cell,
) -> float:
    cost, _ = cost_for_placements(base_occupied, placements + [p], footprint_size, inflation_radius, start, goal)
    return cost


def greedy_attack(
    base_occupied: np.ndarray,
    start: Cell,
    goal: Cell,
    footprint_size: int,
    inflation_radius: int,
    stealth_distance: float,
    k: int,
    candidate_stride: int = 1,
    candidate_buffer_cells: int = 2,
) -> GreedyResult:
    baseline_blocked = inflate(base_occupied, inflation_radius)
    baseline_cost, baseline_path = shortest_path(baseline_blocked, start, goal)
    if math.isinf(baseline_cost):
        raise ValueError("baseline route is infeasible")

    placements: list[Cell] = []
    placed_occupied = np.zeros_like(base_occupied, dtype=bool)
    current_cost = baseline_cost
    current_path = baseline_path
    steps: list[GreedyStep] = []

    for i in range(1, k + 1):
        candidates = _candidate_centers(
            base_occupied, placed_occupied, current_path,
            footprint_size, inflation_radius, candidate_buffer_cells,
            candidate_stride, start, goal, stealth_distance,
        )
        if not candidates:
            break

        best = None
        best_cost = -math.inf
        evaluated = 0
        for p in candidates:
            cost = _score_candidate(base_occupied, placements, p, footprint_size, inflation_radius, start, goal)
            evaluated += 1
            if math.isinf(cost):
                best, best_cost = p, cost
                break
            if cost > best_cost:
                best, best_cost = p, cost

        if best is not None and candidate_stride > 1 and not math.isinf(best_cost):
            br, bc = best
            rad = candidate_stride - 1
            for r in range(br - rad, br + rad + 1):
                for c in range(bc - rad, bc + rad + 1):
                    p = (r, c)
                    if not valid_placement(base_occupied, placed_occupied, p, footprint_size, start, goal, stealth_distance):
                        continue
                    cost = _score_candidate(base_occupied, placements, p, footprint_size, inflation_radius, start, goal)
                    evaluated += 1
                    if math.isinf(cost):
                        best, best_cost = p, cost
                        break
                    if cost > best_cost:
                        best, best_cost = p, cost
                if math.isinf(best_cost):
                    break

        if best is None:
            break

        placements.append(best)
        placed_occupied = apply_placement(placed_occupied, best, footprint_size)
        new_cost, new_path = cost_for_placements(base_occupied, placements, footprint_size, inflation_radius, start, goal)
        marginal = math.inf if math.isinf(new_cost) else new_cost - current_cost
        total_delay = math.inf if math.isinf(new_cost) else new_cost - baseline_cost
        steps.append(GreedyStep(
            iteration=i,
            placement=best,
            total_cost=new_cost,
            total_delay=total_delay,
            marginal_gain=marginal,
            denial=math.isinf(new_cost),
            candidates_evaluated=evaluated,
        ))
        current_cost = new_cost
        current_path = new_path
        if math.isinf(new_cost):
            break

    return GreedyResult(
        baseline_cost=baseline_cost,
        placements=placements,
        steps=steps,
        final_cost=current_cost,
        final_path=current_path,
    )
