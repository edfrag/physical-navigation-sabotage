from __future__ import annotations

import heapq
import math
from typing import Dict, Tuple
import numpy as np

Cell = Tuple[int, int]

_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
]


def _heuristic(a: Cell, b: Cell) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2.0) - 1.0) * min(dr, dc)


def shortest_path(blocked: np.ndarray, start: Cell, goal: Cell) -> tuple[float, list[Cell]]:
    h, w = blocked.shape
    for p in (start, goal):
        if not (0 <= p[0] < h and 0 <= p[1] < w):
            return math.inf, []
        if blocked[p]:
            return math.inf, []

    g: Dict[Cell, float] = {start: 0.0}
    parent: Dict[Cell, Cell] = {}
    pq: list[tuple[float, float, Cell]] = [(_heuristic(start, goal), 0.0, start)]
    closed: set[Cell] = set()

    while pq:
        _, gcost, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return gcost, path
        closed.add(cur)

        r, c = cur
        for dr, dc, step in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w) or blocked[nr, nc]:
                continue
            if dr != 0 and dc != 0:
                if blocked[r + dr, c] or blocked[r, c + dc]:
                    continue
            ng = gcost + step
            if ng < g.get(nxt, math.inf):
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(pq, (ng + _heuristic(nxt, goal), ng, nxt))

    return math.inf, []
