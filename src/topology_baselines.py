"""Topology-aware AOPP baselines: min-cut, chokepoint, edge betweenness.

All three baselines satisfy the same constraints as greedy AOPP:
  - budget k
  - stealth distance d_min from start and goal (Chebyshev cells)
  - object footprint (obj_w x obj_h) matched to grid resolution
  - footprint must lie in free-space (under inflation_r) before placement
  - placements may not overlap each other (footprint regions disjoint)

Each baseline returns: (placements list of (r, c) topleft cells, delay,
is_denial, baseline_cost).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from .map_utils import OccupancyMap
from .planner import astar, inflate, path_cost


# -----------------------------------------------------------------------
# Placement selection helper

def _stamp(grid: np.ndarray, r: int, c: int, w: int, h: int) -> np.ndarray:
    g = grid.copy()
    r0 = max(0, r); r1 = min(g.shape[0], r + h)
    c0 = max(0, c); c1 = min(g.shape[1], c + w)
    g[r0:r1, c0:c1] = 100
    return g


def _chebyshev_to_rect(p: Tuple[int, int], r: int, c: int,
                        h: int, w: int) -> int:
    """Chebyshev distance from point p to axis-aligned rect [r, r+h) x
    [c, c+w). Returns 0 if p is inside the rect."""
    dr = max(r - p[0], 0, p[0] - (r + h - 1))
    dc = max(c - p[1], 0, p[1] - (c + w - 1))
    return max(dr, dc)


def _is_valid_placement(grid: np.ndarray, r: int, c: int, w: int, h: int,
                        inflation_r: int, start: Tuple[int, int],
                        goal: Tuple[int, int], min_dist: int,
                        forbidden: np.ndarray) -> bool:
    """Check (r, c) topleft footprint is valid: in-bounds, in free space,
    far enough from endpoints, not overlapping `forbidden` (footprint mask
    of prior placements)."""
    H, W = grid.shape
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    # Free in unobstructed footprint (reject if any cell is occupied;
    # caller passes the inflated grid, so this also rejects inflation halos)
    sub = grid[r:r+h, c:c+w]
    if np.any(sub > 50):
        return False
    # Stealth: Chebyshev distance from footprint to each endpoint >= min_dist
    if (_chebyshev_to_rect(start, r, c, h, w) < min_dist
            or _chebyshev_to_rect(goal, r, c, h, w) < min_dist):
        return False
    # No overlap with prior placements + halo
    if forbidden[r:r+h, c:c+w].any():
        return False
    return True


def _select_top_k(score_grid: np.ndarray, omap: OccupancyMap,
                   start: Tuple[int, int], goal: Tuple[int, int],
                   k: int, obj_w: int, obj_h: int, inflation_r: int,
                   min_dist: int,
                   inf_grid: Optional[np.ndarray] = None,
                   forbidden: Optional[np.ndarray] = None
                   ) -> List[Tuple[int, int]]:
    """Greedily pick top-k (r, c) topleft positions with the highest summed
    score in the obj footprint, subject to constraints. After each pick,
    add the placed footprint + obj-sized halo to the forbidden mask.

    inf_grid / forbidden may be passed in for callers that need to maintain
    these across outer-loop iterations (e.g. chokepoint with re-planning).
    """
    if inf_grid is None:
        inf_grid = inflate(omap.grid, inflation_r)
    H, W = inf_grid.shape
    if forbidden is None:
        forbidden = np.zeros((H, W), dtype=bool)
    placements: List[Tuple[int, int]] = []
    score = score_grid.astype(np.float64)

    for _ in range(k):
        # Footprint-sum at every topleft via 2D integral image (fast)
        cs = np.cumsum(np.cumsum(score, axis=0), axis=1)
        # Pad with zero row/col on the top-left for the standard formula
        cs_p = np.zeros((H + 1, W + 1), dtype=np.float64)
        cs_p[1:, 1:] = cs
        fp_sum = (cs_p[obj_h:, obj_w:] - cs_p[:-obj_h, obj_w:]
                  - cs_p[obj_h:, :-obj_w] + cs_p[:-obj_h, :-obj_w])
        # fp_sum[r, c] = sum of score in window [r:r+h, c:c+w]; shape
        # (H-h+1, W-w+1) over valid topleft positions.

        # Iterate non-zero topleft positions in descending score order,
        # take the first that passes constraint checks.
        nz = np.argwhere(fp_sum > 0)
        if len(nz) == 0:
            break
        scores = fp_sum[nz[:, 0], nz[:, 1]]
        order = np.argsort(-scores)
        best_pos: Optional[Tuple[int, int]] = None
        for idx in order:
            r0, c0 = int(nz[idx, 0]), int(nz[idx, 1])
            if not _is_valid_placement(inf_grid, r0, c0, obj_w, obj_h,
                                       inflation_r, start, goal, min_dist,
                                       forbidden):
                continue
            best_pos = (r0, c0)
            break
        if best_pos is None:
            break
        placements.append(best_pos)
        # Persist no-overlap halo: forbid the placed footprint plus an
        # obj-sized buffer in each direction so future picks can't overlap
        rr, cc = best_pos
        r0 = max(0, rr - obj_h); r1 = min(H, rr + 2 * obj_h)
        c0 = max(0, cc - obj_w); c1 = min(W, cc + 2 * obj_w)
        forbidden[r0:r1, c0:c1] = True
        # Zero score in this region so we don't re-rank the same footprint
        score[r0:r1, c0:c1] = 0
    return placements


# -----------------------------------------------------------------------
# Baseline 1: Min-cut placement

def compute_min_cut_score(omap: OccupancyMap, start: Tuple[int, int],
                           goal: Tuple[int, int], inflation_r: int,
                           min_dist: int = 0,
                           max_free_cells: int = 600_000
                           ) -> Tuple[np.ndarray, int]:
    """Return (score_grid, cut_size) where score_grid[r, c] = 1 iff cell
    (r, c) is in the minimum vertex s-t cut of the inflated free-space
    graph, restricted so that the cut lies *outside* the (Chebyshev)
    stealth radius `min_dist` from both endpoints.

    We build a node-split max-flow problem:
      - each free cell c becomes (c, 'i') -> (c, 'o') with capacity:
          INF  if Chebyshev(c, start) < min_dist or Chebyshev(c, goal) < min_dist
          1    otherwise
      - 4-connected edges between cells get capacity INF
      - min cut between (start, 'o') and (goal, 'i') consists of unit-cap
        node-split edges; the cells whose split edge is in the cut form the
        vertex cut, and they all lie outside the stealth zones.
    """
    import networkx as nx

    inf_grid = inflate(omap.grid, inflation_r)
    H, W = inf_grid.shape
    free = inf_grid < 99
    if not free[start] or not free[goal]:
        return np.zeros((H, W), dtype=np.int32), 0

    # Downsample if free space exceeds max_free_cells. Coarse cell becomes
    # free iff ALL constituent fine cells are free (conservative).
    n_free = int(free.sum())
    stride = 1
    if n_free > max_free_cells:
        # Pick stride that brings free count down
        import math
        stride = int(math.ceil(math.sqrt(n_free / max_free_cells)))
        # Coarse free: an SxS block is free iff all fine cells in it are free
        new_H = H // stride
        new_W = W // stride
        free_coarse = np.ones((new_H, new_W), dtype=bool)
        for dr in range(stride):
            for dc in range(stride):
                free_coarse &= free[dr:dr + new_H * stride:stride,
                                    dc:dc + new_W * stride:stride][:new_H, :new_W]
        free = free_coarse
        # Re-map endpoints to coarse coords
        start = (start[0] // stride, start[1] // stride)
        goal = (goal[0] // stride, goal[1] // stride)
        # Re-map min_dist (Chebyshev coord scale)
        min_dist = max(1, min_dist // stride)
        H, W = free.shape
        if not free[start[0], start[1]] or not free[goal[0], goal[1]]:
            return np.zeros((omap.grid.shape[0], omap.grid.shape[1]),
                            dtype=np.int32), 0

    INF = 10 ** 9
    G = nx.DiGraph()
    free_set = set()
    for r, c in zip(*np.where(free)):
        r, c = int(r), int(c)
        free_set.add((r, c))
    # Node-split with stealth-aware capacities
    for (r, c) in free_set:
        ds = max(abs(r - start[0]), abs(c - start[1]))
        dg = max(abs(r - goal[0]), abs(c - goal[1]))
        cap = INF if (ds < min_dist or dg < min_dist) else 1
        G.add_edge(((r, c), 'i'), ((r, c), 'o'), capacity=cap)
    # 4-connected edges between cells (both directions)
    for (r, c) in free_set:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr = (r + dr, c + dc)
            if nbr in free_set:
                G.add_edge(((r, c), 'o'), (nbr, 'i'), capacity=INF)
    # Min cut between (start, 'o') and (goal, 'i')
    s_node = (start, 'o')
    t_node = (goal, 'i')
    if s_node not in G or t_node not in G:
        return np.zeros((H, W), dtype=np.int32), 0
    try:
        cut_value, (s_side, t_side) = nx.minimum_cut(
            G, s_node, t_node, capacity='capacity')
    except Exception as e:
        return np.zeros((H, W), dtype=np.int32), 0
    if cut_value >= INF:
        # No stealth-compatible cut possible (every path crosses a
        # protected cell that can't be cut)
        return np.zeros(omap.grid.shape, dtype=np.int32), 0
    # Cut cells = those with (cell, 'i') in s_side and (cell, 'o') in t_side
    cut_cells_coarse = []
    for node in s_side:
        if isinstance(node, tuple) and len(node) == 2 and node[1] == 'i':
            cell = node[0]
            if ((cell, 'o')) in t_side:
                cut_cells_coarse.append(cell)
    # Map coarse cut cells back to original resolution as a stride x stride
    # block stamped in the score grid
    score_grid = np.zeros(omap.grid.shape, dtype=np.int32)
    for (r, c) in cut_cells_coarse:
        r0 = r * stride; r1 = min(r0 + stride, score_grid.shape[0])
        c0 = c * stride; c1 = min(c0 + stride, score_grid.shape[1])
        score_grid[r0:r1, c0:c1] = 1
    return score_grid, len(cut_cells_coarse)


def baseline_min_cut(omap: OccupancyMap, start: Tuple[int, int],
                     goal: Tuple[int, int], k: int, obj_w: int, obj_h: int,
                     inflation_r: int, min_dist: int,
                     score_cache: Optional[np.ndarray] = None
                     ) -> Tuple[List[Tuple[int, int]], int, np.ndarray]:
    """Min vertex cut placement. Returns (placements, cut_size, score_grid).

    The cut cells often sit in corridors too narrow for the (obj_w x obj_h)
    footprint to be centred on them. We dilate the score by inflation_r so
    any placement whose inflation halo covers a cut cell scores positive,
    matching the brief's "the placement that severs this adjacency is
    centered on one of the two cells" intent.
    """
    from scipy.ndimage import binary_dilation
    if score_cache is not None:
        score = score_cache
        cut_size = int((score > 0).sum())
    else:
        score, cut_size = compute_min_cut_score(omap, start, goal,
                                                 inflation_r, min_dist=min_dist)
    if cut_size == 0:
        return [], cut_size, score
    # Dilate cut cells by inflation_r so nearby placements (whose halos
    # cover the cut) become valid candidates.
    score_dilated = binary_dilation(score > 0,
                                     iterations=int(inflation_r)).astype(np.float64)
    placements = _select_top_k(score_dilated, omap, start, goal,
                                k, obj_w, obj_h, inflation_r, min_dist)
    return placements, cut_size, score


# -----------------------------------------------------------------------
# Baseline 2: Chokepoint clearance

def baseline_chokepoint(omap: OccupancyMap, start: Tuple[int, int],
                        goal: Tuple[int, int], k: int, obj_w: int, obj_h: int,
                        inflation_r: int, min_dist: int
                        ) -> List[Tuple[int, int]]:
    """Place at the narrowest cells along the (re-computed) baseline path.
    Recompute the shortest path after each placement so subsequent picks
    can target chokepoints on the detoured path.
    """
    inf_grid = inflate(omap.grid, inflation_r)
    free = (inf_grid < 99).astype(np.uint8)
    base_clearance = distance_transform_edt(free)  # cells

    grid = omap.grid.copy()
    placements: List[Tuple[int, int]] = []
    H, W = grid.shape
    # Persistent forbidden mask across outer-loop iterations (so successive
    # placements respect the no-overlap halo of all prior placements)
    forbidden = np.zeros((H, W), dtype=bool)

    for _ in range(k):
        # Recompute path after each placement
        path = astar(grid, start, goal, inflation_r)
        if path is None:
            break  # already denied; subsequent placements would be no-op
        cur_inf = inflate(grid, inflation_r)
        cur_free = (cur_inf < 99).astype(np.uint8)
        wall_dist = distance_transform_edt(cur_free)
        score = np.zeros((H, W), dtype=np.float64)
        for (r, c) in path:
            score[r, c] = 1.0 / (wall_dist[r, c] + 1.0)
        picks = _select_top_k(score, omap, start, goal, 1, obj_w, obj_h,
                               inflation_r, min_dist,
                               inf_grid=cur_inf, forbidden=forbidden)
        if not picks:
            break
        (r0, c0) = picks[0]
        placements.append((r0, c0))
        # Stamp into the working grid for the next path re-plan
        grid = _stamp(grid, r0, c0, obj_w, obj_h)
        # Persist the no-overlap halo of this placement
        rr, cc = r0, c0
        rh0 = max(0, rr - obj_h); rh1 = min(H, rr + 2 * obj_h)
        ch0 = max(0, cc - obj_w); ch1 = min(W, cc + 2 * obj_w)
        forbidden[rh0:rh1, ch0:ch1] = True
    return placements


# -----------------------------------------------------------------------
# Baseline 3: Edge betweenness centrality (Brandes-sample approximation)

def compute_betweenness_score(omap: OccupancyMap, inflation_r: int,
                               n_paths: int = 200, min_path_dist: int = 50,
                               seed: int = 42) -> np.ndarray:
    """Approximate edge betweenness via sampled shortest paths.
    Returns a per-cell score grid: cell_counts[r, c] = number of sampled
    shortest paths passing through (r, c).
    """
    inf_grid = inflate(omap.grid, inflation_r)
    free = (inf_grid < 99)
    H, W = inf_grid.shape
    free_rs, free_cs = np.where(free)
    n_free = len(free_rs)
    if n_free == 0:
        return np.zeros((H, W), dtype=np.int32)
    rng = np.random.RandomState(seed)
    cell_counts = np.zeros((H, W), dtype=np.int32)
    n_done = 0
    attempts = 0
    while n_done < n_paths and attempts < n_paths * 8:
        attempts += 1
        i = rng.randint(n_free); j = rng.randint(n_free)
        s = (int(free_rs[i]), int(free_cs[i]))
        g = (int(free_rs[j]), int(free_cs[j]))
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) < min_path_dist:
            continue
        p = astar(omap.grid, s, g, inflation_r)
        if p is None:
            continue
        for (r, c) in p:
            cell_counts[r, c] += 1
        n_done += 1
    return cell_counts


def baseline_betweenness(omap: OccupancyMap, start: Tuple[int, int],
                         goal: Tuple[int, int], k: int, obj_w: int, obj_h: int,
                         inflation_r: int, min_dist: int,
                         score_cache: Optional[np.ndarray] = None
                         ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    if score_cache is not None:
        score = score_cache
    else:
        score = compute_betweenness_score(omap, inflation_r)
    if score.sum() == 0:
        return [], score
    placements = _select_top_k(score.astype(np.float64), omap, start, goal,
                                k, obj_w, obj_h, inflation_r, min_dist)
    return placements, score


# -----------------------------------------------------------------------
# Evaluator

def evaluate_placements(omap: OccupancyMap, start: Tuple[int, int],
                        goal: Tuple[int, int],
                        placements: List[Tuple[int, int]], obj_w: int,
                        obj_h: int, inflation_r: int) -> Tuple[float, bool, float]:
    """Return (delay, is_denial, baseline_cost). delay is in cells."""
    base_path = astar(omap.grid, start, goal, inflation_r)
    base_cost = path_cost(base_path) if base_path else float('inf')
    g = omap.grid.copy()
    for (r, c) in placements:
        g = _stamp(g, r, c, obj_w, obj_h)
    p = astar(g, start, goal, inflation_r)
    if p is None:
        return 1e6, True, base_cost
    cost = path_cost(p)
    delay = float(cost - base_cost)
    return delay, False, base_cost


# -----------------------------------------------------------------------
# Jaccard helper

def placement_jaccard(set_a: List[Tuple[int, int]],
                       set_b: List[Tuple[int, int]],
                       obj_w: int, obj_h: int) -> float:
    """Jaccard similarity with footprint-overlap as the match relation.
    Two placements (r1, c1), (r2, c2) "match" if their (obj_h, obj_w)
    footprints overlap, i.e. |r1 - r2| < obj_h and |c1 - c2| < obj_w."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    matched_a = set()
    matched_b = set()
    for i, a in enumerate(set_a):
        for j, b in enumerate(set_b):
            if j in matched_b:
                continue
            if abs(a[0] - b[0]) < obj_h and abs(a[1] - b[1]) < obj_w:
                matched_a.add(i)
                matched_b.add(j)
                break
    matched = len(matched_a)
    union = len(set_a) + len(set_b) - matched
    return matched / union if union else 1.0
