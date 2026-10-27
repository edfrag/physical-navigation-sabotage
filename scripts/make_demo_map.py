#!/usr/bin/env python3
from pathlib import Path
import numpy as np

from aopp.grid import save_occupancy
from aopp.inflation import inflate
from aopp.planning import shortest_path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data" / "maps" / "demo_maze.png"
out.parent.mkdir(parents=True, exist_ok=True)

h = w = 180
occ = np.zeros((h, w), dtype=bool)
occ[[0, -1], :] = True
occ[:, [0, -1]] = True

# Alternating vertical walls create a long corridor-maze route.
for i, c in enumerate(range(30, 160, 25)):
    occ[1:-1, c:c+3] = True
    if i % 2 == 0:
        occ[125:175, c:c+3] = False
    else:
        occ[5:55, c:c+3] = False

start = (160, 10)
goal = (20, 170)
blocked = inflate(occ, 6)
cost, path = shortest_path(blocked, start, goal)
if not path:
    raise SystemExit("generated demo map is not traversable")

save_occupancy(out, occ)
print(f"wrote {out}")
print(f"baseline path cost: {cost:.2f} cells")
