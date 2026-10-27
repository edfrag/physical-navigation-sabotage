# Reproducibility guide

## 1. Offline surrogate

The implemented offline model follows the camera-ready description:

1. Load a binary occupancy grid.
2. Add square physical object footprints to the base occupancy grid.
3. Apply hard-cutoff binary inflation by the configured radius.
4. Compute an A* path on the free-space graph.
5. Define attack delay as attacked path cost minus baseline path cost, with denial represented by no feasible path.
6. Enforce placement budget, endpoint stealth distance, physical realizability, and non-overlap.
7. For greedy AOPP, restrict candidate centers to a margin around the current shortest path, subsample by candidate stride, then refine around the best strided candidate.

### Demo

```bash
pip install -e .
python scripts/make_demo_map.py
python scripts/run_offline.py --config configs/demo_maze.yaml --method greedy --k 3 \
  --output results/offline/demo_greedy_k3.json
```

Random and PathBlock:

```bash
python scripts/run_offline.py --config configs/demo_maze.yaml --method random --k 3 --seed 0
python scripts/run_offline.py --config configs/demo_maze.yaml --method pathblock --k 3
```

## 2. Exact paper reproduction

Exact numerical reproduction requires the original artifacts that are not recoverable from the paper text alone:

- Maze, Hospital A, Hospital B Levels 1/3, warehouse, and depot occupancy grids.
- Exact start/goal coordinate files for all named and sweep routes.
- Hospital B preprocessing output used for experiments.
- Topology-aware attack implementation/output for min-cut, chokepoint, and betweenness.
- Nav2 map YAML files and exact planner/controller parameter files.
- Gazebo world/model assets and box-spawn scripts.
- Per-trial simulation logs.
- Hardware SLAM map, exact tape-measured placement coordinates, trial logs, and rosbags.

Do not synthesize these files to make the paper tables match. Copy the original experiment assets into this repository.

## 3. Parameters reported by the paper

- Offline greedy: deterministic.
- Random baseline seeds: 0 through 9 (`numpy.random.RandomState`).
- Maze: 0.03 m/cell, 6-cell inflation, 10-cell object footprint.
- Hospital A: 0.05 m/cell, 6-cell inflation, 6-cell footprint, candidate stride 2.
- Hospital B: 0.05 m/cell, 6-cell inflation, 6-cell footprint, candidate stride 6; calibration at stride 4 on five Level-1 routes.
- Warehouse: 0.03 m/cell, 10-cell footprint.
- Depot: 0.04 m/cell, 8-cell footprint.
- Stealth distance in the main evaluation: approximately 1 m from start and goal.
- Nav2: NavFn global planner and DWB controller by default; Smac 2D and MPPI used for transfer checks.
- Gazebo timeout: 600 s.
- Hardware HW-1 timeout: 600 s; HW-2 timeout: 300 s.

## 4. Reconstructed-code caveat

The offline implementation was reconstructed from the methodology in the camera-ready manuscript. Where the paper does not specify an implementation detail, this repository uses a documented conventional choice. In particular, the current A* implementation uses an 8-connected grid with Euclidean diagonal cost and prevents diagonal corner cutting. If the original solver used different connectivity/tie-breaking, replace this component with the original source before claiming bit-for-bit reproduction.

## 5. Verify reported statistics from published counts

The aggregate Fisher exact tests and agreement ratios can be recomputed without raw telemetry:

```bash
python scripts/verify_reported_stats.py
```
