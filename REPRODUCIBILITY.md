# Reproducibility

## Check the recorded evidence

Run from the repository root with Python 3.12. Dependency versions were
recovered from the original project's environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test.py
python -m unittest discover -s tests -v
python scripts/check_results.py
python scripts/replay_offline.py
```

The checks verify archived map/result hashes, recount the recorded trials, and
re-evaluate the 20 saved main-sweep placement sets. They do not regenerate the
original observations. The smoke test uses a temporary synthetic corridor.

Maps are named `maze`, `hospital` (Hospital A), `level1` and `level3` (Hospital B),
`warehouse`, `depot`, and `home_living_kitchen` (hardware). Floor-plan sources
are cited in paper references [10] and [11]; the maze, warehouse, and depot
reference maps were obtained from `turtlebot4_navigation`.

## Run experiments

Use a separate working copy for new runs. Recovered scripts use fixed output
paths and may resume from existing results or overwrite them. Preserve archived
inputs and write new observations separately.

| Experiment | Command |
| --- | --- |
| Maze and Hospital A | `python -m experiments.run_inflation6` |
| Warehouse and depot | `python -m experiments.exp_open_envs` |
| Hospital A named routes | `python -m experiments.exp2b_compute_placements` |
| Hospital A sweep | `python -m experiments.exp_route_sweep` |
| Hospital B named routes | `python -m experiments.exp_nph_named_baselines level1` (repeat for `level3`) |
| Hospital B sweep | `python -m experiments.exp_nph_route_sweep level1` (repeat for `level3`) |
| Topology baselines | `python -m experiments.exp_topology_baselines` |
| Hospital A topology margin | `python -m experiments.exp_topology_metric_wide` |
| Stride calibration | `python -m experiments.exp_stride_calibration` |
| Attacker-knowledge sensitivity | `python -m experiments.exp_attacker_knowledge` |

Saved configurations describe each recorded run. Costs are in grid cells;
`1000000` is a denial sentinel. The recovered solver resolves equal gains by
worker completion order, so use saved placements for exact placement replay.
The binary surrogate and Nav2 costmap are distinct evaluation layers.

## Nav2 and Gazebo

`Dockerfile` is the recovered ROS 2 Humble / Ignition Fortress recipe. Build it
with `bash scripts/build_turtlebot4_image.sh`. Upstream packages are not locked;
a fresh image build and a complete simulation rerun have not been verified.

In a separate experiment copy mounted at `/workspace` inside the container:

```bash
source /opt/ros/humble/setup.bash
cd /workspace
python3 -m experiments.stamp_navfn_maps
python3 scripts/nav2_verify_nph_named.py --source nph_named --out /workspace/output/navfn_named.json
```

Other source selectors are `route_sweep`, `nph_stratified`, and `exp2b`. Run these
scripts in an isolated container; they manage their own Nav2 processes.

Gazebo entry points are `scripts/exp1_trials.py` (DWB) and
`scripts/exp4_mppi.py` (MPPI). They require a running TurtleBot4 maze simulation.
The MPPI parameters are in `results/additional_experiments/exp4/nav2_mppi.yaml`.
Use `--out` to select new output files. See [SUMMARY.md](SUMMARY.md) for evidence
availability and remaining gaps.
