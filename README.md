# Physical Navigation Sabotage: Trusted-Input Attacks on Robot Navigation

Artifact repository for the ACSAC paper **“Physical Navigation Sabotage: Trusted-Input Attacks on Robot Navigation.”**

This repository contains a reproducible implementation of the offline Adversarial Object Placement Problem (AOPP) pipeline described in the paper, together with the structure used for planner, simulation, and hardware results.

## What is included

- Binary occupancy-grid inflation.
- A* shortest-path surrogate on an 8-connected grid.
- Physical placement constraints: budget, stealth distance, footprint realizability, and non-overlap.
- Greedy AOPP placement search with path-margin candidate pruning, candidate stride, and local refinement.
- Random and PathBlock baselines.
- A synthetic demo map and configuration.
- Scripts for running and validating offline experiments.
- Published headline results transcribed from the camera-ready manuscript.
- Placeholders and documentation for Nav2/Gazebo and TurtleBot4 artifacts.

## Important artifact note

The offline solver in this repository has been reconstructed from the camera-ready methodology and is functional. Exact reproduction of the paper's numerical tables additionally requires the original occupancy-grid maps, Gazebo worlds, Nav2 parameter files, and archived per-trial logs. Those files should be copied into the locations described in `data/README.md` and `results/README.md` before archival of the final artifact.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/make_demo_map.py
python scripts/run_offline.py --config configs/demo_maze.yaml --method greedy --k 3
pytest -q
```

Or with Docker:

```bash
docker build -t aopp-artifact .
docker run --rm -v "$PWD:/workspace" -w /workspace aopp-artifact \
  python scripts/run_offline.py --config configs/demo_maze.yaml --method greedy --k 3
```

## Repository layout

```text
src/aopp/              Offline AOPP implementation
scripts/               Reproduction and utility scripts
configs/               Experiment configuration files
data/maps/              Occupancy-grid maps
results/offline/        Offline solver outputs
results/gazebo/         Nav2/Gazebo trial outputs
results/hardware/       TurtleBot4 hardware trial outputs
docker/                 Nav2 container notes/reference Dockerfile
```

See `REPRODUCIBILITY.md` for the full workflow and `SUMMARY.md` for the paper's reported headline results.

## Citation

Please cite the ACSAC paper when using this artifact. A `CITATION.cff` file is provided for convenience.
