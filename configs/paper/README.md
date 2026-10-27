# Paper configurations

The manuscript supplies environment-level parameters but not the exact route coordinates or original map filenames. Create one YAML per route using the `configs/demo_maze.yaml` schema after restoring the original maps.

Reported parameters:

| Environment | Resolution | Inflation | Footprint | Candidate stride |
|---|---:|---:|---:|---:|
| Maze | 0.03 m/cell | 6 cells | 10 cells | restore original |
| Hospital A | 0.05 m/cell | 6 cells | 6 cells | 2 |
| Hospital B | 0.05 m/cell | 6 cells | 6 cells | 6 |
| Warehouse | 0.03 m/cell | 6 cells | 10 cells | <= footprint |
| Depot | 0.04 m/cell | 6 cells | 8 cells | <= footprint |

Main-evaluation stealth distance was approximately 1 m from both endpoints.
