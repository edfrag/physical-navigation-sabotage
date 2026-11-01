# Evidence index

Files below are recovered experiment outputs, not values transcribed from the
manuscript. Paths are relative to `results/`.

| Paper experiment | Recorded evidence |
| --- | --- |
| Main offline sweeps (Table 2) | `offline_results_inflation6.json`; `additional_experiments/open_envs/offline_open_envs_inflation6.json` |
| Hospital A named routes (Table 3) | `additional_experiments/exp2b/{placements,planner_only_results}.json` |
| Gazebo DWB trials (Table 4) | `additional_experiments/exp1/exp1_trials.jsonl` |
| Smac planner comparison | `additional_experiments/exp3/cross_planner_results.json` |
| MPPI controller comparison | `additional_experiments/exp4/{exp4_trials.jsonl,nav2_mppi.yaml}` |
| Hospital A sweep | `additional_experiments/route_sweep/{route_sweep,navfn_30routes}.json` |
| Hospital B named routes (Table 6) | `northern_private/{level1_named_baselines,level3_named_baselines,navfn_named}.json` |
| Hospital B sweeps | `northern_private/{level1,level3}_route_sweep/route_sweep.json`; `northern_private/navfn_stratified.json` |
| Topology and stride checks | `topology_baselines/baselines_main.json`; `route_topology_metric_wide/`; `northern_private/level1_stride4_calibration/` |
| Attacker-knowledge sensitivity (offline) | `additional_experiments/attacker_knowledge/results.json` |
| Submodularity diagnostics | `submodularity_inflation6_{maze,hospital}.json` |

`python scripts/check_results.py` recounts the trial records and lists unresolved
planner outcomes. The 80 DWB records include one infrastructure error; the five
MPPI records contain none. Rejected goals and missing planner results are kept
separate from explicit empty-path denials. Some planner snapshots differ from
the manuscript's counts and have not been reconciled.

For hardware experiments, the SLAM map and collection/analysis helpers are
included. Original trial logs, recordings, placement coordinates, start/goal
poses, and the offline inflation setting have not been recovered. No hardware
trial rows have been reconstructed from aggregate counts.
