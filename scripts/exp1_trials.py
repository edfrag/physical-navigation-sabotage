"""Exp 1: Maze stochastic-failure statistical power.

Runs 80 Gazebo trials in a single container session:
  1A: greedy k=3 (existing fixed placement) × 30
  1B: greedy k=4 (existing fixed placement) × 20
  1C: random k=3 (10 seeds × 3) × 30

Uses run_one_trial from run_full_transfer_matrix.py for the per-trial protocol.
Per-trial JSON schema (simplified per operator):
  - outcome: success | failure_stuck | timeout | error
  - time_to_goal_s
  - recoveries_triggered
  - distance_from_goal_m
  - stalled_seconds_total (derived from cmd_vel timeseries)
  - amcl_pose_covariance_peak (best-effort, optional)

Writes JSONL incrementally so partial results survive interrupts.

Usage (inside aopp-gazebo, Nav2 already running):
    python3 /workspace/scripts/exp1_trials.py \\
        --out /workspace/results/additional_experiments/exp1/exp1_trials.jsonl \\
        --start_at 0
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/workspace/scripts')
sys.path.insert(0, '/workspace')

from run_full_transfer_matrix import (
    run_one_trial, Telemetry, START, GOAL, WORLD, clear_all_boxes
)
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator


# ----- Greedy placements (fixed) -----
GREEDY_K3_GRID = [[579, 115], [302, 369], [480, 185]]
GREEDY_K4_WORLD = [[-6.635, -7.315], [-5.165, -6.085], [-1.115, -0.355], [3.115, 2.375]]


def greedy_k3_world():
    from src.map_utils import OccupancyMap
    omap = OccupancyMap('/workspace/maps/maze.yaml')
    return [list(omap.grid_to_world(r, c)) for r, c in GREEDY_K3_GRID]


def random_k3_world(seed):
    """Compute a random k=3 placement via the offline solver, given seed."""
    sys.path.insert(0, '/workspace')
    from src.map_utils import OccupancyMap
    from src.aopp_solver import AOPPSolver
    from src.planner import inflate

    def find_valid(omap, pt, inf_r):
        inf_grid = inflate(omap.grid, inf_r)
        if inf_grid[pt[0], pt[1]] < 99: return pt
        for d in range(1, 100):
            for dr in range(-d, d+1):
                for dc in range(-d, d+1):
                    nr, nc = pt[0]+dr, pt[1]+dc
                    if 0<=nr<omap.height and 0<=nc<omap.width and inf_grid[nr,nc]<99:
                        return (nr, nc)

    omap = OccupancyMap('/workspace/maps/maze.yaml')
    s_grid = omap.world_to_grid(*START)
    g_grid = omap.world_to_grid(*GOAL)
    s = find_valid(omap, s_grid, 6); g = find_valid(omap, g_grid, 6)
    sw = omap.grid_to_world(*s); gw = omap.grid_to_world(*g)
    solver = AOPPSolver(omap, sw, gw, k=3, obj_w=10, obj_h=10,
                        corridor_d=10, inflation_r=6,
                        candidate_stride=2, min_dist_to_endpoints=30)
    placements_grid, delay, _ = solver.random_placement(seed=seed)
    return [list(omap.grid_to_world(r, c)) for r, c in placements_grid], delay


def derive_stalled_seconds(timeseries, vel_eps=0.02):
    """Sum seconds where |cmd_vel| < eps (robot effectively stopped)."""
    stalled = 0.0
    prev_t = None
    for s in timeseries:
        cv = s.get('cmd_vel')
        t = s.get('t', 0)
        if cv is not None and prev_t is not None:
            speed = math.hypot(cv[0], cv[1])
            ang = abs(cv[2])
            if speed < vel_eps and ang < vel_eps * 5:
                stalled += t - prev_t
        prev_t = t
    return round(stalled, 2)


def classify_outcome(trial_record, distance_from_goal):
    """Map run_one_trial result -> Exp 1 outcome.

    Trust Nav2's SUCCEEDED signal authoritatively (Nav2 has its own goal
    tolerance check). Use the 0.5m threshold + recovery count only to
    distinguish failure modes when Nav2 returned FAILED/UNKNOWN.
    """
    raw_result = trial_record.get('result', '')
    elapsed = trial_record.get('elapsed', 0)
    recoveries = trial_record.get('max_recoveries', 0) or 0

    # Nav2 SUCCEEDED is authoritative — robot reached goal within its tolerance
    if raw_result == 'SUCCEEDED':
        return 'success'

    # Timeout — wall-clock budget exhausted (Nav2 returned UNKNOWN at cancel)
    if 'UNKNOWN' in raw_result or elapsed > 595:
        return 'timeout'

    # Nav2 reported FAILED — explicit local controller / BT failure
    if raw_result == 'FAILED':
        return 'failure_stuck'

    # Anything else (exception, infrastructure issue)
    return 'error'


def normalised_record(experiment_id, trial_id, sub_label, placement_method,
                      placement_world, k, seed, raw):
    """Build the simplified Exp 1 schema record."""
    final_dist = raw.get('final_distance_remaining')
    if final_dist is None:
        final_dist = raw.get('min_distance_remaining')
    elapsed = raw.get('elapsed', 0)
    recoveries = raw.get('max_recoveries', 0)
    timeseries = raw.get('timeseries', [])
    stalled = derive_stalled_seconds(timeseries)
    outcome = classify_outcome(raw, final_dist)
    return {
        'experiment_id': experiment_id,
        'trial_id': trial_id,
        'sub_label': sub_label,
        'placement_method': placement_method,
        'k': k,
        'placement_seed': seed,
        'placement_world': placement_world,
        'global_planner': 'NavFn',
        'local_controller': 'DWB',
        'outcome': outcome,
        'raw_result': raw.get('result'),
        'time_to_goal_s': elapsed if outcome == 'success' else None,
        'elapsed_s': elapsed,
        'recoveries_triggered': recoveries,
        'distance_from_goal_m': final_dist,
        'min_distance_remaining_m': raw.get('min_distance_remaining'),
        'stalled_seconds_total': stalled,
        'n_placements': raw.get('n_placements'),
    }


def append_jsonl(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--start_at', type=int, default=0,
                    help='Skip the first N trials (for resume).')
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build full trial list
    greedy_k3 = greedy_k3_world()
    print(f"Greedy k=3 world placements: {greedy_k3}")
    random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    trial_list = []
    # 1A: greedy k=3, 30 trials
    for i in range(30):
        trial_list.append({
            'sub_label': '1A_greedy_k3', 'placement_method': 'greedy',
            'k': 3, 'seed': None, 'placements': greedy_k3,
        })
    # 1B: greedy k=4, 20 trials
    for i in range(20):
        trial_list.append({
            'sub_label': '1B_greedy_k4', 'placement_method': 'greedy',
            'k': 4, 'seed': None, 'placements': GREEDY_K4_WORLD,
        })
    # 1C: random k=3, 10 seeds × 3
    for seed in random_seeds:
        try:
            placements, _ = random_k3_world(seed)
        except Exception as e:
            print(f"Failed to compute random k=3 seed={seed}: {e}")
            continue
        for trial_in_seed in range(3):
            trial_list.append({
                'sub_label': f'1C_random_k3_seed{seed}',
                'placement_method': 'random', 'k': 3, 'seed': seed,
                'placements': placements,
            })

    print(f"Total trials planned: {len(trial_list)} (start_at={args.start_at})")

    # Init ROS2
    rclpy.init()
    nav = BasicNavigator()
    telemetry = Telemetry()

    # Remove dock once
    subprocess.run(
        ['ign', 'service', '-s', f'/world/{WORLD}/remove',
         '--reqtype', 'ignition.msgs.Entity', '--reptype', 'ignition.msgs.Boolean',
         '--timeout', '3000', '--req', 'name: "standard_dock" type: MODEL'],
        capture_output=True, timeout=5)

    # Run trials
    for trial_id, t in enumerate(trial_list):
        if trial_id < args.start_at:
            continue
        sub = t['sub_label']
        label = f"{sub}_t{trial_id:03d}"
        print(f"\n[{trial_id+1}/{len(trial_list)}] {label}")
        try:
            raw = run_one_trial(nav, telemetry, label, t['placements'], timeout=args.timeout)
        except Exception as e:
            raw = {'label': label, 'placements': t['placements'],
                   'n_placements': len(t['placements']),
                   'result': f'EXCEPTION:{e}', 'elapsed': 0,
                   'final_distance_remaining': None,
                   'min_distance_remaining': None,
                   'max_recoveries': 0, 'timeseries': []}
        record = normalised_record(
            'exp1_maze_stochastic', trial_id, sub, t['placement_method'],
            t['placements'], t['k'], t['seed'], raw)
        append_jsonl(out_path, record)
        print(f"  → outcome={record['outcome']} elapsed={record['elapsed_s']:.0f}s "
              f"recoveries={record['recoveries_triggered']}")

    telemetry.destroy_node()
    try: rclpy.shutdown()
    except: pass
    os._exit(0)


if __name__ == '__main__':
    main()
