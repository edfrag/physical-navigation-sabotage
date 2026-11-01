"""Exp 4: MPPI cross-controller check on maze sw_to_ne with greedy k=3.

Reuses the run_one_trial protocol from run_full_transfer_matrix. Assumes Nav2
has already been launched with MPPI as FollowPath (see overnight_orchestrator.sh).

Runs N trials with the same greedy k=3 placement used in Exp 1A.
Outputs JSONL trial records (matching exp1 schema with global_planner=NavFn,
local_controller=MPPI).

Usage (inside aopp-gazebo, MPPI Nav2 already running):
    python3 /workspace/scripts/exp4_mppi.py \\
        --out /workspace/results/additional_experiments/exp4/exp4_trials.jsonl \\
        --n 10
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/workspace/scripts')
sys.path.insert(0, '/workspace')

from run_full_transfer_matrix import (
    run_one_trial, Telemetry, START, GOAL, WORLD
)
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator


GREEDY_K3_GRID = [[579, 115], [302, 369], [480, 185]]


def derive_stalled_seconds(timeseries, vel_eps=0.02):
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


def classify_outcome(trial_record):
    raw_result = trial_record.get('result', '')
    elapsed = trial_record.get('elapsed', 0)
    if raw_result == 'SUCCEEDED':
        return 'success'
    if 'UNKNOWN' in raw_result or elapsed > 595:
        return 'timeout'
    if raw_result == 'FAILED':
        return 'failure_stuck'
    return 'error'


def normalised_record(trial_id, sub_label, placement_world, raw):
    final_dist = raw.get('final_distance_remaining')
    if final_dist is None:
        final_dist = raw.get('min_distance_remaining')
    elapsed = raw.get('elapsed', 0)
    recoveries = raw.get('max_recoveries', 0)
    timeseries = raw.get('timeseries', [])
    stalled = derive_stalled_seconds(timeseries)
    outcome = classify_outcome(raw)
    return {
        'experiment_id': 'exp4_mppi_cross_controller',
        'trial_id': trial_id,
        'sub_label': sub_label,
        'placement_method': 'greedy',
        'k': 3,
        'placement_world': placement_world,
        'global_planner': 'NavFn',
        'local_controller': 'MPPI',
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
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--timeout', type=float, default=600)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Greedy k=3 world coords
    sys.path.insert(0, '/workspace')
    from src.map_utils import OccupancyMap
    omap = OccupancyMap('/workspace/maps/maze.yaml')
    placements = [list(omap.grid_to_world(r, c)) for r, c in GREEDY_K3_GRID]
    print(f"Greedy k=3 world placements: {placements}")

    rclpy.init()
    nav = BasicNavigator()
    telemetry = Telemetry()

    subprocess.run(
        ['ign', 'service', '-s', f'/world/{WORLD}/remove',
         '--reqtype', 'ignition.msgs.Entity', '--reptype', 'ignition.msgs.Boolean',
         '--timeout', '3000', '--req', 'name: "standard_dock" type: MODEL'],
        capture_output=True, timeout=5)

    for trial_id in range(args.n):
        label = f"exp4_mppi_t{trial_id:03d}"
        print(f"\n[{trial_id+1}/{args.n}] {label}")
        try:
            raw = run_one_trial(nav, telemetry, label, placements, timeout=args.timeout)
        except Exception as e:
            raw = {'label': label, 'placements': placements,
                   'n_placements': len(placements),
                   'result': f'EXCEPTION:{e}', 'elapsed': 0,
                   'final_distance_remaining': None,
                   'min_distance_remaining': None,
                   'max_recoveries': 0, 'timeseries': []}
        record = normalised_record(trial_id, 'exp4_mppi_greedy_k3', placements, raw)
        append_jsonl(out_path, record)
        print(f"  → outcome={record['outcome']} elapsed={record['elapsed_s']:.0f}s "
              f"recoveries={record['recoveries_triggered']}")

    telemetry.destroy_node()
    try: rclpy.shutdown()
    except: pass
    os._exit(0)


if __name__ == '__main__':
    main()
