"""Exp 2B: Planner-only verification of greedy k=5 on 5 hospital routes.

Loads results/additional_experiments/exp2b/placements.json (computed by
experiments/exp2b_compute_placements.py) and for each route runs Nav2's
ComputePathToPose under two conditions:
  - no_attack: clean hospital map
  - greedy_k5: hospital map with k=5 boxes stamped at greedy placements

Output:
  results/additional_experiments/exp2b/planner_only_results.json

Usage (inside aopp-planner-exp3 container):
    python3 /workspace/scripts/exp2b_planner_only.py \\
        --out /workspace/results/additional_experiments/exp2b/planner_only_results.json
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/workspace')
sys.stdout.reconfigure(line_buffering=True)


def stamp_boxes_on_pgm(src_yaml: Path, placements_world, obj_cells, out_name):
    import numpy as np
    from PIL import Image
    import yaml
    with open(src_yaml) as f:
        meta = yaml.safe_load(f)
    res = meta['resolution']
    origin = meta['origin'][:2]
    img = np.array(Image.open(src_yaml.parent / meta['image']).convert('L'))
    h, w = img.shape

    def w2g(x, y):
        c = int((x - origin[0]) / res)
        r = h - 1 - int((y - origin[1]) / res)
        return r, c

    for (wx, wy) in placements_world:
        r, c = w2g(wx, wy)
        r0 = max(0, r); r1 = min(h, r + obj_cells)
        c0 = max(0, c); c1 = min(w, c + obj_cells)
        img[r0:r1, c0:c1] = 0

    out_pgm = src_yaml.parent / f'{out_name}.pgm'
    out_yaml = src_yaml.parent / f'{out_name}.yaml'
    Image.fromarray(img).save(out_pgm)
    new_meta = dict(meta); new_meta['image'] = out_pgm.name
    with open(out_yaml, 'w') as f:
        yaml.dump(new_meta, f, default_flow_style=False)
    return out_yaml


def run_navfn_planner(map_yaml: Path, start_world, goal_world, timeout=120):
    import rclpy
    from rclpy.action import ActionClient
    from nav2_msgs.action import ComputePathToPose

    tf1 = subprocess.Popen(
        ['ros2', 'run', 'tf2_ros', 'static_transform_publisher',
         '--x', str(start_world[0]), '--y', str(start_world[1]), '--z', '0',
         '--frame-id', 'map', '--child-frame-id', 'base_link'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tf2 = subprocess.Popen(
        ['ros2', 'run', 'tf2_ros', 'static_transform_publisher',
         '--x', '0', '--y', '0', '--z', '0',
         '--frame-id', 'map', '--child-frame-id', 'odom'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tf3 = subprocess.Popen(
        ['ros2', 'run', 'tf2_ros', 'static_transform_publisher',
         '--x', str(start_world[0]), '--y', str(start_world[1]), '--z', '0',
         '--frame-id', 'odom', '--child-frame-id', 'base_link'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    launch = subprocess.Popen(
        ['ros2', 'launch', 'nav2_bringup', 'bringup_launch.py',
         f'map:={map_yaml}',
         'use_sim_time:=false', 'autostart:=true', 'use_composition:=False'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(25)

    node = None
    try:
        rclpy.init()
        node = rclpy.create_node('exp2b_client')
        ac = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')
        if not ac.wait_for_server(timeout_sec=30):
            return (False, None, 'action server not available')
        goal = ComputePathToPose.Goal()
        goal.start.header.frame_id = 'map'
        goal.start.pose.position.x = float(start_world[0])
        goal.start.pose.position.y = float(start_world[1])
        goal.start.pose.orientation.w = 1.0
        goal.goal.header.frame_id = 'map'
        goal.goal.pose.position.x = float(goal_world[0])
        goal.goal.pose.position.y = float(goal_world[1])
        goal.goal.pose.orientation.w = 1.0
        goal.use_start = True
        goal.planner_id = 'GridBased'

        future = ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        gh = future.result()
        if gh is None or not gh.accepted:
            return (False, None, 'goal not accepted')
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(node, rf, timeout_sec=timeout)
        if rf.result() is None:
            return (False, None, 'no result')
        path = rf.result().result.path
        status = rf.result().status
        if not path.poses:
            return (False, None, f'empty path - DENIAL (status={status})')
        length = 0.0
        for i in range(1, len(path.poses)):
            dx = path.poses[i].pose.position.x - path.poses[i-1].pose.position.x
            dy = path.poses[i].pose.position.y - path.poses[i-1].pose.position.y
            length += math.hypot(dx, dy)
        return (True, length, f'{len(path.poses)} waypoints, status={status}')
    except Exception as e:
        return (False, None, f'{type(e).__name__}: {e}')
    finally:
        try:
            if node: node.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass
        for p in [launch, tf1, tf2, tf3]:
            try: p.terminate(); p.wait(timeout=3)
            except:
                try: p.kill()
                except: pass
        for pat in ['bringup_launch.py', '/lifecycle_manager', '/map_server',
                    '/planner_server', '/controller_server', '/bt_navigator',
                    '/waypoint_follower', '/behavior_server', '/smoother_server',
                    '/velocity_smoother', '/amcl', 'static_transform_publisher']:
            subprocess.run(['pkill', '-9', '-f', pat], capture_output=True, timeout=3)
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    placements_data = json.load(open('/workspace/results/additional_experiments/exp2b/placements.json'))
    from src.map_utils import OccupancyMap
    omap = OccupancyMap('/workspace/maps/hospital.yaml')

    hospital_yaml = Path('/workspace/maps/hospital.yaml')

    results = []
    for r in placements_data:
        if r.get('error') or not r.get('placements'):
            print(f"\n=== {r.get('route')}: SKIP (no placements)")
            continue
        route = r['route']
        start = r['start_world']
        goal = r['goal_world']
        placements_world = [list(omap.grid_to_world(*p)) for p in r['placements']]

        # Baseline
        print(f"\n=== {route} no_attack [NavFn] ===")
        ok, length, notes = run_navfn_planner(hospital_yaml, start, goal)
        print(f"  ok={ok} length={length} {notes}")
        results.append({
            'route': route, 'condition': 'no_attack',
            'planner': 'NavFn', 'success': ok, 'path_length_m': length,
            'notes': notes,
        })

        # Greedy k=5 attack
        print(f"\n=== {route} greedy_k5 [NavFn] ===")
        mod_yaml = stamp_boxes_on_pgm(
            hospital_yaml, placements_world, 6, f'mod_exp2b_{route}_k5')
        ok, length, notes = run_navfn_planner(mod_yaml, start, goal)
        print(f"  ok={ok} length={length} {notes}")
        results.append({
            'route': route, 'condition': 'greedy_k5',
            'planner': 'NavFn', 'success': ok, 'path_length_m': length,
            'notes': notes, 'placements_world': placements_world,
        })

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nSaved: {out_path}")
    n_denial = sum(1 for r in results if r['condition'] == 'greedy_k5' and not r['success'])
    n_attack = sum(1 for r in results if r['condition'] == 'greedy_k5')
    print(f"Denial under attack: {n_denial}/{n_attack}")


if __name__ == '__main__':
    main()
