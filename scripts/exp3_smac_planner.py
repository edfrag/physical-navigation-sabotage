"""Exp 3: Cross-planner check (Smac 2D vs NavFn) on hospital.

For nurse_to_far_room and corridor_endtoend, run greedy k=5 attack with
Smac 2D as the global planner. Compare outcome to existing NavFn result.

Re-uses the planner_only infrastructure from nav2_planner_only.py but
with a custom Nav2 params file that selects Smac 2D.

Usage (inside aopp-planner-style container):
    python3 /workspace/scripts/exp3_smac_planner.py \
        --out /workspace/results/additional_experiments/exp3/cross_planner_results.json
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


DEFAULT_NAV2_PARAMS = '/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml'


def make_smac_params_file(out_path):
    """Copy Nav2's default params and replace only the planner_server section
    with Smac 2D. Returns the path to the new file."""
    import yaml
    with open(DEFAULT_NAV2_PARAMS) as f:
        params = yaml.safe_load(f)
    # Replace planner with Smac 2D
    params['planner_server']['ros__parameters']['planner_plugins'] = ['GridBased']
    params['planner_server']['ros__parameters']['GridBased'] = {
        'plugin': 'nav2_smac_planner/SmacPlanner2D',
        'tolerance': 0.5,
        'downsample_costmap': False,
        'allow_unknown': True,
        'max_iterations': 1000000,
        'max_on_approach_iterations': 1000,
        'max_planning_time': 5.0,
        'cost_travel_multiplier': 2.0,
        'use_final_approach_orientation': False,
        'smoother': {
            'max_iterations': 1000,
            'w_smooth': 0.3,
            'w_data': 0.2,
            'tolerance': 1e-10,
        },
    }
    # Drop the use_astar key from old NavFn config — leftover doesn't matter
    params['planner_server']['ros__parameters'].pop('use_astar', None)
    # Use sim time false (no Gazebo)
    for node, cfg in params.items():
        if isinstance(cfg, dict) and 'ros__parameters' in cfg:
            cfg['ros__parameters']['use_sim_time'] = False
    with open(out_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    return out_path


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


def run_smac_planner(map_yaml: Path, start_world, goal_world, params_path, timeout=120):
    import rclpy
    from rclpy.action import ActionClient
    from nav2_msgs.action import ComputePathToPose

    # Static TFs (planner needs map -> base_link chain even without robot)
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

    # Launch Nav2 with Smac 2D
    launch = subprocess.Popen(
        ['ros2', 'launch', 'nav2_bringup', 'bringup_launch.py',
         f'map:={map_yaml}',
         f'params_file:={params_path}',
         'use_sim_time:=false', 'autostart:=true', 'use_composition:=False'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(25)

    try:
        rclpy.init()
        node = rclpy.create_node('exp3_client')
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
            return (False, None, f'empty path — DENIAL (status={status})')
        length = 0.0
        for i in range(1, len(path.poses)):
            dx = path.poses[i].pose.position.x - path.poses[i-1].pose.position.x
            dy = path.poses[i].pose.position.y - path.poses[i-1].pose.position.y
            length += math.hypot(dx, dy)
        return (True, length, f'{len(path.poses)} waypoints, status={status}')
    except Exception as e:
        return (False, None, f'{type(e).__name__}: {e}')
    finally:
        try: node.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass
        for p in [launch, tf1, tf2, tf3]:
            try: p.terminate(); p.wait(timeout=3)
            except:
                try: p.kill()
                except: pass
        # Hard-kill Nav2 nodes (avoid matching this script)
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

    # Build Smac 2D params file (copy Nav2 defaults, override planner only)
    params_path = Path('/tmp/smac_params.yaml')
    make_smac_params_file(params_path)

    # Load hospital placements
    multi_hosp = json.load(open('/workspace/results/multi_route_hospital.json'))
    from src.map_utils import OccupancyMap
    omap = OccupancyMap('/workspace/maps/hospital.yaml')

    routes = ['corridor_endtoend', 'nurse_to_far_room']
    hospital_yaml = Path('/workspace/maps/hospital.yaml')

    results = []
    for route_name in routes:
        for r in multi_hosp:
            if r.get('route') != route_name or r.get('k') != 5 or not r.get('placements'):
                continue
            placements_world = [list(omap.grid_to_world(*p)) for p in r['placements']]
            start = r['start_world']; goal = r['goal_world']

            # No-attack baseline with Smac 2D
            print(f"\n=== {route_name} no_attack [Smac 2D] ===")
            ok, length, notes = run_smac_planner(hospital_yaml, start, goal, params_path)
            print(f"  {ok} length={length} {notes}")
            results.append({
                'route': route_name, 'condition': 'no_attack',
                'planner': 'Smac 2D', 'success': ok, 'path_length_m': length,
                'notes': notes,
            })

            # Greedy k=5 with Smac 2D
            print(f"\n=== {route_name} greedy_k5 [Smac 2D] ===")
            mod_yaml = stamp_boxes_on_pgm(hospital_yaml, placements_world, 6,
                                          f'mod_smac_{route_name}_k5')
            ok, length, notes = run_smac_planner(mod_yaml, start, goal, params_path)
            print(f"  {ok} length={length} {notes}")
            results.append({
                'route': route_name, 'condition': 'greedy_k5',
                'planner': 'Smac 2D', 'success': ok, 'path_length_m': length,
                'notes': notes, 'placements_world': placements_world,
            })

            # Persist after each
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            break

    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
