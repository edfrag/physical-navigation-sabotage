"""NavFn verification for the 8 NPH named routes × greedy + min_cut.

Runs Nav2's planner-only stack against each (route, attacker) modified map
and records whether NavFn finds a path. Outputs:
  results/northern_private/navfn_named.json

Usage (inside aopp-turtlebot4 container):
  python3 /workspace/scripts/nav2_verify_nph_named.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/workspace')
sys.stdout.reconfigure(line_buffering=True)


def kill_nav2():
    for pat in ['bringup_launch.py', '/lifecycle_manager', '/map_server',
                '/planner_server', '/controller_server', '/bt_navigator',
                '/waypoint_follower', '/behavior_server', '/smoother_server',
                '/velocity_smoother', '/amcl', 'static_transform_publisher']:
        subprocess.run(['pkill', '-9', '-f', pat], timeout=5,
                       capture_output=True)
    time.sleep(3)


def run_navfn(map_yaml, start_world, goal_world, timeout=30):
    import rclpy
    from rclpy.action import ActionClient
    from nav2_msgs.action import ComputePathToPose

    # Static TFs
    tfs = []
    for args in [
        ['--frame-id', 'map', '--child-frame-id', 'base_link',
         '--x', str(start_world[0]), '--y', str(start_world[1])],
        ['--frame-id', 'map', '--child-frame-id', 'odom',
         '--x', '0', '--y', '0'],
        ['--frame-id', 'odom', '--child-frame-id', 'base_link',
         '--x', str(start_world[0]), '--y', str(start_world[1])],
    ]:
        tfs.append(subprocess.Popen(
            ['ros2', 'run', 'tf2_ros', 'static_transform_publisher'] + args
            + ['--z', '0'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(2)

    nav2 = subprocess.Popen(
        ['ros2', 'launch', 'nav2_bringup', 'bringup_launch.py',
         f'map:={map_yaml}', 'use_sim_time:=false', 'autostart:=true',
         'use_composition:=False'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(25)

    node = None
    success = False
    path_len = None
    notes = ''
    try:
        rclpy.init()
        node = rclpy.create_node('verify_nph_named')
        ac = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')
        if not ac.wait_for_server(timeout_sec=20):
            notes = 'action server unavailable'
        else:
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
            future = ac.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
            gh = future.result()
            if gh is None or not gh.accepted:
                notes = 'goal rejected'
            else:
                rfut = gh.get_result_async()
                rclpy.spin_until_future_complete(node, rfut, timeout_sec=timeout)
                rr = rfut.result()
                if rr is None:
                    notes = 'no result'
                else:
                    path = rr.result.path
                    if path and len(path.poses) > 1:
                        success = True
                        # Path length in metres
                        import math
                        path_len = sum(
                            math.hypot(
                                path.poses[i+1].pose.position.x - path.poses[i].pose.position.x,
                                path.poses[i+1].pose.position.y - path.poses[i].pose.position.y)
                            for i in range(len(path.poses) - 1))
                    else:
                        notes = 'empty path (denial)'
    except Exception as e:
        notes = f'exception: {type(e).__name__}: {e}'
    finally:
        try:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        for p in tfs + [nav2]:
            try: p.terminate()
            except Exception: pass
        time.sleep(1)
        kill_nav2()
    return success, path_len, notes


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='nph_named',
                    help='manifest source filter (nph_named | route_sweep | exp2b | topology_baselines)')
    ap.add_argument('--out', default=None,
                    help='output JSON; defaults derived from --source')
    args = ap.parse_args()

    project = Path('/workspace')
    manifest_path = project / 'results' / 'topology_baselines' / 'navfn_manifest.json'
    out_default = {
        'nph_named':         project / 'results/northern_private/navfn_named.json',
        'nph_stratified':    project / 'results/northern_private/navfn_stratified.json',
        'route_sweep':       project / 'results/additional_experiments/route_sweep/navfn_30routes.json',
        'exp2b':             project / 'results/additional_experiments/exp2b/navfn_4routes.json',
        'topology_baselines': project / 'results/topology_baselines/navfn_outcomes.json',
    }
    out_path = Path(args.out) if args.out else out_default[args.source]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(manifest_path))
    nph_named = [e for e in manifest if e.get('source') == args.source]
    print(f'{args.source} entries: {len(nph_named)}')

    results = []
    if out_path.exists():
        results = json.load(open(out_path))
        done = {(r['env'], r['route'], r['method']) for r in results}
        print(f'Resuming: {len(done)} done')
    else:
        done = set()

    for i, e in enumerate(nph_named):
        key = (e['env'], e['route'], e['method'])
        if key in done:
            continue
        print(f"\n[{i+1}/{len(nph_named)}] {e['env']} {e['route']} {e['method']}")
        sw = e['start_world']; gw = e['goal_world']
        if not sw or not gw:
            print(f"  SKIP: missing start/goal")
            continue
        # Map yaml stamped already (e['yaml'])
        map_yaml = project / e['yaml']
        if not map_yaml.exists():
            print(f"  SKIP: yaml not found {map_yaml}")
            continue
        ok, path_len, notes = run_navfn(map_yaml, sw, gw, timeout=30)
        rec = {
            'env': e['env'], 'route': e['route'], 'method': e['method'],
            'k': e['k'],
            'navfn_success': ok, 'navfn_path_length_m': path_len,
            'notes': notes,
        }
        results.append(rec)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        tag = f'PATH {path_len:.1f}m' if ok else 'DENIAL'
        print(f"  -> {tag}  {notes}")

    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
