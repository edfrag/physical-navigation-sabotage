"""Read TurtleBot4's default nav2.yaml and emit a copy with MPPI as FollowPath.

Run inside the aopp-gazebo container.
Usage:
    python3 build_mppi_nav2.py --in /opt/ros/humble/share/turtlebot4_navigation/config/nav2.yaml \
                               --out /tmp/nav2_mppi.yaml
"""
import argparse
import yaml


MPPI_FOLLOWPATH = {
    'plugin': 'nav2_mppi_controller::MPPIController',
    'time_steps': 56,
    'model_dt': 0.05,
    'batch_size': 2000,
    'vx_std': 0.2,
    'vy_std': 0.2,
    'wz_std': 0.4,
    'vx_max': 0.26,
    'vx_min': -0.26,
    'wz_max': 1.0,
    'vy_max': 0.0,
    'ax_max': 2.5,
    'ax_min': -2.5,
    'ay_max': 0.0,
    'az_max': 3.2,
    'iteration_count': 1,
    'prune_distance': 1.7,
    'transform_tolerance': 0.1,
    'temperature': 0.3,
    'gamma': 0.015,
    'motion_model': 'DiffDrive',
    'visualize': False,
    'TrajectoryVisualizer': {
        'trajectory_step': 5,
        'time_step': 3,
    },
    'AckermannConstraints': {
        'min_turning_r': 0.2,
    },
    'critics': [
        'ConstraintCritic', 'ObstaclesCritic', 'GoalCritic', 'GoalAngleCritic',
        'PathAlignCritic', 'PathFollowCritic', 'PathAngleCritic', 'PreferForwardCritic',
    ],
    'ConstraintCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 4.0},
    'GoalCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 5.0, 'threshold_to_consider': 1.4},
    'GoalAngleCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 3.0, 'threshold_to_consider': 0.5},
    'PreferForwardCritic': {'enabled': True, 'cost_power': 1, 'cost_weight': 5.0, 'threshold_to_consider': 0.5},
    'ObstaclesCritic': {
        'enabled': True, 'cost_power': 1,
        'repulsion_weight': 1.5, 'critical_weight': 20.0,
        'consider_footprint': False, 'collision_cost': 10000.0,
        'collision_margin_distance': 0.1, 'near_goal_distance': 0.5,
        'inflation_radius': 0.55, 'cost_scaling_factor': 10.0,
    },
    'PathAlignCritic': {
        'enabled': True, 'cost_power': 1, 'cost_weight': 14.0,
        'max_path_occupancy_ratio': 0.05, 'trajectory_point_step': 4,
        'threshold_to_consider': 0.5, 'offset_from_furthest': 20,
        'use_path_orientations': False,
    },
    'PathFollowCritic': {
        'enabled': True, 'cost_power': 1, 'cost_weight': 5.0,
        'offset_from_furthest': 5, 'threshold_to_consider': 1.4,
    },
    'PathAngleCritic': {
        'enabled': True, 'cost_power': 1, 'cost_weight': 2.0,
        'offset_from_furthest': 4, 'threshold_to_consider': 0.5,
        'max_angle_to_furthest': 1.2, 'mode': 0,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.inp) as f:
        params = yaml.safe_load(f)
    cs = params['controller_server']['ros__parameters']
    cs['FollowPath'] = MPPI_FOLLOWPATH
    cs['controller_frequency'] = 30.0
    with open(args.out, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    print(f'Wrote MPPI config -> {args.out}')


if __name__ == '__main__':
    main()
