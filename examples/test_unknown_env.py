import argparse
import math

import numpy as np


def build_indoor_env():
    env_width = 24.0
    env_height = 18.0

    # Waypoints create several sharp turns to exercise unknown-obstacle detection.
    waypoints = np.array(
        [
            [2.0, 2.0, math.pi / 2],
            [2.0, 15.0, 0.0],
            [9.4, 15.0, -math.pi / 2],
            [9.4, 5.0, 0.0],
            [16.6, 5.0, math.pi / 2],
            [16.6, 13.0, 0.0],
            [22.0, 13.0, -math.pi / 2],
            [22.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    e_wall = 6.0
    # Thin superellipsoids approximate interior walls/room partitions.
    interior_walls = np.array(
        [
            [6.0, 8.0, 0.22, 6.0, e_wall, 0.0, 1.0],
            [7.4, 11.0, 1.4, 0.18, e_wall, 0.0, 1.0],
            [12.8, 11.0, 1.4, 0.18, e_wall, 0.0, 1.0],
            [13.0, 11.0, 0.18, 5.5, e_wall, 0.0, 1.0],
            [14.7, 7.0, 1.3, 0.18, e_wall, 0.0, 1.0],
            [18.6, 7.0, 1.0, 0.18, e_wall, 0.0, 1.0],
            [19.0, 7.0, 0.18, 5.5, e_wall, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    known_circles = np.array(
        [
            [4.0, 4.0, 0.60],
            [8.0, 3.0, 0.55],
            [14.5, 14.8, 0.75],
            [20.8, 6.5, 0.65],
        ],
        dtype=np.float64,
    )
    known_circles = np.hstack((known_circles, np.zeros((known_circles.shape[0], 4))))

    known_furniture_super = np.array(
        [
            [4.3, 11.2, 0.90, 0.50, 6.0, np.pi / 10, 1.0],
            [15.2, 3.0, 0.95, 0.40, 4.0, -np.pi / 7, 1.0],
            [20.9, 10.9, 0.80, 0.55, 6.0, np.pi / 7, 1.0],
        ],
        dtype=np.float64,
    )

    known_obs = np.vstack((known_circles, interior_walls, known_furniture_super))

    # Unknown obstacles intentionally placed on/near the nominal route corridors.
    # Small lateral offsets keep the scenario feasible while forcing meaningful avoidance.
    unknown_obs = np.array(
        [
            # Segment 1: up (x ~= 2.0)
            [2.45, 4.3, 0.19],
            [1.60, 7.0, 0.19],
            [2.40, 10.2, 0.19],
            [1.65, 13.1, 0.19],
            # Segment 2: right (y ~= 15.0)
            [3.5, 14.6, 0.19],
            [6.0, 15.35, 0.19],
            [8.3, 14.6, 0.19],
            # Segment 3: down (x ~= 9.4)
            [8.95, 13.3, 0.19],
            [9.85, 10.8, 0.19],
            [8.95, 8.0, 0.19],
            [9.85, 6.0, 0.19],
            # Segment 4: right (y ~= 5.0)
            [10.9, 5.45, 0.19],
            [13.0, 4.55, 0.19],
            [15.1, 5.45, 0.19],
            # Segment 5: up (x ~= 16.6)
            [16.15, 6.8, 0.19],
            [17.05, 9.0, 0.19],
            [16.15, 11.2, 0.19],
            # Segment 6: right (y ~= 13.0)
            [18.2, 12.6, 0.19],
            [20.2, 13.4, 0.19],
            [21.4, 12.6, 0.19],
            # Segment 7: down (x ~= 22.0)
            [21.55, 11.0, 0.19],
            [22.45, 8.8, 0.19],
            [21.55, 6.3, 0.19],
            [22.45, 4.2, 0.19],
        ],
        dtype=np.float64,
    )

    return env_width, env_height, waypoints, known_obs, unknown_obs


def get_robot_spec(model, unknown_detection):
    robot_spec = {}
    if model == 'SingleIntegrator2D':
        robot_spec = {
            'model': 'SingleIntegrator2D',
            'v_max': 1.0,
            'radius': 0.25,
            'sensor': 'rgbd',
        }
    elif model == 'DoubleIntegrator2D':
        robot_spec = {
            'model': 'DoubleIntegrator2D',
            'v_max': 1.5,
            'a_max': 1.8,
            'radius': 0.23,
            'sensor': 'rgbd',
            'fov_angle': 70.0,
            'cam_range': 4.5,
            'num_constraints': 10,
            'reached_threshold': 0.45,
            'nominal_k_v': 1.8,
            'nominal_k_a': 2.0,
        }
    elif model == 'Unicycle2D':
        robot_spec = {
            'model': 'Unicycle2D',
            'w_max': 0.5,
            'a_max': 0.5,
            'radius': 0.25,
            'sensor': 'rgbd',
        }
    elif model == 'DynamicUnicycle2D':
        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 0.8,
            'a_max': 1.3,
            'v_max': 1.4,
            'radius': 0.23,
            'sensor': 'rgbd',
            'num_constraints': 6,
            'reached_threshold': 0.45,
            'nominal_k_v': 1.6,
            'nominal_k_a': 1.8,
            'nominal_k_omega': 2.8,
        }
    elif model == 'KinematicBicycle2D':
        robot_spec = {
            'model': 'KinematicBicycle2D',
            'a_max': 0.5,
            'radius': 0.50,
            'sensor': 'rgbd',
        }
    elif model == 'Quad2D':
        robot_spec = {
            'model': 'Quad2D',
            'f_min': 3.0,
            'f_max': 10.0,
            'radius': 0.25,
            'sensor': 'rgbd',
        }
    elif model == 'Quad3D':
        robot_spec = {
            'model': 'Quad3D',
            'radius': 0.25,
        }
    elif model == 'Manipulator2D':
        robot_spec = {
            'model': 'Manipulator2D',
            'w_max': 2.0,
            'Kp': 5.0,
            'radius': 0.25,
            'reached_threshold': 0.5,
        }
    else:
        raise ValueError(f"Unknown model: {model}")

    if robot_spec.get('sensor') == 'rgbd':
        robot_spec.setdefault('fov_angle', 90.0)
        robot_spec.setdefault('cam_range', 4.5)
        robot_spec['unknown_obs_detection'] = unknown_detection

    return robot_spec


def apply_algo_tuning(robot_spec, algo):
    model = robot_spec.get('model')
    if model not in ['DynamicUnicycle2D', 'DoubleIntegrator2D']:
        return robot_spec

    if algo == 'cbf_qp':
        robot_spec['cbf_alpha1'] = 1.5
        robot_spec['cbf_alpha2'] = 1.5
    elif algo == 'mpc_cbf':
        robot_spec['mpc_horizon'] = 6
        if model == 'DoubleIntegrator2D':
            robot_spec['mpc_horizon'] = 9
            robot_spec['mpc_cbf_alpha1'] = 0.32
            robot_spec['mpc_cbf_alpha2'] = 0.32
        else:
            robot_spec['mpc_horizon'] = 7
            robot_spec['mpc_cbf_alpha1'] = 0.26
            robot_spec['mpc_cbf_alpha2'] = 0.26

    return robot_spec


def parse_args():
    parser = argparse.ArgumentParser(description='Run unknown-obstacle tracking simulation in indoor-like environment.')
    parser.add_argument(
        '--model',
        type=str,
        default='du',
        choices=['si', 'di', 'un', 'du', 'kb', 'quad', 'quad3d', 'ma'],
        help='Robot model to use (si, di, un, du, kb, quad, quad3d, ma)',
    )
    parser.add_argument(
        '--algo',
        type=str,
        default='mpc_cbf',
        choices=['cbf_qp', 'mpc_cbf'],
        help='Position controller algorithm',
    )
    parser.add_argument(
        '--att_algo',
        type=str,
        default='velocity_tracking_yaw',
        help='Attitude controller algorithm',
    )
    parser.add_argument('--save_anim', action='store_true', help='Save animation as mp4')
    parser.add_argument('--no_render', action='store_true', help='Disable live rendering (headless run)')
    parser.add_argument(
        '--unknown_detection',
        type=str,
        default='fov',
        choices=['fov', 'ray'],
        help='Unknown-obstacle detection mode: fov (default) or ray (legacy)',
    )
    parser.add_argument('--tf', type=float, default=120.0, help='Simulation horizon in seconds')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_render:
        import matplotlib
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from safe_control.utils import env
    from safe_control.utils import plotting
    from safe_control.tracking import LocalTrackingController

    if args.no_render and args.save_anim:
        print('`--save_anim` requires rendering. Ignoring save request because `--no_render` is set.')

    model_map = {
        'si': 'SingleIntegrator2D',
        'di': 'DoubleIntegrator2D',
        'un': 'Unicycle2D',
        'du': 'DynamicUnicycle2D',
        'kb': 'KinematicBicycle2D',
        'quad': 'Quad2D',
        'quad3d': 'Quad3D',
        'ma': 'Manipulator2D',
    }
    model = model_map[args.model]

    env_width, env_height, waypoints, known_obs, unknown_obs = build_indoor_env()
    robot_spec = get_robot_spec(model, args.unknown_detection)
    robot_spec = apply_algo_tuning(robot_spec, args.algo)

    if model == 'Manipulator2D':
        waypoints = np.array([[6.5, 9.0, 0.0], [10.0, 12.0, 0.0], [14.0, 7.0, 0.0]], dtype=np.float64)

    if model in ['SingleIntegrator2D', 'DoubleIntegrator2D', 'Unicycle2D', 'Quad2D', 'Quad3D']:
        x_init = waypoints[0]
    elif model == 'Manipulator2D':
        x_init = np.zeros(3)
    else:
        x_init = np.append(waypoints[0], 1.0)

    controller_type = {
        'pos': args.algo,
        'att': args.att_algo,
    }

    show_animation = not args.no_render
    save_animation = args.save_anim and show_animation

    if show_animation:
        plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
        ax, fig = plot_handler.plot_grid(
            f"Unknown Environment ({model}, detection={args.unknown_detection})"
        )
        # Larger default window for the bigger indoor map.
        fig.set_size_inches(15.0, 9.5, forward=True)
    else:
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, env_width)
        ax.set_ylim(0.0, env_height)
        ax.set_aspect('equal', adjustable='box')

    env_handler = env.Env(width=env_width, height=env_height, known_obs=known_obs)

    tracking_controller = LocalTrackingController(
        x_init,
        robot_spec,
        controller_type=controller_type,
        dt=0.05,
        show_animation=show_animation,
        save_animation=save_animation,
        show_mpc_traj=False,
        ax=ax,
        fig=fig,
        env=env_handler,
    )

    if model == 'Manipulator2D':
        tracking_controller.robot.robot.base_pos = np.array([env_width / 2.0, env_height / 2.0])

    tracking_controller.obs = known_obs
    tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)

    unexpected_beh = tracking_controller.run_all_steps(tf=args.tf)

    if unexpected_beh == -1 or unexpected_beh == 0:
        print('Success!')
    else:
        print('Failed!')


if __name__ == '__main__':
    main()
