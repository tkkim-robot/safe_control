
import numpy as np
import math
import argparse

from safe_control.utils import plotting
from safe_control.utils import env
from safe_control.tracking import LocalTrackingController

def main():
    parser = argparse.ArgumentParser(description='Run single agent tracking simulation.')
    parser.add_argument('--model', type=str, default='du', 
                        choices=['si', 'di', 'un', 'du', 'kb', 'quad', 'quad3d'],
                        help='Robot model to use (si, di, un, du, kb, quad, quad3d)')
    parser.add_argument('--algo', type=str, default='mpc_cbf',
                        choices=['cbf_qp', 'mpc_cbf'],
                        help='Position controller algorithm')
    parser.add_argument('--att_algo', type=str, default='velocity_tracking_yaw',
                        help='Attitude controller algorithm')
    args = parser.parse_args()

    # Map short names to full model names
    model_map = {
        'si': 'SingleIntegrator2D',
        'di': 'DoubleIntegrator2D',
        'un': 'Unicycle2D',
        'du': 'DynamicUnicycle2D',
        'kb': 'KinematicBicycle2D',
        'quad': 'Quad2D',
        'quad3d': 'Quad3D'
    }
    
    model = model_map[args.model]
    dt = 0.05
    
    controller_type = {
        'pos': args.algo,
        'att': args.att_algo
    }

    # Default waypoints
    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [12, 12, 0],
        [12, 2, 0]
    ]

    # Define static obs
    known_obs = np.array([[2.2, 5.0, 0.2], [3.0, 5.0, 0.2], [4.0, 9.0, 0.3], [1.5, 10.0, 0.5], [9.0, 11.0, 1.0], [7.0, 7.0, 3.0], [4.0, 3.5, 1.5],
                        [10.0, 7.3, 0.4],
                        [6.0, 13.0, 0.7], [5.0, 10.0, 0.6], [11.0, 5.0, 0.8], [13.5, 11.0, 0.6]])

    env_width = 14.0
    env_height = 14.0
    
    robot_spec = {}
    
    if model == 'SingleIntegrator2D':
        robot_spec = {
            'model': 'SingleIntegrator2D',
            'v_max': 1.0,
            'radius': 0.25
        }
    elif model == 'DoubleIntegrator2D':
        robot_spec = {
            'model': 'DoubleIntegrator2D',
            'v_max': 1.0,
            'a_max': 1.0,
            'radius': 0.25,
            'sensor': 'rgbd'
        }
    elif model == 'Unicycle2D':
         robot_spec = {
            'model': 'Unicycle2D',
            'w_max': 0.5,
            'a_max': 0.5,
            'sensor': 'rgbd',
            'radius': 0.25
        }
    elif model == 'DynamicUnicycle2D':
        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 0.5,
            'a_max': 0.5,
            'sensor': 'rgbd',
            'radius': 0.25
        }
    elif model == 'KinematicBicycle2D':
        robot_spec = {
            'model': 'KinematicBicycle2D',
            'a_max': 0.5,
            'sensor': 'rgbd',
            'radius': 0.5
        }
    elif model == 'Quad2D':
        robot_spec = {
            'model': 'Quad2D',
            'f_min': 3.0,
            'f_max': 10.0,
            'sensor': 'rgbd',
            'radius': 0.25
        }
    elif model == 'Quad3D':
        robot_spec = {
            'model': 'Quad3D',
            'radius': 0.25
        }
        # override the waypoints with z axis
        waypoints = [
            [2, 2, 0, math.pi/2],
            [2, 12, 1, 0],
            [12, 12, -1, 0],
            [12, 2, 0, 0]
        ]
    
    waypoints = np.array(waypoints, dtype=np.float64)

    if model in ['SingleIntegrator2D', 'DoubleIntegrator2D', 'Unicycle2D', 'Quad2D', 'Quad3D']:
        x_init = waypoints[0]
    else:
        x_init = np.append(waypoints[0], 1.0)
    
    if len(known_obs) > 0 and known_obs.shape[1] != 7:
        known_obs = np.hstack((known_obs, np.zeros((known_obs.shape[0], 4)))) # Set static obs velocity 0.0 at (5, 5)

    plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
    ax, fig = plot_handler.plot_grid(f"Tracking with {model}") 
    env_handler = env.Env()

    tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                  controller_type=controller_type,
                                                  dt=dt,
                                                  show_animation=True,
                                                  save_animation=False,
                                                  show_mpc_traj=False,
                                                  ax=ax, fig=fig,
                                                  env=env_handler)

    # Set obstacles
    tracking_controller.obs = known_obs
    tracking_controller.set_waypoints(waypoints)
    unexpected_beh = tracking_controller.run_all_steps(tf=100)
    
    if unexpected_beh == -1 or unexpected_beh == 0:
        print("Success!")
    else:
        print("Failed!")

if __name__ == '__main__':
    main()
