
import numpy as np
import matplotlib.pyplot as plt

from safe_control.utils import plotting
from safe_control.utils import env
from safe_control.tracking import LocalTrackingController

def main():
    dt = 0.05
    controller_type = {'pos': 'mpc_cbf'}
    save_animation = False

    # temporal
    waypoints = [
        [2, 2, 0],
        [2, 12, 0],
        [12, 12, 0],
        [12, 2, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)

    x_init = np.append(waypoints[0], 0.0)
    x_goal = np.append(waypoints[-1], 0.0)

    plot_handler = plotting.Plotting()
    ax, fig = plot_handler.plot_grid("Multi-Robot Tracking")
    env_handler = env.Env()

    # Robot 0
    robot_spec_0 = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'sensor': 'rgbd',
        'fov_angle': 45.0,
        'cam_range': 3.0,
        'radius': 0.25,
        'robot_id': 0
    }

    controller_0 = LocalTrackingController(x_init, robot_spec_0,
                                           controller_type=controller_type,
                                           dt=dt,
                                           show_animation=True,
                                           save_animation=save_animation,
                                           ax=ax, fig=fig,
                                           env=env_handler)

    # Robot 1
    robot_spec_1 = {
        'model': 'DynamicUnicycle2D',
        'w_max': 1.0,
        'a_max': 1.5,
        'v_max': 2.0,
        'sensor': 'rgbd',
        'fov_angle': 90.0,
        'cam_range': 5.0,
        'radius': 0.25,
        'robot_id': 1
    }

    controller_1 = LocalTrackingController(x_goal, robot_spec_1,
                                           controller_type=controller_type,
                                           dt=dt,
                                           show_animation=True,
                                           save_animation=False,
                                           ax=ax, fig=fig,
                                           env=env_handler)

    controller_0.set_waypoints(waypoints)
    controller_1.set_waypoints(waypoints[::-1])
    tf = 50
    
    print("Start Multi-Agent Tracking Simulation")
    
    for _ in range(int(tf / dt)):
        ret_list = []
        ret_list.append(controller_0.control_step())
        ret_list.append(controller_1.control_step())
        controller_0.draw_plot()
        # if all elements of ret_list are -1, break
        if all([ret == -1 for ret in ret_list]):
            print("Both agents reached goal!")
            break

    if save_animation:
        controller_0.export_video()

if __name__ == '__main__':
    main()
