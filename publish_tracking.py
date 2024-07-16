import numpy as np
import math

from utils import plotting
from utils import env
from tracking import LocalTrackingController

"""
Created on July 16th, 2024
@author: Taekyung Kim

@description: 
It's a template script example for ROS2 support. 
It calls local tracking controller implementedin tracking.py, and publish the
control inputs via ROS2 messages. 

@required-scripts: tracking.py
"""


def single_agent_main(control_type):
    dt = 0.05

    # TODO: modify the template waypoints
    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [10, 12, 0],
        [10, 2, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)

    x_init = waypoints[0]

    plot_handler = plotting.Plotting()
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")
    env_handler = env.Env()

    robot_spec = {
        'model': 'DynamicUnicycle2D', #'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'fov_angle': 70.0,
        'cam_range': 3.0
    }
    tracking_controller = LocalTrackingController(x_init, robot_spec,
                                         control_type=control_type,
                                         dt=dt,
                                         show_animation=False,
                                         save_animation=False,
                                         ax=ax, fig=fig,
                                         env=env_handler)

    tracking_controller.set_waypoints(waypoints)

    tf = 50
    for _ in range(int(tf / dt)):
        ret = tracking_controller.control_step()
        u = tracking_controller.get_control_input()

        # TODO: publish this control input via ROS
        # u is [acceleration, angular velocity] in m/s^2 and rad/s unit
        print(u.flatten())

        #tracking_controller.draw_plot()
        if ret == -1:
            # infeasible
            break


if __name__ == "__main__":
    single_agent_main(control_type='cbf_qp')