#!/usr/bin/env python3
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node

import numpy as np
import math

from utils import plotting
from utils import env
from tracking import LocalTrackingController

from std_msgs.msg import Float32MultiArray

"""
Created on July 16th, 2024
@author: Taekyung Kim

@description: 
It's a template script example for ROS2 support. 
It calls local tracking controller implemented in tracking.py, and publishes the
control inputs via ROS2 messages. 

@required-scripts: tracking.py
"""

class TrackingControllerNode(Node):
    def __init__(self, control_type):
        super().__init__('tracking_controller_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ctrl_vel', 10)
        self.control_type = control_type
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # Initialize waypoints and tracking controller
        waypoints = [
            [2, 2, math.pi/2],
            [2, 12, 0],
            [10, 12, 0],
            [10, 2, 0]
        ]
        waypoints = np.array(waypoints, dtype=np.float64)
        x_init = waypoints[0]

        plot_handler = plotting.Plotting()
        self.ax, self.fig = plot_handler.plot_grid("Local Tracking Controller")
        self.env_handler = env.Env()

        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 0.5,
            'a_max': 0.5,
            'fov_angle': 70.0,
            'cam_range': 3.0
        }
        self.tracking_controller = LocalTrackingController(
            x_init, robot_spec, control_type=self.control_type, dt=self.dt,
            show_animation=False, save_animation=False, ax=self.ax, fig=self.fig,
            env=self.env_handler
        )

        self.tracking_controller.set_waypoints(waypoints)
        self.tf = 50
        self.steps = int(self.tf / self.dt)

    def timer_callback(self):
        if self.steps <= 0:
            return

        ret = self.tracking_controller.control_step()
        u = self.tracking_controller.get_control_input()

        # Convert control input to Float32MultiArray and publish
        msg = Float32MultiArray()
        msg.data = [float(val) for val in u.flatten()]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

        self.steps -= 1

        if ret == -1:
            # Infeasible
            self.get_logger().info('Control infeasible, stopping...')
            self.destroy_timer(self.timer)

def main(args=None):
    rclpy.init(args=args)
    control_type = 'cbf_qp'
    node = TrackingControllerNode(control_type)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
