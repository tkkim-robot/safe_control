#!/usr/bin/env python3
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node

import numpy as np
import math

from utils import plotting
from utils import env
from utils.utils import euler_from_quaternion
from tracking import LocalTrackingController

from nav_msgs.msg import Odometry
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
        self.subscriber_odom = self.create_subscription(Float32MultiArray, '/control/obstacle_circles', self.obs_circles_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.subscriber_obs = self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ctrl_vel', 10)
        
        self.control_type = control_type

        # Initialize waypoints and tracking controller
        # waypoints = [
        #     [0, 0, 0],
        #     [1.0, 0.0, 0],
        #     [1.5, -1.0, 0],
        #     [2.0, 0.0, 0],
        #     [2.5, 1.5, 0],
        # ]

        # load csv file waypoints_vis.csv
        waypoints = np.loadtxt("/workspaces/colcon_ws/exp_waypoints_rrt_2.csv", delimiter=",")
        waypoints = np.array(waypoints, dtype=np.float64)
        waypoints[:, 0] = waypoints[:, 0] - waypoints[0, 0]
        waypoints[:, 1] = waypoints[:, 1] - waypoints[0, 1]
        waypoints[:, 0], waypoints[:, 1] = waypoints[:, 1], -1*waypoints[:, 0]
        waypoints[:, 2] -= np.pi/2

        #print(waypoints)
        x_init = waypoints[0]
        waypoints[0][0] += 0.1


        plot_handler = plotting.Plotting()
        self.ax, self.fig = plot_handler.plot_grid("Local Tracking Controller")
        self.env_handler = env.Env()

        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 1.5,
            'a_max': 0.5,
            'v_max': 1.1,
            'fov_angle': 70.0,
            'cam_range': 3.0
        }
        self.tracking_controller = LocalTrackingController(
            x_init, robot_spec, control_type=self.control_type, dt=0.05,
            show_animation=False, save_animation=False, ax=self.ax, fig=self.fig,
            env=self.env_handler
        )

        self.tracking_controller.set_waypoints(waypoints)

    def obs_circles_callback(self, msg):
        obs_circles = msg.data
        # reshape it as [[x,y,radius], ]
        obs_circles = np.array(obs_circles).reshape(-1, 3)
        self.tracking_controller.set_detected_obs(obs_circles)

    def odom_callback(self, msg):
        goal = self.tracking_controller.goal
        if goal is None:
            print("Reached all the waypoints")
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0]
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: {msg.data}')
            return False
        
        pose = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        velocity = msg.twist.twist.linear
        # Convert quaternion to euler angles
        orientation = euler_from_quaternion(orientation) # roll, pitch, yaw order

        self.tracking_controller.set_robot_state(pose, orientation, velocity)
        print("velocity: ", velocity.x, velocity.y)
        print(self.tracking_controller.robot.X)

        ret = self.tracking_controller.control_step()
        u = self.tracking_controller.get_control_input()
        print("Goal: ", self.tracking_controller.goal)

        # Convert control input to Float32MultiArray and publish
        msg = Float32MultiArray()
        msg.data = [float(val) for val in u.flatten()]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

        if ret == -1:
            # Infeasible
            print("Reached all the waypoints")
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0]
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    control_type = 'cbf_qp'
    node = TrackingControllerNode(control_type)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
