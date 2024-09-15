#!/usr/bin/env python3
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node

import numpy as np
import math

from utils import plotting
from utils import env
from tracking import LocalTrackingController

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleLocalPosition


"""
Created on Sept 14th, 2024
@author: Taekyung Kim

@description: 
It's a modified script from publish_tracking_node.py in "ros2" branch, but it is using vICON state estimation.
It calls local tracking controller implemented in tracking.py, and publishes the
control inputs via ROS2 messages. 

@required-scripts: tracking.py
"""


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class TrackingControllerNode(Node):
    def __init__(self, control_type):
        super().__init__('tracking_controller_node')
        #self.subscriber_obs = self.create_subscription(Float32MultiArray, '/control/obstacle_circles', self.obs_circles_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        #self.subscriber_odom = self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.subscriber_odom = self.create_subscription(VehicleLocalPosition, '/px4_1/fmu/out/vehicle_local_position', self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
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
        waypoints = np.array([[-1.2, -2.4, 0]])

        #print(waypoints)
        x_init = np.array([1.8, 1.8, 0])
        #waypoints[0][0] += 0.1


        plot_handler = plotting.Plotting()
        self.ax, self.fig = plot_handler.plot_grid("Local Tracking Controller")
        self.env_handler = env.Env()

        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 1.0,
            'a_max': 0.5,
            'v_max': 0.95, # 1.1
            'fov_angle': 70.0,
            'cam_range': 3.0
        }
        self.tracking_controller = LocalTrackingController(
            x_init, robot_spec, control_type=self.control_type, dt=0.05,
            show_animation=False, save_animation=False, ax=self.ax, fig=self.fig,
            env=self.env_handler
        )
        #test_type = 'high'
        test_type = 'optimal_decay_mpc_cbf'
        if test_type == 'low':
            self.tracking_controller.pos_controller.cbf_param['alpha1'] = 0.01
            self.tracking_controller.pos_controller.cbf_param['alpha1'] = 0.01
        elif test_type == 'high':
            self.tracking_controller.pos_controller.cbf_param['alpha1'] = 0.2
            self.tracking_controller.pos_controller.cbf_param['alpha1'] = 0.2
        else:
            pass

        self.tracking_controller.obs = np.array([[1.5, 0.3, 0.45],
                                                [0.3, -1.8, 0.45],
                                                 [-0.15, 0.3, 0.45]])
        #np.array([[-0.3, 0.3, 0.35]])
        #np.array([[1.5, 0.3, 0.4],
                                                #[0.3, -2.1, 0.4]])
#                                                [-0.3, 0.3, 0.4],
        self.tracking_controller.set_waypoints(waypoints)

    def odom_callback(self, msg):
        print("goal: ", self.tracking_controller.goal)
        goal = self.tracking_controller.goal
        if goal is None and self.tracking_controller.state_machine != 'stop':
            print("Reached all the waypoints")
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0]
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: {msg.data}')
            return False
        
        # pose = msg.pose.pose.position
        # orientation = msg.pose.pose.orientation
        # velocity = msg.twist.twist.linear
        pose = Odometry().pose.pose.position
        pose.x = msg.x
        pose.y = msg.y
        pose.z = msg.z
        orientation = [0, 0, angle_normalize(msg.heading+np.pi/2)]
        velocity = Odometry().twist.twist.linear
        velocity.x = msg.vx # in earth-fixed frame
        velocity.y = msg.vy # in earth-fixed frame
        # Convert quaternion to euler angles
        #orientation = euler_from_quaternion(orientation) # roll, pitch, yaw order

        self.tracking_controller.set_robot_state(pose, orientation, velocity)
        print("velocity: ", velocity.x, velocity.y)
        #print(self.tracking_controller.robot.X)

        ret = self.tracking_controller.control_step()
        u = self.tracking_controller.get_control_input()

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
    #control_type = 'mpc_cbf'
    #control_type = 'optimal_decay_mpc_cbf'
    control_type = 'optimal_decay_cbf_qp'
    node = TrackingControllerNode(control_type)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
