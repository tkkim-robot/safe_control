#!/usr/bin/env python3
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node

import numpy as np
import math
import matplotlib.pyplot as plt

from utils import plotting
from utils import env
from tracking import LocalTrackingController

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleLocalPosition


"""
Created on Feb 2nd, 2025
@author: Rahul H Kumar

@description: 


@required-scripts: tracking.py
"""

def generate_U_waypoints(start_x=0, start_y=0, height=0.5, width=0.2, num_points=5):
    """ Generate waypoints forming a 'U' shape. """
    left_leg = [(start_x, start_y + i * (height / num_points), 0.0, 0.0, 0.0) for i in range(num_points)]
    bottom = [(start_x + j * (width / num_points), start_y, 0.0, 0.0, 0.0) for j in range(num_points)]
    right_leg = [(start_x + width, start_y + i * (height / num_points), 0.0, 0.0, 0.0) for i in range(num_points)]
    return left_leg + bottom + right_leg

def generate_M_waypoints(start_x=0.5, start_y=0, height=0.5, width=0.2, num_points=5):
    """ Generate waypoints forming an 'M' shape. """
    left_leg = [(start_x, start_y + i * (height / num_points), 0.0, 0.0, 0.0) for i in range(num_points)]
    left_middle = [(start_x + j * (width / num_points), start_y + height - j * (height / num_points), 0.0, 0.0, 0.0) for j in range(num_points)]
    right_middle = [(start_x + width - j * (width / num_points), start_y + height - j * (height / num_points), 0.0, 0.0, 0.0) for j in range(num_points)]
    right_leg = [(start_x + width, start_y + i * (height / num_points), 0.0, 0.0, 0.0) for i in range(num_points)]
    return left_leg + left_middle + right_middle + right_leg


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class TrackingControllerNode(Node):
    def __init__(self, control_type):
        super().__init__('tracking_controller_node')
        #self.subscriber_obs = self.create_subscription(Float32MultiArray, '/control/obstacle_circles', self.obs_circles_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        # self.subscriber_odom = self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.vslam_odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.subscriber_odom = self.create_subscription(VehicleLocalPosition, '/px4_3/fmu/out/vehicle_local_position', self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ctrl_vel', 10)
        
        self.control_type = control_type

        # Initialize waypoints and tracking controller
        # Parameters
        radius = 0.5
        num_waypoints = 8

        # # Generate waypoints for circle
        # self.waypoints = np.array([(radius * np.cos(theta), radius * np.sin(theta), 0.0, 0.0, -np.pi/2) 
        #     for theta in np.linspace(0, 2 * np.pi, num_waypoints, endpoint=False)], dtype=np.float32)

        # Square trajectory of length 1m centered at origin
        self.waypoints = np.array([[0.5, 0.5, 0.0, 0.0, 0.0], [0.5, -0.5, 0.0, 0.0, 0.0], [-0.5, -0.5, 0.0, 0.0, 0.0], [-0.5, 0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
        # Repeating waypoints 10 times
        self.waypoints = np.tile(self.waypoints, (4, 1))

        # Generate waypoints for infinity
        # self.waypoints = np.array([
        #     (radius * np.cos(theta) / (1 + np.sin(theta) ** 2),
        #     radius * np.cos(theta) * np.sin(theta) / (1 + np.sin(theta) ** 2),
        #     0.0, 0.0, 0.0)
        #     for theta in np.linspace(0, 2 * np.pi, num_waypoints, endpoint=False)
        # ], dtype=np.float32)

        # Generate waypoints for U-shape and M-shape
        # Generate waypoints for 'U' and 'M'
        # waypoints_U = generate_U_waypoints()
        # waypoints_M = generate_M_waypoints()

        # # Combine the waypoints into one array
        # self.waypoints = np.array(waypoints_U + waypoints_M, dtype=np.float32)

        # self.waypoints = np.append(self.waypoints, [self.waypoints[0]], axis=0)
        # print("Waypoints:", waypoints)
        # waypoints = np.array([[0.1, 0.08, 0.0, 0.0, 0.0], [0.3, 0.08, 0.0, 0.0, 0.0]], dtype=np.float32)
        for i, wp in enumerate(self.waypoints):
            print(f"Waypoint {i}: {wp}")

        x_init = self.waypoints[0]

        


        plot_handler = plotting.Plotting()
        self.ax, self.fig = plot_handler.plot_grid("Local Tracking Controller")
        self.env_handler = env.Env()

        robot_spec = {
            'model': 'DoubleIntegrator2D',
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
        test_type = 'high'
        test_type = 'optimal_decay_mpc_cbf'
        if test_type == 'low':
            self.tracking_controller.pos_controller.cbf_param['alpha1'] = 0.01
            self.tracking_controller.pos_controller.cbf_param['alpha1'] = 0.01
        elif test_type == 'high':
            self.tracking_controller.pos_controller.cbf_param['alpha1'] = 0.2
            self.tracking_controller.pos_controller.cbf_param['alpha1'] = 0.2
        else:
            pass

        self.tracking_controller.obs = np.array([[50, 50, 0.001]])
        # np.array([[-0.3, 0.3, 0.35]])
        # np.array([[1.5, 0.3, 0.4],
        #                                         [0.3, -2.1, 0.4]])
        #                                        [-0.3, 0.3, 0.4],
        self.tracking_controller.set_waypoints(self.waypoints)
        self.prev_u = np.array([0.0, 0.0])
        self.infeasible_flag = False


        self.odom_x_list = []
        self.odom_y_list = [] # Required for plotting
        self.vslam_x_list = []
        self.vslam_y_list = [] # Required for plotting
        self.fig1, self.ax1 = plt.subplots()
        self.plot_waypoints()

    def plot_waypoints(self):
        waypoints_x, waypoints_y = self.waypoints[:, 0], self.waypoints[:, 1]
        self.ax1.plot(waypoints_x, waypoints_y, 'bo-', label='Waypoints')
        self.ax1.set_xlabel("X Position")
        self.ax1.set_ylabel("Y Position")
        self.ax1.set_title("Waypoints and Robot Trajectory")
        self.ax1.legend()
        self.ax1.grid()

    # def vslam_odom_callback(self, msg):
    #     pose = msg.pose.pose.position
    #     # print("pose: ", pose)
    #     print("vslam pose x and y: ", pose.x, pose.y)
    #     self.vslam_x_list.append(pose.x)
    #     self.vslam_y_list.append(pose.y)
    #     self.ax1.plot(self.vslam_x_list, self.vslam_y_list, 'g-', label='VSLAM_odom')
    #     self.fig1.savefig("/workspaces/colcon_ws/waypoints_and_trajectory.png")


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
        print("theta robot: ", angle_normalize(msg.heading+np.pi/2))
        # Convert quaternion to euler angles
        #orientation = euler_from_quaternion(orientation) # roll, pitch, yaw order

        self.tracking_controller.set_robot_state(pose, orientation, velocity)
        print("velocity: ", velocity.x, velocity.y)
        #print(self.tracking_controller.robot.X)

        ret = self.tracking_controller.control_step()
        u = self.tracking_controller.get_control_input()
        x_next = self.tracking_controller.get_full_state()
        # print("x_next: ", x_next)

        # concatenating x_next and u
        full_state_u = np.concatenate((x_next, u), axis=0)
        # TODO: @TK, append yaw state and ctrl to full_state_u and send on msg (Not implemented)
        print("full_state_u: ", full_state_u)

        if ret == -2 or self.infeasible_flag == True:
            print("Infeasible!!!!")
            self.infeasible_flag = True
            u = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Convert control input to Float32MultiArray and publish
        msg = Float32MultiArray()
        msg.data = [float(val) for val in full_state_u.flatten()]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

        if ret == -1:
            # Infeasible
            print("Reached all the waypoints")
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: {msg.data}')

        # Append to list for plotting
        self.odom_x_list.append(pose.x)
        self.odom_y_list.append(pose.y)
        self.ax1.plot(self.odom_x_list, self.odom_y_list, 'r-', label='Vicon_odom')
        self.fig1.savefig("/workspaces/colcon_ws/waypoints_and_trajectory.png")

    def shutdown(self):
        self.fig.savefig("waypoints_and_trajectory.png")
        self.get_logger().info("Saved waypoint and trajectory plot.")

def main(args=None):
    rclpy.init(args=args)
    control_type = 'mpc_cbf'
    # control_type = 'optimal_decay_mpc_cbf'
    #control_type = 'optimal_decay_cbf_qp'
    #control_type = 'mpc_iccbf'
    node = TrackingControllerNode(control_type)
    rclpy.spin(node)
    node.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
