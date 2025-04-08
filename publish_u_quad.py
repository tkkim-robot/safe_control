#!/usr/bin/env python3
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from px4_msgs.msg import TrajectorySetpoint
from std_msgs.msg import Float32MultiArray
import time
import numpy as np

class SetpointAssignerNode(Node):
    def __init__(self):
        super().__init__('setpoint_assigner')
        self.publisher = self.create_publisher(TrajectorySetpoint, '/px4_3/fmu/in/trajectory_setpoint', 50)
        self.ctrl_vel_subscription = self.create_subscription(
            Float32MultiArray,
            'ctrl_vel',
            self.ctrl_vel_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.msg = TrajectorySetpoint()
        self.control_flag = False
        self.x = 0.0  # Position in m
        self.y = 0.0  # Position in m
        self.vx = 0.0  # Velocity in m/s
        self.vy = 0.0  # Velocity in m/s
        self.acc_x = 0.0  # Linear acceleration in m/s^2
        self.acc_y = 0.0  # Linear acceleration in m/s^2
        self.yaw_rate = 0  # Yaw rate in rad/sec ; +ve moves left, -ve moves right
        self.z_des = -1.0
        self.yaw = 0.0
        self.full_state_u = np.zeros(11)
        self.start_time = time.time()
        self.current_time = self.start_time
        self.L = 0.55 #0.3302  # Wheel base in m
        self.prev_state = np.zeros(11) # to store previous state

        # to store previous vel_x of the robot
        self.vel_longitudinal = 0.0
        self.max_vel = 0.8 # spec of motor
        self.min_vel = 0.02

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.height_attained_flag = False
        self.landing_flag = False

        self.last_received_time = time.time()  # For tracking the ctrl_vel receiving rate
        self.stopped_receiving = False      # Flag to indicate if messages are not being received at ≥ 3Hz

        # New timer to check the receiving rate independent of receiving a new message
        self.receiving_check_timer = self.create_timer(0.1, self.check_receive_rate)

    def ctrl_vel_callback(self, msg):
        # print("Received control inputs")
        # print(msg.data)
        self.control_flag = True

        # Update last received time since we just received a message
        self.last_received_time = time.time()

        self.x = msg.data[0] # desired
        self.y = msg.data[1] 
        self.vx = msg.data[2] # desired
        self.vy = msg.data[3] # desired
        self.acc_x = msg.data[4] # desired
        self.acc_y = msg.data[5] # desired
        z_feedback = msg.data[8] # feedback
        if self.acc_x == 0.0 and self.yaw_rate == 0.0 and self.acc_y==0.0:
            self.control_flag = False
        # self.full_state_u = self.control_full_state(self.prev_state, self.acc_x, self.acc_y, self.yaw_rate, dt=0.05, z_des=-0.3)
        # self.prev_state = self.full_state_u
        tune_gain = 1.5 # set to 2 for temporary testing
        # TODO: make neccessary changes in setpoint publisher (eg: publish_tracking_circle.py) to send actual yaw and yaw_rate
        self.yaw = msg.data[6] # desired
        self.yaw_rate = msg.data[7] # desired
        self.full_state_u = np.array([self.x, self.y, self.z_des, self.vx, self.vy, 0.0, tune_gain*self.acc_x, tune_gain*self.acc_y, 0.0, self.yaw, self.yaw_rate])
        # self.full_state_u = np.array([self.x, self.y, self.z_des, self.vx, self.vy, 0.0, tune_gain*self.acc_x, tune_gain*self.acc_y, 0.0, 0.0, 0.0])
        # print("acc_x: ", self.acc_x)
        #self.get_logger().info(f'Received control inputs - acc_x: {self.acc_x}, yaw_rate: {self.yaw_rate}')

        # if(z_feedback > -0.2 and not self.height_attained_flag):
        #     print("Height not attained yet")
        #     self.full_state_u = np.array([0.0, 0.0, self.z_des, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0])
        #     if(z_feedback < -0.2):
        #         self.height_attained_flag = True

    def timer_callback(self):
        # print("Control flag: ", self.control_flag)
        if self.control_flag:
            self.current_time = time.time()
            dt = 0.05
            self.msg.raw_mode = False

            # Convert to float32 before assignment
            self.msg.position = np.array(self.full_state_u[0:3], dtype=np.float32)
            self.msg.velocity = np.array(self.full_state_u[3:6], dtype=np.float32)
            self.msg.acceleration = np.array(self.full_state_u[6:9], dtype=np.float32)
            self.msg.yaw = float(self.full_state_u[9])  # Ensure Python float
            self.msg.yawspeed = float(self.full_state_u[10])  # Ensure Python float
        else:
            self.msg.raw_mode = False
            self.msg.position = np.array(self.full_state_u[0:3], dtype=np.float32)
            self.msg.velocity = np.zeros(3, dtype=np.float32)
            self.msg.acceleration = np.zeros(3, dtype=np.float32)
            self.msg.yaw = float(0.0)
            self.msg.yawspeed = float(0.0)

        # print("Setpoint: ", self.msg)
        self.publisher.publish(self.msg)
        self.print_rate()  # Call function to print publishing rate

    def print_rate(self):
        current_publish_time = time.time()
        dt = current_publish_time - self.start_time
        if dt > 0:
            rate = 1.0 / dt
        else:
            rate = float('inf')
        # print("Publishing rate: {:.2f} Hz".format(rate))
        self.start_time = current_publish_time

    def check_receive_rate(self):
        current_time = time.time()
        dt = current_time - self.last_received_time
        if dt > 0:
            rate = 1.0 / dt
        else:
            rate = float('inf')
        print("Receiving rate: {:.2f} Hz".format(rate))
        if rate < 3.0:
            self.stopped_receiving = True
            print("should stop moving")
            self.full_state_u = np.array([self.x, self.y, self.z_des, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            print("Holding position")
        else:
            self.stopped_receiving = False

def main(args=None):
    rclpy.init(args=args)
    setpoint_assigner_node = SetpointAssignerNode()
    rclpy.spin(setpoint_assigner_node)
    setpoint_assigner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
