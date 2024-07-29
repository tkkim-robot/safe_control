import numpy as np


class SimpleAttitude:
    def __init__(self, robot, robot_spec):
        self.robot = robot
        self.robot_spec = robot_spec

        self.yaw_rate_const = 0.5

        self.setup_control_problem()

    def setup_control_problem(self):
        pass

    def solve_control_problem(self, robot_state):
        return np.array([self.yaw_rate_const]).reshape(-1, 1)
