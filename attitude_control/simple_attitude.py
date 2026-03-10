import numpy as np


class SimpleAtt:
    def __init__(self, robot, robot_spec):
        self.robot = robot
        self.robot_spec = robot_spec

        self.yaw_rate_const = float(
            robot_spec.get(
                'simple_yaw_rate',
                robot_spec.get('w_max', 0.5),
            )
        )

        self.setup_control_problem()

    def setup_control_problem(self):
        pass

    def solve_control_problem(self, *args, **kwargs) -> np.ndarray:
        return np.array([[self.yaw_rate_const]])
