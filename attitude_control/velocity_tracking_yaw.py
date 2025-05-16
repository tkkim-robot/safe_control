import numpy as np

"""
Created on April 20th, 2025
@author: Taekyung Kim

@description: 
This code implements a velocity tracking yaw controller for various robot dynamics, but mostly for integrators.
It computes the desired yaw angle based on the robot's velocity vector.
Assume tracks velocity yaw perfectly, then it is guaranteed to observe the potential obstacles along the path.

@note: 
- Can be used in general cases.
- Can be used as a backup attitude controller of the gatekeeper.

"""

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class VelocityTrackingYaw:
    """
    Backup attitude controller: for a single-integrator yaw model
    (˙yaw = u_att), aligns heading with the instantaneous velocity.
    """
    def __init__(self, robot, robot_spec, kp=1.5):
        self.robot = robot
        self.robot_spec = robot_spec
        self.model = robot_spec['model']

        self.kp = kp
        self.w_max = robot_spec.get('w_max', 0.5) 

    def solve_control_problem(self,
                              robot_state: np.ndarray,
                              current_yaw: float,
                              u: np.ndarray) -> float:
        # 1) Extract state
        if self.model == 'SingleIntegrator2D':
            vx = u[0, 0]
            vy = u[1, 0]
        elif self.model == 'DoubleIntegrator2D':
            vx = robot_state[2, 0]
            vy = robot_state[3, 0]
        speed = np.hypot(vx, vy)

        # 2) If nearly stationary, hold yaw
        if speed < 1e-2:
            return np.array([0.0]).reshape(-1, 1)
        

        # 3) Compute error
        desired_yaw = np.arctan2(vy, vx)
        yaw_err = angle_normalize(desired_yaw - current_yaw)

        # 5) P‐control and clip
        u_att = self.kp * yaw_err
        u_att = np.clip(u_att, -self.w_max, self.w_max)

        return np.array([u_att]).reshape(-1, 1)