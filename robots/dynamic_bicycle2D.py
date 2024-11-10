import numpy as np
import casadi as ca

"""
Dynamic Bicycle Model for CBF-QP and MPC-CBF (casadi)
"""

def angle_normalize(x):
    if isinstance(x, (np.ndarray, float, int)):
        # NumPy implementation
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        # CasADi implementation
        return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


class DynamicBicycle2D:
    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, theta, v, delta]
            U: [a, delta_dot] (acceleration, steering rate)
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
            relative degree: 2
            
            Equations:
            - β = arctan((L_r / L) * tan(δ)) (Slip angle)
            
            - x_dot = v * cos(θ + β)
            - y_dot = v * sin(θ + β)
            - θ_dot = v / L * cos(β) * tan(δ)
            - v_dot = a (input)
            - δ_dot = δ_dot (input)
        '''
        self.dt = dt
        self.robot_spec = robot_spec
        if 'wheel_base' not in self.robot_spec:
            self.robot_spec['wheel_base'] = 2.0
        if 'front_axle_distance' not in self.robot_spec:
            self.robot_spec['front_axle_distance'] = 1.0
        if 'rear_axle_distance' not in self.robot_spec:
            self.robot_spec['rear_axle_distance'] = 1.0
        if 'v_max' not in self.robot_spec:
            self.robot_spec['v_max'] = 2.0
        if 'delta_max' not in self.robot_spec:
            self.robot_spec['delta_max'] = np.deg2rad(30)

    def beta(self, delta):
        # Computes the slip angle beta
        L_r = self.robot_spec['rear_axle_distance']
        L = self.robot_spec['wheel_base']
        return np.arctan((L_r / L) * np.tan(delta))

    def f(self, X, casadi=False):
        beta = self.beta(X[4, 0])
        v = X[3, 0]

        if casadi:
            return ca.vertcat(
                v * ca.cos(X[2, 0] + beta),
                v * ca.sin(X[2, 0] + beta),
                v / self.robot_spec['wheel_base'] * ca.cos(beta) * ca.tan(X[4, 0]),
                0,
                0
            )
        else:
            return np.array([
                v * np.cos(X[2, 0] + beta),
                v * np.sin(X[2, 0] + beta),
                v / self.robot_spec['wheel_base'] * np.cos(beta) * np.tan(X[4, 0]),
                0,
                0
            ]).reshape(-1, 1)

    def g(self, X, casadi=False):
        if casadi:
            g = ca.SX.zeros(5, 2)
            g[3, 0] = 1  # Acceleration affects velocity
            g[4, 1] = 1  # Steering rate affects delta
            return g
        else:
            return np.array([
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1]
            ])

    def step(self, X, U):
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        X[2, 0] = angle_normalize(X[2, 0])  # Normalize heading angle
        return X

    def nominal_input(self, X, G, d_min=0.05, k_omega=2.0, k_a=1.0, k_v=1.0):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(G.reshape(-1, 1))  # goal state

        distance = max(np.linalg.norm(X[0:2, 0] - G[0:2, 0]) - d_min, 0.05)
        theta_d = np.arctan2(G[1, 0] - X[1, 0], G[0, 0] - X[0, 0])
        error_theta = angle_normalize(theta_d - X[2, 0])

        delta_dot = k_omega * error_theta  # Steering velocity
        if abs(error_theta) > np.deg2rad(90):
            desired_velocity = 0.0
        else:
            desired_velocity = k_v * distance * np.cos(error_theta)

        acceleration = k_a * (desired_velocity - X[3, 0])
        return np.array([acceleration, delta_dot]).reshape(-1, 1)

    def stop(self, X, k_a=1.0):
        # Stop by reducing velocity
        acceleration = -k_a * X[3, 0]
        return np.array([acceleration, 0]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        # Stop condition based on velocity
        return np.abs(X[3, 0]) < tol

    def rotate_to(self, X, theta_des, k_omega=2.0):
        error_theta = angle_normalize(theta_des - X[2, 0])
        delta_dot = k_omega * error_theta
        return np.array([0.0, delta_dot]).reshape(-1, 1)

    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''Continuous Time High Order CBF'''
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius

        h = np.linalg.norm(X[0:2] - obsX)**2 - beta * d_min**2
        dh_dx = 2 * (X[0:2] - obsX)
        return h, dh_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time High Order CBF'''
        x_k1 = self.step(x_k, u_k)

        def h(x, obs, robot_radius, beta=1.01):
            obs_pos = obs[0:2]
            d_min = obs[2][0] + robot_radius
            return np.linalg.norm(x[0:2, 0] - obs_pos)**2 - beta * d_min**2

        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)
        d_h = h_k1 - h_k
        return h_k, d_h
