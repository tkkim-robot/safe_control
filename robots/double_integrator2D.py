import numpy as np
import casadi as ca

"""
Created on July 15th, 2024
@author: Taekyung Kim

@description: 
Double Integrator model for CBF-QP and MPC-CBF (casadi) with separated position and attitude states
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


class DoubleIntegrator2D:

    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, vx, vy]
            theta: yaw angle
            U: [ax, ay]
            U_attitude: [yaw_rate]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
            relative degree: 2
        '''
        self.dt = dt
        self.robot_spec = robot_spec

        if 'a_max' not in self.robot_spec:
            self.robot_spec['a_max'] = 1.0
        if 'v_max' not in self.robot_spec:
            self.robot_spec['v_max'] = 1.0
        if 'ax_max' not in self.robot_spec:
            self.robot_spec['ax_max'] = self.robot_spec['a_max']
        if 'ay_max' not in self.robot_spec:
            self.robot_spec['ay_max'] = self.robot_spec['a_max']
        if 'w_max' not in self.robot_spec:
            self.robot_spec['w_max'] = 0.5

    def f(self, X, casadi=False):
        if casadi:
            return ca.vertcat(
                X[2, 0],
                X[3, 0],
                0,
                0
            )
        else:
            return np.array([X[2, 0],
                             X[3, 0],
                             0,
                             0]).reshape(-1, 1)

    def df_dx(self, X):
        return np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def g(self, X, casadi=False):
        if casadi:
            return ca.DM([
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1]
            ])
        else:
            return np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

    def step(self, X, U):
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        return X

    def step_rotate(self, theta, U_attitude):
        theta = angle_normalize(theta + U_attitude[0, 0] * self.dt)
        return theta

    def nominal_input(self, X, G, d_min=0.05, k_v=1.0, k_a=1.0):
        '''
        nominal input for CBF-QP (position control)
        '''
        G = np.copy(G.reshape(-1, 1))  # goal state
        v_max = self.robot_spec['v_max']  # Maximum velocity (x+y)
        a_max = self.robot_spec['a_max']  # Maximum acceleration

        pos_errors = G[0:2, 0] - X[0:2, 0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)

        # Compute desired velocities for x and y
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag

        # Compute accelerations
        current_v = X[2:4, 0]
        a = k_a * (v_des - current_v)
        a_mag = np.linalg.norm(a)
        if a_mag > a_max:
            a = a * a_max / a_mag

        return a.reshape(-1, 1)

    def nominal_attitude_input(self, theta, theta_des, k_theta=1.0):
        '''
        nominal input for attitude control
        '''
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_theta * error_theta
        return np.array([yaw_rate]).reshape(-1, 1)

    def stop(self, X, k_a=1.0):
        # Set desired velocity to zero
        vx_des, vy_des = 0.0, 0.0
        ax = k_a * (vx_des - X[2, 0])
        ay = k_a * (vy_des - X[3, 0])
        return np.array([ax, ay]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        return np.linalg.norm(X[2:4, 0]) < tol

    def rotate_to(self, theta, theta_des, k_omega=2.0):
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_omega * error_theta
        return np.array([yaw_rate]).reshape(-1, 1)

    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''Continuous Time High Order CBF'''
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius  # obs radius + robot radius

        h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta*d_min**2
        # Lgh is zero => relative degree is 2, f(x)[0:2] actually equals to X[2:4]
        h_dot = 2 * (X[0:2] - obsX[0:2]).T @ (self.f(X)[0:2])

        # these two options are the same
        # df_dx = self.df_dx(X)
        # dh_dot_dx = np.append( ( 2 * self.f(X)[0:2] ).T, np.array([[0,0]]), axis = 1 ) + 2 * ( X[0:2] - obsX[0:2] ).T @ df_dx[0:2,:]
        dh_dot_dx = np.append(2 * X[2:4].T, 2 * (X[0:2] - obsX[0:2]).T, axis=1)
        return h, h_dot, dh_dot_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

        def h(x, obs, robot_radius, beta=1.01):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            y_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 - beta*d_min**2
            return h

        h_k2 = h(x_k2, obs, robot_radius, beta)
        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        dd_h = h_k2 - 2 * h_k1 + h_k
        # hocbf_2nd_order = h_ddot + (gamma1 + gamma2) * h_dot + (gamma1 * gamma2) * h_k

        return h_k, d_h, dd_h