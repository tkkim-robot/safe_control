import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


def angle_normalize(x):
    if isinstance(x, (np.ndarray, float, int)):
        # NumPy implementation
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        # CasADi implementation
        return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

class Quad2D:

    def __init__(self, dt, robot_spec):
        '''
            X: [x, z, theta, x_dot, z_dot, theta_dot]
            U: [u_right, u_left]
            G: [x_goal, z_goal]
            
            x_dot = v_x
            z_dot = v_z
            theta_dot = theta_dot
            v_x_dot = -1/m * sin(theta) * (u_right + u_left)
            v_z_dot = -g + 1/m * cos(theta) * (u_right + u_left)
            theta_dot = r/I * (u_right - u_left)

            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
            relative degree: 2
        '''
        self.dt = dt
        self.robot_spec = robot_spec

        self.robot_spec.setdefault('mass', 1.0)
        self.robot_spec.setdefault('inertia', 0.01)
        self.robot_spec.setdefault('f_min', 1.0)
        self.robot_spec.setdefault('f_max', 10.0)

    def f(self, X, casadi=False):
        m, g = self.robot_spec['mass'], 9.81
        if casadi:
            return ca.vertcat(
                X[3, 0],
                X[4, 0],
                X[5, 0],
                0,
                -g,
                0
            )
        else:
            return np.array([X[3, 0], X[4, 0], X[5, 0], 0, -g, 0]).reshape(-1, 1)
        
    def df_dx(self, X):
        """Jacobian of f with respect to state X."""
        df_dx = np.zeros((6, 6))
        df_dx[0, 3] = 1
        df_dx[1, 4] = 1
        df_dx[2, 5] = 1
        return df_dx
    
    def g(self, X, casadi=False):
        """Control-dependent dynamics"""
        m, I, r = self.robot_spec['mass'], self.robot_spec['inertia'], self.robot_spec['radius']
        theta = X[2, 0]
        if casadi:
            return ca.vertcat(
                ca.horzcat(0, 0, 0, -ca.sin(theta)/m, ca.cos(theta)/m, r/I),
                ca.horzcat(0, 0, 0, -ca.sin(theta)/m, ca.cos(theta)/m, -r/I)
            ).T
        else:
            return np.array([
                [0, 0, 0, -np.sin(theta)/m, np.cos(theta)/m, r/I],
                [0, 0, 0, -np.sin(theta)/m, np.cos(theta)/m, -r/I]
            ]).T
        
    def step(self, X, U):
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        X[2, 0] = angle_normalize(X[2, 0])
        return X
    
    def nominal_input(self, X, G, k_px=3.0, k_dx=0.5, k_pz=0.1, k_dz=0.5,
                k_p_theta=0.05, k_d_theta=0.05):
        """
        Compute the nominal input (rotor forces) for the quad2d dynamics.
        Operates without any programming loops.
        """

        m, g = self.robot_spec['mass'], 9.81
        f_min, f_max = self.robot_spec['f_min'], self.robot_spec['f_max']
        r, I = self.robot_spec['radius'], self.robot_spec['inertia']

        x, z, theta, x_dot, z_dot, theta_dot = X.flatten()

        # Goal position
        x_goal, z_goal = G.flatten()

        # Position and velocity errors
        e_x = x_goal - x
        e_z = z_goal - z
        e_x_dot = -x_dot  # Assuming desired velocity is zero
        e_z_dot = -z_dot

        # Desired accelerations (Outer Loop)
        x_ddot_d = k_px * e_x + k_dx * e_x_dot
        z_ddot_d = k_pz * e_z + k_dz * e_z_dot

        # Desired total acceleration vector
        a_d_x = x_ddot_d
        a_d_z = z_ddot_d + g  # Compensate for gravity

        # Desired thrust magnitude
        a_d_norm = np.sqrt(a_d_x**2 + a_d_z**2)
        T = m * a_d_norm

        # Desired orientation
        theta_d = - np.arctan2(a_d_x, a_d_z) # sign convention

        # Orientation error (Inner Loop)
        e_theta = theta_d - theta
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))  # Normalize angle to [-π, π]
        e_theta_dot = -theta_dot  # Assuming desired angular velocity is zero

        # Desired torque
        tau = k_p_theta * e_theta + k_d_theta * e_theta_dot
        # clip tau
        tau = np.clip(tau, -1, 1)

        # Compute rotor forces
        F_r = (T + tau / r) / 2.0
        F_l = (T - tau / r) / 2.0

        # Enforce constraints
        F_r = np.clip(F_r, f_min, f_max)
        F_l = np.clip(F_l, f_min, f_max)

        return np.array([F_r, F_l]).reshape(-1, 1)
    
    def stop(self, X):
        """
        Compute the nominal input for stopping behavior,
        leveraging the nomianl input function
        """
        x, z, theta, x_dot, z_dot, theta_dot = X.flatten()

        G = np.array([x, z]).reshape(-1, 1) # provide current position as goal
        stop_control = self.nominal_input(X, G)
        return stop_control
        
    def has_stopped(self, X, tol=0.05):
        """Check if quadrotor has stopped within tolerance."""
        return np.linalg.norm(X[3:5, 0]) < tol
    
    def rotate_to(self, X, theta_des, k_omega=2.0):
        """Generate control input to rotate the quadrotor to a target angle."""
        error_theta = angle_normalize(theta_des - X[2, 0])
        omega = k_omega * error_theta
        return np.array([0.0, omega]).reshape(-1, 1)
    
    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''Continuous Time High Order CBF'''
        obsX = obs[0:2].reshape(-1, 1)
        d_min = obs[2] + robot_radius  # obs radius + robot radius

        h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta*d_min**2
        # Lgh is zero => relative degree is 2
        h_dot = 2 * (X[0:2] - obsX[0:2]).T @ (self.f(X)[0:2])

        df_dx = self.df_dx(X)
        dh_dot_dx = np.append((2 * self.f(X)[0:2]).T, np.array([[0, 0, 0, 0]]), axis=1) + 2 * (X[0:2] - obsX[0:2]).T @ df_dx[0:2, :]
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

        # h_k = np.zeros_like(h_k)
        # d_h = np.zeros_like(d_h)
        # dd_h = np.zeros_like(dd_h)
        return h_k, d_h, dd_h

    def render_rigid_body(self, X):
        x, z, theta, _, _, _ = X.flatten()
        # Adjust rectangle's center to align its bottom edge with the robot's circle center
        rect_center_z = z + self.robot_spec['radius'] / 6
        
        # Create a transformation that handles rotation and translation
        transform_rect = Affine2D().rotate(theta).translate(x, rect_center_z) + plt.gca().transData
        
        return transform_rect