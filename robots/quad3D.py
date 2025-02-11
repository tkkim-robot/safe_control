import numpy as np
import casadi as ca

"""
Created on Feb 10th, 2025
@author: Taekyung Kim, Yichen Wang

@description: 
3D Quad model for CBF-QP and MPC-CBF (casadi)
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

class Quad3D:
    
    def __init__(self, dt, robot_spec):
        '''
            X: [px, py, pz, vx, vy, vz, phi, theta, psi]
            U: [f, phi_dot, theta_dot, psi_dot]
            system parameters: m
            cbf: h(x) = ||x(:,0:3)-x_obs||^2 - beta*d_min^2 - sigma(s)
            relative degree: 2

            NOTE: Z is defined as positive downwards
            Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/systems/quad3d.py
        '''
        self.dt = dt
        self.robot_spec = robot_spec
        if 'phi_dot_max' not in self.robot_spec:
            self.robot_spec['phi_dot_max'] = np.deg2rad(25.0)
        if 'theta_dot_max' not in self.robot_spec:
            self.robot_spec['theta_dot_max'] = np.deg2rad(25.0)
        if 'psi_dot_max' not in self.robot_spec:
            self.robot_spec['psi_dot_max'] = np.deg2rad(25.0)
        if 'f_max' not in self.robot_spec:
            self.robot_spec['f_max'] = 50.0
        if 'mass' not in self.robot_spec:
            self.robot_spec['mass'] = 1.0
        self.df_dx = np.vstack([np.hstack([np.zeros([3,3]), np.eye(3),np.zeros([3,3])]),np.zeros([6,9])])
      
        self.m = self.robot_spec['mass']
        self.gravity = 9.8

    def f(self, X, casadi=False):
        vx = X[3, 0]
        vy = X[4, 0]
        vz = X[5, 0]
        if casadi:
            return ca.vertcat(
                vx,
                vy,
                vz,
                0,
                0,
                self.gravity,
                0,
                0,
                0,
            )
        else:
            return np.vstack([
                vx,
                vy,
                vz,
                0,
                0,
                self.gravity,
                0,
                0,
                0,
            ]).reshape(-1,1)
    
    def g(self, X, casadi=False):
        phi = X[6, 0] # assign it to scalar symbolic to flatten (avoid error)
        theta = X[7, 0]
        psi = X[8, 0]
        if casadi:
            # Create a 9x4 list-of-lists with symbolic entries:
            # using block-building to do list-to-SX.
            g_list = [[0]*4 for _ in range(9)]
            g_list[3][0] = -ca.sin(theta) / self.m
            g_list[4][0] = ca.cos(theta) * ca.sin(phi) / self.m
            g_list[5][0] = -ca.cos(theta) * ca.cos(phi) / self.m
            g_list[6][1] = 1
            g_list[7][2] = 1
            g_list[8][3] = 1

            # Convert list-of-lists into a single CasADi SX matrix:
            #  - first convert each row (list) to a CasADi row via horzcat
            #  - then stack the rows via vertcat
            g_sx_rows = []
            for row in g_list:
                g_sx_rows.append(ca.horzcat(*row))  # turn [expr1, expr2, ...] into one row
            g_sx = ca.vertcat(*g_sx_rows)
            return g_sx
        else:
            g = np.zeros([9, 4])
            g[3, 0] = -np.sin(theta) / self.m
            g[4, 0] = np.cos(theta) * np.sin(phi) / self.m
            g[5, 0] = -np.cos(theta) * np.cos(phi) / self.m
            g[6, 1] = 1
            g[7, 2] = 1
            g[8, 3] = 1
            return g
        
    def step(self, X, U, casadi=False): 
        X = X + ( self.f(X, casadi) + self.g(X, casadi) @ U )*self.dt
        X[6,0] = angle_normalize(X[6,0])
        X[7,0] = angle_normalize(X[7,0])
        X[8,0] = angle_normalize(X[8,0])
        return X

    def nominal_input(self, X, goal, k_p = 0.05, k_d = 2, k_ang = 5):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(goal.reshape(-1,1)) # goal state
        phi_dot_max = self.robot_spec['phi_dot_max']
        theta_dot_max = self.robot_spec['theta_dot_max']
        psi_dot_max = self.robot_spec['psi_dot_max']
        f_max = self.robot_spec['f_max']
        f_min = 0.0
        
        u_nom = np.zeros([4,1])
        x_err = np.atleast_2d(goal[0:3]).T - X[0:3]
        v_err = 0 - X[3:6]
        F_des = x_err * k_p + v_err * k_d + np.array([0, 0, - self.gravity * self.m]).reshape(-1,1) #proportional control & gravity compensation
        u_nom[0] = max(min(np.linalg.norm(F_des), f_max), f_min)
        a_des = F_des / np.linalg.norm(F_des)
        theta_des = np.arcsin(-1 * a_des[0])
        psi_des = np.arctan2(x_err[1], x_err[0])
        # if np.abs(theta_des) < 0.01:
        #     phi_des = np.arccos(a_des[1] / np.cos(theta_des))
        # else:
        try:
            phi_des = np.arcsin(a_des[1] / np.cos(theta_des))
        except RuntimeWarning:
            phi_des = np.arccos(a_des[2] / np.cos(theta_des))
        u_nom[1] = max(min((phi_des - X[6]) * k_ang, phi_dot_max), -phi_dot_max)
        u_nom[2] = max(min((theta_des - X[7]) * k_ang, theta_dot_max), -theta_dot_max)
        u_nom[3] = max(min((psi_des - X[8]) * k_ang, psi_dot_max), -psi_dot_max)
        return u_nom
    
    def stop(self, X, k_stop = 1):
        u_stop = np.zeros(4)

        v_curr = X[3:6]
        F_des = v_curr * k_stop + np.array([0, 0, - self.gravity * self.m]).reshape(-1,1) #proportional control & gravity compensation
        u_stop[0] = np.linalg.norm(F_des)
        a_des = F_des / u_stop[0]
        theta_des = np.arcsin(-1 * a_des[0])
        try:
            phi_des = np.arcsin(a_des[1] / np.cos(theta_des))
        except RuntimeWarning:
            phi_des = np.arccos(a_des[2] / np.cos(theta_des))
        u_stop[1] = (phi_des - X[6]) * k_stop
        u_stop[2] = (theta_des - X[7]) * k_stop
        u_stop[3] = -1 * X[8] * k_stop
        return u_stop
    
    def has_stopped(self, X, tol = 0.05):
        return np.linalg.norm(X[3:6]) < tol

    def rotate_to(self, X, ang_des, k_omega = 2.0):
        u = np.zeros(4)
        u[0] = self.gravity * self.m
        u[1] = (0 - X[6]) * k_omega
        u[2] = (0 - X[7]) * k_omega
        u[3] = (ang_des - X[8]) * k_omega
        return u
    
    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''obs: [x, y, r]'''
        '''obstacles are infinite cylinders at x and y with radius r, extending in z direction'''
        '''X : [x y z vx vy yz phi theta psi]'''
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius  # obs radius + robot radius

        h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta*d_min**2
        # Lgh is zero => relative degree is 2
        h_dot = 2 * (X[0:2] - obsX[0:2]).T @ (self.f(X)[0:2])

        dh_dot_dx = np.hstack([np.atleast_2d(2 * self.f(X)[0:2]).T[0,:], 0,
                               np.atleast_2d(2 * (X[0:2] - obsX[0:2])).T[0,:],
                               np.zeros(4)])
        return h, h_dot, dh_dot_dx
        
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta = 1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k, casadi=True)
        x_k2 = self.step(x_k1, u_k, casadi=True)

        def h(x, obs, robot_radius, beta = 1.01):
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
        return h_k, d_h, dd_h