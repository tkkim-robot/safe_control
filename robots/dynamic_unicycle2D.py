import numpy as np
import casadi as ca

"""
Created on July 11th, 2024
@author: Taekyung Kim

@description: 
Dynamic unicycle model for CBF-QP and MPC-CBF (casadi)
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

class DynamicUnicycle2D:
    
    def __init__(self, dt):
        '''
            X: [x, y, theta, v]
            U: [a, omega]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
            relative degree: 2
        '''
        self.model = 'DynamicUnicycle'   
        self.dt = dt     

    def f(self, X, casadi = False):
        if casadi:
            return ca.vertcat(
                X[3,0] * ca.cos(X[2,0]),
                X[3,0] * ca.sin(X[2,0]),
                0,
                0
            )
        else:
            return np.array([X[3,0]*np.cos(X[2,0]),
                            X[3,0]*np.sin(X[2,0]),
                            0,
                            0]).reshape(-1,1)
    
 
    def df_dx(self, X):
        return np.array([  
                         [0, 0, -X[3,0]*np.sin(X[2,0]), np.cos(X[2,0])],
                         [0, 0,  X[3,0]*np.cos(X[2,0]), np.sin(X[2,0])],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]
                         ])
    
    def g(self, X, casadi = False):
        if casadi:
            return ca.DM([
                [0, 0], 
                [0, 0], 
                [0, 1], 
                [1, 0]
            ])
        else:
            return np.array([ [0, 0],[0, 0], [0, 1], [1, 0] ])

        
    def step(self, X, U):
        X = X + ( self.f(X) + self.g(X) @ U )*self.dt
        X[2,0] = angle_normalize(X[2,0])
        return X

    def nominal_input(self, X, G, d_min = 0.05, k_omega = 2.0, k_a = 1.0, k_v = 1.0):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(G.reshape(-1,1)) # goal state
        max_v = 1.0

        distance = max(np.linalg.norm( X[0:2,0]-G[0:2,0] ) - d_min, 0.0) # don't need a min dist since it has accel
        theta_d = np.arctan2( G[1,0]-X[1,0], G[0,0]-X[0,0] )
        error_theta = angle_normalize( theta_d - X[2,0] )

        omega = k_omega * error_theta
        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = min(k_v * distance * np.cos(error_theta), max_v)
        #print("distance: ", distance, "v: ", v, "error_theta: ", error_theta)
        
        accel = k_a * ( v - X[3,0] )
        #print(f"CBF nominal acc: {accel}, omega:{omega}")
        return np.array([accel, omega]).reshape(-1,1)
    
    def stop(self, X, k_a = 1.0):
        # set desired velocity to zero
        v = 0.0
        accel = k_a * ( v - X[3,0] )
        return np.array([accel,0]).reshape(-1,1)
    
    def has_stopped(self, X, tol = 0.01):
        return np.linalg.norm(X[3,0]) < tol
    
    def rotate_to(self, X, theta_des, k_omega = 2.0):
        error_theta = angle_normalize( theta_des - X[2,0] )
        omega = k_omega * error_theta
        return np.array([0.0, omega]).reshape(-1,1)
    
    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''Continuous Time High Order CBF'''
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius # obs radius + robot radius
    
        h = np.linalg.norm( X[0:2] - obsX[0:2] )**2 - beta*d_min**2   
        h_dot = 2 * (X[0:2] - obsX[0:2]).T @ ( self.f(X)[0:2]) # Lgh is zero => relative degree is 2

        df_dx = self.df_dx(X)
        dh_dot_dx = np.append( ( 2 * self.f(X)[0:2] ).T, np.array([[0,0]]), axis = 1 ) + 2 * ( X[0:2] - obsX[0:2] ).T @ df_dx[0:2,:]
        return h, h_dot, dh_dot_dx
    
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta = 1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

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
        #hocbf_2nd_order = h_ddot + (gamma1 + gamma2) * h_dot + (gamma1 * gamma2) * h_k

        return h_k, d_h, dd_h

        
        
    