import numpy as np
import casadi as ca

"""
Created on July 14h, 2024
@author: Taekyung Kim

@description: 
Kinematic unicycle model for CBF-QP and MPC-CBF (casadi)
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

class Unicycle2D:
    
    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, theta]
            U: [v, omega]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2 - sigma(s)
            relative degree: 1
        '''
        self.dt = dt
        self.robot_spec = robot_spec # not used in this model
        
        self.robot_spec.setdefault('model', 'Unicycle2D')
      
        # for exp (CBF for unicycle)
        self.k1 = 0.5 #=#1.0
        self.k2 = 1.8 #0.5

        self.robot_spec.setdefault('v_max', 1.0)
        self.robot_spec.setdefault('w_max', 0.5)

    def f(self, X, casadi=False):
        if casadi:
            return ca.vertcat([
                0, 
                0, 
                0
            ])
        else:
            return np.array([0,0,0]).reshape(-1,1)
    
    def g(self, X, casadi=False):
        if casadi:
            g = ca.SX.zeros(3, 2)
            g[0, 0] = ca.cos(X[2,0])
            g[1, 0] = ca.sin(X[2,0])
            g[2, 1] = 1
            return g
        else:
            return np.array([ [ np.cos(X[2,0]), 0],
                            [ np.sin(X[2,0]), 0],
                            [0, 1] ]) 
         
    def step(self, X, U): 
        X = X + ( self.f(X) + self.g(X) @ U )*self.dt
        X[2,0] = angle_normalize(X[2,0])
        return X

    def nominal_input(self, X, G, d_min = 0.05, k_omega = 2.0, k_v = 1.0):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(G.reshape(-1,1)) # goal state

        distance = max(np.linalg.norm( X[0:2,0]-G[0:2,0] ) - d_min, 0.05)
        theta_d = np.arctan2(G[1,0]-X[1,0],G[0,0]-X[0,0])
        error_theta = angle_normalize( theta_d - X[2,0] )

        omega = k_omega * error_theta   
        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = k_v*( distance )*np.cos( error_theta )

        return np.array([v, omega]).reshape(-1,1)
    
    def stop(self, X):
        return np.array([0,0]).reshape(-1,1)
    
    def has_stopped(self, X):
        # unicycle can always stop immediately
        return True

    def rotate_to(self, X, theta_des, k_omega = 2.0):
        error_theta = angle_normalize( theta_des - X[2,0] )
        omega = k_omega * error_theta
        return np.array([0.0, omega]).reshape(-1,1)
    
    def sigma(self,s):
        #print("s", s)
        return self.k2 * (np.exp(self.k1-s)-1)/(np.exp(self.k1-s)+1)
    
    def sigma_der(self,s):
        return - self.k2 * np.exp(self.k1-s)/( 1+np.exp( self.k1-s ) ) * ( 1 - self.sigma(s)/self.k2 )
    
    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius # obs radius + robot radius

        theta = X[2,0]

        h = np.linalg.norm( X[0:2] - obsX[0:2] )**2 - beta*d_min**2   
        s = ( X[0:2] - obsX[0:2]).T @ np.array( [np.cos(theta),np.sin(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        
        der_sigma = self.sigma_der(s)
        # [dh/dx, dh/dy, dh/dtheta]^T
        dh_dx = np.append( 
                    2*( X[0:2] - obsX[0:2] ).T - der_sigma * ( np.array([ [np.cos(theta), np.sin(theta)] ]) ),
                    - der_sigma * ( -np.sin(theta)*( X[0,0]-obsX[0,0] ) + np.cos(theta)*( X[1,0] - obsX[1,0] ) ),
                     axis=1)
        # print(h)
        # print(dh_dx)
        return h, dh_dx
        
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta = 1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)

        def h(x, obs, robot_radius, beta = 1.01):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            y_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 - beta*d_min**2
            return h

        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        return h_k, d_h