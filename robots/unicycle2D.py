import numpy as np

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class Unicycle2D:
    
    def __init__(self, dt):
        '''
            X: [x, y, theta]
            U: [v, omega]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2 - sigma(s)
            relative degree: 1
        '''
        self.type = 'Unicycle2D'
        self.dt = dt
      
        # for exp 
        self.k1 = 0.5 #=#1.0
        self.k2 = 1.8 #0.5

        self.max_decel = 0.4  # [m/s^2]
        self.max_ang_decel = 0.25  # [rad/s^2]

    def f(self, X):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self, X):
        return np.array([ [ np.cos(X[2,0]), 0],
                          [ np.sin(X[2,0]), 0],
                          [0, 1] ]) 
         
    def step(self, X, U): 
        X = X + ( self.f(X) + self.g(X) @ U )*self.dt
        X[2,0] = angle_normalize(X[2,0])
        return X

    def nominal_input(self, X, G, d_min = 0.05, k_omega = 2.0, k_v = 1.0):
        G = np.copy(G.reshape(-1,1)) # goal state
        distance = max(np.linalg.norm( X[0:2,0]-G[0:2,0] ) - d_min, 0.05) #1.5)
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

    def rotate_to(self, X, theta, k_omega = 2.0):
        error_theta = angle_normalize( theta - X[2,0] )
        omega = k_omega * error_theta
        return np.array([0.0, omega]).reshape(-1,1)
    
    def sigma(self,s):
        #print("s", s)
        return self.k2 * (np.exp(self.k1-s)-1)/(np.exp(self.k1-s)+1)
    
    def sigma_der(self,s):
        return - self.k2 * np.exp(self.k1-s)/( 1+np.exp( self.k1-s ) ) * ( 1 - self.sigma(s)/self.k2 )
    
    def agent_barrier(self, X, obs, robot_radius):
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius # obs radius + robot radius

        beta = 1.01
        theta = X[2,0]

        # if np.linalg.norm( X[0:2] - obsX[0:2] ) > 0.3:
        #     obsX = obsX.copy() * 10
        
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
        