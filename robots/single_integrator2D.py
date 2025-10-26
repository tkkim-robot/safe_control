import numpy as np
import casadi as ca

"""
Single Integrator model for CBF-QP and MPC-CBF (casadi)
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


class SingleIntegrator2D:

    def __init__(self, dt, robot_spec):
        '''
            X: [x, y]
            theta: yaw angle
            U: [vx, vy]
            

            Dynamics:
            dx/dt = vx
            dy/dt = vy
            f(x) = [0, 0], g(x) = I(2x2)
            cbf: h(x,y) = ||x-x_obs||^2 + ||y-y_obs||^2- beta*d_min^2
            relative degree: 1
        '''
        self.dt = dt
        self.robot_spec = robot_spec

        self.robot_spec.setdefault('v_max', 1.0)
        self.robot_spec.setdefault('w_max', 0.5)

    def f(self, X, casadi=False):
        if casadi:
            return ca.vertcat(
                0,
                0
            )
        else:
            return np.array([
                             0,
                             0]).reshape(-1, 1)

    def g(self, X, casadi=False):
        if casadi:
            return ca.DM([
                [1, 0],
                [0, 1]
            ])
        else:
            return np.array([[1, 0], [0, 1]])

    def step(self, X, U):
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        return X

    def step_rotate(self, theta, U_attitude):
        theta = angle_normalize(theta + U_attitude[0, 0] * self.dt)
        return theta
    
    def nominal_input(self, X, G, d_min=0.05, k_v=1.0):
        '''
        nominal input for CBF-QP (position control)
        '''
        G = np.copy(G.reshape(-1, 1))  # goal state
        v_max = self.robot_spec['v_max']  # Maximum velocity (x+y)

        pos_errors = G[0:2, 0] - X[0:2, 0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)

        # Compute desired velocities for x and y
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag

        return v_des.reshape(-1, 1)

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
        return np.array([vx_des, vy_des]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        # Single Integrator can always stop.
        return True

    def rotate_to(self, theta, theta_des, k_omega=2.0):
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_omega * error_theta
        yaw_rate = np.clip(yaw_rate, -self.robot_spec['w_max'], self.robot_spec['w_max'])
        return np.array([yaw_rate]).reshape(-1, 1)

    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''Continuous Time High Order CBF'''

        h = 0
        dh_dx = np.zeros((1, 2))

        if obs[-1] == 0:
            obsX = obs[0:2].reshape(-1, 1)
            d_min = obs[2] + robot_radius  # obs radius + robot radius

            h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta*d_min**2
            # relative degree is 1

            dh_dx = (2 * (X[0:2] - obsX[0:2])).T
        elif obs[-1] == 1:
            ox = obs[0]
            oy = obs[1]
            a = obs[2]
            b = obs[3]
            e = obs[4]
            theta = obs[5]

            pox_prime = np.cos(theta)*(X[0]-ox) + np.sin(theta)*(X[1]-oy)
            poy_prime = -np.sin(theta)*(X[0]-ox) + np.cos(theta)*(X[1]-oy)

            h = (pox_prime/(a + robot_radius))**(e) + (poy_prime/(b + robot_radius))**(e) - 1
            dh_dx = np.array([
                e*(pox_prime**(e-1))*(np.cos(theta)/(a + robot_radius)**e) + e*(poy_prime**(e-1))*(-np.sin(theta)/(b + robot_radius)**e),
                e*(pox_prime**(e-1))*(np.sin(theta)/(a + robot_radius)**e) + e*(poy_prime**(e-1))*(np.cos(theta)/(b + robot_radius)**e)
            ]).reshape(1, -1)


        return h, dh_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)

        def _h_circle(x, obs, robot_radius, beta):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            y_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 - beta*d_min**2
            return h
        
        def _h_superellipsoid(x, obs, robot_radius, beta):
            ox = obs[0]
            oy = obs[1]
            a = obs[2]
            b = obs[3]
            e = obs[4]
            theta = obs[5]

            pox_prime = np.cos(theta)*(x[0,0]-ox) + np.sin(theta)*(x[1,0]-oy)
            poy_prime = -np.sin(theta)*(x[0,0]-ox) + np.cos(theta)*(x[1,0]-oy)

            h = ((pox_prime)/(a + robot_radius))**(e) + ((poy_prime)/(b + robot_radius))**(e) - 1
            return h
        
        def h(x, obs, robot_radius, beta=1.01):
            is_circle = (obs[6] < 0.5) 
            
            return ca.if_else(is_circle,
                                _h_circle(x, obs, robot_radius, beta),
                                _h_superellipsoid(x, obs, robot_radius, beta))

        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        # cbf = h_dot + gamma1 * h_k

        return h_k, d_h