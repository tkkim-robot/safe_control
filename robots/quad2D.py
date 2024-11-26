import numpy as np
import casadi as ca


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

        if 'mass' not in self.robot_spec:
            self.robot_spec['mass'] = 1.0
        if 'inertia' not in self.robot_spec:
            self.robot_spec['inertia'] = 1.0
        if 'radius' not in self.robot_spec:
            self.robot_spec['radius'] = 0.25

        if 'f_min' not in self.robot_spec:
            self.robot_spec['f_min'] = 3.0
        if 'f_max' not in self.robot_spec:
            self.robot_spec['f_max'] = 10.0

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
    
    def nominal_input(self, X, G, d_min=0.5, Kp_pos=1.0, theta_max=np.pi/4):
        m, g = self.robot_spec['mass'], 9.81
        f_min, f_max = self.robot_spec['f_min'], self.robot_spec['f_max']
        r, I = self.robot_spec['radius'], self.robot_spec['inertia']

        # Unpack state variables
        x, z, theta, x_dot, z_dot, theta_dot = X.flatten()
        x_goal, z_goal = G
        x_error, z_error = x_goal - x - d_min, z_goal - z - d_min

        # Vertical control (F_z)
        desired_v_z = Kp_pos * z_error
        v_z_error = desired_v_z - z_dot
        F_z = (m * v_z_error) +  (m * g) / np.cos(theta)

        # Horizontal control (F_theta)
        desired_theta = np.arctan2(x_error, z_error) # - np.pi * 3 / 4 + 
        desired_theta = angle_normalize(desired_theta)
        theta_error = desired_theta - theta
        # theta_error = np.clip(theta_error, -theta_max, theta_max)
        F_theta = - I * theta_error * 0.1

        print(theta_error * 180 / np.pi)

        print("desired_theta: ", desired_theta * 180 / np.pi, "theta: ", theta * 180 / np.pi, "theta_error: ", theta_error * 180 / np.pi)


        # Distribute thrust between left and right motors
        u_right = np.clip(F_z / 2 + F_theta, f_min, f_max)
        u_left = np.clip(F_z / 2 - F_theta, f_min, f_max)

        return np.array([u_right, u_left]).reshape(-1, 1)
    
    def stop(self, X, Kp_stop=2.0, Kd_stop=0.5):
        """
        Compute control inputs to bring the quadrotor to a halt with gravity compensation.
        
        Args:
            X (np.ndarray): Current state [x, z, theta, x_dot, z_dot, theta_dot].
            Kp_stop (float): Proportional gain for stopping. Defaults to 2.0.
            Kd_stop (float): Derivative gain for damping. Defaults to 0.5.
        
        Returns:
            np.ndarray: Control inputs [u_right, u_left].
        """
        m, g = self.robot_spec['mass'], 9.81
        f_min, f_max = self.robot_spec['f_min'], self.robot_spec['f_max']
        r, I = self.robot_spec['radius'], self.robot_spec['inertia']

        # Unpack state variables
        x, z, theta, x_dot, z_dot, theta_dot = X.flatten()

        # Gravity compensation for maintaining altitude
        F_total = m * g / np.cos(theta)

        # Linear velocity stopping force
        F_x = -Kp_stop * x_dot  # Horizontal stop force
        F_z = -Kp_stop * z_dot  # Vertical stop force

        # Angular velocity stopping torque
        tau = -Kp_stop * theta_dot - Kd_stop * theta  # Stop rotation and stabilize theta

        # Adjust total thrust to incorporate stopping in x and z directions
        F_total_adjusted = F_total

        # Distribute forces for left and right rotors
        u_right = (F_total_adjusted / 2) + (tau / (2 * r))
        u_left = (F_total_adjusted / 2) - (tau / (2 * r))

        # Clip thrusts to within physical limits
        u_right = np.clip(u_right, f_min, f_max)
        u_left = np.clip(u_left, f_min, f_max)

        return np.array([u_right, u_left]).reshape(-1, 1)
    
        
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
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius  # obs radius + robot radius

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