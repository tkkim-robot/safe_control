import numpy as np
import casadi as ca

"""
state and control input
X: [x, z, theta, x_dot, z_dot, theta_dot]
U: [u_right, u_left]
--- # TODO: Check the eqautions again cos
x_dot = v_x
z_dot = v_z
theta_dot = theta_dot
v_x_dot = -1/m * (u_R * sin(theta) + u_L * sin(theta))
v_z_dot = -g + 1/m * (u_R * cos(theta) + u_L * cos(theta))
theta_dot = r/I * (u_R - u_L)
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

class Quad2D:

    def __init__(self, dt, robot_spec):
        '''
            X: [x, z, theta, x_dot, z_dot, theta_dot]
            U: [u_right, u_left]
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
        # TODO: Check this
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
        # print("X", X.shape)
        # print("U", U.shape)
        # print("f(X)", self.f(X).shape)
        # print("g(X)", self.g(X).shape)
        # print("X + f(X) + g(X) @ U", (self.f(X) + self.g(X) @ U).shape)
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        X[2, 0] = angle_normalize(X[2, 0])
        # print("X", X.shape)
        return X

    # TODO: fix nominal input
    def nominal_input(self, X, G, d_min=0.05, k_omega=2.0, k_a=1.0, k_v=1.0):
        """
        Nominal input for CBF-QP, outputting u_right and u_left.
        
        Parameters:
            X: Current state [x, z, theta, x_dot, z_dot, theta_dot]
            G: Goal state [x_goal, z_goal, theta_goal]
            d_min: Minimum allowable distance to the goal
            k_omega: Gain for angular velocity control
            k_a: Gain for linear acceleration control
            k_v: Gain for velocity damping
            
        Returns:
            Nominal control inputs [u_right, u_left]
        """
        G = np.copy(G.reshape(-1, 1))  # Ensure goal state is a column vector
        f_max = self.robot_spec['f_max']
        m, I, r = self.robot_spec['mass'], self.robot_spec['inertia'], self.robot_spec['radius']

        # Calculate distance and direction to the goal
        distance = max(np.linalg.norm(X[0:2, 0] - G[0:2, 0]) - d_min, 0.0)
        theta_d = np.arctan2(G[1, 0] - X[1, 0], G[0, 0] - X[0, 0])
        error_theta = angle_normalize(theta_d - X[2, 0])

        # Linear velocity control to reach the goal
        v_desired = k_v * distance
        a_x = k_a * (v_desired - X[3, 0])
        a_z = k_a * (0.0 - X[4, 0])  # Ensure vertical velocity stabilizes at zero

        # Angular velocity control to align orientation
        omega = k_omega * error_theta

        # Convert desired accelerations and angular velocity to force inputs
        u_total = m * np.sqrt(a_x**2 + (a_z + 9.81)**2)
        u_diff = I * omega / r

        # Ensure forces are within physical limits
        u_right = min(max((u_total + u_diff) / 2, 0), f_max)
        u_left = min(max((u_total - u_diff) / 2, 0), f_max)

        return np.array([u_right, u_left]).reshape(-1, 1)
    
    # TODO: fix stop
    def stop(self, X, k_a=1.0, k_theta=1.0):
        """Generate input to decelerate the quadrotor in both linear and angular velocities."""
        # Desired velocities are zero for vx, vz, and theta_dot
        desired_velocity = 0.0

        # Deceleration for x and z velocities
        accel_x = k_a * (desired_velocity - X[3, 0])
        accel_z = k_a * (desired_velocity - X[4, 0])

        # Deceleration for angular velocity theta_dot
        accel_theta = k_theta * (desired_velocity - X[5, 0])

        # Return control inputs as a 3x1 vector (x, z, theta)
        return np.array([9.81/2+accel_z, 9.81/2+accel_z]).reshape(-1, 1) 
    
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