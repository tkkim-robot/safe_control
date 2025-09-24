from robots.kinematic_bicycle2D import KinematicBicycle2D
import numpy as np
import casadi as ca

"""
It is based on the kinematic bicycle 2D model and overrides
only the continous and discrete-time CBF funcitions for Dynamic Parabolic CBF (DPCBF) counterparts:
"""

class KinematicBicycle2D_DPCBF(KinematicBicycle2D):
    def __init__(self, dt, robot_spec, k_lambda=0.1, k_mu=0.5):
        super().__init__(dt, robot_spec)
        self.k_lambda = k_lambda
        self.k_mu = k_mu

    def agent_barrier(self, X, obs, robot_radius, s=1.05):
        """
        '''Continuous Time DPCBF'''
        Compute a Dynamic Parabolic Control Barrier Function for the Kinematic Bicycle2D.
        The barrier's relative degree is "1"
            h_dot = ∂h/∂x ⋅ f(x) + ∂h/∂x ⋅ g(x) ⋅ u
        Define h from the collision cone idea:
            p_rel = [obs_x - x, obs_y - y]
            v_rel = [obs_x_dot-v_cos(theta), obs_y_dot-v_sin(theta)]
            dist = ||p_rel||
            R = robot_radius + obs_r
        """

        theta = X[2, 0]
        v = X[3, 0]

        # Check if obstacles have velocity components (static or moving)
        if obs.shape[0] > 3:
            obs_vel_x = obs[3]
            obs_vel_y = obs[4]

        else:
            obs_vel_x = 0.0
            obs_vel_y = 0.0

        # Combine safety radius (= r_obs + r_rob) with a safe margin (s)
        ego_dim = (obs[2] + robot_radius) * s

        # Compute relative position and velocity
        p_rel = np.array([[obs[0] - X[0, 0]], 
                        [obs[1] - X[1, 0]]])
        v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                        [obs_vel_y - v * np.sin(theta)]])
        # Compute norms
        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)

        p_rel_x = p_rel[0, 0]
        p_rel_y = p_rel[1, 0]

        # Rotation angle and transformation
        rot_angle = np.arctan2(p_rel_y, p_rel_x)
        R = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                    [-np.sin(rot_angle),  np.cos(rot_angle)]])

        # Transform v_rel into the new coordinate frame
        v_rel_new = R @ v_rel
        v_rel_new_x = v_rel_new[0, 0]
        v_rel_new_y = v_rel_new[1, 0]

        # Compute clearance safely
        eps = 1e-6
        d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)

        # DPCBF functions
        func_lamda = self.k_lambda * np.sqrt(d_safe) / v_rel_mag
        func_mu = self.k_mu * np.sqrt(d_safe)

        # Barrier function h(x)
        h = v_rel_new_x + func_lamda * (v_rel_new_y**2) + func_mu

        # Compute dh_dx for DPCBF
        dh_dx = np.zeros((1, 4))
        dh_dx[0, 0] = p_rel_y * v_rel_new_y / p_rel_mag**2 - self.k_lambda * p_rel_x * v_rel_new_y**2 / v_rel_mag / np.sqrt(d_safe) - 2 * self.k_lambda * np.sqrt(d_safe) / v_rel_mag * v_rel_new_y * p_rel_y / p_rel_mag**2 * v_rel_new_x - self.k_mu * p_rel_x / np.sqrt(d_safe)
        dh_dx[0, 1] = - p_rel_x * v_rel_new_y / p_rel_mag**2 - self.k_lambda * p_rel_y * v_rel_new_y**2 / v_rel_mag / np.sqrt(d_safe) + 2 * self.k_lambda * np.sqrt(d_safe) / v_rel_mag * v_rel_new_y * p_rel_x / p_rel_mag**2 * v_rel_new_x - self.k_mu * p_rel_y / np.sqrt(d_safe)
        dh_dx[0, 2] = - v * np.sin(rot_angle-theta) - self.k_lambda * np.sqrt(d_safe) * v * (obs_vel_x * np.sin(theta) - obs_vel_y * np.cos(theta)) * v_rel_new_y**2 / v_rel_mag**3 - 2 * self.k_lambda * np.sqrt(d_safe) * v_rel_new_y * v * np.cos(rot_angle-theta) / v_rel_mag
        dh_dx[0, 3] = - np.cos(rot_angle-theta) - self.k_lambda * np.sqrt(d_safe) / v_rel_mag**3 * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta)) * v_rel_new_y**2 - 2 * self.k_lambda * np.sqrt(d_safe) * v_rel_new_y * np.sin(rot_angle-theta) / v_rel_mag

        return h, dh_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, s=1.05):
        '''Discrete Time DPCBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k, casadi=True)

        def h(x, obs, robot_radius, s=1.05):
            theta = x[2, 0]
            v = x[3, 0]

            # Check if obstacles have velocity components (static or moving)
            if obs.shape[0] > 3:
                obs_vel_x = obs[3]
                obs_vel_y = obs[4]
            else:
                obs_vel_x = 0.0
                obs_vel_y = 0.0

            # Combine radius R
            ego_dim = (obs[2] + robot_radius) * s

            # Compute relative position and velocity
            p_rel = ca.vertcat(obs[0] - x[0, 0], obs[1] - x[1, 0])
            v_rel = ca.vertcat(obs_vel_x - v * ca.cos(theta), obs_vel_y - v * ca.sin(theta))

            # Compute the rotation angle
            rot_angle = ca.atan2(p_rel[1], p_rel[0])

            # Rotation matrix for transforming to the new coordinate frame:
            R = ca.vertcat( 
                ca.horzcat(ca.cos(rot_angle), ca.sin(rot_angle)),
                ca.horzcat(-ca.sin(rot_angle), ca.cos(rot_angle))
            )

            # Transform v_rel into the new coordinate frame
            v_rel_new = ca.mtimes(R, v_rel)

            p_rel_mag = ca.norm_2(p_rel)
            v_rel_mag = ca.norm_2(v_rel)

            eps = 1e-6
            d_safe = ca.fmax(p_rel_mag**2 - ego_dim**2, eps)

            k_lamda, k_mu = 0.1 * ca.sqrt(s**2 - 1)/ego_dim, 0.5 * ca.sqrt(s**2 - 1)/ego_dim

            lamda = k_lamda * ca.sqrt(d_safe) / v_rel_mag
            mu = k_mu * ca.sqrt(d_safe)

            # Compute h
            h = v_rel_new[0] + lamda * v_rel_new[1]**2 + mu

            return h

        h_k1 = h(x_k1, obs, robot_radius, s)
        h_k = h(x_k, obs, robot_radius, s)

        d_h = h_k1 - h_k

        return h_k, d_h
