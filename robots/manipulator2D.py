
import numpy as np
import casadi as ca

class Manipulator2D:
    def __init__(self, dt, robot_spec):
        '''
        State: [theta1, theta2, theta3] (Joint Angles)
        Control: [omega1, omega2, omega3] (Joint Velocities)
        Dynamics: Kinematic (q_dot = u)
        '''
        self.dt = dt
        self.robot_spec = robot_spec
        self.robot_spec.setdefault('model', 'Manipulator2D')
        
        # Parameters from index.html (SCALE = 60)
        # Links: 80, 70, 50 pixels
        scale = 60.0
        self.link_lengths = np.array([80, 70, 50]) / scale
        self.num_links = 3
        
        self.robot_spec.setdefault('w_max', 2.0) 
        self.robot_spec.setdefault('Kp', 3.0) 

        self.base_pos = np.array([0.0, 0.0])

    def f(self, X, casadi=False):
        # Kinematic: f(x) = 0
        if casadi:
            return ca.DM.zeros(3, 1)
        return np.zeros((3, 1))

    def g(self, X, casadi=False):
        # Kinematic: g(x) = I
        if casadi:
            return ca.DM.eye(3)
        return np.eye(3)

    def step(self, X, U):
        # Euler integration
        # X: (3,1), U: (3,1)
        return X + U * self.dt
    
    def get_end_effector(self, X):
        x, y = self.base_pos[0], self.base_pos[1]
        total_angle = 0
        for i in range(self.num_links):
            total_angle += X[i, 0]
            x += self.link_lengths[i] * np.cos(total_angle)
            y += self.link_lengths[i] * np.sin(total_angle)
        return np.array([x, y])

    def get_joint_positions(self, X):
         # Returns list of (x,y) for Base, Joint1, Joint2, EE
         P = [np.copy(self.base_pos)]
         total_angle = 0
         for i in range(self.num_links):
             total_angle += X[i, 0]
             next_p = P[-1] + self.link_lengths[i] * np.array([np.cos(total_angle), np.sin(total_angle)])
             P.append(next_p)
         return P

    def get_jacobian(self, X):
        # Geometric Jacobian for End Effector position
        # J is 2x3. v_ee = J @ q_dot
        J = np.zeros((2, 3))
        
        # Calculate joint positions
        P = [np.copy(self.base_pos)]
        total_angle = 0
        link_angles = []
        for i in range(self.num_links):
            total_angle += X[i, 0]
            link_angles.append(total_angle)
            next_p = P[-1] + self.link_lengths[i] * np.array([np.cos(total_angle), np.sin(total_angle)])
            P.append(next_p)
        
        # EE is P[-1] (which is P[3])
        ee_pos = P[-1]
        
        # J_i column is z_axis x (p_ee - p_joint_i)
        # z_axis is always [0,0,1]. Cross product in 2D is [-dy, dx].
        for i in range(self.num_links):

            dx = ee_pos[0] - P[i][0]
            dy = ee_pos[1] - P[i][1]

            
            jx = 0
            jy = 0
            # Reconstruct angles again to be safe
            angle_sum = 0
            # Forward till i-1
            for k in range(i):
                angle_sum += X[k,0]
                
            for k in range(i, self.num_links):
                angle_sum += X[k, 0]
                jx -= self.link_lengths[k] * np.sin(angle_sum)
                jy += self.link_lengths[k] * np.cos(angle_sum)
                
            J[0, i] = jx
            J[1, i] = jy
            
        return J

    def nominal_input(self, X, G, d_min=0.05):
        # Inverse Kinematics Control
        # G: Goal (x,y)
        ee = self.get_end_effector(X)
        error = G[0:2].flatten() - ee
        
        # Simple P control on EE velocity
        kp = self.robot_spec['Kp']
        v_ee_des = kp * error
        
        J = self.get_jacobian(X)
        omega_des = J.T @ (v_ee_des.reshape(2,1))
        
        # Clamp joint velocities
        w_max = self.robot_spec['w_max']
        omega_des = np.clip(omega_des, -w_max, w_max)
        
        return omega_des

    def get_link_circles(self, X, radius):
        # Returns list of circles (center_x, center_y, radius, link_index)
        circles = []
        
        p_start = np.copy(self.base_pos)
        total_angle = 0
        
        # Discretize links
        step_len = 10.0 / 60.0
        
        for i in range(self.num_links):
            total_angle += X[i, 0]
            dx = self.link_lengths[i] * np.cos(total_angle)
            dy = self.link_lengths[i] * np.sin(total_angle)
            
            p_end = p_start + np.array([dx, dy])
            
            link_dist = self.link_lengths[i]
            num_steps = int(np.ceil(link_dist / step_len))
            
            for j in range(num_steps + 1):
                t = j / num_steps
                pos = p_start + t * np.array([dx, dy])
                circles.append({'x': pos[0], 'y': pos[1], 'r': radius, 'link_idx': i})
                
            p_start = p_end
            
        return circles

    def get_points_jacobian(self, X, point_on_link, link_idx):
        # Compute Jacobian for a specific point on a specific link
        # point_on_link: (x,y) absolute coordinates
        # link_idx: which link this point belongs to (0, 1, 2)
        # Returns J (2x3). Columns > link_idx are zero.
        
        J = np.zeros((2, 3))
        P = [np.copy(self.base_pos)]
        total_angle = 0
        
        # Calculate joint positions P[0]...P[link_idx]
        for i in range(link_idx + 1):
            if i > 0:
                total_angle += X[i-1, 0]
                P.append(P[-1] + self.link_lengths[i-1] * np.array([np.cos(total_angle), np.sin(total_angle)]))
                
        # J_k = z x (p - p_joint_k)
        for k in range(link_idx + 1):
            # Joint k position is P[k]
            # (Note: P[0] is base. Joint 0 rotates Link 0 relative to base)
            
            dx_pt = point_on_link[0] - P[k][0]
            dy_pt = point_on_link[1] - P[k][1]
            
            # Cross product (0,0,1) x (dx, dy, 0) = (-dy, dx, 0)
            J[0, k] = -dy_pt
            J[1, k] = dx_pt
            
        return J


    def agent_barrier(self, X, obs, robot_radius, beta=1.3):
         h_list = []
         dh_dx_list = []
         
         link_circles = self.get_link_circles(X, radius=robot_radius)
         
         # Obs unpacking
         # obs shape: assumes list/array like [x, y, r, ...]
         ox = obs[0]
         oy = obs[1]
         orad = obs[2]
         
         for circle in link_circles:
             cx, cy = circle['x'], circle['y']
             link_idx = circle['link_idx']
             
             # d_min: sum of radii
             d_min = robot_radius + orad
             
             dx = cx - ox
             dy = cy - oy
             dist_sq = dx*dx + dy*dy

             # CBF h
             h = dist_sq - beta * (d_min**2)
             
             # Jacobian for this circle center
             J_p = self.get_points_jacobian(X, [cx, cy], link_idx)
             
             # dh/dx = 2 * (p - obs).T @ dp/dq
             # dp/dq = J_p
             # dh_dx (w.r.t q) = 2 * [dx, dy] @ J_p
             
             dh_dx = 2 * np.array([dx, dy]) @ J_p
             
             h_list.append(h)
             dh_dx_list.append(dh_dx.flatten())
             
         return h_list, dh_dx_list
