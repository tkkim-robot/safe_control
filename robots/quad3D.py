import numpy as np
import casadi as ca

"""
Created on June 2nd, 2025
@author: Taekyung Kim

@description: 
3D Quad model for MPC-CBF (casadi)
Linearized 6-DOF quadrotor model: "Integration of Adaptive Control and Reinforcement Learning for Real-Time Control and Learning", IEEE T-AC, 2023.
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
            New linearized 6-DOF quadrotor model:
            X: [x, y, z, θ, φ, ψ, vx, vy, vz, q, p, r] (12 states)
            U: [u1, u2, u3, u4] (4 control inputs - motor forces)
            
            Dynamics:
            ẋ = vx, v̇x = g*θ
            ẏ = vy, v̇y = -g*φ  
            ż = vz, v̇z = (1/m)*F
            θ̇ = q, q̇ = (1/Iy)*τy
            φ̇ = p, ṗ = (1/Ix)*τx
            ψ̇ = r, ṙ = (1/Iz)*τz
            
            where [F, τy, τx, τz]^T = B2 * [u1, u2, u3, u4]^T
            B2 = [[1, 1, 1, 1],
                  [0, L, 0, -L], 
                  [L, 0, -L, 0],
                  [ν, -ν, ν, -ν]]
        '''
        self.dt = dt
        self.robot_spec = robot_spec
        
        # Default physical parameters
        self.robot_spec.setdefault('mass', 3.0)  # m
        self.robot_spec.setdefault('Ix', 0.5)    # Moment of inertia around x-axis
        self.robot_spec.setdefault('Iy', 0.5)    # Moment of inertia around y-axis  
        self.robot_spec.setdefault('Iz', 0.5)    # Moment of inertia around z-axis
        self.robot_spec.setdefault('L', 0.3)    # Arm length
        self.robot_spec.setdefault('nu', 0.1)   # Torque coefficient
        
        # Control constraints
        self.robot_spec.setdefault('u_max', 10.0)  # Maximum motor force
        self.robot_spec.setdefault('u_min', -10.0)   # Minimum motor force
        
        # Extract parameters
        self.m = self.robot_spec['mass']
        self.Ix = self.robot_spec['Ix']
        self.Iy = self.robot_spec['Iy']  
        self.Iz = self.robot_spec['Iz']
        self.L = self.robot_spec['L']
        self.nu = self.robot_spec['nu']
        self.gravity = 9.8
        
        # B2 matrix for control allocation
        self.B2 = np.array([
            [1,  1,  1,  1],      # F = u1 + u2 + u3 + u4
            [0,  self.L,  0, -self.L],   # τy = L*u2 - L*u4
            [self.L,  0, -self.L,  0],   # τx = L*u1 - L*u3  
            [self.nu, -self.nu, self.nu, -self.nu]  # τz = ν*u1 - ν*u2 + ν*u3 - ν*u4
        ])
        
        # State space matrices
        self.A = np.zeros((12, 12))
        self.A[0, 6] = 1   # ẋ = vx
        self.A[1, 7] = 1   # ẏ = vy
        self.A[2, 8] = 1   # ż = vz
        self.A[3, 9] = 1   # θ̇ = q
        self.A[4, 10] = 1  # φ̇ = p
        self.A[5, 11] = 1  # ψ̇ = r
        self.A[6, 3] = self.gravity   # v̇x = g*θ
        self.A[7, 4] = -self.gravity  # v̇y = -g*φ
        
        self.B1 = np.zeros((12, 4))
        self.B1[8, 0] = 1/self.m      # v̇z = (1/m)*F
        self.B1[9, 1] = 1/self.Iy     # q̇ = (1/Iy)*τy
        self.B1[10, 2] = 1/self.Ix    # ṗ = (1/Ix)*τx
        self.B1[11, 3] = 1/self.Iz    # ṙ = (1/Iz)*τz
        
        self.B = self.B1 @ self.B2
        
        # For derivative computation
        self.df_dx = self.A
      
    def f(self, X, casadi=False):
        """
        Drift dynamics f(x) = Ax
        X: [x, y, z, θ, φ, ψ, vx, vy, vz, q, p, r]
        """
        if casadi:
            return ca.mtimes(self.A, X)
        else:
            return self.A @ X
    
    def g(self, X, casadi=False):
        """
        Control matrix g(x) = B (constant for linear system)
        """
        if casadi:
            return self.B
        else:
            return self.B
        
    def step(self, X, U, casadi=False): 
        """
        Euler integration: x_{k+1} = x_k + (Ax_k + Bu_k) * dt
        """
        # print with 2 decimal places
        # print(f"roll: {float(X[3, 0]):.2f}, pitch: {float(X[4, 0]):.2f}, yaw: {float(X[5, 0]):.2f}")
        # print(f"roll rate: {float(X[9, 0]):.2f}, pitch rate: {float(X[10, 0]):.2f}, yaw rate: {float(X[11, 0]):.2f}")
        if casadi:
            X_next = X + (ca.mtimes(self.A, X) + ca.mtimes(self.B, U)) * self.dt
            # Normalize angles
            X_next[3, 0] = angle_normalize(X_next[3, 0])  # θ
            X_next[4, 0] = angle_normalize(X_next[4, 0])  # φ  
            X_next[5, 0] = angle_normalize(X_next[5, 0])  # ψ
        else:
            X_next = X + (self.A @ X + self.B @ U) * self.dt
            # Normalize angles
            X_next[3, 0] = angle_normalize(X_next[3, 0])  # θ
            X_next[4, 0] = angle_normalize(X_next[4, 0])  # φ
            X_next[5, 0] = angle_normalize(X_next[5, 0])  # ψ
        return X_next

    def nominal_input(self, X, goal, k_p=1.0, k_d=2.0, k_ang=5.0):
        '''
        Nominal input for CBF-QP
        X: [x, y, z, θ, φ, ψ, vx, vy, vz, q, p, r]
        goal: [x_des, y_des, z_des]
        '''
        u_max = self.robot_spec['u_max']
        u_min = self.robot_spec['u_min']
        
        u_nom = np.zeros([4, 1])
        
        # Position and velocity errors
        pos_err = goal[0:3].reshape(-1, 1) - X[0:3]  # [x, y, z] error
        vel_err = -X[6:9]  # desired velocity is 0
        
        # Desired accelerations
        ax_des = k_p * pos_err[0, 0] + k_d * vel_err[0, 0]
        ay_des = k_p * pos_err[1, 0] + k_d * vel_err[1, 0]  
        az_des = k_p * pos_err[2, 0] + k_d * vel_err[2, 0]
        
        # From linearized dynamics: v̇x = g*θ, v̇y = -g*φ, v̇z = (1/m)*F
        theta_des = ax_des / self.gravity
        phi_des = -ay_des / self.gravity
        F_des = self.m * az_des
        
        # Angular errors and rates
        theta_err = theta_des - X[3, 0]
        phi_err = phi_des - X[4, 0]
        psi_err = 0 - X[5, 0]  # Keep yaw at 0
        
        q_err = -X[9, 0]   # desired angular velocity is 0
        p_err = -X[10, 0]
        r_err = -X[11, 0]
        
        # Desired torques
        tau_y_des = self.Iy * (k_ang * theta_err + k_d * q_err)
        tau_x_des = self.Ix * (k_ang * phi_err + k_d * p_err)  
        tau_z_des = self.Iz * (k_ang * psi_err + k_d * r_err)
        
        # Solve for motor forces: [F, τy, τx, τz]^T = B2 * u
        desired_wrench = np.array([[F_des], [tau_y_des], [tau_x_des], [tau_z_des]])
        u_nom = np.linalg.pinv(self.B2) @ desired_wrench
        
        # Apply constraints
        u_nom = np.clip(u_nom, u_min, u_max)
        
        return u_nom
    
    def stop(self, X, k_stop=1.0):
        """
        Control to stop the quadrotor (minimize velocities)
        """
        u_max = self.robot_spec['u_max']
        u_min = self.robot_spec['u_min']
        
        # Desired accelerations to reduce velocities
        ax_des = -k_stop * X[6, 0]  # reduce vx
        ay_des = -k_stop * X[7, 0]  # reduce vy
        az_des = -k_stop * X[8, 0]  # reduce vz
        
        # Convert to desired angles and force
        theta_des = ax_des / self.gravity
        phi_des = -ay_des / self.gravity
        F_des = self.m * az_des
        
        # Angular control to reach desired angles
        tau_y_des = self.Iy * k_stop * (theta_des - X[3, 0] - X[9, 0]/k_stop)
        tau_x_des = self.Ix * k_stop * (phi_des - X[4, 0] - X[10, 0]/k_stop)
        tau_z_des = self.Iz * k_stop * (0 - X[5, 0] - X[11, 0]/k_stop)  # level yaw
        
        # Solve for motor forces
        desired_wrench = np.array([[F_des], [tau_y_des], [tau_x_des], [tau_z_des]])
        u_stop = np.linalg.pinv(self.B2) @ desired_wrench
        
        # Apply constraints
        u_stop = np.clip(u_stop, u_min, u_max)
        
        return u_stop.flatten()
    
    def has_stopped(self, X, tol=0.05):
        """Check if quadrotor has stopped (low velocities and angular rates)"""
        linear_vel = np.linalg.norm(X[6:9])
        angular_vel = np.linalg.norm(X[9:12])
        return linear_vel < tol and angular_vel < tol

    def rotate_to(self, X, ang_des, k_omega=2.0):
        """
        Rotate to desired yaw angle while maintaining position
        """
        u_max = self.robot_spec['u_max']
        u_min = self.robot_spec['u_min']
        
        # Hover force (compensate gravity)
        F_hover = self.m * self.gravity
        
        # No rotation in roll and pitch
        tau_y_des = self.Iy * k_omega * (0 - X[3, 0] - X[9, 0]/k_omega)   # level pitch
        tau_x_des = self.Ix * k_omega * (0 - X[4, 0] - X[10, 0]/k_omega)  # level roll
        tau_z_des = self.Iz * k_omega * (ang_des - X[5, 0] - X[11, 0]/k_omega)  # desired yaw
        
        # Solve for motor forces
        desired_wrench = np.array([[F_hover], [tau_y_des], [tau_x_des], [tau_z_des]])
        u = np.linalg.pinv(self.B2) @ desired_wrench
        
        # Apply constraints
        u = np.clip(u, u_min, u_max)
        
        return u.flatten()
    
    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''obs: [x, y, r]'''
        '''obstacles are infinite cylinders at x and y with radius r, extending in z direction'''
        '''X : [x, y, z, θ, φ, ψ, vx, vy, vz, q, p, r]'''
        raise NotImplementedError("Cannot implement with nominal distance based CBF")
        
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k, casadi=True)
        x_k2 = self.step(x_k1, u_k, casadi=True)

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
        return h_k, d_h, dd_h