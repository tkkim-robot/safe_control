"""
Dynamic Bicycle Model for drift simulation.

This model captures the full nonlinear dynamics of a vehicle including
tire forces with the Fiala brush tire model. Suitable for high-slip
(drifting) conditions.

State: x = [r, beta, V, delta, tau]^T
    - r: yaw rate [rad/s]
    - beta: side slip angle [rad]
    - V: velocity magnitude [m/s]
    - delta: steering angle [rad]
    - tau: rear wheel torque [Nm]

Input: u = [delta_dot, tau_dot]^T
    - delta_dot: steering rate [rad/s]
    - tau_dot: torque rate [Nm/s]

Reference: Based on dynamic bicycle model with Fiala tire model
"""

import numpy as np
import casadi as ca
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt


def angle_normalize(x):
    """Normalize angle to [-pi, pi]."""
    if isinstance(x, (np.ndarray, float, int)):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


class DynamicBicycle2D:
    """
    Dynamic Bicycle Model with Fiala tire model.
    
    This model is suitable for high-slip maneuvers like drifting.
    """
    
    def __init__(self, dt, robot_spec):
        """
        Initialize the dynamic bicycle model.
        
        Args:
            dt: Time step [s]
            robot_spec: Dictionary with vehicle specifications
        """
        self.dt = dt
        self.robot_spec = robot_spec
        
        # Vehicle geometry
        self.robot_spec.setdefault('a', 1.6)  # Front axle to CG [m]
        self.robot_spec.setdefault('b', 0.8)  # Rear axle to CG [m]
        self.robot_spec.setdefault('wheel_base', self.robot_spec['a'] + self.robot_spec['b'])
        
        # Mass and inertia
        self.robot_spec.setdefault('m', 1500.0)  # Vehicle mass [kg]
        self.robot_spec.setdefault('Iz', 2500.0)  # Yaw moment of inertia [kg*m^2]
        
        # Tire parameters
        self.robot_spec.setdefault('Cc_f', 80000.0)  # Front cornering stiffness [N/rad]
        self.robot_spec.setdefault('Cc_r', 120000.0)  # Rear cornering stiffness [N/rad]
        self.robot_spec.setdefault('mu', 1.0)  # Friction coefficient
        self.robot_spec.setdefault('r_w', 0.3)  # Wheel radius [m]
        self.robot_spec.setdefault('gamma', 0.99)  # Numeric stability parameter
        
        # Input limits
        self.robot_spec.setdefault('delta_max', np.deg2rad(35))  # Max steering angle [rad]
        self.robot_spec.setdefault('delta_dot_max', np.deg2rad(60))  # Max steering rate [rad/s]
        self.robot_spec.setdefault('tau_max', 5000.0)  # Max torque [Nm]
        self.robot_spec.setdefault('tau_dot_max', 10000.0)  # Max torque rate [Nm/s]
        
        # State limits
        self.robot_spec.setdefault('v_max', 30.0)  # Max velocity [m/s]
        self.robot_spec.setdefault('v_min', 0.5)  # Min velocity [m/s]
        self.robot_spec.setdefault('r_max', 2.0)  # Max yaw rate [rad/s]
        self.robot_spec.setdefault('beta_max', np.deg2rad(60))  # Max slip angle [rad]
        
        # Visualization parameters
        self.robot_spec.setdefault('body_length', 4.3)
        self.robot_spec.setdefault('body_width', 1.8)
        self.robot_spec.setdefault('front_ax_dist', self.robot_spec['a'])
        self.robot_spec.setdefault('rear_ax_dist', self.robot_spec['b'])
        self.robot_spec.setdefault('radius', 1.2)
        
        # Gravity
        self.gravity = 9.81
        
        # Compute static normal forces
        self._compute_normal_forces()
        
    def _compute_normal_forces(self):
        """Compute static normal forces on front and rear axles."""
        m = self.robot_spec['m']
        a = self.robot_spec['a']
        b = self.robot_spec['b']
        L = a + b
        
        self.Fz_f = m * self.gravity * b / L  # Front normal force
        self.Fz_r = m * self.gravity * a / L  # Rear normal force
        
    def _compute_slip_angles(self, r, beta, V, delta, casadi=False):
        """
        Compute front and rear slip angles.
        
        Args:
            r: Yaw rate [rad/s]
            beta: Side slip angle [rad]
            V: Velocity magnitude [m/s]
            delta: Steering angle [rad]
            casadi: Use CasADi for symbolic computation
            
        Returns:
            alpha_f, alpha_r: Front and rear slip angles [rad]
        """
        a = self.robot_spec['a']
        b = self.robot_spec['b']
        
        if casadi:
            # Prevent division by zero
            V_safe = ca.fmax(V, 0.1)
            
            # Front slip angle
            alpha_f = ca.atan2(V * ca.sin(beta) + a * r, V_safe * ca.cos(beta)) - delta
            
            # Rear slip angle
            alpha_r = ca.atan2(V * ca.sin(beta) - b * r, V_safe * ca.cos(beta))
        else:
            # Prevent division by zero
            V_safe = max(V, 0.1)
            
            # Front slip angle
            alpha_f = np.arctan2(V * np.sin(beta) + a * r, V_safe * np.cos(beta)) - delta
            
            # Rear slip angle
            alpha_r = np.arctan2(V * np.sin(beta) - b * r, V_safe * np.cos(beta))
            
        return alpha_f, alpha_r
    
    def _compute_lateral_force(self, alpha, Cc, Fz, Fx, casadi=False):
        """
        Compute lateral tire force using Fiala brush model.
        
        Args:
            alpha: Slip angle [rad]
            Cc: Cornering stiffness [N/rad]
            Fz: Normal force [N]
            Fx: Longitudinal force [N]
            casadi: Use CasADi for symbolic computation
            
        Returns:
            Fy: Lateral force [N]
        """
        mu = self.robot_spec['mu']
        gamma = self.robot_spec['gamma']
        
        if casadi:
            # Maximum lateral force
            Fy_max = ca.sqrt(ca.fmax((mu * Fz)**2 - gamma * Fx**2, 1.0))
            
            # Slip angle at which tire starts sliding
            alpha_sl = ca.atan(3 * Fy_max / Cc)
            
            # Tire force computation (Fiala model)
            tan_alpha = ca.tan(alpha)
            
            # Linear region force
            Fy_linear = (-Cc * tan_alpha 
                        + (Cc**2 / (3 * Fy_max)) * ca.fabs(tan_alpha) * tan_alpha
                        - (Cc**3 / (27 * Fy_max**2)) * tan_alpha**3)
            
            # Saturated force
            Fy_saturated = -Fy_max * ca.sign(alpha)
            
            # Switch between linear and saturated regions
            Fy = ca.if_else(ca.fabs(alpha) < alpha_sl, Fy_linear, Fy_saturated)
        else:
            # Maximum lateral force
            Fy_max_sq = (mu * Fz)**2 - gamma * Fx**2
            Fy_max = np.sqrt(max(Fy_max_sq, 1.0))
            
            # Slip angle at which tire starts sliding
            alpha_sl = np.arctan(3 * Fy_max / Cc)
            
            tan_alpha = np.tan(alpha)
            
            if abs(alpha) < alpha_sl:
                # Linear region (Fiala brush model)
                Fy = (-Cc * tan_alpha 
                      + (Cc**2 / (3 * Fy_max)) * abs(tan_alpha) * tan_alpha
                      - (Cc**3 / (27 * Fy_max**2)) * tan_alpha**3)
            else:
                # Saturated region
                Fy = -Fy_max * np.sign(alpha)
                
        return Fy
    
    def _compute_tire_forces(self, r, beta, V, delta, tau, casadi=False):
        """
        Compute all tire forces.
        
        Args:
            r, beta, V, delta, tau: Vehicle states
            casadi: Use CasADi for symbolic computation
            
        Returns:
            Fx_f, Fy_f, Fx_r, Fy_r: Tire forces [N]
        """
        r_w = self.robot_spec['r_w']
        
        # Slip angles
        alpha_f, alpha_r = self._compute_slip_angles(r, beta, V, delta, casadi)
        
        # Longitudinal forces (rear wheel drive)
        Fx_f = 0.0
        Fx_r = tau / r_w
        
        # Lateral forces
        Fy_f = self._compute_lateral_force(alpha_f, self.robot_spec['Cc_f'], 
                                           self.Fz_f, Fx_f, casadi)
        Fy_r = self._compute_lateral_force(alpha_r, self.robot_spec['Cc_r'], 
                                           self.Fz_r, Fx_r, casadi)
        
        return Fx_f, Fy_f, Fx_r, Fy_r
        
    def f(self, X, casadi=False):
        """
        Compute drift dynamics f(x).
        
        State: [r, beta, V, delta, tau]^T
        
        Args:
            X: State vector (5x1)
            casadi: Use CasADi for symbolic computation
            
        Returns:
            f(x): Drift term (5x1)
        """
        r = X[0, 0]
        beta = X[1, 0]
        V = X[2, 0]
        delta = X[3, 0]
        tau = X[4, 0]
        
        a = self.robot_spec['a']
        b = self.robot_spec['b']
        m = self.robot_spec['m']
        Iz = self.robot_spec['Iz']
        
        # Get tire forces
        Fx_f, Fy_f, Fx_r, Fy_r = self._compute_tire_forces(r, beta, V, delta, tau, casadi)
        
        if casadi:
            V_safe = ca.fmax(V, 0.1)
            
            # Yaw acceleration
            r_dot = (a * (Fx_f * ca.sin(delta) + Fy_f * ca.cos(delta)) - b * Fy_r) / Iz
            
            # Side slip rate
            beta_dot = ((Fx_f * ca.sin(delta - beta) + Fy_f * ca.cos(delta - beta) 
                        - Fx_r * ca.sin(beta) + Fy_r * ca.cos(beta)) / (m * V_safe) - r)
            
            # Velocity rate
            V_dot = ((Fx_f * ca.cos(delta - beta) - Fy_f * ca.sin(delta - beta)
                     + Fx_r * ca.cos(beta) + Fy_r * ca.sin(beta)) / m)
            
            return ca.vertcat(r_dot, beta_dot, V_dot, 0, 0)
        else:
            V_safe = max(V, 0.1)
            
            # Yaw acceleration
            r_dot = (a * (Fx_f * np.sin(delta) + Fy_f * np.cos(delta)) - b * Fy_r) / Iz
            
            # Side slip rate
            beta_dot = ((Fx_f * np.sin(delta - beta) + Fy_f * np.cos(delta - beta)
                        - Fx_r * np.sin(beta) + Fy_r * np.cos(beta)) / (m * V_safe) - r)
            
            # Velocity rate
            V_dot = ((Fx_f * np.cos(delta - beta) - Fy_f * np.sin(delta - beta)
                     + Fx_r * np.cos(beta) + Fy_r * np.sin(beta)) / m)
            
            return np.array([[r_dot], [beta_dot], [V_dot], [0], [0]])
    
    def g(self, X, casadi=False):
        """
        Compute input matrix g(x).
        
        Args:
            X: State vector (5x1)
            casadi: Use CasADi for symbolic computation
            
        Returns:
            g(x): Input matrix (5x2)
        """
        if casadi:
            g = ca.SX.zeros(5, 2)
            g[3, 0] = 1  # delta_dot
            g[4, 1] = 1  # tau_dot
            return g
        else:
            return np.array([
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1]
            ])
    
    def step(self, X, U, use_casadi=False):
        """
        Step the dynamics forward by dt using Euler integration.
        
        Args:
            X: State [r, beta, V, delta, tau]^T (5x1)
            U: Input [delta_dot, tau_dot]^T (2x1)
            use_casadi: Use CasADi for symbolic computation
            
        Returns:
            X_next: Next state (5x1)
        """
        # Euler integration
        X_next = X + (self.f(X, use_casadi) + self.g(X, use_casadi) @ U) * self.dt
        
        # Apply state constraints
        if use_casadi:
            # Clamp states
            X_next[0, 0] = ca.fmax(ca.fmin(X_next[0, 0], self.robot_spec['r_max']), 
                                   -self.robot_spec['r_max'])
            X_next[1, 0] = ca.fmax(ca.fmin(X_next[1, 0], self.robot_spec['beta_max']), 
                                   -self.robot_spec['beta_max'])
            X_next[2, 0] = ca.fmax(ca.fmin(X_next[2, 0], self.robot_spec['v_max']), 
                                   self.robot_spec['v_min'])
            X_next[3, 0] = ca.fmax(ca.fmin(X_next[3, 0], self.robot_spec['delta_max']), 
                                   -self.robot_spec['delta_max'])
            X_next[4, 0] = ca.fmax(ca.fmin(X_next[4, 0], self.robot_spec['tau_max']), 
                                   -self.robot_spec['tau_max'])
        else:
            # Clamp states
            X_next[0, 0] = np.clip(X_next[0, 0], -self.robot_spec['r_max'], 
                                   self.robot_spec['r_max'])
            X_next[1, 0] = np.clip(X_next[1, 0], -self.robot_spec['beta_max'], 
                                   self.robot_spec['beta_max'])
            X_next[2, 0] = np.clip(X_next[2, 0], self.robot_spec['v_min'], 
                                   self.robot_spec['v_max'])
            X_next[3, 0] = np.clip(X_next[3, 0], -self.robot_spec['delta_max'], 
                                   self.robot_spec['delta_max'])
            X_next[4, 0] = np.clip(X_next[4, 0], -self.robot_spec['tau_max'], 
                                   self.robot_spec['tau_max'])
        
        return X_next
    
    def nominal_input(self, X, G, d_min=0.5, k_delta=1.0, k_tau=500.0):
        """
        Compute nominal input to reach goal (simple proportional control).
        
        Args:
            X: Current state [r, beta, V, delta, tau]^T
            G: Goal position [x_goal, y_goal] (requires global position)
            d_min: Minimum distance threshold
            k_delta: Steering gain
            k_tau: Torque gain
            
        Returns:
            U: Control input [delta_dot, tau_dot]^T
        """
        # For nominal input, we need the global position which is tracked externally
        # This is a placeholder - actual implementation depends on tracking state
        return np.array([[0.0], [0.0]])
    
    def stop(self, X):
        """Return control input to stop the vehicle."""
        # Reduce torque to bring velocity down
        tau = X[4, 0]
        tau_dot = -np.sign(tau) * self.robot_spec['tau_dot_max'] * 0.5
        
        # Center steering
        delta = X[3, 0]
        delta_dot = -np.sign(delta) * self.robot_spec['delta_dot_max'] * 0.5
        
        return np.array([[delta_dot], [tau_dot]])
    
    def has_stopped(self, X, tol=0.5):
        """Check if vehicle has stopped."""
        return abs(X[2, 0]) < tol
    
    def get_global_velocity(self, X):
        """
        Get velocity in global frame from body-fixed states.
        
        Args:
            X: State [r, beta, V, delta, tau]^T
            
        Returns:
            vx, vy: Velocity components in body frame
        """
        V = X[2, 0]
        beta = X[1, 0]
        
        vx = V * np.cos(beta)  # Longitudinal velocity
        vy = V * np.sin(beta)  # Lateral velocity
        
        return vx, vy
    
    def render_rigid_body(self, X_global, U):
        """
        Return transforms for rendering the vehicle.
        
        Args:
            X_global: Global state [x, y, theta, r, beta, V, delta, tau]
            U: Control input
            
        Returns:
            Transforms for body and wheels
        """
        x, y, theta = X_global[0], X_global[1], X_global[2]
        delta = X_global[6] if len(X_global) > 6 else 0
        
        # Body transform
        transform_body = Affine2D().rotate(theta).translate(x, y) + plt.gca().transData
        
        # Wheel positions
        a = self.robot_spec['a']
        b = self.robot_spec['b']
        
        rear_x = x - b * np.cos(theta)
        rear_y = y - b * np.sin(theta)
        front_x = x + a * np.cos(theta)
        front_y = y + a * np.sin(theta)
        
        # Rear wheel transform
        transform_rear = Affine2D().rotate(theta).translate(rear_x, rear_y) + plt.gca().transData
        
        # Front wheel transform (with steering)
        transform_front = Affine2D().rotate(theta + delta).translate(front_x, front_y) + plt.gca().transData
        
        return transform_body, transform_rear, transform_front


# Alias for compatibility
DynamicBicycle = DynamicBicycle2D

