"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Backup Controller - Abstract base class and implementations for backup control strategies.
These controllers provide alternative control behaviors that can be used for safety guarantees
or emergency maneuvers. The controllers are designed to be simple feedback controllers
that can be forward-simulated to predict trajectories.

@required-scripts: None (standalone module)
"""

import numpy as np
from abc import ABC, abstractmethod


def angle_normalize(x):
    """Normalize angle to [-pi, pi]."""
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class BackupController(ABC):
    """
    Abstract base class for backup controllers.
    
    Backup controllers are simple feedback controllers that can be used to
    predict closed-loop trajectories for safety analysis or emergency maneuvers.
    """
    
    def __init__(self, robot_spec, dt):
        """
        Initialize the backup controller.
        
        Args:
            robot_spec: Dictionary with robot specifications
            dt: Time step for simulation
        """
        self.robot_spec = robot_spec
        self.dt = dt
        
    @abstractmethod
    def compute_control(self, state, target):
        """
        Compute control input given current state and target.
        
        Args:
            state: Current state vector (model-specific)
            target: Target for the controller (behavior-specific)
            
        Returns:
            Control input vector
        """
        pass
    
    @abstractmethod
    def simulate_trajectory(self, initial_state, target, horizon, friction=1.0):
        """
        Forward simulate the closed-loop trajectory.
        
        Args:
            initial_state: Initial state vector
            target: Target for the controller
            horizon: Number of steps to simulate
            friction: Friction coefficient for simulation
            
        Returns:
            trajectory: Array of states over the horizon (n_states x horizon)
        """
        pass
    
    def get_behavior_name(self):
        """Return the name of the backup behavior."""
        return self.__class__.__name__


class LaneChangeController(BackupController):
    """
    Lane change backup controller using cascaded PD control.
    
    This controller steers the vehicle to change to a target lane Y position
    and stabilize there. Uses a cascaded control structure:
    1. Outer loop: Lateral position (y) → desired heading (theta_des)
    2. Inner loop: Heading error → desired steering angle (delta_des)
    3. Actuator loop: Steering error → steering rate (delta_dot)
    """
    
    def __init__(self, robot_spec, dt, direction='left'):
        """
        Initialize the lane change controller.
        
        Args:
            robot_spec: Dictionary with robot specifications
            dt: Time step for simulation
            direction: 'left' or 'right' lane change direction
        """
        super().__init__(robot_spec, dt)
        if self.robot_spec.get('model') != 'DriftingCar':
            raise NotImplementedError("LaneChangeController is only implemented for DriftingCar model.")
        self.direction = direction
        
        # Cascaded control gains (tuned for smooth lane change)
        # Outer loop: lateral position -> desired heading
        self.Kp_y = 0.15      # Proportional gain for lateral error -> heading
        self.Kd_y = 0.8       # Derivative gain (using current heading as proxy for lateral velocity)
        
        # Inner loop: heading -> desired steering
        self.Kp_theta = 1.5   # Proportional gain for heading error -> steering
        self.Kd_theta = 0.3   # Derivative gain (using yaw rate)
        
        # Actuator loop: steering -> steering rate
        self.Kp_delta = 3.0   # Proportional gain for steering error -> steering rate
        
        # Velocity control gains
        self.Kp_v = 500.0     # Proportional gain for velocity tracking
        self.target_velocity = robot_spec.get('v_ref', 8.0)  # Target velocity during lane change
        
        # Limits
        self.delta_max = robot_spec.get('delta_max', np.deg2rad(20))
        self.delta_dot_max = robot_spec.get('delta_dot_max', np.deg2rad(15))
        self.tau_max = robot_spec.get('tau_max', 4000.0)
        self.tau_dot_max = robot_spec.get('tau_dot_max', 8000.0)
        
        # Maximum desired heading during lane change (limits aggressiveness)
        self.theta_des_max = np.deg2rad(15)  # Max 15 degrees heading toward target lane
    
    def compute_control(self, state, target_y):
        """
        Compute control input for lane change using cascaded PD control.
        
        The controller stabilizes the vehicle at the target_y position:
        - When far from target: steer toward target lane
        - When close to target: straighten out and maintain lane
        
        Args:
            state: Current state [x, y, theta, r, beta, V, delta, tau]
            target_y: Target lateral position (lane center Y coordinate)
            
        Returns:
            Control input [delta_dot, tau_dot]
        """
        # Extract state
        x, y, theta, r, beta, V, delta, tau = state.flatten()
        
        # Ensure minimum velocity for stability
        V = max(V, 0.1)
        
        # ===== Outer loop: Lateral position control =====
        # Lateral error (positive = target is above current position)
        y_error = target_y - y
        
        # Desired heading angle to reach target lane
        # Use arctan with a gain to limit maximum heading angle
        # As y_error -> 0, theta_des -> 0 (straighten out)
        theta_des = np.arctan(self.Kp_y * y_error)
        
        # Limit maximum desired heading
        theta_des = np.clip(theta_des, -self.theta_des_max, self.theta_des_max)
        
        # ===== Inner loop: Heading control =====
        # Heading error (how far are we from desired heading)
        theta_error = angle_normalize(theta_des - theta)
        
        # Desired steering angle: proportional to heading error + damping from yaw rate
        # Negative yaw rate (r) means turning right, so we subtract it for damping
        delta_des = self.Kp_theta * theta_error - self.Kd_theta * r
        
        # Limit steering angle
        delta_des = np.clip(delta_des, -self.delta_max, self.delta_max)
        
        # ===== Actuator loop: Steering rate control =====
        # Steering error
        delta_error = delta_des - delta
        
        # Steering rate
        delta_dot = self.Kp_delta * delta_error
        delta_dot = np.clip(delta_dot, -self.delta_dot_max, self.delta_dot_max)
        
        # ===== Velocity control =====
        # Maintain target velocity during lane change
        V_error = self.target_velocity - V
        tau_des = self.Kp_v * V_error
        tau_des = np.clip(tau_des, -self.tau_max, self.tau_max)
        
        # Torque rate
        tau_error = tau_des - tau
        tau_dot = 2000.0 * np.sign(tau_error) * min(abs(tau_error), 1.0)
        tau_dot = np.clip(tau_dot, -self.tau_dot_max, self.tau_dot_max)
        
        return np.array([[delta_dot], [tau_dot]])
    
    def simulate_trajectory(self, initial_state, target_y, horizon, friction=1.0):
        """
        Forward simulate the lane change trajectory.
        
        Args:
            initial_state: Initial state [x, y, theta, r, beta, V, delta, tau]
            target_y: Target lateral position (lane center Y coordinate)
            horizon: Number of steps to simulate
            friction: Friction coefficient for simulation
            
        Returns:
            trajectory: Array of states over the horizon (8 x horizon+1)
        """
        # Import dynamics model
        try:
            from robots.dynamic_bicycle2D import DynamicBicycle2D
        except ImportError:
            from safe_control.robots.dynamic_bicycle2D import DynamicBicycle2D
        
        # Create dynamics model with given friction
        sim_spec = self.robot_spec.copy()
        sim_spec['mu'] = friction
        dynamics = DynamicBicycle2D(self.dt, sim_spec)
        
        # Initialize trajectory storage
        state = initial_state.copy().reshape(-1, 1)
        trajectory = np.zeros((8, horizon + 1))
        trajectory[:, 0] = state.flatten()
        
        for i in range(horizon):
            # Compute control
            U = self.compute_control(state, target_y)
            
            # Extract dynamics state [r, beta, V, delta, tau]
            X_dyn = state[3:8, :]
            
            # Step dynamics
            X_dyn_next = dynamics.step(X_dyn, U)
            
            # Update global state
            theta = state[2, 0]
            V = X_dyn_next[2, 0]
            beta = X_dyn_next[1, 0]
            r = X_dyn_next[0, 0]
            
            # Velocity in global frame
            vx_global = V * np.cos(theta + beta)
            vy_global = V * np.sin(theta + beta)
            
            # Update state
            state[0, 0] += vx_global * self.dt
            state[1, 0] += vy_global * self.dt
            state[2, 0] += r * self.dt
            state[2, 0] = angle_normalize(state[2, 0])
            state[3:8, :] = X_dyn_next
            
            trajectory[:, i + 1] = state.flatten()
        
        return trajectory
    
    def get_behavior_name(self):
        return f"LaneChange_{self.direction}"


class StoppingController(BackupController):
    """
    Stopping backup controller using PD control.
    
    This controller brings the vehicle to a complete stop by:
    1. Applying negative torque proportional to velocity (braking)
    2. Centering the steering wheel to maintain straight heading
    3. Holding position once stopped
    """
    
    def __init__(self, robot_spec, dt):
        """
        Initialize the stopping controller.
        
        Args:
            robot_spec: Dictionary with robot specifications
            dt: Time step for simulation
        """
        super().__init__(robot_spec, dt)
        if self.robot_spec.get('model') != 'DriftingCar':
            raise NotImplementedError("StoppingController is only implemented for DriftingCar model.")

        
        # Braking control gains
        self.Kp_v = 1000.0     # Proportional gain for velocity -> torque (braking)
        
        # Steering control gains (to straighten out)
        self.Kp_theta = 2.0    # Proportional gain for heading correction
        self.Kd_theta = 0.5    # Derivative gain (yaw rate damping)
        self.Kp_delta = 3.0    # Proportional gain for steering rate
        
        # Limits
        self.delta_max = robot_spec.get('delta_max', np.deg2rad(20))
        self.delta_dot_max = robot_spec.get('delta_dot_max', np.deg2rad(15))
        self.tau_max = robot_spec.get('tau_max', 4000.0)
        self.tau_dot_max = robot_spec.get('tau_dot_max', 8000.0)
        
        # Velocity threshold for "stopped" (very small to ensure complete stop)
        self.stop_velocity_threshold = 0.05  # m/s
        
        # Holding torque to keep vehicle stationary after stopping
        self.holding_torque = -100.0  # Small negative torque to resist rolling
    
    def compute_control(self, state, target=None):
        """
        Compute control input for stopping.
        
        Args:
            state: Current state [x, y, theta, r, beta, V, delta, tau]
            target: Not used for stopping controller (can be None)
            
        Returns:
            Control input [delta_dot, tau_dot]
        """
        # Extract state
        x, y, theta, r, beta, V, delta, tau = state.flatten()
        
        # ===== Braking control =====
        # Target velocity is 0
        # Apply negative torque proportional to current velocity until stopped
        if V > self.stop_velocity_threshold:
            # Still moving - apply strong braking proportional to velocity
            # Use maximum braking at high speeds, proportional at low speeds
            tau_des = -self.Kp_v * V
            # Ensure we're always applying significant braking when moving
            tau_des = min(tau_des, -500.0)  # At least -500 Nm when moving
        else:
            # Stopped - apply small holding torque to prevent rolling
            # This keeps the vehicle stationary
            tau_des = self.holding_torque
        
        # Clamp desired torque
        tau_des = np.clip(tau_des, -self.tau_max, self.tau_max)
        
        # Torque rate to reach desired torque - fast response for emergency braking
        tau_error = tau_des - tau
        tau_dot = 5000.0 * np.sign(tau_error) * min(abs(tau_error) / 50.0, 1.0)
        tau_dot = np.clip(tau_dot, -self.tau_dot_max, self.tau_dot_max)
        
        # ===== Steering control =====
        # Try to straighten out (theta -> 0 is not always desired, 
        # but we want to reduce yaw rate and center steering)
        
        # Desired steering: reduce yaw rate, center steering
        delta_des = -self.Kd_theta * r  # Damping on yaw rate
        delta_des = np.clip(delta_des, -self.delta_max, self.delta_max)
        
        # Steering rate
        delta_error = delta_des - delta
        delta_dot = self.Kp_delta * delta_error
        delta_dot = np.clip(delta_dot, -self.delta_dot_max, self.delta_dot_max)
        
        return np.array([[delta_dot], [tau_dot]])
    
    def simulate_trajectory(self, initial_state, target, horizon, friction=1.0):
        """
        Forward simulate the stopping trajectory.
        
        Args:
            initial_state: Initial state [x, y, theta, r, beta, V, delta, tau]
            target: Not used (can be None)
            horizon: Number of steps to simulate
            friction: Friction coefficient for simulation
            
        Returns:
            trajectory: Array of states over the horizon (8 x horizon+1)
        """
        # Import dynamics model
        try:
            from robots.dynamic_bicycle2D import DynamicBicycle2D
        except ImportError:
            from safe_control.robots.dynamic_bicycle2D import DynamicBicycle2D
        
        # Create dynamics model with given friction
        sim_spec = self.robot_spec.copy()
        sim_spec['mu'] = friction
        dynamics = DynamicBicycle2D(self.dt, sim_spec)
        
        # Initialize trajectory storage
        state = initial_state.copy().reshape(-1, 1)
        trajectory = np.zeros((8, horizon + 1))
        trajectory[:, 0] = state.flatten()
        
        for i in range(horizon):
            # Compute control
            U = self.compute_control(state, target)
            
            # Extract dynamics state [r, beta, V, delta, tau]
            X_dyn = state[3:8, :]
            
            # Step dynamics
            X_dyn_next = dynamics.step(X_dyn, U)
            
            # Update global state
            theta = state[2, 0]
            V = X_dyn_next[2, 0]
            beta = X_dyn_next[1, 0]
            r = X_dyn_next[0, 0]
            
            # Velocity in global frame
            vx_global = V * np.cos(theta + beta)
            vy_global = V * np.sin(theta + beta)
            
            # Update state
            state[0, 0] += vx_global * self.dt
            state[1, 0] += vy_global * self.dt
            state[2, 0] += r * self.dt
            state[2, 0] = angle_normalize(state[2, 0])
            state[3:8, :] = X_dyn_next
            
            trajectory[:, i + 1] = state.flatten()
        
        return trajectory
    
    def get_behavior_name(self):
        return "Stopping"


class EvadeBackupController(BackupController):
    """
    Evade backup controller for double integrator model.
    
    This controller steers the robot to the center of a safe pocket using PD control.
    The evade_bullet_bill scenario uses this to hide from a fast-moving obstacle.
    Uses two-phase control:
    1. First, move toward the safe x-range if outside it
    2. Then, move toward the safe y position (into the pocket)
    """
    
    def __init__(self, robot_spec, dt, safe_pocket_center, safe_pocket_bounds, goal_bounds=None):
        """
        Initialize the evade backup controller.
        
        Args:
            robot_spec: Dictionary with robot specifications (for DoubleIntegrator2D)
            dt: Time step for simulation
            safe_pocket_center: [x, y] center of the safe pocket
            safe_pocket_bounds: Dict with x_min, x_max, y_min, y_max
            goal_bounds: Optional dict with goal zone bounds. If provided, staying in goal is safe.
        """
        super().__init__(robot_spec, dt)
        
        self.safe_center = np.array(safe_pocket_center).flatten()
        self.safe_bounds = safe_pocket_bounds
        self.goal_bounds = goal_bounds
        
        # PD control gains
        self.Kp = 2.0  # Position gain
        self.Kd = 2.0  # Velocity damping
        
        # Limits
        self.a_max = robot_spec.get('a_max', 1.0)
        self.v_max = robot_spec.get('v_max', 1.0)
        
    def compute_control(self, state, target=None):
        """
        Compute control input to reach safe pocket.
        
        The controller uses a multi-phase approach:
        1. If in GOAL ZONE (and goal_bounds provided): stay put
        2. If already inside safe pocket: stay put (zero velocity)
        3. If within safe x-range: move vertically into the pocket
        4. If outside safe x-range: move along center line (y=0) toward pocket x-range
        
        Args:
            state: Current state [x, y, vx, vy]
            target: Not used (safe pocket is set in constructor)
            
        Returns:
            Control input [ax, ay]
        """
        state = np.array(state).flatten()
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        
        # Check priority 1: Goal Zone
        if self.goal_bounds is not None:
            # Check if in goal zone
            if (self.goal_bounds['x_min'] <= x <= self.goal_bounds['x_max'] and 
                self.goal_bounds['y_min'] <= y <= self.goal_bounds['y_max']):
                # In goal zone - brake to stop
                ax = -self.Kd * vx
                ay = -self.Kd * vy
                
                # Clamp and return
                a_mag = np.sqrt(ax**2 + ay**2)
                if a_mag > self.a_max:
                    ax = ax * self.a_max / a_mag
                    ay = ay * self.a_max / a_mag
                return np.array([[ax], [ay]])
        
        # Safe pocket bounds
        x_min = self.safe_bounds['x_min']
        x_max = self.safe_bounds['x_max']
        y_min = self.safe_bounds['y_min']  # Bottom of pocket (top of hallway)
        y_max = self.safe_bounds['y_max']  # Top of pocket
        
        pocket_center_y = self.safe_center[1]
        pocket_center_x = self.safe_center[0]
        
        # Margin for "inside pocket" check - ensure robot is fully inside
        margin = self.robot_spec.get('radius', 0.5) + 0.1
        
        # Phase 2: Check if inside safe pocket - stay put
        if (x_min + margin <= x <= x_max - margin and 
            y_min + margin <= y <= y_max - margin):
            # Inside pocket - brake to stop
            ax = -self.Kd * vx
            ay = -self.Kd * vy
        
        # Phase 3: Approach pocket carefully
        # Strategy: Align X with pocket center first, THEN enter Y
        # This prevents "cutting corners" and hitting the wall
        elif x_min - 2.0 <= x <= x_max + 2.0:
            # We are near the pocket x-range
            
            # Check if we are safe to enter Y (strictly within x bounds with margin)
            safe_x_entry = (x_min + margin <= x <= x_max - margin)
            
            if safe_x_entry:
                # X is safe, now we can move Y into pocket
                # Align X to center, Move Y to center
                error_x = pocket_center_x - x
                error_y = pocket_center_y - y
                
                # Full control on Y, weaker on X to maintain safety
                ax = self.Kp * error_x - self.Kd * vx
                ay = self.Kp * error_y - self.Kd * vy
            else:
                # Not fully X-aligned yet, stay in hallway center (y=0) and align X
                error_x = pocket_center_x - x
                target_y = 0.0
                error_y = target_y - y
                
                ax = self.Kp * error_x - self.Kd * vx
                ay = self.Kp * error_y - self.Kd * vy
        
        
        # Phase 4: Outside pocket x-range - move along center line (y=0) toward pocket x
        else:
            # First, correct y to center line (y=0), then move x toward pocket
            target_y = 0.0  # Center of hallway
            target_x = pocket_center_x  # Move toward pocket center x
            
            error_x = target_x - x
            error_y = target_y - y
            
            # Full PD control for both axes, but x moves toward pocket
            ax = self.Kp * np.sign(error_x) * min(abs(error_x), 3.0) - self.Kd * vx
            ay = self.Kp * error_y - self.Kd * vy
        
        # Clamp accelerations
        a_mag = np.sqrt(ax**2 + ay**2)
        if a_mag > self.a_max:
            ax = ax * self.a_max / a_mag
            ay = ay * self.a_max / a_mag
        
        return np.array([[ax], [ay]])
    
    def simulate_trajectory(self, initial_state, target, horizon, friction=1.0):
        """
        Forward simulate the evade trajectory.
        
        Args:
            initial_state: Initial state [x, y, vx, vy]
            target: Not used
            horizon: Number of steps to simulate
            friction: Not used (for interface compatibility)
            
        Returns:
            trajectory: Array of states over the horizon (4 x horizon+1)
        """
        # Initialize trajectory storage
        state = np.array(initial_state).flatten().reshape(-1, 1)
        trajectory = np.zeros((4, horizon + 1))
        trajectory[:, 0] = state.flatten()
        
        for i in range(horizon):
            # Compute control
            U = self.compute_control(state)
            
            # Simple double integrator dynamics
            # x' = x + vx * dt
            # y' = y + vy * dt  
            # vx' = vx + ax * dt
            # vy' = vy + ay * dt
            
            ax, ay = U[0, 0], U[1, 0]
            
            # Clamp velocities
            vx_new = state[2, 0] + ax * self.dt
            vy_new = state[3, 0] + ay * self.dt
            v_mag = np.sqrt(vx_new**2 + vy_new**2)
            if v_mag > self.v_max:
                vx_new = vx_new * self.v_max / v_mag
                vy_new = vy_new * self.v_max / v_mag
            
            # Update state
            state[0, 0] += state[2, 0] * self.dt
            state[1, 0] += state[3, 0] * self.dt
            state[2, 0] = vx_new
            state[3, 0] = vy_new
            
            trajectory[:, i + 1] = state.flatten()
        
        return trajectory
    
    def get_behavior_name(self):
        return "EvadeToPocket"

