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


class BackupControllerManager:
    """
    Manager for backup controllers with trajectory visualization.
    """
    
    def __init__(self, robot_spec, dt, ax=None):
        """
        Initialize the backup controller manager.
        
        Args:
            robot_spec: Dictionary with robot specifications
            dt: Time step
            ax: Matplotlib axis for visualization (optional)
        """
        self.robot_spec = robot_spec
        self.dt = dt
        self.ax = ax
        
        # Available backup controllers
        self.controllers = {}
        
        # Visualization handles
        self.backup_traj_line = None
        self.backup_traj_points = None
        
        # Default backup horizon
        self.backup_horizon = 100  # steps
        
        if ax is not None:
            self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup visualization handles for backup trajectory."""
        # Backup trajectory line (distinct color - orange/yellow)
        self.backup_traj_line, = self.ax.plot(
            [], [], 'o-', color='orange', linewidth=2, alpha=0.7,
            markersize=3, markerfacecolor='yellow', markeredgecolor='orange',
            label='Backup trajectory', zorder=7
        )
    
    def add_controller(self, name, controller):
        """
        Add a backup controller.
        
        Args:
            name: Name/identifier for the controller
            controller: BackupController instance
        """
        self.controllers[name] = controller
    
    def create_lane_change_controller(self, direction='left'):
        """
        Create and add a lane change controller.
        
        Args:
            direction: 'left' or 'right'
            
        Returns:
            The created controller
        """
        name = f'lane_change_{direction}'
        controller = LaneChangeController(self.robot_spec, self.dt, direction)
        self.add_controller(name, controller)
        return controller
    
    def simulate_backup(self, controller_name, current_state, target, horizon=None, friction=1.0):
        """
        Simulate backup trajectory using specified controller.
        
        Args:
            controller_name: Name of the controller to use
            current_state: Current robot state
            target: Target for the controller
            horizon: Simulation horizon (uses default if None)
            friction: Current friction coefficient
            
        Returns:
            trajectory: Simulated trajectory (8 x horizon+1)
        """
        if controller_name not in self.controllers:
            raise ValueError(f"Unknown controller: {controller_name}")
        
        controller = self.controllers[controller_name]
        horizon = horizon or self.backup_horizon
        
        return controller.simulate_trajectory(current_state, target, horizon, friction)
    
    def update_visualization(self, trajectory):
        """
        Update the backup trajectory visualization.
        
        Args:
            trajectory: Trajectory to visualize (8 x N)
        """
        if self.backup_traj_line is not None:
            # Extract x, y positions
            x = trajectory[0, :]
            y = trajectory[1, :]
            self.backup_traj_line.set_data(x, y)
    
    def set_backup_horizon(self, horizon):
        """Set the backup trajectory horizon."""
        self.backup_horizon = horizon

