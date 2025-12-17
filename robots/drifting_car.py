"""
Drifting Car - A car model for drift simulation with detailed visualization.

This module provides a dynamic bicycle model based car with detailed
visualization. The car uses the DynamicBicycle2D dynamics which includes
nonlinear tire forces suitable for high-slip (drifting) maneuvers.

The visualization is similar to the MATLAB drift_parking example with:
- A body polygon with realistic shape
- Four wheels (2 front that steer, 2 rear)
- Trajectory tracking

Global State: X_global = [x, y, theta, r, beta, V, delta, tau]^T
    - x, y: Global position [m]
    - theta: Global heading [rad]
    - r: Yaw rate [rad/s]
    - beta: Side slip angle [rad]
    - V: Velocity magnitude [m/s]
    - delta: Steering angle [rad]
    - tau: Rear wheel torque [Nm]

Input: U = [delta_dot, tau_dot]^T
    - delta_dot: Steering rate [rad/s]
    - tau_dot: Torque rate [Nm/s]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

try:
    from robots.dynamic_bicycle2D import DynamicBicycle2D, angle_normalize
except ImportError:
    from safe_control.robots.dynamic_bicycle2D import DynamicBicycle2D, angle_normalize


class DriftingCar:
    """
    A car model for drift simulation with detailed visualization.
    
    Uses DynamicBicycle2D dynamics and provides MATLAB-style visualization
    with body and wheel rendering.
    """
    
    def __init__(self, X0, robot_spec, dt, ax=None):
        """
        Initialize the drifting car.
        
        Args:
            X0: Initial state. Can be:
                - [x, y, theta]: Position and heading (velocity=5, others=0)
                - [x, y, theta, V]: Position, heading, velocity (others=0)
                - [x, y, theta, r, beta, V, delta, tau]: Full state
            robot_spec: Dictionary with robot specifications
            dt: Time step
            ax: Matplotlib axis for plotting
        """
        self.dt = dt
        self.ax = ax
        self.robot_spec = robot_spec.copy()
        
        # Initialize dynamics model
        self.dynamics = DynamicBicycle2D(dt, self.robot_spec)
        
        # Parse initial state
        X0 = np.array(X0, dtype=float).flatten()
        if len(X0) == 3:
            # [x, y, theta] -> add default velocity and zeros
            self.X = np.array([X0[0], X0[1], X0[2], 0, 0, 5.0, 0, 0]).reshape(-1, 1)
        elif len(X0) == 4:
            # [x, y, theta, V] -> add zeros for r, beta, delta, tau
            self.X = np.array([X0[0], X0[1], X0[2], 0, 0, X0[3], 0, 0]).reshape(-1, 1)
        elif len(X0) == 8:
            self.X = X0.reshape(-1, 1)
        else:
            raise ValueError(f"Invalid initial state dimension: {len(X0)}")
        
        # Control input: [delta_dot, tau_dot]
        self.U = np.zeros((2, 1))
        
        # Trajectory history
        self.trajectory = [self.X[:2, 0].copy()]
        
        # MPC predicted trajectory (will be updated by controller)
        self.mpc_predicted_states = None
        self.mpc_predicted_inputs = None
        
        # Vehicle geometry for visualization
        self._setup_vehicle_geometry()
        
        # Initialize plot handles
        self.body_patch = None
        self.tire_patches = {}
        self.trajectory_line = None
        self.mpc_trajectory_line = None
        self.cg_marker = None
        
        if ax is not None:
            self._setup_plot_handles()
    
    def _setup_vehicle_geometry(self):
        """Setup vehicle body and tire geometry vertices."""
        a = self.robot_spec.get('a', 1.6)
        b = self.robot_spec.get('b', 0.8)
        L = self.robot_spec.get('body_length', 4.3)
        W = self.robot_spec.get('body_width', 1.8)
        
        # Body vertices (centered at CG)
        rear_overhang = (L - a - b) * 0.4
        front_overhang = (L - a - b) * 0.6
        
        # Main body outline (counterclockwise from rear-left)
        self.body_vertices = np.array([
            [-b - rear_overhang, -W/2],
            [-b - rear_overhang, W/2],
            [-b - rear_overhang + 0.3, W/2 + 0.05],
            [a + front_overhang - 0.8, W/2 + 0.05],
            [a + front_overhang - 0.3, W/2 * 0.7],
            [a + front_overhang, W/2 * 0.5],
            [a + front_overhang, -W/2 * 0.5],
            [a + front_overhang - 0.3, -W/2 * 0.7],
            [a + front_overhang - 0.8, -W/2 - 0.05],
            [-b - rear_overhang + 0.3, -W/2 - 0.05],
        ]).T
        
        # Tire geometry
        self.tire_length = 0.6
        self.tire_width = 0.25
        
        tire_y_offset = W / 2 - self.tire_width / 2 - 0.1
        self.tire_positions = {
            'front_left': np.array([a, tire_y_offset]),
            'front_right': np.array([a, -tire_y_offset]),
            'rear_left': np.array([-b, tire_y_offset]),
            'rear_right': np.array([-b, -tire_y_offset])
        }
        
        tl = self.tire_length / 2
        tw = self.tire_width / 2
        self.tire_vertices = np.array([
            [-tl, -tw],
            [-tl, tw],
            [tl, tw],
            [tl, -tw]
        ]).T
        
        # Colors
        self.body_color = np.array([0, 0.45, 0.74])
        self.tire_color = np.array([0.3, 0.3, 0.3])
        
    def _setup_plot_handles(self):
        """Initialize matplotlib plot handles for animation."""
        if self.ax is None:
            return
            
        # Body polygon
        self.body_patch = MplPolygon(
            self.body_vertices.T, closed=True,
            facecolor=self.body_color, edgecolor='black',
            linewidth=1.5, alpha=0.9, zorder=10
        )
        self.ax.add_patch(self.body_patch)
        
        # Four tires
        self.tire_patches = {}
        for name in ['front_left', 'front_right', 'rear_left', 'rear_right']:
            tire = MplPolygon(
                self.tire_vertices.T, closed=True,
                facecolor=self.tire_color, edgecolor='black',
                linewidth=1, alpha=0.9, zorder=11
            )
            self.ax.add_patch(tire)
            self.tire_patches[name] = tire
        
        # CG marker
        self.cg_marker, = self.ax.plot([], [], 'ko', markersize=5, zorder=12)
        
        # Trajectory line
        self.trajectory_line, = self.ax.plot(
            [], [], 'b-', linewidth=2, alpha=0.7, zorder=5
        )
        
        # MPC predicted trajectory
        self.mpc_trajectory_line, = self.ax.plot(
            [], [], 'r--', linewidth=1.5, alpha=0.8, zorder=6
        )
        
        # Initial render
        self.render_plot()
    
    # ==================== State accessors ====================
    
    def get_position(self):
        """Return current [x, y] position."""
        return self.X[:2, 0].copy()
    
    def get_orientation(self):
        """Return current heading angle theta."""
        return self.X[2, 0]
    
    def get_velocity(self):
        """Return current velocity magnitude."""
        return self.X[5, 0]
    
    def get_yaw_rate(self):
        """Return current yaw rate."""
        return self.X[3, 0]
    
    def get_slip_angle(self):
        """Return current side slip angle."""
        return self.X[4, 0]
    
    def get_steering_angle(self):
        """Return current steering angle."""
        return self.X[6, 0]
    
    def get_torque(self):
        """Return current rear wheel torque."""
        return self.X[7, 0]
    
    def get_state(self):
        """Return full state vector."""
        return self.X.copy()
    
    def get_dynamics_state(self):
        """Return state for dynamics model [r, beta, V, delta, tau]."""
        return self.X[3:8, :].copy()
    
    # ==================== Dynamics ====================
    
    def f(self):
        """Return drift dynamics f(x) for dynamics state."""
        return self.dynamics.f(self.get_dynamics_state())
    
    def g(self):
        """Return input matrix g(x) for dynamics state."""
        return self.dynamics.g(self.get_dynamics_state())
    
    def f_casadi(self, X_dyn):
        """Return CasADi drift dynamics f(x)."""
        return self.dynamics.f(X_dyn, casadi=True)
    
    def g_casadi(self, X_dyn):
        """Return CasADi input matrix g(x)."""
        return self.dynamics.g(X_dyn, casadi=True)
    
    def step(self, U):
        """
        Step the car dynamics forward.
        
        Args:
            U: Control input [delta_dot, tau_dot]^T (2x1)
            
        Returns:
            New state X
        """
        self.U = np.array(U).reshape(-1, 1)
        
        # Get current dynamics state
        X_dyn = self.get_dynamics_state()
        
        # Step dynamics
        X_dyn_next = self.dynamics.step(X_dyn, self.U)
        
        # Update global state
        # Global position update using body velocities
        theta = self.X[2, 0]
        V = X_dyn_next[2, 0]
        beta = X_dyn_next[1, 0]
        r = X_dyn_next[0, 0]
        
        # Velocity in global frame
        vx_global = V * np.cos(theta + beta)
        vy_global = V * np.sin(theta + beta)
        
        # Update global position
        self.X[0, 0] += vx_global * self.dt
        self.X[1, 0] += vy_global * self.dt
        self.X[2, 0] += r * self.dt
        self.X[2, 0] = angle_normalize(self.X[2, 0])
        
        # Update dynamics states
        self.X[3:8, :] = X_dyn_next
        
        # Record trajectory
        self.trajectory.append(self.X[:2, 0].copy())
        
        return self.X.copy()
    
    def nominal_input(self, goal, d_min=0.5):
        """
        Compute nominal control input to reach goal.
        
        Args:
            goal: Target position [x, y] or [x, y, theta]
            d_min: Minimum distance threshold
            
        Returns:
            Control input [delta_dot, tau_dot]
        """
        goal = np.array(goal).flatten()
        pos = self.get_position()
        theta = self.get_orientation()
        V = self.get_velocity()
        delta = self.get_steering_angle()
        tau = self.get_torque()
        
        # Distance to goal
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Desired heading
        theta_des = np.arctan2(dy, dx)
        heading_error = angle_normalize(theta_des - theta)
        
        # Steering control
        delta_des = np.clip(2.0 * heading_error, 
                           -self.robot_spec['delta_max'], 
                           self.robot_spec['delta_max'])
        delta_dot = np.clip(3.0 * (delta_des - delta),
                           -self.robot_spec['delta_dot_max'],
                           self.robot_spec['delta_dot_max'])
        
        # Torque control (simple velocity tracking)
        V_des = min(distance, self.robot_spec['v_max'])
        tau_des = 500.0 * (V_des - V)
        tau_des = np.clip(tau_des, -self.robot_spec['tau_max'], self.robot_spec['tau_max'])
        tau_dot = np.clip(2000.0 * (tau_des - tau),
                         -self.robot_spec['tau_dot_max'],
                         self.robot_spec['tau_dot_max'])
        
        return np.array([[delta_dot], [tau_dot]])
    
    def stop(self):
        """Return control input to stop the vehicle."""
        return self.dynamics.stop(self.get_dynamics_state())
    
    def has_stopped(self, tol=0.5):
        """Check if car has stopped."""
        return self.dynamics.has_stopped(self.get_dynamics_state(), tol)
    
    # ==================== MPC interface ====================
    
    def set_mpc_prediction(self, states, inputs):
        """
        Store MPC predicted trajectory for visualization.
        
        Args:
            states: Predicted states over horizon (n_states x horizon)
            inputs: Predicted inputs over horizon (n_inputs x horizon)
        """
        self.mpc_predicted_states = states
        self.mpc_predicted_inputs = inputs
    
    def get_mpc_prediction(self):
        """Get stored MPC prediction."""
        return self.mpc_predicted_states, self.mpc_predicted_inputs
    
    # ==================== Visualization ====================
    
    def render_plot(self):
        """Update the plot with current car state."""
        if self.ax is None:
            return
            
        x, y, theta = self.X[0, 0], self.X[1, 0], self.X[2, 0]
        delta = self.X[6, 0]
        
        # Rotation matrix
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Transform body vertices
        body_world = R @ self.body_vertices + np.array([[x], [y]])
        self.body_patch.set_xy(body_world.T)
        
        # Transform tires
        for name, pos in self.tire_positions.items():
            pos_world = R @ pos.reshape(-1, 1) + np.array([[x], [y]])
            
            if 'front' in name:
                tire_angle = theta + delta
            else:
                tire_angle = theta
                
            cos_tire, sin_tire = np.cos(tire_angle), np.sin(tire_angle)
            R_tire = np.array([[cos_tire, -sin_tire], [sin_tire, cos_tire]])
            
            tire_world = R_tire @ self.tire_vertices + pos_world
            self.tire_patches[name].set_xy(tire_world.T)
        
        # Update CG marker
        self.cg_marker.set_data([x], [y])
        
        # Update trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.trajectory_line.set_data(traj[:, 0], traj[:, 1])
        
        # Update MPC prediction visualization
        if self.mpc_predicted_states is not None:
            # Extract x, y from predicted states
            pred_x = self.mpc_predicted_states[0, :]
            pred_y = self.mpc_predicted_states[1, :]
            self.mpc_trajectory_line.set_data(pred_x, pred_y)


class DriftingCarSimulator:
    """
    Simulator for the drifting car with environment integration.
    """
    
    def __init__(self, car, env, show_animation=True):
        """
        Initialize the simulator.
        
        Args:
            car: DriftingCar instance
            env: Environment with check_collision method
            show_animation: Whether to show real-time animation
        """
        self.car = car
        self.env = env
        self.show_animation = show_animation
        
        self.collision_detected = False
        self.collision_marker = None
        
    def check_collision(self):
        """Check for collision with environment boundaries."""
        position = self.car.get_position()
        robot_radius = self.car.robot_spec.get('radius', 1.2)
        
        if hasattr(self.env, 'check_collision_detailed'):
            result = self.env.check_collision_detailed(position, robot_radius)
            collision = result['collision']
        else:
            collision = self.env.check_collision(position, robot_radius)
        
        if collision and not self.collision_detected:
            self.collision_detected = True
            self._draw_collision_marker()
            
        return collision
    
    def _draw_collision_marker(self):
        """Draw red exclamation mark at collision point."""
        if self.car.ax is not None:
            pos = self.car.get_position()
            self.collision_marker = self.car.ax.text(
                pos[0] + 0.5, pos[1] + 0.5, '!',
                color='red', fontsize=28, fontweight='bold',
                ha='center', va='center', zorder=100
            )
    
    def step(self, U):
        """
        Execute one simulation step.
        
        Args:
            U: Control input [delta_dot, tau_dot]
            
        Returns:
            dict with 'collision', 'state', 'done' keys
        """
        self.car.step(U)
        collision = self.check_collision()
        
        if self.show_animation:
            self.car.render_plot()
        
        return {
            'collision': collision,
            'state': self.car.get_state(),
            'done': collision
        }
    
    def draw_plot(self, pause=0.01):
        """Refresh the plot."""
        if self.show_animation and self.car.ax is not None:
            self.car.ax.figure.canvas.draw_idle()
            self.car.ax.figure.canvas.flush_events()
            plt.pause(pause)

