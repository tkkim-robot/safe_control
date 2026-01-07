"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Gatekeeper - Safety shielding algorithm that guarantees safety for all time.
It manages candidate trajectory generation and updates the committed trajectory.
The controller switches between a nominal trajectory and a backup trajectory
according to a pre-defined candidate horizon and event offset.

The gatekeeper:
1. Takes a nominal controller (e.g., MPCC) and a backup controller (e.g., lane change)
2. Forward simulates the closed-loop trajectory by merging nominal + backup trajectories
3. Checks if the candidate trajectory is valid (no collisions)
4. If invalid, discounts the switching time and retries
5. Maximizes the switching time to stay on nominal trajectory as long as safely possible
6. Only sends control from the committed trajectory to the robot

Two modes supported:
1. Forward propagation mode: Use nominal_controller and backup_controller functions
2. External trajectory mode: Use externally provided control sequences (e.g., from MPC)

@required-scripts: safe_control/robots/dynamic_bicycle2D.py
"""

import numpy as np
from scipy.integrate import solve_ivp


def angle_normalize(x):
    """Normalize angle to [-pi, pi]."""
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class Gatekeeper:
    """
    Gatekeeper safety shielding algorithm.
    
    Validates nominal trajectories against safety constraints and maintains
    a committed trajectory that is guaranteed to be safe.
    """
    
    def __init__(self, robot, robot_spec, dt=0.05, 
                 backup_horizon=2.0, event_offset=0.5, ax=None,
                 nominal_horizon=None):
        """
        Initialize the Gatekeeper controller.

        Args:
            robot: Robot instance (e.g., DriftingCar, DoubleIntegrator2D) with dynamics
            robot_spec: Robot specification dictionary
            dt: Simulation timestep
            backup_horizon: Duration (seconds) for backup trajectory (TB in paper)
            event_offset: Time offset before next candidate generation event
            ax: Matplotlib axis for visualization (optional)
            nominal_horizon: Max nominal horizon (seconds). If None, uses backup_horizon.
            
        Note: nominal_horizon is automatically determined from MPCC's trajectory length
              if using external trajectory mode, otherwise uses the nominal_horizon param.
        """
        self.robot = robot
        self.robot_spec = robot_spec
        self.dt = dt
        self.backup_horizon = backup_horizon
        self.event_offset = event_offset
        self.horizon_discount = dt * 2  # Discount step for binary search (finer granularity)
        self.nominal_horizon = nominal_horizon if nominal_horizon is not None else backup_horizon
        
        # Infer state/control dimensions from robot model
        model = robot_spec.get('model', 'DynamicBicycle2D')
        if model in ['DoubleIntegrator2D', 'double_integrator']:
            # Double Integrator: [x, y, vx, vy] (4 states), [ax, ay] (2 controls)
            self.n_states = 4
            self.n_controls = 2
        else:
            # Default: Drifting car model
            # State: [x, y, theta, r, beta, V, delta, tau] (8 states)
            # Control: [delta_dot, tau_dot] (2 inputs)
            self.n_states = 8
            self.n_controls = 2
        
        # Controllers (will be set externally)
        self.nominal_controller = None  # Function: state -> control
        self.backup_controller = None   # Function: state, target -> control
        self.backup_target = None       # Target for backup controller (e.g., target_y for lane change)
        
        # Environment for collision checking (will be set externally)
        self.env = None
        
        # Moving obstacles (for forward simulation during validation)
        # Each obstacle: {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'radius': r} or similar
        self.moving_obstacles = None  # Set via set_moving_obstacles()
        
        # Timing
        self.next_event_time = 0.0
        self.current_time_idx = int(backup_horizon / dt)  # Start from backup (safe init)
        self.committed_horizon = 0.0  # Length of nominal portion in committed trajectory
        
        # Trajectory storage
        self.committed_x_traj = None   # Committed state trajectory
        self.committed_u_traj = None   # Committed control trajectory
        self.candidate_x_traj = None   # Current candidate state trajectory
        self.candidate_u_traj = None   # Current candidate control trajectory
        
        # External nominal trajectory (from MPC)
        self.nominal_x_traj = None
        self.nominal_u_traj = None
        
        # Visualization
        self.ax = ax
        self.visualize_backup = True
        self.backup_trajs = []  # Store backup trajectories for visualization
        self.save_every_N = 1
        self.curr_step = 0
        
        # Visualization handles
        self.committed_nominal_line = None  # Green: nominal portion of committed
        self.committed_backup_line = None   # Blue: backup portion of committed
        self.switching_point_marker = None
        
        # Track the actual nominal horizon steps used (from MPCC trajectory)
        self.actual_nominal_steps = 0
        
        if ax is not None:
            self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup visualization handles for trajectory display."""
        if self.ax is None:
            return
        
        # Committed nominal portion (green = following MPCC)
        self.committed_nominal_line, = self.ax.plot(
            [], [], '-', color='lime', linewidth=3, alpha=0.9,
            label='Committed nominal', zorder=8
        )
        
        # Committed backup portion (blue = lane change)
        self.committed_backup_line, = self.ax.plot(
            [], [], '-', color='dodgerblue', linewidth=3, alpha=0.9,
            label='Committed backup', zorder=8
        )
        
        # Switching point marker
        self.switching_point_marker, = self.ax.plot(
            [], [], 'mo', markersize=12, markerfacecolor='magenta',
            markeredgecolor='white', markeredgewidth=2,
            label='Switching point', zorder=9
        )
    
    def set_nominal_controller(self, nominal_controller):
        """
        Set the nominal controller function.
        
        Args:
            nominal_controller: Function that takes (state) and returns control input
        """
        self.nominal_controller = nominal_controller
    
    def set_backup_controller(self, backup_controller, target=None):
        """
        Set the backup controller.
        
        Args:
            backup_controller: BackupController instance with compute_control method
            target: Target for the backup controller (e.g., target_y for lane change)
        """
        self.backup_controller = backup_controller
        self.backup_target = target
    
    def set_environment(self, env):
        """
        Set the environment for collision checking.
        
        Args:
            env: Environment instance with collision checking methods
        """
        self.env = env
    
    def set_nominal_trajectory(self, nominal_x_traj, nominal_u_traj):
        """
        Set the nominal trajectory from external source (e.g., MPC).
        
        Args:
            nominal_x_traj: State trajectory (n_steps x n_states) or (n_states x n_steps)
            nominal_u_traj: Control trajectory (n_steps x n_controls) or (n_controls x n_steps)
        """
        # Ensure trajectories are in (n_steps, n_dim) format
        if nominal_x_traj is not None:
            if nominal_x_traj.ndim == 2 and nominal_x_traj.shape[0] < nominal_x_traj.shape[1]:
                nominal_x_traj = nominal_x_traj.T
            self.nominal_x_traj = np.array(nominal_x_traj)
        
        if nominal_u_traj is not None:
            if nominal_u_traj.ndim == 2 and nominal_u_traj.shape[0] < nominal_u_traj.shape[1]:
                nominal_u_traj = nominal_u_traj.T
            self.nominal_u_traj = np.array(nominal_u_traj)
    
    def _dynamics_step(self, state, control, friction=None):
        """
        Single step of dynamics integration.
        
        Args:
            state: Current state (model-dependent)
                   - DynamicBicycle2D: [x, y, theta, r, beta, V, delta, tau]
                   - DoubleIntegrator2D: [x, y, vx, vy]
            control: Control input (model-dependent)
            friction: Friction coefficient (uses robot default if None)
            
        Returns:
            next_state: State after one timestep
        """
        state = np.array(state).flatten()
        control = np.array(control).flatten()
        
        model = self.robot_spec.get('model', 'DynamicBicycle2D')
        
        if model in ['DoubleIntegrator2D', 'double_integrator']:
            # Double Integrator: state = [x, y, vx, vy], control = [ax, ay]
            x, y, vx, vy = state[0], state[1], state[2], state[3]
            ax, ay = control[0], control[1]
            
            # Apply acceleration
            vx_new = vx + ax * self.dt
            vy_new = vy + ay * self.dt
            
            # Clamp velocity to v_max if specified
            v_max = self.robot_spec.get('v_max', float('inf'))
            v_mag = np.sqrt(vx_new**2 + vy_new**2)
            if v_mag > v_max:
                vx_new = vx_new * v_max / v_mag
                vy_new = vy_new * v_max / v_mag
            
            # Update position
            next_state = np.zeros(4)
            next_state[0] = x + vx * self.dt
            next_state[1] = y + vy * self.dt
            next_state[2] = vx_new
            next_state[3] = vy_new
            
        else:
            # DynamicBicycle2D model
            from safe_control.robots.dynamic_bicycle2D import DynamicBicycle2D
            
            # Create dynamics model with appropriate friction
            sim_spec = self.robot_spec.copy()
            if friction is not None:
                sim_spec['mu'] = friction
            dynamics = DynamicBicycle2D(self.dt, sim_spec)
            
            # Extract dynamics state [r, beta, V, delta, tau]
            X_dyn = state[3:8].reshape(-1, 1)
            U = control.reshape(-1, 1)
            
            # Step dynamics
            X_dyn_next = dynamics.step(X_dyn, U)
            
            # Update global state
            theta = state[2]
            V = X_dyn_next[2, 0]
            beta = X_dyn_next[1, 0]
            r = X_dyn_next[0, 0]
            
            # Velocity in global frame
            vx_global = V * np.cos(theta + beta)
            vy_global = V * np.sin(theta + beta)
            
            # Update state
            next_state = np.zeros(8)
            next_state[0] = state[0] + vx_global * self.dt
            next_state[1] = state[1] + vy_global * self.dt
            next_state[2] = angle_normalize(state[2] + r * self.dt)
            next_state[3:8] = X_dyn_next.flatten()
        
        return next_state
    
    def _forward_simulate_nominal(self, initial_state, horizon_steps, friction=None):
        """
        Forward simulate the nominal trajectory using nominal controller.
        
        Args:
            initial_state: Initial state
            horizon_steps: Number of steps to simulate
            friction: Friction coefficient
            
        Returns:
            x_traj: State trajectory (horizon_steps+1, n_states)
            u_traj: Control trajectory (horizon_steps, n_controls)
        """
        if horizon_steps <= 0:
            return np.empty((0, self.n_states)), np.empty((0, self.n_controls))
        
        x_traj = np.zeros((horizon_steps + 1, self.n_states))
        u_traj = np.zeros((horizon_steps, self.n_controls))
        
        state = np.array(initial_state).flatten()
        x_traj[0] = state
        
        for i in range(horizon_steps):
            # Get control from nominal controller
            if self.nominal_controller is not None:
                control = self.nominal_controller(state.reshape(-1, 1))
                control = np.array(control).flatten()
            else:
                control = np.zeros(self.n_controls)
            
            u_traj[i] = control
            state = self._dynamics_step(state, control, friction)
            x_traj[i + 1] = state
        
        return x_traj, u_traj
    
    def _forward_simulate_backup(self, initial_state, horizon_steps, friction=None):
        """
        Forward simulate the backup trajectory using backup controller.
        
        Args:
            initial_state: Initial state
            horizon_steps: Number of steps to simulate
            friction: Friction coefficient
            
        Returns:
            x_traj: State trajectory (horizon_steps+1, n_states) - excludes initial state
            u_traj: Control trajectory (horizon_steps, n_controls)
        """
        if horizon_steps <= 0:
            return np.empty((0, self.n_states)), np.empty((0, self.n_controls))
        
        x_traj = np.zeros((horizon_steps, self.n_states))
        u_traj = np.zeros((horizon_steps, self.n_controls))
        
        state = np.array(initial_state).flatten()
        
        for i in range(horizon_steps):
            # Get control from backup controller
            if self.backup_controller is not None:
                control = self.backup_controller.compute_control(
                    state.reshape(-1, 1), 
                    self.backup_target
                )
                control = np.array(control).flatten()
            else:
                control = np.zeros(self.n_controls)
            
            u_traj[i] = control
            state = self._dynamics_step(state, control, friction)
            x_traj[i] = state
        
        return x_traj, u_traj
    
    def _generate_candidate_trajectory(self, initial_state, nominal_horizon_steps, friction=None):
        """
        Generate candidate trajectory by concatenating nominal and backup.
        
        Args:
            initial_state: Current state
            nominal_horizon_steps: Number of steps for nominal trajectory
            friction: Friction coefficient
            
        Returns:
            candidate_x_traj: Full state trajectory
            candidate_u_traj: Full control trajectory
            actual_nominal_steps: Actual number of nominal steps used
        """
        backup_horizon_steps = int(self.backup_horizon / self.dt)
        actual_nominal_steps = nominal_horizon_steps
        
        # Generate nominal trajectory
        if self.nominal_controller is None and self.nominal_x_traj is not None:
            # Use externally provided nominal trajectory (from MPC)
            n_avail = len(self.nominal_x_traj)
            # n_use = number of states (includes initial state)
            n_use = min(nominal_horizon_steps + 1, n_avail)
            actual_nominal_steps = max(0, n_use - 1)  # Actual steps = states - 1
            
            if n_use > 0:
                nominal_x_traj = self.nominal_x_traj[:n_use]
                nominal_u_traj = self.nominal_u_traj[:actual_nominal_steps] if actual_nominal_steps > 0 else np.empty((0, self.n_controls))
            else:
                nominal_x_traj = initial_state.reshape(1, -1)
                nominal_u_traj = np.empty((0, self.n_controls))
                actual_nominal_steps = 0
        else:
            # Forward simulate using nominal controller
            nominal_x_traj, nominal_u_traj = self._forward_simulate_nominal(
                initial_state, nominal_horizon_steps, friction
            )
            actual_nominal_steps = len(nominal_u_traj)
        
        # State at end of nominal trajectory (switching point)
        if len(nominal_x_traj) > 0:
            state_at_switch = nominal_x_traj[-1]
        else:
            state_at_switch = np.array(initial_state).flatten()
        
        # Generate backup trajectory from switching point
        backup_x_traj, backup_u_traj = self._forward_simulate_backup(
            state_at_switch, backup_horizon_steps, friction
        )
        
        # Concatenate trajectories
        if len(backup_x_traj) > 0:
            self.candidate_x_traj = np.vstack([nominal_x_traj, backup_x_traj])
            self.candidate_u_traj = np.vstack([nominal_u_traj, backup_u_traj])
        else:
            self.candidate_x_traj = nominal_x_traj
            self.candidate_u_traj = nominal_u_traj
        
        return self.candidate_x_traj, self.candidate_u_traj, actual_nominal_steps
    
    def set_moving_obstacles(self, obstacles):
        """
        Set moving obstacles for forward simulation during validation.
        
        Args:
            obstacles: List of obstacle state dicts or callable that returns them.
                       Each dict should have: 'x', 'y', 'vx', 'vy', and size info.
                       If callable, will be called each validation step.
        """
        self.moving_obstacles = obstacles
    
    def _is_collision(self, state, safety_margin=0.0, obstacle_state=None):
        """
        Check if a state collides with obstacles or boundaries.
        
        Args:
            state: State vector [x, y, theta, ...] or [x, y, vx, vy]
            safety_margin: Additional buffer distance for conservative checking
            obstacle_state: Optional dict with obstacle position at this timestep
                           {'x': x, 'y': y, 'length': l, 'width': w} for rectangular
                           or {'x': x, 'y': y, 'radius': r} for circular
            
        Returns:
            bool: True if collision detected
        """
        if self.env is None:
            return False
        
        position = state[:2]
        robot_radius = self.robot_spec.get('radius', 1.5) + safety_margin
        
        # Check boundary collision
        if hasattr(self.env, 'check_collision'):
            if self.env.check_collision(position, robot_radius):
                return True
        
        # Check static obstacle collision (from environment)
        if hasattr(self.env, 'check_obstacle_collision'):
            collision, _ = self.env.check_obstacle_collision(position, robot_radius)
            if collision:
                return True
        
        # Check moving obstacle collision (if provided)
        if obstacle_state is not None:
            collision = self._check_moving_obstacle_collision(
                position, robot_radius, obstacle_state
            )
            if collision:
                return True
        
        return False
    
    def _check_moving_obstacle_collision(self, position, robot_radius, obstacle):
        """
        Check collision with a moving obstacle at a specific state.
        
        Args:
            position: Robot position [x, y]
            robot_radius: Robot collision radius
            obstacle: Dict with obstacle state at this timestep
            
        Returns:
            bool: True if collision detected
        """
        x, y = position[0], position[1]
        obs_x = obstacle.get('x', 0)
        obs_y = obstacle.get('y', 0)
        
        # Check for rectangular obstacle (like bullet bill)
        if 'length' in obstacle and 'width' in obstacle:
            obs_length = obstacle['length']
            obs_width = obstacle['width']
            
            # Rectangular hitbox
            obs_x_min = obs_x - obs_length / 2
            obs_x_max = obs_x + obs_length / 2
            obs_y_min = obs_y - obs_width / 2
            obs_y_max = obs_y + obs_width / 2
            
            # Check rectangle-circle collision
            closest_x = np.clip(x, obs_x_min, obs_x_max)
            closest_y = np.clip(y, obs_y_min, obs_y_max)
            dist = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
            
            return dist < robot_radius
        else:
            # Circular obstacle
            obs_radius = obstacle.get('radius', 1.0)
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            return dist < (robot_radius + obs_radius)
    
    def _forward_simulate_obstacle(self, obstacle_state, n_steps):
        """
        Forward simulate obstacle positions for n_steps.
        
        Args:
            obstacle_state: Initial obstacle state dict with 'x', 'y', 'vx', 'vy'
            n_steps: Number of timesteps to simulate
            
        Returns:
            list: List of obstacle state dicts for each timestep
        """
        states = []
        obs = obstacle_state.copy()
        
        for i in range(n_steps):
            # Create state dict for this timestep
            step_state = obs.copy()
            states.append(step_state)
            
            # Advance position
            obs = obs.copy()
            obs['x'] = obs.get('x', 0) + obs.get('vx', 0) * self.dt
            obs['y'] = obs.get('y', 0) + obs.get('vy', 0) * self.dt
        
        return states
    
    def _is_candidate_valid(self, candidate_x_traj, safety_margin=1.0, obstacle_state=None):
        """
        Check if the candidate trajectory is valid (collision-free).
        
        Args:
            candidate_x_traj: Trajectory to validate (n_steps, n_states)
            safety_margin: Additional buffer distance for conservative checking
            obstacle_state: Optional initial obstacle state for forward simulation.
                           If provided, obstacle will be forward-simulated alongside
                           the robot trajectory for time-synchronized collision checking.
            
        Returns:
            bool: True if trajectory is valid (collision-free)
        """
        if candidate_x_traj is None or len(candidate_x_traj) == 0:
            return True
        
        # Forward simulate obstacle if provided
        obstacle_states = None
        if obstacle_state is not None:
            obstacle_states = self._forward_simulate_obstacle(
                obstacle_state, len(candidate_x_traj)
            )
        
        # Check each timestep
        for i, state in enumerate(candidate_x_traj):
            obs_at_t = obstacle_states[i] if obstacle_states else None
            if self._is_collision(state, safety_margin, obs_at_t):
                return False
        
        return True
    
    def _update_committed_trajectory(self, actual_nominal_steps):
        """
        Update the committed trajectory with the valid candidate.
        
        Args:
            actual_nominal_steps: Actual number of steps in nominal portion
        """
        self.committed_x_traj = self.candidate_x_traj.copy()
        self.committed_u_traj = self.candidate_u_traj.copy()
        self.next_event_time = self.event_offset
        # Reset to 0 - we'll use index 0 this iteration, then increment at end
        self.current_time_idx = 0
        self.actual_nominal_steps = actual_nominal_steps  # Store for visualization
        self.committed_horizon = actual_nominal_steps * self.dt
        
        # Store backup trajectory for visualization
        if self.visualize_backup and self.curr_step % self.save_every_N == 0:
            # Backup starts after nominal states (+1 because states include initial)
            backup_start_idx = actual_nominal_steps + 1
            if backup_start_idx < len(self.candidate_x_traj):
                backup_portion = self.candidate_x_traj[backup_start_idx:]
                self.backup_trajs.append(backup_portion.copy())
        self.curr_step += 1
    
    def solve_control_problem(self, robot_state, friction=None):
        """
        Main gatekeeper control loop.
        
        This method should be called at every control timestep.
        It manages event timing, generates candidate trajectories,
        validates them, and outputs control from the committed trajectory.
        
        Args:
            robot_state: Current robot state
            friction: Current friction coefficient (for dynamics simulation)
            
        Returns:
            control_input: Control output for current timestep
        """
        robot_state = np.array(robot_state).flatten()
        
        # Initialize committed trajectory if not yet done
        if self.committed_x_traj is None or self.committed_u_traj is None:
            backup_horizon_steps = int(self.backup_horizon / self.dt)
            backup_x_traj, backup_u_traj = self._forward_simulate_backup(
                robot_state, backup_horizon_steps, friction
            )
            # Include initial state
            self.committed_x_traj = np.vstack([robot_state.reshape(1, -1), backup_x_traj])
            self.committed_u_traj = backup_u_traj
            self.committed_horizon = 0.0  # Pure backup initially
            self.actual_nominal_steps = 0
            self.current_time_idx = 0
            self.next_event_time = 0.0  # Trigger event immediately on next call
        
        # Try updating committed trajectory if event triggered
        if self.current_time_idx >= self.next_event_time / self.dt:
            # Determine max nominal steps from available trajectory or controller
            if self.nominal_x_traj is not None:
                # Use externally provided trajectory
                max_nominal_steps = len(self.nominal_x_traj) - 1  # -1 because states include initial
            elif self.nominal_controller is not None:
                # Use nominal_horizon with forward simulation
                max_nominal_steps = int(self.nominal_horizon / self.dt)
            else:
                max_nominal_steps = 0
            
            discount_steps = max(1, int(self.horizon_discount / self.dt))
            
            # Binary search for maximum valid nominal horizon
            found_valid = False
            for i in range(max_nominal_steps // discount_steps + 2):
                nominal_horizon_steps = max_nominal_steps - i * discount_steps
                
                if nominal_horizon_steps < 0:
                    nominal_horizon_steps = 0
                
                # Generate candidate trajectory
                candidate_x_traj, candidate_u_traj, actual_steps = self._generate_candidate_trajectory(
                    robot_state, nominal_horizon_steps, friction
                )
                
                # Get moving obstacle state for validation
                obstacle_state = None
                if self.moving_obstacles is not None:
                    if callable(self.moving_obstacles):
                        obstacle_state = self.moving_obstacles()
                    else:
                        obstacle_state = self.moving_obstacles
                
                # Check validity with safety margin (conservative)
                if self._is_candidate_valid(candidate_x_traj, safety_margin=1.0, obstacle_state=obstacle_state):
                    self._update_committed_trajectory(actual_steps)
                    found_valid = True
                    break
            
            # If no valid trajectory found, DO NOT update committed trajectory
            # Keep following the previously committed backup trajectory
            # This prevents "creeping forward" when all candidates are invalid
            if not found_valid:
                # Just reschedule the next event, don't change the committed trajectory
                # This ensures we stay on the previously committed (safe) backup path
                self.next_event_time = self.current_time_idx * self.dt + self.event_offset
        
        # Output control from committed trajectory using current index
        if self.current_time_idx < len(self.committed_u_traj):
            control = self.committed_u_traj[self.current_time_idx].reshape(-1, 1)
        else:
            # Fallback: use backup controller directly
            if self.backup_controller is not None:
                control = self.backup_controller.compute_control(
                    robot_state.reshape(-1, 1),
                    self.backup_target
                )
            else:
                control = np.zeros((self.n_controls, 1))
        
        # Increment index AFTER getting control (for next iteration)
        self.current_time_idx += 1
        
        # Update visualization
        self._update_visualization()
        
        return control
    
    def _update_visualization(self):
        """Update visualization of committed and candidate trajectories."""
        if self.ax is None:
            return
        
        if self.committed_x_traj is None:
            return
        
        # Split committed trajectory into nominal and backup portions
        # nominal portion: indices 0 to actual_nominal_steps (inclusive)
        # backup portion: indices actual_nominal_steps to end
        nominal_end_idx = self.actual_nominal_steps + 1  # +1 because it's state count
        
        # Update committed nominal portion (green)
        if self.committed_nominal_line is not None:
            if nominal_end_idx > 0:
                x = self.committed_x_traj[:nominal_end_idx, 0]
                y = self.committed_x_traj[:nominal_end_idx, 1]
                self.committed_nominal_line.set_data(x, y)
            else:
                self.committed_nominal_line.set_data([], [])
        
        # Update committed backup portion (blue)
        if self.committed_backup_line is not None:
            # Start from switching point state to show continuity
            backup_start_idx = max(0, nominal_end_idx - 1)  # Overlap at switching point
            if backup_start_idx < len(self.committed_x_traj):
                x = self.committed_x_traj[backup_start_idx:, 0]
                y = self.committed_x_traj[backup_start_idx:, 1]
                self.committed_backup_line.set_data(x, y)
            else:
                self.committed_backup_line.set_data([], [])
        
        # Update switching point marker (at the junction of nominal and backup)
        if self.switching_point_marker is not None:
            switch_idx = self.actual_nominal_steps  # Index of switching state
            if 0 <= switch_idx < len(self.committed_x_traj):
                switch_state = self.committed_x_traj[switch_idx]
                self.switching_point_marker.set_data([switch_state[0]], [switch_state[1]])
            else:
                self.switching_point_marker.set_data([], [])
    
    def get_committed_trajectory(self):
        """Get the current committed trajectory."""
        return self.committed_x_traj, self.committed_u_traj
    
    def get_candidate_trajectory(self):
        """Get the current candidate trajectory."""
        return self.candidate_x_traj, self.candidate_u_traj
    
    def get_committed_horizon(self):
        """Get the committed nominal horizon in seconds."""
        return self.committed_horizon
    
    def get_backup_trajectories(self):
        """Get stored backup trajectories for plotting."""
        return self.backup_trajs.copy() if self.visualize_backup else []
    
    def clear_trajectories(self):
        """Clear stored backup trajectories."""
        self.backup_trajs.clear()
    
    def is_using_backup(self):
        """Check if currently in backup mode."""
        nominal_steps = int(self.committed_horizon / self.dt)
        return self.current_time_idx >= nominal_steps
    
    def get_status(self):
        """Get current gatekeeper status."""
        return {
            'current_time_idx': self.current_time_idx,
            'committed_horizon': self.committed_horizon,
            'next_event_time': self.next_event_time,
            'using_backup': self.is_using_backup(),
            'committed_length': len(self.committed_u_traj) if self.committed_u_traj is not None else 0,
        }

