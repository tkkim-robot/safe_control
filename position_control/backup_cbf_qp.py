"""
Created on January 13th, 2026
@author: Taekyung Kim

@description:
Backup CBF (Control Barrier Function) - Safety shielding algorithm using backup trajectory constraints.

This algorithm implements Backup CBF-QP which:
1. Rolls out the backup controller trajectory from current state
2. Propagates sensitivity matrices along backup trajectory
3. Imposes CBF constraints at each discretized state along the backup horizon
4. Imposes a terminal invariant set constraint at the final state
5. Solves a QP to find the safe control closest to the nominal

The QP formulation:
    minimize    ||u - u_ref||²
    subject to  ∇h(φ_i) · S_i · g(x_0) · u ≥ -∇h(φ_i) · S_i · f(x_0) - α(h(φ_i))  for i=1..N
                ∇h_S(φ_N) · S_N · g(x_0) · u ≥ -∇h_S(φ_N) · S_N · f(x_0) - α_S(h_S(φ_N))

where:
- φ_i is the state at backup step i
- S_i is the sensitivity matrix propagated to step i
- h is the safety CBF (obstacle/boundary avoidance)
- h_S is the terminal control invariant set CBF

@required-scripts: safe_control/robots/, safe_control/position_control/backup_controller.py
"""

import numpy as np
import cvxpy as cp


class BackupCBF:
    """
    Backup CBF safety shielding algorithm.
    
    Uses backup trajectory rollout and sensitivity propagation to constrain
    current control for future safety.
    """
    
    def __init__(self, robot, robot_spec, dt=0.05, backup_horizon=2.0, ax=None):
        """
        Initialize the Backup CBF controller.

        Args:
            robot: Robot instance with dynamics (f, g, step methods)
            robot_spec: Robot specification dictionary
            dt: Simulation timestep
            backup_horizon: Duration (seconds) for backup trajectory
            ax: Matplotlib axis for visualization (optional)
        """
        self.robot = robot
        self.robot_spec = robot_spec
        self.dt = dt
        self.backup_horizon = backup_horizon
        self.N = int(backup_horizon / dt)  # Number of backup steps
        
        # Infer state/control dimensions from robot model
        model = robot_spec.get('model', 'DoubleIntegrator2D')
        if model in ['DoubleIntegrator2D', 'double_integrator']:
            self.n_states = 4
            self.n_controls = 2
        elif model in ['DriftingCar', 'DynamicBicycle2D']:
            self.n_states = 8
            self.n_controls = 2
        else:
            # Default fallback
            self.n_states = 4
            self.n_controls = 2
        
        # Controllers (set externally)
        self.nominal_controller = None
        self.backup_controller = None
        self.backup_target = None
        
        # Environment for collision checking
        self.env = None
        
        # Moving obstacles
        self.moving_obstacles = None
        
        # External nominal trajectory (from MPC/MPCC)
        self.nominal_x_traj = None
        self.nominal_u_traj = None
        
        # CBF parameters - higher alpha = constraint activates sooner (more aggressive)
        self.alpha = 1.0  # Aligned with theoretical stability requirements
        self.alpha_terminal = 2.0  # Class-K function gain for terminal constraint
        
        # Safety margin - depends on scenario
        # Can be overridden via robot_spec
        if 'safety_margin' in robot_spec:
            self.safety_margin = robot_spec['safety_margin']
        elif model in ['DriftingCar', 'DynamicBicycle2D']:
            self.safety_margin = 0.5  # Realistic margin
        else:
            self.safety_margin = 0.5  # Standard default margin (consistent with test configs)
        
        # Stabilization weights for QP objective
        # Balance between steering (small) and torque (large)
        if model in ['DriftingCar', 'DynamicBicycle2D']:
            # Steering is ~0.1, Torque is ~2000. Weight steering more or torque less.
            # Normalize to ~1.0 range
            # Steering and torque weights [steering, torque]
            # Penalize torque deviation (braking) more heavily to encourage steering avoidance
            self.Q_u = np.array([1.0, 10.0])
        else:
            self.Q_u = np.array([1.0, 1.0])
            
        # Visualization
        self.ax = ax
        self.visualize_backup = False  # Disabled by default for performance
        self.backup_trajs = []
        self.save_every_N = 5
        self.curr_step = 0
        
        # Visualization handles
        self.backup_traj_line = None
        if ax is not None:
            self._setup_visualization()
        
        # Status tracking
        self._using_backup = False
        self._last_intervention = False
        self._last_h_min = 1.0
        self.global_min_h = float('inf')  # Track global minimum for debugging
        
    def _setup_visualization(self):
        """Setup visualization handles."""
        if self.ax is None:
            return
        
        self.backup_traj_line, = self.ax.plot(
            [], [], '-', color='cyan', linewidth=2, alpha=0.7,
            label='Backup trajectory', zorder=18
        )
    
    def set_nominal_controller(self, nominal_controller):
        """Set the nominal controller function."""
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
        """Set the environment for collision checking and invariant set."""
        self.env = env
    
    def set_nominal_trajectory(self, nominal_x_traj, nominal_u_traj):
        """Set the nominal trajectory from external source (e.g., MPC)."""
        if nominal_x_traj is not None:
            if nominal_x_traj.ndim == 2 and nominal_x_traj.shape[0] < nominal_x_traj.shape[1]:
                nominal_x_traj = nominal_x_traj.T
            self.nominal_x_traj = np.array(nominal_x_traj)
        
        if nominal_u_traj is not None:
            if nominal_u_traj.ndim == 2 and nominal_u_traj.shape[0] < nominal_u_traj.shape[1]:
                nominal_u_traj = nominal_u_traj.T
            self.nominal_u_traj = np.array(nominal_u_traj)
    
    def set_moving_obstacles(self, obstacles):
        """Set moving obstacles for forward simulation during validation."""
        self.moving_obstacles = obstacles
    
    def _get_nominal_control(self, state):
        """Get nominal control from controller or trajectory."""
        if self.nominal_u_traj is not None and len(self.nominal_u_traj) > 0:
            return self.nominal_u_traj[0].flatten()
        elif self.nominal_controller is not None:
            return np.array(self.nominal_controller(state.reshape(-1, 1))).flatten()
        else:
            return np.zeros(self.n_controls)
    
    def _dynamics_f(self, x):
        """Get full-state dynamics f(x) from robot."""
        model = self.robot_spec.get('model', '')
        if model in ['DriftingCar', 'DynamicBicycle2D']:
            # DriftingCar has separate f_full() that includes kinematic coupling
            return self.robot.f_full(x)
        else:
            # DoubleIntegrator2D and others: f() already returns full-state dynamics
            x = np.array(x).reshape(-1, 1)
            return self.robot.f(x).flatten()
    
    def _dynamics_g(self, x):
        """Get full-state control matrix g(x) from robot."""
        model = self.robot_spec.get('model', '')
        if model in ['DriftingCar', 'DynamicBicycle2D']:
            # DriftingCar has separate g_full() that extends to full state
            return self.robot.g_full(x)
        else:
            # DoubleIntegrator2D and others: g() already returns full-state matrix
            x = np.array(x).reshape(-1, 1)
            return self.robot.g(x)
    
    
    def _dynamics_df_dx(self, x):
        """Get Jacobian of f with respect to x."""
        x = np.array(x).reshape(-1, 1)
        if hasattr(self.robot, 'df_dx'):
            return self.robot.df_dx(x)
        else:
            # Finite difference fallback
            return self._finite_difference(self._dynamics_f, x.flatten())
    
    def _finite_difference(self, func, x, eps=1e-5):
        """Compute Jacobian via finite differences."""
        x = np.array(x, dtype=float).flatten()
        base = func(x.copy())
        jac = np.zeros((base.size, x.size))
        for i in range(x.size):
            x_pert = x.copy()
            x_pert[i] += eps
            jac[:, i] = (func(x_pert) - base) / eps
        return jac
    
    def _backup_control(self, x):
        """Get backup control for state x."""
        if self.backup_controller is not None:
            u = self.backup_controller.compute_control(
                x.reshape(-1, 1), 
                self.backup_target
            )
            return np.array(u).flatten()
        else:
            return np.zeros(self.n_controls)
    
    def _integrate_backup_trajectory(self, x0):
        """
        Integrate the backup trajectory and sensitivity matrices.
        
        Uses robot.step() for state integration to handle different dynamics models.
        Sensitivity matrices computed via finite differences.
        
        Args:
            x0: Initial state (n_states,)
            
        Returns:
            phi: States along backup trajectory (N, n_states)
            S: Sensitivity matrices (N, n_states, n_states)
        """
        phi = np.zeros((self.N, self.n_states))
        S = np.zeros((self.N, self.n_states, self.n_states))
        
        # Initial conditions
        x = x0.flatten().copy()
        S_curr = np.eye(self.n_states)
        
        phi[0] = x
        S[0] = S_curr
        
        for i in range(1, self.N):
            # Get backup control
            u_b = self._backup_control(x)
            
            # Use robot.step() for state integration (handles all robot types)
            if hasattr(self.robot, 'step'):
                try:
                    # Stateless step: robot.step(X, U)
                    x_next = self.robot.step(x.reshape(-1, 1), u_b.reshape(-1, 1))
                    x_next = np.array(x_next).flatten()
                except Exception:
                    # Fallback to Euler
                    f_val = self._dynamics_f(x)
                    g_val = self._dynamics_g(x)
                    x_dot = f_val + (g_val @ u_b)[:len(x)]
                    x_next = x + self.dt * x_dot
            else:
                # Euler integration for robots without step()
                f_val = self._dynamics_f(x)
                g_val = self._dynamics_g(x)
                x_dot = f_val + (g_val @ u_b)[:len(x)]
                x_next = x + self.dt * x_dot
            
            # Sensitivity via finite differences: S_{k+1} = dPhi/dx * S_k
            # where Phi(x) is one step of dynamics
            eps = 1e-5
            n = len(x)
            A_discrete = np.zeros((n, n))
            for j in range(n):
                x_pert = x.copy()
                x_pert[j] += eps
                
                u_b_pert = self._backup_control(x_pert)
                
                if hasattr(self.robot, 'step'):
                    try:
                        # Step with perturbed state AND perturbed control
                        x_next_pert = self.robot.step(x_pert.reshape(-1, 1), u_b_pert.reshape(-1, 1))
                        x_next_pert = np.array(x_next_pert).flatten()
                    except:
                        f_pert = self._dynamics_f(x_pert)
                        g_pert = self._dynamics_g(x_pert)
                        x_dot_pert = f_pert + (g_pert @ u_b_pert)[:len(x)]
                        x_next_pert = x_pert + self.dt * x_dot_pert
                else:
                    f_pert = self._dynamics_f(x_pert)
                    g_pert = self._dynamics_g(x_pert)
                    x_dot_pert = f_pert + (g_pert @ u_b_pert)[:len(x)]
                    x_next_pert = x_pert + self.dt * x_dot_pert
                
                A_discrete[:, j] = (x_next_pert - x_next) / eps
            
            S_curr = A_discrete @ S_curr
            
            x = x_next
            phi[i] = x
            S[i] = S_curr
        
        return phi, S
    
    def _get_obstacle_at_time(self, t):
        """Get obstacle state at time t."""
        if self.moving_obstacles is None:
            return None
        
        if callable(self.moving_obstacles):
            try:
                return self.moving_obstacles(t)
            except TypeError:
                return self.moving_obstacles()
        else:
            return self.moving_obstacles
    
    def _h_safety(self, x, t=0.0):
        """
        Compute safety CBF value h(x).
        
        Returns positive value when safe, negative when unsafe.
        For evade scenario: avoids bullet and stays within valid regions (hallway + pocket).
        """
        x = np.array(x).flatten()
        position = x[:2]
        robot_radius = self.robot_spec.get('radius', 0.5)
        
        h_min = float('inf')
        
        # Environment boundary constraint
        if self.env is not None:
            if hasattr(self.env, 'half_width') and hasattr(self.env, 'pocket_x_min'):
                # Evade scenario: compute proper distance to boundaries
                px, py = position[0], position[1]
                
                # Bottom boundary (always applies)
                h_bottom = py + self.env.half_width - robot_radius
                h_min = min(h_min, h_bottom)
                
                # Left boundary (start of hallway)
                h_left = px - robot_radius
                h_min = min(h_min, h_left)
                
                # Right boundary (end of hallway)
                h_right = self.env.hallway_length - px - robot_radius
                h_min = min(h_min, h_right)
                
                # Top boundary - complex due to pocket
                # If in pocket x-range: can go up to pocket_y_max
                # Otherwise: limited to half_width
                if self.env.pocket_x_min <= px <= self.env.pocket_x_max:
                    # In pocket x-range
                    h_top = self.env.pocket_y_max - py - robot_radius
                    h_min = min(h_min, h_top)
                    
                    # Also check pocket side walls when in upper region
                    if py > self.env.half_width:
                        h_pocket_left = px - self.env.pocket_x_min - robot_radius
                        h_pocket_right = self.env.pocket_x_max - px - robot_radius
                        h_min = min(h_min, h_pocket_left, h_pocket_right)
                else:
                    # Outside pocket x-range
                    h_top = self.env.half_width - py - robot_radius
                    h_min = min(h_min, h_top)
                    
            elif hasattr(self.env, 'track_width'):
                # Track environment (drift car)
                half_width = self.env.track_width / 2
                h_upper = half_width - position[1] - robot_radius
                h_lower = position[1] + half_width - robot_radius
                h_min = min(h_min, h_upper, h_lower)
        
        # Static obstacle constraints  
        if self.env is not None and hasattr(self.env, 'obstacles'):
            for obs in self.env.obstacles:
                obs_pos = np.array([obs.get('x', 0), obs.get('y', 0)])
                # Handle different obstacle formats:
                # - Simple dict: {'x': x, 'y': y, 'radius': r}
                # - DriftingEnv: {'x': x, 'y': y, 'spec': {'radius': r}}
                if 'spec' in obs:
                    obs_radius = obs['spec'].get('radius', 2.5)
                else:
                    obs_radius = obs.get('radius', 1.0)
                dist = np.linalg.norm(position - obs_pos)
                h_obs = dist - robot_radius - obs_radius - self.safety_margin
                h_min = min(h_min, h_obs)
        
        # Moving obstacle constraint (bullet)
        obs_state = self._get_obstacle_at_time(t)
        if obs_state is not None and obs_state.get('active', True):
            obs_x = obs_state.get('x', 0)
            obs_y = obs_state.get('y', 0)
            
            if 'length' in obs_state and 'width' in obs_state:
                # Rectangular obstacle (bullet bill)
                obs_length = obs_state['length']
                obs_width = obs_state['width']
                
                # Distance to rectangle (signed distance)
                dx = max(abs(position[0] - obs_x) - obs_length / 2, 0)
                dy = max(abs(position[1] - obs_y) - obs_width / 2, 0)
                dist = np.sqrt(dx**2 + dy**2)
                h_obs = dist - robot_radius - self.safety_margin
            else:
                # Circular obstacle
                obs_radius = obs_state.get('radius', 1.0)
                dist = np.sqrt((position[0] - obs_x)**2 + (position[1] - obs_y)**2)
                h_obs = dist - robot_radius - obs_radius - self.safety_margin
            
            h_min = min(h_min, h_obs)
        
        return h_min if h_min != float('inf') else 1.0
    
    def _grad_h_safety(self, x, t=0.0):
        """Compute gradient of safety CBF ∇h(x) via finite differences."""
        eps = 1e-5
        x = np.array(x).flatten()
        grad = np.zeros(self.n_states)
        h0 = self._h_safety(x, t)
        
        for i in range(self.n_states):
            x_pert = x.copy()
            x_pert[i] += eps
            grad[i] = (self._h_safety(x_pert, t) - h0) / eps
        
        return grad
    
    def _h_terminal(self, x):
        """
        Compute terminal control invariant set CBF.
        
        The terminal set is where the backup controller can keep the robot safe indefinitely.
        For evade scenario: inside safe pocket with low velocity
        For drift car: in target lane with stable heading
        """
        x = np.array(x).flatten()
        position = x[:2]
        
        h_terminal = float('inf')
        
        # Check if in safe zone based on environment type
        if self.env is not None:
            # Evade scenario: safe pocket
            if hasattr(self.env, 'get_pocket_bounds'):
                pocket_bounds = self.env.get_pocket_bounds()
                robot_radius = self.robot_spec.get('radius', 0.5)
                # More generous margin (wide terminal set)
                margin = robot_radius + 0.2
                
                h_x_min = position[0] - pocket_bounds['x_min'] - margin
                h_x_max = pocket_bounds['x_max'] - position[0] - margin
                h_y_min = position[1] - pocket_bounds['y_min'] - margin
                h_y_max = pocket_bounds['y_max'] - position[1] - margin
                
                h_terminal = min(h_x_min, h_x_max, h_y_min, h_y_max)
                
            # Drift car: in left lane (backup target lane)
            # Drift car: ensure track boundaries AND loosely centered on target
            elif hasattr(self.env, 'track_width'):
                 half_width = self.env.track_width / 2
                 robot_radius = self.robot_spec.get('radius', 1.5)
                 
                 # 1. Track boundary constraints (Hard limits)
                 h_track_upper = half_width - position[1] - robot_radius
                 h_track_lower = position[1] + half_width - robot_radius
                 
                 h_list = [h_track_upper, h_track_lower]
                 
                 # 2. Target lane centering (terminal set)
                 if self.backup_target is not None:
                     target_margin = 12.0 # generous terminal set
                     h_target_upper = (self.backup_target + target_margin) - position[1] - robot_radius
                     h_target_lower = position[1] - (self.backup_target - target_margin) - robot_radius
                     h_list.extend([h_target_upper, h_target_lower])
                 
                 h_terminal = min(h_list)
                 
                 # DEBUG: Print if large violation
                 if h_terminal < -10.0:
                     print(f"DEBUG: Large h_terminal violation: {h_terminal}")
                     print(f"  Pos: {position}, Target: {self.backup_target}")
                     print(f"  Track: [{h_track_upper:.2f}, {h_track_lower:.2f}]")
                     if self.backup_target is not None:
                         print(f"  Target: [{h_target_upper:.2f}, {h_target_lower:.2f}] (margin 5.0)")
                         
        # Velocity constraint: should be slowing down
        
        # Velocity constraint: should be slowing down
        if self.n_states >= 4:
            if self.n_states == 4:
                # Double integrator: [x, y, vx, vy]
                velocity = np.sqrt(x[2]**2 + x[3]**2)
            else:
                # Drifting car: V is at index 5
                velocity = abs(x[5]) if len(x) > 5 else 0
            
            v_max = self.robot_spec.get('v_max', 1.5)
            h_velocity = v_max - velocity
            h_terminal = min(h_terminal, h_velocity)

        # Check safety constraint at terminal state (ensure terminal set is subset of safe set)
        if hasattr(self, 'backup_horizon'):
             t_terminal = self.backup_horizon
             h_safe_terminal = self._h_safety(x, t_terminal)
             h_terminal = min(h_terminal, h_safe_terminal)
        
        return h_terminal if h_terminal != float('inf') else 1.0
    
    def _grad_h_terminal(self, x):
        """Compute gradient of terminal CBF via finite differences."""
        eps = 1e-5
        x = np.array(x).flatten()
        grad = np.zeros(self.n_states)
        h0 = self._h_terminal(x)
        
        for i in range(self.n_states):
            x_pert = x.copy()
            x_pert[i] += eps
            grad[i] = (self._h_terminal(x_pert) - h0) / eps
        
        return grad
    
    def _alpha(self, h):
        """Class-K function for CBF constraint."""
        return self.alpha * h
    
    def _alpha_terminal(self, h):
        """Class-K function for terminal constraint."""
        return self.alpha_terminal * h
    
    def solve_control_problem(self, robot_state, friction=None):
        """
        Main Backup CBF-QP control computation.
        
        When the backup trajectory is unsafe (collision imminent), switches to
        using backup control as the reference instead of nominal control.
        
        Args:
            robot_state: Current robot state
            friction: Current friction coefficient (for dynamics simulation)
            
        Returns:
            control_input: Safe control output for current timestep
        """
        robot_state = np.array(robot_state).flatten()
        
        # Integrate backup trajectory and sensitivity
        phi, S = self._integrate_backup_trajectory(robot_state)
        
        # Calculate safety of the nonlinear backup trajectory for status reporting
        # Pass time t_i to correctly handle moving obstacles during the rollout
        h_vals = [self._h_safety(phi[i], i * self.dt) for i in range(len(phi))]
        h_safety_min = np.min(h_vals)
        h_term = self._h_terminal(phi[-1])
        self._last_h_min = min(h_safety_min, h_term)
        
        # Track global minimum
        if self._last_h_min < self.global_min_h:
            self.global_min_h = self._last_h_min
        
        # Store for visualization
        if self.visualize_backup and self.curr_step % self.save_every_N == 0:
            self.backup_trajs.append(phi.copy())
        self.curr_step += 1
        
        # Get nominal control reference - BackupCBF uses nominal as QP reference
        # The CBF constraints will modify it when needed for safety
        u_ref = self._get_nominal_control(robot_state)
        
        # Check terminal constraint
        h_terminal = self._h_terminal(phi[-1])
        
        # Get continuous dynamics at initial state
        # _dynamics_f now includes proper kinematic coupling for all models
        f0 = self._dynamics_f(robot_state)
        g0 = self._dynamics_g(robot_state)
        
        # Build constraint matrices
        G_list = []
        h_list = []
        
        # Safety constraints along backup horizon
        for i in range(1, self.N):
            x_i = phi[i]
            S_i = S[i]
            t_i = i * self.dt
            
            h_val = self._h_safety(x_i, t_i)
            grad_h = self._grad_h_safety(x_i, t_i)
            
            # 1. Moving obstacle time derivative dh/dt
            # Finite difference approx if moving obstacles exist
            if self.moving_obstacles is not None:
                h_val_dt = self._h_safety(x_i, t_i + self.dt)
                dh_dt = (h_val_dt - h_val) / self.dt
            else:
                dh_dt = 0.0
                
            # 2. Backup trajectory drift term f_pi (velocity of backup plan)
            # f_pi = (phi[i+1] - phi[i]) / dt (approx)
            if i < self.N - 1:
                f_pi_i = (phi[i+1] - phi[i]) / self.dt
            else:
                f_pi_i = (phi[i] - phi[i-1]) / self.dt
                
            # Constraint: 
            # ∇h S (f0 + g0 u) - ∇h f_pi + dh_dt >= -α(h)
            # ∇h S g0 u >= -∇h S f0 + ∇h f_pi - dh_dt - α(h)
            
            lhs = grad_h @ S_i @ g0
            rhs = -(grad_h @ S_i @ f0) + (grad_h @ f_pi_i) - dh_dt - self._alpha(h_val)
            
            # Skip constraints that are already very satisfied (for numerical stability)
            if np.linalg.norm(lhs) > 1e-6:
                G_list.append(lhs)
                h_list.append(rhs)
        
        # Terminal constraint
        # Note: Terminal constraint does NOT use the f_pi drift term 
        # because the terminal time T is fixed relative to start for infinite safety
        x_T = phi[-1]
        S_T = S[-1]
        
        grad_h_terminal = self._grad_h_terminal(x_T)
        
        lhs = grad_h_terminal @ S_T @ g0
        rhs = -(grad_h_terminal @ S_T @ f0 + self._alpha_terminal(h_terminal))
        
        if np.linalg.norm(lhs) > 1e-6:
            G_list.append(lhs)
            h_list.append(rhs)
        
        # Solve QP if we have constraints
        if len(G_list) > 0:
            G = np.array(G_list)
            h = np.array(h_list)
            
            # SCALING: Scale variables to be O(1) using actuator limits
            # Generalized for different models
            u_scale_list = []
            
            model = self.robot_spec.get('model', '')
            
            if model in ['DriftingCar', 'DynamicBicycle2D']:
                 delta_dot_max = self.robot_spec.get('delta_dot_max', np.deg2rad(15))
                 tau_dot_max = self.robot_spec.get('tau_dot_max', 8000.0)
                 u_scale_list = [delta_dot_max, tau_dot_max]
            elif model in ['DoubleIntegrator2D', 'double_integrator']:
                 a_max = self.robot_spec.get('a_max', 2.0)
                 u_scale_list = [a_max, a_max]
            else:
                 # Default fallback if unknown model (should restrict or warn)
                 u_scale_list = [1.0] * self.n_controls
                 
            u_scale = np.array(u_scale_list)
            
            S_mat = np.diag(u_scale)           # u = S * u_scaled
            
            # Scaled variables: u_scaled \in [-1, 1] approximately
            u_scaled = cp.Variable(self.n_controls)
            
            S_inv = np.diag(1.0/u_scale)
            u_ref_scaled = (S_inv @ u_ref).flatten()
            
            W = np.diag(self.Q_u)
            
            # Expression to minimize: W @ (u_scaled - u_ref_scaled)
            # This penalizes relative deviation (percentage of actuation range)
            # which is much better conditioned than physical units
            error_expr = W @ (u_scaled - u_ref_scaled)
            objective = cp.Minimize(cp.sum_squares(error_expr))
            
            # Scaled CBF constraints
            # G * u >= h  =>  G * (S * u_scaled) >= h
            constraints = [(G @ S_mat) @ u_scaled >= h]
            
            # Control limits (scaled to [-1, 1])
            constraints.extend([
                u_scaled >= -1.0,
                u_scaled <= 1.0,
            ])
            
            prob = cp.Problem(objective, constraints)
            
            # Solve with strict feasibility requirement
            # Try OSQP first (faster), fallback to SCS (more robust)
            try:
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except:
                prob.solve(solver=cp.SCS, verbose=False)
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                if u_scaled.value is None:
                        raise ValueError("QP solved but returned None value")

                # Unscale result to physical units for output
                u_safe = S_mat @ u_scaled.value
                
                # Compare SCALED output to SCALED reference to determine backup usage
                # This ensures torque (large values) doesn't dominate steering (small values)
                # u_ref_scaled was calculated earlier
                u_diff_scaled = u_scaled.value - u_ref_scaled
                
                # error_val = W * diff (element-wise since W is diagonal Q_u)
                error_val = self.Q_u * u_diff_scaled
                control_diff = np.linalg.norm(error_val)
                
                # Threshold for intervention (in normalized cost units)
                self._last_intervention = control_diff > 0.1
                self._using_backup = control_diff > 0.1

            else:
                if self._last_h_min > 0.01:
                    u_backup = self.backup_controller.compute_control(robot_state, self.backup_target).flatten()
                    u_safe = u_backup
                    self._last_intervention = True
                    self._using_backup = True
                else:
                    print(f"\n*** CBF-QP {prob.status.upper()} ***")
                    print(f"CRITICAL: Backup trajectory is also UNSAFE or marginally safe (h={self._last_h_min:.4f}).")
                    if self.env and hasattr(self.env, 'obstacles'):
                        print(f"Obstacles: {self.env.obstacles}")
                    
                    print("\n--- INFEASIBILITY ANALYSIS ---")
                    try:
                        for i in range(len(phi)):
                            if i < len(phi):
                                # Pass time i*dt to correctly evaluate moving obstacles
                                h_val = self._h_safety(phi[i], i * self.dt)
                                if h_val < 0.1:
                                    print(f"  Step {i} is dangerously close: h={h_val:.4f}")
                    except:
                        pass
                    
                    raise ValueError(f"CBF-QP {prob.status} and Backup Unsafe (h_min={self._last_h_min:.4f})")
        else:
            # No constraints needed - use reference directly
            u_safe = u_ref
            self._last_intervention = False
            self._using_backup = False
        
        # Update visualization
        self._update_visualization(phi)
        
        return u_safe.reshape(-1, 1)
    
    def _update_visualization(self, phi):
        """Update visualization of backup trajectory."""
        if self.ax is None or self.backup_traj_line is None:
            return
        
        x = phi[:, 0]
        y = phi[:, 1]
        self.backup_traj_line.set_data(x, y)
    
    def is_using_backup(self):
        """Check if CBF is actively intervening."""
        return self._using_backup
    
    def get_status(self):
        """Get current controller status."""
        return {
            'using_backup': self._using_backup,
            'last_intervention': self._last_intervention,
            'backup_horizon': self.backup_horizon,
            'h_min': self._last_h_min,
            'global_min_h': self.global_min_h,
            'num_constraints': self.N,
        }
    
    def get_backup_trajectories(self):
        """Get stored backup trajectories for plotting."""
        return self.backup_trajs.copy() if self.visualize_backup else []
    
    def clear_trajectories(self):
        """Clear stored backup trajectories."""
        self.backup_trajs.clear()
