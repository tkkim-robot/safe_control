"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Model Predictive Contouring Control (MPCC) for path following.
Jointly minimizes contouring, lag, heading errors and control effort
using do-mpc framework with CasADi optimization.

State: [x, y, theta, r, beta, V, delta, tau, psi]
Input: [delta_dot, tau_dot, v_psi]

@required-scripts: safe_control/robots/dynamic_bicycle2D.py
"""

import numpy as np
import casadi as ca
import do_mpc


class MPCC:
    """
    Model Predictive Contouring Controller.
    """
    
    def __init__(self, robot, robot_spec, show_mpc_traj=False, horizon=None):
        """
        Initialize the MPCC controller.
        
        Args:
            robot: Robot instance (DriftingCar with DynamicBicycle2D dynamics)
            robot_spec: Robot specifications dictionary
            show_mpc_traj: Whether to show MPC trajectory in plots
            horizon: MPC prediction horizon in steps (default: 30 = 1.5s at dt=0.05)
        """
        self.robot = robot
        self.robot_spec = robot_spec
        if self.robot_spec.get('model') != 'DriftingCar':
             raise NotImplementedError("MPCC is only implemented for DriftingCar model.")
        self.status = 'optimal'
        self.show_mpc_traj = show_mpc_traj
        
        # Get dynamics from robot
        self.dynamics = robot.dynamics
        
        # MPC parameters
        self.horizon = horizon if horizon is not None else 30  # Default 1.5s at dt=0.05
        self.dt = robot.dt
        
        # State: [x, y, theta, r, beta, V, delta, tau, psi]
        self.n_dyn_states = 5  # [r, beta, V, delta, tau]
        self.n_states = 9      # Full state with x, y, theta, psi
        self.n_controls = 3    # [delta_dot, tau_dot, v_psi]
        
        # Cost function weights (tuned for smooth tracking)
        self.Q_c = 100.0     # Contouring error (reduced for less aggressive correction)
        self.Q_l = 5.0       # Lag error  
        self.Q_theta = 30.0  # Heading error
        self.Q_v = 20.0      # Velocity tracking
        self.Q_r = 10.0      # Yaw rate penalty (to reduce oscillation)
        self.v_ref = 5.0     # Target velocity [m/s]
        # Control weights: [delta_dot, tau_dot, v_psi]
        # All weights must be non-negative for proper cost minimization
        self.R = np.array([50.0, 0.01, 0.1])
        self.v_psi_ref = 2.0  # Slower progress rate
        
        # Reference path
        self.path_x = None
        self.path_y = None
        self.path_theta = None
        self.path_s = None
        self.path_length = 0.0
        self.path_curvature = None
        
        # Current path parameter
        self._current_psi = 0.0
        self._psi_initialized = False  # Will be set True after first global search
        
        # MPC solution storage
        self.predicted_states = None
        self.predicted_inputs = None
        
        # Reference horizon for visualization
        self.reference_horizon = None
        
        self.setup_control_problem()
        
    def setup_control_problem(self):
        """Setup the MPC optimization problem."""
        self.model = self._create_model()
        self.mpc = self._create_mpc()
        self.simulator = self._create_simulator()
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        
    def _create_model(self):
        """Create the do-mpc model using dynamics from DynamicBicycle2D."""
        model = do_mpc.model.Model('discrete')
        
        # Full state: [x, y, theta, r, beta, V, delta, tau, psi]
        _x = model.set_variable(var_type='_x', var_name='x', shape=(self.n_states, 1))
        
        # Controls: [delta_dot, tau_dot, v_psi]
        _u = model.set_variable(var_type='_u', var_name='u', shape=(self.n_controls, 1))
        
        # TVP for path reference
        _path_ref = model.set_variable(var_type='_tvp', var_name='path_ref', shape=(3, 1))
        
        # Extract states
        x_pos = _x[0, 0]
        y_pos = _x[1, 0]
        theta = _x[2, 0]
        r = _x[3, 0]
        beta = _x[4, 0]
        V = _x[5, 0]
        delta = _x[6, 0]
        tau = _x[7, 0]
        psi = _x[8, 0]
        
        # Extract controls
        delta_dot = _u[0, 0]
        tau_dot = _u[1, 0]
        v_psi = _u[2, 0]
        
        # ========== Use dynamics from DynamicBicycle2D ==========
        # Dynamics state: [r, beta, V, delta, tau]
        X_dyn = ca.vertcat(r, beta, V, delta, tau)
        U_dyn = ca.vertcat(delta_dot, tau_dot)
        
        # Get f and g from the dynamics model
        f_dyn = self.dynamics.f(X_dyn, casadi=True)  # (5x1)
        g_dyn = self.dynamics.g(X_dyn, casadi=True)  # (5x2)
        
        # Dynamics state update: X_dyn_next = X_dyn + (f + g @ U) * dt
        X_dyn_next = X_dyn + (f_dyn + ca.mtimes(g_dyn, U_dyn)) * self.dt
        
        # Extract updated dynamics states
        r_next = X_dyn_next[0, 0]
        beta_next = X_dyn_next[1, 0]
        V_next = X_dyn_next[2, 0]
        delta_next = X_dyn_next[3, 0]
        tau_next = X_dyn_next[4, 0]
        
        # Global position update using current velocities
        vx_global = V * ca.cos(theta + beta)
        vy_global = V * ca.sin(theta + beta)
        
        x_next = x_pos + vx_global * self.dt
        y_next = y_pos + vy_global * self.dt
        theta_next = theta + r * self.dt
        
        # Path parameter update
        psi_next = psi + v_psi * self.dt
        
        # Full state update
        state_next = ca.vertcat(
            x_next, y_next, theta_next,
            r_next, beta_next, V_next, delta_next, tau_next,
            psi_next
        )
        
        model.set_rhs('x', state_next)
        
        # ========== Cost Function ==========
        x_ref = _path_ref[0, 0]
        y_ref = _path_ref[1, 0]
        theta_ref = _path_ref[2, 0]
        
        dx = x_pos - x_ref
        dy = y_pos - y_ref
        
        # Contouring error (perpendicular to path)
        e_c = ca.sin(theta_ref) * dx - ca.cos(theta_ref) * dy
        
        # Lag error (along path)
        e_l = -ca.cos(theta_ref) * dx - ca.sin(theta_ref) * dy
        
        # Heading error (with proper angle wrapping using atan2)
        # This ensures the error is always in [-pi, pi]
        e_theta = ca.atan2(ca.sin(theta - theta_ref), ca.cos(theta - theta_ref))
        
        # Velocity error (keep velocity near target)
        e_v = V - self.v_ref
        
        cost = (self.Q_c * e_c**2 + 
                self.Q_l * e_l**2 + 
                self.Q_theta * e_theta**2 +
                self.Q_v * e_v**2 +
                self.Q_r * r**2)  # Yaw rate penalty to reduce oscillation
        
        model.set_expression(expr_name='cost', expr=cost)
        model.set_expression('e_c', e_c)
        model.set_expression('e_l', e_l)
        model.set_expression('e_v', e_v)
        
        model.setup()
        return model
    
    def _create_mpc(self):
        """Create the MPC controller."""
        mpc = do_mpc.controller.MPC(self.model)
        mpc.settings.supress_ipopt_output()
        
        setup_mpc = {
            'n_horizon': self.horizon,
            't_step': self.dt,
            'n_robust': 0,
            'state_discretization': 'discrete',
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)
        
        # IPOPT solver options for faster and more robust solving
        mpc.set_param(nlpsol_opts={
            'ipopt.max_iter': 100,           # Limit iterations
            'ipopt.print_level': 0,
            'ipopt.acceptable_tol': 1e-4,    # Relaxed tolerance
            'ipopt.acceptable_obj_change_tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes',
            'print_time': 0,
        })
        
        mterm = self.model.aux['cost']
        lterm = self.model.aux['cost']
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=self.R)
        
        # State bounds from robot_spec (relaxed for better feasibility)
        v_max = self.robot_spec.get('v_max', 15.0)
        v_min = self.robot_spec.get('v_min', 0.5)
        delta_max = self.robot_spec.get('delta_max', np.deg2rad(35))
        tau_max = self.robot_spec.get('tau_max', 3000.0)
        r_max = self.robot_spec.get('r_max', 2.0)  # More relaxed yaw rate
        beta_max = self.robot_spec.get('beta_max', np.deg2rad(45))  # More relaxed slip angle
        
        mpc.bounds['lower', '_x', 'x', 3] = -r_max
        mpc.bounds['upper', '_x', 'x', 3] = r_max
        mpc.bounds['lower', '_x', 'x', 4] = -beta_max
        mpc.bounds['upper', '_x', 'x', 4] = beta_max
        mpc.bounds['lower', '_x', 'x', 5] = v_min
        mpc.bounds['upper', '_x', 'x', 5] = v_max
        mpc.bounds['lower', '_x', 'x', 6] = -delta_max
        mpc.bounds['upper', '_x', 'x', 6] = delta_max
        mpc.bounds['lower', '_x', 'x', 7] = -tau_max
        mpc.bounds['upper', '_x', 'x', 7] = tau_max
        
        # Input bounds
        delta_dot_max = self.robot_spec.get('delta_dot_max', np.deg2rad(45))
        tau_dot_max = self.robot_spec.get('tau_dot_max', 8000.0)
        v_psi_max = self.robot_spec.get('v_psi_max', 15.0)
        
        mpc.bounds['lower', '_u', 'u'] = np.array([-delta_dot_max, -tau_dot_max, 0.1])
        mpc.bounds['upper', '_u', 'u'] = np.array([delta_dot_max, tau_dot_max, v_psi_max])
        
        mpc = self._set_tvp(mpc)
        mpc.setup()
        return mpc
    
    def _set_tvp(self, mpc):
        """Set time-varying parameters function."""
        def tvp_fun(t_now):
            tvp_template = mpc.get_tvp_template()
            
            # Store reference horizon for visualization
            ref_horizon_x = []
            ref_horizon_y = []
            
            for k in range(self.horizon + 1):
                psi_k = self._current_psi + k * self.v_psi_ref * self.dt
                x_ref, y_ref, theta_ref, _ = self._get_path_reference(psi_k)
                tvp_template['_tvp', k, 'path_ref'] = np.array([x_ref, y_ref, theta_ref])
                ref_horizon_x.append(x_ref)
                ref_horizon_y.append(y_ref)
            
            # Store for visualization
            self.reference_horizon = np.array([ref_horizon_x, ref_horizon_y])
                
            return tvp_template
        
        mpc.set_tvp_fun(tvp_fun)
        return mpc
    
    def _create_simulator(self):
        """Create the simulator."""
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.dt)
        tvp_template = simulator.get_tvp_template()
        
        def tvp_fun(t_now):
            return tvp_template
        
        simulator.set_tvp_fun(tvp_fun)
        simulator.setup()
        return simulator
    
    def set_reference_path(self, path_x, path_y, path_theta=None):
        """Set the reference path."""
        self.path_x = np.array(path_x)
        self.path_y = np.array(path_y)
        
        # Compute arc length
        dx = np.diff(self.path_x)
        dy = np.diff(self.path_y)
        ds = np.sqrt(dx**2 + dy**2)
        self.path_s = np.concatenate([[0], np.cumsum(ds)])
        self.path_length = self.path_s[-1]
        
        # Compute heading - keep in [-pi, pi] range
        if path_theta is None:
            self.path_theta = np.arctan2(
                np.gradient(self.path_y),
                np.gradient(self.path_x)
            )
        else:
            self.path_theta = np.array(path_theta)
        
        # Compute curvature - need to handle angle wrapping
        dtheta = np.diff(self.path_theta)
        # Wrap dtheta to [-pi, pi] for continuous curvature
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        dtheta = np.concatenate([[dtheta[0]], dtheta])  # Pad to match length
        ds_safe = np.maximum(np.gradient(self.path_s), 0.01)
        self.path_curvature = dtheta / ds_safe
        
        self._current_psi = 0.0
        
    def _get_path_reference(self, psi):
        """Get reference at arc length psi."""
        if self.path_x is None or self.path_s is None:
            return 0.0, 0.0, 0.0, 0.0
        
        # Wrap for looped tracks
        if self.path_length > 0:
            psi = psi % self.path_length
        
        x_ref = np.interp(psi, self.path_s, self.path_x)
        y_ref = np.interp(psi, self.path_s, self.path_y)
        
        # Interpolate angle using sin/cos to handle wrapping properly
        sin_theta = np.interp(psi, self.path_s, np.sin(self.path_theta))
        cos_theta = np.interp(psi, self.path_s, np.cos(self.path_theta))
        theta_ref = np.arctan2(sin_theta, cos_theta)
        
        kappa = np.interp(psi, self.path_s, self.path_curvature)
        
        return x_ref, y_ref, theta_ref, kappa
    
    def _find_closest_path_point(self, x, y, search_window=None, force_global_search=False):
        """
        Find the arc length (psi) of the closest point on the path to (x, y).
        Only searches in a local window around current psi to prevent jumping
        to the other side of a closed track.
        
        Args:
            x, y: Robot position
            search_window: Search window size in meters (default: 30m ahead, 10m behind)
            force_global_search: If True, search entire path (use for initialization)
            
        Returns:
            psi: Arc length of closest point on path
        """
        if self.path_x is None:
            return 0.0
        
        # For initialization or when explicitly requested, do global search
        if force_global_search or not hasattr(self, '_psi_initialized') or not self._psi_initialized:
            distances = np.sqrt((self.path_x - x)**2 + (self.path_y - y)**2)
            closest_idx = np.argmin(distances)
            self._psi_initialized = True
            return self.path_s[closest_idx]
        
        if search_window is None:
            search_window = 40.0  # Search within 40m of current position
        
        # Get current arc length
        current_psi = self._current_psi
        
        # Create search range around current position
        # For closed tracks, handle wrapping
        n_points = len(self.path_s)
        
        # Find indices within the search window
        if self.path_length > 0:
            # Search forward and backward from current position
            psi_min = current_psi - 10.0  # 10m behind
            psi_max = current_psi + search_window  # search_window ahead
            
            # Handle wrapping for closed tracks
            valid_mask = np.zeros(n_points, dtype=bool)
            for i, s in enumerate(self.path_s):
                # Check if this point is within search window
                # Account for wrapping
                dist_forward = (s - current_psi) % self.path_length
                dist_backward = (current_psi - s) % self.path_length
                
                if dist_forward <= search_window or dist_backward <= 10.0:
                    valid_mask[i] = True
        else:
            # Non-looped track - simple range check
            valid_mask = (self.path_s >= current_psi - 10.0) & (self.path_s <= current_psi + search_window)
        
        # If no valid points, fall back to full search
        if not np.any(valid_mask):
            distances = np.sqrt((self.path_x - x)**2 + (self.path_y - y)**2)
            closest_idx = np.argmin(distances)
            return self.path_s[closest_idx]
        
        # Compute distances only for valid points
        valid_indices = np.where(valid_mask)[0]
        distances = np.sqrt((self.path_x[valid_indices] - x)**2 + (self.path_y[valid_indices] - y)**2)
        closest_local_idx = np.argmin(distances)
        closest_idx = valid_indices[closest_local_idx]
        
        return self.path_s[closest_idx]
    
    def solve_control_problem(self, robot_state, control_ref=None, nearest_obs=None):
        """
        Solve the MPCC optimization.
        
        Args:
            robot_state: Full state [x, y, theta, r, beta, V, delta, tau]
            
        Returns:
            u: Control input [delta_dot, tau_dot]
        """
        # Sync path parameter with robot's actual position
        # This ensures reference doesn't run away from the robot
        robot_x = robot_state[0, 0] if robot_state.ndim > 1 else robot_state[0]
        robot_y = robot_state[1, 0] if robot_state.ndim > 1 else robot_state[1]
        self._current_psi = self._find_closest_path_point(robot_x, robot_y)
        
        # Build MPC state: [x, y, theta, r, beta, V, delta, tau, psi]
        mpc_state = np.vstack([robot_state[:8], [[self._current_psi]]])
        
        self.mpc.x0 = mpc_state
        self.mpc.set_initial_guess()
        
        try:
            u_mpc = self.mpc.make_step(mpc_state)
            self.status = 'optimal'
        except Exception as e:
            print(f"MPC solve failed: {e}")
            self.status = 'infeasible'
            return np.zeros((2, 1))
        
        # Store predictions
        self._store_predictions()
        
        # Update simulator
        y_next = self.simulator.make_step(u_mpc)
        x_next = self.estimator.make_step(y_next)
        
        return u_mpc[:2]
    
    def _store_predictions(self):
        """Store predictions for visualization and gatekeeper."""
        try:
            x_pred = self.mpc.data.prediction(('_x', 'x'))
            if x_pred is not None and x_pred.size > 0:
                if x_pred.ndim >= 3:
                    x_pred = x_pred[:, :, 0] if x_pred.ndim == 3 else x_pred[:, :, 0, 0]
                self.predicted_states = x_pred[:2, :]  # x, y positions for visualization
                
                # Store full state trajectory for gatekeeper (8 states: x,y,theta,r,beta,V,delta,tau)
                self.predicted_full_states = x_pred[:8, :]  # (8, horizon+1)
        except:
            pass
        
        try:
            u_pred = self.mpc.data.prediction(('_u', 'u'))
            if u_pred is not None and u_pred.size > 0:
                if u_pred.ndim >= 3:
                    u_pred = u_pred[:, :, 0] if u_pred.ndim == 3 else u_pred[:, :, 0, 0]
                self.predicted_inputs = u_pred[:2, :]  # (2, horizon) - delta_dot, tau_dot
        except:
            pass
    
    def get_predictions(self):
        """Get stored predictions."""
        return self.predicted_states, self.predicted_inputs
    
    def get_full_predictions(self):
        """
        Get full state and control predictions for gatekeeper.
        
        Returns:
            states: Full state trajectory (8, horizon+1) or None
            controls: Control trajectory (2, horizon) or None
        """
        states = getattr(self, 'predicted_full_states', None)
        controls = self.predicted_inputs
        return states, controls
    
    def get_reference_horizon(self):
        """Get reference path over the current horizon (for visualization)."""
        return self.reference_horizon
    
    def set_cost_weights(self, Q_c=None, Q_l=None, Q_theta=None, Q_v=None, Q_r=None, R=None, v_ref=None):
        """Update cost weights and rebuild MPC."""
        if Q_c is not None:
            self.Q_c = Q_c
        if Q_l is not None:
            self.Q_l = Q_l
        if Q_theta is not None:
            self.Q_theta = Q_theta
        if Q_v is not None:
            self.Q_v = Q_v
        if Q_r is not None:
            self.Q_r = Q_r
        if v_ref is not None:
            self.v_ref = v_ref
        if R is not None:
            self.R = np.array(R)
        self.setup_control_problem()
    
    def set_progress_rate(self, v_psi_ref):
        """Set desired progress rate."""
        self.v_psi_ref = v_psi_ref
