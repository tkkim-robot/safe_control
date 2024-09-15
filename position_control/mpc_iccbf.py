import numpy as np
import casadi as ca
import do_mpc

import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'probabilistic_ensemble_nn'))
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'DistributionallyRobustCVaR'))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from position_control.probabilistic_ensemble_nn.dynamics.nn_vehicle import ProbabilisticEnsembleNN
from position_control.DistributionallyRobustCVaR.distributionally_robust_cvar import DistributionallyRobustCVaR, plot_gmm_with_cvar
from sklearn.preprocessing import MinMaxScaler



class AdaptiveCBFParameterSelector:
    def __init__(self, model_name, scaler_name, distance_margin=0.07, step_size=0.02, epistemic_threshold=0.2):
        self.penn = ProbabilisticEnsembleNN()
        self.penn.load_model(model_name)
        self.penn.load_scaler(scaler_name)
        self.lower_bound = 0.01
        self.upper_bound = 0.2
        self.distance_margin = distance_margin
        self.step_size = step_size
        self.epistemic_threshold = epistemic_threshold

    def sample_cbf_parameters(self, current_gamma1, current_gamma2):
        gamma1_range = np.arange(max(self.lower_bound, current_gamma1 - 0.2), min(self.upper_bound, current_gamma1 + 0.2 + self.step_size), self.step_size)
        gamma2_range = np.arange(max(self.lower_bound, current_gamma2 - 0.2), min(self.upper_bound, current_gamma2 + 0.2 + self.step_size), self.step_size)
        return gamma1_range, gamma2_range

    def get_rel_state_wt_obs(self, controller, robot_state, nearest_obs):
        robot_pos = robot_state[:2, 0].flatten()
        robot_radius = controller.robot.robot_radius
        try:
            near_obs = nearest_obs.flatten()
        except:
            near_obs = [100, 100, 0.2]
        
        distance = np.linalg.norm(robot_pos - near_obs[:2]) - 0.45 + robot_radius + near_obs[2]
        velocity = robot_state[3][0]
        theta = np.arctan2(near_obs[1] - robot_pos[1], near_obs[0] - robot_pos[0])
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        gamma1 = controller.cbf_param['alpha1']
        gamma2 = controller.cbf_param['alpha2']
        
        return [distance, velocity, theta, gamma1, gamma2], robot_pos, robot_radius, near_obs
        # return [distance, 7, 9, velocity, theta, gamma1, gamma2]

    def predict_with_penn(self, current_state, gamma1_range, gamma2_range):
        batch_input = []
        for gamma1 in gamma1_range:
            for gamma2 in gamma2_range:
                state = current_state.copy()
                state[3] = gamma1
                state[4] = gamma2
                # state[5] = gamma1
                # state[6] = gamma2
                batch_input.append(state)
        
        batch_input = np.array(batch_input)
        y_pred_safety_loss, y_pred_deadlock_time, epistemic_uncertainty = self.penn.predict(batch_input)
        predictions = []

        for i, (gamma1, gamma2) in enumerate(zip(gamma1_range.repeat(len(gamma2_range)), np.tile(gamma2_range, len(gamma1_range)))):
            predictions.append((gamma1, gamma2, y_pred_safety_loss[i], y_pred_deadlock_time[i][0], epistemic_uncertainty[i]))

        return predictions

    def filter_by_epistemic_uncertainty(self, predictions):
        epistemic_uncertainties = [pred[4] for pred in predictions]
        if all(pred > 1.0 for pred in epistemic_uncertainties):
            filtered_predictions = []
            # print("High epistemic uncertainty detected. Filtering out all predictions.")
        else:
            scaler = MinMaxScaler()
            normalized_epistemic_uncertainties = scaler.fit_transform(np.array(epistemic_uncertainties).reshape(-1, 1)).flatten()
            filtered_predictions = [pred for pred, norm_uncert in zip(predictions, normalized_epistemic_uncertainties) if norm_uncert <= self.epistemic_threshold]
        return filtered_predictions

    def calculate_cvar_boundary(self, robot_radius, near_obs):
        alpha_1 = 0.4
        beta_1 = 100.0 
        # beta_2 = 2.5
        # delta_theta = np.arctan2(near_obs[1] - robot_pos[1], near_obs[0] - robot_pos[0]) - theta
        # distance = np.linalg.norm(robot_pos - near_obs[:2])
        min_distance = self.distance_margin
        cvar_boundary = alpha_1 / (beta_1 * min_distance**2 + 1)
        return cvar_boundary

    def filter_by_aleatoric_uncertainty(self, filtered_predictions, robot_pos, robot_radius, theta, near_obs):
        final_predictions = []
        cvar_boundary = self.calculate_cvar_boundary(robot_radius, near_obs)
        for pred in filtered_predictions:
            _, _, y_pred_safety_loss, _, _ = pred
            gmm = self.penn.create_gmm(y_pred_safety_loss)
            cvar_filter = DistributionallyRobustCVaR(gmm)
            
            # dr_cvar, cvar_values, dr_cvar_index = cvar_filter.compute_dr_cvar(alpha=0.95)
            # within_boundary = cvar_filter.is_within_boundary(cvar_boundary, alpha=0.95)
            # print(f"Distributionally Robust CVaR: {dr_cvar} | boundary: {cvar_boundary}")
            # if within_boundary:
            #     print(f"Distributionally Robust CVaR: {dr_cvar} | boundary: {cvar_boundary}")

            if cvar_filter.is_within_boundary(cvar_boundary):
                final_predictions.append(pred)
        return final_predictions

    def select_best_parameters(self, final_predictions, controller):
        # If no predictions were selected, gradually decrease the parameter
        if not final_predictions:
            current_gamma1 = controller.cbf_param['alpha1']
            current_gamma2 = controller.cbf_param['alpha2']
            gamma1 = max(self.lower_bound, current_gamma1 - 0.02)
            gamma2 = max(self.lower_bound, current_gamma2 - 0.02)
            return gamma1, gamma2
        min_deadlock_time = min(final_predictions, key=lambda x: x[3])[3]
        best_predictions = [pred for pred in final_predictions if pred[3][0] < 1e-3]
        # If no predictions under 1e-3, use the minimum deadlock time
        if not best_predictions:
            best_predictions = [pred for pred in final_predictions if pred[3] == min_deadlock_time]
        # If there are multiple best predictions, use harmonic mean to select the best one
        if len(best_predictions) != 1:
            best_prediction = max(best_predictions, key=lambda x: 2 * (x[0] * x[1]) / (x[0] + x[1]) if (x[0] + x[1]) != 0 else 0)
            return best_prediction[0], best_prediction[1]
        return best_predictions[0][0], best_predictions[0][1]

    def adaptive_parameter_selection(self, controller, robot_state, nearest_obs):
        current_state, robot_pos, robot_radius, near_obs = self.get_rel_state_wt_obs(controller, robot_state, nearest_obs)
        gamma1_range, gamma2_range = self.sample_cbf_parameters(current_state[3], current_state[4])
        # gamma1_range, gamma2_range = self.sample_cbf_parameters(current_state[5], current_state[6])
        predictions = self.predict_with_penn(current_state, gamma1_range, gamma2_range)
        filtered_predictions = self.filter_by_epistemic_uncertainty(predictions)
        final_predictions = self.filter_by_aleatoric_uncertainty(filtered_predictions, robot_pos, robot_radius, current_state[2], near_obs)
        best_gamma1, best_gamma2 = self.select_best_parameters(final_predictions, controller)
        if best_gamma1 is not None and best_gamma2 is not None:
            print(f"CBF parameters updated to: {best_gamma1:.2f}, {best_gamma2:.2f} | Total prediction count: {len(predictions)} | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric DR-CVaR")        
        else:
            print(f"CBF parameters updated to: NONE, NONE | Total prediction count: {len(predictions)} | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric DR-CVaR")        
            
        return best_gamma1, best_gamma2








class MPC_ICCBF:
    def __init__(self, robot, robot_spec):
        self.robot = robot
        self.robot_spec = robot_spec
        self.status = 'optimal'  # TODO: not implemented

        # MPC parameters
        self.horizon = 10
        self.dt = robot.dt

        # Cost function weights
        if self.robot_spec['model'] == 'Unicycle2D':
            self.Q = np.diag([50, 50, 0.01])  # State cost matrix
            self.R = np.array([0.5, 0.5])  # Input cost matrix
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            self.Q = np.diag([50, 50, 0.01, 30])  # State cost matrix
            self.R = np.array([0.5, 0.5])  # Input cost matrix
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            self.Q = np.diag([50, 50, 20, 20])  # State cost matrix
            self.R = np.array([0.5, 0.5])  # Input cost matrix

        # DT CBF parameters should scale from 0 to 1
        self.cbf_param = {}
        if self.robot_spec['model'] == 'Unicycle2D':
            self.cbf_param['alpha'] = 0.05
            self.n_states = 3
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            self.cbf_param['alpha1'] = 0.15
            self.cbf_param['alpha2'] = 0.15
            self.n_states = 4
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            self.cbf_param['alpha1'] = 0.15
            self.cbf_param['alpha2'] = 0.15
            self.n_states = 4
        self.n_controls = 2

        self.goal = np.array([0, 0])
        self.obs = None

        self.adaptive_selector = AdaptiveCBFParameterSelector('penn_model_0907.pth', 'scaler_0907.save')

        self.setup_control_problem()

    def setup_control_problem(self):
        self.model = self.create_model()
        self.mpc = self.create_mpc()
        self.simulator = self.create_simulator()
        self.estimator = do_mpc.estimator.StateFeedback(self.model)

    def create_model(self):
        model = do_mpc.model.Model('discrete')

        # States
        _x = model.set_variable(
            var_type='_x', var_name='x', shape=(self.n_states, 1))

        # Inputs
        _u = model.set_variable(
            var_type='_u', var_name='u', shape=(self.n_controls, 1))

        # Parameters
        _goal = model.set_variable(
            var_type='_tvp', var_name='goal', shape=(self.n_states, 1))
        _obs = model.set_variable(
            var_type='_tvp', var_name='obs', shape=(5, 3))

        if self.robot_spec['model'] == 'Unicycle2D':
            _alpha = model.set_variable(
                var_type='_tvp', var_name='alpha', shape=(1, 1))
        elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D']:
            _alpha1 = model.set_variable(
                var_type='_tvp', var_name='alpha1', shape=(1, 1))
            _alpha2 = model.set_variable(
                var_type='_tvp', var_name='alpha2', shape=(1, 1))

        # System dynamics
        f_x = self.robot.f_casadi(_x)
        g_x = self.robot.g_casadi(_x)
        x_next = _x + (f_x + ca.mtimes(g_x, _u)) * self.dt

        # Set right hand side of ODE
        model.set_rhs('x', x_next)

        # Defines the objective function wrt the state cost
        cost = ca.mtimes([(_x - _goal).T, self.Q, (_x - _goal)])
        model.set_expression(expr_name='cost', expr=cost)

        model.setup()
        return model

    def create_mpc(self):
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

        # Configure objective function
        mterm = self.model.aux['cost']  # Terminal cost
        lterm = self.model.aux['cost']  # Stage cost
        mpc.set_objective(mterm=mterm, lterm=lterm)
        # Input penalty (R diagonal matrix in objective fun)
        mpc.set_rterm(u=self.R)

        # State and input bounds
        if self.robot_spec['model'] == 'Unicycle2D':
            mpc.bounds['lower', '_u', 'u'] = np.array(
                [-self.robot_spec['v_max'], -self.robot_spec['w_max']])
            mpc.bounds['upper', '_u', 'u'] = np.array(
                [self.robot_spec['v_max'], self.robot_spec['w_max']])
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            mpc.bounds['lower', '_x', 'x', 3] = -self.robot_spec['v_max']
            mpc.bounds['upper', '_x', 'x', 3] = self.robot_spec['v_max']
            mpc.bounds['lower', '_u', 'u'] = np.array(
                [-self.robot_spec['a_max'], -self.robot_spec['w_max']])
            mpc.bounds['upper', '_u', 'u'] = np.array(
                [self.robot_spec['a_max'], self.robot_spec['w_max']])
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            mpc.bounds['lower', '_u', 'u'] = np.array(
                [-self.robot_spec['ax_max'], -self.robot_spec['ay_max']])
            mpc.bounds['upper', '_u', 'u'] = np.array(
                [self.robot_spec['ax_max'], self.robot_spec['ay_max']])

        mpc = self.set_tvp(mpc)
        mpc = self.set_cbf_constraint(mpc)

        mpc.setup()
        return mpc

    def set_tvp(self, mpc):
        # Set time-varying parameters
        def tvp_fun(t_now):
            tvp_template = mpc.get_tvp_template()

            # Set goal
            tvp_template['_tvp', :, 'goal'] = np.concatenate([self.goal, [0] * (self.n_states - 2)])

            # Handle up to 5 obstacles (if fewer than 5, substitute dummy obstacles)
            if self.obs is None:
                # Before detecting any obstacle, set 5 dummy obstacles far away
                dummy_obstacles = np.tile(np.array([1000, 1000, 0]), (5, 1))  # 5 far away obstacles
                tvp_template['_tvp', :, 'obs'] = dummy_obstacles
            else:
                num_obstacles = self.obs.shape[0]
                if num_obstacles < 5:
                    # Add dummy obstacles for missing ones
                    dummy_obstacles = np.tile(np.array([1000, 1000, 0]), (5 - num_obstacles, 1))
                    tvp_template['_tvp', :, 'obs'] = np.vstack([self.obs, dummy_obstacles])
                else:
                    # Use the detected obstacles directly
                    tvp_template['_tvp', :, 'obs'] = self.obs[:5, :]  # Limit to 5 obstacles

            if self.robot_spec['model'] == 'Unicycle2D':
                tvp_template['_tvp', :, 'alpha'] = self.cbf_param['alpha']
            elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D']:
                tvp_template['_tvp', :, 'alpha1'] = self.cbf_param['alpha1']
                tvp_template['_tvp', :, 'alpha2'] = self.cbf_param['alpha2']

            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        return mpc

    def set_cbf_constraint(self, mpc):
        _x = self.model.x['x']
        _u = self.model.u['u']  # Current control input [0] acc, [1] omega
        _obs = self.model.tvp['obs']

        # Add a separate constraint for each of the 5 obstacles
        for i in range(5):
            obs_i = _obs[i, :]  # Select the i-th obstacle
            cbf_constraint = self.compute_cbf_constraint(_x, _u, obs_i)
            mpc.set_nl_cons(f'cbf_{i}', -cbf_constraint, ub=0)

        return mpc

    def compute_cbf_constraint(self, _x, _u, _obs):
        '''compute cbf constraint value
        We reuse this function to print the CBF constraint'''

        if self.robot_spec['model'] == 'Unicycle2D':
            _alpha = self.model.tvp['alpha']
            h_k, d_h = self.robot.agent_barrier_dt(_x, _u, _obs)
            cbf_constraint = d_h + _alpha * h_k
        elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D']:
            _alpha1 = self.model.tvp['alpha1']
            _alpha2 = self.model.tvp['alpha2']
            h_k, d_h, dd_h = self.robot.agent_barrier_dt(_x, _u, _obs)
            cbf_constraint = dd_h + (_alpha1 + _alpha2) * \
                d_h + _alpha1 * _alpha2 * h_k
        else:
            raise NotImplementedError('Model not implemented')

        return cbf_constraint

    def create_simulator(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.dt)
        tvp_template = simulator.get_tvp_template()

        def tvp_fun(t_now):
            return tvp_template
        simulator.set_tvp_fun(tvp_fun)
        simulator.setup()
        return simulator

    def update_tvp(self, goal, obs):
        # Update the tvp variables
        self.goal = np.array(goal)
        
        if obs is None or len(obs) == 0:
            # No obstacles detected, set 5 dummy obstacles far away
            self.obs = np.tile(np.array([1000, 1000, 0]), (5, 1))
        else:
            num_obstacles = len(obs)
            if num_obstacles < 5:
                # Add dummy obstacles for missing ones
                dummy_obstacles = np.tile(np.array([1000, 1000, 0]), (5 - num_obstacles, 1))
                self.obs = np.vstack([obs, dummy_obstacles])
            else:
                # Use the detected obstacles directly (up to 5)
                self.obs = np.array(obs[:5])

    def solve_control_problem(self, robot_state, control_ref, nearest_obs):
        nearest_obs = nearest_obs[0]
        best_gamma1, best_gamma2 = self.adaptive_selector.adaptive_parameter_selection(self, robot_state, nearest_obs)
        self.cbf_param['alpha1'] = best_gamma1
        self.cbf_param['alpha2'] = best_gamma2
        
        # Set initial state and reference
        self.mpc.x0 = robot_state
        self.mpc.set_initial_guess()
        goal = control_ref['goal']
        self.update_tvp(goal, nearest_obs)

        
        if control_ref['state_machine'] != 'track':
            # if outer loop is doing something else, just return the reference
            return control_ref['u_ref']

        # Solve MPC problem
        u = self.mpc.make_step(robot_state)

        # Update simulator and estimator
        y_next = self.simulator.make_step(u)
        x_next = self.estimator.make_step(y_next)

        # if nearest_obs is not None:
        #     cbf_constraint = self.compute_cbf_constraint(
        #         x_next, u, nearest_obs)  # here use actual value, not symbolic
        # self.status = 'optimal' if self.mpc.optimal else 'infeasible'
        # print(self.mpc.opt_x_num['_x', :, 0, 0])
        return u
