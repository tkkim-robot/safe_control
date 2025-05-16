import numpy as np
import casadi as ca
import do_mpc
import copy

class NotCompatibleError(Exception):
    '''
    Exception raised for errors when the robot model is not compatible with the controller.
    '''

    def __init__(self, message="Currently not compatible with the robot model. Only compatible with DynamicUnicycle2D now"):
        self.message = message
        super().__init__(self.message)

class OptimalDecayMPCCBF:
    def __init__(self, robot, robot_spec):
        self.robot = robot
        self.robot_spec = robot_spec
        if self.robot_spec['model'] not in ['DynamicUnicycle2D', 'KinematicBicycle2D']: # TODO: not compatible with other robot models yet
            raise NotCompatibleError("Infeasible or Collision")
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
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            self.Q = np.diag([50, 50, 1, 1])  # State cost matrix
            self.R = np.array([0.5, 50.0])  # Input cost matrix
            
        # DT CBF parameters should scale from 0 to 1
        self.cbf_param = {}
        if self.robot_spec['model'] == 'Unicycle2D':
            self.cbf_param['alpha'] = 0.01
            self.n_states = 3
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            self.cbf_param['alpha1'] = 0.01
            self.cbf_param['alpha2'] = 0.01
            self.n_states = 4
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            self.cbf_param['alpha1'] = 0.01
            self.cbf_param['alpha2'] = 0.01
            self.n_states = 4
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            self.cbf_param['alpha1'] = 0.05
            self.cbf_param['alpha2'] = 0.05
            self.n_states = 4
        self.n_controls = 2

        # Optimal-decay parameters
        self.cbf_param['omega1'] = 1.0  # Initial omega1
        self.cbf_param['p_sb1'] = 10**1  # Penalty parameter for omega1
        self.cbf_param['omega2'] = 1.0  # Initial omega2
        self.cbf_param['p_sb2'] = 10**1  # Penalty parameter for omega2
        self.omega1 = None
        self.omega2 = None

        self.goal = np.array([0, 0])
        self.obs = None

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

        # Add omega parameters
        _omega1 = model.set_variable(var_type='_u', var_name='omega1', shape=(1, 1))
        _omega2 = model.set_variable(var_type='_u', var_name='omega2', shape=(1, 1))

        if self.robot_spec['model'] == 'Unicycle2D':
            _alpha = model.set_variable(
                var_type='_tvp', var_name='alpha', shape=(1, 1))
        elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D', 'KinematicBicycle2D']:
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
        # cost = ca.mtimes([(_x - _goal).T, self.Q, (_x - _goal)]) + \
        #     self.cbf_param['p_sb1'] * ca.sumsqr(_omega1 - self.cbf_param['omega1']) + \
        #     self.cbf_param['p_sb2'] * ca.sumsqr(_omega2 - self.cbf_param['omega2'])
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
        expression = sum(self.R[i] * self.model.u['u'][i]**2 for i in range(self.n_controls))
        control_input_rterm = expression
        omega1_rterm = self.cbf_param['p_sb1'] * (self.model.u['omega1'] - self.cbf_param['omega1'])**2
        omega2_rterm = self.cbf_param['p_sb2'] * (self.model.u['omega2'] - self.cbf_param['omega2'])**2
        mpc.set_rterm(control_input_rterm)
        mpc.set_rterm(omega1_rterm + omega2_rterm)
        # MPC.set_rterm(u=self.R)


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
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            mpc.bounds['lower', '_x', 'x', 3] = -self.robot_spec['v_max']
            mpc.bounds['upper', '_x', 'x', 3] = self.robot_spec['v_max']
            mpc.bounds['lower', '_u', 'u'] = np.array(
                [-self.robot_spec['a_max'], -self.robot_spec['beta_max']])
            mpc.bounds['upper', '_u', 'u'] = np.array(
                [self.robot_spec['a_max'], self.robot_spec['beta_max']])


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
            elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D', 'KinematicBicycle2D']:
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
        elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D', 'KinematicBicycle2D']:
            _alpha1 = self.model.tvp['alpha1']
            _alpha2 = self.model.tvp['alpha2']
            omega1 = self.model.u['omega1']
            omega2 = self.model.u['omega2']
            h_k, d_h, dd_h = self.robot.agent_barrier_dt(_x, _u, _obs)
            cbf_constraint = (dd_h + (_alpha1 * omega1 + _alpha2 * omega2) * d_h 
                            + _alpha1 * _alpha2 * h_k * omega1 * omega2)
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
        
        obs = obs.reshape(-1,3)
        
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
        
        # omega1 = float(self.mpc.opt_x_num['_u', 0, 0][2])
        # omega2 = float(self.mpc.opt_x_num['_u', 0, 0][3])
        # # print(f"Omega1: {omega1}, Omega2: {omega2}")
        # print(self.mpc.opt_x_num['_u', :, 0])

        # if nearest_obs is not None:
        #     cbf_constraint = self.compute_cbf_constraint(
        #         x_next, u, nearest_obs)  # here use actual value, not symbolic
        # self.status = 'optimal' if self.mpc.optimal else 'infeasible'
        
        # print(self.mpc.opt_x_num['_u', :, 0])
        # cost_value = self.mpc.opt_aux_num['_aux'][0]  # Access the 'cost' expression
        # print(f"Cost: {cost_value}")
        
        return u[:self.n_controls]