import numpy as np
import casadi as ca
import do_mpc


class MPCCBF:
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
            var_type='_tvp', var_name='obs', shape=(3, 1))

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
            # for k in range(self.horizon + 1):
            tvp_template['_tvp', :, 'goal'] = np.concatenate(
                [self.goal, [0] * (self.n_states - 2)])
            if self.obs is None:
                # before detecting any obstacle, set a dummy obstacle far away
                tvp_template['_tvp', :, 'obs'] = np.concatenate(
                    [self.goal+1000, [0]])
            else:
                tvp_template['_tvp', :, 'obs'] = self.obs
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        return mpc

    def set_cbf_constraint(self, mpc):
        # set CBF constraint for MPC, and it will be automatically updated if you update self.tvp
        _x = self.model.x['x']
        _u = self.model.u['u']  # Current control input [0] acc, [1] omega
        _obs = self.model.tvp['obs']

        cbf_constraint = self.compute_cbf_constraint(
            _x, _u, _obs)  # here, use symbolic variable
        mpc.set_nl_cons('cbf', -cbf_constraint, ub=0)

        return mpc

    def compute_cbf_constraint(self, _x, _u, _obs):
        '''compute cbf constraint value
        We reuse this function to print the CBF constraint'''

        if self.robot_spec['model'] == 'Unicycle2D':
            h_k, d_h = self.robot.agent_barrier_dt(_x, _u, _obs)
            cbf_constraint = d_h + self.cbf_param['alpha'] * h_k
        elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D']:
            h_k, d_h, dd_h = self.robot.agent_barrier_dt(_x, _u, _obs)
            cbf_constraint = dd_h + \
                (self.cbf_param['alpha1'] + self.cbf_param['alpha2']) * d_h + \
                self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * h_k
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
        if obs is None:
            self.obs = np.concatenate([self.goal[:2]*1000, [0]])
        else:
            self.obs = np.array(obs).flatten()

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

        if nearest_obs is not None:
            cbf_constraint = self.compute_cbf_constraint(
                x_next, u, nearest_obs)  # here use actual value, not symbolic
        # self.status = 'optimal' if self.mpc.optimal else 'infeasible'

        return u
