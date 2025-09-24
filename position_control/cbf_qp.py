import numpy as np
import cvxpy as cp

class CBFQP:
    def __init__(self, robot, robot_spec, num_obs=1): # When solving a QP with constraints for n obstacles, set 'num_obs=n'
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs

        self.cbf_param = {}

        if self.robot_spec['model'] == "SingleIntegrator2D":
            self.cbf_param['alpha'] = 1.0
        elif self.robot_spec['model'] == 'Unicycle2D':
            self.cbf_param['alpha'] = 1.0
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_DPCBF':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'Quad2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'Quad3D':
            self.cbf_param['alpha'] = 1.5

        self.setup_control_problem()

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.A1 = cp.Parameter((self.num_obs, 2), value=np.zeros((self.num_obs, 2)))
        self.b1 = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))

        if self.robot_spec['model'] == 'SingleIntegrator2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <=  self.robot_spec['v_max'],
                           cp.abs(self.u[1]) <=  self.robot_spec['v_max']]
        elif self.robot_spec['model'] == 'Unicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['v_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['a_max']]
        elif 'KinematicBicycle2D' in self.robot_spec['model']:
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['beta_max']]
        elif self.robot_spec['model'] == 'Quad2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           self.robot_spec["f_min"] <= self.u[0],
                           self.u[0] <= self.robot_spec["f_max"],
                           self.robot_spec["f_min"] <= self.u[1],
                           self.u[1] <= self.robot_spec["f_max"]]
        elif self.robot_spec['model'] == 'Quad3D':
            # overwrite the variables
            self.u = cp.Variable((4, 1))
            self.u_ref = cp.Parameter((4, 1), value=np.zeros((4, 1)))
            self.A1 = cp.Parameter((1, 4), value=np.zeros((1, 4)))
            self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           self.u[0] <= self.robot_spec['f_max'],
                            self.u[0] >= 0.0,
                           cp.abs(self.u[1]) <= self.robot_spec['phi_dot_max'],
                           cp.abs(self.u[2]) <= self.robot_spec['theta_dot_max'],
                           cp.abs(self.u[3]) <= self.robot_spec['psi_dot_max']]

        self.cbf_controller = cp.Problem(objective, constraints)

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        for i in range(min(self.num_obs, len(obs_list))):
            obs = obs_list[i]
            # 3. Update the CBF constraints
            if obs is None:
                # deactivate the CBF constraints
                self.A1.value = np.zeros_like(self.A1.value)
                self.b1.value = np.zeros_like(self.b1.value)
            elif self.robot_spec['model'] in ['SingleIntegrator2D', 'Unicycle2D', 'KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF', 'Quad3D']:
                h, dh_dx = self.robot.agent_barrier(obs)
                self.A1.value[i,:] = dh_dx @ self.robot.g()
                self.b1.value[i,:] = dh_dx @ self.robot.f() + self.cbf_param['alpha'] * h
            elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D', 'KinematicBicycle2D', 'Quad2D']:
                h, h_dot, dh_dot_dx = self.robot.agent_barrier(obs)
                self.A1.value[i,:] = dh_dot_dx @ self.robot.g()
                self.b1.value[i,:] = dh_dot_dx @ self.robot.f() + (self.cbf_param['alpha1']+self.cbf_param['alpha2']) * h_dot + self.cbf_param['alpha1']*self.cbf_param['alpha2']*h
            
            #print(f'h: {h} | value: {self.A1.value[i,:] @ self.u.value + self.b1.value[i,:]}')
        
        self.u_ref.value = control_ref['u_ref']

        # 4. Solve this yields a new 'self.u'
        self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)

        # print(f'h: {h} | value: {self.A1.value[0,:] @ self.u.value + self.b1.value[0,:]}')
        
        # Check QP error in tracking.py
        self.status = self.cbf_controller.status
        # if self.cbf_controller.status != 'optimal':
        #     raise QPError("CBF-QP optimization failed")

        return self.u.value