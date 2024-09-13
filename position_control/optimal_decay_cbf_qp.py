import numpy as np
import cvxpy as cp



# FIXME: Only works for DynamicUnicycle2D for now
class OptimalDecayCBFQP:
    def __init__(self, robot, robot_spec):
        self.robot = robot
        self.robot_spec = robot_spec

        self.cbf_param = {}
        
        self.cbf_param['alpha1'] = 2.5 
        self.cbf_param['alpha2'] = 2.5
        self.cbf_param['omega1'] = 1.0  # Initial omega
        self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
        self.cbf_param['omega2'] = 1.0  # Initial omega
        self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay

        self.setup_control_problem()

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.omega1 = cp.Variable((1, 1))  # Optimal-decay parameter
        self.omega2 = cp.Variable((1, 1))  # Optimal-decay parameter
        self.A1 = cp.Parameter((1, 2), value=np.zeros((1, 2)))
        self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref) 
                                + self.cbf_param['p_sb1'] * cp.square(self.omega1 - self.cbf_param['omega1'])
                                + self.cbf_param['p_sb2'] * cp.square(self.omega2 - self.cbf_param['omega2']))

        constraints = [self.A1 @ self.u + self.b1 >= 0,
                       cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                       cp.abs(self.u[1]) <= self.robot_spec['w_max'],]

        self.cbf_controller = cp.Problem(objective, constraints)

    def solve_control_problem(self, robot_state, control_ref, nearest_obs):
        # Update the CBF constraints
        if nearest_obs is None:
            self.A1.value = np.zeros_like(self.A1.value)
            self.b1.value = np.zeros_like(self.b1.value)
        else:
            h, h_dot, dh_dot_dx = self.robot.agent_barrier(nearest_obs)
            self.A1.value[0,:] = dh_dot_dx @ self.robot.g()
            
            print(self.omega1.value, self.omega2.value)

            # self.b1.value[0,:] = dh_dot_dx @ self.robot.f() + (self.cbf_param['alpha1'] + self.cbf_param['alpha2']) * h_dot + self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * h
            self.b1.value[0, :] = dh_dot_dx @ self.robot.f() + \
                                (self.cbf_param['alpha1'] * self.omega1 + self.cbf_param['alpha2'] * self.omega2) * h_dot + \
                                self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * h * self.omega1 * self.omega2
            
            self.b1.value[0, :] = dh_dot_dx @ self.robot.f() + \
                    cp.multiply(self.cbf_param['alpha1'], self.omega1) @ h_dot + \
                    cp.multiply(self.cbf_param['alpha1'], self.cbf_param['alpha2']) @ cp.multiply(h, self.omega1, self.omega2)

        self.u_ref.value = control_ref['u_ref']

        print(self.omega1.value, self.omega2.value)

        # Solve the optimization problem
        self.cbf_controller.solve(solver=cp.GUROBI)
        self.status = self.cbf_controller.status
        
        return self.u.value
