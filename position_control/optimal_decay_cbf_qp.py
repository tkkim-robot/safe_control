import numpy as np
import cvxpy as cp

class NotCompatibleError(Exception):
    '''
    Exception raised for errors when the robot model is not compatible with the controller.
    '''

    def __init__(self, message="Currently not compatible with the robot model."):
        self.message = message
        super().__init__(self.message)
        
class OptimalDecayCBFQP:
    def __init__(self, robot, robot_spec):
        self.robot = robot
        self.robot_spec = robot_spec
        if self.robot_spec['model'] == 'DynamicUnicycle2D': # TODO: not compatible with other robot models yet
            self.cbf_param = {}
            self.cbf_param['alpha1'] = 0.5
            self.cbf_param['alpha2'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
            self.cbf_param['omega2'] = 1.0  # Initial omega
            self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            self.cbf_param = {}
            self.cbf_param['alpha1'] = 0.5
            self.cbf_param['alpha2'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
            self.cbf_param['omega2'] = 1.0  # Initial omega
            self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay
        elif self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            self.cbf_param = {}            
            self.cbf_param['alpha'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
        elif self.robot_spec['model'] == 'Quad2D':
            self.cbf_param = {}            
            self.cbf_param['alpha1'] = 0.5
            self.cbf_param['alpha2'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
            self.cbf_param['omega2'] = 1.0  # Initial omega
            self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay
        elif self.robot_spec['model'] == 'Quad3D':
            self.cbf_param = {}            
            self.cbf_param['alpha'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay         
        else:
            raise NotCompatibleError("Infeasible or Collision")

        self.setup_control_problem()

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.omega1 = cp.Variable((1, 1))  # Optimal-decay parameter
        self.omega2 = cp.Variable((1, 1))  # Optimal-decay parameter
        self.A1 = cp.Parameter((1, 2), value=np.zeros((1, 2)))
        self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
        self.h = cp.Parameter((1, 1), value=np.zeros((1, 1)))
        self.h_dot = cp.Parameter((1, 1), value=np.zeros((1, 1)))
        
        if self.robot_spec['model'] in ['KinematicBicycle2D_C3BF', 'Quad3D']:
            objective = cp.Minimize(
                cp.sum_squares(self.u - self.u_ref) +
                self.cbf_param['p_sb1'] * cp.square(self.omega1 - self.cbf_param['omega1'])
            )
        else:
            objective = cp.Minimize(
                cp.sum_squares(self.u - self.u_ref) +
                self.cbf_param['p_sb1'] * cp.square(self.omega1 - self.cbf_param['omega1']) +
                self.cbf_param['p_sb2'] * cp.square(self.omega2 - self.cbf_param['omega2'])
            )
        
        # objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref) 
        #                         + self.cbf_param['p_sb1'] * cp.square(self.omega1 - self.cbf_param['omega1'])
        #                         + self.cbf_param['p_sb2'] * cp.square(self.omega2 - self.cbf_param['omega2']))

        if self.robot_spec['model'] == 'DynamicUnicycle2D':
            constraints = [
                self.A1 @ self.u + self.b1 + 
                (self.cbf_param['alpha1'] + self.cbf_param['alpha2'])* self.omega1 @ self.h_dot +
                self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * self.h @ self.omega2 >= 0,
                cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                cp.abs(self.u[1]) <= self.robot_spec['w_max'],
            ]
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            constraints = [
                self.A1 @ self.u + self.b1 + 
                (self.cbf_param['alpha1'] + self.cbf_param['alpha2'])* self.omega1 @ self.h_dot +
                self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * self.h @ self.omega2 >= 0,
                cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                cp.abs(self.u[1]) <= self.robot_spec['beta_max'],
            ]
        elif self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            constraints = [
                self.A1 @ self.u + self.b1 + self.cbf_param['alpha'] * self.h @ self.omega1 >= 0,
                cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                cp.abs(self.u[1]) <= self.robot_spec['beta_max'],
            ]
        elif self.robot_spec['model'] == 'Quad2D':
            constraints = [
                self.A1 @ self.u + self.b1 + 
                (self.cbf_param['alpha1'] + self.cbf_param['alpha2']) * self.omega1 @ self.h_dot + 
                self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * self.h @ self.omega2 >= 0,
                self.u[0] >= self.robot_spec['f_min'],
                self.u[0] <= self.robot_spec['f_max'],
                self.u[1] >= self.robot_spec['f_min'],
                self.u[1] <= self.robot_spec['f_max'],
            ]
        elif self.robot_spec['model'] == 'Quad3D':
            self.u = cp.Variable((4, 1))
            self.u_ref = cp.Parameter((4, 1), value=np.zeros((4, 1)))
            self.A1 = cp.Parameter((1, 4), value=np.zeros((1, 4)))
            self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            self.h = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            self.omega1 = cp.Variable((1, 1))
            constraints = [
                self.A1 @ self.u + self.b1 + self.cbf_param['alpha'] * self.h @ self.omega1 >= 0,
                self.u[0] >= 0.0,
                self.u[0] <= self.robot_spec['f_max'],
                cp.abs(self.u[1]) <= self.robot_spec['phi_dot_max'],
                cp.abs(self.u[2]) <= self.robot_spec['theta_dot_max'],
                cp.abs(self.u[3]) <= self.robot_spec['psi_dot_max'],
            ]

        self.cbf_controller = cp.Problem(objective, constraints)

    def solve_control_problem(self, robot_state, control_ref, nearest_obs):
        # Update the CBF constraints
        if nearest_obs is None:
            self.A1.value = np.zeros_like(self.A1.value)
            self.b1.value = np.zeros_like(self.b1.value)
            self.h.value = np.zeros_like(self.h.value)
            self.h_dot.value = np.zeros_like(self.h_dot.value)
        elif self.robot_spec['model'] in ['KinematicBicycle2D_C3BF', 'Quad3D']:
            h, dh_dx = self.robot.agent_barrier(nearest_obs)
            self.A1.value[0,:] = dh_dx @ self.robot.g()
            self.b1.value[0,:] = dh_dx @ self.robot.f()
            self.h.value[0,:] = h
        elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'Quad2D']:
            h, h_dot, dh_dot_dx = self.robot.agent_barrier(nearest_obs)
            self.A1.value[0,:] = dh_dot_dx @ self.robot.g()
            self.b1.value[0,:] = dh_dot_dx @ self.robot.f()
            self.h.value[0,:] = h
            self.h_dot.value[0,:] = h_dot

        # print(self.omega1.value, self.omega2.value)

        self.u_ref.value = control_ref['u_ref']

        # Solve the optimization problem
        self.cbf_controller.solve(solver=cp.GUROBI)
        self.status = self.cbf_controller.status
        
        return self.u.value