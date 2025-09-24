from tracking import LocalTrackingController
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import subprocess
import csv

class InfeasibleError(Exception):
    '''
    Exception raised for errors when QP is infeasible or 
    the robot collides with the obstacle
    '''

    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)

class LocalTrackingControllerDyn(LocalTrackingController):

    def __init__(self, X0, robot_spec,
                 controller_type=None,
                 dt=0.05,
                 show_animation=False, save_animation=False, show_mpc_traj=False,
                 enable_rotation=True, raise_error=False,
                 ax=None, fig=None, env=None):
        super().__init__(X0, robot_spec,
                         controller_type=controller_type,
                         dt=dt,
                         show_animation=show_animation, save_animation=save_animation, show_mpc_traj=show_mpc_traj,
                         enable_rotation=enable_rotation, raise_error=raise_error,
                         ax=ax, fig=fig, env=env)
        
        # Create a list to hold the arrow patches for obstacle velocities
        self.obs_vel_arrows = []
        
        # Initialize moving obstacles
        self.dyn_obs_patch = None # will be initialized after the first step
        self.init_obs_info = None
        self.init_obs_circle = None
    

    def setup_robot(self, X0):
        from dynamic_env.robot import BaseRobotDyn
        self.robot = BaseRobotDyn(
            X0.reshape(-1, 1), self.robot_spec, self.dt, self.ax)
        
    # Update dynamic obs position
    def step_dyn_obs(self):
        """if self.obs (n,5) array (ex) [x, y, r, vx, vy], update obs position per time step"""
        if len(self.obs) != 0 and self.obs.shape[1] >= 5:
            self.obs[:, 0] += self.obs[:, 3] * self.dt
            self.obs[:, 1] += self.obs[:, 4] * self.dt
    
    def render_dyn_obs(self):
        if len(self.obs_vel_arrows) != len(self.obs):
            for arrow in self.obs_vel_arrows:
                arrow.remove()
            self.obs_vel_arrows = []
            
            if self.obs.shape[0] > 0 and self.obs.shape[1] >= 5:
                for _ in range(len(self.obs)):
                    arrow = patches.Arrow(0, 0, 0, 0, width=0.2, color='orange', zorder=5)
                    self.ax.add_patch(arrow)
                    self.obs_vel_arrows.append(arrow)

        for i, obs_info in enumerate(self.obs):
            # obs: [x, y, r, vx, vy]
            ox, oy, r = obs_info[:3]
            self.dyn_obs_patch[i].center = ox, oy
            self.dyn_obs_patch[i].set_radius(r)

            # Check if there are arrows to update
            if i < len(self.obs_vel_arrows):
                vx, vy = obs_info[3], obs_info[4]
                
                # Remove the old arrow and add a new one to update its properties
                # This is a robust way to handle patches in matplotlib animations
                self.obs_vel_arrows[i].remove()
                
                # You can scale the vector length for better visualization, e.g., multiply by 0.5
                arrow_scale = 1.0 
                new_arrow = patches.Arrow(ox, oy, vx * arrow_scale, vy * arrow_scale, 
                                          width=0.2, color='orange', zorder=5)
                
                self.ax.add_patch(new_arrow)
                self.obs_vel_arrows[i] = new_arrow

    def draw_plot(self, pause=0.01, force_save=False):
        if self.show_animation:
            if self.dyn_obs_patch is None:
                # Initialize moving obstacles
                self.dyn_obs_patch = [self.ax.add_patch(plt.Circle(
                    (0, 0), 0, edgecolor='black', facecolor='gray', fill=True)) for _ in range(len(self.obs))]
                self.init_obs_info = self.obs.copy()
                
            self.render_dyn_obs()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # move the square frame of the plot based on robot's x position
            # if self.robot_spec['model'] in ['VTOL2D']:
            #     x = np.clip(self.robot.X[0, 0], 7.5, 67.5)
            #     self.ax.set_xlim(x-7.5, x+7.5)
            #     self.ax.set_ylim(0, 15)
            #     self.fig.tight_layout()
                
            plt.pause(pause)
            if self.save_animation:
                self.ani_idx += 1
                if force_save or self.ani_idx % self.save_per_frame == 0:
                    plt.savefig(self.current_directory_path +
                                "/output/animations/" + "t_step_" + str(self.ani_idx//self.save_per_frame).zfill(4) + ".png", dpi=300)
                    # plt.savefig(self.current_directory_path +
                    #             "/output/animations/" + "t_step_" + str(self.ani_idx//self.save_per_frame).zfill(4) + ".svg")

    def control_step(self):
        '''
        Simulate one step of tracking control with CBF-QP with the given waypoints.
        Output: 
            - -2 or QPError: if the QP is infeasible or the robot collides with the obstacle
            - -1: all waypoints reached
            - 0: normal
            - 1: visibility violation
        '''
        # update state machine
        if self.state_machine == 'stop':
            if self.robot.has_stopped():
                if self.enable_rotation:
                    self.state_machine = 'rotate'
                else:
                    self.state_machine = 'track'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        # 1. Update the detected obstacles
        detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)
        # self.nearest_obs = self.get_nearest_obs(detected_obs)
        self.nearest_multi_obs = self.get_nearest_unpassed_obs(detected_obs, obs_num=self.num_constraints)
        if self.nearest_multi_obs is not None:
            self.nearest_obs = self.nearest_multi_obs[0].reshape(-1, 1)

        # 2. Update Moving Obstacles
        self.step_dyn_obs()

        # 3. Compuite nominal control input, pre-defined in the robot class
        if self.state_machine == 'rotate':
            goal_angle = np.arctan2(self.goal[1] - self.robot.X[1, 0],
                                    self.goal[0] - self.robot.X[0, 0])
            if self.robot_spec['model'] in ['SingleIntegrator2D', 'DoubleIntegrator2D']:
                self.u_att = self.robot.rotate_to(goal_angle)
                u_ref = self.robot.stop()
            elif self.robot_spec['model'] in ['Unicycle2D', 'DynamicUnicycle2D', 'KinematicBicycle2D', 'KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF', 'Quad2D', 'Quad3D', 'VTOL2D']:
                u_ref = self.robot.rotate_to(goal_angle)
        elif self.goal is None:
            u_ref = self.robot.stop()
        else:
            # Normal waypoint tracking
            if self.pos_controller_type == 'optimal_decay_cbf_qp':
                u_ref = self.robot.nominal_input(self.goal, k_omega=3.0, k_a=0.5, k_v=0.5)
            else:
                u_ref = self.robot.nominal_input(self.goal)

        # 4. Update the CBF constraints & 5. Solve the control problem
        control_ref = {'state_machine': self.state_machine,
                       'u_ref': u_ref,
                       'goal': self.goal}
        
        if self.pos_controller_type in ['optimal_decay_cbf_qp', 'cbf_qp']:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_multi_obs) 
        else:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_multi_obs)
        plt.figure(self.fig.number)

        # 6. Draw collision cones/parabolas for C3BF/DPCBF
        if self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            self.robot.draw_collision_cone(self.robot.X, self.nearest_multi_obs, self.ax)
        elif self.robot_spec['model'] == 'KinematicBicycle2D_DPCBF':
            self.robot.draw_collision_parabola(self.robot.X, self.nearest_multi_obs, self.ax) 

        # 7. Update the attitude controller
        if self.state_machine == 'track' and self.att_controller is not None:
            # att_controller is only defined for integrators
            self.u_att = self.att_controller.solve_control_problem(
                    self.robot.X, self.robot.yaw, u)

        # 8. Raise an error if the QP is infeasible, or the robot collides with the obstacle
        collide = self.is_collide_unknown()
        if self.pos_controller.status != 'optimal' or collide:
            cause = "Collision" if collide else "Infeasible"
            self.draw_infeasible()
            print(f"{cause} detected !!")
            if self.raise_error:
                raise InfeasibleError(f"{cause} detected !!")
            return -2

        # 9. Step the robot
        self.robot.step(u, self.u_att)
        self.u_pos = u
    
        if self.show_animation:
            self.robot.render_plot()

        # 10. Update sensing information
        if 'sensor' in self.robot_spec and self.robot_spec['sensor'] == 'rgbd':
            self.robot.update_sensing_footprints()
            self.robot.update_safety_area()

            beyond_flag = self.robot.is_beyond_sensing_footprints()
            if beyond_flag and self.show_animation:
                pass
                # print("Visibility Violation")
        else:
            beyond_flag = 0 # not checking sensing footprint

        if self.goal is None and self.state_machine != 'stop':
            return -1  # all waypoints reached
        return beyond_flag

def single_agent_main(controller_type):
    dt = 0.05
    model = 'KinematicBicycle2D_DPCBF' # SingleIntegrator2D, DoubleIntegrator2D, DynamicUnicycle2D, KinematicBicycle2D, KinematicBicycle2D_C3BF, KinematicBicycle2D_DPCBF, Quad2D

    waypoints = [
         [1, 7.5, 0],
         [20, 7.5, 0],
    ]

    # Define dynamic obs
    known_obs = np.array([
        [8.0, 9.0, 0.5],  # obstacle 1
        [10.0, 4.0, 0.5],  # obstacle 3
        [12.0, 5.0, 0.5],  # obstacle 5
        [14.0, 9.0, 0.5],  # obstacle 7
        [16.0, 6.0, 0.5],  # obstacle 9
        [18.0, 14.0, 0.5],  # obstacle 11
        [20.0, 4.0, 0.5],  # obstacle 13
        [22.0, 12.0, 0.5],  # obstacle 15
    ])

    dynamic_obs = []  
    for i, obs_info in enumerate(known_obs):
        ox, oy, r = obs_info[:3]
        if i % 2 == 0:
            vx, vy = -0.5, 0.5
        else:
            vx, vy = -0.5, -0.5
        y_min, y_max = 0.0, 15.0
        dynamic_obs.append([ox, oy, r, vx, vy, y_min, y_max])
    known_obs = np.array(dynamic_obs)

    env_width = 22.0
    env_height = 15.0
    if model == 'SingleIntegrator2D':
        robot_spec = {
            'model': 'SingleIntegrator2D',
            'v_max': 1.0,
            'radius': 0.25
        }
    elif model == 'DoubleIntegrator2D':
        robot_spec = {
            'model': 'DoubleIntegrator2D',
            'v_max': 1.0,
            'a_max': 1.0,
            'radius': 0.25,
            'sensor': 'rgbd'
        }
    elif model == 'DynamicUnicycle2D':
        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 0.5,
            'a_max': 0.5,
            'sensor': 'rgbd',
            'radius': 0.25
        }
    elif model == 'KinematicBicycle2D':
        robot_spec = {
            'model': 'KinematicBicycle2D',
            'a_max': 0.5,
            'sensor': 'rgbd',
            'radius': 0.5
        }
    elif model == 'KinematicBicycle2D_C3BF':
        robot_spec = {
            'model': 'KinematicBicycle2D_C3BF',
            'a_max': 5.0,
            'radius': 0.3
        }
    elif model == 'KinematicBicycle2D_DPCBF':
        robot_spec = {
            'model': 'KinematicBicycle2D_DPCBF',
            'a_max': 5.0,
            # 'sensor': 'rgbd',
            'radius': 0.3
        }
    elif model == 'Quad2D':
        robot_spec = {
            'model': 'Quad2D',
            'f_min': 3.0,
            'f_max': 10.0,
            'sensor': 'rgbd',
            'radius': 0.25
        }
    elif model == 'Quad3D':
        robot_spec = {
            'model': 'Quad3D',
            'radius': 0.25
        }
        # override the waypoints with z axis
        waypoints = [
            [2, 2, 0, math.pi/2],
            [2, 12, 1, 0],
            [12, 12, -1, 0],
            [12, 2, 0, 0]
        ]

    waypoints = np.array(waypoints, dtype=np.float64)

    if model in ['SingleIntegrator2D', 'DoubleIntegrator2D', 'Quad2D', 'Quad3D']:
        x_init = waypoints[0]
    elif model == 'VTOL2D':
        v_init = robot_spec['v_max'] # m/s
        x_init = np.hstack((waypoints[0][0:2], 0.0, v_init, 0.0, 0.0))
    else:
        x_init = np.append(waypoints[0], 1.0)
    
    if known_obs.shape[1] != 7:
        known_obs = np.hstack((known_obs, np.zeros((known_obs.shape[0], 2)))) # Set static obs velocity 0.0 at (5, 5)
    
    plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
    ax, fig = plot_handler.plot_grid("") # you can set the title of the plot here
    env_handler = env.Env()

    tracking_controller = LocalTrackingControllerDyn(x_init, robot_spec,
                                                  controller_type=controller_type,
                                                  dt=dt,
                                                  show_animation=True,
                                                  save_animation=False,
                                                  show_mpc_traj=False,
                                                  ax=ax, fig=fig,
                                                  env=env_handler)

    # Set obstacles
    tracking_controller.obs = known_obs
    # tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    unexpected_beh = tracking_controller.run_all_steps(tf=100)

if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math

    single_agent_main(controller_type={'pos': 'cbf_qp'})
    # single_agent_main(controller_type={'pos': 'mpc_cbf'})
    # single_agent_main(controller_type={'pos': 'mpc_cbf', 'att': 'gatekeeper'}) # only Integrators have attitude controller, otherwise ignored
