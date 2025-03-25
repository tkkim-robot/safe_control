import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import subprocess

"""
Created on June 20th, 2024
@author: Taekyung Kim

@description: 
This code implements a local tracking controller for 2D robot navigation using Control Barrier Functions (CBF) and Quadratic Programming (QP).
It supports both kinematic (Unicycle2D) and dynamic (DynamicUnicycle2D) unicycle models, with functionality for obstacle avoidance and waypoint following.
The controller includes real-time visualization capabilities and can handle both known and unknown obstacles.
The main functions demonstrate single and multi-agent scenarios, showcasing the controller's ability to navigate complex environments.

@required-scripts: robots/robot.py
"""


class InfeasibleError(Exception):
    '''
    Exception raised for errors when QP is infeasible or 
    the robot collides with the obstacle
    '''

    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)


class LocalTrackingController:
    def __init__(self, X0, robot_spec, control_type='cbf_qp', dt=0.05,
                 show_animation=False, save_animation=False, ax=None, fig=None, env=None):

        self.robot_spec = robot_spec
        self.control_type = control_type  # 'cbf_qp' or 'mpc_cbf'
        self.dt = dt

        self.state_machine = 'idle'  # Can be 'idle', 'track', 'stop', 'rotate'
        self.rotation_threshold = 0.1  # Radians

        self.current_goal_index = 0  # Index of the current goal in the path
        self.reached_threshold = 0.07

        if self.robot_spec['model'] == 'Unicycle2D':
            if 'v_max' not in self.robot_spec:
                self.robot_spec['v_max'] = 1.0
            if 'w_max' not in self.robot_spec:
                self.robot_spec['w_max'] = 0.5
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            # v_max is set to 1.0 inside the robot class
            if 'a_max' not in self.robot_spec:
                self.robot_spec['a_max'] = 0.5
            if 'w_max' not in self.robot_spec:
                self.robot_spec['w_max'] = 0.5
            if 'v_max' not in self.robot_spec:
                self.robot_spec['v_max'] = 1.0
            if X0.shape[0] == 3:  # set initial velocity to 0.0
                X0 = np.array([X0[0], X0[1], X0[2], 0.0]).reshape(-1, 1)
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            if 'a_max' not in self.robot_spec:
                self.robot_spec['a_max'] = 1.0
            if 'v_max' not in self.robot_spec:
                self.robot_spec['v_max'] = 1.0
            if 'ax_max' not in self.robot_spec:
                self.robot_spec['ax_max'] = self.robot_spec['a_max']
            if 'ay_max' not in self.robot_spec:
                self.robot_spec['ay_max'] = self.robot_spec['a_max']
            if X0.shape[0] == 3:
                X0 = np.array([X0[0], X0[1], 0.0, 0.0, X0[2]]).reshape(-1, 1)
            elif X0.shape[0] == 2:
                X0 = np.array([X0[0], X0[1], 0.0, 0.0, 0.0]).reshape(-1, 1)
            elif X0.shape[0] != 5:
                raise ValueError(
                    "Invalid initial state dimension for DoubleIntegrator2D")
            
        self.u_att = None

        if 'fov_angle' not in self.robot_spec:
            self.robot_spec['fov_angle'] = 70.0
        if 'cam_range' not in self.robot_spec:
            self.robot_spec['cam_range'] = 3.0

        self.show_animation = show_animation
        self.save_animation = save_animation
        if self.save_animation:
            self.setup_animation_saving()

        self.ax = ax
        self.fig = fig
        self.obs = np.array(env.obs_circle)
        self.unknown_obs = None

        if show_animation:
            self.setup_animation_plot()
        else:
            self.ax = plt.axes()  # dummy placeholder

        # Setup control problem
        self.setup_robot(X0)
        self.control_type = control_type
        self.num_constraints = 5 # number of max obstacle constraints to consider in the controller
        if control_type == 'cbf_qp':
            from position_control.cbf_qp import CBFQP
            self.pos_controller = CBFQP(self.robot, self.robot_spec)
        elif control_type == 'mpc_cbf':
            from position_control.mpc_cbf import MPCCBF
            self.pos_controller = MPCCBF(self.robot, self.robot_spec)
        elif control_type == 'optimal_decay_cbf_qp':
            from position_control.optimal_decay_cbf_qp import OptimalDecayCBFQP
            self.pos_controller = OptimalDecayCBFQP(self.robot, self.robot_spec)
        elif control_type == 'optimal_decay_mpc_cbf':
            from position_control.optimal_decay_mpc_cbf import OptimalDecayMPCCBF
            self.pos_controller = OptimalDecayMPCCBF(self.robot, self.robot_spec)
        elif control_type == 'mpc_iccbf':
            from position_control.mpc_iccbf import MPC_ICCBF
            self.pos_controller = MPC_ICCBF(self.robot, self.robot_spec)
            
        if True:  # TODO: currently only have one attitude controller
            from attitude_control.simple_attitude import SimpleAttitude
            self.att_controller = SimpleAttitude(self.robot, self.robot_spec)
        self.goal = None

    def set_robot_state(self, pose, orientation, velocity):
        self.robot.set_robot_state(pose, orientation, velocity)

    def setup_animation_saving(self):
        self.current_directory_path = os.getcwd()
        if not os.path.exists(self.current_directory_path + "/output/animations"):
            os.makedirs(self.current_directory_path + "/output/animations")
        self.save_per_frame = 2
        self.ani_idx = 0

    def setup_animation_plot(self):
        # Initialize plotting
        if self.ax is None:
            self.ax = plt.axes()
        if self.fig is None:
            self.fig = plt.figure()
        plt.ion()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_aspect(1)
        self.waypoints_scatter = self.ax.scatter(
            [], [], s=10, facecolors='g', edgecolors='g', alpha=0.5)

    def setup_robot(self, X0):
        from robots.robot import BaseRobot
        self.robot = BaseRobot(
            X0.reshape(-1, 1), self.robot_spec, self.dt, self.ax)

    def set_waypoints(self, waypoints):
        if type(waypoints) == list:
            waypoints = np.array(waypoints, dtype=float)
        self.waypoints = self.filter_waypoints(waypoints)
        self.current_goal_index = 0

        self.goal = self.update_goal()
        if self.goal is not None:
            if not self.robot.is_in_fov(self.goal):
                self.state_machine = 'stop'
                self.goal = None  # let the robot stop then rotate
            else:
                self.state_machine = 'track'

        if self.show_animation:
            self.waypoints_scatter.set_offsets(self.waypoints[:, :2])

    def filter_waypoints(self, waypoints):
        '''
        Initially filter out waypoints that are too close to the robot
        '''
        if len(waypoints) < 2:
            return waypoints

        robot_pos = self.robot.get_position()
        aug_waypoints = np.vstack((robot_pos, waypoints[:, :2]))

        distances = np.linalg.norm(np.diff(aug_waypoints, axis=0), axis=1)
        mask = np.concatenate(([False], distances >= self.reached_threshold))
        return aug_waypoints[mask]

    def goal_reached(self, current_position, goal_position):
        return np.linalg.norm(current_position[:2] - goal_position[:2]) < self.reached_threshold

    def has_reached_goal(self):
        # return whethere the self.goal is None or not
        if self.state_machine in ['stop']:
            return False
        return self.goal is None

    def set_unknown_obs(self, unknown_obs):
        # set initially
        self.unknown_obs = unknown_obs
        for (ox, oy, r) in self.unknown_obs:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='orange',
                    fill=True,
                    alpha=0.4
                )
            )

    def get_nearest_unpassed_obs(self, detected_obs, angle_unpassed=np.pi, obs_num=5):
        def angle_normalize(x):
            return (((x + np.pi) % (2 * np.pi)) - np.pi)
        '''
        Get the nearest 5 obstacles that haven't been passed by (i.e., they're still in front of the robot).
        '''
        if len(detected_obs) == 0:
            return None
        
        unpassed_obs = []
        robot_pos = self.robot.get_position()
        robot_yaw = self.robot.get_orientation()
        
        # Iterate through each detected obstacle
        for obs in detected_obs:
            obs_pos = np.array([obs[0], obs[1]])
            to_obs_vector = obs_pos - robot_pos
            
            # Calculate the angle between the robot's heading and the vector to the obstacle
            robot_heading_vector = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
            angle_to_obs = np.arctan2(to_obs_vector[1], to_obs_vector[0])
            angle_diff = abs(angle_normalize(angle_to_obs - robot_yaw))
            
            # If the obstacle is within the forward-facing 180 degrees, consider it
            if angle_diff <= angle_unpassed/2:
                unpassed_obs.append(obs)
        
        # If no unpassed obstacles are found, return the nearest obstacles from the full detected_obs list
        if len(unpassed_obs) == 0:
            detected_obs = np.array(detected_obs)
            distances = np.linalg.norm(detected_obs[:, :2] - robot_pos, axis=1)
            nearest_indices = np.argsort(distances)[:5]  # Get indices of the nearest 5 obstacles
            return detected_obs[nearest_indices]
        
        # Now, find the nearest unpassed obstacles
        unpassed_obs = np.array(unpassed_obs)
        distances = np.linalg.norm(unpassed_obs[:, :2] - robot_pos, axis=1)
        nearest_indices = np.argsort(distances)[:obs_num]  # Get indices of the nearest 'obs_num' (max 5) unpassed obstacles
        
        return unpassed_obs[nearest_indices]

    def get_nearest_obs(self, detected_obs):
        # if there was new obstacle detected, update the obs
        if len(detected_obs) != 0:
            if len(self.obs) == 0:
                all_obs = np.array(detected_obs)
            else:
                all_obs = np.vstack((self.obs, detected_obs))
            # return np.array(detected_obs).reshape(-1, 1) just returning the detected obs
        else:
            all_obs = self.obs

        if len(all_obs) == 0:
            return None

        if all_obs.ndim == 1:
            all_obs = all_obs.reshape(1, -1)

        radius = all_obs[:, 2]
        distances = np.linalg.norm(all_obs[:, :2] - self.robot.X[:2].T, axis=1)
        min_distance_index = np.argmin(distances-radius)
        nearest_obstacle = all_obs[min_distance_index]
        return nearest_obstacle.reshape(-1, 1)

    def is_collide_unknown(self):
        # if self.unknown_obs is None:
        #     return False
        robot_radius = self.robot.robot_radius
        
        if self.unknown_obs is not None:
            for obs in self.unknown_obs:
                # check if the robot collides with the obstacle
                distance = np.linalg.norm(self.robot.X[:2, 0] - obs[:2])
                if distance < (obs[2] + robot_radius):
                    return True
                
        if self.obs is not None:
            for obs in self.obs:
                # check if the robot collides with the obstacle
                distance = np.linalg.norm(self.robot.X[:2, 0] - obs[:2])
                if distance < (obs[2] + robot_radius):
                    return True
        return False

    def update_goal(self):
        '''
        Update the goal from waypoints
        '''
        if self.state_machine == 'rotate':
            # in-place rotation
            self.state_machine = 'track'
            self.u_att = None
            # current_angle = self.robot.get_orientation()
            # goal_angle = np.arctan2(self.waypoints[0][1] - self.robot.X[1, 0],
            #                         self.waypoints[0][0] - self.robot.X[0, 0])
            # if abs(current_angle - goal_angle) > self.rotation_threshold:
            #     return self.waypoints[0][:2]
            # else:
            #     self.state_machine = 'track'
            #     self.u_att = None
            #     print("set u_att to none")

        # Check if all waypoints are reached;
        if self.current_goal_index >= len(self.waypoints):
            return None

        if self.goal_reached(self.robot.X, np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)):
            self.current_goal_index += 1

            if self.current_goal_index >= len(self.waypoints):
                self.state_machine = 'idle'
                return None

        # set goal to next waypoint's (x,y)
        goal = np.array(self.waypoints[self.current_goal_index][0:2])
        return goal

    def draw_plot(self, pause=0.01, force_save=False):
        if self.show_animation:
            self.fig.canvas.draw()
            plt.pause(pause)
            if self.save_animation:
                self.ani_idx += 1
                if force_save or self.ani_idx % self.save_per_frame == 0:
                    plt.savefig(self.current_directory_path +
                                "/output/animations/" + "t_step_" + str(self.ani_idx//self.save_per_frame).zfill(4) + ".png")

    def control_step(self):
        '''
        Simulate one step of tracking control with CBF-QP with the given waypoints.
        Output: 
            - -1: all waypoints reached
            - 0: normal
            - 1: visibility violation
            - raise QPError: if the QP is infeasible or the robot collides with the obstacle
        '''
        # update state machine
        if self.state_machine == 'stop':
            if self.robot.has_stopped():
                self.state_machine = 'rotate'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        # 1. Update the detected obstacles
        # detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)
        # self.nearest_obs = self.get_nearest_obs(detected_obs)
        self.nearest_multi_obs = self.get_nearest_unpassed_obs(self.obs, obs_num=self.num_constraints)
        self.nearest_obs = self.nearest_multi_obs[0].reshape(3,1)

        # 2. Compuite nominal control input, pre-defined in the robot class
        if self.state_machine == 'rotate':
            goal_angle = np.arctan2(self.goal[1] - self.robot.X[1, 0],
                                    self.goal[0] - self.robot.X[0, 0])
            if self.robot_spec['model'] == 'DoubleIntegrator2D':
                self.u_att = self.robot.rotate_to(goal_angle)
                u_ref = self.robot.stop()
            elif self.robot_spec['model'] in ['Unicycle2D', 'DynamicUnicycle2D']:
                u_ref = self.robot.rotate_to(goal_angle)
        elif self.goal is None:
            u_ref = self.robot.stop()
        else:
            if self.control_type == 'optimal_decay_cbf_qp':
                u_ref = self.robot.nominal_input(self.goal, k_omega=3.0, k_a=0.5, k_v=0.5)
            else:
                u_ref = self.robot.nominal_input(self.goal)
            self.u_att = self.att_controller.solve_control_problem(
                    self.robot.X)

        # 3. Update the CBF constraints & # 4. Solve the control problem
        control_ref = {'state_machine': self.state_machine,
                       'u_ref': u_ref,
                       'goal': self.goal}
        print(self.state_machine)
        if self.control_type == 'optimal_decay_cbf_qp' or self.control_type == 'cbf_qp':
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_obs)
        else:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_multi_obs)

        # 5. Raise an error if the QP is infeasible, or the robot collides with the obstacle
        collide = self.is_collide_unknown()
        if self.pos_controller.status != 'optimal' or collide:
            self.draw_infeasible()
            print("Infeasible or Collision")
            return -2
            # raise InfeasibleError("Infeasible or Collision")

        # 6. Step the robot
        self.robot.step(u, self.u_att)
        # Generate the pseudo state for the next step for hardware implementation
        self.next_x_according_to_dynamics = self.robot.pseudo_step(u, self.u_att)
        # print("step", x)
        self.control_input = u
        if self.show_animation:
            self.robot.render_plot()

        # 7. Update sensing information
        self.robot.update_sensing_footprints()
        self.robot.update_safety_area()

        beyond_flag = self.robot.is_beyond_sensing_footprints()
        if beyond_flag and self.show_animation:
            pass
            # print("Visibility Violation")

        if self.goal is None and self.state_machine != 'stop':
            return -1  # all waypoints reached
        return beyond_flag

    def get_control_input(self):
        return self.control_input

    def get_full_state(self):
        return self.next_x_according_to_dynamics

    def draw_infeasible(self):
        if self.show_animation:
            self.robot.render_plot()
            current_position = self.robot.get_position()
            self.ax.text(current_position[0]+0.5, current_position[1] +
                         0.5, '!', color='red', weight='bold', fontsize=22)
            self.draw_plot(pause=5, force_save=True)

    def export_video(self):
        # convert the image sequence to a video
        if self.show_animation and self.save_animation:
            subprocess.call(['ffmpeg',
                             '-framerate', '30',  # Input framerate (adjust if needed)
                             '-i', self.current_directory_path + "/output/animations/t_step_%04d.png",
                             '-vf', 'scale=1920:982,fps=60',  # Ensure height is divisible by 2 and set output framerate
                             '-pix_fmt', 'yuv420p',
                             self.current_directory_path + "/output/animations/tracking.mp4"])

            for file_name in glob.glob(self.current_directory_path +
                                       "/output/animations/*.png"):
                os.remove(file_name)

    # # If the 'upper' function is not compatible with your device, please use the function provided below
    # def export_video(self):
    #     # convert the image sequence to a video
    #     if self.show_animation and self.save_animation:
    #         subprocess.call(['ffmpeg',
    #                          # Input framerate (adjust if needed)
    #                          '-framerate', '30',
    #                          '-i', self.current_directory_path+"/output/animations/t_step_%04d.png",
    #                          '-filter:v', 'fps=60',  # Output framerate
    #                          '-pix_fmt', 'yuv420p',
    #                          self.current_directory_path+"/output/animations/tracking.mp4"])

    #         for file_name in glob.glob(self.current_directory_path +
    #                                    "/output/animations/*.png"):
    #             os.remove(file_name)
    
    def run_all_steps(self, tf=30):
        print("===================================")
        print("============ Tracking =============")
        print("Start following the generated path.")
        unexpected_beh = 0

        for _ in range(int(tf / self.dt)):
            ret = self.control_step()
            self.draw_plot()
            unexpected_beh += ret
            if ret == -1:  # all waypoints reached
                break

        self.export_video()

        print("=====   Tracking finished    =====")
        print("===================================\n")
        if self.show_animation:
            plt.ioff()
            plt.close()

        return unexpected_beh


def single_agent_main(control_type):
    dt = 0.05

    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [10, 12, 0],
        [10, 2, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)
    x_init = np.append(waypoints[0], 1.0)
    
    known_obs = np.array([[1.5, 3.0, 0.2],[2.6, 3.0, 0.2],[2.2, 5.0, 0.2], [3.0, 5.0, 0.2],
                            [10.0, 7.3, 0.4],])
    
    plot_handler = plotting.Plotting(known_obs=known_obs)
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")
    env_handler = env.Env()

    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'fov_angle': 70.0,
        'cam_range': 3.0
    }
    # robot_spec = {
    #     'model': 'DoubleIntegrator2D',
    #     'w_max': 0.5,
    #     'a_max': 0.5,
    #     'fov_angle': 70.0,
    #     'cam_range': 5.0
    # }
    tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                  control_type=control_type,
                                                  dt=dt,
                                                  show_animation=True,
                                                  save_animation=False,
                                                  ax=ax, fig=fig,
                                                  env=env_handler)

    # Set gamma values
    tracking_controller.pos_controller.cbf_param['alpha1'] = 0.15
    tracking_controller.pos_controller.cbf_param['alpha2'] = 0.15
    
    tracking_controller.obs = known_obs
    # tracking_controller.set_unknown_obs(known_obs)
    tracking_controller.set_waypoints(waypoints)
    unexpected_beh = tracking_controller.run_all_steps(tf=100)

def multi_agent_main(control_type):
    dt = 0.05

    # temporal
    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [10, 12, 0],
        [10, 2, math.pi/2]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)

    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting()
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")
    env_handler = env.Env()

    robot_spec = {
        'model': 'DoubleIntegrator2D',  # 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'fov_angle': 70.0,
        'cam_range': 3.0
    }

    robot_spec['robot_id'] = 0
    controller_0 = LocalTrackingController(x_init, robot_spec,
                                           control_type=control_type,
                                           dt=dt,
                                           show_animation=True,
                                           save_animation=False,
                                           ax=ax, fig=fig,
                                           env=env_handler)

    robot_spec['robot_id'] = 1
    controller_1 = LocalTrackingController(x_goal, robot_spec,
                                           control_type=control_type,
                                           dt=dt,
                                           show_animation=True,
                                           save_animation=False,
                                           ax=ax, fig=fig,
                                           env=env_handler)

    # unknown_obs = np.array([[9.0, 8.8, 0.3]])
    # tracking_controller.set_unknown_obs(unknown_obs)
    controller_0.set_waypoints(waypoints)
    controller_1.set_waypoints(waypoints[::-1])
    tf = 50
    for _ in range(int(tf / dt)):
        ret_list = []
        ret_list.append(controller_0.control_step())
        ret_list.append(controller_1.control_step())
        controller_0.draw_plot()
        # if all elements of ret_list are -1, break
        if all([ret == -1 for ret in ret_list]):
            break


if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math

    # single_agent_main('mpc_cbf')
    # single_agent_main('cbf_qp')
    # single_agent_main('optimal_decay_cbf_qp')
    # single_agent_main('optimal_decay_mpc_cbf')
    single_agent_main('mpc_iccbf')
