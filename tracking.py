import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import subprocess
import csv

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
    def __init__(self, X0, robot_spec,
                 controller_type=None,
                 dt=0.05,
                 show_animation=False, save_animation=False, show_mpc_traj=False,
                 enable_rotation=True, raise_error=False,
                 ax=None, fig=None, env=None):

        self.robot_spec = robot_spec
        self.pos_controller_type = controller_type.get('pos', 'cbf_qp')  # 'cbf_qp' or 'mpc_cbf'
        self.att_controller_type = controller_type.get('att', 'velocity_tracking_yaw')  # 'simple' or 'velocity_tracking_yaw'
        self.dt = dt

        self.state_machine = 'idle'  # Can be 'idle', 'track', 'stop', 'rotate'
        self.rotation_threshold = 0.1  # Radians

        self.current_goal_index = 0  # Index of the current goal in the path
        self.reached_threshold = 0.2
        # if robot_spec specifies a different reached_threshold, use that (ex. VTOL)
        if 'reached_threshold' in robot_spec:
            self.reached_threshold = robot_spec['reached_threshold']
            print("Using custom reached_threshold: ", self.reached_threshold)

        if self.robot_spec['model'] == 'SingleIntegrator2D':
            if X0.shape[0] == 2:
                X0 = np.array([X0[0], X0[1], 0.0]).reshape(-1, 1)
            elif X0.shape[0] != 3:
                raise ValueError(
                    "Invalid initial state dimension for SingleIntegrator2D")
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            if X0.shape[0] == 3:  # set initial velocity to 0.0
                X0 = np.array([X0[0], X0[1], X0[2], 0.0]).reshape(-1, 1)
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            if X0.shape[0] == 3:
                X0 = np.array([X0[0], X0[1], 0.0, 0.0, X0[2]]).reshape(-1, 1)
            elif X0.shape[0] == 2:
                X0 = np.array([X0[0], X0[1], 0.0, 0.0, 0.0]).reshape(-1, 1)
            elif X0.shape[0] != 5:
                raise ValueError(
                    "Invalid initial state dimension for DoubleIntegrator2D")
        elif self.robot_spec['model'] in ['KinematicBicycle2D', 'KinematicBicycle2D_C3BF']:
            if X0.shape[0] == 3:  # set initial velocity to 0.0
                X0 = np.array([X0[0], X0[1], X0[2], 0.0]).reshape(-1, 1)
        elif self.robot_spec['model'] in ['Quad2D']:
            if X0.shape[0] in [2, 3]: # only initialize the x,z position if don't provide the full state
                X0 = np.array([X0[0], X0[1], 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
            elif X0.shape[0] != 6:
                raise ValueError("Invalid initial state dimension for Quad2D")
        elif self.robot_spec['model'] == 'Quad3D':
            if X0.shape[0] == 2:
                X0 = np.array([X0[0], X0[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
            elif X0.shape[0] == 3:
                X0 = np.array([X0[0], X0[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, X0[2]]).reshape(-1, 1)
            elif X0.shape[0] == 4:
                X0 = np.array([X0[0], X0[1], X0[2], 0.0, 0.0, 0.0, 0.0, 0.0, X0[3]]).reshape(-1, 1)
            elif X0.shape[0] != 9:
                raise ValueError("Invalid initial state dimension for Quad3D")
        elif self.robot_spec['model'] in ['VTOL2D']:
            if X0.shape[0] in [2, 3]: 
                # set initial velocity to 5.0
                X0 = np.array([X0[0], X0[1], 0.0, 5.0, 0.0, 0.0]).reshape(-1, 1)
            elif X0.shape[0] != 6:
                raise ValueError("Invalid initial state dimension for VTOL2D")
    
            
        self.u_att = None

        self.show_animation = show_animation
        self.save_animation = save_animation
        self.show_mpc_traj = show_mpc_traj
        self.enable_rotation = enable_rotation
        self.raise_error = raise_error
        if self.save_animation:
            self.setup_animation_saving()

        self.ax = ax
        self.fig = fig
        self.obs = np.array(env.obs_circle)

        # Initialize moving obstacles
        self.dyn_obs_patch = None # will be initialized after the first step
        self.init_obs_info = None
        self.init_obs_circle = None
        
        self.known_obs = np.array([])
        self.unknown_obs = np.array([])

        if show_animation:
            self.setup_animation_plot()
        else:
            self.ax = plt.axes()  # dummy placeholder

        # Setup control problem
        self.setup_robot(X0)
        self.num_constraints = 5 # number of max obstacle constraints to consider in the controller
        if self.pos_controller_type == 'cbf_qp':
            from position_control.cbf_qp import CBFQP
            self.pos_controller = CBFQP(self.robot, self.robot_spec)
        elif self.pos_controller_type == 'mpc_cbf':
            from position_control.mpc_cbf import MPCCBF
            self.pos_controller = MPCCBF(self.robot, self.robot_spec, show_mpc_traj=self.show_mpc_traj)
        elif self.pos_controller_type == 'optimal_decay_cbf_qp':
            from position_control.optimal_decay_cbf_qp import OptimalDecayCBFQP
            self.pos_controller = OptimalDecayCBFQP(self.robot, self.robot_spec)
        elif self.pos_controller_type == 'optimal_decay_mpc_cbf':
            from position_control.optimal_decay_mpc_cbf import OptimalDecayMPCCBF
            self.pos_controller = OptimalDecayMPCCBF(self.robot, self.robot_spec)
        else:
            raise ValueError(
                f"Unknown controller type: {self.pos_controller_type}")
            
        if self.enable_rotation and self.robot_spec['model'] in ['SingleIntegrator2D', 'DoubleIntegrator2D']:
            if self.att_controller_type == 'simple':
                from attitude_control.simple_attitude import SimpleAtt
                self.att_controller = SimpleAtt(self.robot, self.robot_spec)
            elif self.att_controller_type == 'velocity_tracking_yaw':
                from attitude_control.velocity_tracking_yaw import VelocityTrackingYaw
                self.att_controller = VelocityTrackingYaw(self.robot, self.robot_spec)
            elif self.att_controller_type == 'visibility_raycast':
                from safe_control.attitude_control.visibility_raycast import VisibilityRayCastAtt
                self.att_controller = VisibilityRayCastAtt(self.robot, self.robot_spec)
            elif self.att_controller_type == 'visibility_area':
                from attitude_control.visibility_area import VisibilityAreaAtt
                self.att_controller = VisibilityAreaAtt(self.robot, self.robot_spec)
            elif self.att_controller_type == 'gatekeeper':
                from attitude_control.gatekeeper_attitude import GatekeeperAtt
                self.att_controller = GatekeeperAtt(self.robot, self.robot_spec)
                self.att_controller.setup_pos_controller(self.pos_controller)
            else:
                raise ValueError(
                    f"Unknown attitude controller type: {self.att_controller_type}")
        else:
            self.att_controller = None
        self.goal = None

    def setup_animation_saving(self):
        self.current_directory_path = os.getcwd()
        if not os.path.exists(self.current_directory_path + "/output/animations"):
            os.makedirs(self.current_directory_path + "/output/animations")
        self.save_per_frame = 1
        self.ani_idx = 0

    def setup_animation_plot(self):
        # Initialize plotting
        if self.ax is None:
            self.ax = plt.axes()
        if self.fig is None:
            self.fig = plt.figure()
        plt.ion()
        self.ax.set_xlabel("X [m]")
        if self.robot_spec['model'] in ['Quad2D', 'VTOL2D']:
            self.ax.set_ylabel("Z [m]")
        else:
            self.ax.set_ylabel("Y [m]")
        self.ax.set_aspect(1)
        self.fig.tight_layout()
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
                if self.robot_spec['exploration']:
                    # when tracking class used in exploration scenario,
                    # the goal is updated usually when the robot is far away from the unsafe area (considering sensing range)
                    self.state_machine = 'rotate'
                else:
                    # normal tracking mode
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
        if self.robot_spec['model'] in ['Quad3D']:
            n_pos = 3
            robot_pos = np.hstack([robot_pos, self.robot.get_z()])
            aug_waypoints = np.vstack((robot_pos, waypoints[:, :n_pos]))
        else:
            n_pos = 2
            aug_waypoints = np.vstack((robot_pos, waypoints[:, :n_pos]))

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
        unknown_obs = np.array(unknown_obs)
        if unknown_obs.shape[1] == 3:
            zeros = np.zeors((unknown_obs.shape[0], 2))
            unknown_obs = np.hstack((unknown_obs, zeros))
        self.unknown_obs = unknown_obs
        for obs_info in self.unknown_obs:
            ox, oy, r = obs_info[:3]
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='orange',
                    fill=True,
                    alpha=0.4
                )
            )

    def get_nearest_unpassed_obs(self, detected_obs, angle_unpassed=np.pi*2, obs_num=5):
        def angle_normalize(x):
            return (((x + np.pi) % (2 * np.pi)) - np.pi)
        '''
        Get the nearest 5 obstacles that haven't been passed by (i.e., they're still in front of the robot or the robot should still consider the obstacle).
        '''
        
        if self.robot_spec['model'] in ['SingleIntegrator2D', 'DoubleIntegrator2D', 'Quad2D']:
            angle_unpassed=np.pi*2
        elif self.robot_spec['model'] in ['Unicycle2D', 'DynamicUnicycle2D', 'KinematicBicycle2D', 'KinematicBicycle2D_C3BF', 'Quad3D', 'VTOL2D']:
            angle_unpassed=np.pi*1.2
        
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
        
        unpassed_obs = []
        robot_pos = self.robot.get_position()
        robot_yaw = self.robot.get_orientation()

        # Iterate through each detected obstacle
        for obs in all_obs:
            obs_pos = np.array([obs[0], obs[1]])
            to_obs_vector = obs_pos - robot_pos
            
            # Calculate the angle between the robot's heading and the vector to the obstacle
            robot_heading_vector = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
            angle_to_obs = np.arctan2(to_obs_vector[1], to_obs_vector[0])
            angle_diff = abs(angle_normalize(angle_to_obs - robot_yaw))
            
            # If the obstacle is within the forward-facing 180 degrees, consider it
            if angle_diff <= angle_unpassed/2:
                unpassed_obs.append(obs)
        
        # If no unpassed obstacles are found, return the nearest obstacles from the full all_obs list
        if len(unpassed_obs) == 0:
            all_obs = np.array(all_obs)
            distances = np.linalg.norm(all_obs[:, :2] - robot_pos, axis=1)
            nearest_indices = np.argsort(distances)[:5]  # Get indices of the nearest 5 obstacles
            return all_obs[nearest_indices]
        
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

    # Update dynamic obs position
    def step_dyn_obs(self):
        """if self.obs (n,5) array (ex) [x, y, r, vx, vy], update obs position per time step"""
        if len(self.obs) != 0 and self.obs.shape[1] >= 5:
            self.obs[:, 0] += self.obs[:, 3] * self.dt
            self.obs[:, 1] += self.obs[:, 4] * self.dt
    
    def render_dyn_obs(self):
        for i, obs_info in enumerate(self.obs):
            # obs: [x, y, r, vx, vy]
            ox, oy, r = obs_info[:3]
            self.dyn_obs_patch[i].center = ox, oy
            self.dyn_obs_patch[i].set_radius(r)

        for i, obs_info in enumerate(self.init_obs_info):
            ox, oy, r = obs_info[:3]
            self.init_obs_circle[i].center = ox, oy
            self.init_obs_circle[i].set_radius(r)

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

        # Collision with the ground
        if self.robot_spec['model'] in ['VTOL2D']:
            if self.robot.X[1, 0] < 0:
                return True
            if np.abs(self.robot.X[2, 0]) > self.robot_spec['pitch_max']:
                return True
        return False

    def update_goal(self):
        '''
        Update the goal from waypoints
        '''
        if self.robot_spec['model'] in ['Quad3D']:
            n_pos = 3
        else:
            n_pos = 2

        if self.state_machine == 'rotate':
            # in-place rotation
            current_angle = self.robot.get_orientation()
            goal_angle = np.arctan2(self.waypoints[0][1] - self.robot.X[1, 0],
                                    self.waypoints[0][0] - self.robot.X[0, 0])
            if self.robot_spec['model'] in ['Quad2D', 'VTOL2D']: # These skip 'rotate' state since there is no yaw angle
                self.state_machine = 'track'
            if not self.enable_rotation:
                self.state_machine = 'track'
            if abs(current_angle - goal_angle) > self.rotation_threshold:
                return self.waypoints[0][:n_pos]
            else:
                self.state_machine = 'track'
                self.u_att = None
                print("set u_att to none")

        # Check if all waypoints are reached;
        if self.current_goal_index >= len(self.waypoints):
            return None

        if self.goal_reached(self.robot.X, np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)):
            self.current_goal_index += 1

            if self.current_goal_index >= len(self.waypoints):
                self.state_machine = 'idle'
                return None

        goal = np.array(self.waypoints[self.current_goal_index][0:n_pos])
        return goal

    def draw_plot(self, pause=0.01, force_save=False):
        if self.show_animation:
            if self.dyn_obs_patch is None:
                # Initialize moving obstacles
                self.dyn_obs_patch = [self.ax.add_patch(plt.Circle(
                    (0, 0), 0, edgecolor='black', facecolor='gray', fill=True)) for _ in range(len(self.obs))]
                self.init_obs_info = self.obs.copy()
                self.init_obs_circle = [self.ax.add_patch(plt.Circle((0, 0), 0, edgecolor='black', facecolor='none', linestyle='--')) for _ in self.init_obs_info]
            
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
            elif self.robot_spec['model'] in ['Unicycle2D', 'DynamicUnicycle2D', 'KinematicBicycle2D', 'KinematicBicycle2D_C3BF', 'Quad2D', 'Quad3D', 'VTOL2D']:
                u_ref = self.robot.rotate_to(goal_angle)
        elif self.goal is None:
            u_ref = self.robot.stop()
        else:
            # Normal waypoint tracking
            if self.pos_controller_type == 'optimal_decay_cbf_qp':
                u_ref = self.robot.nominal_input(self.goal, k_omega=3.0, k_a=0.5, k_v=0.5)
            else:
                u_ref = self.robot.nominal_input(self.goal)

        # 4. Update the CBF constraints & 5. Solve the control problem & 6. Draw Collision Cones for C3BF
        control_ref = {'state_machine': self.state_machine,
                       'u_ref': u_ref,
                       'goal': self.goal}
        if self.pos_controller_type in ['optimal_decay_cbf_qp', 'cbf_qp']:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_obs)
            self.robot.draw_collision_cone(self.robot.X, [self.nearest_obs], self.ax)
        else:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_multi_obs)
            self.robot.draw_collision_cone(self.robot.X, self.nearest_multi_obs, self.ax)
        plt.figure(self.fig.number)

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

    def get_control_input(self):
        return self.u_pos
    
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
    
    def run_all_steps(self, tf=30, write_csv=False):
        print("===================================")
        print("============ Tracking =============")
        print("Start following the generated path.")
        unexpected_beh = 0

        if write_csv:
            # create a csv file to record the states, control inputs, and CBF parameters
            with open('output.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['states', 'control_inputs', 'alpha1', 'alpha2'])

        for _ in range(int(tf / self.dt)):
            ret = self.control_step()
            self.draw_plot()
            unexpected_beh += ret

            # get states of the robot
            robot_state = self.robot.X[:,0].flatten()
            control_input = self.get_control_input().flatten()
            # print(f"Robot state: {robot_state}")
            # print(f"Control input: {control_input}")

            if write_csv:
                # append the states, control inputs, and CBF parameters by appending to csv
                with open('output.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(np.append(robot_state, np.append(control_input, [self.pos_controller.cbf_param['alpha1'], self.pos_controller.cbf_param['alpha2']])))


            if ret == -1 or ret == -2:  # all waypoints reached
                break

        self.export_video()

        print("=====   Tracking finished    =====")
        print("===================================\n")
        if self.show_animation:
            plt.ioff()
            plt.close()

        return unexpected_beh


def single_agent_main(controller_type):
    dt = 0.05
    model = 'DynamicUnicycle2D' # SingleIntegrator2D, DynamicUnicycle2D, KinematicBicycle2D, KinematicBicycle2D_C3BF, DoubleIntegrator2D, Quad2D, Quad3D, VTOL2D

    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [12, 12, 0],
        [12, 2, 0]
    ]

    # Define static obs
    known_obs = np.array([[2.2, 5.0, 0.2], [3.0, 5.0, 0.2], [4.0, 9.0, 0.3], [1.5, 10.0, 0.5], [9.0, 11.0, 1.0], [7.0, 7.0, 3.0], [4.0, 3.5, 1.5],
                        [10.0, 7.3, 0.4],
                        [6.0, 13.0, 0.7], [5.0, 10.0, 0.6], [11.0, 5.0, 0.8], [13.5, 11.0, 0.6]])

    env_width = 14.0
    env_height = 14.0
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
            'a_max': 0.5,
            'radius': 0.5
        }
        dynamic_obs = []  
        for i, obs_info in enumerate(known_obs):
            ox, oy, r = obs_info[:3]
            if i % 2 == 0:
                vx, vy = 0.1, 0.05
            else:
                vx, vy = -0.1, 0.05
            dynamic_obs.append([ox, oy, r, vx, vy])
        known_obs = np.array(dynamic_obs)

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
            'f_max': 100.0,
            'radius': 0.25
        }
        # override the waypoints with z axis
        waypoints = [
            [2, 2, 0, math.pi/2],
            [2, 12, 1, 0],
            [12, 12, -1, 0],
            [12, 2, 0, 0]
        ]
    elif model == 'VTOL2D':
        # VTOL has pretty different dynacmis, so create a special test case
        robot_spec = {
            'model': 'VTOL2D',
            'radius': 0.6,
            'v_max': 20.0,
            'reached_threshold': 1.0 # meter
        }
        # override the waypoints and known_obs
        waypoints = [
            [2, 10],
            [70, 10],
            [70, 0.5]
        ]
        pillar_1_x = 67.0
        pillar_2_x = 73.0
        known_obs = np.array([
            # [pillar_1_x, 1.0, 0.5],
            # [pillar_1_x, 2.0, 0.5],
            # [pillar_1_x, 3.0, 0.5],
            # [pillar_1_x, 4.0, 0.5],
            # [pillar_1_x, 5.0, 0.5],
            [pillar_1_x, 6.0, 0.5],
            [pillar_1_x, 7.0, 0.5],
            [pillar_1_x, 8.0, 0.5],
            [pillar_1_x, 9.0, 0.5],
            [pillar_2_x, 1.0, 0.5],
            [pillar_2_x, 2.0, 0.5],
            [pillar_2_x, 3.0, 0.5],
            [pillar_2_x, 4.0, 0.5],
            [pillar_2_x, 5.0, 0.5],
            [pillar_2_x, 6.0, 0.5],
            [pillar_2_x, 7.0, 0.5],
            [pillar_2_x, 8.0, 0.5],
            [pillar_2_x, 9.0, 0.5],
            [pillar_2_x, 10.0, 0.5],
            [pillar_2_x, 11.0, 0.5],
            [pillar_2_x, 12.0, 0.5],
            [pillar_2_x, 13.0, 0.5],
            [pillar_2_x, 14.0, 0.5],
            [pillar_2_x, 15.0, 0.5],
            [60.0, 12.0, 1.5]
        ])

        env_width = 75.0
        env_height = 20.0
        plt.rcParams['figure.figsize'] = [12, 5]

    waypoints = np.array(waypoints, dtype=np.float64)

    if model in ['SingleIntegrator2D', 'DoubleIntegrator2D', 'Quad2D', 'Quad3D']:
        x_init = waypoints[0]
    elif model == 'VTOL2D':
        v_init = robot_spec['v_max'] # m/s
        x_init = np.hstack((waypoints[0][0:2], 0.0, v_init, 0.0, 0.0))
    else:
        x_init = np.append(waypoints[0], 1.0)
    
    if known_obs.shape[1] != 5:
        known_obs = np.hstack((known_obs, np.zeros((known_obs.shape[0], 2)))) # Set static obs velocity 0.0 at (5, 5)

    plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
    ax, fig = plot_handler.plot_grid("") # you can set the title of the plot here
    env_handler = env.Env()

    tracking_controller = LocalTrackingController(x_init, robot_spec,
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

def multi_agent_main(controller_type, save_animation=False):
    dt = 0.05

    # temporal
    waypoints = [
        [2, 2, 0],
        [2, 12, 0],
        [12, 12, 0],
        [12, 2, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)

    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting()
    ax, fig = plot_handler.plot_grid("")
    env_handler = env.Env()

    robot_spec = {
        'model': 'DynamicUnicycle2D', #'DoubleIntegrator2D'
        'w_max': 0.5,
        'a_max': 0.5,
        'sensor': 'rgbd',
        'fov_angle': 45.0,
        'cam_range': 3.0,
        'radius': 0.25
    }

    robot_spec['robot_id'] = 0
    controller_0 = LocalTrackingController(x_init, robot_spec,
                                           controller_type=controller_type,
                                           dt=dt,
                                           show_animation=True,
                                           save_animation=save_animation,
                                           ax=ax, fig=fig,
                                           env=env_handler)

    robot_spec = {
        'model': 'DynamicUnicycle2D', #'DoubleIntegrator2D'
        'w_max': 1.0,
        'a_max': 1.5,
        'v_max': 2.0,
        'sensor': 'rgbd',
        'fov_angle': 90.0,
        'cam_range': 5.0,
        'radius': 0.25
    }

    robot_spec['robot_id'] = 1
    controller_1 = LocalTrackingController(x_goal, robot_spec,
                                           controller_type=controller_type,
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

    if save_animation:
        controller_0.export_video()


if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math

    #single_agent_main(controller_type={'pos': 'mpc_cbf'})
    #single_agent_main(controller_type={'pos': 'mpc_cbf', 'att': 'simple'})
    single_agent_main(controller_type={'pos': 'mpc_cbf', 'att': 'visibility_raycast'})
    # multi_agent_main('mpc_cbf', save_animation=True)
    # single_agent_main('cbf_qp')
    # single_agent_main('optimal_decay_cbf_qp')
    # single_agent_main('optimal_decay_mpc_cbf')