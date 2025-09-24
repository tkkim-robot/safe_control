from robots.robot import BaseRobot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely.geometry import Polygon, Point, LineString
from shapely import is_valid_reason
from utils.geometry import custom_merge

class BaseRobotDyn(BaseRobot):

    def __init__(self, X0, robot_spec, dt, ax):
        super().__init__(X0, robot_spec, dt, ax)

        self.collision_parabola_patches = []
        self.collision_parabola_patch = None
        
        self.collision_cone_patches = []
        self.collision_cone_patch = None

        self.rel_vel_patches = []

    def draw_collision_cone(self, X, obs_list, ax):
        '''
        Render the collision cone based on phi
        obs: [obs_x, obs_y, obs_r]
        '''
        if self.robot_spec['model'] != 'KinematicBicycle2D_C3BF':
            return
        
        # Remove previous collision cones safely
        if not hasattr(self, 'collision_cone_patches'):
            self.collision_cone_patches = [] # Initialize attribute
        
        # Remove previous relative vel safely
        if not hasattr(self, 'rel_vel_patches'):
            self.rel_vel_patches = []

        for patch in list(self.collision_cone_patches):
            if patch in ax.patches:
                patch.remove()
        self.collision_cone_patches.clear()

        for arrow in self.rel_vel_patches:
            arrow.remove()
        self.rel_vel_patches.clear()

        # Robot and obstacle positions
        robot_pos = self.get_position()
        theta = X[2, 0]
        v = X[3, 0]

        obstacles_with_dist = []
        for obs in obs_list:
            obs_pos = np.array([obs[0], obs[1]])
            distance = np.linalg.norm(obs_pos - robot_pos)
            obstacles_with_dist.append((distance, obs))
        
        obstacles_with_dist.sort(key=lambda item: item[0])

        num_to_plot = min(20, len(obstacles_with_dist))
        closest_obs_list = [item[1] for item in obstacles_with_dist[:num_to_plot]]
        
        if num_to_plot > 0:
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_to_plot))
        else:
            colors = []

        for i, obs in enumerate(closest_obs_list):
            obs = np.array(obs).flatten()
            obs_pos = np.array([obs[0], obs[1]])
            obs_radius = obs[2]
            obs_vel_x = obs[3]
            obs_vel_y = obs[4]
            beta = 1.05

            # Combine radius R
            ego_dim = obs_radius + self.robot_spec['radius'] * beta # max(c1,c2) + robot_width/2

            v = X[3, 0]
            p_rel = np.array([[obs[0] - X[0, 0]], 
                        [obs[1] - X[1, 0]]])
            v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                        [obs_vel_y - v * np.sin(theta)]])

            p_rel_mag = np.linalg.norm(p_rel)

            # Calculate Collision cone angle
            phi = np.arcsin(ego_dim / p_rel_mag)

            cone_dir = -p_rel / p_rel_mag
            rot_matrix_left = np.array([[np.cos(phi), -np.sin(phi)],
                                        [np.sin(phi),  np.cos(phi)]])
            rot_matrix_right = np.array([[np.cos(-phi), -np.sin(-phi)],
                                        [np.sin(-phi),  np.cos(-phi)]])
            cone_left = (rot_matrix_left @ cone_dir).flatten()
            cone_right = (rot_matrix_right @ cone_dir).flatten()

            # Extend cone boundaries
            cone_left = (robot_pos + 4 * cone_left).tolist()
            cone_right = (robot_pos + 4 * cone_right).tolist()

            # Draw the cone
            cone_points = np.array ([robot_pos.tolist(), cone_left, cone_right])
            collision_cone_patch = patches.Polygon( # only edgecolors different
                cone_points, closed = True,
                edgecolor=colors[i], linestyle='--', alpha=0.5, label=f"Obstacle {i}"
            )
            ax.add_patch(collision_cone_patch)
            self.collision_cone_patches.append(collision_cone_patch)

            offset_angle = 0.003 * (i - (len(obs_list)//2))
            R_offset = np.array([
                [np.cos(offset_angle), -np.sin(offset_angle)],
                [np.sin(offset_angle),  np.cos(offset_angle)]
            ])
            v_rel_offset = R_offset @ v_rel

            arrow = ax.arrow(float(robot_pos[0]), float(robot_pos[1]),
                            float(v_rel_offset[0]), float(v_rel_offset[1]),
                            color=colors[i], width=0.01, alpha=1.0)
            self.rel_vel_patches.append(arrow)

    def draw_collision_parabola(self, X, obs_list, ax):
        '''
        Render the collision parabola based on functions
        obs: [obs_x, obs_y, obs_r]
        '''
        if self.robot_spec['model'] not in ['KinematicBicycle2D_DPCBF']:
            return

        # Remove previous collision parabolas safely
        if not hasattr(self, 'collision_parabola_patches'):
            self.collision_parabola_patches = [] # Initialize attribute

        # Remove previous relative vel safely
        if not hasattr(self, 'rel_vel_patches'):
            self.rel_vel_patches = []

        for line in self.collision_parabola_patches:
                line.remove()
        self.collision_parabola_patches.clear()

        for arrow in self.rel_vel_patches:
                arrow.remove()
        self.rel_vel_patches.clear()

        # Robot and obstacle positions
        robot_pos = self.get_position()

        obstacles_with_dist = []
        for obs in obs_list:
            obs_pos = np.array([obs[0], obs[1]])
            distance = np.linalg.norm(obs_pos - robot_pos)
            obstacles_with_dist.append((distance, obs))

        obstacles_with_dist.sort(key=lambda item: item[0])

        num_to_plot = min(20, len(obstacles_with_dist))
        closest_obs_list = (item[1] for item in obstacles_with_dist[:num_to_plot])
        
        if num_to_plot > 0:
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_to_plot))
        else:
            colors = []

        for i, obs in enumerate(closest_obs_list):

            theta = X[2, 0]
            v = X[3, 0]

            obs = np.array(obs).flatten()
            obs_pos = np.array([obs[0], obs[1]])
            obs_radius = obs[2]
            obs_vel_x = obs[3]
            obs_vel_y = obs[4]

            # safety margin
            beta = 1.05
            # Combine radius R
            ego_dim = (obs_radius + self.robot_spec['radius']) * beta # max(c1,c2) + robot_width/2 (we suppose safe r as radius)

            p_rel = obs_pos - robot_pos
            v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                            [obs_vel_y - v * np.sin(theta)]])

            p_rel_mag = np.linalg.norm(p_rel)
            v_rel_mag = np.linalg.norm(v_rel)

            # Compute d_safe safely
            eps = 1e-6
            d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)

            # DPCBF functions
            k_lambda, k_mu = 0.1 * np.sqrt(beta**2 - 1)/ego_dim, 0.5 * np.sqrt(beta**2 - 1)/ego_dim
            func_lambda = k_lambda * np.sqrt(d_safe) / v_rel_mag
            func_mu = k_mu * np.sqrt(d_safe)

            rot_angle = np.arctan2(p_rel[1], p_rel[0])
            R = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                        [-np.sin(rot_angle),  np.cos(rot_angle)]])

            L = 1.5
            y_disp = np.linspace(-L, L, 100)
            x_disp = (-func_lambda * (y_disp**2) - func_mu)

            pts_world = robot_pos.reshape(2,1) + R.T @ np.vstack([x_disp, y_disp])
            line, = ax.plot(pts_world[0,:], pts_world[1, :],
                            color=colors[i], linestyle='-', linewidth=2.0,
                            label=f"Quadratic Obs {i}")
            self.collision_parabola_patches.append(line)

            offset_angle = 0.0 * (i - (len(obs_list)//2))
            R_offset = np.array([
                [np.cos(offset_angle), -np.sin(offset_angle)],
                [np.sin(offset_angle),  np.cos(offset_angle)]
            ])
            v_rel_offset = R_offset @ v_rel

            x_offset = 0.8
            y_offset = 0.8
        
            arrow = ax.arrow(float(robot_pos[0]), float(robot_pos[1]),
                            float(x_offset * v_rel_offset[0]), float(y_offset * v_rel_offset[1]),
                            color=colors[i], width=0.02, alpha=1.0)
            self.rel_vel_patches.append(arrow)

             