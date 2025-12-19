"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Drifting Environment - A racing track environment for drift simulation.
This module provides a simple racing track with left and right boundaries
for collision checking. Supports straight, oval, and L-shaped track types.

@required-scripts: None (standalone module)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


class DriftingEnv:
    """
    A racing track environment for drift simulation.
    
    The track is defined as a simple oval/rectangular racing track with
    configurable dimensions. Boundaries are computed for collision checking.
    """
    
    def __init__(self, track_type='straight', track_width=8.0, track_length=100.0, num_lanes=1):
        """
        Initialize the drifting environment.
        
        Args:
            track_type: Type of track ('straight', 'oval', 'l_shape')
            track_width: Width of the track in meters (total width for all lanes)
            track_length: Length of the track in meters
            num_lanes: Number of lanes (default 1, use 5 for multi-lane)
        """
        self.track_type = track_type
        self.track_width = track_width
        self.track_length = track_length
        self.num_lanes = num_lanes
        self.lane_width = track_width / num_lanes if num_lanes > 1 else track_width
        
        # Generate track boundaries
        self.left_boundary = None
        self.right_boundary = None
        self.centerline = None
        self.lane_centers = []  # Y-positions of lane centers
        
        self._generate_track()
        
        # Colors for visualization (similar to MATLAB style)
        self.road_color = np.array([150, 150, 150]) / 255  # Gray asphalt
        self.shoulder_color = np.array([100, 100, 100]) / 255  # Darker gray for outer lanes
        self.grass_color = np.array([100, 180, 100]) / 255  # Green grass
        self.line_color = 'white'
        self.center_line_color = 'yellow'
        
        # Plot handles
        self.ax = None
        self.road_patch = None
        self.lane_patches = []  # Patches for individual lanes
        self.left_boundary_line = None
        self.right_boundary_line = None
        self.center_line = None
        self.lane_lines = []  # Lane divider lines
        
        # Puddles on the road (list of dicts with 'x', 'y', 'radius', 'friction')
        self.puddles = []
        self.puddle_patches = []
        
        # Static obstacles (other cars) - list of dicts with 'x', 'y', 'theta', 'spec'
        self.obstacles = []
        self.obstacle_patches = []
        
    def _generate_track(self):
        """Generate track boundaries based on track type."""
        if self.track_type == 'straight':
            self._generate_straight_track()
        elif self.track_type == 'oval':
            self._generate_oval_track()
        elif self.track_type == 'l_shape':
            self._generate_l_shape_track()
        else:
            raise ValueError(f"Unknown track type: {self.track_type}")
    
    def _generate_straight_track(self):
        """Generate a straight track with optional multiple lanes."""
        # Centerline points
        n_points = 100
        x = np.linspace(0, self.track_length, n_points)
        y = np.zeros(n_points)
        
        self.centerline = np.column_stack([x, y])
        
        # Compute boundaries (perpendicular to centerline)
        half_width = self.track_width / 2
        self.left_boundary = np.column_stack([x, y + half_width])
        self.right_boundary = np.column_stack([x, y - half_width])
        
        # Calculate lane centers (from left to right, i.e., top to bottom in plot)
        # Lane 0 is leftmost (top), Lane num_lanes-1 is rightmost (bottom)
        self.lane_centers = []
        if self.num_lanes > 1:
            for i in range(self.num_lanes):
                # Lane center Y position
                lane_y = half_width - (i + 0.5) * self.lane_width
                self.lane_centers.append(lane_y)
        else:
            self.lane_centers = [0.0]  # Single lane at center
        
        # Track bounds for plotting
        self.x_min = -5
        self.x_max = self.track_length + 5
        self.y_min = -self.track_width - 5
        self.y_max = self.track_width + 5
    
    def get_lane_center(self, lane_idx):
        """
        Get the Y-position of a lane center.
        
        Args:
            lane_idx: Lane index (0 = leftmost/top, num_lanes-1 = rightmost/bottom)
            
        Returns:
            float: Y-coordinate of the lane center
        """
        if lane_idx < 0 or lane_idx >= len(self.lane_centers):
            raise ValueError(f"Invalid lane index {lane_idx}. Must be 0 to {len(self.lane_centers)-1}")
        return self.lane_centers[lane_idx]
    
    def get_middle_lane_idx(self):
        """Get the index of the middle lane."""
        return self.num_lanes // 2
        
    def _generate_oval_track(self):
        """Generate an oval track with gentler curves."""
        n_points = 200
        
        # Oval parameters - make semi-minor axis larger for gentler turns
        a = self.track_length / 2  # Semi-major axis
        b = self.track_length / 2.5  # Semi-minor axis
        
        # Parametric oval
        t = np.linspace(0, 2 * np.pi, n_points)
        x = a * np.cos(t) + a
        y = b * np.sin(t)
        
        self.centerline = np.column_stack([x, y])
        
        # Compute boundaries using normal vectors
        half_width = self.track_width / 2
        
        # Compute tangent and normal vectors
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length  # Normal x
        ny = dx / length   # Normal y
        
        self.left_boundary = np.column_stack([x + half_width * nx, y + half_width * ny])
        self.right_boundary = np.column_stack([x - half_width * nx, y - half_width * ny])
        
        # Track bounds
        self.x_min = -10
        self.x_max = 2 * a + 10
        self.y_min = -b - self.track_width - 5
        self.y_max = b + self.track_width + 5
        
        # Lane centers (for oval, just use centerline - no multi-lane support)
        self.lane_centers = [0.0]
        
    def _generate_l_shape_track(self):
        """Generate an L-shaped track."""
        half_width = self.track_width / 2
        
        # L-shape centerline (two segments)
        seg1_length = self.track_length * 0.6
        seg2_length = self.track_length * 0.4
        
        # First segment (horizontal)
        n1 = 60
        x1 = np.linspace(0, seg1_length, n1)
        y1 = np.zeros(n1)
        
        # Corner (arc)
        n_corner = 20
        corner_radius = self.track_width
        theta = np.linspace(-np.pi/2, 0, n_corner)
        x_corner = seg1_length + corner_radius + corner_radius * np.cos(theta)
        y_corner = corner_radius + corner_radius * np.sin(theta)
        
        # Second segment (vertical)
        n2 = 40
        x2 = np.full(n2, seg1_length + corner_radius)
        y2 = np.linspace(corner_radius, corner_radius + seg2_length, n2)
        
        # Combine
        x = np.concatenate([x1, x_corner, x2])
        y = np.concatenate([y1, y_corner, y2])
        
        self.centerline = np.column_stack([x, y])
        
        # Compute boundaries
        dx = np.gradient(x)
        dy = np.gradient(y)
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length
        ny = dx / length
        
        self.left_boundary = np.column_stack([x + half_width * nx, y + half_width * ny])
        self.right_boundary = np.column_stack([x - half_width * nx, y - half_width * ny])
        
        # Track bounds
        self.x_min = -5
        self.x_max = seg1_length + 2 * corner_radius + 5
        self.y_min = -self.track_width - 5
        self.y_max = corner_radius + seg2_length + 5
        
        # Lane centers (for L-shape, just use centerline - no multi-lane support)
        self.lane_centers = [0.0]
    
    def setup_plot(self, ax=None, fig=None):
        """
        Setup the plot with track visualization.
        
        Args:
            ax: Matplotlib axis (optional)
            fig: Matplotlib figure (optional)
            
        Returns:
            ax, fig: The axis and figure handles
        """
        if fig is None:
            fig = plt.figure(figsize=(14, 6))
        if ax is None:
            ax = fig.add_subplot(111)
            
        self.ax = ax
        self.fig = fig
        
        # Draw grass background
        ax.set_facecolor(self.grass_color)
        
        # Draw road surface with multiple lanes if applicable
        if self.num_lanes > 1 and self.track_type == 'straight':
            self._draw_multi_lane_road(ax)
        else:
            # Single lane road
            road_vertices = np.vstack([self.left_boundary, self.right_boundary[::-1]])
            road_polygon = MplPolygon(road_vertices, closed=True, 
                                       facecolor=self.road_color, edgecolor='none')
            ax.add_patch(road_polygon)
            self.road_patch = road_polygon
        
        # Draw boundaries
        self.left_boundary_line, = ax.plot(
            self.left_boundary[:, 0], self.left_boundary[:, 1],
            color=self.line_color, linewidth=3, solid_capstyle='round'
        )
        self.right_boundary_line, = ax.plot(
            self.right_boundary[:, 0], self.right_boundary[:, 1],
            color=self.line_color, linewidth=3, solid_capstyle='round'
        )
        
        # Draw lane divider lines for multi-lane roads
        if self.num_lanes > 1 and self.track_type == 'straight':
            self._draw_lane_dividers(ax)
        else:
            # Draw center line (dashed yellow) for single lane
            self.center_line, = ax.plot(
                self.centerline[:, 0], self.centerline[:, 1],
                color=self.center_line_color, linewidth=2, linestyle='--'
            )
        
        # Set axis properties
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.3)
        
        return ax, fig
    
    def _draw_multi_lane_road(self, ax):
        """Draw multi-lane road with colored outer lanes."""
        half_width = self.track_width / 2
        x_start = self.left_boundary[0, 0]
        x_end = self.left_boundary[-1, 0]
        
        self.lane_patches = []
        
        for i in range(self.num_lanes):
            # Lane boundaries (top and bottom of lane)
            lane_top = half_width - i * self.lane_width
            lane_bottom = half_width - (i + 1) * self.lane_width
            
            # Determine lane color (outer lanes are darker)
            if i == 0 or i == self.num_lanes - 1:
                lane_color = self.shoulder_color  # Outer lanes (shoulders)
            else:
                lane_color = self.road_color  # Middle lanes
            
            # Create lane polygon
            lane_vertices = np.array([
                [x_start, lane_top],
                [x_end, lane_top],
                [x_end, lane_bottom],
                [x_start, lane_bottom]
            ])
            
            lane_patch = MplPolygon(lane_vertices, closed=True,
                                     facecolor=lane_color, edgecolor='none', zorder=1)
            ax.add_patch(lane_patch)
            self.lane_patches.append(lane_patch)
    
    def _draw_lane_dividers(self, ax):
        """Draw lane divider lines."""
        half_width = self.track_width / 2
        x_start = self.left_boundary[0, 0]
        x_end = self.left_boundary[-1, 0]
        
        self.lane_lines = []
        
        for i in range(1, self.num_lanes):
            # Y position of lane divider
            divider_y = half_width - i * self.lane_width
            
            # Dashed white line for lane dividers
            line, = ax.plot([x_start, x_end], [divider_y, divider_y],
                           color='white', linewidth=2, linestyle='--', alpha=0.8, zorder=2)
            self.lane_lines.append(line)
    
    def check_collision(self, position, robot_radius=0.0):
        """
        Check if a position collides with track boundaries.
        
        Args:
            position: [x, y] position to check
            robot_radius: Radius of the robot for collision margin
            
        Returns:
            bool: True if collision detected
        """
        x, y = position[0], position[1]
        
        # Find closest point on centerline
        distances = np.linalg.norm(self.centerline - np.array([x, y]), axis=1)
        closest_idx = np.argmin(distances)
        
        # Get corresponding boundary points
        left_pt = self.left_boundary[closest_idx]
        right_pt = self.right_boundary[closest_idx]
        center_pt = self.centerline[closest_idx]
        
        # Compute distance from center to boundary (half track width)
        half_width = np.linalg.norm(left_pt - center_pt)
        
        # Check if position is outside track
        dist_from_center = np.linalg.norm(np.array([x, y]) - center_pt)
        
        if dist_from_center + robot_radius > half_width:
            return True
        
        return False
    
    def check_collision_detailed(self, position, robot_radius=0.0):
        """
        Detailed collision check returning which boundary was hit.
        
        Args:
            position: [x, y] position to check
            robot_radius: Radius of the robot for collision margin
            
        Returns:
            dict: {'collision': bool, 'boundary': 'left'/'right'/None, 'distance': float}
        """
        x, y = position[0], position[1]
        
        # Find closest points on boundaries
        dist_to_left = np.min(np.linalg.norm(self.left_boundary - np.array([x, y]), axis=1))
        dist_to_right = np.min(np.linalg.norm(self.right_boundary - np.array([x, y]), axis=1))
        
        # Find closest point on centerline for local track direction
        distances = np.linalg.norm(self.centerline - np.array([x, y]), axis=1)
        closest_idx = np.argmin(distances)
        
        # Get local track coordinate
        left_pt = self.left_boundary[closest_idx]
        right_pt = self.right_boundary[closest_idx]
        center_pt = self.centerline[closest_idx]
        
        # Vector from right to left boundary (perpendicular to track)
        track_normal = left_pt - right_pt
        track_normal = track_normal / np.linalg.norm(track_normal)
        
        # Vector from center to position
        pos_vec = np.array([x, y]) - center_pt
        
        # Signed distance (positive = towards left, negative = towards right)
        signed_dist = np.dot(pos_vec, track_normal)
        half_width = self.track_width / 2
        
        result = {
            'collision': False,
            'boundary': None,
            'distance': min(dist_to_left, dist_to_right),
            'signed_distance': signed_dist
        }
        
        if signed_dist > half_width - robot_radius:
            result['collision'] = True
            result['boundary'] = 'left'
        elif signed_dist < -(half_width - robot_radius):
            result['collision'] = True
            result['boundary'] = 'right'
            
        return result
    
    def get_track_bounds(self):
        """Return the track boundary data for external use."""
        return {
            'left_boundary': self.left_boundary.copy(),
            'right_boundary': self.right_boundary.copy(),
            'centerline': self.centerline.copy(),
            'track_width': self.track_width
        }
    
    def add_puddle(self, x, y, radius, friction=0.3):
        """
        Add a puddle (low friction area) to the track.
        
        Args:
            x: X position of puddle center
            y: Y position of puddle center
            radius: Radius of the puddle
            friction: Friction coefficient in the puddle (default 0.3)
        """
        puddle = {
            'x': x,
            'y': y,
            'radius': radius,
            'friction': friction
        }
        self.puddles.append(puddle)
        
        # Draw puddle if plot is set up
        if self.ax is not None:
            from matplotlib.patches import Circle
            puddle_patch = Circle(
                (x, y), radius,
                facecolor=(0.3, 0.5, 0.8, 0.5),  # Blue with transparency
                edgecolor=(0.2, 0.4, 0.7),
                linewidth=2,
                zorder=2
            )
            self.ax.add_patch(puddle_patch)
            self.puddle_patches.append(puddle_patch)
    
    def get_friction_at_position(self, position, default_friction=1.0):
        """
        Get the friction coefficient at a given position.
        
        Args:
            position: [x, y] position to check
            default_friction: Default friction if not in any puddle
            
        Returns:
            float: Friction coefficient at the position
        """
        x, y = position[0], position[1]
        
        for puddle in self.puddles:
            dist = np.sqrt((x - puddle['x'])**2 + (y - puddle['y'])**2)
            if dist <= puddle['radius']:
                return puddle['friction']
        
        return default_friction
    
    def add_obstacle_car(self, x, y, theta, robot_spec=None):
        """
        Add a static obstacle car to the track.
        
        Args:
            x: X position of obstacle car
            y: Y position of obstacle car
            theta: Heading angle of obstacle car
            robot_spec: Robot specification dict (optional)
            
        Returns:
            int: Index of the added obstacle
        """
        if robot_spec is None:
            robot_spec = {
                'body_length': 4.5,
                'body_width': 2.0,
                'a': 1.4,  # Front axle to CG
                'b': 1.4,  # Rear axle to CG
                'radius': 2.5,  # Collision radius (larger for safety)
            }
        
        obstacle = {
            'x': x,
            'y': y,
            'theta': theta,
            'spec': robot_spec
        }
        self.obstacles.append(obstacle)
        
        # Draw obstacle car if plot is set up
        if self.ax is not None:
            self._draw_obstacle_car(obstacle)
        
        return len(self.obstacles) - 1
    
    def _draw_obstacle_car(self, obstacle):
        """Draw a static obstacle car."""
        x, y, theta = obstacle['x'], obstacle['y'], obstacle['theta']
        spec = obstacle['spec']
        
        a = spec.get('a', 1.6)
        b = spec.get('b', 0.8)
        L = spec.get('body_length', 4.3)
        W = spec.get('body_width', 1.8)
        
        # Body vertices (centered at CG)
        rear_overhang = (L - a - b) * 0.4
        front_overhang = (L - a - b) * 0.6
        
        # Main body outline (counterclockwise from rear-left)
        body_vertices = np.array([
            [-b - rear_overhang, -W/2],
            [-b - rear_overhang, W/2],
            [-b - rear_overhang + 0.3, W/2 + 0.05],
            [a + front_overhang - 0.8, W/2 + 0.05],
            [a + front_overhang - 0.3, W/2 * 0.7],
            [a + front_overhang, W/2 * 0.5],
            [a + front_overhang, -W/2 * 0.5],
            [a + front_overhang - 0.3, -W/2 * 0.7],
            [a + front_overhang - 0.8, -W/2 - 0.05],
            [-b - rear_overhang + 0.3, -W/2 - 0.05],
        ]).T
        
        # Rotation matrix
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Transform body vertices
        body_world = R @ body_vertices + np.array([[x], [y]])
        
        # Draw body
        body_patch = MplPolygon(
            body_world.T, closed=True,
            facecolor=(0.7, 0.2, 0.2),  # Dark red color
            edgecolor='black',
            linewidth=1.5, alpha=0.9, zorder=8
        )
        self.ax.add_patch(body_patch)
        self.obstacle_patches.append(body_patch)
        
        # Draw tires
        tire_length = 0.6
        tire_width = 0.25
        tire_y_offset = W / 2 - tire_width / 2 - 0.1
        tire_positions = {
            'front_left': np.array([a, tire_y_offset]),
            'front_right': np.array([a, -tire_y_offset]),
            'rear_left': np.array([-b, tire_y_offset]),
            'rear_right': np.array([-b, -tire_y_offset])
        }
        
        tl = tire_length / 2
        tw = tire_width / 2
        tire_vertices = np.array([
            [-tl, -tw],
            [-tl, tw],
            [tl, tw],
            [tl, -tw]
        ]).T
        
        for name, pos in tire_positions.items():
            pos_world = R @ pos.reshape(-1, 1) + np.array([[x], [y]])
            tire_world = R @ tire_vertices + pos_world
            tire_patch = MplPolygon(
                tire_world.T, closed=True,
                facecolor=(0.3, 0.3, 0.3),
                edgecolor='black',
                linewidth=1, alpha=0.9, zorder=9
            )
            self.ax.add_patch(tire_patch)
            self.obstacle_patches.append(tire_patch)
    
    def check_obstacle_collision(self, position, robot_radius=0.0):
        """
        Check if a position collides with any obstacle cars.
        
        Args:
            position: [x, y] position to check
            robot_radius: Radius of the robot for collision margin
            
        Returns:
            bool: True if collision detected
            int or None: Index of collided obstacle, or None
        """
        x, y = position[0], position[1]
        
        for i, obstacle in enumerate(self.obstacles):
            obs_x, obs_y = obstacle['x'], obstacle['y']
            obs_radius = obstacle['spec'].get('radius', 2.5)
            
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dist < (obs_radius + robot_radius):
                return True, i
        
        return False, None
    
    def update_plot_frame(self, ax, position, window_size=(40, 20)):
        """
        Update the plot frame to center around the robot position.
        
        Args:
            ax: Matplotlib axis
            position: [x, y] robot position
            window_size: (width, height) of the view window
        """
        x, y = position[0], position[1]
        half_w, half_h = window_size[0] / 2, window_size[1] / 2
        
        # Calculate new limits, but keep within track bounds
        x_min = max(self.x_min, x - half_w)
        x_max = min(self.x_max, x + half_w)
        y_min = max(self.y_min, y - half_h)
        y_max = min(self.y_max, y + half_h)
        
        # Adjust if we hit boundaries
        if x_max - x_min < window_size[0]:
            if x_min == self.x_min:
                x_max = x_min + window_size[0]
            elif x_max == self.x_max:
                x_min = x_max - window_size[0]
        
        if y_max - y_min < window_size[1]:
            if y_min == self.y_min:
                y_max = y_min + window_size[1]
            elif y_max == self.y_max:
                y_min = y_max - window_size[1]
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

