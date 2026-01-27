"""
Created on December 21st, 2025
@author: Taekyung Kim

@description:
Evade Environment - A hallway environment for the "evade_bullet_bill" safety scenario.
A robot navigates a straight hallway while evading a fast-moving obstacle (bullet bill).
The robot can hide in a safe pocket on the side of the hallway to let the obstacle pass.

@required-scripts: None (standalone module)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection


class EvadeEnv:
    """
    A hallway environment for evade simulation.
    
    The environment consists of:
    - A straight hallway corridor
    - A safe pocket on the left (top in 2D plot) side
    - A goal zone at the end of the hallway
    - A moving obstacle (bullet bill) that travels along the hallway
    """
    
    def __init__(self, hallway_length=50.0, hallway_width=4.0, 
                 pocket_x=20.0, pocket_length=10.0, pocket_width=4.0,
                 goal_length=5.0,
                 bullet_speed=3.0, bullet_width=None, bullet_length=3.0,
                 bullet_start_x=None):
        """
        Initialize the evade environment.
        
        Args:
            hallway_length: Length of the hallway in meters
            hallway_width: Width of the hallway in meters
            pocket_x: X position where safe pocket starts
            pocket_length: Length of the safe pocket (x direction)
            pocket_width: Width of the safe pocket (y direction, extends from hallway)
            goal_length: Length of goal zone at end of hallway
            bullet_speed: Speed of the bullet bill (m/s)
            bullet_width: Width of bullet (default: covers entire hallway width)
            bullet_length: Length of bullet in x direction
            bullet_start_x: Initial x position of bullet (default: just before hallway)
        """
        self.hallway_length = hallway_length
        self.hallway_width = hallway_width
        self.pocket_x = pocket_x
        self.pocket_length = pocket_length
        self.pocket_width = pocket_width
        self.goal_length = goal_length
        self.bullet_speed = bullet_speed
        self.bullet_width = bullet_width if bullet_width else hallway_width
        self.bullet_length = bullet_length
        
        # Derived dimensions
        self.half_width = hallway_width / 2
        
        # Safe pocket bounds (top left in plot)
        self.pocket_x_min = pocket_x
        self.pocket_x_max = pocket_x + pocket_length
        self.pocket_y_min = self.half_width  # Top of hallway
        self.pocket_y_max = self.half_width + pocket_width
        self.pocket_center = np.array([
            (self.pocket_x_min + self.pocket_x_max) / 2,
            (self.pocket_y_min + self.pocket_y_max) / 2
        ])
        
        # Goal zone bounds
        self.goal_x_min = hallway_length - goal_length
        self.goal_x_max = hallway_length
        
        # Bullet bill state
        self.bullet_start_x = bullet_start_x if bullet_start_x is not None else -self.bullet_length
        self.bullet_x = self.bullet_start_x
        self.bullet_y = 0.0  # Center of hallway
        self.bullet_vx = bullet_speed
        self.bullet_active = True
        
        # Colors for visualization
        self.wall_color = np.array([60, 60, 80]) / 255  # Dark blue-gray
        self.hallway_color = np.array([180, 180, 190]) / 255  # Light gray floor
        self.pocket_color = np.array([100, 150, 200]) / 255  # Light blue safe zone
        self.goal_color = np.array([100, 200, 100]) / 255  # Green goal
        self.bullet_body_color = np.array([30, 30, 30]) / 255  # Black body
        self.bullet_face_color = np.array([255, 255, 255]) / 255  # White face
        
        # Plot handles
        self.ax = None
        self.fig = None
        self.bullet_patches = []
        self.collision_marker = None
        self.goal_marker = None
        
        # Track bounds for plotting
        self.x_min = -5
        self.x_max = hallway_length + 5
        self.y_min = -hallway_width - 5
        self.y_max = hallway_width + pocket_width + 5
        
    def setup_plot(self, ax=None, fig=None):
        """
        Setup the plot with environment visualization.
        
        Args:
            ax: Matplotlib axis (optional)
            fig: Matplotlib figure (optional)
            
        Returns:
            ax, fig: The axis and figure handles
        """
        if fig is None:
            fig = plt.figure(figsize=(14, 8))
        if ax is None:
            ax = fig.add_subplot(111)
            
        self.ax = ax
        self.fig = fig
        
        # Background color
        ax.set_facecolor(np.array([40, 40, 50]) / 255)  # Dark background
        
        # Draw walls (as filled regions outside the hallway)
        # Top wall
        top_wall = Rectangle(
            (self.x_min, self.half_width), 
            self.x_max - self.x_min, 
            self.y_max - self.half_width,
            facecolor=self.wall_color, edgecolor='none', zorder=1
        )
        ax.add_patch(top_wall)
        
        # Bottom wall
        bottom_wall = Rectangle(
            (self.x_min, self.y_min), 
            self.x_max - self.x_min, 
            -self.half_width - self.y_min,
            facecolor=self.wall_color, edgecolor='none', zorder=1
        )
        ax.add_patch(bottom_wall)
        
        # Draw hallway floor
        hallway_floor = Rectangle(
            (0, -self.half_width), 
            self.hallway_length, 
            self.hallway_width,
            facecolor=self.hallway_color, edgecolor='none', zorder=2
        )
        ax.add_patch(hallway_floor)
        
        # Draw safe pocket (cutout in top wall)
        pocket = FancyBboxPatch(
            (self.pocket_x_min, self.pocket_y_min),
            self.pocket_length,
            self.pocket_width,
            boxstyle="round,pad=0.1,rounding_size=0.5",
            facecolor=self.pocket_color,
            edgecolor=(0.3, 0.5, 0.7),
            linewidth=3,
            zorder=3
        )
        ax.add_patch(pocket)
        
        # Draw pocket label
        ax.text(
            self.pocket_center[0], self.pocket_center[1] + 0.5,
            'SAFE', fontsize=10, fontweight='bold',
            color='white', ha='center', va='center', zorder=4
        )
        
        # Draw goal zone
        goal = FancyBboxPatch(
            (self.goal_x_min, -self.half_width),
            self.goal_length,
            self.hallway_width,
            boxstyle="round,pad=0.05,rounding_size=0.3",
            facecolor=self.goal_color,
            edgecolor=(0.2, 0.6, 0.2),
            linewidth=3,
            zorder=3
        )
        ax.add_patch(goal)
        
        # Draw goal flag
        flag_x = self.goal_x_min + self.goal_length / 2
        flag_y = 0
        ax.plot([flag_x, flag_x], [flag_y - 0.5, flag_y + 1.5], 
                'k-', linewidth=3, zorder=5)
        flag_verts = np.array([
            [flag_x, flag_y + 1.5],
            [flag_x + 1.5, flag_y + 1.0],
            [flag_x, flag_y + 0.5]
        ])
        flag = MplPolygon(flag_verts, closed=True, 
                          facecolor=(0.2, 0.8, 0.3), edgecolor='darkgreen',
                          linewidth=2, zorder=6)
        ax.add_patch(flag)
        
        # Draw hallway boundaries
        ax.plot([0, self.hallway_length], [self.half_width, self.half_width],
                'w-', linewidth=4, zorder=3)
        ax.plot([0, self.hallway_length], [-self.half_width, -self.half_width],
                'w-', linewidth=4, zorder=3)
        
        # Draw pocket opening (gap in top boundary)
        ax.plot([0, self.pocket_x_min], [self.half_width, self.half_width],
                'w-', linewidth=4, zorder=4)
        ax.plot([self.pocket_x_max, self.hallway_length], [self.half_width, self.half_width],
                'w-', linewidth=4, zorder=4)
        
        # Draw pocket walls
        ax.plot([self.pocket_x_min, self.pocket_x_min], 
                [self.half_width, self.pocket_y_max],
                'w-', linewidth=3, zorder=4)
        ax.plot([self.pocket_x_max, self.pocket_x_max], 
                [self.half_width, self.pocket_y_max],
                'w-', linewidth=3, zorder=4)
        ax.plot([self.pocket_x_min, self.pocket_x_max], 
                [self.pocket_y_max, self.pocket_y_max],
                'w-', linewidth=3, zorder=4)
        
        # Draw center dashed line in hallway
        ax.plot([0, self.hallway_length], [0, 0],
                'y--', linewidth=2, alpha=0.5, zorder=2)
        
        # Initialize bullet bill
        self._draw_bullet_bill()
        
        # Set axis properties
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        ax.set_title('Evade Bullet Bill', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, color='white')
        
        return ax, fig
    
    def _draw_bullet_bill(self):
        """Draw the bullet bill at its current position."""
        if self.ax is None:
            return
            
        # Clear old patches
        for patch in self.bullet_patches:
            patch.remove()
        self.bullet_patches = []
        
        if not self.bullet_active:
            return
            
        x = self.bullet_x
        y = self.bullet_y
        length = self.bullet_length
        width = self.bullet_width
        
        # Main body (black oval-ish shape)
        body = FancyBboxPatch(
            (x - length/2, y - width/2),
            length, width,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=self.bullet_body_color,
            edgecolor=(0.2, 0.2, 0.2),
            linewidth=2,
            zorder=10
        )
        self.ax.add_patch(body)
        self.bullet_patches.append(body)
        
        # Front nose (pointed)
        nose_verts = np.array([
            [x + length/2, y - width/3],
            [x + length/2 + length/3, y],
            [x + length/2, y + width/3]
        ])
        nose = MplPolygon(nose_verts, closed=True,
                          facecolor=self.bullet_body_color,
                          edgecolor=(0.2, 0.2, 0.2),
                          linewidth=2, zorder=10)
        self.ax.add_patch(nose)
        self.bullet_patches.append(nose)
        
        # Eyes (white circles with black pupils)
        eye_y_offset = width / 4
        eye_x = x + length/4
        eye_radius = width / 5
        pupil_radius = eye_radius / 2
        
        # Left eye
        left_eye = Circle((eye_x, y + eye_y_offset), eye_radius,
                          facecolor='white', edgecolor='black',
                          linewidth=1, zorder=11)
        self.ax.add_patch(left_eye)
        self.bullet_patches.append(left_eye)
        
        left_pupil = Circle((eye_x + pupil_radius/3, y + eye_y_offset), pupil_radius,
                            facecolor='black', zorder=12)
        self.ax.add_patch(left_pupil)
        self.bullet_patches.append(left_pupil)
        
        # Right eye
        right_eye = Circle((eye_x, y - eye_y_offset), eye_radius,
                           facecolor='white', edgecolor='black',
                           linewidth=1, zorder=11)
        self.ax.add_patch(right_eye)
        self.bullet_patches.append(right_eye)
        
        right_pupil = Circle((eye_x + pupil_radius/3, y - eye_y_offset), pupil_radius,
                             facecolor='black', zorder=12)
        self.ax.add_patch(right_pupil)
        self.bullet_patches.append(right_pupil)
        
        # Angry eyebrows
        brow_width = eye_radius * 1.5
        left_brow = plt.Line2D(
            [eye_x - brow_width/2, eye_x + brow_width/2],
            [y + eye_y_offset + eye_radius*1.2, y + eye_y_offset + eye_radius*0.6],
            color='black', linewidth=3, zorder=13
        )
        self.ax.add_line(left_brow)
        self.bullet_patches.append(left_brow)
        
        right_brow = plt.Line2D(
            [eye_x - brow_width/2, eye_x + brow_width/2],
            [y - eye_y_offset - eye_radius*1.2, y - eye_y_offset - eye_radius*0.6],
            color='black', linewidth=3, zorder=13
        )
        self.ax.add_line(right_brow)
        self.bullet_patches.append(right_brow)
        
        # Arms (small rectangles on sides)
        arm_width = length / 6
        arm_height = width / 3
        
        top_arm = Rectangle(
            (x - length/4, y + width/2 - arm_height/4),
            arm_width, arm_height,
            facecolor=(0.1, 0.1, 0.1),
            edgecolor='black',
            linewidth=1, zorder=9
        )
        self.ax.add_patch(top_arm)
        self.bullet_patches.append(top_arm)
        
        bottom_arm = Rectangle(
            (x - length/4, y - width/2 - arm_height*3/4),
            arm_width, arm_height,
            facecolor=(0.1, 0.1, 0.1),
            edgecolor='black',
            linewidth=1, zorder=9
        )
        self.ax.add_patch(bottom_arm)
        self.bullet_patches.append(bottom_arm)
    
    def step_bullet(self, dt):
        """
        Advance the bullet bill position.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            bool: True if bullet respawned
        """
        if not self.bullet_active:
            return False
            
        self.bullet_x += self.bullet_vx * dt
        
        # Check if bullet passed end of hallway - respawn at start position
        respawned = False
        if self.bullet_x > self.hallway_length + self.bullet_length:
            self.bullet_x = self.bullet_start_x
            respawned = True
        
        # Update visualization
        self._draw_bullet_bill()
        
        return respawned
    
    def get_bullet_state(self):
        """
        Get the current bullet bill state.
        
        Returns:
            dict: Bullet state with position, velocity, and dimensions
        """
        # Collision bounds match check_obstacle_collision:
        # Includes a nose of length/3 at the front (positive x direction)
        effective_length = self.bullet_length * (1 + 1/3)
        collision_center_x = self.bullet_x + (self.bullet_length / 6)
        
        return {
            'x': collision_center_x,
            'y': self.bullet_y,
            'vx': self.bullet_vx,
            'vy': 0.0,
            'length': effective_length,
            'width': self.bullet_width,
            'active': self.bullet_active
        }
    
    def check_collision(self, position, robot_radius=0.0):
        """
        Check if a position collides with environment boundaries.
        
        Args:
            position: [x, y] position to check
            robot_radius: Radius of the robot for collision margin
            
        Returns:
            bool: True if collision detected
        """
        x, y = position[0], position[1]
        
        # Check if outside hallway bounds
        # Bottom boundary
        if y - robot_radius < -self.half_width:
            return True
        
        # Top boundary (with pocket exception)
        if y + robot_radius > self.half_width:
            # Check if in pocket area
            if self.pocket_x_min <= x <= self.pocket_x_max:
                # In pocket region, check pocket bounds
                if y + robot_radius > self.pocket_y_max:
                    return True
                # Check pocket side walls
                if x - robot_radius < self.pocket_x_min:
                    if y > self.half_width:
                        return True
                if x + robot_radius > self.pocket_x_max:
                    if y > self.half_width:
                        return True
            else:
                # Outside pocket x-range, collision with top wall
                return True
        
        # Left boundary (start of hallway)
        if x - robot_radius < 0:
            return True
        
        # Right boundary (end of hallway)
        if x + robot_radius > self.hallway_length:
            return True
        
        return False
    
    def check_obstacle_collision(self, position, robot_radius=0.0):
        """
        Check if a position collides with the bullet bill.
        
        Args:
            position: [x, y] position to check
            robot_radius: Radius of the robot for collision margin
            
        Returns:
            tuple: (collision: bool, obstacle_idx: int or None)
        """
        if not self.bullet_active:
            return False, None
        
        x, y = position[0], position[1]
        
        # Bullet bill hitbox (rectangular)
        bullet_x_min = self.bullet_x - self.bullet_length / 2
        bullet_x_max = self.bullet_x + self.bullet_length / 2 + self.bullet_length / 3  # Include nose
        bullet_y_min = self.bullet_y - self.bullet_width / 2
        bullet_y_max = self.bullet_y + self.bullet_width / 2
        
        # Check rectangular collision with circle
        closest_x = np.clip(x, bullet_x_min, bullet_x_max)
        closest_y = np.clip(y, bullet_y_min, bullet_y_max)
        
        dist = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        
        if dist < robot_radius:
            return True, 0
        
        return False, None
    
    def check_goal_reached(self, position):
        """
        Check if the robot has reached the goal zone.
        
        Args:
            position: [x, y] position to check
            
        Returns:
            bool: True if in goal zone
        """
        x, y = position[0], position[1]
        
        return (self.goal_x_min <= x <= self.goal_x_max and 
                -self.half_width <= y <= self.half_width)
    
    def is_in_safe_pocket(self, position, margin=0.0):
        """
        Check if position is inside the safe pocket.
        
        Args:
            position: [x, y] position to check
            margin: Additional margin (negative = stricter)
            
        Returns:
            bool: True if in safe pocket
        """
        x, y = position[0], position[1]
        
        return (self.pocket_x_min + margin <= x <= self.pocket_x_max - margin and
                self.pocket_y_min + margin <= y <= self.pocket_y_max - margin)
    
    def show_collision(self, position):
        """Show collision marker (red exclamation mark) at position."""
        if self.ax is None:
            return
            
        if self.collision_marker is not None:
            try:
                self.collision_marker.remove()
            except:
                pass
        
        x, y = position[0], position[1]
        
        # Simple text exclamation mark like in tracking.py
        self.collision_marker = self.ax.text(
            x + 0.5, y + 0.5, '!', 
            color='red', weight='bold', fontsize=22, zorder=25
        )
    

    def show_goal_reached(self, position):
        """Show goal reached marker (green circle with checkmark) at position."""
        if self.ax is None:
            return
        if self.goal_marker is not None:
            try:
                if isinstance(self.goal_marker, list):
                    for artist in self.goal_marker:
                        artist.remove()
                else:
                    self.goal_marker.remove()
            except Exception:
                pass

        x, y = position[0], position[1]
        
        marker_y = y + 2.0
        radius = 1.2
        circle_color = '#32CD32'  # LimeGreen
        edge_color = '#006400'    # DarkGreen

        from matplotlib.patches import Circle as MplCircle
        bg_circle = MplCircle((x, marker_y), radius, 
                              facecolor=circle_color, 
                              edgecolor=edge_color, 
                              linewidth=2, 
                              zorder=25)
        self.ax.add_patch(bg_circle)

        check_text = self.ax.text(x, marker_y, r"$\checkmark$", 
                                  fontsize=45, 
                                  color='white',
                                  ha='center', 
                                  va='center', 
                                  fontweight='bold',
                                  zorder=26)

        self.goal_marker = [bg_circle, check_text]

    def update_plot_frame(self, ax, position, window_size=(30, 15)):
        """
        Update the plot frame to center around the robot position.
        
        Args:
            ax: Matplotlib axis
            position: [x, y] robot position
            window_size: (width, height) of the view window
        """
        x, y = position[0], position[1]
        half_w, half_h = window_size[0] / 2, window_size[1] / 2
        
        # Calculate new limits, but keep within environment bounds
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
    
    def get_pocket_center(self):
        """Get the center position of the safe pocket."""
        return self.pocket_center.copy()
    
    def get_pocket_bounds(self):
        """Get the bounds of the safe pocket."""
        return {
            'x_min': self.pocket_x_min,
            'x_max': self.pocket_x_max,
            'y_min': self.pocket_y_min,
            'y_max': self.pocket_y_max
        }

