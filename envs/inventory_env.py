"""
Created on February 4th, 2026
@author: Antigravity

@description:
Inventory Environment.
A grid-like world with circular static obstacles and moving dynamic obstacles (ghosts).
"""
import numpy as np
import matplotlib.pyplot as plt

class InventoryEnv:
    def __init__(self, level=1, dt=0.05):
        self.level = level
        self.dt = dt
        
        # World Bounds
        self.width = 100.0
        self.height = 100.0
        
        # Grid Config
        # 5 Hallways: 10, 30, 50, 70, 90
        self.hallways_x = [10, 30, 50, 70, 90]
        self.hallways_y = [10, 30, 50, 70, 90]
        self.obs_radius = 7.0 # Large enough to block diagonal but leave hallway
        
        # Static Obstacles (between hallways)
        # Centers at (20, 20), (20, 40)...
        self.static_obstacles = []
        for x in [20, 40, 60, 80]:
            for y in [20, 40, 60, 80]:
                self.static_obstacles.append({'x': x, 'y': y, 'radius': self.obs_radius})
        self.obstacles = self.static_obstacles # Alias for Baselines
                
        # Start and Goal
        self.start_pos = np.array([10.0, 10.0])
        self.goal_pos = np.array([90.0, 90.0])
        self.goal_radius = 5.0
        
        # Robot State
        self.robot_pos = self.start_pos.copy()
        
        # Ghosts
        self.ghosts = [] # list of dict {'x', 'y', 'vx', 'vy', 'radius'}
        self.ghost_radius = 2.0
        self.ghost_speed = 4.0
        
        self.reset()
        
        # Visualization
        self.fig = None
        self.ax = None
        self.ghost_patches = []
        
    def reset(self):
        self.robot_pos = self.start_pos.copy()
        self.ghosts = []
        
        # Spawn Ghosts based on Level
        # Level 0: None
        # Level 1: One ghost moving along a hallway (e.g. y=50)
        
        if self.level >= 1:
            # Ghost 1: Horizontal at Y=50
            self.ghosts.append({
                'x': 95.0, 'y': 50.0,
                'vx': -self.ghost_speed, 'vy': 0.0,
                'radius': self.ghost_radius
            })
            
        if self.level >= 2:
            # Ghost 2: Vertical at X=50
            self.ghosts.append({
                'x': 50.0, 'y': 5.0,
                'vx': 0.0, 'vy': self.ghost_speed,
                'radius': self.ghost_radius
            })
            
        if self.level >= 3:
            # Ghost 3: Horizontal at Y=70
            self.ghosts.append({
                'x': 5.0, 'y': 70.0,
                'vx': self.ghost_speed, 'vy': 0.0,
                'radius': self.ghost_radius
            })
            
        if self.level >= 4:
            # Ghost 4: Vertical at X=30
            self.ghosts.append({
                'x': 30.0, 'y': 95.0,
                'vx': 0.0, 'vy': -self.ghost_speed,
                'radius': self.ghost_radius
            })
            
        if self.level >= 5:
            # Ghost 5: Fast random movement? Or just another hallway.
            self.ghosts.append({
                'x': 90.0, 'y': 90.0, # Near Goal
                'vx': -self.ghost_speed * 0.7, 'vy': -self.ghost_speed * 0.7,
                'radius': self.ghost_radius
            })
            
    def step(self):
        # Update Ghosts
        for ghost in self.ghosts:
            ghost['x'] += ghost['vx'] * self.dt
            ghost['y'] += ghost['vy'] * self.dt
            
            # Bounce or Wrap? Let's Bounce
            if ghost['x'] < 2.0 or ghost['x'] > 98.0:
                ghost['vx'] *= -1
            if ghost['y'] < 2.0 or ghost['y'] > 98.0:
                ghost['vy'] *= -1
                
    def get_static_obstacles(self):
        return self.static_obstacles
        
    def get_dynamic_obstacles(self):
        return self.ghosts
        
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')
        
        # Draw Static Obstacles
        for obs in self.static_obstacles:
            circle = plt.Circle((obs['x'], obs['y']), obs['radius'], color='gray', alpha=0.5)
            self.ax.add_patch(circle)
            
        # Draw Goal
        goal = plt.Circle((self.goal_pos[0], self.goal_pos[1]), self.goal_radius, color='green', alpha=0.3)
        self.ax.add_patch(goal)
        self.ax.text(self.goal_pos[0], self.goal_pos[1], 'GOAL', ha='center', va='center')
        
        # Init Ghosts
        self.ghost_patches = []
        for ghost in self.ghosts:
            patch = plt.Circle((ghost['x'], ghost['y']), ghost['radius'], color='red')
            self.ax.add_patch(patch)
            self.ghost_patches.append(patch)
            
        return self.fig, self.ax
        
    def update_plot(self):
        for i, ghost in enumerate(self.ghosts):
            if i < len(self.ghost_patches):
                self.ghost_patches[i].center = (ghost['x'], ghost['y'])
                
    def check_collision(self, pos, radius):
        """Check boundary collision."""
        x, y = pos
        if x < radius or x > self.width - radius:
            return True
        if y < radius or y > self.height - radius:
            return True
        return False
        
    def check_obstacle_collision(self, pos, radius):
        """Check static obstacle collision."""
        # Returns (collision_bool, obstacle_object)
        for obs in self.static_obstacles:
            dist = np.linalg.norm(pos - np.array([obs['x'], obs['y']]))
            if dist < (obs['radius'] + radius):
                return True, obs
        return False, None

    def get_nominal_waypoints(self):
        """Diagonal ZigZag path."""
        # Nodes: (10,10)->(30,10)->(30,30)->(50,30)->(50,50)->(70,50)->(70,70)->(90,70)->(90,90)
        wps = [
            (10, 10), 
            (30, 10), (30, 30),
            (50, 30), (50, 50),
            (70, 50), (70, 70),
            (90, 70), (90, 90)
        ]
        return np.array(wps)
