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
        # If Hero Level -> denser grid? Or just more obstacles?
        if str(self.level).lower() == 'hero':
             self.obs_radius = 7.0 # Back to standard
        else:
             self.obs_radius = 7.0 
        
        # Static Obstacles (between hallways)
        self.static_obstacles = []
        # Standard Grid for ALL levels including Hero
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
        if str(self.level).lower() == 'hero':
             self.ghost_speed = 2.5
        
        self.reset()
        
        # Visualization
        self.fig = None
        self.ax = None
        self.ghost_patches = []
        
    def reset(self):
        self.robot_pos = self.start_pos.copy()
        self.ghosts = []
        
        # Parse Level
        lvl = 0
        is_hero_layout = False
        target_speed = 4.0
        
        # New Level Mapping:
        # 0: None
        # 1: Old L3 (3 Ghosts) - Speed 4.0
        # 2: Old L5 (5 Ghosts) - Speed 4.0
        # 3: Hero (All Ghosts) - Speed 2.5
        # 4: Hard Hero - Speed 3.0
        # 5: Nightmare - Speed 3.5
        
        s_level = str(self.level).lower()
        if s_level == 'hero':
            lvl = 3
        else:
            try:
                lvl = int(self.level)
            except:
                lvl = 1
        
        # Determine configuration based on mapped level
        if lvl == 0:
            pass # No ghosts
            
        elif lvl == 1: # Old Level 3
            # Ghost 1 (H, y=50)
            self.ghosts.append({'x': 95.0, 'y': 50.0, 'vx': -4.0, 'vy': 0.0, 'radius': self.ghost_radius})
            # Ghost 2 (V, x=50)
            self.ghosts.append({'x': 50.0, 'y': 5.0, 'vx': 0.0, 'vy': 4.0, 'radius': self.ghost_radius})
            # Ghost 3 (H, y=70)
            self.ghosts.append({'x': 5.0, 'y': 70.0, 'vx': 4.0, 'vy': 0.0, 'radius': self.ghost_radius})
            
        elif lvl == 2: # Old Level 5
            # Ghosts 1-3 from above
            self.ghosts.append({'x': 95.0, 'y': 50.0, 'vx': -4.0, 'vy': 0.0, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 50.0, 'y': 5.0, 'vx': 0.0, 'vy': 4.0, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 5.0, 'y': 70.0, 'vx': 4.0, 'vy': 0.0, 'radius': self.ghost_radius})
            # Ghost 4 (V, x=30)
            self.ghosts.append({'x': 30.0, 'y': 95.0, 'vx': 0.0, 'vy': -4.0, 'radius': self.ghost_radius})
            # Ghost 5 (Random/Diagonal)
            self.ghosts.append({'x': 90.0, 'y': 90.0, 'vx': -2.8, 'vy': -2.8, 'radius': self.ghost_radius})
            
        elif lvl >= 3: # Hero Layouts (Lvl 3, 4, 5)
            # Level Configuration
            extra_ghosts = []
            
            # Base Speed
            if lvl == 3: 
                base_speed = 2.5
                num_extra = 0
            elif lvl == 4: 
                base_speed = 3.0
                num_extra = 3 # Add 3 more
            elif lvl >= 5: 
                base_speed = 3.0
                num_extra = 6 # Add 6 more (Total 11+6=17)
            else:
                base_speed = 2.5
                num_extra = 0

            # 1. Standard Set (11 Ghosts from Hero)
            # Use base_speed for main ghosts
            speed = base_speed
            
            # Main 5
            self.ghosts.append({'x': 95.0, 'y': 50.0, 'vx': -speed, 'vy': 0.0, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 50.0, 'y': 5.0, 'vx': 0.0, 'vy': speed, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 5.0, 'y': 70.0, 'vx': speed, 'vy': 0.0, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 30.0, 'y': 95.0, 'vx': 0.0, 'vy': -speed, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 90.0, 'y': 90.0, 'vx': -speed*0.7, 'vy': -speed*0.7, 'radius': self.ghost_radius})
            
            # Extra Density (6 Ghosts)
            self.ghosts.append({'x': 95.0, 'y': 10.0, 'vx': -speed, 'vy': 0.0, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 5.0, 'y': 30.0, 'vx': speed, 'vy': 0.0, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 95.0, 'y': 90.0, 'vx': -speed, 'vy': 0.0, 'radius': self.ghost_radius})
            
            self.ghosts.append({'x': 10.0, 'y': 95.0, 'vx': 0.0, 'vy': -speed, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 70.0, 'y': 5.0, 'vx': 0.0, 'vy': speed, 'radius': self.ghost_radius})
            self.ghosts.append({'x': 90.0, 'y': 95.0, 'vx': 0.0, 'vy': -speed, 'radius': self.ghost_radius})

            # 2. Level 4 Additions (Varied Velocities)
            if num_extra >= 3:
                 # Slower "Blockers"
                 s_slow = 1.5
                 self.ghosts.append({'x': 20.0, 'y': 20.0, 'vx': s_slow, 'vy': s_slow, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 80.0, 'y': 80.0, 'vx': -s_slow, 'vy': -s_slow, 'radius': self.ghost_radius})
                 # Fast "Interceptor" (Max Speed)
                 self.ghosts.append({'x': 20.0, 'y': 80.0, 'vx': speed, 'vy': -speed, 'radius': self.ghost_radius})

            # 3. Level 5 Additions (Chaos)
            if num_extra >= 6:
                 # More fast chaos
                 self.ghosts.append({'x': 80.0, 'y': 20.0, 'vx': -speed, 'vy': speed, 'radius': self.ghost_radius})
                 # Horizontal sweepers at 40 and 60 (Empty hallways?)
                 self.ghosts.append({'x': 5.0, 'y': 40.0, 'vx': speed*0.8, 'vy': 0.0, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 95.0, 'y': 60.0, 'vx': -speed*0.8, 'vy': 0.0, 'radius': self.ghost_radius})
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
