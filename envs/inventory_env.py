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
            self.ghosts.append({'x': 5.0, 'y': 70.0, 'vx': 2.5, 'vy': 0.0, 'radius': self.ghost_radius})
            # Ghost 4 (V, x=30)
            self.ghosts.append({'x': 30.0, 'y': 95.0, 'vx': 0.0, 'vy': -4.0, 'radius': self.ghost_radius})
            # Ghost 5 (Random/Diagonal)
            self.ghosts.append({'x': 90.0, 'y': 90.0, 'vx': -2.0, 'vy': -2.0, 'radius': self.ghost_radius})
            
        elif lvl >= 3: # Hero Layouts (Lvl 3, 4, 5, 6)
            # Level Configuration
            # Base Speed
            if lvl == 3: 
                base_speed = 2.5
                num_extra = 0
            elif lvl == 4: 
                base_speed = 3.0
                num_extra = 3 # Add 3 more
            elif lvl == 5: 
                base_speed = 3.0
                num_extra = 6 # Add 6 more (Total 11+6=17)
            elif lvl == 6:
                base_speed = 2.7
                num_extra = 10 # Add 10 more (Super dense)
            elif lvl >= 7:
                base_speed = 2.6
                num_extra = 0 # Level 7 uses custom cross-flow only
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

            # Levels 3-5: reduce the y=70 sweeper speed for MPCBF feasibility
            if lvl in [3, 4, 5]:
                 target_speed = 1.5 if lvl == 3 else 2.5
                 for g in self.ghosts:
                      if abs(g.get('y', 0.0) - 70.0) < 1e-6 and abs(g.get('vy', 0.0)) < 1e-9:
                           g['vx'] = target_speed if g.get('vx', 0.0) > 0 else -target_speed

            # Levels 4-5: shift slow blockers away from the diagonal corridor
            if lvl in [4, 5]:
                 for g in self.ghosts:
                      if abs(g.get('x', 0.0) - 20.0) < 1e-6 and abs(g.get('y', 0.0) - 20.0) < 1e-6:
                           g['y'] = 15.0
                           g['vy'] = 0.0
                      if abs(g.get('x', 0.0) - 80.0) < 1e-6 and abs(g.get('y', 0.0) - 80.0) < 1e-6:
                           g['y'] = 85.0
                           g['vy'] = 0.0

            # 4. Level 6 Additions (Super Dense)
            if num_extra >= 10:
                 # Dense horizontal sweepers
                 self.ghosts.append({'x': 5.0, 'y': 15.0, 'vx': speed*0.9, 'vy': 0.0, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 95.0, 'y': 80.0, 'vx': -speed*0.9, 'vy': 0.0, 'radius': self.ghost_radius})
                 # Dense vertical sweepers
                 self.ghosts.append({'x': 20.0, 'y': 95.0, 'vx': 0.0, 'vy': -speed*0.9, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 80.0, 'y': 5.0, 'vx': 0.0, 'vy': speed*0.9, 'radius': self.ghost_radius})
                 # Diagonal interceptors
                 self.ghosts.append({'x': 15.0, 'y': 85.0, 'vx': speed*0.7, 'vy': -speed*0.7, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 85.0, 'y': 15.0, 'vx': -speed*0.7, 'vy': speed*0.7, 'radius': self.ghost_radius})
                 # Corridor blockers (Level 6 only): increase density on nominal zig-zag lines
                 self.ghosts.append({'x': 35.0, 'y': 5.0, 'vx': 0.0, 'vy': speed*0.9, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 75.0, 'y': 95.0, 'vx': 0.0, 'vy': -speed*0.9, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 95.0, 'y': 30.0, 'vx': -speed*0.9, 'vy': 0.0, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 5.0, 'y': 65.0, 'vx': speed*0.9, 'vy': 0.0, 'radius': self.ghost_radius})
                 # Slow vertical blocker that drifts into the mid corridor
                 self.ghosts.append({'x': 50.0, 'y': 10.0, 'vx': 0.0, 'vy': speed*0.45, 'radius': self.ghost_radius})

            # 5. Level 7 Additions (Cross-Flow)
            if lvl >= 7:
                 flow_speed = speed * 0.8
                 # Left-to-right lanes (start left)
                 for y in [44.0, 58.0, 72.0, 86.0]:
                      self.ghosts.append({'x': 5.0, 'y': y, 'vx': flow_speed, 'vy': 0.0, 'radius': self.ghost_radius})
                 # Bottom-to-up lanes (start bottom)
                 for x in [44.0, 58.0, 72.0, 86.0]:
                      self.ghosts.append({'x': x, 'y': 5.0, 'vx': 0.0, 'vy': flow_speed, 'radius': self.ghost_radius})
                 # Diagonal crossers
                 self.ghosts.append({'x': 5.0, 'y': 54.0, 'vx': flow_speed, 'vy': -flow_speed*0.4, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 54.0, 'y': 5.0, 'vx': -flow_speed*0.4, 'vy': flow_speed, 'radius': self.ghost_radius})
                 # Targeted crossers near the mid corridor
                 self.ghosts.append({'x': 40.0, 'y': 5.0, 'vx': 0.0, 'vy': flow_speed*1.1, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 6.0, 'y': 40.0, 'vx': flow_speed*1.1, 'vy': 0.0, 'radius': self.ghost_radius})
                 # Timed sweeper to intersect mid-path around t=20s
                 self.ghosts.append({'x': 95.0, 'y': 34.0, 'vx': -flow_speed*1.1, 'vy': 0.0, 'radius': self.ghost_radius})
                 # Head-on sweeper on y=30 (faster) to pressure nominal corridor
                 self.ghosts.append({'x': 95.0, 'y': 30.0, 'vx': -speed, 'vy': 0.0, 'radius': self.ghost_radius})
                 # Late-stage vertical trap near x=70 to force lateral avoidance
                 self.ghosts.append({'x': 66.0, 'y': 80.0, 'vx': 0.0, 'vy': -flow_speed*1.2, 'radius': self.ghost_radius})
                 self.ghosts.append({'x': 66.0, 'y': 60.0, 'vx': 0.0, 'vy': flow_speed*1.2, 'radius': self.ghost_radius})
                 # Timed horizontal sweeper to intersect nominal at (70,70) around t=20s
                 self.ghosts.append({'x': 20.0, 'y': 70.0, 'vx': speed*1.02, 'vy': 0.0, 'radius': self.ghost_radius})
            
            # Level 6: shift the y=30 sweeper away from the main corridor while staying dense
            if lvl >= 6:
                 for g in self.ghosts:
                      if abs(g.get('x', 0.0) - 5.0) < 1e-6 and abs(g.get('y', 0.0) - 30.0) < 1e-6:
                           g['y'] = 25.0
                      # Redirect slow blocker that crosses (30,30)
                      if abs(g.get('x', 0.0) - 20.0) < 1e-6 and abs(g.get('y', 0.0) - 20.0) < 1e-6:
                           g['x'] = 45.0
                           g['y'] = 25.0
                           g['vy'] = 0.0
                      # Shift vertical sweeper off the x=50 corridor
                      if abs(g.get('x', 0.0) - 50.0) < 1e-6 and abs(g.get('y', 0.0) - 5.0) < 1e-6:
                           g['x'] = 95.0
                      # Shift the x=70 vertical sweeper off the main corridor
                      if abs(g.get('x', 0.0) - 70.0) < 1e-6 and abs(g.get('y', 0.0) - 5.0) < 1e-6:
                           g['x'] = 25.0
                      # Shift the y=50 horizontal sweeper off the main corridor
                      if abs(g.get('y', 0.0) - 50.0) < 1e-6 and abs(g.get('vy', 0.0)) < 1e-9 and g.get('vx', 0.0) < 0:
                           g['y'] = 55.0
                      # Redirect the upper-left slow blocker away from the center corridor
                      if abs(g.get('x', 0.0) - 80.0) < 1e-6 and abs(g.get('y', 0.0) - 80.0) < 1e-6:
                           g['y'] = 85.0
                           g['vy'] = 0.0
                      # Level 6: relieve left-boundary crowding near the start
                      if abs(g.get('x', 0.0) - 5.0) < 1e-6 and abs(g.get('y', 0.0) - 15.0) < 1e-6:
                           g['x'] = 95.0
                           g['vx'] = -abs(g.get('vx', 0.0))
                      if abs(g.get('x', 0.0) - 5.0) < 1e-6 and abs(g.get('y', 0.0) - 25.0) < 1e-6:
                           g['x'] = 95.0
                           g['vx'] = -abs(g.get('vx', 0.0))
                      if abs(g.get('x', 0.0) - 5.0) < 1e-6 and abs(g.get('y', 0.0) - 40.0) < 1e-6:
                           g['x'] = 95.0
                           g['vx'] = -abs(g.get('vx', 0.0))
                      if abs(g.get('x', 0.0) - 5.0) < 1e-6 and abs(g.get('y', 0.0) - 65.0) < 1e-6:
                           g['x'] = 95.0
                           g['vx'] = -abs(g.get('vx', 0.0))
                      if abs(g.get('x', 0.0) - 5.0) < 1e-6 and abs(g.get('y', 0.0) - 70.0) < 1e-6:
                           g['x'] = 95.0
                           g['vx'] = -abs(g.get('vx', 0.0))
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
