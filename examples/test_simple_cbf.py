
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

from safe_control.robots.robot import BaseRobot
from safe_control.position_control.cbf_qp import CBFQP

# Parameters
DT = 0.01  # 100Hz
T_MAX = 10.0 # 10 seconds per trial
WORKSPACE_X = [0, 10.0] 
WORKSPACE_Y = [0, 6.7]
SCALE = 60.0 

def to_meters(pixels):
    return pixels / SCALE

def to_pixels(meters):
    return meters * SCALE

class SimulationRunner:
    def __init__(self, model_name, mode, alpha, num_trials, num_obs, save_plots, max_vel=None):
        self.model_name = model_name
        self.mode = mode # 'cbf' or 'hard'
        self.alpha_val = alpha
        self.num_trials = num_trials
        self.num_obs = num_obs
        self.save_plots = save_plots
        self.max_vel = max_vel # For double integrator sweep
        
        self.robot = None
        self.cbf_controller = None
        self.setup_robot()
        
    def setup_robot(self):
        # Setup Figure for BaseRobot
        self.fig, self.ax = plt.subplots(figsize=(10, 6.7))
        self.ax.set_xlim(WORKSPACE_X)
        self.ax.set_ylim(WORKSPACE_Y)
        self.ax.set_aspect('equal')
        
        radius = to_meters(15) 
        
        spec = {
            'radius': radius,
            'cbf_mode': self.mode,
            'no_heading': True
        }
        
        if self.model_name == 'single':
            spec['model'] = 'SingleIntegrator2D'
            spec['v_max'] = 5.0
            # Alpha
            self.cbf_alpha = self.alpha_val
            
        elif self.model_name == 'double':
            spec['model'] = 'DoubleIntegrator2D'
            spec['a_max'] = 1.0 # default
            spec['v_max'] = self.max_vel if self.max_vel else 5.0
            # Alpha
            self.cbf_alpha = self.alpha_val
            
        elif self.model_name == 'manipulator':
            spec['model'] = 'Manipulator2D'
            spec['w_max'] = 2.0
            spec['Kp'] = 3.0
            spec['radius'] = to_meters(15) 
            # Alpha
            self.cbf_alpha = self.alpha_val

        self.spec = spec
        
        # Init BaseRobot
        # Initial state dummy, will reset in trial
        if self.model_name == 'manipulator':
            x0 = np.zeros(3)
        elif self.model_name == 'double':
             # x, y, vx, vy, theta
            x0 = np.zeros(5)
        else:
             # x, y, theta (BaseRobot calls set_orientation on idx 2)
            x0 = np.zeros(3)
            
        self.robot = BaseRobot(x0, spec, DT, self.ax)
        
        if self.model_name == 'manipulator':
             # Set logical center
             self.robot.robot.base_pos = np.array([5.0, 3.35])

        # Init CBFQP
        # Num constraints:
        # Manipulator: num_obs * 3 links.
        # Others: num_obs.
        if self.model_name == 'manipulator':
            # Each obstacle creates constraints for ALL link circles.
            # Links: ~23 circles total (High coverage).
            # Allocate 30 per obstacle to be safe.
            n_constraints = self.num_obs * 30
        else:
            n_constraints = self.num_obs
            
        self.cbf_controller = CBFQP(self.robot, spec, num_obs=n_constraints)
        
        # Override alpha in controller
        if self.model_name == 'single':
             self.cbf_controller.cbf_param['alpha'] = self.alpha_val
        elif self.model_name == 'double':
             self.cbf_controller.cbf_param['alpha1'] = self.alpha_val
             self.cbf_controller.cbf_param['alpha2'] = self.alpha_val
        elif self.model_name == 'manipulator':
             self.cbf_controller.cbf_param['alpha'] = self.alpha_val

        # Visualization Data (Init once)
        self.goal_marker, = self.ax.plot([], [], 'g*', markersize=15, zorder=10)
        self.obs_patches = []

        for _ in range(self.num_obs + 5): 
            c = plt.Circle((0,0), 0.1, color='r', alpha=0.5, visible=False)
            self.ax.add_patch(c)
            self.obs_patches.append(c)

    def generate_random_state(self):
        if self.model_name == 'manipulator':
            start_angles = np.random.uniform(-np.pi, np.pi, (3,1))
            # Goal (Cartesian)
            r = np.random.uniform(0.5, 2.0)
            theta = np.random.uniform(0, 2*np.pi)
            base = self.robot.robot.base_pos
            goal_pos = base + np.array([r * np.cos(theta), r * np.sin(theta)])
            return start_angles, goal_pos.reshape(2,1)
        else:
            start_pos = np.random.uniform([WORKSPACE_X[0], WORKSPACE_Y[0]], [WORKSPACE_X[1], WORKSPACE_Y[1]])
            goal_pos = np.random.uniform([WORKSPACE_X[0], WORKSPACE_Y[0]], [WORKSPACE_X[1], WORKSPACE_Y[1]])
            while np.linalg.norm(start_pos - goal_pos) < 1.0:
                 goal_pos = np.random.uniform([WORKSPACE_X[0], WORKSPACE_Y[0]], [WORKSPACE_X[1], WORKSPACE_Y[1]])
            
            if self.model_name == 'double':
                start_state = np.zeros((4,1))
                start_state[0:2, 0] = start_pos
                # # Random Velocity
                # v_limit = self.spec['v_max']
                # start_state[2, 0] = np.random.uniform(-v_limit, v_limit)
                # start_state[3, 0] = np.random.uniform(-v_limit, v_limit)
            else:
                start_state = start_pos.reshape(2,1)
            return start_state, goal_pos.reshape(2,1)
            
    def check_goal_reached(self, state, goal):
        if self.model_name == 'manipulator':
            ee = self.robot.robot.get_end_effector(state)
            dist = np.linalg.norm(ee - goal.flatten())
        else:
            dist = np.linalg.norm(state[0:2] - goal[0:2])
        return dist < 0.2

    def check_collision(self, state, obstacles):
        # Actual physics collision: dist < d_min (no margin)
        # obstacles is list of [x,y,r,...]
        if self.model_name == 'manipulator':
            link_circles = self.robot.robot.get_link_circles(state, self.spec['radius'])
            for obs in obstacles:
                ox, oy, r = obs[0], obs[1], obs[2]
                for c in link_circles:
                     d = np.sqrt((c['x']-ox)**2 + (c['y']-oy)**2)
                     if d < c['r'] + r: return True
        else:
            px, py = state[0,0], state[1,0]
            rob_r = self.spec['radius']
            for obs in obstacles:
                 d = np.sqrt((px-obs[0])**2 + (py-obs[1])**2)
                 if d < rob_r + obs[2]: return True
        return False

    def check_valid_obstacle(self, obs, state):
        # Use barrier margin (sqrt(beta)) to ensure h >= 0 at start
        # This is stricter than collision check
        margin = np.sqrt(1.3)  # Match barrier beta=1.3
        if self.model_name == 'manipulator':
            link_circles = self.robot.robot.get_link_circles(state, self.spec['radius'])
            ox, oy, r = obs[0], obs[1], obs[2]
            for c in link_circles:
                 d = np.sqrt((c['x']-ox)**2 + (c['y']-oy)**2)
                 if d < margin * (c['r'] + r): return False
            return True
        else:
            px, py = state[0,0], state[1,0]
            rob_r = self.spec['radius']
            d = np.sqrt((px-obs[0])**2 + (py-obs[1])**2)
            return d >= margin * (rob_r + obs[2])

    def run_trials(self):
        stats = {'collisions': 0, 'infeasible': 0}
        obs_radius = to_meters(25)
        
        for i in range(self.num_trials):
            np.random.seed(i)
            start_state, goal_state = self.generate_random_state()
            self.robot.X = np.copy(start_state)
            if self.model_name in ['single', 'manipulator']:
                 # Reset previous controls
                 pass
                 
            # Obstacles
            obstacles = []
            
            attempts = 0
            while len(obstacles) < self.num_obs and attempts < 1000:
                ox = np.random.uniform(WORKSPACE_X[0], WORKSPACE_X[1])
                oy = np.random.uniform(WORKSPACE_Y[0], WORKSPACE_Y[1])
                obs_vec = np.array([ox, oy, obs_radius, 0, 0, 0, 0])
                
                # Check collision with Robot Start
                if self.check_valid_obstacle(obs_vec, start_state):
                    # Check collision with Goal? (Optional, but good practice)
                    obstacles.append(obs_vec)
                attempts += 1
                
            # Loop
            trial_collided = False
            trial_infeasible = False
            
            # Update Visualization Data (Pre-Loop)
            if self.save_plots and i == self.num_trials-1:
                # Obstacles
                for idx, obs in enumerate(obstacles):
                    if idx < len(self.obs_patches):
                        self.obs_patches[idx].center = (obs[0], obs[1])
                        self.obs_patches[idx].set_radius(obs[2])
                        self.obs_patches[idx].set_visible(True)
                for idx in range(len(obstacles), len(self.obs_patches)):
                        self.obs_patches[idx].set_visible(False)
                # Goal
                self.goal_marker.set_data([goal_state[0]], [goal_state[1]])
                 
            for t in np.arange(0, T_MAX, DT):
                if self.model_name == 'double':
                     u_nom = self.robot.nominal_input(goal_state, k_v=4.0, k_a=4.0)
                else:
                     u_nom = self.robot.nominal_input(goal_state)
                control_ref = {'u_ref': u_nom} # cbf_qp expects dict
                
                # Solve QP with error handling
                try:
                    u_safe = self.cbf_controller.solve_control_problem(self.robot.X, control_ref, obstacles)
                    
                    # Check status
                    if self.cbf_controller.status not in ['optimal', 'optimal_inaccurate']:
                        trial_infeasible = True
                        u_safe = np.zeros_like(u_nom)
                except Exception as e:
                    # Solver crash - treat as collision
                    print(f"Solver error: {e}")
                    trial_collided = True
                    break
                    
                # Step
                self.robot.step(u_safe)
                
                # Check Collision
                if self.check_collision(self.robot.X, obstacles):
                    trial_collided = True
                    break
                    
                # Check Goal Reached -> Update Goal
                if self.check_goal_reached(self.robot.X, goal_state):
                    # Generate new goal
                    if self.model_name == 'manipulator':
                        _, goal_state = self.generate_random_state()
                    else:
                        # Random point in workspace
                        pt = np.random.uniform([WORKSPACE_X[0], WORKSPACE_Y[0]], [WORKSPACE_X[1], WORKSPACE_Y[1]])
                        goal_state = pt.reshape(2,1)
                
                if self.save_plots and i == self.num_trials-1:
                    # Update Goal (in case it changed)
                    self.goal_marker.set_data([goal_state[0]], [goal_state[1]])
                    
                    # Robot Update
                    self.robot.render_plot()
                    
                    self.fig.canvas.flush_events()
                    if t % (DT*10) == 0:
                        pass
                    plt.pause(0.001)

            if i == self.num_trials - 1 and self.save_plots:
                 plt.savefig(f"trial_{self.model_name}_{self.mode}.png")

            # Prioritize Collision over Infeasibility for reporting
            if trial_collided:
                trial_infeasible = False
            
            if trial_collided: stats['collisions'] += 1
            if trial_infeasible: stats['infeasible'] += 1
            
            print(f"Trial {i}: C={trial_collided}, I={trial_infeasible}")
            
        return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default='cbf')
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_vel", type=float, default=None)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--obs", type=int, default=10)
    parser.add_argument("--save_plots", action='store_true')
    
    args = parser.parse_args()
    
    runner = SimulationRunner(args.model, args.mode, args.alpha, args.trials, args.obs, args.save_plots, args.max_vel)
    res = runner.run_trials()
    
    # Save
    tag = f"{args.model}_{args.mode}_alpha{args.alpha}"
    if args.max_vel: tag += f"_v{args.max_vel}"
    with open(f"results_{tag}.json", 'w') as f:
        json.dump(res, f)
