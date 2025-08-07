#!/usr/bin/env python3
"""
BarrierNet Dataset Generator

This script generates training datasets for BarrierNet by running simulations with
LocalTrackingController using BarrierNet position controller.

Dataset Format:
Each row contains one timestep of simulation data:
- robot_x, robot_y, robot_theta, robot_v: Robot state
- obs_x, obs_y, obs_R: Obstacle position and radius
- goal_x, goal_y: Fixed goal position
- range_goal: Distance from robot to goal
- range_obs: Distance from robot to obstacle center
- u0, u1, ...: Control outputs (flattened)

Usage:
    python generate_bn_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking import LocalTrackingController, InfeasibleError
from utils import plotting, env

# ===== CONFIGURATION PARAMETERS =====
# Robot model selection
robot_model_list = ["DynamicUnicycle2D", "Quad2D", "Quad3D"]
SELECTED_ROBOT_MODEL = robot_model_list[0]  # 0: DynamicUnicycle2D, 1: Quad2D, 2: Quad3D

# Simulation parameters
N_SIMS = 100                    # Number of simulations to run
DT = 0.05                       # Controller timestep
T_MAX = 25.0                    # Maximum simulation time

# Obstacle parameters
OBS_X_MIN, OBS_X_MAX = 1.0, 3.0    # Obstacle x position range
OBS_Y_MIN, OBS_Y_MAX = 1.0, 3.0    # Obstacle y position range  
OBS_R_MIN, OBS_R_MAX = 0.2, 0.5    # Obstacle radius range

# Robot initial conditions
VEL_MIN, VEL_MAX = 0.1, 1.0        # Velocity range
THETA_MIN, THETA_MAX = 0.0, np.pi/2 # Heading range in radians

# Fixed parameters
GOAL_POSITION = [8.0, 2.0]          # Fixed goal position
GAMMA0 = 0.1                        # Fixed CBF parameter
GAMMA1 = 0.1                        # Fixed CBF parameter

# Display options
SHOW_ANIMATION = False              # Show animation for first simulation

# Hard-coded checkpoint paths for each model
CHECKPOINT_PATHS = {
    "DynamicUnicycle2D": "position_control/BarrierNet/2D_Robot/model_bn.pth",
    "Quad2D": "position_control/BarrierNet/2D_Robot/model_bn.pth",  # Same as unicycle for now
    "Quad3D": "position_control/BarrierNet/3D_Robot/model_bn.pth"
}

def get_robot_spec(model):
    """Get minimal robot specification for the given model."""
    base_spec = {"model": model, "radius": 0.25}
    
    # Add model-specific defaults
    if model == "DynamicUnicycle2D":
        base_spec.update({
            "w_max": 0.5,
            "a_max": 0.5
        })
    elif model == "Quad2D":
        base_spec.update({
            "f_min": 3.0,
            "f_max": 10.0
        })
    elif model == "Quad3D":
        base_spec.update({
            "u_min": -10.0,
            "u_max": 10.0
        })
    
    return base_spec

def run_single_simulation(sim_id, model, obs_x, obs_y, obs_R, vel, theta, 
                         dt=0.05, T_max=25.0, show_animation=False):
    """
    Run a single simulation and return training data.
    
    Returns:
        DataFrame with training data, or None if simulation failed
    """
    try:
        # Robot specification
        robot_spec = get_robot_spec(model)
        
        # Initial robot state
        if model == "DynamicUnicycle2D":
            x_init = np.array([1.0, 2.0, theta, vel])
        elif model == "Quad2D":
            x_init = np.array([1.0, 2.0, theta, vel, 0.0, 0.0])
        elif model == "Quad3D":
            x_init = np.array([1.0, 2.0, 0.0, 0.0, 0.0, theta, vel, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Waypoints
        waypoints = np.array([GOAL_POSITION + [0.0]], dtype=np.float64)
        
        # Obstacles
        obstacles = np.array([[obs_x, obs_y, obs_R, 0.0, 0.0]])  # Static obstacle
        
        # Environment setup
        env_width, env_height = 10.0, 4.0
        plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=obstacles)
        ax, fig = plot_handler.plot_grid("BarrierNet Data Generation")
        env_handler = env.Env()
        
        # Controller setup
        controller_type = {
            'pos': 'barriernet',
            'ckpt': CHECKPOINT_PATHS[model]
        }
        
        tracking_controller = LocalTrackingController(
            x_init, robot_spec,
            controller_type=controller_type,
            dt=dt,
            show_animation=show_animation,
            save_animation=False,
            enable_rotation=False,
            raise_error=True,
            ax=ax, fig=fig,
            env=env_handler
        )
        
        # Set obstacles and waypoints
        tracking_controller.obs = obstacles
        tracking_controller.set_waypoints(waypoints)
        
        # Training data storage
        training_data = []
        
        # Simulation loop
        sim_time = 0.0
        max_steps = int(T_max / dt)
        
        for step in range(max_steps):
            try:
                ret = tracking_controller.control_step()
                
                if show_animation:
                    tracking_controller.draw_plot()
                
                # Get robot state
                robot_state = tracking_controller.robot.X.flatten()
                robot_pos = tracking_controller.robot.get_position()
                
                # Get control input
                control_input = tracking_controller.get_control_input().flatten()
                
                # Calculate distances
                range_goal = np.linalg.norm(robot_pos - np.array(GOAL_POSITION))
                range_obs = np.linalg.norm(robot_pos - np.array([obs_x, obs_y]))
                
                # Create training row
                row_data = {
                    'robot_x': robot_pos[0],
                    'robot_y': robot_pos[1],
                    'robot_theta': robot_state[2] if len(robot_state) > 2 else 0.0,
                    'robot_v': robot_state[3] if len(robot_state) > 3 else 0.0,
                    'obs_x': obs_x,
                    'obs_y': obs_y,
                    'obs_R': obs_R,
                    'goal_x': GOAL_POSITION[0],
                    'goal_y': GOAL_POSITION[1],
                    'range_goal': range_goal,
                    'range_obs': range_obs
                }
                
                # Add control outputs
                for i, u_val in enumerate(control_input):
                    row_data[f'u{i}'] = u_val
                
                training_data.append(row_data)
                
                sim_time += dt
                
                # Check if goal reached
                if ret == -1:
                    break
                    
            except InfeasibleError:
                print(f"Warning: Simulation {sim_id} failed due to infeasible QP at step {step}")
                plt.close()
                return None
        
        plt.close()
        
        if training_data:
            return pd.DataFrame(training_data)
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Simulation {sim_id} failed with error: {e}")
        return None

def main():
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print(f"Generating dataset for {SELECTED_ROBOT_MODEL}")
    print(f"Parameters:")
    print(f"  Simulations: {N_SIMS}")
    print(f"  Obstacle x: [{OBS_X_MIN}, {OBS_X_MAX}]")
    print(f"  Obstacle y: [{OBS_Y_MIN}, {OBS_Y_MAX}]")
    print(f"  Obstacle R: [{OBS_R_MIN}, {OBS_R_MAX}]")
    print(f"  Velocity: [{VEL_MIN}, {VEL_MAX}]")
    print(f"  Theta: [{THETA_MIN}, {THETA_MAX}]")
    print(f"  Goal: {GOAL_POSITION}")
    print()
    
    successful_sims = 0
    
    for sim_id in range(N_SIMS):
        # Randomize parameters
        obs_x = np.random.uniform(OBS_X_MIN, OBS_X_MAX)
        obs_y = np.random.uniform(OBS_Y_MIN, OBS_Y_MAX)
        obs_R = np.random.uniform(OBS_R_MIN, OBS_R_MAX)
        vel = np.random.uniform(VEL_MIN, VEL_MAX)
        theta = np.random.uniform(THETA_MIN, THETA_MAX)
        
        # Show animation only for first simulation if requested
        show_anim = SHOW_ANIMATION and sim_id == 0
        
        print(f"Running simulation {sim_id+1}/{N_SIMS}...", end=' ')
        
        # Run simulation
        df = run_single_simulation(
            sim_id, SELECTED_ROBOT_MODEL, obs_x, obs_y, obs_R, vel, theta,
            dt=DT, T_max=T_MAX, show_animation=show_anim
        )
        
        if df is not None:
            # Save to CSV
            filename = f"data/bn_train_sim_{SELECTED_ROBOT_MODEL}_{sim_id:04d}.csv"
            df.to_csv(filename, index=False)
            print(f"✓ Saved {len(df)} timesteps to {filename}")
            successful_sims += 1
        else:
            print("✗ Failed")
    
    print(f"\nDataset generation complete!")
    print(f"Successful simulations: {successful_sims}/{N_SIMS}")
    print(f"Data files saved in: data/")

if __name__ == "__main__":
    main() 