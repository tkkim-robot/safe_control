"""
Created on December 21st, 2025
@author: Taekyung Kim

@description:
Test script for safety shielding algorithms (Gatekeeper and MPS) in the evade_bullet_bill scenario.
A double integrator robot navigates a hallway while evading a fast-moving obstacle.
The robot uses the shielding algorithm to hide in a safe pocket when the bullet approaches.

Algorithms:
- gatekeeper: Uses nominal horizon with backward search
- mps: Model Predictive Shielding - uses single-step nominal horizon

Usage:
    uv run python examples/evade/test_evade.py [--algo ALGO] [--save]

Examples:
    # Test with gatekeeper (default)
    uv run python examples/evade/test_evade.py --algo gatekeeper
    
    # Test with MPS algorithm
    uv run python examples/evade/test_evade.py --algo mps
    
    # Save animation
    uv run python examples/evade/test_evade.py --save

@required-scripts: safe_control/shielding/gatekeeper.py, safe_control/shielding/mps.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

from safe_control.envs.evade_env import EvadeEnv
from safe_control.robots.double_integrator2D import DoubleIntegrator2D
from safe_control.shielding.gatekeeper import Gatekeeper
from safe_control.shielding.mps import MPS
from safe_control.utils.animation import AnimationSaver


# =============================================================================
# Algorithm Types
# =============================================================================

ALGO_TYPES = ['gatekeeper', 'mps']


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""
    hallway_length: float = 60.0
    hallway_width: float = 4.0
    pocket_x: float = 25.0  # Safe pocket in middle of hallway
    pocket_length: float = 10.0
    pocket_width: float = 4.0
    goal_length: float = 5.0
    bullet_speed: float = 3.0  # Slower bullet (robot v_max=1.5)
    bullet_length: float = 3.0
    bullet_start_x: float = 0.0  # Start closer to hallway for faster respawn


@dataclass
class RobotConfig:
    """Robot configuration parameters."""
    radius: float = 0.5
    a_max: float = 2.0
    v_max: float = 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'radius': self.radius,
            'a_max': self.a_max,
            'v_max': self.v_max,
            'model': 'DoubleIntegrator2D',
        }


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    dt: float = 0.05
    tf: float = 60.0
    backup_horizon_time: float = 12.0  
    event_offset: float = 0.2
    safety_margin: float = 0.5
    initial_x: float = 20.0  # Start ahead of bullet
    target_velocity: float = 1.0


@dataclass
class TestConfig:
    """Complete test configuration."""
    name: str = "Evade Bullet Bill"
    description: str = "Robot evades fast-moving obstacle by hiding in safe pocket"
    env: EnvironmentConfig = None
    robot: RobotConfig = None
    simulation: SimulationConfig = None
    save_animation: bool = False
    algo_type: str = 'gatekeeper'  # Algorithm type: 'gatekeeper' or 'mps'
    
    def __post_init__(self):
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.robot is None:
            self.robot = RobotConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()






# =============================================================================
# Controllers for Evade Scenario
# =============================================================================

class EvadeNominalController:
    """
    Nominal controller for evade scenario.
    Tracks center of hallway (y=0) while moving forward to goal.
    """
    
    def __init__(self, robot_spec: Dict):
        self.robot_spec = robot_spec
        self.v_max = robot_spec.get('v_max', 1.0)
        self.a_max = robot_spec.get('a_max', 1.0)
    
    def compute_control(self, state: np.ndarray, target=None) -> np.ndarray:
        """Compute nominal control - PD controller to track y=0 and move forward."""
        state = np.array(state).flatten()
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        
        # Target: center of hallway (y=0), moving forward at max speed
        target_y = 0.0
        target_vx = self.v_max
        target_vy = 0.0
        
        # PD control
        Kp_y = 2.0
        Kd = 2.0
        
        error_y = target_y - y
        error_vx = target_vx - vx
        error_vy = target_vy - vy
        
        ax = Kd * error_vx  # Accelerate to target speed
        ay = Kp_y * error_y + Kd * error_vy  # Track center line
        
        # Clamp
        a_mag = np.sqrt(ax**2 + ay**2)
        if a_mag > self.a_max:
            ax = ax * self.a_max / a_mag
            ay = ay * self.a_max / a_mag
        
        return np.array([[ax], [ay]])
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.compute_control(state)

class EvadeBackupController:
    """
    Backup controller for evade scenario.
    Goes to pocket center OR stays in goal zone if reached.
    """
    
    def __init__(self, robot_spec: Dict, pocket_center: np.ndarray, 
                 pocket_bounds: Dict, goal_bounds: Dict = None):
        self.robot_spec = robot_spec
        self.pocket_center = pocket_center
        self.pocket_bounds = pocket_bounds
        self.goal_bounds = goal_bounds
        self.a_max = robot_spec.get('a_max', 1.0)
    
    def compute_control(self, state: np.ndarray, target=None) -> np.ndarray:
        """
        Compute backup control.
        
        Priority:
        1. If in GOAL ZONE: stay there (brake to stop) - this is also safe!
        2. If in hallway and outside pocket x-range: first move x toward pocket center
        3. If in hallway and within pocket x-range: move up into pocket
        4. If in pocket: move to pocket center
        """
        state = np.array(state).flatten()
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        
        # Safe pocket bounds
        x_min = self.pocket_bounds['x_min']
        x_max = self.pocket_bounds['x_max']
        y_min = self.pocket_bounds['y_min']
        y_max = self.pocket_bounds['y_max']
        
        pocket_center_x = self.pocket_center[0]
        pocket_center_y = self.pocket_center[1]
        
        # PD control gains
        Kp = 2.0
        Kd = 2.0
        
        # PRIORITY 1: Check if in GOAL ZONE - stay there (also safe!)
        if self.goal_bounds is not None:
            goal_x_min = self.goal_bounds['x_min']
            goal_x_max = self.goal_bounds['x_max']
            goal_y_min = self.goal_bounds['y_min']
            goal_y_max = self.goal_bounds['y_max']
            
            in_goal = (goal_x_min <= x <= goal_x_max and 
                       goal_y_min <= y <= goal_y_max)
            
            if in_goal:
                # In goal zone - brake to stop and stay!
                ax = -Kd * vx
                ay = -Kd * vy
                
                # Clamp and return early
                a_mag = np.sqrt(ax**2 + ay**2)
                if a_mag > self.a_max:
                    ax = ax * self.a_max / a_mag
                    ay = ay * self.a_max / a_mag
                return np.array([[ax], [ay]])
        
        # Margin for "in pocket" check
        inside_margin = 0.5
        
        # Check if robot is IN the pocket (y > y_min and x in range)
        in_pocket = (x_min - inside_margin <= x <= x_max + inside_margin and 
                     y >= y_min - inside_margin)
        
        # Check if close enough to pocket center
        dist_to_center = np.sqrt((x - pocket_center_x)**2 + (y - pocket_center_y)**2)
        
        if dist_to_center < 1.0:
            # Very close to pocket center - brake to stop
            ax = -Kd * vx
            ay = -Kd * vy
        
        elif in_pocket:
            # Robot is in the pocket - move toward pocket center
            error_x = pocket_center_x - x
            error_y = pocket_center_y - y
            
            ax = Kp * error_x - Kd * vx
            ay = Kp * error_y - Kd * vy
        
        elif x_min <= x <= x_max:
            # Robot is in hallway but within pocket x-range - safe to go up
            target_x = pocket_center_x
            target_y = pocket_center_y
            
            error_x = target_x - x
            error_y = target_y - y
            
            ax = Kp * error_x - Kd * vx
            ay = Kp * error_y - Kd * vy
        
        else:
            # Robot is in hallway OUTSIDE pocket x-range
            # First move x toward pocket center WHILE staying in hallway (y=0)
            target_x = pocket_center_x
            target_y = 0.0
            
            error_x = target_x - x
            error_y = target_y - y
            
            ax = Kp * error_x - Kd * vx
            ay = Kp * error_y - Kd * vy
        
        # Clamp accelerations
        a_mag = np.sqrt(ax**2 + ay**2)
        if a_mag > self.a_max:
            ax = ax * self.a_max / a_mag
            ay = ay * self.a_max / a_mag
        
        return np.array([[ax], [ay]])




# =============================================================================
# Robot Visualization
# =============================================================================

class RobotVisualizer:
    """Simple robot visualizer as a circle."""
    
    def __init__(self, ax, radius=0.5, color='orange'):
        self.ax = ax
        self.radius = radius
        self.body = Circle((0, 0), radius, facecolor=color, 
                          edgecolor='black', linewidth=2, zorder=15)
        ax.add_patch(self.body)
        
        # Direction indicator
        self.direction, = ax.plot([], [], 'k-', linewidth=2, zorder=16)
        
        # Trail
        self.trail_x = []
        self.trail_y = []
        self.trail, = ax.plot([], [], 'orange', linewidth=1, alpha=0.5, zorder=5)
    
    def update(self, state):
        """Update robot visualization."""
        x, y, vx, vy = state.flatten()[:4]
        
        self.body.center = (x, y)
        
        # Direction arrow
        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag > 0.1:
            dir_x = x + vx / v_mag * self.radius * 1.5
            dir_y = y + vy / v_mag * self.radius * 1.5
            self.direction.set_data([x, dir_x], [y, dir_y])
        
        # Update trail
        self.trail_x.append(x)
        self.trail_y.append(y)
        if len(self.trail_x) > 200:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
        self.trail.set_data(self.trail_x, self.trail_y)


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(config: TestConfig, animation_saver: Optional['AnimationSaver'] = None) -> Dict[str, Any]:
    """Run the evade simulation."""
    print("\n" + "=" * 70)
    print(f"  TEST: {config.name}")
    print(f"  {config.description}")
    print("=" * 70)
    
    # Setup environment
    env = EvadeEnv(
        hallway_length=config.env.hallway_length,
        hallway_width=config.env.hallway_width,
        pocket_x=config.env.pocket_x,
        pocket_length=config.env.pocket_length,
        pocket_width=config.env.pocket_width,
        goal_length=config.env.goal_length,
        bullet_speed=config.env.bullet_speed,
        bullet_length=config.env.bullet_length,
        bullet_start_x=config.env.bullet_start_x
    )
    
    plt.ion()
    ax, fig = env.setup_plot()
    algo_name = config.algo_type.upper()
    fig.canvas.manager.set_window_title(f'Evade Bullet Bill - {algo_name}')
    
    # Initial robot state [x, y, vx, vy]
    initial_state = np.array([
        config.simulation.initial_x,
        0.0,  # Center of hallway
        0.0,  # No initial velocity
        0.0
    ])
    
    robot_spec = config.robot.to_dict()
    
    # Get pocket info
    pocket_center = env.get_pocket_center()
    pocket_bounds = env.get_pocket_bounds()
    
    # Create goal bounds
    goal_bounds = {
        'x_min': env.goal_x_min,
        'x_max': env.goal_x_max,
        'y_min': -env.half_width,  # Hallway y-range
        'y_max': env.half_width
    }
    
    print(f"  Pocket center: ({pocket_center[0]:.1f}, {pocket_center[1]:.1f})")
    print(f"  Pocket bounds: x=[{pocket_bounds['x_min']:.1f}, {pocket_bounds['x_max']:.1f}], "
          f"y=[{pocket_bounds['y_min']:.1f}, {pocket_bounds['y_max']:.1f}]")
    print(f"  Goal bounds: x=[{goal_bounds['x_min']:.1f}, {goal_bounds['x_max']:.1f}], "
          f"y=[{goal_bounds['y_min']:.1f}, {goal_bounds['y_max']:.1f}]")
    
    # Create controllers
    nominal_controller = EvadeNominalController(robot_spec)
    backup_controller = EvadeBackupController(robot_spec, pocket_center, pocket_bounds, goal_bounds)
    
    # Dynamics model for simulation
    dynamics = DoubleIntegrator2D(config.simulation.dt, robot_spec)
    
    # Setup shielding algorithm
    backup_horizon_time = config.simulation.backup_horizon_time
    
    if config.algo_type == 'mps':
        print(f"  Algorithm: MPS (one-step nominal horizon)")
        shielding = MPS(
            robot=dynamics,
            robot_spec=robot_spec,
            dt=config.simulation.dt,
            backup_horizon=backup_horizon_time,
            event_offset=config.simulation.event_offset,
            ax=ax
        )
    else:
        print(f"  Algorithm: GATEKEEPER (backward search)")
        shielding = Gatekeeper(
            robot=dynamics,
            robot_spec=robot_spec,
            dt=config.simulation.dt,
            backup_horizon=backup_horizon_time,
            nominal_horizon=backup_horizon_time, # Use full backup horizon as nominal for Gatekeeper
            event_offset=config.simulation.event_offset,
            ax=ax
        )
    
    # Configure shielding
    shielding.set_nominal_controller(nominal_controller)
    shielding.set_backup_controller(backup_controller)
    shielding.set_environment(env) 
    shielding.set_moving_obstacles(env.get_bullet_state) # Callable that returns obstacle list/dict
    
    # Setup visualization
    robot_viz = RobotVisualizer(ax, config.robot.radius, 'orange')
    
    # Simulation loop
    state = initial_state.reshape(-1, 1)
    num_steps = int(config.simulation.tf / config.simulation.dt)
    
    # For stats
    collision_occurred = False
    goal_reached = False
    nominal_steps = 0
    backup_steps = 0
    
    print(f"\nRunning simulation for {config.simulation.tf}s...")
    print(f"  Bullet speed: {config.env.bullet_speed} m/s")
    print(f"  Robot max speed: {config.robot.v_max} m/s")
    
    for step in range(num_steps):
        pos = state[:2, 0]
        
        # Shielding control
        # Note: shielding.solve_control_problem calls get_bullet_state internally via moving_obstacles callback
        control = shielding.solve_control_problem(state)
        
        # Track mode
        if shielding.is_using_backup():
            backup_steps += 1
        else:
            nominal_steps += 1
        
        # Apply control (double integrator dynamics)
        state = dynamics.step(state, control)
        
        # Clamp velocity (simple model constraint)
        vx, vy = state[2, 0], state[3, 0]
        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag > config.robot.v_max:
            state[2, 0] = vx * config.robot.v_max / v_mag
            state[3, 0] = vy * config.robot.v_max / v_mag
        
        # Step bullet
        env.step_bullet(config.simulation.dt)
        
        # Update visualization
        robot_viz.update(state)
        env.update_plot_frame(ax, pos, window_size=(35, 18))
        
        plt.pause(0.001)
        fig.canvas.flush_events()
        
        # Save animation frame
        if animation_saver is not None:
             animation_saver.save_frame(fig)
             
        # Check collision
        collision, _ = env.check_obstacle_collision(pos, config.robot.radius)
        if collision:
            collision_occurred = True
            env.show_collision(pos)
            print(f"\n*** COLLISION at step {step} ***")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            if animation_saver is not None:
                animation_saver.save_frame(fig, force=True)
            plt.pause(2.0)
            break
            
        # Check goal
        if env.check_goal_reached(pos):
            goal_reached = True
            env.show_goal_reached(pos)
            print(f"\n*** GOAL REACHED at step {step} ***")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            if animation_saver is not None:
                animation_saver.save_frame(fig, force=True)
            plt.pause(2.0)
            break

        # Status output
        if step % 20 == 0:
            status = shielding.get_status()
            mode = "BACKUP" if status['using_backup'] else "NOMINAL"
            in_pocket = env.is_in_safe_pocket(pos)
            pocket_str = " [IN POCKET]" if in_pocket else ""
            print(f"Step {step:4d}: x={pos[0]:6.2f}, y={pos[1]:6.2f}, mode={mode}{pocket_str}")
    
    # Export video if requested
    if animation_saver is not None:
        animation_saver.export_video()
    
    plt.ioff()
    plt.close()
    
    # Results
    total_steps = nominal_steps + backup_steps
    results = {
        'collision': collision_occurred,
        'goal_reached': goal_reached,
        'total_steps': total_steps,
        'nominal_steps': nominal_steps,
        'backup_steps': backup_steps,
        'nominal_ratio': nominal_steps / max(total_steps, 1),
        'backup_ratio': backup_steps / max(total_steps, 1),
    }
    
    print("\n" + "-" * 50)
    print("Results:")
    print(f"  Collision: {'YES' if collision_occurred else 'NO'}")
    print(f"  Goal Reached: {'YES' if goal_reached else 'NO'}")
    print(f"  Total steps: {total_steps}")
    print(f"  Nominal: {nominal_steps} ({100*results['nominal_ratio']:.1f}%)")
    print(f"  Backup: {backup_steps} ({100*results['backup_ratio']:.1f}%)")
    
    if goal_reached and not collision_occurred:
        print("\n  ✓ TEST PASSED (reached goal without collision)")
        results['passed'] = True
    else:
        print("\n  ✗ TEST FAILED")
        results['passed'] = False
    
    print("-" * 50)
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Evade Bullet Bill scenario')
    parser.add_argument('--algo', type=str, default='gatekeeper',
                        choices=ALGO_TYPES,
                        help='Shielding algorithm: gatekeeper (default) or mps')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as video')
    
    args = parser.parse_args()
    
    config = TestConfig()
    config.algo_type = args.algo
    config.save_animation = args.save
    config.name = f"Evade Bullet Bill ({args.algo})"
    
    # Setup animation saver if enabled
    animation_saver = None
    if config.save_animation:
        safe_name = config.name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        output_dir = f"output/animations/{safe_name}"
        animation_saver = AnimationSaver(output_dir=output_dir, save_per_frame=1, fps=30)
        print(f"\n  Animation saving enabled -> {output_dir}/")
    
    results = run_simulation(config, animation_saver)
    
    return results


if __name__ == "__main__":
    main()
