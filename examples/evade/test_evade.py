"""
Created on December 21st, 2025
@author: Taekyung Kim

@description:
Test script for Gatekeeper safety shielding in the evade_bullet_bill scenario.
A double integrator robot navigates a hallway while evading a fast-moving obstacle.
The robot uses gatekeeper to hide in a safe pocket when the bullet approaches.

Usage:
    uv run python examples/evade/test_evade.py [--save]

@required-scripts: safe_control/shielding/gatekeeper.py, safe_control/envs/evade_env.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from safe_control.envs.evade_env import EvadeEnv
from safe_control.robots.double_integrator2D import DoubleIntegrator2D
from safe_control.position_control.backup_controller import EvadeBackupController


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
    bullet_speed: float = 2.5  # Slower bullet (robot v_max=1.5)
    bullet_length: float = 3.0
    bullet_start_x: float = -25.0  # Start behind the robot


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
    backup_horizon_time: float = 4.0
    event_offset: float = 0.2
    safety_margin: float = 0.5
    initial_x: float = 3.0
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
    
    def __post_init__(self):
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.robot is None:
            self.robot = RobotConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()


# =============================================================================
# Simple Gatekeeper for Double Integrator
# =============================================================================

class EvadeGatekeeper:
    """
    Simplified gatekeeper for the evade scenario with double integrator.
    
    Validates nominal trajectories against safety constraints including
    forward-simulated moving obstacles.
    """
    
    def __init__(self, robot_spec, dt, backup_horizon, event_offset, 
                 backup_controller, env, ax=None):
        """
        Initialize the EvadeGatekeeper.
        
        Args:
            robot_spec: Robot specification dictionary
            dt: Simulation timestep
            backup_horizon: Duration (seconds) for backup trajectory
            event_offset: Time between re-evaluation events
            backup_controller: EvadeBackupController instance
            env: EvadeEnv instance
            ax: Matplotlib axis for visualization
        """
        self.robot_spec = robot_spec
        self.dt = dt
        self.backup_horizon = backup_horizon
        self.event_offset = event_offset
        self.backup_controller = backup_controller
        self.env = env
        self.ax = ax
        
        self.a_max = robot_spec.get('a_max', 1.0)
        self.v_max = robot_spec.get('v_max', 1.0)
        
        # State tracking
        self.committed_x_traj = None
        self.committed_u_traj = None
        self.current_time_idx = 0
        self.next_event_time = 0.0
        self.committed_horizon = 0.0
        self.using_backup = True
        
        # Visualization
        self.committed_line = None
        self.backup_line = None
        if ax is not None:
            self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup visualization handles (matching drift_car gatekeeper style)."""
        if self.ax is None:
            return
        
        # Committed nominal portion (lime = following nominal controller)
        self.committed_line, = self.ax.plot(
            [], [], '-', color='lime', linewidth=3, alpha=0.9,
            label='Committed nominal', zorder=8
        )
        
        # Committed backup portion (dodgerblue = evade to pocket)
        self.backup_line, = self.ax.plot(
            [], [], '-', color='dodgerblue', linewidth=3, alpha=0.9,
            label='Committed backup', zorder=8
        )
        
        # Switching point marker (magenta)
        self.switching_marker, = self.ax.plot(
            [], [], 'mo', markersize=12, markerfacecolor='magenta',
            markeredgecolor='white', markeredgewidth=2,
            label='Switching point', zorder=9
        )
    
    def _forward_simulate(self, initial_state, controller, horizon_steps, target=None):
        """Forward simulate trajectory with given controller."""
        state = np.array(initial_state).flatten().reshape(-1, 1)
        x_traj = np.zeros((horizon_steps + 1, 4))
        u_traj = np.zeros((horizon_steps, 2))
        x_traj[0] = state.flatten()
        
        for i in range(horizon_steps):
            if controller == 'nominal':
                U = self._nominal_control(state)
            else:
                U = self.backup_controller.compute_control(state)
            
            u_traj[i] = U.flatten()
            
            # Double integrator dynamics
            ax, ay = U[0, 0], U[1, 0]
            vx_new = state[2, 0] + ax * self.dt
            vy_new = state[3, 0] + ay * self.dt
            v_mag = np.sqrt(vx_new**2 + vy_new**2)
            if v_mag > self.v_max:
                vx_new = vx_new * self.v_max / v_mag
                vy_new = vy_new * self.v_max / v_mag
            
            state[0, 0] += state[2, 0] * self.dt
            state[1, 0] += state[3, 0] * self.dt
            state[2, 0] = vx_new
            state[3, 0] = vy_new
            
            x_traj[i + 1] = state.flatten()
        
        return x_traj, u_traj
    
    def _nominal_control(self, state):
        """
        Nominal PD controller: track center of hallway heading to goal.
        """
        state = np.array(state).flatten()
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        
        # Target: center of hallway (y=0), moving forward (x -> hallway_length)
        target_y = 0.0
        target_vx = self.v_max  # Move forward at max speed
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
    
    def _is_trajectory_valid(self, x_traj, obstacle_state, safety_margin=0.5):
        """Check if trajectory is collision-free considering moving obstacle."""
        robot_radius = self.robot_spec.get('radius', 0.5) + safety_margin
        
        # Forward simulate obstacle
        obs = obstacle_state.copy()
        
        for i, state in enumerate(x_traj):
            pos = state[:2]
            
            # Check boundary collision
            if self.env.check_collision(pos, robot_radius):
                return False
            
            # Check moving obstacle collision (at simulated time)
            # Obstacle position at timestep i
            obs_x = obstacle_state['x'] + obstacle_state.get('vx', 0) * i * self.dt
            obs_y = obstacle_state['y'] + obstacle_state.get('vy', 0) * i * self.dt
            
            # Check if obstacle respawned (wrapped around)
            if obs_x > self.env.hallway_length + obstacle_state['length']:
                obs_x = self.env.bullet_start_x + (obs_x - self.env.hallway_length - obstacle_state['length'])
            
            # Rectangle-circle collision
            obs_length = obstacle_state['length']
            obs_width = obstacle_state['width']
            obs_x_min = obs_x - obs_length / 2
            obs_x_max = obs_x + obs_length / 2 + obs_length / 3  # Include nose
            obs_y_min = obs_y - obs_width / 2
            obs_y_max = obs_y + obs_width / 2
            
            closest_x = np.clip(pos[0], obs_x_min, obs_x_max)
            closest_y = np.clip(pos[1], obs_y_min, obs_y_max)
            dist = np.sqrt((pos[0] - closest_x)**2 + (pos[1] - closest_y)**2)
            
            if dist < robot_radius:
                return False
        
        return True
    
    def solve_control_problem(self, robot_state, obstacle_state):
        """
        Main control loop - validates nominal and switches to backup if needed.
        
        Args:
            robot_state: Current robot state [x, y, vx, vy]
            obstacle_state: Current bullet bill state dict
            
        Returns:
            control: Control input [ax, ay]
        """
        robot_state = np.array(robot_state).flatten()
        backup_steps = int(self.backup_horizon / self.dt)
        
        # Initialize committed trajectory if needed
        if self.committed_x_traj is None:
            backup_x, backup_u = self._forward_simulate(
                robot_state, 'backup', backup_steps
            )
            self.committed_x_traj = backup_x
            self.committed_u_traj = backup_u
            self.committed_horizon = 0.0
            self.current_time_idx = 0
            self.next_event_time = 0.0
            self.using_backup = True
        
        # Check if we should re-evaluate (only at event times)
        if self.current_time_idx >= self.next_event_time / self.dt:
            # Try nominal+backup trajectory
            nominal_steps = backup_steps // 2  # Shorter nominal horizon
            nominal_x, nominal_u = self._forward_simulate(
                robot_state, 'nominal', nominal_steps
            )
            
            # Append backup from end of nominal
            if len(nominal_x) > 0:
                backup_x, backup_u = self._forward_simulate(
                    nominal_x[-1], 'backup', backup_steps
                )
                candidate_x = np.vstack([nominal_x, backup_x[1:]])
                candidate_u = np.vstack([nominal_u, backup_u])
            else:
                candidate_x = nominal_x
                candidate_u = nominal_u
            
            # Validate candidate
            if self._is_trajectory_valid(candidate_x, obstacle_state):
                # Valid candidate found - commit to it
                self.committed_x_traj = candidate_x
                self.committed_u_traj = candidate_u
                self.committed_horizon = nominal_steps * self.dt
                self.current_time_idx = 0
                self.next_event_time = self.event_offset
                self.using_backup = False
            else:
                # Candidate invalid - KEEP following previously committed trajectory
                # Do NOT replace with a new backup trajectory
                # Only reschedule the next event
                self.next_event_time = self.current_time_idx * self.dt + self.event_offset
                
                # Update using_backup flag based on where we are in committed trajectory
                remaining_nominal_steps = int(self.committed_horizon / self.dt) - self.current_time_idx
                self.using_backup = remaining_nominal_steps <= 0
        
        # Get control from committed trajectory
        if self.current_time_idx < len(self.committed_u_traj):
            control = self.committed_u_traj[self.current_time_idx].reshape(-1, 1)
        else:
            # Fallback to backup controller
            control = self.backup_controller.compute_control(robot_state.reshape(-1, 1))
            self.using_backup = True
        
        self.current_time_idx += 1
        
        # Update visualization
        self._update_visualization()
        
        return control
    
    def _update_visualization(self):
        """Update trajectory visualization (matching drift_car gatekeeper style)."""
        if self.ax is None or self.committed_x_traj is None:
            return
        
        nominal_end = int(self.committed_horizon / self.dt) + 1
        
        # Update committed nominal portion (lime)
        if self.committed_line is not None:
            if nominal_end > 0:
                self.committed_line.set_data(
                    self.committed_x_traj[:nominal_end, 0],
                    self.committed_x_traj[:nominal_end, 1]
                )
            else:
                self.committed_line.set_data([], [])
        
        # Update committed backup portion (dodgerblue)
        if self.backup_line is not None:
            backup_start = max(0, nominal_end - 1)  # Overlap at switching point
            if backup_start < len(self.committed_x_traj):
                self.backup_line.set_data(
                    self.committed_x_traj[backup_start:, 0],
                    self.committed_x_traj[backup_start:, 1]
                )
            else:
                self.backup_line.set_data([], [])
        
        # Update switching point marker (magenta)
        if hasattr(self, 'switching_marker') and self.switching_marker is not None:
            switch_idx = int(self.committed_horizon / self.dt)
            if 0 <= switch_idx < len(self.committed_x_traj):
                switch_state = self.committed_x_traj[switch_idx]
                self.switching_marker.set_data([switch_state[0]], [switch_state[1]])
            else:
                self.switching_marker.set_data([], [])
    
    def is_using_backup(self):
        """Check if currently executing backup trajectory."""
        return self.using_backup


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

def run_simulation(config: TestConfig) -> Dict[str, Any]:
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
    fig.canvas.manager.set_window_title('Evade Bullet Bill - Gatekeeper')
    
    # Initial robot state [x, y, vx, vy]
    initial_state = np.array([
        config.simulation.initial_x,
        0.0,  # Center of hallway
        0.0,  # No initial velocity
        0.0
    ])
    
    robot_spec = config.robot.to_dict()
    
    # Setup backup controller
    pocket_center = env.get_pocket_center()
    pocket_bounds = env.get_pocket_bounds()
    backup_controller = EvadeBackupController(
        robot_spec, config.simulation.dt, pocket_center, pocket_bounds
    )
    
    # Setup gatekeeper
    gatekeeper = EvadeGatekeeper(
        robot_spec=robot_spec,
        dt=config.simulation.dt,
        backup_horizon=config.simulation.backup_horizon_time,
        event_offset=config.simulation.event_offset,
        backup_controller=backup_controller,
        env=env,
        ax=ax
    )
    
    # Setup visualization
    robot_viz = RobotVisualizer(ax, config.robot.radius, 'orange')
    
    # Dynamics model
    dynamics = DoubleIntegrator2D(config.simulation.dt, robot_spec)
    
    # Simulation loop
    state = initial_state.reshape(-1, 1)
    num_steps = int(config.simulation.tf / config.simulation.dt)
    
    collision_occurred = False
    goal_reached = False
    nominal_steps = 0
    backup_steps = 0
    
    print(f"\nRunning simulation for {config.simulation.tf}s...")
    print(f"  Bullet speed: {config.env.bullet_speed} m/s")
    print(f"  Robot max speed: {config.robot.v_max} m/s")
    print(f"  Safe pocket at x=[{pocket_bounds['x_min']:.1f}, {pocket_bounds['x_max']:.1f}]")
    
    for step in range(num_steps):
        pos = state[:2, 0]
        
        # Get bullet state
        bullet_state = env.get_bullet_state()
        
        # Gatekeeper control
        control = gatekeeper.solve_control_problem(state.flatten(), bullet_state)
        
        # Track mode
        if gatekeeper.is_using_backup():
            backup_steps += 1
        else:
            nominal_steps += 1
        
        # Apply control (double integrator dynamics)
        state = dynamics.step(state, control)
        
        # Clamp velocity
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
        
        # Check collision with bullet
        collision, _ = env.check_obstacle_collision(pos, config.robot.radius)
        if collision:
            collision_occurred = True
            env.show_collision(pos)
            print(f"\n*** COLLISION at step {step} ***")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            plt.pause(2.0)
            break
        
        # Check goal
        if env.check_goal_reached(pos):
            goal_reached = True
            env.show_goal_reached(pos)
            print(f"\n*** GOAL REACHED at step {step} ***")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            plt.pause(2.0)
            break
        
        # Status output
        if step % 100 == 0:
            mode = "BACKUP" if gatekeeper.is_using_backup() else "NOMINAL"
            in_pocket = env.is_in_safe_pocket(pos)
            pocket_str = " [IN POCKET]" if in_pocket else ""
            print(f"Step {step:4d}: x={pos[0]:6.2f}, y={pos[1]:6.2f}, mode={mode}{pocket_str}")
    
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
    
    plt.ioff()
    plt.show()
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Evade Bullet Bill scenario')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as video')
    
    args = parser.parse_args()
    
    config = TestConfig()
    config.save_animation = args.save
    
    results = run_simulation(config)
    
    return results


if __name__ == "__main__":
    main()
