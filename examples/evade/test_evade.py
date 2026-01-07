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

@required-scripts: safe_control/envs/evade_env.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from safe_control.envs.evade_env import EvadeEnv
from safe_control.robots.double_integrator2D import DoubleIntegrator2D
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
    algo_type: str = 'gatekeeper'  # Algorithm type: 'gatekeeper' or 'mps'
    
    def __post_init__(self):
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.robot is None:
            self.robot = RobotConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()

@dataclass
class ShieldingState:
    """State for shielding algorithm (function-based approach)."""
    committed_x_traj: Optional[np.ndarray] = None
    committed_u_traj: Optional[np.ndarray] = None
    current_time_idx: int = 0
    next_event_time: float = 0.0
    committed_horizon: float = 0.0
    using_backup: bool = True
    
    # Visualization handles
    committed_line: Any = None
    backup_line: Any = None
    switching_marker: Any = None


# =============================================================================
# Shielding Functions
# =============================================================================

def setup_visualization(ax: plt.Axes, state: ShieldingState):
    """Setup visualization handles for trajectory display."""
    if ax is None:
        return
    
    # Committed nominal portion (lime = following nominal controller)
    # Use high zorder (20+) to draw above robot which is at zorder=15
    state.committed_line, = ax.plot(
        [], [], '-', color='lime', linewidth=3, alpha=0.9,
        label='Committed nominal', zorder=20
    )
    
    # Committed backup portion (dodgerblue = evade to pocket)
    state.backup_line, = ax.plot(
        [], [], '-', color='dodgerblue', linewidth=3, alpha=0.9,
        label='Committed backup', zorder=20
    )
    
    # Switching point marker (magenta)
    state.switching_marker, = ax.plot(
        [], [], 'mo', markersize=12, markerfacecolor='magenta',
        markeredgecolor='white', markeredgewidth=2,
        label='Switching point', zorder=25
    )


def update_visualization(ax: plt.Axes, state: ShieldingState, dt: float):
    """Update trajectory visualization."""
    if ax is None or state.committed_x_traj is None:
        return
    
    nominal_end = int(state.committed_horizon / dt) + 1
    
    # Update committed nominal portion (lime)
    if state.committed_line is not None:
        if nominal_end > 0:
            state.committed_line.set_data(
                state.committed_x_traj[:nominal_end, 0],
                state.committed_x_traj[:nominal_end, 1]
            )
        else:
            state.committed_line.set_data([], [])
    
    # Update committed backup portion (dodgerblue)
    if state.backup_line is not None:
        backup_start = max(0, nominal_end - 1)  # Overlap at switching point
        if backup_start < len(state.committed_x_traj):
            state.backup_line.set_data(
                state.committed_x_traj[backup_start:, 0],
                state.committed_x_traj[backup_start:, 1]
            )
        else:
            state.backup_line.set_data([], [])
    
    # Update switching point marker (magenta)
    if state.switching_marker is not None:
        switch_idx = int(state.committed_horizon / dt)
        if 0 <= switch_idx < len(state.committed_x_traj):
            switch_state = state.committed_x_traj[switch_idx]
            state.switching_marker.set_data([switch_state[0]], [switch_state[1]])
        else:
            state.switching_marker.set_data([], [])


def forward_simulate(initial_state: np.ndarray, controller_type: str, horizon_steps: int,
                     robot_spec: Dict, dt: float, pocket_center: np.ndarray, 
                     pocket_bounds: Dict) -> tuple:
    """Forward simulate trajectory with given controller."""
    state = np.array(initial_state).flatten().reshape(-1, 1)
    x_traj = np.zeros((horizon_steps + 1, 4))
    u_traj = np.zeros((horizon_steps, 2))
    x_traj[0] = state.flatten()
    
    a_max = robot_spec.get('a_max', 1.0)
    v_max = robot_spec.get('v_max', 1.0)
    
    for i in range(horizon_steps):
        if controller_type == 'nominal':
            U = compute_nominal_control(state, robot_spec)
        else:
            U = compute_backup_control(state, robot_spec, pocket_center, pocket_bounds)
        
        u_traj[i] = U.flatten()
        
        # Double integrator dynamics
        ax, ay = U[0, 0], U[1, 0]
        vx_new = state[2, 0] + ax * dt
        vy_new = state[3, 0] + ay * dt
        v_mag = np.sqrt(vx_new**2 + vy_new**2)
        if v_mag > v_max:
            vx_new = vx_new * v_max / v_mag
            vy_new = vy_new * v_max / v_mag
        
        state[0, 0] += state[2, 0] * dt
        state[1, 0] += state[3, 0] * dt
        state[2, 0] = vx_new
        state[3, 0] = vy_new
        
        x_traj[i + 1] = state.flatten()
    
    return x_traj, u_traj


def compute_nominal_control(state: np.ndarray, robot_spec: Dict) -> np.ndarray:
    """Nominal PD controller: track center of hallway heading to goal."""
    state = np.array(state).flatten()
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    
    v_max = robot_spec.get('v_max', 1.0)
    a_max = robot_spec.get('a_max', 1.0)
    
    # Target: center of hallway (y=0), moving forward (x -> hallway_length)
    target_y = 0.0
    target_vx = v_max  # Move forward at max speed
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
    if a_mag > a_max:
        ax = ax * a_max / a_mag
        ay = ay * a_max / a_mag
    
    return np.array([[ax], [ay]])


def compute_backup_control(state: np.ndarray, robot_spec: Dict, 
                           pocket_center: np.ndarray, pocket_bounds: Dict) -> np.ndarray:
    """
    Backup controller: go to pocket center (DEEP inside the pocket).
    
    IMPORTANT: Avoid cutting corners! The trajectory must:
    1. If in hallway and outside pocket x-range: first move x toward pocket center
    2. If in hallway and within pocket x-range: move up into pocket
    3. If in pocket: move to pocket center
    """
    state = np.array(state).flatten()
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    
    a_max = robot_spec.get('a_max', 1.0)
    
    # Safe pocket bounds
    x_min = pocket_bounds['x_min']  # 25
    x_max = pocket_bounds['x_max']  # 35
    y_min = pocket_bounds['y_min']  # 2 (top of hallway, bottom of pocket)
    y_max = pocket_bounds['y_max']  # 6
    
    pocket_center_x = pocket_center[0]  # 30
    pocket_center_y = pocket_center[1]  # 4
    
    # PD control gains
    Kp = 2.0
    Kd = 2.0
    
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
        # Move toward pocket entrance first, then center
        target_x = pocket_center_x
        target_y = pocket_center_y
        
        error_x = target_x - x
        error_y = target_y - y
        
        ax = Kp * error_x - Kd * vx
        ay = Kp * error_y - Kd * vy
    
    else:
        # Robot is in hallway OUTSIDE pocket x-range
        # CRITICAL: First move x toward pocket center WHILE staying in hallway (y=0)
        # This avoids cutting through the corner at (35, 2) or (25, 2)
        target_x = pocket_center_x  # Move x toward center of pocket
        target_y = 0.0  # Stay in hallway center
        
        error_x = target_x - x
        error_y = target_y - y
        
        # Strong x control to get back to pocket range quickly
        ax = Kp * error_x - Kd * vx
        ay = Kp * error_y - Kd * vy
    
    # Clamp accelerations
    a_mag = np.sqrt(ax**2 + ay**2)
    if a_mag > a_max:
        ax = ax * a_max / a_mag
        ay = ay * a_max / a_mag
    
    return np.array([[ax], [ay]])


def is_trajectory_valid(x_traj: np.ndarray, obstacle_state: Dict, 
                        env: EvadeEnv, robot_radius: float, 
                        dt: float, safety_margin: float = 0.5,
                        debug: bool = False) -> bool:
    """Check if trajectory is collision-free considering moving obstacle."""
    radius = robot_radius + safety_margin
    
    for i, state in enumerate(x_traj):
        pos = state[:2]
        
        # Check boundary collision
        if env.check_collision(pos, radius):
            if debug:
                print(f"    INVALID: boundary collision at step {i}, pos=({pos[0]:.2f}, {pos[1]:.2f})")
            return False
        
        # Check moving obstacle collision (at simulated time)
        # Obstacle position at timestep i
        obs_x = obstacle_state['x'] + obstacle_state.get('vx', 0) * i * dt
        obs_y = obstacle_state['y'] + obstacle_state.get('vy', 0) * i * dt
        
        # Handle multiple respawns (bullet can pass through multiple times)
        hallway_length = env.hallway_length
        bullet_length = obstacle_state['length']
        respawn_threshold = hallway_length + bullet_length
        bullet_start = env.bullet_start_x
        cycle_length = respawn_threshold - bullet_start  # Total distance per cycle
        
        # Wrap bullet position using modulo for proper multi-cycle handling
        if obs_x > respawn_threshold:
            cycles = (obs_x - bullet_start) / cycle_length
            obs_x = bullet_start + (obs_x - bullet_start) % cycle_length
        
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
        
        if dist < radius:
            if debug:
                print(f"    INVALID: bullet collision at step {i}, "
                      f"pos=({pos[0]:.2f}, {pos[1]:.2f}), "
                      f"bullet_x={obs_x:.2f}, dist={dist:.2f} < radius={radius:.2f}")
            return False
    
    return True


def solve_gatekeeper(robot_state: np.ndarray, obstacle_state: Dict,
                     state: ShieldingState, config: TestConfig, env: EvadeEnv,
                     pocket_center: np.ndarray, pocket_bounds: Dict, ax: plt.Axes) -> np.ndarray:
    """
    Gatekeeper algorithm with backward search for maximum valid nominal horizon.
    """
    robot_state = np.array(robot_state).flatten()
    robot_spec = config.robot.to_dict()
    dt = config.simulation.dt
    backup_steps = int(config.simulation.backup_horizon_time / dt)
    
    # Initialize committed trajectory if needed
    if state.committed_x_traj is None:
        backup_x, backup_u = forward_simulate(
            robot_state, 'backup', backup_steps, robot_spec, dt, pocket_center, pocket_bounds
        )
        state.committed_x_traj = backup_x
        state.committed_u_traj = backup_u
        state.committed_horizon = 0.0
        state.current_time_idx = 0
        state.next_event_time = 0.0
        state.using_backup = True
    
    # Check if we should re-evaluate (at event times OR in fallback mode)
    should_evaluate = (state.current_time_idx >= state.next_event_time / dt or
                      state.current_time_idx >= len(state.committed_u_traj))
    
    if should_evaluate:
        # Try nominal+backup trajectory with backward search
        found_valid = False
        max_nominal_steps = backup_steps // 2  # Start with half backup horizon
        
        for nominal_steps in range(max_nominal_steps, -1, -1):
            nominal_x, nominal_u = forward_simulate(
                robot_state, 'nominal', nominal_steps, robot_spec, dt, pocket_center, pocket_bounds
            )
            
            # Append backup from end of nominal
            if len(nominal_x) > 0:
                backup_x, backup_u = forward_simulate(
                    nominal_x[-1], 'backup', backup_steps, robot_spec, dt, pocket_center, pocket_bounds
                )
                candidate_x = np.vstack([nominal_x, backup_x[1:]])
                candidate_u = np.vstack([nominal_u, backup_u])
            else:
                candidate_x = nominal_x
                candidate_u = nominal_u
            
            # Validate candidate
            if is_trajectory_valid(candidate_x, obstacle_state, env, config.robot.radius, dt):
                # Valid candidate found - commit to it
                state.committed_x_traj = candidate_x
                state.committed_u_traj = candidate_u
                state.committed_horizon = nominal_steps * dt
                state.current_time_idx = 0
                state.next_event_time = config.simulation.event_offset
                state.using_backup = (nominal_steps == 0)
                found_valid = True
                break
        
        if not found_valid:
            # No valid trajectory found - commit pure backup from current position
            backup_x, backup_u = forward_simulate(
                robot_state, 'backup', backup_steps, robot_spec, dt, pocket_center, pocket_bounds
            )
            state.committed_x_traj = backup_x
            state.committed_u_traj = backup_u
            state.committed_horizon = 0.0
            state.current_time_idx = 0
            state.next_event_time = config.simulation.event_offset
            state.using_backup = True
    
    # Get control from committed trajectory
    if state.current_time_idx < len(state.committed_u_traj):
        control = state.committed_u_traj[state.current_time_idx].reshape(-1, 1)
    else:
        # Fallback to backup controller (should not happen now)
        control = compute_backup_control(robot_state.reshape(-1, 1), robot_spec, pocket_center, pocket_bounds)
        state.using_backup = True
    
    state.current_time_idx += 1
    
    # Update visualization
    update_visualization(ax, state, dt)
    
    return control


def solve_mps(robot_state: np.ndarray, obstacle_state: Dict,
              state: ShieldingState, config: TestConfig, env: EvadeEnv,
              pocket_center: np.ndarray, pocket_bounds: Dict, ax: plt.Axes) -> np.ndarray:
    """
    MPS (Model Predictive Shielding) algorithm with one-step nominal horizon.
    
    Key difference from Gatekeeper:
    - Only tries 1 step of nominal + full backup
    - Re-evaluates EVERY step (not at intervals)
    - When valid: applies nominal control (toward goal)
    - When invalid: applies backup control (toward pocket)
    """
    robot_state = np.array(robot_state).flatten()
    robot_spec = config.robot.to_dict()
    dt = config.simulation.dt
    backup_steps = int(config.simulation.backup_horizon_time / dt)
    
    # MPS re-evaluates EVERY step
    # Try ONE step of nominal + backup
    nominal_steps = 1
    nominal_x, nominal_u = forward_simulate(
        robot_state, 'nominal', nominal_steps, robot_spec, dt, pocket_center, pocket_bounds
    )
    
    # Append backup from end of nominal
    backup_x, backup_u = forward_simulate(
        nominal_x[-1], 'backup', backup_steps, robot_spec, dt, pocket_center, pocket_bounds
    )
    candidate_x = np.vstack([nominal_x, backup_x[1:]])
    candidate_u = np.vstack([nominal_u, backup_u])
    
    # Validate candidate trajectory
    is_valid = is_trajectory_valid(candidate_x, obstacle_state, env, config.robot.radius, dt)
    
    # Debug output
    obs_x = obstacle_state['x']
    if state.current_time_idx % 20 == 0:  # Print every 20 steps (1 second)
        print(f"  MPS: robot=({robot_state[0]:.1f}, {robot_state[1]:.1f}), "
              f"bullet_x={obs_x:.1f}, valid={is_valid}")
        # If invalid, show why
        if not is_valid:
            is_trajectory_valid(candidate_x, obstacle_state, env, config.robot.radius, dt, debug=True)
    
    if is_valid:
        # Valid: apply NOMINAL control (first control from trajectory)
        # This makes the robot move toward the goal
        control = nominal_u[0].reshape(-1, 1)
        state.using_backup = False
        
        # Update committed trajectory for visualization
        state.committed_x_traj = candidate_x
        state.committed_u_traj = candidate_u
        state.committed_horizon = nominal_steps * dt
    else:
        # Invalid: apply BACKUP control directly
        # This makes the robot move toward the pocket
        control = compute_backup_control(robot_state.reshape(-1, 1), robot_spec, pocket_center, pocket_bounds)
        state.using_backup = True
        
        # Update committed trajectory for visualization (pure backup)
        state.committed_x_traj = backup_x
        state.committed_u_traj = backup_u
        state.committed_horizon = 0.0
    
    state.current_time_idx += 1
    
    # Update visualization
    update_visualization(ax, state, dt)
    
    return control


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
    
    print(f"  Pocket center: ({pocket_center[0]:.1f}, {pocket_center[1]:.1f})")
    print(f"  Pocket bounds: x=[{pocket_bounds['x_min']:.1f}, {pocket_bounds['x_max']:.1f}], "
          f"y=[{pocket_bounds['y_min']:.1f}, {pocket_bounds['y_max']:.1f}]")
    
    # Setup shielding state and visualization
    shielding_state = ShieldingState()
    setup_visualization(ax, shielding_state)
    
    # Select algorithm
    if config.algo_type == 'mps':
        solve_func = solve_mps
        print(f"  Algorithm: MPS (one-step nominal horizon)")
    else:
        solve_func = solve_gatekeeper
        print(f"  Algorithm: GATEKEEPER (backward search)")
    
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
    
    for step in range(num_steps):
        pos = state[:2, 0]
        
        # Get bullet state
        bullet_state = env.get_bullet_state()
        
        # Shielding control
        control = solve_func(state.flatten(), bullet_state, shielding_state, 
                            config, env, pocket_center, pocket_bounds, ax)
        
        # Track mode
        if shielding_state.using_backup:
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
        
        # Save animation frame
        if animation_saver is not None:
            animation_saver.save_frame(fig)
        
        # Check collision with bullet
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
        if step % 100 == 0:
            mode = "BACKUP" if shielding_state.using_backup else "NOMINAL"
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
    
    # Export video if animation was saved
    if animation_saver is not None:
        animation_saver.export_video(output_name=f"{config.name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.mp4")
    
    return results


if __name__ == "__main__":
    main()
