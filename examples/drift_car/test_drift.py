"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Test script for safety shielding algorithms (Gatekeeper and MPS) in the drift_car environment.
Contains modular test cases to validate shielding behavior under different conditions.

Algorithms:
- gatekeeper: Searches backward for maximum valid nominal horizon
- mps: Model Predictive Shielding - uses single-step nominal horizon

Test Cases:
1. High Friction - Normal conditions, algorithm should avoid obstacle
2. Low Friction - Slippery surface everywhere, algorithm should still avoid obstacle
3. Puddle Surprise - Puddle in front of obstacle causes algorithm to fail
   (demonstrates limitation: algorithm plans with current friction estimate)

Backup Controllers:
- lane_change: Lane change to left lane (avoids obstacle by changing lanes)
- stop: Emergency braking to stop the vehicle (expected to fail in puddle scenario)

Number of Obstacles:
- 1: Single obstacle in middle lane (default)
- 2: Two obstacles - one in middle lane, one in left lane (blocks lane change backup)

Usage:
    uv run python examples/drift_car/test_drift.py [--test TEST] [--algo ALGO] [--backup BACKUP] [--obs NUM] [--save]

Examples:
    # Test with gatekeeper (default)
    uv run python examples/drift_car/test_drift.py --test high_friction --algo gatekeeper
    
    # Test with MPS algorithm
    uv run python examples/drift_car/test_drift.py --test high_friction --algo mps
    
    # Test with stopping backup (expected to fail in puddle scenario)
    uv run python examples/drift_car/test_drift.py --test puddle_surprise --backup stop
    
    # Test with 2 obstacles (lane change will fail due to blocked left lane)
    uv run python examples/drift_car/test_drift.py --test high_friction --backup lane_change --obs 2
    
    # Save animation
    uv run python examples/drift_car/test_drift.py --test high_friction --save

@required-scripts: safe_control/shielding/gatekeeper.py, safe_control/shielding/mps.py
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

from safe_control.envs.drifting_env import DriftingEnv
from safe_control.robots.drifting_car import DriftingCar, DriftingCarSimulator
from safe_control.position_control.mpcc import MPCC
from safe_control.position_control.backup_controller import LaneChangeController, StoppingController
from safe_control.shielding.gatekeeper import Gatekeeper
from safe_control.shielding.mps import MPS
from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.utils.animation import AnimationSaver


# =============================================================================
# Algorithm Types
# =============================================================================

ALGO_TYPES = ['gatekeeper', 'mps', 'backupcbf']


# =============================================================================
# Backup Controller Types
# =============================================================================

BACKUP_TYPES = ['lane_change', 'stop']


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class TrackConfig:
    """Track configuration parameters."""
    track_type: str = 'straight'
    track_length: float = 300.0
    lane_width: float = 4.0
    num_lanes: int = 5


@dataclass
class VehicleConfig:
    """Vehicle configuration parameters."""
    # Geometry
    a: float = 1.4              # Front axle to CG [m]
    b: float = 1.4              # Rear axle to CG [m]
    wheel_base: float = 2.8     # Total wheelbase [m]
    body_length: float = 4.5
    body_width: float = 2.0
    radius: float = 1.5         # Collision radius
    
    # Mass and inertia
    m: float = 2500.0           # Vehicle mass [kg]
    Iz: float = 5000.0          # Yaw moment of inertia [kg*m^2]
    
    # Tire parameters - lower stiffness = more slip at low friction
    Cc_f: float = 80000.0       # Front cornering stiffness [N/rad] (reduced for more slip)
    Cc_r: float = 100000.0      # Rear cornering stiffness [N/rad] (reduced for more slip)
    mu: float = 1.0             # Friction coefficient (default high)
    r_w: float = 0.35           # Wheel radius [m]
    gamma: float = 0.95         # Numeric stability parameter (lower = more realistic slip)
    
    # Input limits
    delta_max: float = np.deg2rad(20)      # Max steering [rad]
    delta_dot_max: float = np.deg2rad(15)  # Max steering rate [rad/s]
    tau_max: float = 4000.0                # Max torque [Nm]
    tau_dot_max: float = 8000.0            # Max torque rate [Nm/s]
    
    # State limits
    v_max: float = 20.0         # Max velocity [m/s]
    v_min: float = 0.0          # Min velocity [m/s] - allow complete stop
    r_max: float = 2.0          # Max yaw rate [rad/s]
    beta_max: float = np.deg2rad(45)  # Max slip angle [rad]
    
    # MPCC specific
    v_psi_max: float = 15.0     # Max progress rate [m/s]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for robot_spec."""
        return {
            'a': self.a, 'b': self.b, 'wheel_base': self.wheel_base,
            'body_length': self.body_length, 'body_width': self.body_width,
            'radius': self.radius, 'm': self.m, 'Iz': self.Iz,
            'Cc_f': self.Cc_f, 'Cc_r': self.Cc_r, 'mu': self.mu,
            'r_w': self.r_w, 'gamma': self.gamma,
            'delta_max': self.delta_max, 'delta_dot_max': self.delta_dot_max,
            'tau_max': self.tau_max, 'tau_dot_max': self.tau_dot_max,
            'v_max': self.v_max, 'v_min': self.v_min,
            'r_max': self.r_max, 'beta_max': self.beta_max,
            'v_psi_max': self.v_psi_max,
        }


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    dt: float = 0.05
    tf: float = 14.0
    nominal_horizon_time: float = 1.5    # MPCC prediction horizon [s]
    backup_horizon_time: float = 3.0     # Backup trajectory horizon [s]
    event_offset: float = 0.1            # Gatekeeper re-evaluation interval [s]
    safety_margin: float = 1.5           # Collision checking margin [m]
    initial_velocity: float = 10.0        # Starting velocity [m/s]
    target_velocity: float = 10.0         # Target velocity [m/s]


@dataclass
class ObstacleConfig:
    """Single obstacle configuration."""
    x: float = 80.0             # X position
    y: Optional[float] = None   # Y position (None = middle lane)
    theta: float = 0.0          # Heading angle
    body_length: float = 4.5
    body_width: float = 2.0
    radius: float = 2.5         # Collision radius (larger for safety)



# Number of obstacles options
NUM_OBSTACLES = [1, 2]


@dataclass
class PuddleConfig:
    """Puddle (low friction area) configuration."""
    x: float
    y: float
    radius: float
    friction: float = 0.3


@dataclass 
class TestConfig:
    """Complete test configuration."""
    name: str
    description: str
    track: TrackConfig
    vehicle: VehicleConfig
    simulation: SimulationConfig
    obstacles: list  # List of ObstacleConfig
    puddles: list  # List of PuddleConfig
    expected_collision: bool = False  # Whether collision is expected
    save_animation: bool = False  # Whether to save animation as video
    backup_type: str = 'lane_change'  # Backup controller type: 'lane_change' or 'stop'
    num_obstacles: int = 1  # Number of obstacles to use
    algo_type: str = 'gatekeeper'  # Algorithm type: 'gatekeeper' or 'mps'


# =============================================================================
# Test Environment Setup
# =============================================================================

def setup_environment(config: TestConfig) -> Tuple[DriftingEnv, plt.Axes, plt.Figure]:
    """Setup the test environment."""
    track = config.track
    total_width = track.lane_width * track.num_lanes
    
    env = DriftingEnv(
        track_type=track.track_type,
        track_width=total_width,
        track_length=track.track_length,
        num_lanes=track.num_lanes
    )
    
    plt.ion()
    ax, fig = env.setup_plot()
    algo_name = config.algo_type.upper()
    fig.canvas.manager.set_window_title(f'{algo_name} Test: {config.name}')
    
    return env, ax, fig


def setup_vehicle(config: TestConfig, env: DriftingEnv, ax: plt.Axes) -> Tuple[DriftingCar, np.ndarray, float, float]:
    """Setup the vehicle and get lane positions."""
    middle_lane = env.get_middle_lane_idx()
    middle_lane_y = env.get_lane_center(middle_lane)
    left_lane_y = env.get_lane_center(middle_lane - 1)
    
    # Initial state in middle lane
    X0 = np.array([
        5.0,                        # x
        middle_lane_y,              # y
        np.deg2rad(0),              # theta
        0, 0,                       # r, beta
        config.simulation.initial_velocity,  # V
        0, 0                        # delta, tau
    ])
    
    robot_spec = config.vehicle.to_dict()
    car = DriftingCar(X0, robot_spec, config.simulation.dt, ax)
    
    return car, X0, middle_lane_y, left_lane_y


def setup_controllers(
    config: TestConfig, 
    car: DriftingCar, 
    env: DriftingEnv,
    middle_lane_y: float,
    left_lane_y: float,
    ax: plt.Axes
) -> Tuple[MPCC, Union[Gatekeeper, MPS]]:
    """Setup MPCC and shielding controller (Gatekeeper or MPS)."""
    sim = config.simulation
    robot_spec = config.vehicle.to_dict()
    
    # Reference path along middle lane
    ref_x = env.centerline[:, 0]
    ref_y = np.full_like(ref_x, middle_lane_y)
    
    # MPCC controller
    nominal_horizon_steps = int(sim.nominal_horizon_time / sim.dt)
    mpcc = MPCC(car, car.robot_spec, horizon=nominal_horizon_steps)
    mpcc.set_reference_path(ref_x, ref_y)
    mpcc.set_cost_weights(
        Q_c=30.0,       # Contouring error (reduced - less aggressive correction)
        Q_l=1.0,        # Lag error
        Q_theta=20.0,   # Heading error (reduced)
        Q_v=50.0,       # Velocity tracking
        Q_r=80.0,       # Yaw rate penalty (increased - more damping)
        v_ref=sim.target_velocity,
        R=np.array([300.0, 0.5, 0.1]),  # Steering rate penalty increased for smoother control
    )
    mpcc.set_progress_rate(sim.target_velocity)
    
    # Backup controller - choose based on config
    if config.backup_type == 'stop':
        backup_controller = StoppingController(car.robot_spec, sim.dt)
        backup_target = None  # Stopping doesn't need a target
        print(f"  Using STOPPING backup controller")
    else:  # 'lane_change' (default)
        backup_controller = LaneChangeController(car.robot_spec, sim.dt, direction='left')
        # Target is the center of the left lane as requested by user
        backup_target = left_lane_y  # never change the target here.
        print(f"  Using LANE CHANGE backup controller (target y={backup_target:.2f})")
    
    # Shielding algorithm - choose based on config
    if config.algo_type == 'mps':
        shielding = MPS(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            event_offset=sim.event_offset,
            ax=ax
        )
        print(f"  Using MPS algorithm (one-step nominal horizon)")
    elif config.algo_type == 'backupcbf':
        shielding = BackupCBF(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            ax=ax
        )
        print(f"  Using BACKUPCBF algorithm (backup CBF-QP)")
    else:  # 'gatekeeper' (default)
        shielding = Gatekeeper(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            event_offset=sim.event_offset,
            ax=ax
        )
        print(f"  Using GATEKEEPER algorithm (backward search)")
    
    shielding.set_backup_controller(backup_controller, target=backup_target)
    shielding.set_environment(env)
    
    return mpcc, shielding


def setup_obstacles_and_puddles(
    config: TestConfig, 
    env: DriftingEnv, 
    middle_lane_y: float,
    left_lane_y: float
):
    """Add obstacles and puddles to the environment."""
    # Add obstacles (up to num_obstacles)
    for i, obs in enumerate(config.obstacles[:config.num_obstacles]):
        # Determine Y position
        if obs.y is not None:
            obs_y = obs.y
        elif i == 0:
            obs_y = middle_lane_y  # First obstacle in middle lane
        else:
            obs_y = left_lane_y  # Additional obstacles in left lane by default
        
        obstacle_spec = {
            'body_length': obs.body_length,
            'body_width': obs.body_width,
            'a': 1.4, 'b': 1.4,
            'radius': obs.radius,
        }
        env.add_obstacle_car(x=obs.x, y=obs_y, theta=obs.theta, robot_spec=obstacle_spec)
        print(f"  Obstacle {i+1}: x={obs.x:.1f}, y={obs_y:.1f}")
    
    # Add puddles
    for puddle in config.puddles:
        env.add_puddle(x=puddle.x, y=puddle.y, radius=puddle.radius, friction=puddle.friction)


def setup_visualization(
    ax: plt.Axes, 
    env: DriftingEnv, 
    middle_lane_y: float, 
    left_lane_y: float
) -> Tuple:
    """Setup visualization elements."""
    ref_x = env.centerline[:, 0]
    
    # Reference paths
    ax.plot(ref_x, np.full_like(ref_x, middle_lane_y), 
            'g-', linewidth=1, alpha=0.3, label='Reference (middle lane)')
    ax.plot(ref_x, np.full_like(ref_x, left_lane_y), 
            'orange', linewidth=1, alpha=0.3, linestyle=':', label='Backup target')
    
    # Dynamic visualization
    ref_horizon_line, = ax.plot([], [], 'y-', linewidth=3, alpha=0.9, label='MPCC horizon')
    mpc_pred_line, = ax.plot([], [], 'r--', linewidth=2, alpha=0.8, label='MPCC prediction')
    
    ax.legend(loc='upper right', fontsize=8)
    
    return ref_horizon_line, mpc_pred_line


# =============================================================================
# Simulation Loop
# =============================================================================

def run_simulation(
    config: TestConfig,
    car: DriftingCar,
    env: DriftingEnv,
    mpcc: MPCC,
    shielding: Union[Gatekeeper, MPS],
    simulator: DriftingCarSimulator,
    ref_horizon_line,
    mpc_pred_line,
    ax: plt.Axes,
    fig: plt.Figure,
    animation_saver: Optional[AnimationSaver] = None,
) -> Dict[str, Any]:
    """Run the simulation loop and return results."""
    sim = config.simulation
    robot_spec = config.vehicle.to_dict()
    
    num_steps = int(sim.tf / sim.dt)
    window_size = (60, 30)
    last_friction = robot_spec['mu']
    
    # Statistics
    nominal_steps = 0
    backup_steps = 0
    collision_occurred = False
    collision_step = None
    
    print(f"\nRunning simulation for {sim.tf}s...")
    
    for step in range(num_steps):
        state = car.get_state()
        pos = car.get_position()
        
        # Update friction based on position (puddle check)
        current_friction = env.get_friction_at_position(pos, default_friction=robot_spec['mu'])
        if abs(current_friction - car.get_friction()) > 0.01:
            car.set_friction(current_friction)
            if abs(current_friction - last_friction) > 0.01:
                if current_friction < robot_spec['mu']:
                    print(f"Step {step:4d}: *** ENTERED PUDDLE - friction: {current_friction:.2f} ***")
                else:
                    print(f"Step {step:4d}: *** LEFT PUDDLE - friction: {current_friction:.2f} ***")
                last_friction = current_friction
        
        # Get MPCC's nominal plan
        try:
            mpcc_control = mpcc.solve_control_problem(state)
            pred_states, pred_controls = mpcc.get_full_predictions()
            
            if pred_states is not None and pred_controls is not None:
                shielding.set_nominal_trajectory(pred_states, pred_controls)
        except Exception as e:
            print(f"MPCC error: {e}")
            pred_states, pred_controls = None, None
        
        # Shielding validates and returns committed control
        U = shielding.solve_control_problem(state, friction=car.get_friction())
        
        # Track mode
        if shielding.is_using_backup():
            backup_steps += 1
        else:
            nominal_steps += 1
        
        # Apply control
        result = simulator.step(U)
        
        # Update visualizations
        ref_horizon = mpcc.get_reference_horizon()
        if ref_horizon is not None:
            ref_horizon_line.set_data(ref_horizon[0, :], ref_horizon[1, :])
        
        pred_states_viz, _ = mpcc.get_predictions()
        if pred_states_viz is not None:
            mpc_pred_line.set_data(pred_states_viz[0, :], pred_states_viz[1, :])
        
        env.update_plot_frame(ax, pos, window_size=window_size)
        simulator.draw_plot(pause=0.001)
        
        # Save animation frame
        if animation_saver is not None:
            animation_saver.save_frame(fig)
        
        # Status output
        if step % 50 == 0:
            V = car.get_velocity()
            status = shielding.get_status()
            mode = "BACKUP" if status['using_backup'] else "NOMINAL"
            h_min = status.get('h_min', 1.0)
            print(f"Step {step:4d}: x={pos[0]:6.2f}, y={pos[1]:6.2f}, V={V:5.2f} m/s, mode={mode:8s}, h_min={h_min:6.3f}")
        
        # Check collision
        if result['collision']:
            collision_occurred = True
            collision_step = step
            collision_type = getattr(simulator, 'collision_type', 'unknown')
            print(f"\n*** COLLISION ({collision_type}) at step {step} ***")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            if animation_saver is not None:
                animation_saver.save_frame(fig, force=True)
            plt.pause(2.0)
            break
        
        # End if reached track end
        if pos[0] > env.track_length - 10:
            print("\nReached end of track!")
            break
    
    # Return results
    total_steps = nominal_steps + backup_steps
    return {
        'collision': collision_occurred,
        'collision_step': collision_step,
        'total_steps': total_steps,
        'nominal_steps': nominal_steps,
        'backup_steps': backup_steps,
        'nominal_ratio': nominal_steps / max(total_steps, 1),
        'backup_ratio': backup_steps / max(total_steps, 1),
    }


# =============================================================================
# Main Test Runner
# =============================================================================

def run_test(config: TestConfig) -> Dict[str, Any]:
    """Run a complete test with the given configuration."""
    print("\n" + "=" * 70)
    print(f"  TEST: {config.name}")
    print(f"  {config.description}")
    print("=" * 70)
    
    # Setup
    env, ax, fig = setup_environment(config)
    car, X0, middle_lane_y, left_lane_y = setup_vehicle(config, env, ax)
    mpcc, shielding = setup_controllers(config, car, env, middle_lane_y, left_lane_y, ax)
    setup_obstacles_and_puddles(config, env, middle_lane_y, left_lane_y)
    ref_horizon_line, mpc_pred_line = setup_visualization(ax, env, middle_lane_y, left_lane_y)
    
    simulator = DriftingCarSimulator(car, env, show_animation=True)
    
    # Setup animation saver if enabled
    animation_saver = None
    if config.save_animation:
        # Create unique output directory for this test
        safe_name = config.name.lower().replace(' ', '_')
        output_dir = f"output/animations/{safe_name}"
        animation_saver = AnimationSaver(output_dir=output_dir, save_per_frame=1, fps=30)
        print(f"\n  Animation saving enabled -> {output_dir}/")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Algorithm: {config.algo_type}")
    print(f"  Friction: μ = {config.vehicle.mu}")
    print(f"  Obstacles: {config.num_obstacles}")
    print(f"  Puddles: {len(config.puddles)}")
    print(f"  Backup type: {config.backup_type}")
    print(f"  Expected collision: {config.expected_collision}")
    
    # Run simulation
    results = run_simulation(
        config, car, env, mpcc, shielding, simulator,
        ref_horizon_line, mpc_pred_line, ax, fig, animation_saver
    )
    
    # Export video if animation was saved
    if animation_saver is not None:
        animation_saver.export_video(output_name=f"{config.name.lower().replace(' ', '_')}.mp4")
    
    # Print results
    print("\n" + "-" * 50)
    print("Results:")
    print(f"  Collision: {'YES' if results['collision'] else 'NO'}")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Nominal: {results['nominal_steps']} ({100*results['nominal_ratio']:.1f}%)")
    print(f"  Backup: {results['backup_steps']} ({100*results['backup_ratio']:.1f}%)")
    
    # Check expectation
    if results['collision'] == config.expected_collision:
        print(f"\n  ✓ TEST PASSED (collision={'expected' if config.expected_collision else 'avoided'} as expected)")
        results['passed'] = True
    else:
        print(f"\n  ✗ TEST FAILED (expected collision={config.expected_collision}, got {results['collision']})")
        results['passed'] = False
    
    print("-" * 50)
    
    plt.ioff()
    plt.show(block=False)
    plt.pause(2)  # Show results briefly
    plt.close('all')
    
    return results


# =============================================================================
# Test Case Definitions
# =============================================================================

def create_high_friction_test() -> TestConfig:
    """Test Case 1: High friction - normal conditions."""
    return TestConfig(
        name="High Friction",
        description="Normal high friction conditions. Gatekeeper should avoid obstacle.",
        track=TrackConfig(),
        vehicle=VehicleConfig(mu=1.0),  # High friction
        simulation=SimulationConfig(),
        obstacles=[
            ObstacleConfig(x=80.0, y=None),       # First obstacle in middle lane
            ObstacleConfig(x=85.0, y=None),       # Second obstacle in left lane (y=None uses default)
        ],
        puddles=[],  # No puddles
        expected_collision=False,
    )


def create_low_friction_test() -> TestConfig:
    """Test Case 2: Low friction everywhere."""
    return TestConfig(
        name="Low Friction",
        description="Low friction surface everywhere. Gatekeeper should still avoid obstacle.",
        track=TrackConfig(),
        vehicle=VehicleConfig(mu=0.3),  # Low friction everywhere
        simulation=SimulationConfig(),
        obstacles=[
            ObstacleConfig(x=80.0, y=None),       # First obstacle in middle lane
            ObstacleConfig(x=85.0, y=None),       # Second obstacle in left lane
        ],
        puddles=[],  # No puddles - friction is globally low
        expected_collision=False,
    )


def create_puddle_surprise_test() -> TestConfig:
    """Test Case 3: Puddle surprise - gatekeeper should fail."""
    # Get middle lane Y position
    track = TrackConfig()
    half_width = track.lane_width * track.num_lanes / 2
    middle_lane_y = half_width - (track.num_lanes // 2 + 0.5) * track.lane_width
    
    return TestConfig(
        name="Puddle Surprise",
        description="Large puddle in front of obstacle. Gatekeeper plans with high friction "
                    "but encounters low friction during execution, causing it to fail.",
        track=track,
        vehicle=VehicleConfig(mu=1.0),  # Start with high friction
        simulation=SimulationConfig(),
        obstacles=[
            ObstacleConfig(x=80.0, y=None),       # First obstacle in middle lane
            ObstacleConfig(x=85.0, y=None),       # Second obstacle in left lane
        ],
        puddles=[
            # Large puddle right in front of obstacle
            PuddleConfig(x=70.0, y=middle_lane_y, radius=15.0, friction=0.25),
        ],
        expected_collision=True,  # Expected to fail!
    )


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test safety shielding algorithms (Gatekeeper/MPS)')
    parser.add_argument('--test', type=str, default='high_friction',
                        choices=['high_friction', 'low_friction', 'puddle_surprise', 'all'],
                        help='Which test to run')
    parser.add_argument('--algo', type=str, default='gatekeeper',
                        choices=ALGO_TYPES,
                        help='Shielding algorithm: gatekeeper (default) or mps')
    parser.add_argument('--backup', type=str, default='lane_change',
                        choices=BACKUP_TYPES,
                        help='Backup controller type: lane_change (default) or stop')
    parser.add_argument('--obs', type=int, default=1,
                        choices=NUM_OBSTACLES,
                        help='Number of obstacles: 1 (default) or 2 (blocks lane change)')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as video')
    
    args = parser.parse_args()
    
    # Test configurations with expected collisions based on backup type
    # For stopping backup, puddle surprise is expected to collide
    # For lane change backup with high/low friction, no collision expected
    test_configs = {
        'high_friction': create_high_friction_test,
        'low_friction': create_low_friction_test,
        'puddle_surprise': create_puddle_surprise_test,
    }
    
    # Expected collision matrix based on (backup_type, num_obstacles)
    # Key: (backup_type, num_obstacles, test_name) -> expected_collision
    def get_expected_collision(test_name, backup_type, num_obstacles):
        """Determine expected collision based on test configuration."""
        # With 2 obstacles, lane change backup will fail (left lane blocked)
        if num_obstacles == 2 and backup_type == 'lane_change':
            return True  # Both lanes blocked - collision expected
        
        # With stopping backup
        if backup_type == 'stop':
            if test_name == 'puddle_surprise':
                return True  # Can't stop in time with puddle
            else:
                return False  # Should be able to stop with high/low friction
        
        # Default: lane change with 1 obstacle
        if test_name == 'puddle_surprise':
            return True  # Puddle causes failure
        return False  # Lane change should work
    
    results = {}
    
    if args.test == 'all':
        print("\n" + "=" * 70)
        print(f"  RUNNING ALL {args.algo.upper()} TESTS (backup: {args.backup}, obstacles: {args.obs})")
        print("=" * 70)
        
        for name, create_config in test_configs.items():
            config = create_config()
            config.save_animation = args.save
            config.algo_type = args.algo
            config.backup_type = args.backup
            config.num_obstacles = args.obs
            # Update expected collision based on configuration
            config.expected_collision = get_expected_collision(name, args.backup, args.obs)
            # Update name to include algo, backup type and obstacle count
            config.name = f"{config.name} ({args.algo}, {args.backup}, {args.obs} obs)"
            results[name] = run_test(config)
            input("\nPress Enter to continue to next test...")
        
        # Summary
        print("\n" + "=" * 70)
        print(f"  TEST SUMMARY ({args.algo}, backup: {args.backup}, obstacles: {args.obs})")
        print("=" * 70)
        for name, result in results.items():
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            collision = "collision" if result['collision'] else "no collision"
            print(f"  {name}: {status} ({collision})")
        
        passed = sum(1 for r in results.values() if r['passed'])
        total = len(results)
        print(f"\n  Total: {passed}/{total} tests passed")
        print("=" * 70)
    else:
        config = test_configs[args.test]()
        config.save_animation = args.save
        config.algo_type = args.algo
        config.backup_type = args.backup
        config.num_obstacles = args.obs
        # Update expected collision based on configuration
        config.expected_collision = get_expected_collision(args.test, args.backup, args.obs)
        # Update name to include algo, backup type and obstacle count
        config.name = f"{config.name} ({args.algo}, {args.backup}, {args.obs} obs)"
        results[args.test] = run_test(config)
    
    return results


if __name__ == "__main__":
    main()

