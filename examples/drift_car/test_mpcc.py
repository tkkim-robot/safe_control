"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Test script for MPCC (Model Predictive Contouring Control) with Dynamic Bicycle Model.
Demonstrates path following using contouring MPC with Fiala tire dynamics.

Track Modes:
- straight: Simple straight track path following (no obstacles)
- oval: Oval track path following (no obstacles)
- gatekeeper: Straight track with obstacles, puddles, and Gatekeeper safety shielding

Usage:
    uv run python mpcbf/examples/test_mpcc.py [--track straight|oval|gatekeeper] [--save]

Examples:
    # Run pure MPCC on straight track
    uv run python mpcbf/examples/test_mpcc.py --track straight
    
    # Run pure MPCC on oval track and save animation
    uv run python mpcbf/examples/test_mpcc.py --track oval --save
    
    # Run MPCC with gatekeeper safety shielding (default)
    uv run python mpcbf/examples/test_mpcc.py --track gatekeeper --save

@required-scripts: mpcbf/envs/drifting_env.py, safe_control/robots/drifting_car.py,
                   safe_control/robots/dynamic_bicycle2D.py, safe_control/position_control/mpcc.py
"""

import numpy as np
import matplotlib.pyplot as plt

from safe_control.envs.drifting_env import DriftingEnv
from safe_control.robots.drifting_car import DriftingCar, DriftingCarSimulator
from safe_control.position_control.mpcc import MPCC
from safe_control.utils.animation import AnimationSaver


def create_robot_spec_high_friction():
    """
    Create robot specification for high friction conditions.
    
    Returns:
        Dictionary with vehicle parameters
    """
    return {
        # Geometry
        'a': 1.4,           # Front axle to CG [m]
        'b': 1.4,           # Rear axle to CG [m]
        'wheel_base': 2.8,  # Total wheelbase [m]
        'body_length': 4.5,
        'body_width': 2.0,
        'radius': 1.5,      # Collision radius
        
        # Mass and inertia
        'm': 2500.0,        # Vehicle mass [kg]
        'Iz': 5000.0,       # Yaw moment of inertia [kg*m^2]
        
        # Tire parameters
        'Cc_f': 150000.0,   # Front cornering stiffness [N/rad]
        'Cc_r': 180000.0,   # Rear cornering stiffness [N/rad]
        'mu': 1.0,          # Friction coefficient
        'r_w': 0.35,        # Wheel radius [m]
        'gamma': 0.99,      # Numeric stability parameter
        
        # Input limits
        'delta_max': np.deg2rad(20),     # Max steering [rad]
        'delta_dot_max': np.deg2rad(15), # Max steering rate [rad/s]
        'tau_max': 4000.0,               # Max torque [Nm]
        'tau_dot_max': 8000.0,           # Max torque rate [Nm/s]
        
        # State limits
        'v_max': 20.0,      # Max velocity [m/s]
        'v_min': 0.5,       # Min velocity [m/s]
        'r_max': 2.0,       # Max yaw rate [rad/s]
        'beta_max': np.deg2rad(45),  # Max slip angle [rad]
        
        # MPCC specific
        'v_psi_max': 15.0,  # Max progress rate [m/s]
    }


def run_mpcc_oval_track(save_animation=False):
    """Run MPCC on an oval track."""
    print("=" * 60)
    print("       MPCC Test - Oval Track (High Friction)")
    print("=" * 60)
    
    # Simulation parameters
    dt = 0.05
    tf = 20.0  # Total simulation time
    
    # Create oval track
    env = DriftingEnv(
        track_type='oval',
        track_width=16.0,
        track_length=100.0
    )
    
    # Setup plot
    plt.ion()
    ax, fig = env.setup_plot()
    fig.canvas.manager.set_window_title('MPCC - Oval Track')
    
    # Robot specification
    robot_spec = create_robot_spec_high_friction()
    
    # Initial state: start at specified position along the track
    n_points = len(env.centerline)
    start_idx = int(n_points * 0.70)  # Start at 70% around the track
    x0 = env.centerline[start_idx, 0]
    y0 = env.centerline[start_idx, 1]
    
    # Initial heading (tangent to track)
    if start_idx < len(env.centerline) - 1:
        dx = env.centerline[start_idx + 1, 0] - env.centerline[start_idx, 0]
        dy = env.centerline[start_idx + 1, 1] - env.centerline[start_idx, 1]
        theta0 = np.arctan2(dy, dx)
    else:
        theta0 = 0.0
    
    print(f"\nStarting at track index {start_idx}/{n_points}")
    
    # Initial velocity
    V0 = 7.0  # Start with target velocity
    
    # Full initial state: [x, y, theta, r, beta, V, delta, tau]
    X0 = np.array([x0, y0, theta0, 0, 0, V0, 0, 0])
    
    print(f"\nInitial state:")
    print(f"  Position: ({x0:.2f}, {y0:.2f}) m")
    print(f"  Heading: {np.rad2deg(theta0):.1f}°")
    print(f"  Velocity: {V0:.1f} m/s")
    
    # Create car
    car = DriftingCar(X0, robot_spec, dt, ax)
    
    # Create MPCC controller
    print("\nInitializing MPCC controller...")
    mpcc = MPCC(car, car.robot_spec, show_mpc_traj=False)
    
    # Set reference path (track centerline)
    mpcc.set_reference_path(
        env.centerline[:, 0],
        env.centerline[:, 1]
    )
    
    # Set MPCC cost function weights
    mpcc.set_cost_weights(
        Q_c=30.0,       # Contouring error weight
        Q_l=0.1,        # Lag error weight
        Q_theta=1500.0, # Heading error weight
        Q_v=100.0,      # Velocity tracking weight
        Q_r=20.0,       # Yaw rate penalty weight
        v_ref=7.0,      # Target velocity [m/s]
        R=np.array([50.0, 0.1, 0.0]),  # Control effort weights
    )
    mpcc.set_progress_rate(7.0)
    
    # Plot full track centerline (faint)
    ax.plot(
        env.centerline[:, 0], env.centerline[:, 1],
        'g-', linewidth=1, alpha=0.3, label='Track centerline'
    )
    
    # Reference horizon (local lookahead) - will be updated each step
    ref_horizon_line, = ax.plot(
        [], [], 'g-', linewidth=3, alpha=0.8, label='Reference horizon'
    )
    
    # MPC predicted trajectory
    mpc_pred_line, = ax.plot(
        [], [], 'r--', linewidth=2, alpha=0.8, label='MPC prediction'
    )
    
    ax.legend(loc='upper right')
    
    # Create simulator
    simulator = DriftingCarSimulator(car, env, show_animation=True)
    
    # Setup animation saver if enabled
    animation_saver = None
    if save_animation:
        animation_saver = AnimationSaver(output_dir="output/animations/mpcc_oval", save_per_frame=1, fps=30)
        print("\n  Animation saving enabled -> output/animations/mpcc_oval/")
    
    print(f"\nRunning simulation for {tf:.0f} seconds...")
    print("The car should follow the oval track.\n")
    
    # Simulation loop
    num_steps = int(tf / dt)
    
    # Speed checker
    low_speed_threshold = 2.0  # m/s
    low_speed_counter = 0
    low_speed_limit = 20  # steps
    
    for step in range(num_steps):
        # Get current state
        state = car.get_state()
        
        # Solve MPCC
        try:
            U = mpcc.solve_control_problem(state)
        except Exception as e:
            print(f"MPCC error at step {step}: {e}")
            U = car.stop()
        
        # Step simulation
        result = simulator.step(U)
        
        # Update visualizations
        # Reference horizon (local lookahead)
        ref_horizon = mpcc.get_reference_horizon()
        if ref_horizon is not None:
            ref_horizon_line.set_data(ref_horizon[0, :], ref_horizon[1, :])
        
        # MPC predicted trajectory
        pred_states, _ = mpcc.get_predictions()
        if pred_states is not None:
            mpc_pred_line.set_data(pred_states[0, :], pred_states[1, :])
            car.set_mpc_prediction(pred_states, None)
        
        # Update plot
        simulator.draw_plot(pause=0.001)
        
        # Save animation frame
        if animation_saver is not None:
            animation_saver.save_frame(fig)
        
        # Get current velocity
        V = car.get_velocity()
        
        # Print status periodically
        if step % 50 == 0:
            pos = car.get_position()
            beta = car.get_slip_angle()
            delta = car.get_steering_angle()
            print(f"Step {step:4d}: pos=({pos[0]:6.2f},{pos[1]:6.2f}), V={V:4.1f} m/s, "
                  f"delta={np.rad2deg(delta):5.1f}°, beta={np.rad2deg(beta):4.1f}°")
        
        # Check for collision
        if result['collision']:
            print(f"\nCOLLISION at step {step}")
            pos = car.get_position()
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"  Velocity: {V:.2f} m/s")
            print(f"  Steering: {np.rad2deg(car.get_steering_angle()):.1f}°")
            plt.pause(3.0)
            break
        
        # Speed checker
        if V < low_speed_threshold:
            low_speed_counter += 1
            if low_speed_counter >= low_speed_limit:
                print(f"\nLOW SPEED TIMEOUT at step {step}")
                pos = car.get_position()
                print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
                print(f"  Velocity: {V:.2f} m/s (below {low_speed_threshold} for {low_speed_limit} steps)")
                print(f"  Steering: {np.rad2deg(car.get_steering_angle()):.1f}°")
                print(f"  Yaw rate: {np.rad2deg(car.get_yaw_rate()):.1f}°/s")
                print(f"  Slip angle: {np.rad2deg(car.get_slip_angle()):.1f}°")
                print(f"  Control: U = [{U[0,0]:.3f}, {U[1,0]:.1f}]")
                plt.pause(3.0)
                break
        else:
            low_speed_counter = 0
    
    # Export video if animation was saved
    if animation_saver is not None:
        animation_saver.export_video(output_name="mpcc_oval.mp4")
    
    print("\nSimulation complete!")
    plt.ioff()
    plt.show()


def run_mpcc_straight_track(save_animation=False):
    """Run MPCC on a straight track - pure path following without obstacles."""
    print("=" * 60)
    print("       MPCC Test - Straight Track")
    print("=" * 60)
    
    dt = 0.05
    tf = 10.0  # Simulation time
    
    # Create straight track
    lane_width = 4.0
    num_lanes = 3
    total_width = lane_width * num_lanes
    
    env = DriftingEnv(
        track_type='straight',
        track_width=total_width,
        track_length=300.0,
        num_lanes=num_lanes
    )
    
    plt.ion()
    ax, fig = env.setup_plot()
    fig.canvas.manager.set_window_title('MPCC - Straight Track')
    
    robot_spec = create_robot_spec_high_friction()
    
    # Middle lane
    middle_lane = env.get_middle_lane_idx()
    middle_lane_y = env.get_lane_center(middle_lane)
    
    print(f"\nTrack configuration: {num_lanes} lanes, {lane_width}m wide each")
    print(f"Robot starts in middle lane (y={middle_lane_y:.1f}m)")
    
    # Initial state
    V0 = 8.0
    X0 = np.array([6.0, middle_lane_y, np.deg2rad(10), 0, 0, V0, 0, 0])
    
    print(f"\nInitial state: x={X0[0]:.1f}, y={X0[1]:.1f}, V={V0:.1f} m/s")
    
    car = DriftingCar(X0, robot_spec, dt, ax)
    
    # Create MPCC controller
    ref_x = env.centerline[:, 0]
    ref_y = np.full_like(ref_x, middle_lane_y)
    
    mpcc = MPCC(car, car.robot_spec)
    mpcc.set_reference_path(ref_x, ref_y)
    mpcc.set_cost_weights(
        Q_c=50.0,       # Contouring error weight
        Q_l=1.0,        # Lag error weight
        Q_theta=30.0,   # Heading error weight
        Q_v=50.0,       # Velocity tracking weight
        Q_r=20.0,       # Yaw rate penalty weight
        v_ref=V0,       # Target velocity [m/s]
        R=np.array([150.0, 0.1, 0.1]),
    )
    mpcc.set_progress_rate(V0)
    
    # Visualization
    ax.plot(ref_x, ref_y, 'g-', linewidth=1, alpha=0.3, label='Reference path')
    ref_horizon_line, = ax.plot([], [], 'g-', linewidth=3, alpha=0.8, label='Reference horizon')
    mpc_pred_line, = ax.plot([], [], 'r--', linewidth=2, alpha=0.8, label='MPC prediction')
    ax.legend(loc='upper right', fontsize=8)
    
    simulator = DriftingCarSimulator(car, env, show_animation=True)
    
    # Setup animation saver if enabled
    animation_saver = None
    if save_animation:
        animation_saver = AnimationSaver(output_dir="output/animations/mpcc_straight", save_per_frame=1, fps=30)
        print("\n  Animation saving enabled -> output/animations/mpcc_straight/")
    
    print(f"\nRunning simulation for {tf:.0f} seconds...")
    print("The car should follow the straight path.\n")
    
    num_steps = int(tf / dt)
    window_size = (60, 20)
    
    for step in range(num_steps):
        state = car.get_state()
        pos = car.get_position()
        
        try:
            U = mpcc.solve_control_problem(state)
        except Exception as e:
            print(f"MPCC error: {e}")
            U = car.stop()
        
        result = simulator.step(U)
        
        # Update visualizations
        ref_horizon = mpcc.get_reference_horizon()
        if ref_horizon is not None:
            ref_horizon_line.set_data(ref_horizon[0, :], ref_horizon[1, :])
        
        pred_states, _ = mpcc.get_predictions()
        if pred_states is not None:
            mpc_pred_line.set_data(pred_states[0, :], pred_states[1, :])
        
        env.update_plot_frame(ax, pos, window_size=window_size)
        simulator.draw_plot(pause=0.001)
        
        # Save animation frame
        if animation_saver is not None:
            animation_saver.save_frame(fig)
        
        if step % 50 == 0:
            V = car.get_velocity()
            delta = car.get_steering_angle()
            print(f"Step {step:4d}: x={pos[0]:6.2f}, y={pos[1]:6.2f}, V={V:5.2f} m/s, "
                  f"delta={np.rad2deg(delta):5.1f}°")
        
        if result['collision']:
            print(f"\nCOLLISION at step {step}")
            plt.pause(3.0)
            break
        
        if pos[0] > env.track_length - 10:
            print("\nReached end of track!")
            break
    
    # Export video if animation was saved
    if animation_saver is not None:
        animation_saver.export_video(output_name="mpcc_straight.mp4")
    
    print("\nSimulation complete!")
    plt.ioff()
    plt.show()


def test_parameter_combinations():
    """Test various parameter combinations."""
    print("=" * 60)
    print("       Parameter Sweep Test")
    print("=" * 60)
    
    # Test different friction coefficients and weights
    test_configs = [
        {'mu': 1.0, 'Q_c': 100, 'Q_l': 30, 'name': 'High friction, balanced'},
        {'mu': 1.0, 'Q_c': 200, 'Q_l': 10, 'name': 'High friction, tracking focus'},
        {'mu': 0.8, 'Q_c': 150, 'Q_l': 50, 'name': 'Medium friction'},
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print(f"  mu={config['mu']}, Q_c={config['Q_c']}, Q_l={config['Q_l']}")
        # Would run simulation here - keeping brief for demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MPCC controller')
    parser.add_argument('--track', type=str, default='straight',
                        choices=['straight', 'oval', 'test'],
                        help='Track type to test')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as video')
    
    args = parser.parse_args()
    
    if args.track == 'straight':
        run_mpcc_straight_track(save_animation=args.save)
    elif args.track == 'oval':
        run_mpcc_oval_track(save_animation=args.save)
    elif args.track == 'test':
        test_parameter_combinations()

