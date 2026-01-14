"""
Created on January 6th, 2026
@author: Taekyung Kim

@description:
MPS (Model Predictive Shielding) - Safety shielding algorithm that guarantees safety.
This algorithm is based on Gatekeeper but uses a simpler one-step nominal horizon strategy.

Key Difference from Gatekeeper:
- Gatekeeper: Searches backward from full nominal horizon to find maximum valid switching time
- MPS: Only tries ONE step of nominal trajectory, then stitches backup

The MPS algorithm:
1. At each timestep, attempts to commit trajectory: 1 step nominal + full backup
2. If this candidate is valid (collision-free), commit it as the new trajectory
3. If invalid, stick with the previously committed trajectory (which is pure backup)

This leads to a more conservative behavior where the switching time is always
just one step ahead of the current robot time (or pure backup if invalid).

@required-scripts: safe_control/shielding/gatekeeper.py
"""

import numpy as np
from .gatekeeper import Gatekeeper


class MPS(Gatekeeper):
    """
    Model Predictive Shielding (MPS) algorithm.
    
    Inherits from Gatekeeper but simplifies the trajectory update logic:
    - Only attempts one-step nominal + backup trajectory
    - No backward search for switching time
    - Simple if-else: valid → commit, invalid → keep previous trajectory
    """
    
    def __init__(self, robot, robot_spec, dt=0.05, 
                 backup_horizon=2.0, event_offset=0.5, ax=None):
        """
        Initialize the MPS controller.

        Args:
            robot: Robot instance (e.g., DriftingCar) with dynamics
            robot_spec: Robot specification dictionary
            dt: Simulation timestep
            backup_horizon: Duration (seconds) for backup trajectory (TB in paper)
            event_offset: Time offset before next candidate generation event
            ax: Matplotlib axis for visualization (optional)
        """
        super().__init__(robot, robot_spec, dt, backup_horizon, event_offset, ax)
        self._control_matches_nominal = False  # Track if output matches u_ref
    
    def is_using_backup(self):
        """Check if currently in backup mode based on control comparison."""
        return not self._control_matches_nominal
        
    def solve_control_problem(self, robot_state, friction=None):
        """
        Main MPS control loop.
        
        Unlike Gatekeeper which searches backward for maximum valid nominal horizon,
        MPS only attempts ONE step of nominal trajectory.
        
        Args:
            robot_state: Current robot state
            friction: Current friction coefficient (for dynamics simulation)
            
        Returns:
            control_input: Control output for current timestep
        """
        robot_state = np.array(robot_state).flatten()
        
        # Initialize committed trajectory if not yet done
        if self.committed_x_traj is None or self.committed_u_traj is None:
            backup_horizon_steps = int(self.backup_horizon / self.dt)
            backup_x_traj, backup_u_traj = self._forward_simulate_backup(
                robot_state, backup_horizon_steps, friction
            )
            # Include initial state
            self.committed_x_traj = np.vstack([robot_state.reshape(1, -1), backup_x_traj])
            self.committed_u_traj = backup_u_traj
            self.committed_horizon = 0.0  # Pure backup initially
            self.actual_nominal_steps = 0
            self.current_time_idx = 0
            self.next_event_time = 0.0  # Trigger event immediately on next call
        
        # MPS: Re-evaluate EVERY step (key difference from Gatekeeper)
        # MPS: Only try ONE step of nominal
        nominal_horizon_steps = 1
        
        # Check if we can generate a trajectory (either external or via controller)
        can_generate = (self.nominal_x_traj is not None and len(self.nominal_x_traj) > 1) or \
                       (self.nominal_controller is not None)
        
        if can_generate:
            # Generate candidate trajectory with one-step nominal + backup
            candidate_x_traj, candidate_u_traj, actual_steps = self._generate_candidate_trajectory(
                robot_state, nominal_horizon_steps, friction
            )
            
            # Get moving obstacle states for validation
            obstacle_states = None
            if self.moving_obstacles is not None:
                if callable(self.moving_obstacles):
                    # Generate obstacle states over the horizon
                    obstacle_states = []
                    for k in range(len(candidate_x_traj)):
                        t = k * self.dt
                        try:
                            obs_state = self.moving_obstacles(t)
                        except TypeError:
                            obs_state = self.moving_obstacles()
                        obstacle_states.append(obs_state)
                elif isinstance(self.moving_obstacles, list) and len(self.moving_obstacles) == len(candidate_x_traj):
                     obstacle_states = self.moving_obstacles
                else:
                     obstacle_states = [self.moving_obstacles] * len(candidate_x_traj)
            
            # Check validity with safety margin (conservative)
            if self._is_candidate_valid(candidate_x_traj, safety_margin=1.0, obstacle_states=obstacle_states):
                # Valid: commit the one-step nominal + backup trajectory
                self._update_committed_trajectory(actual_steps)
            else:
                # Invalid: keep previously committed trajectory, reschedule event
                self.next_event_time = self.current_time_idx * self.dt + self.event_offset
        
        # Output control from committed trajectory using current index
        if self.current_time_idx < len(self.committed_u_traj):
            control = self.committed_u_traj[self.current_time_idx].reshape(-1, 1)
        else:
            # Fallback: use backup controller directly
            if self.backup_controller is not None:
                control = self.backup_controller.compute_control(
                    robot_state.reshape(-1, 1),
                    self.backup_target
                )
            else:
                control = np.zeros((self.n_controls, 1))
        
        # Compare output control to what nominal controller would produce
        # This determines mode: NOMINAL (u ≈ u_ref) or BACKUP (u ≠ u_ref)
        u_ref = None
        if self.nominal_controller is not None:
            u_ref = np.array(self.nominal_controller(robot_state.reshape(-1, 1))).reshape(-1, 1)
        elif self.nominal_u_traj is not None and len(self.nominal_u_traj) > 0:
            u_ref = self.nominal_u_traj[0].reshape(-1, 1)
        
        if u_ref is not None:
            control_diff = np.linalg.norm(control.flatten() - u_ref.flatten())
            self._control_matches_nominal = control_diff < 1e-2  # Threshold for matching
        else:
            self._control_matches_nominal = False
        
        # Increment index AFTER getting control (for next iteration)
        self.current_time_idx += 1
        
        # Update visualization
        self._update_visualization()
        
        return control
