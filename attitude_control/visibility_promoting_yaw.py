import numpy as np
from shapely.geometry import Point, LineString

"""
Created on May 11th, 2025
@author: Taekyung Kim

@description: 
This code implements a visibility promoting yaw controller that smoothly rotates towards
the direction of maximum unexplored area. The controller uses P control to achieve smooth
rotation while still prioritizing exploration of unknown regions.

@note: 
- Uses P control for smooth rotation towards target angle
- Target angle is determined by finding the largest gap in visibility
- Uses sensing footprints to determine unexplored areas
- Can be used as a nominal attitude controller for the gatekeeper
"""

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class VisibilityAtt:
    """
    Visibility promoting attitude controller that smoothly rotates towards
    the direction of maximum unexplored area using P control.
    """
    def __init__(self, robot, robot_spec, kp=1.5):
        self.robot = robot
        self.robot_spec = robot_spec
        self.w_max = robot_spec.get('w_max', 0.5)  # Maximum angular velocity
        self.kp = kp  # P gain for smooth rotation

    def solve_control_problem(self,
                            robot_state: np.ndarray,
                            current_yaw: float,
                            u: np.ndarray) -> float:
        """
        Determine rotation direction based on sensing footprints using P control.
        
        Parameters
        ----------
        robot_state : numpy.ndarray
            Current robot state
        current_yaw : float
            Current yaw angle
        u : numpy.ndarray
            Position control input (ignored)
            
        Returns
        -------
        float
            Angular velocity control input
        """
        # Get robot position
        robot_pos = robot_state[:2].flatten()
        
        # Get current sensing footprints
        sensing_footprints = self.robot.sensing_footprints
        
        # If no sensing footprints yet, rotate right
        if sensing_footprints.is_empty:
            return np.array([self.w_max]).reshape(-1, 1)
            
        # Get the boundary of sensing footprints
        boundary = sensing_footprints.boundary
        
        # Sample points on the boundary
        if boundary.geom_type == 'LineString':
            boundary_points = list(boundary.coords)
        else:
            # For MultiLineString, get all points
            boundary_points = []
            for line in boundary.geoms:
                boundary_points.extend(list(line.coords))
                
        # Convert to numpy array for easier computation
        boundary_points = np.array(boundary_points)
        
        # Calculate vectors from robot to boundary points
        vectors = boundary_points - robot_pos
        
        # Calculate angles to boundary points
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # Normalize angles relative to current yaw
        angle_diffs = angle_normalize(angles - current_yaw)
        
        # Find the largest gap in visibility
        # Sort angles and find the largest gap
        sorted_angles = np.sort(angle_diffs)
        angle_gaps = np.diff(sorted_angles)
        largest_gap_idx = np.argmax(angle_gaps)
        
        # The direction to rotate is towards the middle of the largest gap
        target_angle = (sorted_angles[largest_gap_idx] + sorted_angles[largest_gap_idx + 1]) / 2
        
        # Use P control to compute control input
        error = angle_normalize(target_angle - current_yaw)
        u_att = self.kp * error
        
        # Clip to maximum angular velocity
        u_att = np.clip(u_att, -self.w_max, self.w_max)
        
        return np.array([u_att]).reshape(-1, 1) 