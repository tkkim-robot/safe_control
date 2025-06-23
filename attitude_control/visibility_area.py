import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

"""
Created on June 23rd, 2025
@author: Taekyung Kim

@description: 
This code implements a visibility-promoting yaw controller based on maximising expected newly
observable area inside the camera field-of-view (FoV). The controller samples candidate headings,
evaluates the incremental unexplored area each heading would reveal, and steers toward the heading
that maximises this metric.

@approach:
1. Sample candidate yaw headings uniformly across the full range
2. For each candidate, construct a FoV sector polygon
3. Calculate the unexplored area within each sector
4. Select the heading with maximum prospective information gain
5. Apply P control to smoothly rotate towards optimal direction

@note: 
- Uses sensing footprints to determine unexplored areas
- More computationally efficient than ray casting approaches (no ray casting, compute area as triangle)
- Can be used as a nominal attitude controller for the gatekeeper
"""

def angle_normalize(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def build_fov_sector(origin: np.ndarray,
                     yaw: float,
                     fov_angle: float,
                     radius: float,
                     resolution: int = 30) -> Polygon:
    """
    Construct a polygon representing the robot's field-of-view sector.
    """
    half = fov_angle / 2.0
    arc_angles = np.linspace(yaw - half, yaw + half, resolution)
    points = [(origin[0] + radius * np.cos(a),
               origin[1] + radius * np.sin(a)) for a in arc_angles]
    # prepend the apex
    points.insert(0, (origin[0], origin[1]))
    return Polygon(points)


class VisibilityAreaAtt:
    """
    Visibility-promoting yaw controller based on maximising expected newly
    observable area inside the camera field-of-view (FoV).
    """

    def __init__(self,
                 robot,
                 robot_spec: dict,
                 kp: float = 1.5,
                 n_yaw_samples: int = 36,
                 arc_resolution: int = 30):
        self.robot = robot
        self.robot_spec = robot_spec

        # Control parameters
        self.kp = kp
        self.w_max = float(robot_spec.get('w_max', 0.5))

        # Sensor model
        self.fov_angle = float(getattr(robot, 'fov_angle',
                                       np.deg2rad(robot_spec.get('fov_angle',
                                                                 70.0))))
        self.cam_range = float(getattr(robot, 'cam_range',
                                       robot_spec.get('cam_range', 3.0)))

        # Sampling parameters
        self.n_yaw_samples = int(n_yaw_samples)
        self.arc_resolution = int(arc_resolution)

    # --------------------------------------------------------------------- #
    # Public API – identical to legacy controller
    # --------------------------------------------------------------------- #
    def solve_control_problem(self,
                              robot_state: np.ndarray,
                              current_yaw: float,
                              u: np.ndarray) -> np.ndarray:
        """
        Compute angular velocity command for optimal exploration using area-based approach.

        Parameters
        ----------
        robot_state : np.ndarray
            Full robot state; only the first two entries (x, y) are used.
        current_yaw : float
            Current heading [rad].
        u : np.ndarray
            Positional control input (ignored – kept for drop-in compatibility).

        Returns
        -------
        np.ndarray, shape (1, 1)
            Angular velocity command [rad/s].
        """
        # Fast path: if the map is empty, simply rotate in place.
        if self.robot.sensing_footprints.is_empty:
            return np.array([[self.w_max]])

        pos = robot_state[:2].flatten()
        footprints = self.robot.sensing_footprints

        # Pre-compute union with a small buffer to avoid topological artefacts
        footprints = footprints.buffer(0.0)

        # Generate candidate headings (absolute angles)
        candidate_yaws = np.linspace(-np.pi, np.pi,
                                     self.n_yaw_samples, endpoint=False)

        unexplored_areas = []
        for yaw in candidate_yaws:
            sector = build_fov_sector(pos, yaw,
                                      self.fov_angle,
                                      self.cam_range,
                                      resolution=self.arc_resolution)
            # Newly observable = FoV sector \ already-seen region
            new_area = sector.difference(footprints)
            unexplored_areas.append(new_area.area)

        # Select heading with maximum prospective gain
        best_idx = int(np.argmax(unexplored_areas))
        target_yaw = candidate_yaws[best_idx]

        # P-controller (wrapped error)
        yaw_error = angle_normalize(target_yaw - current_yaw)
        w_cmd = self.kp * yaw_error
        w_cmd = np.clip(w_cmd, -self.w_max, self.w_max)

        return np.array([[w_cmd]])