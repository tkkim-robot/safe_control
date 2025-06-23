import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

"""
Created on June 23rd, 2025
@author: Taekyung Kim

@description: 
This code implements an exploration-based yaw controller that uses ray casting and 
area analysis to find the optimal direction for maximum information gain. 

@approach:
1. Cast rays in multiple directions from robot position
2. For each direction, calculate potential new visible area if robot faces that way
3. Use weighted scoring based on distance to unknown areas and potential information gain
4. Apply P control to smoothly rotate towards optimal direction

@note: 
- Uses sensing footprints to determine unexplored areas
- Can be used as a nominal attitude controller for the gatekeeper
"""

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class VisibilityRayCastAtt:
    """
    Exploration-based attitude controller that rotates towards directions with
    maximum potential for new area discovery using ray casting and area analysis.
    """
    def __init__(self, robot, robot_spec, kp=2.0, num_rays=36, max_ray_length=5.0):
        self.robot = robot
        self.robot_spec = robot_spec
        self.w_max = robot_spec.get('w_max', 0.5)  # Maximum angular velocity
        self.kp = kp  # P gain for smooth rotation
        
        # Ray casting parameters
        self.num_rays = num_rays  # Number of rays to cast (every 10 degrees for 36)
        
        # Get FOV parameters from robot
        self.fov_angle = getattr(robot, 'fov_angle', np.deg2rad(70.0))
        self.cam_range = getattr(robot, 'cam_range', 3.0)
        self.max_ray_length = max(max_ray_length, self.cam_range)

    def cast_exploration_rays(self, robot_pos, sensing_footprints):
        """
        Cast rays in all directions and calculate exploration potential.
        
        Parameters
        ----------
        robot_pos : numpy.ndarray
            Current robot position [x, y]
        sensing_footprints : shapely.geometry
            Current mapped area
            
        Returns
        -------
        tuple
            (ray_angles, exploration_scores)
        """
        ray_angles = np.linspace(0, 2*np.pi, self.num_rays, endpoint=False)
        exploration_scores = []
        
        for angle in ray_angles:
            # Create ray direction
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            ray_end = robot_pos + self.max_ray_length * ray_dir
            ray = LineString([robot_pos, ray_end])
            
            # Calculate score based on multiple factors
            score = self._calculate_exploration_score(robot_pos, angle, ray, sensing_footprints)
            exploration_scores.append(score)
            
        return ray_angles, np.array(exploration_scores)
    
    def _calculate_exploration_score(self, robot_pos, angle, ray, sensing_footprints):
        """
        Calculate exploration score for a given direction.
        
        Parameters
        ----------
        robot_pos : numpy.ndarray
            Current robot position
        angle : float
            Ray angle
        ray : shapely.geometry.LineString
            Ray geometry
        sensing_footprints : shapely.geometry
            Current mapped area
            
        Returns
        -------
        float
            Exploration score (higher is better)
        """
        score = 0.0
        
        # Factor 1: Distance to mapped boundary in this direction
        boundary_distance = self._calculate_boundary_distance(robot_pos, angle, sensing_footprints)
        
        # Factor 2: Potential new FOV area if facing this direction
        potential_area = self._calculate_potential_fov_area(robot_pos, angle, sensing_footprints)
        
        # Factor 3: Ray intersection analysis
        ray_score = self._calculate_ray_intersection_score(ray, sensing_footprints)
        
        # Combine factors with weights
        score = (
            0.3 * (1.0 / (boundary_distance + 0.1)) +  # Closer to boundary is better
            0.5 * potential_area +                      # More potential area is better  
            0.2 * ray_score                             # Ray extending beyond map is better
        )
        
        return score
    
    def _calculate_boundary_distance(self, robot_pos, angle, sensing_footprints):
        """Calculate distance to sensing footprints boundary in given direction."""
        if sensing_footprints.is_empty:
            return 0.1  # Small value if no map yet
            
        # Create a ray in the given direction
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        ray_end = robot_pos + self.max_ray_length * ray_dir
        ray = LineString([robot_pos, ray_end])
        
        # Find intersection with boundary
        try:
            intersection = ray.intersection(sensing_footprints.boundary)
            
            if intersection.is_empty:
                return self.max_ray_length  # Ray stays inside mapped area
                
            # Get closest intersection point
            if intersection.geom_type == 'Point':
                intersect_point = np.array(intersection.coords[0])
            elif intersection.geom_type == 'MultiPoint':
                # Multiple intersections, find closest
                distances = []
                for geom in intersection.geoms:
                    pt = np.array(geom.coords[0])
                    distances.append(np.linalg.norm(pt - robot_pos))
                min_idx = np.argmin(distances)
                intersect_point = np.array(list(intersection.geoms)[min_idx].coords[0])
            else:
                # LineString or other geometry - use centroid
                intersect_point = np.array(intersection.centroid.coords[0])
                
            return np.linalg.norm(intersect_point - robot_pos)
            
        except Exception:
            return self.max_ray_length
    
    def _calculate_potential_fov_area(self, robot_pos, target_angle, sensing_footprints):
        """
        Calculate potential new visible area if robot faces target_angle direction.
        
        Parameters
        ----------
        robot_pos : numpy.ndarray
            Current robot position
        target_angle : float
            Target facing angle
        sensing_footprints : shapely.geometry
            Current mapped area
            
        Returns
        -------
        float
            Normalized potential new area (0 to 1)
        """
        if sensing_footprints.is_empty:
            return 1.0  # Maximum potential if no map yet
            
        # Create potential FOV triangle if robot faces target_angle
        angle_left = target_angle - self.fov_angle / 2
        angle_right = target_angle + self.fov_angle / 2
        
        fov_left = robot_pos + self.cam_range * np.array([np.cos(angle_left), np.sin(angle_left)])
        fov_right = robot_pos + self.cam_range * np.array([np.cos(angle_right), np.sin(angle_right)])
        
        potential_fov = Polygon([robot_pos, fov_left, fov_right])
        
        # Calculate new area that would be visible
        try:
            new_area = potential_fov.difference(sensing_footprints).area
            total_potential_area = potential_fov.area
            
            if total_potential_area > 0:
                return new_area / total_potential_area
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _calculate_ray_intersection_score(self, ray, sensing_footprints):
        """
        Calculate score based on how ray intersects with sensing footprints.
        Higher score for rays that extend beyond the mapped area.
        """
        if sensing_footprints.is_empty:
            return 1.0
            
        try:
            # Check if ray extends beyond mapped area
            intersection = ray.intersection(sensing_footprints)
            
            if intersection.is_empty:
                # Ray is completely outside mapped area
                return 1.0
            elif intersection.length < ray.length * 0.8:
                # Ray extends significantly beyond mapped area
                return 0.8
            else:
                # Ray mostly within mapped area
                return 0.2
        except Exception:
            return 0.5
    
    def find_optimal_direction(self, robot_pos, current_yaw, sensing_footprints):
        """
        Find the optimal direction for exploration.
        
        Parameters
        ----------
        robot_pos : numpy.ndarray
            Current robot position
        current_yaw : float
            Current yaw angle
        sensing_footprints : shapely.geometry
            Current mapped area
            
        Returns
        -------
        float
            Target angle for optimal exploration
        """
        # Cast rays and get exploration scores
        ray_angles, scores = self.cast_exploration_rays(robot_pos, sensing_footprints)
        
        # Apply angular distance penalty (prefer directions closer to current heading)
        angular_distances = np.abs(angle_normalize(ray_angles - current_yaw))
        distance_penalty = angular_distances / np.pi  # Normalize to [0, 1]
        
        # Combined score with angular distance consideration
        combined_scores = scores * (1 - 0.3 * distance_penalty)
        
        # Find best direction
        best_idx = np.argmax(combined_scores)
        target_angle = ray_angles[best_idx]
        
        return target_angle

    def solve_control_problem(self,
                            robot_state: np.ndarray,
                            current_yaw: float,
                            u: np.ndarray) -> np.ndarray:
        """
        Determine rotation direction for optimal exploration using P control.
        
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
        numpy.ndarray
            Angular velocity control input
        """
        # Get robot position
        robot_pos = robot_state[:2].flatten()
        
        # Get current sensing footprints
        sensing_footprints = self.robot.sensing_footprints
        
        # If no sensing footprints yet, rotate at maximum speed
        if sensing_footprints.is_empty:
            return np.array([self.w_max]).reshape(-1, 1)
        
        # Find optimal exploration direction
        target_angle = self.find_optimal_direction(robot_pos, current_yaw, sensing_footprints)
        
        # Calculate angular error
        angular_error = angle_normalize(target_angle - current_yaw)
        
        # Apply P control
        u_att = self.kp * angular_error
        
        # Clip to maximum angular velocity
        u_att = np.clip(u_att, -self.w_max, self.w_max)
        
        return np.array([u_att]).reshape(-1, 1) 