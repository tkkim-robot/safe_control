"""
Official BarrierNet style feature extraction with multi-obstacle support.

This module provides feature extraction following the Official BarrierNet repo structure,
but extended to support multiple obstacles.

Feature format: [px, py, theta, v, dst, obs1_dx, obs1_dy, obs1_r, ..., obsK_dx, obsK_dy, obsK_r]
where:
    - px, py, theta, v, dst: robot features (Official style)
    - obsK_dx, obsK_dy, obsK_r: obstacle K features (relative to robot position)
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def extract_features_official_style(
    robot_state: np.ndarray,
    goal: np.ndarray,
    obstacles: Optional[np.ndarray],
    k_obs: int = 5,
    robot_radius: float = 0.25,
) -> np.ndarray:
    """
    Extract features in Official BarrierNet style with multi-obstacle support.
    
    Args:
        robot_state: Robot state vector [x, y, theta, v, ...] (robot model dependent)
            For DynamicUnicycle2D: [x, y, theta, v]
            For Quad2D: [x, z, theta, vx, vz, ...]
            For Quad3D: [x, y, z, ..., vx, vy, vz, ...]
        goal: Goal position [gx, gy, ...] (robot model dependent)
        obstacles: Array of obstacles, each row is [x, y, r, ...] or None
            Can be (N, 3), (N, 5), or (N, 7) format
        k_obs: Number of obstacles to include in features (default 5)
        robot_radius: Robot radius for clearance calculation
    
    Returns:
        features: np.ndarray of shape (5 + 3*k_obs,)
            [px, py, theta, v, dst, obs1_x, obs1_y, obs1_r, ..., obsK_x, obsK_y, obsK_r]
            where:
                - px, py: robot position (absolute)
                - theta: robot heading
                - v: robot velocity scalar
                - dst: distance to goal
                - obsK_x, obsK_y: obstacle K absolute position (for barrier calculation compatibility with Official repo)
                - obsK_r: obstacle K radius
            
        Note: Obstacle positions are stored as absolute coordinates (like Official repo's hardcoded obs_x, obs_y),
        but extracted from the obstacle array instead of being hardcoded.
    """
    # Extract robot position and heading (model-dependent)
    px = float(robot_state[0])
    py = float(robot_state[1])
    theta = float(robot_state[2])
    
    # Extract velocity scalar (model-dependent)
    if len(robot_state) >= 4:
        # 2D ground vehicles: scalar velocity at index 3
        v = float(robot_state[3])
    elif len(robot_state) >= 5:
        # Quad2D: [x, z, theta, vx, vz] -> v = sqrt(vx^2 + vz^2)
        vx = float(robot_state[3])
        vz = float(robot_state[4])
        v = float(np.hypot(vx, vz))
    else:
        v = 0.0
    
    # Goal distance
    gx = float(goal[0])
    if len(goal) > 1:
        gy = float(goal[1])
    else:
        gy = 0.0
    
    # Calculate distance to goal (model-dependent)
    if len(robot_state) >= 3 and len(goal) >= 2:
        if len(goal) >= 3 and len(robot_state) >= 3:
            # 3D case
            gz = float(goal[2])
            z = float(robot_state[2]) if len(robot_state) > 2 else 0.0
            dst = float(np.sqrt((gx - px)**2 + (gy - py)**2 + (gz - z)**2))
        else:
            # 2D case
            dst = float(np.hypot(gx - px, gy - py))
    else:
        dst = 0.0
    
    # Robot features (Official style)
    features = [px, py, theta, v, dst]
    
    # Obstacle features (relative to robot)
    if obstacles is None or (isinstance(obstacles, np.ndarray) and obstacles.size == 0):
        # No obstacles: pad with dummy obstacles (far away, zero radius)
        # Use absolute coordinates (like Official repo's hardcoded obs_x=40, obs_y=15)
        # For padding, use coordinates far enough that barrier will be positive (safe)
        far_x, far_y = px + 100.0, py + 100.0  # far from current robot position
        for _ in range(k_obs):
            features.extend([far_x, far_y, 0.0])  # absolute position, zero radius
    else:
        # Convert to numpy array if needed
        if not isinstance(obstacles, np.ndarray):
            obstacles = np.array(obstacles, dtype=np.float64)
        
        if obstacles.ndim == 1:
            obstacles = obstacles.reshape(1, -1)
        
        # Select closest k_obs obstacles by clearance (signed distance)
        obs_list = []
        for obs in obstacles:
            ox = float(obs[0])  # obstacle absolute x
            oy = float(obs[1])  # obstacle absolute y
            or_r = float(obs[2]) if len(obs) > 2 else 0.2  # default radius
            
            # Calculate clearance (signed distance) for sorting
            dist = float(np.hypot(ox - px, oy - py))
            clearance = dist - (or_r + robot_radius)
            
            # Store absolute obstacle position (like Official repo's hardcoded obs_x, obs_y)
            obs_list.append((clearance, ox, oy, or_r))
        
        # Sort by clearance (closest first)
        obs_list.sort(key=lambda x: x[0])
        
        # Add closest k_obs obstacles with absolute positions
        for i in range(min(k_obs, len(obs_list))):
            _, ox, oy, or_r = obs_list[i]
            features.extend([ox, oy, or_r])  # absolute position (Official style)
        
        # Pad if needed (fewer than k_obs obstacles)
        far_x, far_y = px + 100.0, py + 100.0
        while len(features) < 5 + 3 * k_obs:
            features.extend([far_x, far_y, 0.0])  # absolute position
    
    return np.array(features, dtype=np.float64)


def extract_features_for_robot_model(
    robot_model: str,
    robot_state: np.ndarray,
    goal: np.ndarray,
    obstacles: Optional[np.ndarray],
    k_obs: int = 5,
    robot_radius: float = 0.25,
) -> np.ndarray:
    """
    Robot-model-specific feature extraction wrapper.
    
    Handles robot-specific differences (Quad2D uses z instead of y, etc.)
    """
    # For now, use the general function (can be extended for robot-specific handling)
    return extract_features_official_style(
        robot_state=robot_state,
        goal=goal,
        obstacles=obstacles,
        k_obs=k_obs,
        robot_radius=robot_radius,
    )


# Test function for validation
if __name__ == "__main__":
    # Test DynamicUnicycle2D
    robot_state = np.array([2.5, 2.0, 0.1, 0.5], dtype=np.float64)
    goal = np.array([9.5, 2.0], dtype=np.float64)
    obstacles = np.array([
        [4.0, 2.5, 0.3],
        [6.0, 1.5, 0.4],
        [7.0, 2.0, 0.2],
    ], dtype=np.float64)
    
    features = extract_features_official_style(
        robot_state=robot_state,
        goal=goal,
        obstacles=obstacles,
        k_obs=5,
        robot_radius=0.25,
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Expected shape: (5 + 3*5,) = (20,)")
    print(f"Features: {features}")
    print(f"Robot features: {features[:5]}")
    print(f"Obstacle features (first): {features[5:8]}")
    print(f"Obstacle features (second): {features[8:11]}")

