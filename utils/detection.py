import numpy as np
from shapely.geometry import LineString, Point


def _angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


def _normalize_unknown_obs(unknown_obs):
    if unknown_obs is None or len(unknown_obs) == 0:
        return None
    unknown_obs = np.asarray(unknown_obs, dtype=float)
    if unknown_obs.ndim == 1:
        unknown_obs = unknown_obs.reshape(1, -1)
    return unknown_obs


def _find_extreme_points(detected_points, robot_pos, robot_yaw):
    points = np.array(detected_points, dtype=float)
    vectors_to_points = points - robot_pos
    angles = np.arctan2(vectors_to_points[:, 1], vectors_to_points[:, 0]) - robot_yaw
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    leftmost_point = points[np.argmin(angles)]
    rightmost_point = points[np.argmax(angles)]
    return leftmost_point, rightmost_point


def _circle_intersects_fov(robot_pos, robot_yaw, fov_angle, cam_range, center, radius):
    to_center = np.asarray(center, dtype=float) - robot_pos
    dist = np.linalg.norm(to_center)
    if dist <= radius:
        return True
    if dist - radius > cam_range:
        return False

    angle_to_center = np.arctan2(to_center[1], to_center[0])
    angle_diff = abs(_angle_normalize(angle_to_center - robot_yaw))
    if angle_diff <= fov_angle / 2:
        return True

    angular_radius = np.arcsin(np.clip(radius / max(dist, 1e-9), 0.0, 1.0))
    return angle_diff <= (fov_angle / 2 + angular_radius)


def detect_unknown_obs_fov(robot, unknown_obs):
    unknown_obs = _normalize_unknown_obs(unknown_obs)
    if unknown_obs is None:
        return None, [], []

    robot_pos = robot.get_position()
    robot_yaw = robot.get_orientation()
    sorted_unknown_obs = sorted(
        unknown_obs, key=lambda obs: np.linalg.norm(np.array(obs[0:2]) - robot_pos)
    )

    detected_obs = []
    detected_points = []
    for obs in sorted_unknown_obs:
        if obs.shape[0] < 3:
            continue

        padded = np.zeros(7)
        padded[: min(obs.shape[0], 7)] = obs[: min(obs.shape[0], 7)]

        # Once detected, unknown obstacles are approximated as circular obstacles.
        if padded[6] >= 0.5:
            padded[2] = max(padded[2], padded[3], 0.0)
            padded[3:6] = 0.0
            padded[6] = 0.0

        if _circle_intersects_fov(
            robot_pos,
            robot_yaw,
            robot.fov_angle,
            robot.cam_range,
            padded[0:2],
            padded[2],
        ):
            detected_obs.append(padded)
            detected_points.append((padded[0], padded[1]))

    if len(detected_obs) == 0:
        return None, [], []

    detected_obs = np.array(detected_obs)
    # Keep first one for backward-compatible single-obstacle rendering.
    return detected_obs[0], detected_points, detected_obs


def detect_unknown_obs_ray(robot, unknown_obs, obs_margin=0.05):
    unknown_obs = _normalize_unknown_obs(unknown_obs)
    if unknown_obs is None:
        return None, [], []

    detected_points = []
    robot_pos = robot.get_position()
    robot_yaw = robot.get_orientation()
    robot_pos_point = Point(robot.X[0, 0], robot.X[1, 0])

    sorted_unknown_obs = sorted(
        unknown_obs, key=lambda obs: np.linalg.norm(np.array(obs[0:2]) - robot_pos)
    )
    for obs in sorted_unknown_obs:
        obs_radius = max(float(obs[2]) - obs_margin, 1e-3)
        obs_circle = Point(obs[0], obs[1]).buffer(obs_radius)
        intersected_area = robot.sensing_footprints.intersection(obs_circle)

        points = []
        if intersected_area.geom_type == 'Polygon':
            points.extend(intersected_area.exterior.coords)
        elif intersected_area.geom_type == 'MultiPolygon':
            for poly in intersected_area.geoms:
                points.extend(poly.exterior.coords)

        for point in points:
            line_to_point = LineString([robot_pos_point, Point(point)])
            # Legacy logic: front-side check via ray crossing.
            if not line_to_point.crosses(obs_circle):
                detected_points.append(point)

        if len(detected_points) > 0:
            break

    if len(detected_points) == 0:
        return None, [], []

    leftmost_point, rightmost_point = _find_extreme_points(detected_points, robot_pos, robot_yaw)
    center = (leftmost_point + rightmost_point) / 2
    radius = np.linalg.norm(rightmost_point - leftmost_point) / 2

    detected_obs = np.array([center[0], center[1], radius, 0.0, 0.0, 0.0, 0.0])
    return detected_obs, detected_points, detected_obs


def detect_unknown_obs(robot, unknown_obs, detection_mode='fov', obs_margin=0.05):
    if detection_mode == 'fov':
        return detect_unknown_obs_fov(robot, unknown_obs)
    if detection_mode == 'ray':
        return detect_unknown_obs_ray(robot, unknown_obs, obs_margin=obs_margin)
    raise ValueError(f"Unsupported unknown_obs_detection mode: {detection_mode}")
