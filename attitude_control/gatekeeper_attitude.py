import numpy as np
from shapely.geometry import Point, LineString

from .simple_attitude import SimpleAtt
from .velocity_tracking_yaw import VelocityTrackingYaw
from .visibility_area import VisibilityAreaAtt, build_fov_sector


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class GatekeeperAtt:
    """
    Gatekeeper-based attitude controller.

    Core idea:
    1. Generate candidate yaw trajectory = nominal segment + backup segment.
    2. Validate candidate with critical-point visibility condition.
    3. Commit the longest valid nominal segment; otherwise keep previous committed backup.
    """

    def __init__(
        self,
        robot,
        robot_spec: dict,
        dt: float = 0.05,
        ctrl_config: dict = None,
        nominal_horizon: float = 1.0,
        backup_horizon: float = 2.0,
        event_offset: float = 0.5,
    ):
        self.robot = robot
        self.robot_spec = robot_spec
        self.dt = float(dt)

        self.nominal_horizon = float(
            robot_spec.get("gatekeeper_nominal_horizon", nominal_horizon)
        )
        self.backup_horizon = float(
            robot_spec.get("gatekeeper_backup_horizon", backup_horizon)
        )
        self.event_offset = float(
            robot_spec.get("gatekeeper_event_offset", event_offset)
        )
        self.horizon_discount = float(
            robot_spec.get("gatekeeper_horizon_discount", max(5.0 * self.dt, self.dt))
        )

        self.next_event_time = 0.0
        self.current_time_idx = int(np.ceil(self.backup_horizon / self.dt))
        self.committed_horizon = 0.0
        self.actual_nominal_steps = 0

        self.committed_x_traj = None
        self.committed_u_traj = None
        self.candidate_x_traj = None
        self.candidate_u_traj = None
        self.pos_committed_x_traj = None
        self.pos_committed_u_traj = None
        self.total_replan_events = 0
        self.accepted_replan_events = 0
        self.rejected_replan_events = 0
        self.nominal_commit_events = 0
        self.nominal_committed_steps = 0
        self.max_nominal_committed_steps = 0

        # Monitor tuning: conservative but tunable.
        self.validation_slack = float(
            robot_spec.get("gatekeeper_validation_slack", 0.05)
        )
        self.braking_distance_scale = float(
            robot_spec.get("gatekeeper_braking_distance_scale", 1.0)
        )
        self.braking_distance_margin = float(
            robot_spec.get(
                "gatekeeper_braking_distance_margin",
                float(self.robot.robot_radius) + 0.10,
            )
        )
        self.max_yaw_rate = float(robot_spec.get("w_max", 0.5))

        ctrl_cfg = {} if ctrl_config is None else dict(ctrl_config)
        nominal_key = ctrl_cfg.get(
            "nominal",
            robot_spec.get("gatekeeper_nominal", "visibility_area"),
        )
        backup_key = ctrl_cfg.get(
            "backup",
            robot_spec.get("gatekeeper_backup", "velocity_tracking_yaw"),
        )

        self._ctrl_map = {
            "simple": SimpleAtt,
            "velocity tracking yaw": VelocityTrackingYaw,
            "velocity_tracking_yaw": VelocityTrackingYaw,
            "visibility area": VisibilityAreaAtt,
            "visibility_area": VisibilityAreaAtt,
        }

        nominal_name = self._normalize_ctrl_name(nominal_key)
        backup_name = self._normalize_ctrl_name(backup_key)
        if nominal_name not in self._ctrl_map:
            raise ValueError(f"Unknown gatekeeper nominal controller '{nominal_key}'")
        if backup_name not in self._ctrl_map:
            raise ValueError(f"Unknown gatekeeper backup controller '{backup_key}'")

        self.nominal_ctrl = self._ctrl_map[nominal_name](robot, robot_spec)
        self.backup_ctrl = self._ctrl_map[backup_name](robot, robot_spec)
        self.nominal_controller = self.nominal_ctrl.solve_control_problem
        self.backup_controller = self.backup_ctrl.solve_control_problem

    @staticmethod
    def _normalize_ctrl_name(name):
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    def setup_pos_controller(self, pos_controller):
        self.pos_controller = pos_controller

    def _dynamics(self, x, u):
        x_col = np.asarray(x, dtype=float).reshape(-1, 1)
        u_col = np.asarray(u, dtype=float).reshape(-1, 1)
        dx = self.robot.robot.f(x_col) + self.robot.robot.g(x_col) @ u_col
        return np.asarray(dx, dtype=float).reshape(-1, 1)

    def _update_pos_committed_trajectory(self, robot_state):
        """
        Pull positional prediction from MPC and extend it to cover
        nominal + backup horizon.
        """
        required_control_steps = int(
            np.ceil((self.nominal_horizon + self.backup_horizon) / self.dt)
        )
        required_state_steps = required_control_steps + 1

        x_traj = None
        u_traj = None
        try:
            x_traj_casadi = self.pos_controller.mpc.opt_x_num["_x", :, 0, 0]
            u_traj_casadi = self.pos_controller.mpc.opt_x_num["_u", :, 0]

            x_np = []
            for x_dm in x_traj_casadi:
                x_np.append(np.asarray(x_dm.full(), dtype=float).reshape(-1))
            u_np = []
            for u_dm in u_traj_casadi:
                u_np.append(np.asarray(u_dm.full(), dtype=float).reshape(-1))

            if len(x_np) > 0:
                x_traj = np.vstack(x_np)
            if len(u_np) > 0:
                u_traj = np.vstack(u_np)
        except Exception:
            x_traj = None
            u_traj = None

        if x_traj is None or x_traj.size == 0:
            # Fallback: constant-velocity roll-out from current state.
            x0 = np.asarray(robot_state, dtype=float).reshape(-1)
            state_dim = x0.size
            control_dim = 2
            x_traj = np.zeros((required_state_steps, state_dim), dtype=float)
            u_traj = np.zeros((required_control_steps, control_dim), dtype=float)
            x_traj[0] = x0
            for i in range(1, required_state_steps):
                x_traj[i] = x_traj[i - 1]
                if state_dim >= 4:
                    x_traj[i, 0] += x_traj[i - 1, 2] * self.dt
                    x_traj[i, 1] += x_traj[i - 1, 3] * self.dt
            self.pos_committed_x_traj = x_traj
            self.pos_committed_u_traj = u_traj
            return

        if u_traj is None or u_traj.size == 0:
            u_traj = np.zeros((max(x_traj.shape[0] - 1, 1), 2), dtype=float)

        # Ensure state/control length consistency: n_state = n_control + 1.
        if x_traj.shape[0] == u_traj.shape[0]:
            last_x = x_traj[-1].reshape(-1, 1)
            last_u = u_traj[-1].reshape(-1, 1)
            next_x = (last_x + self.dt * self._dynamics(last_x, last_u)).reshape(-1)
            x_traj = np.vstack((x_traj, next_x))
        elif x_traj.shape[0] < u_traj.shape[0] + 1:
            u_traj = u_traj[: max(x_traj.shape[0] - 1, 0)]
        elif x_traj.shape[0] > u_traj.shape[0] + 1:
            x_traj = x_traj[: u_traj.shape[0] + 1]

        # Extend prediction if needed.
        while u_traj.shape[0] < required_control_steps:
            if u_traj.shape[0] > 0:
                last_u = np.zeros_like(u_traj[-1]).reshape(-1, 1)
            else:
                last_u = np.zeros((2, 1))
            last_x = x_traj[-1].reshape(-1, 1)
            next_x = (last_x + self.dt * self._dynamics(last_x, last_u)).reshape(-1)
            x_traj = np.vstack((x_traj, next_x))
            u_traj = np.vstack((u_traj, last_u.reshape(1, -1)))

        if x_traj.shape[0] < required_state_steps:
            missing = required_state_steps - x_traj.shape[0]
            for _ in range(missing):
                last_u = np.zeros((u_traj.shape[1], 1))
                last_x = x_traj[-1].reshape(-1, 1)
                next_x = (last_x + self.dt * self._dynamics(last_x, last_u)).reshape(-1)
                x_traj = np.vstack((x_traj, next_x))

        self.pos_committed_x_traj = x_traj[:required_state_steps]
        self.pos_committed_u_traj = u_traj[:required_control_steps]

    def _generate_trajectory(self, initial_yaw, n_steps, controller, start_step=0):
        """
        Roll out yaw dynamics for n_steps controls:
          yaw_{k+1} = yaw_k + dt * u_att_k
        """
        n_steps = max(int(n_steps), 0)
        x_traj = np.zeros((n_steps + 1,), dtype=float)
        u_traj = np.zeros((n_steps,), dtype=float)
        x_traj[0] = angle_normalize(float(initial_yaw))

        if self.pos_committed_x_traj is None or self.pos_committed_u_traj is None:
            return x_traj, u_traj

        yaw = x_traj[0]
        for k in range(n_steps):
            pos_idx = min(start_step + k, self.pos_committed_x_traj.shape[0] - 1)
            if self.pos_committed_u_traj.shape[0] > 0:
                u_idx = min(start_step + k, self.pos_committed_u_traj.shape[0] - 1)
                pos_u = self.pos_committed_u_traj[u_idx].reshape(-1, 1)
            else:
                pos_u = np.zeros((2, 1))
            pos_x = self.pos_committed_x_traj[pos_idx].reshape(-1, 1)

            raw = controller(pos_x, yaw, pos_u)
            yaw_rate = float(np.asarray(raw, dtype=float).reshape(-1)[0])
            yaw_rate = float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))

            u_traj[k] = yaw_rate
            yaw = angle_normalize(yaw + yaw_rate * self.dt)
            x_traj[k + 1] = yaw

        return x_traj, u_traj

    def _generate_candidate_trajectory(self, nominal_horizon_steps, current_yaw):
        nominal_horizon_steps = max(int(nominal_horizon_steps), 0)
        backup_steps = max(int(np.ceil(self.backup_horizon / self.dt)), 0)

        nominal_x, nominal_u = self._generate_trajectory(
            current_yaw, nominal_horizon_steps, self.nominal_controller, start_step=0
        )
        backup_x, backup_u = self._generate_trajectory(
            nominal_x[-1], backup_steps, self.backup_controller, start_step=nominal_horizon_steps
        )

        # Remove duplicate switching state.
        candidate_x = np.concatenate((nominal_x, backup_x[1:]), axis=0)
        candidate_u = np.concatenate((nominal_u, backup_u), axis=0)

        self.candidate_x_traj = candidate_x
        self.candidate_u_traj = candidate_u
        return candidate_x, candidate_u, nominal_horizon_steps

    @staticmethod
    def _segment_boundary_crossing(known_region, p0, p1, max_iter=24):
        inside0 = bool(known_region.covers(Point(float(p0[0]), float(p0[1]))))
        inside1 = bool(known_region.covers(Point(float(p1[0]), float(p1[1]))))
        if inside0 == inside1:
            return None

        lo = np.array(p0, dtype=float)
        hi = np.array(p1, dtype=float)
        ref_inside = inside0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            mid_inside = bool(known_region.covers(Point(float(mid[0]), float(mid[1]))))
            if mid_inside == ref_inside:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _compute_critical_point(self, max_state_steps):
        """
        Critical point = first boundary crossing where predicted positional
        trajectory exits the currently known sensing footprint.
        """
        if self.pos_committed_x_traj is None or self.pos_committed_x_traj.shape[0] < 2:
            return None, None
        if self.robot.sensing_footprints.is_empty:
            return None, None

        known_region = self.robot.sensing_footprints.buffer(0.0)
        if known_region.is_empty:
            return None, None

        path = self.pos_committed_x_traj[
            : min(max_state_steps, self.pos_committed_x_traj.shape[0]), :2
        ]
        if path.shape[0] < 2:
            return None, None

        first_inside = bool(known_region.covers(Point(float(path[0, 0]), float(path[0, 1]))))
        if not first_inside:
            return np.array(path[0], dtype=float), 0

        prev = path[0]
        prev_inside = first_inside
        for i in range(1, path.shape[0]):
            cur = path[i]
            cur_inside = bool(known_region.covers(Point(float(cur[0]), float(cur[1]))))
            if prev_inside and cur_inside:
                prev = cur
                prev_inside = cur_inside
                continue

            cp = self._segment_boundary_crossing(known_region, prev, cur)
            if cp is None:
                cp = np.array(cur if not cur_inside else prev, dtype=float)
            return cp, i

        return None, None

    def _max_braking_distance(self):
        model = self.robot_spec.get("model", "")
        if model == "DoubleIntegrator2D":
            v_max = float(
                self.robot_spec.get(
                    "v_max",
                    max(np.linalg.norm(np.asarray(self.robot.X[2:4, 0], dtype=float)), 1e-3),
                )
            )
            a_max = float(self.robot_spec.get("a_max", 1.0))
            if a_max <= 1e-6:
                return np.inf
            base = (v_max * v_max) / (2.0 * a_max)
        elif model == "SingleIntegrator2D":
            v_max = float(self.robot_spec.get("v_max", 1.0))
            base = v_max * self.dt
        else:
            base = float(self.robot_spec.get("cam_range", 3.0)) * 0.25
        return self.braking_distance_scale * base + self.braking_distance_margin

    def _is_point_in_fov(self, robot_state, robot_yaw, point, is_in_cam_range=False):
        robot_pos = np.asarray(robot_state[:2], dtype=float).reshape(-1)
        pt = np.asarray(point, dtype=float).reshape(-1)
        to_point = pt - robot_pos

        angle_to_point = np.arctan2(to_point[1], to_point[0])
        angle_diff = abs(angle_normalize(angle_to_point - float(robot_yaw)))
        in_fov = angle_diff <= (self.robot.fov_angle / 2.0)
        if not is_in_cam_range:
            return in_fov
        return in_fov and (np.linalg.norm(to_point) <= float(self.robot.cam_range))

    def _is_candidate_valid(self, critical_point, crossing_step, candidate_x_traj):
        if self.pos_committed_x_traj is None:
            return True

        n_states = min(candidate_x_traj.shape[0], self.pos_committed_x_traj.shape[0])
        if n_states <= 0:
            return False

        # If there is no boundary crossing within the checked horizon,
        # nominal behavior is admissible.
        if critical_point is None:
            return True

        brake_dist = self._max_braking_distance()
        deadline_step = None
        if crossing_step is not None:
            deadline_step = int(max(crossing_step, 0))

        path_xy = self.pos_committed_x_traj[:n_states, :2]

        # 1) Global critical-point check (boundary exit point).
        for i in range(n_states):
            pos_i = path_xy[i]
            if np.linalg.norm(pos_i - critical_point) <= (brake_dist + self.validation_slack):
                deadline_step = i if deadline_step is None else min(deadline_step, i)
                break

        seen_global = False
        if deadline_step is not None:
            for i in range(n_states):
                pos = self.pos_committed_x_traj[i]
                yaw = candidate_x_traj[i]
                if not seen_global and self._is_point_in_fov(pos, yaw, critical_point, is_in_cam_range=True):
                    seen_global = True
                if i >= deadline_step and not seen_global:
                    return False

        # 2) Stepwise braking-lookahead tube check (instantaneous FoV version).
        # At each step, the braking tube to the local critical point must be
        # covered by the current FoV. This enforces timely detection.
        for i in range(n_states):
            pos_i = path_xy[i]
            yaw_i = float(candidate_x_traj[i])
            sector = build_fov_sector(
                pos_i,
                yaw_i,
                float(self.robot.fov_angle),
                float(self.robot.cam_range),
                resolution=18,
            )
            sector = sector.buffer(self.validation_slack).buffer(0.0)

            cp_i = self._critical_point_along_path(path_xy, i, brake_dist)
            if cp_i is None:
                continue
            safety_tube = LineString(
                [
                    (float(pos_i[0]), float(pos_i[1])),
                    (float(cp_i[0]), float(cp_i[1])),
                ]
            ).buffer(float(self.robot.robot_radius))
            if not sector.covers(safety_tube):
                return False

        return True

    @staticmethod
    def _critical_point_along_path(path_xy, start_idx, lookahead_dist):
        if path_xy is None or len(path_xy) == 0:
            return None
        n = len(path_xy)
        i0 = int(np.clip(start_idx, 0, n - 1))
        if i0 >= n - 1:
            return np.array(path_xy[-1], dtype=float)

        if lookahead_dist <= 1e-9:
            return np.array(path_xy[i0], dtype=float)

        remaining = float(lookahead_dist)
        p_prev = np.array(path_xy[i0], dtype=float)
        for k in range(i0 + 1, n):
            p_next = np.array(path_xy[k], dtype=float)
            seg = p_next - p_prev
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1e-9:
                p_prev = p_next
                continue
            if remaining <= seg_len:
                alpha = remaining / seg_len
                return p_prev + alpha * seg
            remaining -= seg_len
            p_prev = p_next
        return np.array(path_xy[-1], dtype=float)

    def _update_committed_trajectory(self, candidate_x, candidate_u, actual_nominal_steps):
        self.committed_x_traj = np.array(candidate_x, dtype=float).copy()
        self.committed_u_traj = np.array(candidate_u, dtype=float).copy()
        self.actual_nominal_steps = int(max(actual_nominal_steps, 0))
        self.committed_horizon = self.actual_nominal_steps * self.dt
        self.next_event_time = self.event_offset
        self.current_time_idx = 0

    def get_stats(self):
        accepted = max(self.accepted_replan_events, 1)
        return {
            "replans": int(self.total_replan_events),
            "accepted": int(self.accepted_replan_events),
            "rejected": int(self.rejected_replan_events),
            "nominal_commits": int(self.nominal_commit_events),
            "nominal_steps_total": int(self.nominal_committed_steps),
            "nominal_seconds_total": float(self.nominal_committed_steps * self.dt),
            "nominal_seconds_avg_per_commit": float(
                self.nominal_committed_steps * self.dt / accepted
            ),
            "nominal_seconds_max": float(self.max_nominal_committed_steps * self.dt),
        }

    def solve_control_problem(self, robot_state: np.ndarray, current_yaw: float, u: np.ndarray) -> float:
        """
        Gatekeeper main loop for yaw control.
        """
        robot_state = np.asarray(robot_state, dtype=float).reshape(-1)
        current_yaw = float(np.asarray(current_yaw, dtype=float).reshape(-1)[0])

        self._update_pos_committed_trajectory(robot_state)

        if self.committed_x_traj is None or self.committed_u_traj is None:
            backup_steps = max(int(np.ceil(self.backup_horizon / self.dt)), 1)
            init_x, init_u = self._generate_trajectory(
                current_yaw, backup_steps, self.backup_controller, start_step=0
            )
            self._update_committed_trajectory(init_x, init_u, actual_nominal_steps=0)
            self.next_event_time = 0.0

        if self.current_time_idx >= (self.next_event_time / self.dt):
            self.total_replan_events += 1
            max_nominal_steps = max(int(np.ceil(self.nominal_horizon / self.dt)), 0)
            backup_steps = max(int(np.ceil(self.backup_horizon / self.dt)), 0)
            discount_steps = max(int(np.ceil(self.horizon_discount / self.dt)), 1)

            critical_point, crossing_step = self._compute_critical_point(
                max_state_steps=max_nominal_steps + backup_steps + 2
            )

            found_valid = False
            for i in range(max_nominal_steps // discount_steps + 2):
                nominal_steps = max_nominal_steps - i * discount_steps
                if nominal_steps < 0:
                    nominal_steps = 0

                cand_x, cand_u, actual_nom_steps = self._generate_candidate_trajectory(
                    nominal_steps, current_yaw
                )
                if self._is_candidate_valid(critical_point, crossing_step, cand_x):
                    self.accepted_replan_events += 1
                    self.nominal_committed_steps += int(max(actual_nom_steps, 0))
                    self.max_nominal_committed_steps = max(
                        self.max_nominal_committed_steps, int(max(actual_nom_steps, 0))
                    )
                    if int(max(actual_nom_steps, 0)) > 0:
                        self.nominal_commit_events += 1
                    self._update_committed_trajectory(cand_x, cand_u, actual_nom_steps)
                    found_valid = True
                    break

            if not found_valid:
                self.rejected_replan_events += 1
                # Keep current committed trajectory (safe fallback), retry later.
                self.next_event_time = self.current_time_idx * self.dt + self.event_offset

        if self.current_time_idx < len(self.committed_u_traj):
            yaw_rate = float(self.committed_u_traj[self.current_time_idx])
        else:
            raw = self.backup_controller(
                robot_state.reshape(-1, 1), current_yaw, np.asarray(u, dtype=float).reshape(-1, 1)
            )
            yaw_rate = float(np.asarray(raw, dtype=float).reshape(-1)[0])
        yaw_rate = float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))

        self.current_time_idx += 1
        return np.array([[yaw_rate]])
