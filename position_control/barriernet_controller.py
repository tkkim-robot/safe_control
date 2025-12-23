import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from safe_control.position_control.BarrierNet.models import BarrierNet, ROBOT_CFG


class BarrierNetController:
    """
    Drop-in controller wrapper for `safe_control/tracking.py`.

    It builds:
    - z (neural input): per-obstacle (px, py, theta, v, dst), flattened over M closest obstacles
    - ctx (QP context): [robot_state_reduced, goal, M obstacles (7D each)] used only for CBF constraints
    then runs BarrierNet(z_norm, ctx) and applies a final deployment clip.
    """

    def __init__(
        self,
                 robot_spec: Dict[str, Any],
                 ckpt_path: str,
                 meta_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.robot_spec = robot_spec
        self.ckpt_path = ckpt_path
        self.robot_model = robot_spec["model"]
        if self.robot_model not in ROBOT_CFG:
            raise ValueError(f"Unsupported robot model for BarrierNet: {self.robot_model}")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self._load_model(ckpt_path, meta_path)
        self.status = "optimal"
        
        # Propagate runtime parameters (radius, bounds, bicycle geometry)
        if "radius" in self.robot_spec:
                self.model.set_robot_radius(float(self.robot_spec["radius"]))

        if self.robot_model == "DynamicUnicycle2D":
            self.model.set_control_bounds(
                a_max=float(self.robot_spec.get("a_max", 0.5)),
                w_max=float(self.robot_spec.get("w_max", 0.5)),
            )
        elif self.robot_model == "Quad2D":
            self.model.set_control_bounds(
                f_min=float(self.robot_spec.get("f_min", 1.0)),
                f_max=float(self.robot_spec.get("f_max", 10.0)),
            )
        elif self.robot_model == "Quad3D":
            self.model.set_control_bounds(
                u_min=float(self.robot_spec.get("u_min", -10.0)),
                u_max=float(self.robot_spec.get("u_max", 10.0)),
            )
        elif self.robot_model == "KinematicBicycle2D_DPCBF":
            rear_ax_dist = float(self.robot_spec.get("rear_ax_dist", 0.2))
            self.model.set_rear_ax_dist(rear_ax_dist)
            wheel_base = float(self.robot_spec.get("wheel_base", 0.4))
            delta_max = float(self.robot_spec.get("delta_max", np.deg2rad(32)))
            beta_max = self.robot_spec.get("beta_max", None)
            if beta_max is None:
                beta_max = float(np.arctan((rear_ax_dist / wheel_base) * np.tan(delta_max)))
            self.model.set_control_bounds(
                a_max=float(self.robot_spec.get("a_max", 5.0)),
                beta_max=float(beta_max),
            )
    
    def _load_model(self, ckpt_path: str, meta_path: Optional[str]) -> BarrierNet:
        if meta_path is None:
            meta_path = ckpt_path.replace(".pth", "_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta_robot_model = meta.get("robot_model")
        if meta_robot_model and meta_robot_model != self.robot_model:
            raise ValueError(f"robot_spec model={self.robot_model} but meta robot_model={meta_robot_model}")
        
        model = BarrierNet(
            robot_model=self.robot_model,
            mean=np.array(meta["mean"], dtype=np.float64),
            std=np.array(meta["std"], dtype=np.float64),
            device=self.device,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model.eval()
        return model.to(self.device)
    
    @staticmethod
    def _obs_as_7d(obs: Optional[np.ndarray]) -> np.ndarray:
        """Normalize obstacles to shape (N,7): [x,y,r,vx,vy,extra,flag]."""
        if obs is None:
            return np.zeros((0, 7), dtype=np.float64)
        obs_arr = np.array(obs, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        if obs_arr.shape[1] < 7:
            pad = np.zeros((obs_arr.shape[0], 7 - obs_arr.shape[1]), dtype=np.float64)
            obs_arr = np.hstack([obs_arr, pad])
        return obs_arr[:, :7]

    def _select_closest_by_clearance(self, robot_state: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Select the closest M obstacles by signed clearance:
          clearance = ||p_obs - p_robot|| - (r_obs + r_robot)
        and pad to (M,7) with far dummy obstacles.
        """
        spec = ROBOT_CFG[self.robot_model]
        M = spec.k_obs
        r_robot = float(self.robot_spec.get("radius", getattr(self.model, "robot_radius", 0.25)))

        if obs.shape[0] == 0:
            sel = np.zeros((0, 7), dtype=np.float64)
        else:
            if self.robot_model == "Quad3D":
                x, y, z = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
                ox, oy = obs[:, 0], obs[:, 1]
                oz = np.zeros_like(ox)
                d = np.sqrt((ox - x) ** 2 + (oy - y) ** 2 + (oz - z) ** 2)
            elif self.robot_model == "Quad2D":
                x, z = float(robot_state[0]), float(robot_state[1])
                ox, oz = obs[:, 0], obs[:, 1]
                d = np.sqrt((ox - x) ** 2 + (oz - z) ** 2)
            else:
                x, y = float(robot_state[0]), float(robot_state[1])
                ox, oy = obs[:, 0], obs[:, 1]
                d = np.sqrt((ox - x) ** 2 + (oy - y) ** 2)

            clearance = d - (obs[:, 2] + r_robot)
            order = np.argsort(clearance)
            sel = obs[order[:M]]

        # pad/truncate to M
        if sel.shape[0] >= M:
            return sel[:M]
        pad_n = M - sel.shape[0]
        far = np.zeros((pad_n, 7), dtype=np.float64)
        far[:, 0] = 1e6
        far[:, 1] = 1e6
        far[:, 2] = 0.0
        return np.vstack([sel, far])

    @staticmethod
    def build_z_ctx_for(robot_model: str, robot_spec: Dict[str, Any], robot_state: np.ndarray, goal: np.ndarray, obs: Optional[np.ndarray]):
        """Return (z_flat, ctx_flat) where z_flat is (M*5,) and ctx_flat is (ctx_dim,)."""
        if robot_state.ndim > 1:
            robot_state = robot_state.flatten()
        goal = np.array(goal, dtype=np.float64).flatten()
        obs7 = BarrierNetController._obs_as_7d(obs)
        spec = ROBOT_CFG[robot_model]
        M = spec.k_obs
        r_robot = float(robot_spec.get("radius", 0.25))

        # select closest by signed clearance and pad to M
        if obs7.shape[0] == 0:
            sel = np.zeros((0, 7), dtype=np.float64)
        else:
            if robot_model == "Quad3D":
                x, y, z = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
                ox, oy = obs7[:, 0], obs7[:, 1]
                oz = np.zeros_like(ox)
                d = np.sqrt((ox - x) ** 2 + (oy - y) ** 2 + (oz - z) ** 2)
            elif robot_model == "Quad2D":
                x, z = float(robot_state[0]), float(robot_state[1])
                ox, oz = obs7[:, 0], obs7[:, 1]
                d = np.sqrt((ox - x) ** 2 + (oz - z) ** 2)
            else:
                x, y = float(robot_state[0]), float(robot_state[1])
                ox, oy = obs7[:, 0], obs7[:, 1]
                d = np.sqrt((ox - x) ** 2 + (oy - y) ** 2)

            clearance = d - (obs7[:, 2] + r_robot)
            order = np.argsort(clearance)
            sel = obs7[order[:M]]

        if sel.shape[0] >= M:
            obs_sel = sel[:M]
        else:
            pad_n = M - sel.shape[0]
            far = np.zeros((pad_n, 7), dtype=np.float64)
            far[:, 0] = 1e6
            far[:, 1] = 1e6
            far[:, 2] = 0.0
            obs_sel = np.vstack([sel, far])

        # Extract theta and v scalar depending on model
        if robot_model in ("DynamicUnicycle2D", "KinematicBicycle2D_DPCBF"):
            theta = float(robot_state[2])
            v = float(robot_state[3])
        elif robot_model == "Quad2D":
            theta = float(robot_state[2])
            vx = float(robot_state[3])
            vz = float(robot_state[4])
            v = float(np.hypot(vx, vz))
        elif robot_model == "Quad3D":
            # Use yaw (psi) if available; otherwise 0.0
            theta = float(robot_state[5]) if robot_state.shape[0] >= 6 else 0.0
            vx = float(robot_state[6])
            vy = float(robot_state[7])
            vz = float(robot_state[8])
            v = float(np.sqrt(vx * vx + vy * vy + vz * vz))
        else:
            raise ValueError(robot_model)

        # goal distance dst
        if robot_model == "Quad3D":
            gx = float(goal[0])
            gy = float(goal[1]) if goal.shape[0] > 1 else 0.0
            gz = float(goal[2]) if goal.shape[0] > 2 else 0.0
            x, y, z = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
            dst = float(np.sqrt((gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2))
        elif robot_model == "Quad2D":
            gx = float(goal[0])
            gz = float(goal[1]) if goal.shape[0] > 1 else 0.0
            x, z = float(robot_state[0]), float(robot_state[1])
            dst = float(np.hypot(gx - x, gz - z))
        else:
            gx = float(goal[0])
            gy = float(goal[1]) if goal.shape[0] > 1 else 0.0
            x, y = float(robot_state[0]), float(robot_state[1])
            dst = float(np.hypot(gx - x, gy - y))

        # Build z blocks: (px,py,theta,v,dst) per obstacle using obstacle-relative px,py
        blocks = []
        if robot_model == "Quad3D":
            x, y, z = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
            for i in range(obs_sel.shape[0]):
                px = float(obs_sel[i, 0] - x)
                py = float(obs_sel[i, 1] - y)
                blocks.extend([px, py, theta, v, dst])
        elif robot_model == "Quad2D":
            x, zpos = float(robot_state[0]), float(robot_state[1])
            for i in range(obs_sel.shape[0]):
                px = float(obs_sel[i, 0] - x)
                py = float(obs_sel[i, 1] - zpos)
                blocks.extend([px, py, theta, v, dst])
        else:
            x, y = float(robot_state[0]), float(robot_state[1])
            for i in range(obs_sel.shape[0]):
                px = float(obs_sel[i, 0] - x)
                py = float(obs_sel[i, 1] - y)
                blocks.extend([px, py, theta, v, dst])

        z_flat = np.array(blocks, dtype=np.float64)
        if z_flat.shape[0] != spec.z_dim:
            raise ValueError(f"z_dim mismatch for {robot_model}: got {z_flat.shape[0]} expected {spec.z_dim}")

        # Build ctx: [state_reduced, goal, obs_sel_flat]
        if robot_model == "Quad3D":
            # reduced state: [x,y,z,vx,vy,vz]
            x, y, z = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
            vx, vy, vz = float(robot_state[6]), float(robot_state[7]), float(robot_state[8])
            state_red = np.array([x, y, z, vx, vy, vz], dtype=np.float64)
            goal_red = np.array([gx, gy, gz], dtype=np.float64)
        elif robot_model == "Quad2D":
            state_red = np.array(robot_state[:6], dtype=np.float64)
            gx = float(goal[0])
            gz = float(goal[1]) if goal.shape[0] > 1 else 0.0
            goal_red = np.array([gx, gz], dtype=np.float64)
        else:
            state_red = np.array(robot_state[:4], dtype=np.float64)
            gx = float(goal[0])
            gy = float(goal[1]) if goal.shape[0] > 1 else 0.0
            goal_red = np.array([gx, gy], dtype=np.float64)

        ctx_flat = np.concatenate([state_red, goal_red, obs_sel.reshape(-1)]).astype(np.float64)
        if ctx_flat.shape[0] != spec.ctx_dim:
            raise ValueError(f"ctx_dim mismatch for {robot_model}: got {ctx_flat.shape[0]} expected {spec.ctx_dim}")
        return z_flat, ctx_flat

    def build_z_ctx(self, robot_state: np.ndarray, goal: np.ndarray, obs: Optional[np.ndarray]):
        return BarrierNetController.build_z_ctx_for(self.robot_model, self.robot_spec, robot_state, goal, obs)
    
    def solve_control_problem(self, robot_state: np.ndarray, control_ref: Dict[str, Any], obs: Optional[np.ndarray] = None) -> np.ndarray:
        goal = control_ref.get("goal")
        u_ref = control_ref.get("u_ref")
        if goal is None:
            self.status = "optimal"
            return np.zeros(ROBOT_CFG[self.robot_model].n_u, dtype=np.float64)
        if u_ref is None:
            raise ValueError("BarrierNetController requires control_ref['u_ref'] (nominal control) for the QP objective.")

        try:
            z, ctx = self.build_z_ctx(np.array(robot_state, dtype=np.float64), np.array(goal, dtype=np.float64), obs)
            z_norm = (z - self.model.mean.detach().cpu().numpy()) / self.model.std.detach().cpu().numpy()
            z_t = torch.from_numpy(z_norm).double().unsqueeze(0).to(self.device)
            ctx_t = torch.from_numpy(ctx).double().unsqueeze(0).to(self.device)
            u_ref = np.array(u_ref, dtype=np.float64).reshape(-1)
            u_ref_t = torch.from_numpy(u_ref).double().unsqueeze(0).to(self.device)

            with torch.no_grad():
                u = self.model(z_t, ctx_t, u_ref_t, sgn=0)
            u_np = u.detach().cpu().numpy().reshape(-1)
            u_np = self._clip_control(u_np)
            self.status = "optimal"
            return u_np
        except Exception as e:
            self.status = "infeasible"
            print(f"BarrierNet inference failed: {e}")
            return np.zeros(ROBOT_CFG[self.robot_model].n_u, dtype=np.float64)
    
    def _clip_control(self, u: np.ndarray) -> np.ndarray:
        """Final deployment guard: clip to robot_spec bounds."""
        if self.robot_model == "DynamicUnicycle2D":
            a_max = float(self.robot_spec.get("a_max", 0.5))
            w_max = float(self.robot_spec.get("w_max", 0.5))
            u[0] = np.clip(u[0], -a_max, a_max)
            u[1] = np.clip(u[1], -w_max, w_max)
        elif self.robot_model == "Quad2D":
            f_min = float(self.robot_spec.get("f_min", 1.0))
            f_max = float(self.robot_spec.get("f_max", 10.0))
            u[0] = np.clip(u[0], f_min, f_max)
            u[1] = np.clip(u[1], f_min, f_max)
        elif self.robot_model == "Quad3D":
            u_min = float(self.robot_spec.get("u_min", -10.0))
            u_max = float(self.robot_spec.get("u_max", 10.0))
            for i in range(u.shape[0]):
                u[i] = np.clip(u[i], u_min, u_max)
        elif self.robot_model == "KinematicBicycle2D_DPCBF":
            a_max = float(self.robot_spec.get("a_max", 5.0))
            beta_max = float(self.robot_spec.get("beta_max", np.deg2rad(32)))
            u[0] = np.clip(u[0], -a_max, a_max)
            u[1] = np.clip(u[1], -beta_max, beta_max)
        return u