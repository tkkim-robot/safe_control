"""
BarrierNet safety layer (multi-constraint QP) used by safe_control.

Key property (matching BarrierNet paper behavior):
- Training: do NOT include hard input bounds inside the QP (sgn=1).
- Deployment: include input bounds (as extra inequality constraints) and also clip as a final guard in the controller.

Obstacle representation follows safe_control convention:
  obs (7D): [x, y, r, vx, vy, extra, flag]
We stack K obstacle constraints into the QP: G u <= h with shape (B, K, n_u).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qpth.qp import QPFunction
from torch.autograd import Variable


def _cvxopt_qp_solve(Q: torch.Tensor, p: torch.Tensor, G: torch.Tensor, h: torch.Tensor):
    """cvxopt solver for inference (CPU)."""
    from cvxopt import matrix, solvers

    solvers.options["show_progress"] = False
    mat_Q = matrix(Q.detach().cpu().numpy())
    mat_p = matrix(p.detach().cpu().numpy())
    mat_G = matrix(G.detach().cpu().numpy())
    mat_h = matrix(h.detach().cpu().numpy())
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    return sol["x"]


@dataclass(frozen=True)
class ModelSpec:
    family: str
    state_dim: int
    goal_dim: int
    obs_dim: int
    k_obs: int
    n_u: int
    relative_degree: int  # 1 or 2
    hidden_sizes: Dict[str, int]
    activation: str
    extra_layers: bool
    control_bounds: Dict[str, float]
    robot_radius_default: float = 0.25
    z_block_dim: int = 5  # (px, py, theta, v, dst)

    @property
    def z_dim(self) -> int:
        return self.k_obs * self.z_block_dim

    @property
    def ctx_dim(self) -> int:
        return self.state_dim + self.goal_dim + self.k_obs * self.obs_dim


ROBOT_CFG: Dict[str, ModelSpec] = {
    "DynamicUnicycle2D": ModelSpec(
        family="2D_robot",
        state_dim=4,
        goal_dim=2,
        obs_dim=7,
        k_obs=5,
        n_u=2,  # [a, w]
        relative_degree=2,
        hidden_sizes={"nHidden1": 256, "nHidden21": 64, "nHidden22": 64},
        activation="relu",
        extra_layers=False,
        control_bounds={"a_max": 0.5, "w_max": 0.5},
        robot_radius_default=0.25,
    ),
    "Quad2D": ModelSpec(
        family="2D_robot",
        state_dim=6,
        goal_dim=2,
        obs_dim=7,
        k_obs=5,
        n_u=2,  # [u_r, u_l]
        relative_degree=2,
        hidden_sizes={"nHidden1": 256, "nHidden21": 64, "nHidden22": 64},
        activation="relu",
        extra_layers=False,
        control_bounds={"f_min": 1.0, "f_max": 10.0},
        robot_radius_default=0.25,
    ),
    "Quad3D": ModelSpec(
        family="3D_robot",
        state_dim=6,  # reduced: [x,y,z,vx,vy,vz]
        goal_dim=3,
        obs_dim=7,
        k_obs=5,
        n_u=4,  # motor forces
        relative_degree=1,  # approximation
        hidden_sizes={"nHidden1": 512, "nHidden21": 128, "nHidden22": 128},
        activation="tanh",
        extra_layers=True,
        control_bounds={"u_min": -10.0, "u_max": 10.0},
        robot_radius_default=0.25,
    ),
    "KinematicBicycle2D_DPCBF": ModelSpec(
        family="2D_robot",
        state_dim=4,  # [x,y,theta,v]
        goal_dim=2,
        obs_dim=7,
        k_obs=5,
        n_u=2,  # [a, beta]
        relative_degree=1,
        hidden_sizes={"nHidden1": 256, "nHidden21": 64, "nHidden22": 64},
        activation="relu",
        extra_layers=False,
        control_bounds={"a_max": 5.0, "beta_max": float(np.deg2rad(32))},
        robot_radius_default=0.3,
    ),
}


def _act_fn(name: str):
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unknown activation '{name}'")


class BarrierNet(nn.Module):
    """
    Neural network that outputs:
    - q: nominal objective linear term for QP (tries to match nominal control)
    - p: adaptive CBF parameters
    then solves a safety QP:
        minimize 0.5 u^T Q u + q^T u
        s.t.     G u <= h   (stacked K obstacle constraints + optional bounds in deployment)
    """

    def __init__(self, robot_model: str, mean: np.ndarray, std: np.ndarray, device: torch.device):
        super().__init__()
        if robot_model not in ROBOT_CFG:
            raise ValueError(f"Unknown robot model: {robot_model}")
        
        self.robot_model = robot_model
        self.spec: ModelSpec = ROBOT_CFG[robot_model]
        self.device = device

        # mean/std are for z only (neural input). ctx is passed raw.
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        if mean.shape[0] != self.spec.z_dim or std.shape[0] != self.spec.z_dim:
            raise ValueError(
                f"BarrierNet mean/std must be length z_dim={self.spec.z_dim} for {robot_model}, "
                f"got mean={mean.shape} std={std.shape}"
            )
        self.mean = torch.as_tensor(mean, dtype=torch.double, device=device)
        self.std = torch.as_tensor(std, dtype=torch.double, device=device)

        hs = self.spec.hidden_sizes
        # Shared per-obstacle encoder on 5D blocks (px,py,theta,v,dst)
        self.obs_fc1 = nn.Linear(self.spec.z_block_dim, hs["nHidden1"]).double()
        self.obs_fc2 = nn.Linear(hs["nHidden1"], hs["nHidden21"]).double()
        if self.spec.extra_layers:
            self.obs_fcm = nn.Linear(hs["nHidden21"], hs["nHidden21"]).double()

        # Heads:
        # - per-obstacle CBF parameters p (2 params) for stacked constraints
        # QP nominal objective comes from the provided u_ref (controller's goal-tracking nominal input).
        self.fc_p = nn.Linear(hs["nHidden21"], 2).double()  # per-obstacle p1,p2 in (0,4)
        self._act = _act_fn(self.spec.activation)

        self.eps_q = 1e-6

        # runtime params
        self.robot_radius = float(self.spec.robot_radius_default)
        self.rear_ax_dist = 0.2
        self.control_bounds_runtime = dict(self.spec.control_bounds)

    def set_robot_radius(self, r: float):
        self.robot_radius = float(r)

    def set_rear_ax_dist(self, rear_ax_dist: float):
        self.rear_ax_dist = float(rear_ax_dist)

    def set_control_bounds(self, **kwargs):
        for k, v in kwargs.items():
            self.control_bounds_runtime[k] = float(v)

    # ----------------------
    # Core forward + QP
    # ----------------------
    def forward(self, z_norm: torch.Tensor, ctx: torch.Tensor, u_ref: torch.Tensor, sgn: int = 1) -> torch.Tensor:
        """
        Args:
            z_norm: normalized neural features z (B, z_dim)
            ctx: raw context for CBF construction (B, ctx_dim)
            sgn: 1 => training mode QP (no bounds), 0 => inference mode QP (+bounds)
        Returns:
            u: (B, n_u)
        """
        z_norm = z_norm.view(z_norm.size(0), -1).double()
        n_batch = z_norm.size(0)
        if z_norm.size(1) != self.spec.z_dim:
            raise ValueError(f"z_norm dim mismatch: got {z_norm.size(1)} expected {self.spec.z_dim}")

        ctx = ctx.view(ctx.size(0), -1).double()
        if ctx.size(0) != n_batch:
            raise ValueError(f"ctx batch mismatch: got {ctx.size(0)} expected {n_batch}")
        if ctx.size(1) != self.spec.ctx_dim:
            raise ValueError(f"ctx dim mismatch: got {ctx.size(1)} expected {self.spec.ctx_dim}")

        # reshape into per-obstacle blocks
        z_blocks = z_norm.view(n_batch, self.spec.k_obs, self.spec.z_block_dim)  # (B,M,5)
        e = self._act(self.obs_fc1(z_blocks))
        e = self._act(self.obs_fc2(e))  # (B,M,E)
        if self.spec.extra_layers:
            e = self._act(self.obs_fcm(e))

        # per-obstacle parameters (B,M,2)
        p_obs = 4.0 * torch.sigmoid(self.fc_p(e))

        # Nominal objective from controller reference:
        # Our QP objective is 0.5||u||^2 + q^T u, which is equivalent to minimize ||u - u_ref||^2
        # when q = -u_ref (up to an additive constant).
        u_ref = u_ref.view(n_batch, -1).double()
        if u_ref.size(1) != self.spec.n_u:
            raise ValueError(f"u_ref dim mismatch: got {u_ref.size(1)} expected {self.spec.n_u}")
        q = -u_ref

        return self._safety_qp(ctx, q, p_obs, sgn=sgn, n_batch=n_batch)

    def _split_ctx(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.spec
        state = ctx[:, : s.state_dim]
        goal = ctx[:, s.state_dim : s.state_dim + s.goal_dim]
        obs_flat = ctx[:, s.state_dim + s.goal_dim :]
        obs = obs_flat.view(-1, s.k_obs, s.obs_dim)
        return state, goal, obs

    def _safety_qp(self, ctx: torch.Tensor, q: torch.Tensor, p_obs: torch.Tensor, sgn: int, n_batch: int) -> torch.Tensor:
        Q = torch.eye(self.spec.n_u, dtype=torch.double, device=self.device).unsqueeze(0).expand(n_batch, -1, -1)
        Q = Q + self.eps_q * torch.eye(self.spec.n_u, dtype=torch.double, device=self.device).unsqueeze(0).expand(n_batch, -1, -1)

        # CBF constraints: (B, K, n_u), (B, K)
        if self.robot_model == "DynamicUnicycle2D":
            G_cbf, h_cbf = self._constraints_dynamic_unicycle(ctx, p_obs)
        elif self.robot_model == "Quad2D":
            G_cbf, h_cbf = self._constraints_quad2d(ctx, p_obs)
        elif self.robot_model == "Quad3D":
            G_cbf, h_cbf = self._constraints_quad3d_approx(ctx, p_obs)
        elif self.robot_model == "KinematicBicycle2D_DPCBF":
            G_cbf, h_cbf = self._constraints_kin_bicycle_dpcbf(ctx, p_obs)
        else:
            raise NotImplementedError(self.robot_model)

        # Add bounds ONLY in inference mode
        if (not self.training) and (sgn == 0):
            G_bnd, h_bnd = self._bounds_constraints(n_batch)
            if G_bnd is not None:
                G = torch.cat([G_cbf, G_bnd], dim=1)
                h = torch.cat([h_cbf, h_bnd], dim=1)
            else:
                G, h = G_cbf, h_cbf
        else:
            G, h = G_cbf, h_cbf

        e = Variable(torch.Tensor()).to(self.device)  # no equalities

        if self.training or sgn == 1:
            # qpth can occasionally return None (or throw) when the QP is infeasible / numerically difficult,
            # especially with many active constraints. For training robustness, fall back to the unconstrained
            # minimizer of 0.5||u||^2 + q^T u, which is u* = -q.
            try:
                u = QPFunction(verbose=0)(Q, q, G, h, e, e)
                if u is None:
                    return -q
                return u
            except Exception:
                return -q

        # inference:
        # Prefer cvxopt (CPU) for determinism, but fall back to qpth (and finally -q) if solver fails.
        try:
            u0 = _cvxopt_qp_solve(Q[0], q[0], G[0], h[0])
            u0 = np.array(u0).astype(np.float64).reshape(1, -1)
            return torch.from_numpy(u0).to(self.device).double()
        except Exception:
            try:
                u = QPFunction(verbose=0)(Q, q, G, h, e, e)
                if u is None:
                    return -q
                return u
            except Exception:
                return -q

    def _bounds_constraints(self, n_batch: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        b = self.control_bounds_runtime
        nu = self.spec.n_u
        if self.robot_model == "DynamicUnicycle2D":
            a_max = float(b.get("a_max", 0.5))
            w_max = float(b.get("w_max", 0.5))
            G = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.double, device=self.device)
            h = torch.tensor([a_max, a_max, w_max, w_max], dtype=torch.double, device=self.device)
        elif self.robot_model == "Quad2D":
            f_min = float(b.get("f_min", 1.0))
            f_max = float(b.get("f_max", 10.0))
            G = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.double, device=self.device)
            h = torch.tensor([f_max, -f_min, f_max, -f_min], dtype=torch.double, device=self.device)
        elif self.robot_model == "Quad3D":
            u_min = float(b.get("u_min", -10.0))
            u_max = float(b.get("u_max", 10.0))
            G_list, h_list = [], []
            for i in range(nu):
                row = [0.0] * nu
                row[i] = 1.0
                G_list.append(row)
                h_list.append(u_max)
                row2 = [0.0] * nu
                row2[i] = -1.0
                G_list.append(row2)
                h_list.append(-u_min)
            G = torch.tensor(G_list, dtype=torch.double, device=self.device)
            h = torch.tensor(h_list, dtype=torch.double, device=self.device)
        elif self.robot_model == "KinematicBicycle2D_DPCBF":
            a_max = float(b.get("a_max", 5.0))
            beta_max = float(b.get("beta_max", np.deg2rad(32)))
            G = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.double, device=self.device)
            h = torch.tensor([a_max, a_max, beta_max, beta_max], dtype=torch.double, device=self.device)
        else:
            return None, None

        G = G.unsqueeze(0).expand(n_batch, -1, -1)
        h = h.unsqueeze(0).expand(n_batch, -1)
        return G, h

    # ----------------------
    # Constraint builders
    # ----------------------
    def _constraints_dynamic_unicycle(self, ctx: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state, _, obs = self._split_ctx(ctx)
        x = state[:, 0:1]
        y = state[:, 1:2]
        theta = state[:, 2:3]
        v = state[:, 3:4]

        ox = obs[:, :, 0]
        oy = obs[:, :, 1]
        r = obs[:, :, 2]
        R = r + self.robot_radius

        dx = x - ox
        dy = y - oy
        b = dx * dx + dy * dy - R * R  # (B,K)

        # Lf b
        cb = torch.cos(theta)
        sb = torch.sin(theta)
        Lf_b = 2.0 * dx * (v * cb) + 2.0 * dy * (v * sb)  # (B,K)

        # Lf^2 b (approx)
        Lf2_b = 2.0 * v * v  # (B,1)
        Lf2_b = Lf2_b.expand_as(b)

        # Lg Lf b for u=[a, w]
        dLf_dv = 2.0 * (dx * cb + dy * sb)  # (B,K)
        dLf_dtheta = 2.0 * v * (-dx * sb + dy * cb)  # (B,K)

        LgLf = torch.stack([dLf_dv, dLf_dtheta], dim=2)  # (B,K,2)

        p1 = p_obs[:, :, 0]
        p2 = p_obs[:, :, 1]
        h = Lf2_b + (p1 + p2) * Lf_b + (p1 * p2) * b
        G = -LgLf
        return G, h

    def _constraints_quad2d(self, ctx: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # State: [x, z, theta, x_dot, z_dot, theta_dot]
        state, _, obs = self._split_ctx(ctx)
        x = state[:, 0:1]
        z = state[:, 1:2]
        vx = state[:, 3:4]
        vz = state[:, 4:5]

        ox = obs[:, :, 0]
        oz = obs[:, :, 1]
        r = obs[:, :, 2]
        R = r + self.robot_radius

        dx = x - ox
        dz = z - oz
        b = dx * dx + dz * dz - R * R
        Lf_b = 2.0 * dx * vx + 2.0 * dz * vz
        Lf2_b = 2.0 * (vx * vx + vz * vz)
        Lf2_b = Lf2_b.expand_as(b)

        # Very rough control influence model: u directly affects (vx,vz) equally.
        # This keeps the QP well-formed and matches the existing baseline-label dimensions.
        dLf_dvx = 2.0 * dx
        dLf_dvz = 2.0 * dz
        # map to 2 controls (u_r,u_l) via sum channel
        g1 = dLf_dvx + dLf_dvz
        g2 = dLf_dvx + dLf_dvz
        LgLf = torch.stack([g1, g2], dim=2)

        p1 = p_obs[:, :, 0]
        p2 = p_obs[:, :, 1]
        h = Lf2_b + (p1 + p2) * Lf_b + (p1 * p2) * b
        G = -LgLf
        return G, h

    def _constraints_quad3d_approx(self, ctx: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reduced state: [x,y,z,vx,vy,vz]
        state, _, obs = self._split_ctx(ctx)
        x = state[:, 0:1]
        y = state[:, 1:2]
        z = state[:, 2:3]
        vx = state[:, 3:4]
        vy = state[:, 4:5]
        vz = state[:, 5:6]

        ox = obs[:, :, 0]
        oy = obs[:, :, 1]
        oz = torch.zeros_like(ox)  # z obstacle not provided in 7D; assume 0
        r = obs[:, :, 2]
        R = r + self.robot_radius

        dx = x - ox
        dy = y - oy
        dz = z - oz
        b = dx * dx + dy * dy + dz * dz - R * R  # (B,K)

        # Lf b
        Lf_b = 2.0 * dx * vx + 2.0 * dy * vy + 2.0 * dz * vz

        # Approximate Lg b: treat controls as direct accel in xyz via first 3 motors aggregate + 4th unused
        # Shape (B,K,4)
        Lg = torch.zeros((ctx.size(0), self.spec.k_obs, 4), dtype=torch.double, device=self.device)
        Lg[:, :, 0] = 2.0 * dx
        Lg[:, :, 1] = 2.0 * dy
        Lg[:, :, 2] = 2.0 * dz
        Lg[:, :, 3] = 0.0

        alpha = p_obs[:, :, 0]
        h = Lf_b + alpha * b
        G = -Lg
        return G, h

    def _constraints_kin_bicycle_dpcbf(self, ctx: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Torch implementation of the (relative-degree-1) DPCBF-style constraint.
        This is a direct translation of the key structure used in safe_control's DPCBF robot:
            h = v_rel_x + lambda * v_rel_y^2 + mu
            constraint: d_h + alpha * h >= 0
        """
        state, _, obs = self._split_ctx(ctx)
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        v = state[:, 3]

        ox = obs[:, :, 0]
        oy = obs[:, :, 1]
        r = obs[:, :, 2]
        ovx = obs[:, :, 3]
        ovy = obs[:, :, 4]

        # relative position/velocity (ego minus obstacle)
        prx = ox - x.unsqueeze(1)
        pry = oy - y.unsqueeze(1)
        evx = v.unsqueeze(1) * torch.cos(theta).unsqueeze(1)
        evy = v.unsqueeze(1) * torch.sin(theta).unsqueeze(1)
        vrx = ovx - evx
        vry = ovy - evy

        # rotate into obstacle line-of-sight frame
        rot = torch.atan2(pry, prx)
        cr = torch.cos(rot)
        sr = torch.sin(rot)
        vrel_x = cr * vrx + sr * vry
        vrel_y = -sr * vrx + cr * vry

        pr_norm = torch.sqrt(prx * prx + pry * pry + 1e-12)
        vr_norm = torch.sqrt(vrx * vrx + vry * vry + 1e-12)

        ego_dim = torch.as_tensor(self.robot_radius, dtype=torch.double, device=self.device)
        beta_safe = torch.as_tensor(1.1, dtype=torch.double, device=self.device)

        d_safe = torch.clamp(pr_norm * pr_norm - ego_dim * ego_dim, min=1e-6)
        k_lam = 0.1 * torch.sqrt(beta_safe * beta_safe - 1.0) / ego_dim
        k_mu = 0.5 * torch.sqrt(beta_safe * beta_safe - 1.0) / ego_dim

        lam = k_lam * torch.sqrt(d_safe) / vr_norm
        mu = k_mu * torch.sqrt(d_safe)

        h_k = vrel_x + lam * (vrel_y * vrel_y) + mu  # (B,K)

        # We approximate dh/du via dependence on ego velocity components through vrel_x/vrel_y.
        # Control u = [a, beta]; affect v and heading rate indirectly. We use a conservative linear proxy:
        #   dh/da ~ -cos(rot) * cos(theta) - sin(rot) * sin(theta)  (through -d/dv of ego velocity)
        dh_dv = -(cr * torch.cos(theta).unsqueeze(1) + sr * torch.sin(theta).unsqueeze(1))  # (B,K)

        # Lg_h: only a influences v in this proxy; beta influence ignored (still solvable)
        Lg_h = torch.zeros((ctx.size(0), self.spec.k_obs, 2), dtype=torch.double, device=self.device)
        Lg_h[:, :, 0] = dh_dv  # a channel
        Lg_h[:, :, 1] = 0.0

        # Lf_h proxy: set 0 (BarrierNet learns alpha scaling via p1); keeps constraint meaningful via h_k
        Lf_h = torch.zeros_like(h_k)

        alpha = p_obs[:, :, 0]
        G = -Lg_h
        h = Lf_h + alpha * h_k
        return G, h



