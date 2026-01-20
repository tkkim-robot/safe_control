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
    import numpy as np

    solvers.options["show_progress"] = False
    mat_Q = matrix(Q.detach().cpu().numpy())
    mat_p = matrix(p.detach().cpu().numpy())
    mat_G = matrix(G.detach().cpu().numpy())
    mat_h = matrix(h.detach().cpu().numpy())
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    
    # Check if solution is valid
    if sol["x"] is None:
        # QP is infeasible - return None to trigger fallback
        return None
    
    # Convert cvxopt matrix to numpy array
    x_np = np.array(sol["x"]).astype(np.float64).flatten()
    return x_np


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

        u_in_dim = hs["nHidden21"] + self.spec.state_dim + self.spec.goal_dim + self.spec.n_u
        self.u_fc1 = nn.Linear(u_in_dim, hs["nHidden22"]).double()
        if self.spec.extra_layers:
            self.u_fcm = nn.Linear(hs["nHidden22"], hs["nHidden22"]).double()
        self.u_out = nn.Linear(hs["nHidden22"], self.spec.n_u).double()
        self._act = _act_fn(self.spec.activation)

        self.eps_q = 1e-6
        self.use_differentiable_qp_in_training = True
        self.train_slack_rho = 1e4  # penalty weight on slack variables during training

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
    def forward(
        self,
        z_norm: torch.Tensor,
        ctx: torch.Tensor,
        u_ref: torch.Tensor,
        sgn: int = 1,
        return_aux: bool = False,
    ):
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

        u_ref = u_ref.view(n_batch, -1).double()
        if u_ref.size(1) != self.spec.n_u:
            raise ValueError(f"u_ref dim mismatch: got {u_ref.size(1)} expected {self.spec.n_u}")

        state, goal, _ = self._split_ctx(ctx)
        e_pool = e.mean(dim=1)  # (B,E) mean over K obstacles
        u_head_in = torch.cat([e_pool, state, goal, u_ref], dim=1)  # (B, u_in_dim)
        du = self._act(self.u_fc1(u_head_in))
        if self.spec.extra_layers:
            du = self._act(self.u_fcm(du))
        du = self.u_out(du)
        u_nom = u_ref + du
        q = -u_ref # F
        # q = -u_nom # F

        state, goal, obs = self._split_ctx(ctx)
        
        # Debug: Check barrier function values
        if not self.training and sgn == 0:
            # Only in inference mode
            with torch.no_grad():
                G_cbf_debug, h_cbf_debug = self._cbf_constraints(state, obs, p_obs)
                # Check if any barrier function is negative (unsafe)
                if self.robot_model == "DynamicUnicycle2D":
                    x, y = state[0, 0:1], state[0, 1:2]
                    ox, oy = obs[0, :, 0], obs[0, :, 1]
                    r = obs[0, :, 2]
                    beta = 1.01
                    d_min = r + self.robot_radius
                    R_sq = beta * d_min * d_min
                    dx = x - ox
                    dy = y - oy
                    b_debug = dx * dx + dy * dy - R_sq
                    min_b = b_debug.min().item()
                    min_p1 = p_obs[0, :, 0].min().item()
                    min_p2 = p_obs[0, :, 1].min().item()
                    if min_b < 0:
                        print(f"WARNING: Barrier function b(x) < 0 (unsafe!): min_b={min_b:.4f}, min_p1={min_p1:.4f}, min_p2={min_p2:.4f}")
        
        u = self._safety_qp(state, obs, q, p_obs, sgn=sgn, n_batch=n_batch)
        if return_aux:
            G_cbf, h_cbf = self._cbf_constraints(state, obs, p_obs)
            return u, {"p_obs": p_obs, "G_cbf": G_cbf, "h_cbf": h_cbf, "u_nom": u_nom}
        return u

    def _cbf_constraints(self, robot_state: torch.Tensor, obs: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (G_cbf, h_cbf) for the current robot model.
        
        Args:
            robot_state: (B, state_dim) - raw robot state (no normalization)
            obs: (B, K, obs_dim) - raw obstacle data (no normalization)
            p_obs: (B, K, 2) - per-obstacle CBF parameters from NN
        """
        if self.robot_model == "DynamicUnicycle2D":
            return self._constraints_dynamic_unicycle(robot_state, obs, p_obs)
        if self.robot_model == "Quad2D":
            return self._constraints_quad2d(robot_state, obs, p_obs)
        if self.robot_model == "Quad3D":
            return self._constraints_quad3d_approx(robot_state, obs, p_obs)
        if self.robot_model == "KinematicBicycle2D_DPCBF":
            return self._constraints_kin_bicycle_dpcbf(robot_state, obs, p_obs)
        raise NotImplementedError(self.robot_model)

    def _split_ctx(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.spec
        state = ctx[:, : s.state_dim]
        goal = ctx[:, s.state_dim : s.state_dim + s.goal_dim]
        obs_flat = ctx[:, s.state_dim + s.goal_dim :]
        obs = obs_flat.view(-1, s.k_obs, s.obs_dim)
        return state, goal, obs

    def _safety_qp(self, robot_state: torch.Tensor, obs: torch.Tensor, q: torch.Tensor, p_obs: torch.Tensor, sgn: int, n_batch: int) -> torch.Tensor:
        Q_u = torch.eye(self.spec.n_u, dtype=torch.double, device=self.device).unsqueeze(0).expand(n_batch, -1, -1)
        Q_u = Q_u + self.eps_q * torch.eye(self.spec.n_u, dtype=torch.double, device=self.device).unsqueeze(0).expand(n_batch, -1, -1)

        # CBF constraints: (B, K, n_u), (B, K)
        G_cbf, h_cbf = self._cbf_constraints(robot_state, obs, p_obs)

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
            if self.use_differentiable_qp_in_training and (G_cbf is not None):
                K = G_cbf.size(1)
                nu = self.spec.n_u
                rho = float(self.train_slack_rho)

                # Build augmented Q and p
                Q = torch.zeros((n_batch, nu + K, nu + K), dtype=torch.double, device=self.device)
                Q[:, :nu, :nu] = Q_u
                Q[:, nu:, nu:] = rho * torch.eye(K, dtype=torch.double, device=self.device).unsqueeze(0).expand(n_batch, -1, -1)
                Q = Q + self.eps_q * torch.eye(nu + K, dtype=torch.double, device=self.device).unsqueeze(0).expand(n_batch, -1, -1)

                p = torch.zeros((n_batch, nu + K), dtype=torch.double, device=self.device)
                p[:, :nu] = q

                # Constraints
                I = torch.eye(K, dtype=torch.double, device=self.device).unsqueeze(0).expand(n_batch, -1, -1)
                G1 = torch.cat([G_cbf, -I], dim=2)  # (B,K,nu+K)
                h1 = h_cbf  # (B,K)
                G2 = torch.cat([torch.zeros((n_batch, K, nu), dtype=torch.double, device=self.device), -I], dim=2)  # -s <= 0
                h2 = torch.zeros((n_batch, K), dtype=torch.double, device=self.device)
                G_aug = torch.cat([G1, G2], dim=1)  # (B,2K,nu+K)
                h_aug = torch.cat([h1, h2], dim=1)  # (B,2K)

                grad_anchor = p_obs.sum(dim=(1, 2)).unsqueeze(1)  # (B,1)
                try:
                    x = QPFunction(verbose=0)(Q, p, G_aug, h_aug, e, e)
                    if x is None:
                        return (-q) + 0.0 * grad_anchor.expand(-1, q.size(1))
                    return x[:, :nu]
                except Exception:
                    return (-q) + 0.0 * grad_anchor.expand(-1, q.size(1))

            grad_anchor = p_obs.sum(dim=(1, 2)).unsqueeze(1)  # (B,1)
            try:
                u = QPFunction(verbose=0)(Q_u, q, G, h, e, e)
                if u is None:
                    return (-q) + 0.0 * grad_anchor.expand(-1, q.size(1))
                return u
            except Exception:
                return (-q) + 0.0 * grad_anchor.expand(-1, q.size(1))

        # inference:
        # Prefer cvxopt (CPU) for determinism, but fall back to qpth (and finally safe stop) if solver fails.
        try:
            u0 = _cvxopt_qp_solve(Q_u[0], q[0], G[0], h[0])
            if u0 is not None:
                u0 = np.array(u0).astype(np.float64).reshape(1, -1)
                return torch.from_numpy(u0).to(self.device).double()
            # If cvxopt returns None (infeasible), try qpth
        except Exception as e:
            # If cvxopt throws exception, try qpth
            pass
        
        # Fallback to qpth
        try:
            u = QPFunction(verbose=0)(Q_u, q, G, h, e, e)
            if u is not None:
                return u
        except Exception:
            pass
        
        print(f"WARNING: BarrierNet QP failed, returning safe stop control")
        return torch.zeros((n_batch, self.spec.n_u), dtype=torch.double, device=self.device)

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
    def _constraints_dynamic_unicycle(self, robot_state: torch.Tensor, obs: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CBF constraints for DynamicUnicycle2D matching agent_barrier structure.
        Based on agent_barrier: h = ||X[0:2] - obsX[0:2]||^2 - beta*d_min^2
        """
        x = robot_state[:, 0:1]
        y = robot_state[:, 1:2]
        theta = robot_state[:, 2:3]
        v = robot_state[:, 3:4]

        ox = obs[:, :, 0]
        oy = obs[:, :, 1]
        r = obs[:, :, 2]
        beta = 1.01  # matching agent_barrier default
        d_min = r + self.robot_radius
        R_sq = beta * d_min * d_min  # beta * d_min^2

        # Barrier function: h = ||X[0:2] - obsX[0:2]||^2 - beta*d_min^2
        dx = x - ox
        dy = y - oy
        b = dx * dx + dy * dy - R_sq  # (B,K)

        # Lf b = h_dot = 2 * (X[0:2] - obsX[0:2]).T @ f(X)[0:2]
        # f(X)[0:2] = [v*cos(theta), v*sin(theta)]
        cb = torch.cos(theta)
        sb = torch.sin(theta)
        Lf_b = 2.0 * dx * (v * cb) + 2.0 * dy * (v * sb)  # (B,K)

        Lf2_b = 2.0 * v * v  # (B,1)
        Lf2_b = Lf2_b.expand_as(b)

        # Lg Lf b for u=[a, w]
        # dLf_dv = 2*(dx*cos(theta) + dy*sin(theta))
        # dLf_dtheta = 2*v*(-dx*sin(theta) + dy*cos(theta))
        dLf_dv = 2.0 * (dx * cb + dy * sb)  # (B,K)
        dLf_dtheta = 2.0 * v * (-dx * sb + dy * cb)  # (B,K)

        LgLf = torch.stack([dLf_dv, dLf_dtheta], dim=2)  # (B,K,2)

        # HOCBF constraint: Lf^2 b + (p1+p2)*Lf b + (p1*p2)*b >= -Lg Lf b @ u
        # Rewrite as: -Lg Lf b @ u <= Lf^2 b + (p1+p2)*Lf b + (p1*p2)*b
        p1 = p_obs[:, :, 0]  # from NN
        p2 = p_obs[:, :, 1]  # from NN
        h = Lf2_b + (p1 + p2) * Lf_b + (p1 * p2) * b
        
        # Debug: Check constraint values in inference
        if not self.training:
            with torch.no_grad():
                # Check if constraints are active (h should be positive for safety)
                min_h = h.min().item()
                min_p1_val = p1.min().item()
                min_p2_val = p2.min().item()
                min_b_val = b.min().item()
                if min_h < 0 or min_b_val < 0:
                    print(f"WARNING: CBF constraint h < 0 or b < 0: min_h={min_h:.4f}, min_b={min_b_val:.4f}, min_p1={min_p1_val:.4f}, min_p2={min_p2_val:.4f}")
        
        G = -LgLf
        return G, h

    def _constraints_quad2d(self, robot_state: torch.Tensor, obs: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CBF constraints for Quad2D matching agent_barrier structure.
        Based on agent_barrier: h = ||X[0:2] - obsX[0:2]||^2 - beta*d_min^2
        State: [x, z, theta, x_dot, z_dot, theta_dot]
        """
        x = robot_state[:, 0:1]
        z = robot_state[:, 1:2]
        vx = robot_state[:, 3:4]
        vz = robot_state[:, 4:5]

        ox = obs[:, :, 0]
        oz = obs[:, :, 1]
        r = obs[:, :, 2]
        beta = 1.01  # matching agent_barrier default
        d_min = r + self.robot_radius
        R_sq = beta * d_min * d_min  # beta * d_min^2

        # Barrier function: h = ||X[0:2] - obsX[0:2]||^2 - beta*d_min^2
        dx = x - ox
        dz = z - oz
        b = dx * dx + dz * dz - R_sq  # (B,K)

        # Lf b = h_dot = 2 * (X[0:2] - obsX[0:2]).T @ f(X)[0:2]
        # f(X)[0:2] = [vx, vz]
        Lf_b = 2.0 * dx * vx + 2.0 * dz * vz  # (B,K)

        g = 9.81  # gravity constant
        Lf2_b = 2.0 * (vx * vx + vz * vz) - 2.0 * g * dz  # (B,K)

        # Lg Lf b for u=[u_r, u_l]
        dLf_dvx = 2.0 * dx
        dLf_dvz = 2.0 * dz
        # map to 2 controls (u_r,u_l) via sum channel
        g1 = dLf_dvx + dLf_dvz
        g2 = dLf_dvx + dLf_dvz
        LgLf = torch.stack([g1, g2], dim=2)  # (B,K,2)

        # HOCBF constraint
        p1 = p_obs[:, :, 0]  # from NN
        p2 = p_obs[:, :, 1]  # from NN
        h = Lf2_b + (p1 + p2) * Lf_b + (p1 * p2) * b
        G = -LgLf
        return G, h

    def _constraints_quad3d_approx(self, robot_state: torch.Tensor, obs: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CBF constraints for Quad3D (approximation, relative degree 1).
        Reduced state: [x,y,z,vx,vy,vz]
        Note: agent_barrier for Quad3D raises NotImplementedError, so we use approximation.
        """
        x = robot_state[:, 0:1]
        y = robot_state[:, 1:2]
        z = robot_state[:, 2:3]
        vx = robot_state[:, 3:4]
        vy = robot_state[:, 4:5]
        vz = robot_state[:, 5:6]

        ox = obs[:, :, 0]
        oy = obs[:, :, 1]
        oz = torch.zeros_like(ox)  # z obstacle not provided in 7D; assume 0
        r = obs[:, :, 2]
        beta = 1.01  # matching agent_barrier default
        d_min = r + self.robot_radius
        R_sq = beta * d_min * d_min  # beta * d_min^2

        # Barrier function: h = ||X[0:3] - obsX[0:3]||^2 - beta*d_min^2
        dx = x - ox
        dy = y - oy
        dz = z - oz
        b = dx * dx + dy * dy + dz * dz - R_sq  # (B,K)

        # Lf b (relative degree 1 approximation)
        Lf_b = 2.0 * dx * vx + 2.0 * dy * vy + 2.0 * dz * vz  # (B,K)

        # Approximate Lg b: treat controls as direct accel in xyz via first 3 motors aggregate + 4th unused
        # Shape (B,K,4)
        Lg = torch.zeros((robot_state.size(0), self.spec.k_obs, 4), dtype=torch.double, device=self.device)
        Lg[:, :, 0] = 2.0 * dx
        Lg[:, :, 1] = 2.0 * dy
        Lg[:, :, 2] = 2.0 * dz
        Lg[:, :, 3] = 0.0

        # CBF constraint (relative degree 1): Lf b + alpha * b >= -Lg b @ u
        # Rewrite as: -Lg b @ u <= Lf b + alpha * b
        alpha = p_obs[:, :, 0]  # from NN
        h = Lf_b + alpha * b
        G = -Lg
        return G, h

    def _constraints_kin_bicycle_dpcbf(self, robot_state: torch.Tensor, obs: torch.Tensor, p_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Torch implementation of the (relative-degree-1) DPCBF-style constraint.
        This matches agent_barrier structure from KinematicBicycle2D_DPCBF:
            h = v_rel_x + lambda * v_rel_y^2 + mu
            constraint: Lf h + alpha * h >= -Lg h @ u
        """
        x = robot_state[:, 0]
        y = robot_state[:, 1]
        theta = robot_state[:, 2]
        v = robot_state[:, 3]

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

        # Matching agent_barrier: ego_dim = (obs[2] + robot_radius) * beta
        beta_safe = torch.as_tensor(1.1, dtype=torch.double, device=self.device)
        ego_dim = (r + self.robot_radius) * beta_safe  # (B,K) - matching agent_barrier exactly

        d_safe = torch.clamp(pr_norm * pr_norm - ego_dim * ego_dim, min=1e-6)
        # Matching agent_barrier: k_lamda = 0.5 * sqrt(beta^2 - 1) / ego_dim, k_mu = 1.0 * sqrt(beta^2 - 1) / ego_dim
        k_lam = 0.5 * torch.sqrt(beta_safe * beta_safe - 1.0) / ego_dim
        k_mu = 1.0 * torch.sqrt(beta_safe * beta_safe - 1.0) / ego_dim

        lam = k_lam * torch.sqrt(d_safe) / vr_norm
        mu = k_mu * torch.sqrt(d_safe)

        h_k = vrel_x + lam * (vrel_y * vrel_y) + mu  # (B,K)

        # We approximate dh/du via dependence on ego velocity components through vrel_x/vrel_y.
        # Control u = [a, beta]; affect v and heading rate indirectly. We use a conservative linear proxy:
        #   dh/da ~ -cos(rot) * cos(theta) - sin(rot) * sin(theta)  (through -d/dv of ego velocity)
        dh_dv = -(cr * torch.cos(theta).unsqueeze(1) + sr * torch.sin(theta).unsqueeze(1))  # (B,K)

        # Lg_h: only a influences v in this proxy; beta influence ignored (still solvable)
        Lg_h = torch.zeros((robot_state.size(0), self.spec.k_obs, 2), dtype=torch.double, device=self.device)
        Lg_h[:, :, 0] = dh_dv  # a channel
        Lg_h[:, :, 1] = 0.0

        # Lf_h proxy: set 0 (BarrierNet learns alpha scaling via p1); keeps constraint meaningful via h_k
        Lf_h = torch.zeros_like(h_k)

        alpha = p_obs[:, :, 0]
        G = -Lg_h
        h = Lf_h + alpha * h_k
        return G, h



