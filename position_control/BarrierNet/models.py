# models.py
"""
Unified BarrierNet model definitions for 2D and 3D robots.

ROBOT MODEL MAPPING TABLE:
================================================================================
| Robot Model        | State Vector                    | Feature Vector          |
|-------------------|----------------------------------|-------------------------|
| DynamicUnicycle2D | [x, y, theta, v]                | [px, py, theta, v, dst_y] |
|                   | Control: [a, omega]             | dst_x fixed at 50.0     |
|                   | CBF: h(x) = ||x-x_obs||^2 - R^2 | Relative degree: 2      |
|                   | Lf b = 2*(px-obs_x)*v*cos(θ) +  | Lf^2 b = 2*v^2          |
|                   |      2*(py-obs_y)*v*sin(θ)      | Lg Lf b = [∂h/∂θ, ∂h/∂v]|
|-------------------|----------------------------------|-------------------------|
| Quad2D            | [x, z, theta, x_dot, z_dot, θ_dot] | [x, z, theta, x_dot, z_dot, θ_dot] |
|                   | Control: [u_right, u_left]      | 6 features, 2 controls  |
|                   | CBF: h(x) = ||x-x_obs||^2 - R^2 | Relative degree: 2      |
|                   | Dynamics: v̇x = -sin(θ)/m * Σu   | Lf b = 2*(x-obs_x)*vx + |
|                   |         v̇z = -g + cos(θ)/m * Σu |      2*(z-obs_z)*vz     |
|                   |         θ̇ = r/I * (u_r - u_l)   | Lf^2 b = 2*(vx^2 + vz^2)|
|-------------------|----------------------------------|-------------------------|
| Quad3D            | [x,y,z,θ,φ,ψ,vx,vy,vz,q,p,r]    | [x, vx, y, vy, z, vz]   |
|                   | Control: [u1, u2, u3, u4]       | 6 features, 3 controls  |
|                   | CBF: h(x) = ||x-x_obs||^4 - R^4 | Relative degree: 1      |
|                   | Superquadratic barrier          | Lf b = 4*Σ(xi-obs_i)^3*vi|
|                   | Linearized dynamics: v̇x = g*θ   | Lg b = [4*(x-obs_x)^3,  |
|                   |                    v̇y = -g*φ   |     4*(y-obs_y)^3,      |
|                   |                    v̇z = F/m    |     4*(z-obs_z)^3]      |
================================================================================

CBF Inequality Form:
- For relative degree 2: -Lg Lf b(x) * u <= Lf^2 b(x) + (p1+p2) Lf b(x) + (p1*p2) b(x)
- For relative degree 1: -Lg b(x) * u <= Lf b(x) + p1 * b(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from qpth.qp import QPFunction
import numpy as np
import json
import os

# Import the test_solver from the training script
def test_solver(Q, p, G, h):
    from cvxopt import solvers, matrix
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())
    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    return sol['x']

# Per-model configuration registry
ROBOT_CFG = {
    "DynamicUnicycle2D": {
        "family": "2D_robot",
        "n_features": 5,
        "n_cls": 2,
        "hidden_sizes": {"nHidden1": 128, "nHidden21": 32, "nHidden22": 32},
        "activation": F.relu,
        "extra_layers": False,
        "obs_default": [40, 15, 6],  # [obs_x, obs_y, R]
        "control_bounds": {"a_max": 0.5, "w_max": 0.5},
        "relative_degree": 2
    },
    "Quad2D": {
        "family": "2D_robot", 
        "n_features": 6,
        "n_cls": 2,
        "hidden_sizes": {"nHidden1": 128, "nHidden21": 32, "nHidden22": 32},
        "activation": F.relu,
        "extra_layers": False,
        "obs_default": [40, 15, 6],  # [obs_x, obs_y, R] - same as unicycle for now
        "control_bounds": {"f_min": 1.0, "f_max": 10.0},
        "relative_degree": 2
    },
    "Quad3D": {
        "family": "3D_robot",
        "n_features": 6,
        "n_cls": 3,
        "hidden_sizes": {"nHidden1": 512, "nHidden21": 128, "nHidden22": 128},
        "activation": torch.tanh,
        "extra_layers": True,
        "obs_default": [10, 10, 9, 7],  # [obs_x, obs_y, obs_z, R]
        "control_bounds": {"u_min": -10.0, "u_max": 10.0},
        "relative_degree": 1
    }
}

class BarrierNet(nn.Module):
    def __init__(self, robot_model, mean, std, device, obs_positions=None, bn=False):
        super().__init__()
        
        # Validate robot model
        if robot_model not in ROBOT_CFG:
            raise ValueError(f"Unknown robot model: {robot_model}. Supported: {list(ROBOT_CFG.keys())}")
        
        self.robot_model = robot_model
        self.cfg = ROBOT_CFG[robot_model].copy()
        self.robot_family = self.cfg["family"]
        
        # Store normalization parameters
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device
        self.bn = bn
        
        # Extract configuration
        self.nFeatures = self.cfg["n_features"]
        self.nCls = self.cfg["n_cls"]
        self.act_fn = self.cfg["activation"]
        
        # Set obstacle positions (override defaults if provided)
        if obs_positions is not None:
            if self.robot_family == "2D_robot":
                self.cfg["obs_x"] = obs_positions[0]
                self.cfg["obs_y"] = obs_positions[1]
                if len(obs_positions) > 2:
                    self.cfg["R"] = obs_positions[2]
            elif self.robot_family == "3D_robot":
                self.cfg["obs_x"] = obs_positions[0]
                self.cfg["obs_y"] = obs_positions[1]
                self.cfg["obs_z"] = obs_positions[2]
                if len(obs_positions) > 3:
                    self.cfg["R"] = obs_positions[3]
        else:
            # Use defaults
            if self.robot_family == "2D_robot":
                self.cfg["obs_x"] = self.cfg["obs_default"][0]
                self.cfg["obs_y"] = self.cfg["obs_default"][1]
                self.cfg["R"] = self.cfg["obs_default"][2]
            elif self.robot_family == "3D_robot":
                self.cfg["obs_x"] = self.cfg["obs_default"][0]
                self.cfg["obs_y"] = self.cfg["obs_default"][1]
                self.cfg["obs_z"] = self.cfg["obs_default"][2]
                self.cfg["R"] = self.cfg["obs_default"][3]

        # Build network layers
        hidden_sizes = self.cfg["hidden_sizes"]
        self.fc1 = nn.Linear(self.nFeatures, hidden_sizes["nHidden1"]).double()
        self.fc21 = nn.Linear(hidden_sizes["nHidden1"], hidden_sizes["nHidden21"]).double()
        self.fc22 = nn.Linear(hidden_sizes["nHidden1"], hidden_sizes["nHidden22"]).double()
        
        if self.cfg["extra_layers"]:
            self.fcm1 = nn.Linear(hidden_sizes["nHidden21"], hidden_sizes["nHidden21"]).double()
            self.fcm2 = nn.Linear(hidden_sizes["nHidden22"], hidden_sizes["nHidden22"]).double()
        
        self.fc31 = nn.Linear(hidden_sizes["nHidden21"], self.nCls).double()
        # For 3D robot, fc32 outputs 2 parameters; for 2D robots, it outputs nCls parameters
        fc32_output = 2 if self.robot_family == "3D_robot" else self.nCls
        self.fc32 = nn.Linear(hidden_sizes["nHidden22"], fc32_output).double()

    def forward(self, x, sgn):
        nBatch = x.size(0)
        x = x.view(nBatch, -1)
        
        # Convert to double precision for QP compatibility
        x = x.double()
        
        # Denormalize features
        x0 = x * self.std + self.mean
        
        # Forward pass through network
        x = self.act_fn(self.fc1(x))
        x21 = self.act_fn(self.fc21(x))
        x22 = self.act_fn(self.fc22(x))
        
        if self.cfg["extra_layers"]:
            x21 = self.act_fn(self.fcm1(x21))
            x22 = self.act_fn(self.fcm2(x22))
        
        x31 = self.fc31(x21)
        x32 = 4 * nn.Sigmoid()(self.fc32(x22))  # Ensure CBF parameters are positive
        
        return self.dCBF(x0, x31, x32, sgn, nBatch)

    def dCBF(self, x0, x31, x32, sgn, nBatch):
        """Compute CBF-based control using robot-specific dynamics."""
        Q = Variable(torch.eye(self.nCls)).unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        
        if self.robot_model == "DynamicUnicycle2D":
            return self._dCBF_unicycle2D(x0, x31, x32, sgn, nBatch, Q)
        elif self.robot_model == "Quad2D":
            return self._dCBF_quad2D(x0, x31, x32, sgn, nBatch, Q)
        elif self.robot_model == "Quad3D":
            return self._dCBF_quad3D(x0, x31, x32, sgn, nBatch, Q)
        else:
            raise NotImplementedError(f"CBF not implemented for {self.robot_model}")

    def _dCBF_unicycle2D(self, x0, x31, x32, sgn, nBatch, Q):
        """CBF for DynamicUnicycle2D - relative degree 2"""
        # Extract state variables: [px, py, theta, v, dst_y]
        px, py, theta, v = x0[:,0], x0[:,1], x0[:,2], x0[:,3]
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
        
        # Barrier function: h(x) = ||x-x_obs||^2 - R^2
        barrier = (px - self.cfg["obs_x"])**2 + (py - self.cfg["obs_y"])**2 - self.cfg["R"]**2
        
        # First derivative: Lf b = 2*(px-obs_x)*v*cos(θ) + 2*(py-obs_y)*v*sin(θ)
        barrier_dot = 2*(px - self.cfg["obs_x"])*v*cos_theta + 2*(py - self.cfg["obs_y"])*v*sin_theta
        
        # Second derivative: Lf^2 b = 2*v^2
        Lf2b = 2*v**2
        
        # Control matrix: Lg Lf b = [∂h/∂θ, ∂h/∂v]
        LgLfbu1 = torch.reshape(-2*(px - self.cfg["obs_x"])*v*sin_theta + 2*(py - self.cfg["obs_y"])*v*cos_theta, (nBatch, 1))
        LgLfbu2 = torch.reshape(2*(px - self.cfg["obs_x"])*cos_theta + 2*(py - self.cfg["obs_y"])*sin_theta, (nBatch, 1))
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        
        # CBF constraint: -Lg Lf b * u <= Lf^2 b + (p1+p2) Lf b + (p1*p2) b
        h = torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1))
        
        return self._solve_qp(Q, x31, G, h, sgn, nBatch)

    def _dCBF_quad2D(self, x0, x31, x32, sgn, nBatch, Q):
        """CBF for Quad2D - relative degree 2"""
        # Extract state variables: [x, z, theta, x_dot, z_dot, theta_dot]
        x, z, theta = x0[:,0], x0[:,1], x0[:,2]
        x_dot, z_dot = x0[:,3], x0[:,4]
        
        # Barrier function: h(x) = ||x-x_obs||^2 - R^2 (simplified to x-z plane)
        barrier = (x - self.cfg["obs_x"])**2 + (z - self.cfg["obs_z"])**2 - self.cfg["R"]**2
        
        # First derivative: Lf b = 2*(x-obs_x)*x_dot + 2*(z-obs_z)*z_dot
        barrier_dot = 2*(x - self.cfg["obs_x"])*x_dot + 2*(z - self.cfg["obs_z"])*z_dot
        
        # Second derivative: Lf^2 b = 2*(x_dot^2 + z_dot^2)
        Lf2b = 2*(x_dot**2 + z_dot**2)
        
        # Control matrix: Lg Lf b for quadrotor dynamics
        # For quadrotor: v̇x = -sin(θ)/m * Σu, v̇z = -g + cos(θ)/m * Σu
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
        m = 1.0  # Default mass from quad2D.py
        
        # Lg Lf b = [∂(Lf b)/∂u_right, ∂(Lf b)/∂u_left]
        # This is a simplified version - in practice, need full quadrotor dynamics
        LgLfbu1 = torch.reshape(-2*(x - self.cfg["obs_x"])*sin_theta/m + 2*(z - self.cfg["obs_z"])*cos_theta/m, (nBatch, 1))
        LgLfbu2 = torch.reshape(-2*(x - self.cfg["obs_x"])*sin_theta/m + 2*(z - self.cfg["obs_z"])*cos_theta/m, (nBatch, 1))
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        
        # CBF constraint
        h = torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1))
        
        return self._solve_qp(Q, x31, G, h, sgn, nBatch)

    def _dCBF_quad3D(self, x0, x31, x32, sgn, nBatch, Q):
        """CBF for Quad3D - relative degree 1 (superquadratic barrier)"""
        # Extract state variables: [x, vx, y, vy, z, vz]
        px, vx = x0[:,0], x0[:,1]
        py, vy = x0[:,2], x0[:,3]
        pz, vz = x0[:,4], x0[:,5]
        
        # Superquadratic barrier: h(x) = ||x-x_obs||^4 - R^4
        barrier = (px - self.cfg["obs_x"])**4 + (py - self.cfg["obs_y"])**4 + (pz - self.cfg["obs_z"])**4 - self.cfg["R"]**4
        
        # First derivative: Lf b = 4*Σ(xi-obs_i)^3*vi
        barrier_dot = 4*(px - self.cfg["obs_x"])**3*vx + 4*(py - self.cfg["obs_y"])**3*vy + 4*(pz - self.cfg["obs_z"])**3*vz
        
        # Second derivative: Lf^2 b = 12*Σ(xi-obs_i)^2*vi^2
        Lf2b = 12*(px - self.cfg["obs_x"])**2*vx**2 + 12*(py - self.cfg["obs_y"])**2*vy**2 + 12*(pz - self.cfg["obs_z"])**2*vz**2
        
        # Control matrix: Lg b = [∂h/∂x, ∂h/∂y, ∂h/∂z] (relative degree 1)
        LgLfbu1 = torch.reshape(4*(px - self.cfg["obs_x"])**3, (nBatch, 1))
        LgLfbu2 = torch.reshape(4*(py - self.cfg["obs_y"])**3, (nBatch, 1))
        LgLfbu3 = torch.reshape(4*(pz - self.cfg["obs_z"])**3, (nBatch, 1))
        G = torch.cat([-LgLfbu1, -LgLfbu2, -LgLfbu3], dim=1)
        
        # CBF constraint: -Lg b * u <= Lf b + p1 * b
        h = torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1))
        
        return self._solve_qp(Q, x31, G, h, sgn, nBatch)

    def _solve_qp(self, Q, x31, G, h, sgn, nBatch):
        """Solve the QP problem for CBF-based control."""
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        h = torch.reshape(h, (nBatch, 1)).to(self.device)
        e = Variable(torch.Tensor()).to(self.device)
        
        if self.training or sgn == 1:
            # Use QPFunction for training
            x = QPFunction(verbose=0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            # Use test_solver for inference
            x = test_solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
        
        return x

    @staticmethod
    def load_from_checkpoint(ckpt_path, meta_path=None, device="cpu"):
        """
        Load a BarrierNet model from checkpoint and metadata.
        
        Args:
            ckpt_path: Path to the .pth checkpoint file
            meta_path: Path to the metadata JSON file (optional, auto-detected if None)
            device: Device to load the model on
            
        Returns:
            Instantiated BarrierNet model ready for inference
        """
        if meta_path is None:
            meta_path = ckpt_path.replace(".pth", "_meta.json")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Determine robot model from metadata (preferred) or checkpoint path
        robot_model = meta.get("robot_model")
        if robot_model is None:
            # Fallback to path-based detection
            if "2D_Robot" in ckpt_path:
                robot_model = "DynamicUnicycle2D"  # Default for 2D
            elif "3D_Robot" in ckpt_path:
                robot_model = "Quad3D"  # Default for 3D
            else:
                raise ValueError("Cannot determine robot model from checkpoint path")
        
        # Validate robot model
        if robot_model not in ROBOT_CFG:
            raise ValueError(f"Unknown robot model in metadata: {robot_model}")
        
        # Create model
        model = BarrierNet(
            robot_model=robot_model,
            mean=np.array(meta["mean"]),
            std=np.array(meta["std"]),
            device=device,
            obs_positions=meta.get("obs_positions"),
            bn=False
        )
        
        # Load weights
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        return model 