# barriernet_controller.py
"""
BarrierNet inference controller for real-time robot control.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, Any, Optional, List

# Import robot classes for control bounds
from safe_control.robots.dynamic_unicycle2D import DynamicUnicycle2D
from safe_control.robots.quad2D import Quad2D
from safe_control.robots.quad3D import Quad3D

# Import the BarrierNet model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'BarrierNet'))
from safe_control.position_control.BarrierNet.models import BarrierNet


class BarrierNetController:
    def __init__(self,
                 robot_spec: Dict[str, Any],
                 ckpt_path: str,
                 meta_path: Optional[str] = None,
                 obs_positions: Optional[List[float]] = None,
                 device: str = "auto"):
        """
        Initialize BarrierNet controller for inference.
        
        Args:
            robot_spec: Robot specification dictionary containing 'model' key
            ckpt_path: Path to the trained model checkpoint (.pth file)
            meta_path: Path to metadata JSON file (auto-detected if None)
            obs_positions: Override obstacle positions [x, y, z, R] or [x, y, R]
            device: Device to run inference on ("auto", "cpu", "cuda")
            
        Example:
            robot_spec = {"model": "Quad3D"}
            controller = BarrierNetController(robot_spec, "model_bn.pth")
            control = controller.solve_control_problem(robot_state, {"goal": goal})
        """
        self.robot_spec = robot_spec
        self.ckpt_path = ckpt_path
        self.robot_model = robot_spec["model"]
        
        # Robot model to robot family mapping
        self._robot_family_map = {
            "DynamicUnicycle2D": "2D_robot",
            "Quad2D": "2D_robot", 
            "Quad3D": "3D_robot"
        }
        
        # Determine robot family from robot_spec
        self.robot_family = self._determine_robot_family(self.robot_model)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and metadata
        self.model = self._load_model(ckpt_path, meta_path)
        
        # Store obstacle positions (can be updated during inference)
        self.obs_positions = obs_positions
    
    def _determine_robot_family(self, robot_model: str) -> str:
        """Determine robot family from robot model name."""
        if robot_model in self._robot_family_map:
            return self._robot_family_map[robot_model]
        else:
            raise ValueError(f"Unknown robot model: {robot_model}. "
                           f"Supported models: {list(self._robot_family_map.keys())}")
    
    def _load_model(self, ckpt_path: str, meta_path: Optional[str] = None) -> BarrierNet:
        """Load BarrierNet model from checkpoint and metadata."""
        if meta_path is None:
            meta_path = ckpt_path.replace(".pth", "_meta.json")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Verify robot model consistency
        meta_robot_model = meta.get("robot_model")
        if meta_robot_model and meta_robot_model != self.robot_model:
            print(f"Warning: Metadata robot_model ({meta_robot_model}) differs from robot_spec ({self.robot_model})")
        
        # Create model
        model = BarrierNet(
            robot_model=self.robot_model,
            mean=np.array(meta["mean"]),
            std=np.array(meta["std"]),
            device=self.device,
            obs_positions=meta.get("obs_positions"),
            bn=False
        )
        
        # Load weights
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model.eval()
        
        return model
    
    def _extract_features(self, robot_state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Extract features from robot state and goal for BarrierNet input.
        
        Args:
            robot_state: Robot state vector (varies by robot model)
            goal: Goal position [x, y] or [x, y, z]
            
        Returns:
            Feature vector for BarrierNet
        """
        if self.robot_model == "DynamicUnicycle2D":
            # State: [x, y, theta, v]
            # Features: [px, py, theta, v, dst_y] (dst_x is fixed at 50.0)
            x, y, theta, v = robot_state
            dst_x = 50.0  # Fixed destination x (as in original training)
            dst_y = goal[1] if len(goal) >= 2 else goal[0]
            features = np.array([x, y, theta, v, dst_y])
            
        elif self.robot_model == "Quad2D":
            # State: [x, z, theta, x_dot, z_dot, theta_dot]
            # Features: [x, z, theta, x_dot, z_dot, theta_dot]
            features = robot_state[:6]
            
        elif self.robot_model == "Quad3D":
            # State: [x, y, z, θ, φ, ψ, vx, vy, vz, q, p, r]
            # Features: [x, vx, y, vy, z, vz]
            x, y, z = robot_state[0], robot_state[1], robot_state[2]
            vx, vy, vz = robot_state[6], robot_state[7], robot_state[8]
            features = np.array([x, vx, y, vy, z, vz])
            
        else:
            raise ValueError(f"Feature extraction not implemented for {self.robot_model}")
        
        return features
    
    def solve_control_problem(self,
                             robot_state: np.ndarray,
                             control_ref: Dict[str, Any],
                             obs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve control problem using BarrierNet.
        
        Args:
            robot_state: Current robot state
            control_ref: Dictionary containing 'goal' key
            obs: Optional obstacle information (not used in current implementation)
            
        Returns:
            Control input as numpy array
        """
        # Extract goal from control_ref
        goal = control_ref['goal']
        
        # Extract features
        features = self._extract_features(robot_state, goal)
        
        # Update obstacle positions if provided
        if self.obs_positions is not None:
            # Update the model's obstacle configuration
            if self.robot_family == "2D_robot":
                self.model.cfg["obs_x"] = self.obs_positions[0]
                self.model.cfg["obs_y"] = self.obs_positions[1]
                if len(self.obs_positions) > 2:
                    self.model.cfg["R"] = self.obs_positions[2]
            elif self.robot_family == "3D_robot":
                self.model.cfg["obs_x"] = self.obs_positions[0]
                self.model.cfg["obs_y"] = self.obs_positions[1]
                self.model.cfg["obs_z"] = self.obs_positions[2]
                if len(self.obs_positions) > 3:
                    self.model.cfg["R"] = self.obs_positions[3]
        
        # Normalize features
        features_normalized = (features - self.model.mean.cpu().numpy()) / self.model.std.cpu().numpy()
        
        # Convert to tensor and ensure double precision
        features_tensor = torch.from_numpy(features_normalized).double().unsqueeze(0).to(self.device)
        
        # Run inference (sgn=0 for inference mode)
        with torch.no_grad():
            control_output = self.model(features_tensor, sgn=0)
        
        # Convert output to numpy array
        if isinstance(control_output, torch.Tensor):
            control = control_output.cpu().numpy().flatten()
        else:
            # Handle test_solver output
            control = np.array(control_output).flatten()
        
        # Clip control to robot input bounds
        control = self._clip_control(control)
        
        return control
    
    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        """Clip control inputs to robot-specific bounds from robot classes."""
        if self.robot_model == "DynamicUnicycle2D":
            # Control: [a, omega]
            a_max = DynamicUnicycle2D(0.1, {}).robot_spec.get('a_max', 0.5)
            w_max = DynamicUnicycle2D(0.1, {}).robot_spec.get('w_max', 0.5)
            control[0] = np.clip(control[0], -a_max, a_max)  # acceleration
            control[1] = np.clip(control[1], -w_max, w_max)  # angular velocity
        elif self.robot_model == "Quad2D":
            # Control: [u_right, u_left]
            f_min = Quad2D(0.1, {}).robot_spec.get('f_min', 1.0)
            f_max = Quad2D(0.1, {}).robot_spec.get('f_max', 10.0)
            control[0] = np.clip(control[0], f_min, f_max)  # right rotor force
            control[1] = np.clip(control[1], f_min, f_max)  # left rotor force
        elif self.robot_model == "Quad3D":
            # Control: [u1, u2, u3, u4] - motor forces
            u_min = Quad3D(0.1, {}).robot_spec.get('u_min', -10.0)
            u_max = Quad3D(0.1, {}).robot_spec.get('u_max', 10.0)
            for i in range(len(control)):
                control[i] = np.clip(control[i], u_min, u_max)
        
        return control
    
    def update_obstacle_positions(self, obs_positions: List[float]):
        """Update obstacle positions for the controller."""
        self.obs_positions = obs_positions 