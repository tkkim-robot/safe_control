#!/usr/bin/env python3
"""
Test script for BarrierNetController
"""

import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from barriernet_controller import BarrierNetController

def test_controller():
    """Test the BarrierNetController with dummy data."""
    
    # Test robot specification
    robot_spec = {
        "model": "DynamicUnicycle2D",
        "a_max": 0.5,
        "w_max": 0.5,
        "v_max": 1.0
    }
    
    # Test checkpoint path (this should exist)
    ckpt_path = "BarrierNet/2D_Robot/model_bn.pth"
    
    try:
        # Initialize controller
        print("Initializing BarrierNetController...")
        controller = BarrierNetController(
            robot_spec=robot_spec,
            ckpt_path=ckpt_path,
            device="cpu"
        )
        print("✓ Controller initialized successfully")
        
        # Test robot state and goal
        robot_state = np.array([0.0, 0.0, 0.0, 0.5])  # [x, y, theta, v]
        control_ref = {"goal": np.array([2.0, 1.0])}
        
        # Test control computation
        print("Computing control...")
        control = controller.solve_control_problem(robot_state, control_ref)
        print(f"✓ Control computed: {control}")
        
        # Test obstacle position update
        print("Testing obstacle position update...")
        controller.update_obstacle_positions([30.0, 10.0, 5.0])
        print("✓ Obstacle positions updated")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_controller() 