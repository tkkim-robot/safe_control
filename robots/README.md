# Dynamic Parabolic Control Barrier Function

`kinematic_bicycle2D_dpcbf.py` details the implementation of the Dynamic Parabolic Control Barrier Function (DPCBF), a novel safety controller designed for the kinematic bicycle model within the `safe_control` library. This method is based on the paper:

Beyon Collision Cones: Dynamic Obstacle Avoidance for Nonholonomic Robots via Dynamic Parabolic Control Barrier Functions.

The core advantage of our DPCBF is its ability to reduce control conservatism compared to collision cone-based CBF methods, improving navigation success rates in dense, dynamic environments.

## How It Works

Collision cone-based methods define a fixed cone as unsafe set in the relative velocity space. This can be overly restrictive, as the robot is prevented from moving toward an obstacle regardless of its distance or relative speed.
DPCBF replaces this fixed cone with an adaptive parabolic safety boundary. The key idea is to define a safety boundary that dynamically adjusts its shape based on both the robot's clearance from the obstacle and the magnitude of their relative velocity.
This is achieved through the following steps:
- Coordinate Transformation: The relative position and velocity vectors are transformed from the world frame into a Line-of-Sight (LoS) frame.  In the LoS frame, the new x-axis aligns with the vecotir connecting the robot to the obstacle. This simplifies the geometry of the safety constraint.
- State-Dependent Parabolic: A parabolic boundary is defined in the LoS frame. Its shape is governed by two key functions, 
- CBF Formulation

## Dynamics


## Code Implementation
The following code implements the DPCBF for a `kinematicBicycle2D` model. It overrides the base class's `agent_barrier` method to compute the continous-time DPCBF and its gradient, which are then used by a CBF-QP controller.
```python
class KinematicBicycle2D_DPCBF(KinematicBicycle2D):
    def __init__(self, dt, robot_spec):
        super().__init__(dt, robot_spec)


## Getting Started

Familiarize with APIs and examples with the scripts in [`tracking.py`](https://github.com/tkkim-robot/safe_control/blob/main/tracking.py)

### Basic Example
You can run our test example by:

```bash
python tracking.py
```

Alternatively, you can import `LocalTrackingController` from [`tracking.py`](https://github.com/tkkim-robot/safe_control/blob/main/tracking.py).

```python
from safe_control.tracking import LocalTrackingController
controller = LocalTrackingController(x_init, robot_spec,
                                control_type=control_type,
                                dt=dt,
                                show_animation=True,
                                save_animation=False,
                                ax=ax, fig=fig,
                                env=env_handler)

# assuming you have set the workspace (environment) in utils/env.py
controller.set_waypoints(waypoints)
_ _= controller.run_all_steps(tf=100)
```

You can also design your own navigation logic using the controller. `contorl_step()` function simulates one time step.

```python
# self refers to the LocalTrackingController class.
for _ in range(int(tf / self.dt)):
    ret = self.control_step()
    self.draw_plot()
    if ret == -1:  # all waypoints reached
        break
self.export_video()
```

The sample results from the basic example:

|     mu(x)-only case              |              lambda(x)-only case        |              DPCBF        |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://github.com/user-attachments/assets/a3571760-b8f3-48cc-91cd-895e1252c0f7"  height="200px"> | <img src="https://github.com/user-attachments/assets/d22bb161-e5d6-4e78-ac30-48847bcf01aa"  height="200px"> | <img src="https://github.com/user-attachments/assets/b229e974-5343-4d56-affc-41682895daa6"  height="200px"> |


|      Navigation with MPC-CBF controller            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/78e6dfb4-8eca-42e5-b9a4-363965bf599e"  height="350px"> |

The green points are the given waypoints, and the blue area is the accumulated sensing footprints.

The gray circles are the obstacles that are known a priori.

### Detect/Avoid Unknown Obstacles
You can also simulate online detection of unknown obstacles and avoidance of obstacles that are detected on-the-fly using safety-critical constratins.
The configuration of the obstacles is (x, y, radius).

```python
unknown_obs = np.array([[2, 2, 0.2], [3.0, 3.0, 0.3]])
controller.set_unknown_obs(unknown_obs)

# then, run simulation
```

The unknown obstacles are visualized in orange.

|      Navigation with MPC-CBF controller            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/8be5453f-8629-4f1d-aa36-c0f9160fd2ee"  height="350px"> |


### Multi-Robot Example

You can simulate heterogeneous, multiple robots navigating in the same environment. 

```python
robot_spec = {
    'model': 'Unicycle2D',
    'robot_id': 0
    }
controller_0 = LocalTrackingController(x_init, robot_spec)

robot_spec = {
    'model': 'DynamicUnicycle2D',
    'robot_id': 1,
    'a_max': 1.5,
    'fov_angle': 90.0,
    'cam_range': 5.0,
    'radius': 0.25
}
controller_1 = LocalTrackingController(x_init, robot_spec)

# then, run simulation
```

First determine the specification of each robot (with different `id`), then run the simulation.

|     Homogeneous Robots              |              Heterogeneous Robots        |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://github.com/user-attachments/assets/d55518d3-79ec-46ec-8cfb-3f78dd7d6e82"  height="200px"> | <img src="https://github.com/user-attachments/assets/7c3db292-f9fa-4734-8578-3034e85ab4fb"  height="200px"> |


### Positional Control Algorithms

Supported positional controllers can be found in the [`position_control/`](https://github.com/tkkim-robot/safe_control/tree/main/position_control) directory:

- `cbf_qp`: A simple CBF-QP controller for collision avoidance (ref: [[1]](https://ieeexplore.ieee.org/document/8796030))

### Visualization
The online visualization is performed using [`matplotlib.pyplot.ion`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ion.html).

It allows you to interatively scroll, resize, change view, etc. during simulation.

You can visualize and save animation by setting the arguments:
```python
controller = LocalTrackingController(..., show_animation=True, save_animation=True)
```

## Citing

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2025how, 
    author    = {Kim, Taekyung and Beard, Randal W. and Panagou, Dimitra},
    title     = {How to Adapt Control Barrier Functions? A Learning-Based Approach with Applications to a VTOL Quadplane}, 
    booktitle = {IEEE Conference on Decision and Control (CDC)},
    shorttitle = {How to Adapt Control Barrier Functions},
    year      = {2025}
}
```

## Related Works

This repository has been utilized in several other projects. Here are some repositories that build upon or use components from this library:

- [Visibility-Aware RRT*](https://github.com/tkkim-robot/visibility-rrt): Safety-critical Global Path Planning (GPP) using Visibility Control Barrier Functions
- [Online Adaptive CBF](https://github.com/tkkim-robot/online_adaptive_cbf): Online adaptation of CBF parameters for input constrained robot systems
- [Multi-Robot Exploration and Mapping](): to be public soon
- [UGV Experiments with ROS2](https://github.com/tkkim-robot/px4_ugv_exp): Environmental setup for rovers using PX4, ros2 humble, Vicon MoCap, and NVIDIA VSLAM + NvBlox
- [Quadrotor Experiments with ROS2](https://github.com/RahulHKumar/px4_quad_exp): Environmental setup for quadrotors using PX4, ros2 humble, Vicon MoCap, and NVIDIA VSLAM + NvBlox
