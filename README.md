# safe_control

`safe_control` is a Python library that provides a unified codebase for safety-critical controllers in robotic navigation. It implements various control barrier function (CBF) based controllers and other types of safety-critical controllers.

## Features

- Implementation of various safety-critical controllers, including `CBF-QP` and `MPC-CBF`
- Support for different robot dynamics models (e.g., unicycle, double integrator)
- Support sensing and mapping simulation for RGB-D type camera sensors (limited FOV)
- Support both single and multi agents navigation
- Interactive plotting and visualization

## Installation

To install `safe_control`, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/safe_control.git
   cd safe_control
   ```

2. (Optional) Create and activate a virtual environment:

3. Install the package and its dependencie:
   ```bash
   python -m pip install -e .
   ```
   Or, install packages manually (see [`setup.py`](https://github.com/tkkim-robot/safe_control/blob/main/setup.py)).

This will install `safe_control` in editable mode, allowing you to modify the source code and immediately see the effects without reinstalling.


# Getting Started

Familiarize with APIs and examples with the scripts in [`tracking.py`](https://github.com/tkkim-robot/safe_control/blob/main/tracking.py)

## Basic Example
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

|      Navigation with MPC-CBF controller            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/3e294698-b8e4-45a9-b6c2-55b1779cd5e5"  height="350px"> |

The green points are the given waypoints, and the blue area is the accumulated sensing footprints.

The gray circles are the obstacles that are known a priori.

## Detect/Avoid Unknown Obstacles
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


## Multi-Robot Example

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


# Module Breakdown

## Dynamics

Supported robot dynamics can be found in the [`robots/`](https://github.com/tkkim-robot/safe_control/tree/main/robots) directory:

- `unicycle2D`
- `dynamic_unicycle2D`: A unicycle model that uses velocity as state and acceleration as input.
- `double_integrator2D`
- `double_integrator2D with camera angle`: under development

## Positional Control Algorithms

Supported positional controllers can be found in the [`position_control/`](https://github.com/tkkim-robot/safe_control/tree/main/position_control) directory:

- `cbf_qp`: A simple CBF-QP controller for collision avoidance (ref: [[1]](https://ieeexplore.ieee.org/document/8796030))
- `mpc_cbf`: An MPC controller using discrete-time CBF (ref: [[2]](https://ieeexplore.ieee.org/document/9483029))
- `optimal_decay_cbf_qp`: A modified CBF-QP for point-wise feasibility guarantee (ref: [[3]](https://ieeexplore.ieee.org/document/9482626))
- `optimal_decay_mpc_cbf`: The same technique applied to MPC-CBF (ref: [[4]](https://ieeexplore.ieee.org/document/9683174))

To use a specific control algorithm, specify it when initializing the `LocalTrackingController`:

```python
controller = LocalTrackingController(..., control_type='cbf_qp', ...)
```

## Attitude Control Algorithms

Supported attitude controllers can be found in the [`attitude_control/`](https://github.com/tkkim-robot/safe_control/tree/main/attitude_control) directory:

- `gatekeeper`: under development (ref: [[5]](https://ieeexplore.ieee.org/document/10665919))

## Customizing Environments

You can modify the environment in [`utils/env.py](https://github.com/tkkim-robot/safe_control/blob/main/utils/env.py).

## Visualization
The online visualization is performed using [`matplotlib.pyplot.ion`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ion.html).

It allows you to interatively scroll, resize, change view, etc. during simulation.

You can visualize and save animation by setting the arguments:
```python
controller = LocalTrackingController(..., show_animation=True, save_animation=True)
```

# Citing

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2024learning, 
    author    = {Taekyung Kim and Robin Inho Kee and Dimitra Panagou},
    title     = {Learning to Refine Input Constrained Control Barrier Functions via Uncertainty-Aware Online Parameter Adaptation}, 
    booktitle = {},
    shorttitle = {Online-Adaptive-CBF},
    year      = {2024}
}
```

# Related Works

This repository has been utilized in several other projects. Here are some repositories that build upon or use components from this library:

- [Visibility-Aware RRT*](https://github.com/tkkim-robot/visibility-rrt): Safety-critical Global Path Planning (GPP) using Visibility Control Barrier Functions
- [Online Adaptive CBF](https://github.com/tkkim-robot/online_adaptive_cbf): Online adaptation of CBF parameters for input constrained robot systems
- [Multi-Robot Exploration and Mapping](): to be public soon
- [UGV Experiments with ROS2](https://github.com/tkkim-robot/px4_ugv_exp): Environmental setup for rovers using PX4, ros2 humble, Vicon MoCap, and NVIDIA VSLAM + NvBlox