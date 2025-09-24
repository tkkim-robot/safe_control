# Dynamic Parabolic Control Barrier Functions

`kinematic_bicycle2D_dpcbf.py` details the implementation of the Dynamic Parabolic Control Barrier Function (DPCBF), a safety controller designed for the kinematic bicycle model within the `safe_control` library. The core advantage of our DPCBF is its ability to reduce control conservatism compared to collision cone-based CBF methods, improving navigation success rates in dense, dynamic environments. Please see our paper ["Beyon Collision Cones: Dynamic Obstacle Avoidance for Nonholonomic Robots via Dynamic Parabolic Control Barrier Functions"]() for more details.

<div align="center">
<img src="https://github.com/user-attachments/assets/0f6c1b74-7f00-4340-a2a8-16da081e68bf" >
<div align="center">

[[Homepage]](https://www.taekyung.me/dpcbf)
[[Arxiv]](https://arxiv.org/abs/)
[[Video]](https://youtu.be/57qgoe7YJao)
[[Research Group]](https://dasc-lab.github.io/)

</div>
</div>

## How to Run Example
You can run our test example by:
```bash
python -m dynamic_env.main
```

### Comparison (Surrounded by Obstacles)
|      C3BF            |      MA-CBF-VO            |
| :-------------------------------: | :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/d5b8c868-1113-41b5-8e2d-c40b2551160a"  height="350px"> |  <img src="https://github.com/user-attachments/assets/3d2388df-e04b-4886-8e68-3f47377b62cc"  height="350px"> |

|      Dynamic Zone-based CBF            |      DPCBF            |
| :-------------------------------: | :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/430a9a41-068f-4763-ad70-2fcfe26f3a44"  height="350px"> |  <img src="https://github.com/user-attachments/assets/3eac2226-6b01-4ded-9448-55124637fe69"  height="350px"> |


### More Examples

|        max_obs_radius = 0.3 m          |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/c0e20a14-f8b6-41ab-ac90-85df3ab8775b"  height="350px"> |


|      max_obs_radius = 0.5 m            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/63bc2053-2cd6-4473-8718-302bc137670a"  height="350px"> |


|      max_obs_radius = 0.7 m            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/1506e504-d6e3-4cfa-a7ba-ef7b9138cdbb"  height="350px"> |


## Implementation Details

The following code implements the DPCBF for a `kinematicBicycle2D` model. It overrides the base class's `agent_barrier` method to compute the continous-time DPCBF and its gradient, which are then used by a CBF-QP controller.
```python
class KinematicBicycle2D_DPCBF(KinematicBicycle2D):
    def __init__(self, dt, robot_spec):
        super().__init__(dt, robot_spec)

```

Define the relative position and velocity between the robot and the obstacle.
```python
# Compute relative position and velocity
        p_rel = np.array([[obs[0] - X[0, 0]], 
                        [obs[1] - X[1, 0]]])
        v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                        [obs_vel_y - v * np.sin(theta)]])
# Compute norms
        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)
```

By rotating the coordinates by an angle rot_angle, the new x-axis of the new coordinate frame is aligned with the vector from the robot to the obstacle, p_rel. We refer to the rotated frame by the rotation matrix R as LoS frame. Then, we define the relative velocity in the LoS frame. This transformation simplifies the definition of the parabolic safety boundary.
```python
# Rotation angle and transformation
rot_angle = np.arctan2(p_rel_y, p_rel_x)
R = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
            [-np.sin(rot_angle),  np.cos(rot_angle)]])

# Transform v_rel into the new coordinate frame
v_rel_new = R @ v_rel
v_rel_new_x = v_rel_new[0, 0]
v_rel_new_y = v_rel_new[1, 0]
```

```python
# Compute clearance safely
eps = 1e-6
d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)
```

We introduce DPCBF functions with tunable parameters, which can adjust the curvature of the parabola and shift the parabola forward by the safe distance margin. Finally, we propose Dynamic Parabolic CBF (DPCBF).
```python
# Introduce tunable parameters
k_lamda, k_mu = 0.1 * np.sqrt(s**2 - 1)/ego_dim, 0.5 * np.sqrt(s**2 - 1)/ego_dim
# DPCBF functions
lamda = k_lamda * np.sqrt(d_safe) / v_rel_mag
mu = k_mu * np.sqrt(d_safe)

# Barrier function h(x)
h = v_rel_new_x + lamda * (v_rel_new_y**2) + mu
```

Three examples describe how a parabolic region in the new plane shapes the safety boundary:

|     mu(x)-only case              |              lambda(x)-only case        |              DPCBF        |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://github.com/user-attachments/assets/a3571760-b8f3-48cc-91cd-895e1252c0f7"  height="250px"> | <img src="https://github.com/user-attachments/assets/5114daea-75f0-4ea1-a575-e37837d8d19d"  height="250px"> | <img src="https://github.com/user-attachments/assets/e7898647-94bb-4dea-ac25-17e9004b68e3"  height="250px"> |

Now, our DPCBF formulation defines a parabolic safety boundary with a Line-of-Sight coordinate frame to explicitly consider both the distance to an obstacle and the relative velocity. DPCBF considers the robot is safe as long as its relative velocity vector stays outside the parabolic region.

## Citing

If you find this repository useful, please consider citing our paper:

```
@inproceedings{park2025dpcbf, 
    author    = {Park, Hun Kuk and Kim, Taekyung and Panagou, Dimitra},
    title     = {Beyond Collision Cones: Dynamic Obstacle Avoidance for Nonholonomic Robots via Dynamic Parabolic Control Barrier Functions}, 
    booktitle = {arXiv preprint arXiv.25},
    shorttitle = {DPCBF},
    year      = {2025}
}
```