#!/usr/bin/env python3
"""
BarrierNet dataset generation using safe_control baseline controllers.

Generates train/valid/test .mat files containing rows:
  [features..., labels...]
where features = [state, goal, K obstacles (7D each)] and labels = baseline controller output u.

Important: we intentionally DO NOT remove input bounds in the baseline controller.
The "no input bounds" rule applies to BarrierNet training (QP layer), not data collection.

Edit the CONFIG section below to change behavior (no CLI required).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import scipy.io as sio

# Ensure *parent of safe_control* is on PYTHONPATH so `import safe_control` works
# both when:
# - online_adaptive_cbf imports safe_control as a subrepo, and
# - user runs scripts from inside the safe_control repo.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from safe_control.position_control.BarrierNet.models import ROBOT_CFG
from safe_control.position_control.barriernet_controller import BarrierNetController
from safe_control.tracking import LocalTrackingController, InfeasibleError
from safe_control.dynamic_env.main import LocalTrackingControllerDyn
from safe_control.utils import env


# =========================
# CONFIG (edit in code)
# =========================
ROBOT_MODEL = os.environ.get(
    "BN_ROBOT_MODEL",
    "DynamicUnicycle2D",
)  # DynamicUnicycle2D, Quad2D, Quad3D, KinematicBicycle2D_DPCBF

N_SIMS = 10
DT = 0.05
T_MAX = 30.0

# Optional env overrides (for automation scripts)
BN_TARGET_ROWS = int(os.environ.get("BN_TARGET_ROWS", "0"))  # if >0, collect at least this many rows (pre-split)
BN_MAX_SIMS = int(os.environ.get("BN_MAX_SIMS", "0"))        # safety cap when BN_TARGET_ROWS is used (0 => use N_SIMS)
N_SIMS = int(os.environ.get("BN_N_SIMS", str(N_SIMS)))
DT = float(os.environ.get("BN_DT", str(DT)))
T_MAX = float(os.environ.get("BN_T_MAX", str(T_MAX)))

# ---------------------------------------------------------
# FAIR COMPARISON ENV (match gat_data_generation.py)
# ---------------------------------------------------------
# Waypoints: start near (2.5,2.0), goal near (9.5,2.0)
START_XY = np.array([2.5, 2.0], dtype=np.float64)
GOAL_XY = np.array([9.5, 2.0], dtype=np.float64)
GOAL_XYZ = np.array([9.5, 2.0, 0.0], dtype=np.float64)

# Obstacles: count uniform in [2,10], placement/radius ranges + clearance rules
OBS_N_MIN, OBS_N_MAX = 2, 10
OBS_X_MIN, OBS_X_MAX = 0.5, 7.5
OBS_Y_MIN, OBS_Y_MAX = 0.5, 3.5
OBS_R_MIN, OBS_R_MAX = 0.2, 0.4
OBS_MAX_ATTEMPTS = 500
START_GOAL_MARGIN = 0.5  # safety margin in gat_data_generation.py

# dataset split
TRAIN_FRAC = 0.8
VALID_FRAC = 0.1

# output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


"""
Dataset rows are stored as:
  [z (z_dim), ctx (ctx_dim), u* (n_u)]
where:
  - z is the NN input built from closest obstacles by signed clearance
  - ctx is the raw context used by the BarrierNet QP constraints
"""


def _robot_spec_for(model: str) -> dict:
    if model == "DynamicUnicycle2D":
        return {"model": "DynamicUnicycle2D", "w_max": 0.5, "a_max": 0.5, "v_max": 1.0, "radius": 0.25}
    if model == "Quad2D":
        return {"model": "Quad2D", "f_min": 3.0, "f_max": 10.0, "radius": 0.25}
    if model == "Quad3D":
        return {"model": "Quad3D", "u_min": -10.0, "u_max": 10.0, "radius": 0.25}
    if model == "KinematicBicycle2D_DPCBF":
        return {"model": "KinematicBicycle2D_DPCBF", "a_max": 5.0, "beta_max": float(np.deg2rad(32)), "radius": 0.3}
    raise ValueError(model)


def _baseline_controller_type(model: str) -> dict:
    # Follow safe_control baseline usage:
    # - Use MPC-CBF for DynamicUnicycle2D / Quad2D / Quad3D (user requested)
    if model in ("DynamicUnicycle2D", "Quad2D", "Quad3D"):
        return {"pos": "mpc_cbf"}
    return {"pos": "cbf_qp"}

def _sample_obstacles_like_gat(model: str, robot_radius: float, num_obstacles: int) -> np.ndarray:
    """
    Sample obstacles using the same distribution/filters as gat_data_generation.py.
    Returns obs array (N,7): [x, y, r, vx, vy, extra, flag] with vx=vy=0 for fairness.
    """
    min_gap = robot_radius * 2.0
    obstacles = []
    attempts = 0
    while len(obstacles) < num_obstacles and attempts < OBS_MAX_ATTEMPTS:
        ox = np.random.uniform(OBS_X_MIN, OBS_X_MAX)
        oy = np.random.uniform(OBS_Y_MIN, OBS_Y_MAX)
        radius = np.random.uniform(OBS_R_MIN, OBS_R_MAX)

        # Skip if too close to start or goal (accounting for radii + margin)
        dist_start = np.hypot(ox - START_XY[0], oy - START_XY[1])
        dist_goal = np.hypot(ox - GOAL_XY[0], oy - GOAL_XY[1])
        min_clear_start = radius + robot_radius + START_GOAL_MARGIN
        min_clear_goal = radius + robot_radius + START_GOAL_MARGIN
        if dist_start < min_clear_start or dist_goal < min_clear_goal:
            attempts += 1
            continue

        valid = True
        for ex, ey, er, *_ in obstacles:
            center_dist = np.hypot(ox - ex, oy - ey)
            min_clearance = er + radius + min_gap
            if center_dist < min_clearance * 1.2:
                valid = False
                break
        if not valid:
            attempts += 1
            continue

        obstacles.append([ox, oy, radius, 0.0, 0.0, 0.0, 0.0])  # flag=0 circle
        attempts += 1

    return np.array(obstacles, dtype=np.float64)


def run_single_sim(model: str, sim_id: int) -> np.ndarray | None:
    """
    Run a single baseline rollout and return data rows (N, z_dim+ctx_dim+n_u), or None if failed.
    """
    robot_spec = _robot_spec_for(model)
    controller_type = _baseline_controller_type(model)

    # Simple environment
    env_handler = env.Env()

    # sample obstacles like gat_data_generation.py (random count 2..10)
    n_obs = int(np.random.randint(OBS_N_MIN, OBS_N_MAX + 1))
    robot_radius = float(robot_spec.get("radius", 0.25))
    known_obs = _sample_obstacles_like_gat(model, robot_radius=robot_radius, num_obstacles=n_obs)
    if known_obs.size == 0:
        # keep controller happy; will be padded later for features
        known_obs = np.zeros((0, 7), dtype=np.float64)

    # init state
    theta0 = float(np.random.uniform(-0.01, 0.01))  # match gat theta_range
    if model == "DynamicUnicycle2D":
        v0 = float(np.random.uniform(0.0, 1.0))
        x0 = np.array([START_XY[0], START_XY[1], theta0, v0], dtype=np.float64)
    elif model == "Quad2D":
        vx_init = float(np.random.uniform(0.0, 1.0))
        vz_init = float(np.random.uniform(0.0, 1.0))
        x0 = np.array([START_XY[0], START_XY[1], theta0, vx_init, vz_init, 0.0], dtype=np.float64)
    elif model == "Quad3D":
        # [x,y,z,theta,phi,psi,vx,vy,vz,q,p,r]
        vx_init = float(np.random.uniform(0.0, 1.0))
        vz_init = float(np.random.uniform(0.0, 1.0))
        x0 = np.array([START_XY[0], START_XY[1], 0.0, 0.0, 0.0, 0.0, vx_init, 0.0, vz_init, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "KinematicBicycle2D_DPCBF":
        v0 = float(np.random.uniform(0.2, 1.0))  # match DPCBF min speed
        x0 = np.array([START_XY[0], START_XY[1], theta0, v0], dtype=np.float64)
    else:
        raise ValueError(model)

    # waypoints
    if model == "Quad3D":
        waypoints = np.array([[START_XY[0], START_XY[1], 0.0], [GOAL_XYZ[0], GOAL_XYZ[1], GOAL_XYZ[2]]], dtype=np.float64)
    else:
        waypoints = np.array([[START_XY[0], START_XY[1], theta0], [GOAL_XY[0], GOAL_XY[1], 0.0]], dtype=np.float64)

    # controller
    try:
        if model == "KinematicBicycle2D_DPCBF":
            ctrl = LocalTrackingControllerDyn(
                x0, robot_spec,
                controller_type=controller_type,
                dt=DT,
                show_animation=False,
                save_animation=False,
                env=env_handler,
            )
        else:
            ctrl = LocalTrackingController(
                x0, robot_spec,
                controller_type=controller_type,
                dt=DT,
                show_animation=False,
                save_animation=False,
                env=env_handler,
            )
    except Exception as e:
        print(f"[sim {sim_id}] controller init failed for {model}: {e}")
        return None

    # seed known obstacles + goal
    ctrl.obs = known_obs
    ctrl.set_waypoints(waypoints)

    rows = []
    max_steps = int(T_MAX / DT)
    for _ in range(max_steps):
        try:
            ret = ctrl.control_step()
        except InfeasibleError as e:
            print(f"[sim {sim_id}] infeasible for {model}: {e}")
            return None
        except Exception as e:
            print(f"[sim {sim_id}] exception during control_step for {model}: {e}")
            return None

        # If the controller reported infeasible/collision, do not record this step
        # (the controller may not have a valid u_pos for this step).
        if ret == -2:
            break

        # goal may become None after completion
        cur_goal = ctrl.goal
        if cur_goal is None:
            break

        # Use the controller's current obstacle list; for dynamic env this is updated each step.
        obs_now = getattr(ctrl, "obs", None)

        # Nominal control for objective: use the robot's nominal tracking input toward the current goal.
        u_ref = np.array(ctrl.robot.nominal_input(cur_goal), dtype=np.float64).reshape(-1)
        n_u = ROBOT_CFG[model].n_u
        if u_ref.shape[0] < n_u:
            u_ref = np.pad(u_ref, (0, n_u - u_ref.shape[0]), mode="constant")
        elif u_ref.shape[0] > n_u:
            u_ref = u_ref[:n_u]

        # labels = applied control (expert / baseline output) (pad/slice to expected control dimension)
        u = np.array(ctrl.get_control_input(), dtype=np.float64).reshape(-1)
        if u.shape[0] < n_u:
            u = np.pad(u, (0, n_u - u.shape[0]), mode="constant")
        elif u.shape[0] > n_u:
            u = u[:n_u]
        z, ctx = BarrierNetController.build_z_ctx_for(model, robot_spec, ctrl.robot.X.flatten(), np.array(cur_goal).flatten(), obs_now)
        # row: [z, ctx, u_ref, u*]
        rows.append(np.concatenate([z, ctx, u_ref.astype(np.float64), u.astype(np.float64)], axis=0))

        if ret == -1:
            break

    if not rows:
        return None
    return np.vstack(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(0)

    all_rows = []
    ok = 0

    # collection loop: either fixed N_SIMS, or target-row driven
    max_sims = N_SIMS if BN_TARGET_ROWS <= 0 else (BN_MAX_SIMS if BN_MAX_SIMS > 0 else max(N_SIMS, 100))
    sim_id = 0
    while sim_id < max_sims:
        data = run_single_sim(ROBOT_MODEL, sim_id)
        if data is not None:
            all_rows.append(data)
            ok += 1
            if BN_TARGET_ROWS > 0:
                # row count before split
                cur_rows = sum(d.shape[0] for d in all_rows)
                if cur_rows >= BN_TARGET_ROWS:
                    break
        sim_id += 1

    if not all_rows:
        raise RuntimeError("No successful simulations; dataset is empty.")

    data = np.vstack(all_rows)
    n_total = data.shape[0]
    n_train = int(TRAIN_FRAC * n_total)
    n_valid = int(VALID_FRAC * n_total)

    train = data[:n_train]
    valid = data[n_train : n_train + n_valid]
    test = data[n_train + n_valid :]

    base = os.path.join(OUT_DIR, f"{ROBOT_MODEL}_data")
    sio.savemat(base + "_train.mat", {"data": train})
    sio.savemat(base + "_valid.mat", {"data": valid})
    sio.savemat(base + "_test.mat", {"data": test})

    print(f"Saved: {base}_train.mat ({train.shape})")
    print(f"Saved: {base}_valid.mat ({valid.shape})")
    print(f"Saved: {base}_test.mat  ({test.shape})")
    print(f"Successful sims: {ok}")


if __name__ == "__main__":
    main()


