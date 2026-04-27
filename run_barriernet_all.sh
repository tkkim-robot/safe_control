#!/usr/bin/env bash
set -euo pipefail

# End-to-end BarrierNet pipeline runner:
# - activate conda env `cbf`
# - generate datasets for 4 robot models
# - train BarrierNet models
# - run one-step deployment inference sanity checks
#
# This script is designed to work whether `safe_control` is:
# - a standalone repo, or
# - a subrepo inside a larger project.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Always run from the parent directory so `import safe_control` works naturally.
cd "${PARENT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cbf
else
  echo "ERROR: conda not found on PATH. Please install conda/mamba and ensure 'conda' is available." >&2
  exit 1
fi

# ----------------------------
# Configuration (edit here)
# ----------------------------
# Target number of *raw* (pre-split) state-action rows per model.
# For a fair comparison to a 200k-sample method, set BN_TARGET_ROWS=200000.
BN_TARGET_ROWS="${BN_TARGET_ROWS:-200000}"
BN_MAX_SIMS="${BN_MAX_SIMS:-3000}"

# Training hyperparams
BN_EPOCHS="${BN_EPOCHS:-100}"
BN_BATCH_SIZE="${BN_BATCH_SIZE:-128}"
BN_LR="${BN_LR:-1e-3}"
BN_DEVICE="${BN_DEVICE:-auto}"

export BN_TARGET_ROWS BN_MAX_SIMS BN_EPOCHS BN_BATCH_SIZE BN_LR BN_DEVICE

MODELS=("DynamicUnicycle2D" "Quad2D" "Quad3D" "KinematicBicycle2D_DPCBF")

echo "=== BarrierNet: data_gen -> train -> deploy-check ==="
echo "BN_TARGET_ROWS=${BN_TARGET_ROWS} BN_MAX_SIMS=${BN_MAX_SIMS}"
echo "BN_EPOCHS=${BN_EPOCHS} BN_BATCH_SIZE=${BN_BATCH_SIZE} BN_LR=${BN_LR} BN_DEVICE=${BN_DEVICE}"
echo

for M in "${MODELS[@]}"; do
  echo "=== [${M}] Generating dataset ==="
  BN_ROBOT_MODEL="${M}" python safe_control/position_control/BarrierNet/generate_dataset.py
  echo

  echo "=== [${M}] Training ==="
  BN_ROBOT_MODEL="${M}" python safe_control/position_control/BarrierNet/train.py
  echo
done

echo "=== Deployment sanity checks (one step) ==="
python - <<'PY'
import numpy as np
from safe_control.position_control.barriernet_controller import BarrierNetController

def _default_u_ref(model: str):
    # BarrierNet now requires u_ref (nominal control) for its QP objective.
    # For this *sanity check only*, we use zeros of the correct control dimension.
    if model in ("DynamicUnicycle2D", "Quad2D", "KinematicBicycle2D_DPCBF"):
        return np.zeros(2, dtype=float)
    if model == "Quad3D":
        return np.zeros(4, dtype=float)
    raise ValueError(model)

def run(model, ckpt, robot_spec, x, goal, obs):
    c = BarrierNetController(robot_spec, ckpt)
    u_ref = _default_u_ref(model)
    u = c.solve_control_problem(
        np.array(x, dtype=float),
        {"goal": np.array(goal, dtype=float), "u_ref": u_ref},
        np.array(obs, dtype=float),
    )
    print(f"{model}: status={c.status} u={u}")

run(
    "DynamicUnicycle2D",
    "safe_control/position_control/BarrierNet/checkpoints/DynamicUnicycle2D_barriernet.pth",
    {"model":"DynamicUnicycle2D","a_max":0.5,"w_max":0.5,"radius":0.25},
    [1.0,1.0,0.2,0.5],
    [8.0,2.0],
    [[2.0,2.0,0.3,0.0,0.0,0.0,0.0],[3.0,1.5,0.4,0.0,0.0,0.0,0.0]],
)

run(
    "Quad2D",
    "safe_control/position_control/BarrierNet/checkpoints/Quad2D_barriernet.pth",
    {"model":"Quad2D","f_min":1.0,"f_max":10.0,"radius":0.25},
    [0.5,1.0,0.0,0.0,0.0,0.0],
    [5.0,2.0],
    [[2.0,2.0,0.4,0.0,0.0,0.0,0.0]],
)

run(
    "Quad3D",
    "safe_control/position_control/BarrierNet/checkpoints/Quad3D_barriernet.pth",
    {"model":"Quad3D","u_min":-10.0,"u_max":10.0,"radius":0.25},
    [0.0,0.0,1.0,0.0,0.0,0.0, 0.2,-0.1,0.0, 0.0,0.0,0.0],
    [3.0,2.0,1.0],
    [[1.0,1.0,0.5,0.0,0.0,0.0,0.0],[2.0,0.0,0.3,0.0,0.0,0.0,0.0]],
)

run(
    "KinematicBicycle2D_DPCBF",
    "safe_control/position_control/BarrierNet/checkpoints/KinematicBicycle2D_DPCBF_barriernet.pth",
    {"model":"KinematicBicycle2D_DPCBF","a_max":5.0,"beta_max":0.6,"rear_ax_dist":0.2,"wheel_base":0.4,"radius":0.3},
    [1.0,1.0,0.0,0.5],
    [8.0,2.0],
    [[2.0,2.0,0.3,0.1,-0.2,0.0,0.0],[3.0,1.5,0.4,0.0,0.0,0.0,0.0]],
)
PY

echo
echo "=== Done ==="


