#!/usr/bin/env bash
# ==============================================================================
# Lightweight pip venv setup for DynaArm RL (ROS 2 Jazzy compatible)
# Uses --system-site-packages so the venv sees ROS 2 rclpy/messages while
# keeping PyTorch & MuJoCo isolated in .venv.
#
# Usage:
#   ./setup_env.sh          # Fast robot-PC deploy install (CPU PyTorch + MuJoCo)
#   ./setup_env.sh --train  # Full training/eval/render install (default PyTorch)
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
MODE="${1:-deploy}"

echo "[*] Creating virtual environment with --system-site-packages at ${VENV_DIR}..."
python3 -m venv --system-site-packages "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip

if [[ "${MODE}" == "--train" ]]; then
    echo "[*] Installing full training + evaluation + rendering dependencies..."
    pip install -r "${REPO_DIR}/requirements.txt"
else
    echo "[*] Installing lightweight robot-PC deploy dependencies (CPU PyTorch + MuJoCo)..."
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r "${REPO_DIR}/requirements-deploy.txt"
fi

echo "[*] Verifying installation..."
python3 -c "import numpy, torch, mujoco; print(f'[✓] numpy={numpy.__version__}, torch={torch.__version__}, mujoco={mujoco.__version__}')"
python3 "${REPO_DIR}/rl/deploy_ros2_node.py" --self-test --target-mode helix --residual-gain 1.0

echo ""
echo "[✓] Environment ready! Run deployment with:"
echo "    ./run_deploy.sh --controller-mode qp_osc --target-mode helix --residual-gain 0.0 --speed-scale 0.3"
