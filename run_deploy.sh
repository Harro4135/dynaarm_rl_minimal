#!/usr/bin/env bash
# ==============================================================================
# Launcher for DynaArm 6D Pose RL ROS 2 deployment node
# Automatically sources ROS 2 Jazzy + .venv and exports CycloneDDS settings.
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[!] .venv not found. Run ./setup_env.sh first."
    exit 1
fi

if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-8}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

exec python3 "${REPO_DIR}/rl/deploy_ros2_node.py" "$@"
