# DynaArm 6D Pose RL Tracking (Minimal)

Minimal, self-contained repository for training, evaluating, exporting, and deploying the no-load **6D Cartesian pose tracking RL policy** for the Clearpath Ridgeback + Duatic DynaArm (with Orbbec Femto Bolt hand-mounted camera and Nov 19 Asymmetric Stribeck friction).

## Contents
- `models/ridgeback_duatic.xml` — Calibrated Nov 19 MuJoCo model (with hand-mounted Femto Bolt).
- `datasets/calibration_real/sysid_parameters.json` — Nov 19 Asymmetric Stribeck friction parameters.
- `rl/dynaarm_reach_env.py` — 500 Hz 6-DOF Box-Constrained QP-OSC + 2nd-order CBF + Asymmetric Stribeck + latency/noise DR Gymnasium environment.
- `rl/train_ppo_overnight.py` — Squashed Tanh-Gaussian PPO trainer with running observation normalization (`obs_rms`) and truncation bootstrapping.
- `rl/evaluate_policy.py` — 6D pose benchmark comparing trained RL against the 500 Hz nominal QP-OSC baseline.
- `rl/export_policy.py` — Exports checkpoint + `obs_rms` into standalone TorchScript (`policy_deploy.ts`).
- `rl/deploy_ros2_node.py` — 50 Hz ROS 2 deployment node for `/r100_0208` (`qp_osc`, `crisp`, or `jtc` modes).
- `rl/render_policy_animation.py` — High-quality 960×720 MP4 + GIF renderer (`validation`, `helix`, `circle`, `figure8`).
- `rl/checkpoints/stribeck_pose6d_qp/` — Pre-trained 1M-step checkpoint (`best_policy.pt`) and exported TorchScript policy (`policy_deploy.ts`).

## Quick Start

```bash
# 1. Offline deployment self-test (no ROS 2 required)
python3 rl/deploy_ros2_node.py --self-test --target-mode helix --residual-gain 1.0

# 2. Train a fresh 1M-step policy
python3 rl/train_ppo_overnight.py --exp-name stribeck_pose6d_qp --total-steps 1000000 --num-envs 12

# 3. Export standalone TorchScript policy (bakes in obs_rms)
python3 rl/export_policy.py --checkpoint rl/checkpoints/stribeck_pose6d_qp/best_policy.pt

# 4. Deploy on robot PC alongside C++ 500 Hz qp_osc_rl_controller (ROS 2 Jazzy)
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=8 python3 rl/deploy_ros2_node.py \
  --controller-mode qp_osc --target-mode helix --residual-gain 0.0 --speed-scale 0.3
```
