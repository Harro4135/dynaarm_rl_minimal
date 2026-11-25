#!/usr/bin/env python3
"""
========================================================================================
DUATIC DYNAARM 6D POSE RL POLICY EVALUATOR & NOMINAL BASELINE BENCHMARK
========================================================================================
Evaluates a trained 6D pose tracking policy against the Zero-Action Nominal 500 Hz OSC
Baseline on:
  1. Seeded Random 6D Dynamic Tracking Episodes (position mm + orientation deg + smoothness)
  2. Held-Out Real Hardware Trajectory (`datasets/calibration_real/validation_trajectory.csv`)
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch
import mujoco

from dynaarm_reach_env import DynaArmReachEnv, JOINT_NAMES
from train_ppo_overnight import ActorCritic, RunningMeanStd

RL_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(RL_DIR)
DEFAULT_CKPT = os.path.join(RL_DIR, "checkpoints", "best_policy.pt")
VAL_CSV = os.path.join(SIM_DIR, "datasets", "calibration_real", "validation_trajectory.csv")
PLOTS_DIR = os.path.join(SIM_DIR, "media", "plots")


def load_policy_checkpoint(checkpoint_path: str, obs_dim: int, act_dim: int, device: torch.device):
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent = ActorCritic(obs_dim=obs_dim, action_dim=act_dim).to(device)
    obs_rms = None
    config = {}

    if isinstance(raw, dict) and "model_state_dict" in raw:
        agent.load_state_dict(raw["model_state_dict"])
        config = raw.get("config", {})
        if "obs_rms" in raw and raw["obs_rms"] is not None:
            obs_rms = RunningMeanStd(shape=(obs_dim,))
            obs_rms.load_state_dict(raw["obs_rms"])
    else:
        agent.load_state_dict(raw)

    agent.eval()
    return agent, obs_rms, config


def build_validation_reference(env: DynaArmReachEnv, df: pd.DataFrame) -> dict:
    """Precompute exact 6D pose reference (p, v, a, R, w, alpha) from validation CSV."""
    q_cmd = np.column_stack([df[f"q_cmd_{j}"].values for j in JOINT_NAMES])
    dq_cmd = np.column_stack([df[f"dq_cmd_{j}"].values for j in JOINT_NAMES])

    m = env.nom_model
    d = mujoco.MjData(m)
    n_pts = len(df)
    p_ref = np.zeros((n_pts, 3), dtype=np.float64)
    R_ref = np.zeros((n_pts, 3, 3), dtype=np.float64)
    v_ref = np.zeros((n_pts, 3), dtype=np.float64)
    w_ref = np.zeros((n_pts, 3), dtype=np.float64)
    jac_pos = np.zeros((3, m.nv), dtype=np.float64)
    jac_rot = np.zeros((3, m.nv), dtype=np.float64)

    for k in range(n_pts):
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[env.qpos_idx] = q_cmd[k]
        d.qvel[env.qvel_idx] = dq_cmd[k]
        mujoco.mj_forward(m, d)
        p_ref[k] = d.site_xpos[env.ee_site_id]
        R_ref[k] = d.site_xmat[env.ee_site_id].reshape(3, 3)
        mujoco.mj_jacSite(m, d, jac_pos, jac_rot, env.ee_site_id)
        v_ref[k] = jac_pos[:, env.qvel_idx] @ dq_cmd[k]
        w_ref[k] = jac_rot[:, env.qvel_idx] @ dq_cmd[k]

    dt_sample = float(np.median(np.diff(df["timestamp"].values)))
    win = min(51, n_pts if n_pts % 2 == 1 else n_pts - 1)
    if win >= 5:
        a_ref = savgol_filter(v_ref, window_length=win, polyorder=3, deriv=1, delta=dt_sample, axis=0)
        alpha_ref = savgol_filter(w_ref, window_length=win, polyorder=3, deriv=1, delta=dt_sample, axis=0)
    else:
        a_ref = np.gradient(v_ref, dt_sample, axis=0)
        alpha_ref = np.gradient(w_ref, dt_sample, axis=0)

    stride = max(1, int(round(env.dt_ctrl / dt_sample)))
    idx_50hz = np.arange(0, n_pts, stride)
    return {
        "q_cmd": q_cmd,
        "dq_cmd": dq_cmd,
        "p_ref": p_ref,
        "v_ref": v_ref,
        "a_ref": a_ref,
        "R_ref": R_ref,
        "w_ref": w_ref,
        "alpha_ref": alpha_ref,
        "idx_50hz": idx_50hz,
    }


def _run_random_episodes(env: DynaArmReachEnv, num_episodes: int, policy_fn) -> dict:
    ep_returns, ep_mean_errs, ep_rms_errs, ep_peak_errs, ep_ori_errs, succ_15, succ_5 = [], [], [], [], [], [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=100 + ep)
        ep_reward = 0.0
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy_fn(obs))
            ep_reward += r
            done = term or trunc
        ep_returns.append(ep_reward)
        ep_mean_errs.append(info["ep_mean_dist_mm"])
        ep_rms_errs.append(info["ep_rms_dist_mm"])
        ep_peak_errs.append(info["ep_peak_dist_mm"])
        ep_ori_errs.append(info["ep_mean_ori_deg"])
        succ_15.append(info["ep_mean_dist_mm"] < 15.0 and info["ep_mean_ori_deg"] < 5.0)
        succ_5.append(info["ep_mean_dist_mm"] < 5.0 and info["ep_mean_ori_deg"] < 3.0)

    return {
        "mean_return": float(np.mean(ep_returns)),
        "mean_err_mm": float(np.mean(ep_mean_errs)),
        "rms_err_mm": float(np.mean(ep_rms_errs)),
        "peak_err_mm": float(np.max(ep_peak_errs)),
        "mean_ori_deg": float(np.mean(ep_ori_errs)),
        "succ_15_pct": float(np.mean(succ_15) * 100.0),
        "succ_5_pct": float(np.mean(succ_5) * 100.0),
    }


def _run_validation_trajectory(env: DynaArmReachEnv, ref: dict, policy_fn) -> dict:
    q_cmd, dq_cmd = ref["q_cmd"], ref["dq_cmd"]
    p_ref, v_ref, a_ref = ref["p_ref"], ref["v_ref"], ref["a_ref"]
    R_ref, w_ref, alpha_ref = ref["R_ref"], ref["w_ref"], ref["alpha_ref"]
    idx_50hz = ref["idx_50hz"]

    obs = env.reset_to_state(q_cmd[0], dq_cmd[0], p_ref[0], v_ref[0], a_ref[0], R_ref[0], w_ref[0], alpha_ref[0])
    pos_errs, ori_errs, taus, dqs = [], [], [], []

    for k in idx_50hz:
        ext = {
            "p": p_ref[k],
            "v": v_ref[k],
            "a": a_ref[k],
            "R": R_ref[k],
            "w": w_ref[k],
            "alpha": alpha_ref[k],
        }
        obs, _, _, _, info = env.step(policy_fn(obs), external_target=ext)
        pos_errs.append(info["dist_mm"])
        ori_errs.append(info["ori_err_deg"])
        taus.append(env.data.ctrl[env.act_idx].copy())
        dqs.append(env.data.qvel[env.qvel_idx].copy())

    taus = np.asarray(taus)
    dqs = np.asarray(dqs)
    dt = env.dt_ctrl
    dtau_rms = np.sqrt(np.mean(np.diff(taus, axis=0) ** 2, axis=0))
    jerk_rms = np.sqrt(np.mean((np.diff(dqs, n=2, axis=0) / (dt ** 2)) ** 2, axis=0))

    return {
        "pos_errs_mm": np.asarray(pos_errs),
        "ori_errs_deg": np.asarray(ori_errs),
        "dtau_rms_nm": dtau_rms,
        "jerk_rms": jerk_rms,
    }


def evaluate_policy(
    checkpoint_path: str = DEFAULT_CKPT,
    num_episodes: int = 15,
    label: str = "",
    use_stribeck_friction: bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    raw_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = raw_meta.get("config", {}) if isinstance(raw_meta, dict) else {}
    action_scale = float(ckpt_cfg.get("action_scale", 8.0))
    action_scale_ori = float(ckpt_cfg.get("action_scale_ori", 12.0))
    action_lpf_alpha = float(ckpt_cfg.get("action_lpf_alpha", 0.5))
    squashed_tanh = bool(ckpt_cfg.get("squashed_tanh", True))

    env = DynaArmReachEnv(
        domain_randomization=False,
        dynamic_targets=True,
        use_stribeck_friction=use_stribeck_friction,
        action_scale=action_scale,
        action_scale_ori=action_scale_ori,
        action_lpf_alpha=action_lpf_alpha,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent, obs_rms, _ = load_policy_checkpoint(checkpoint_path, obs_dim, act_dim, device)

    def rl_policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
            act = agent.get_deterministic_action(obs_t) if squashed_tanh else agent.actor_mean(obs_t)
            return act.cpu().numpy()[0]

    def zero_baseline_fn(obs: np.ndarray) -> np.ndarray:
        return np.zeros(act_dim, dtype=np.float32)

    print("=" * 86)
    print("     DUATIC DYNAARM 6D POSE RL EVALUATION (500 Hz OSC + ASYMMETRIC STRIBECK)")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Plant      : Stribeck={use_stribeck_friction} | No Load | EMA alpha={action_lpf_alpha:.2f} | Obs Norm={obs_rms is not None}")
    print("=" * 86)

    base_rand = _run_random_episodes(env, num_episodes, zero_baseline_fn)
    rl_rand = _run_random_episodes(env, num_episodes, rl_policy_fn)

    print(f"  [Random 6D Dynamic Tracking ({num_episodes} eps)]")
    print(
        f"    Nominal Baseline : Pos={base_rand['mean_err_mm']:5.2f} mm (RMS {base_rand['rms_err_mm']:5.2f}) | "
        f"Ori={base_rand['mean_ori_deg']:4.2f} deg | <15mm&5deg: {base_rand['succ_15_pct']:5.1f}%"
    )
    print(
        f"    Trained RL Policy: Pos={rl_rand['mean_err_mm']:5.2f} mm (RMS {rl_rand['rms_err_mm']:5.2f}) | "
        f"Ori={rl_rand['mean_ori_deg']:4.2f} deg | <15mm&5deg: {rl_rand['succ_15_pct']:5.1f}%"
    )

    val_mean_err = val_peak_err = base_val_mean = val_ori_mean = 0.0
    if os.path.exists(VAL_CSV):
        df = pd.read_csv(VAL_CSV)
        ref = build_validation_reference(env, df)
        base_val = _run_validation_trajectory(env, ref, zero_baseline_fn)
        rl_val = _run_validation_trajectory(env, ref, rl_policy_fn)

        base_val_mean = float(np.mean(base_val["pos_errs_mm"]))
        base_val_peak = float(np.max(base_val["pos_errs_mm"]))
        base_ori_mean = float(np.mean(base_val["ori_errs_deg"]))
        val_mean_err = float(np.mean(rl_val["pos_errs_mm"]))
        val_peak_err = float(np.max(rl_val["pos_errs_mm"]))
        val_ori_mean = float(np.mean(rl_val["ori_errs_deg"]))

        print("\n  [Held-Out Real Hardware 6D Trajectory: validation_trajectory.csv]")
        print(
            f"    Nominal Baseline : Pos={base_val_mean:5.2f} mm (Peak {base_val_peak:5.2f}) | "
            f"Ori={base_ori_mean:4.2f} deg | Max dTau={np.max(base_val['dtau_rms_nm']):4.2f} Nm | Max Jerk={np.max(base_val['jerk_rms']):5.1f}"
        )
        print(
            f"    Trained RL Policy: Pos={val_mean_err:5.2f} mm (Peak {val_peak_err:5.2f}) | "
            f"Ori={val_ori_mean:4.2f} deg | Max dTau={np.max(rl_val['dtau_rms_nm']):4.2f} Nm | Max Jerk={np.max(rl_val['jerk_rms']):5.1f}"
        )

        os.makedirs(PLOTS_DIR, exist_ok=True)
        tag = f"_{label}" if label else ""
        out_plot = os.path.join(PLOTS_DIR, f"rl_policy_evaluation{tag}.png")
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=150, sharex=True)
        axes[0].plot(base_val["pos_errs_mm"], label=f"Nominal 500Hz OSC (Mean: {base_val_mean:.2f} mm)", color="#7f7f7f", linestyle="--")
        axes[0].plot(rl_val["pos_errs_mm"], label=f"Trained RL ({label or 'Policy'}) (Mean: {val_mean_err:.2f} mm)", color="#2ca02c")
        axes[0].set_ylabel("3D Position Error [mm]", fontweight="bold")
        axes[0].set_title(f"No-Load 6D Pose Tracking (500Hz OSC + Asymmetric Stribeck): {label or 'Validation'}", fontweight="bold")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(base_val["ori_errs_deg"], label=f"Nominal 500Hz OSC (Mean: {base_ori_mean:.2f} deg)", color="#7f7f7f", linestyle="--")
        axes[1].plot(rl_val["ori_errs_deg"], label=f"Trained RL ({label or 'Policy'}) (Mean: {val_ori_mean:.2f} deg)", color="#1f77b4")
        axes[1].set_ylabel("Orientation Error [deg]", fontweight="bold")
        axes[1].set_xlabel("Trajectory Step (50 Hz)", fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_plot)
        plt.close()
        print(f"\n[✓] Saved evaluation plot: {out_plot}")

    print("=" * 86)
    return {
        "mean_dist_mm": rl_rand["mean_err_mm"],
        "rms_dist_mm": rl_rand["rms_err_mm"],
        "max_dist_mm": rl_rand["peak_err_mm"],
        "mean_ori_deg": rl_rand["mean_ori_deg"],
        "succ_rate": rl_rand["succ_15_pct"],
        "baseline_mean_dist_mm": base_rand["mean_err_mm"],
        "val_mean_err_mm": val_mean_err,
        "val_peak_err_mm": val_peak_err,
        "val_ori_mean_deg": val_ori_mean,
        "baseline_val_mean_err_mm": base_val_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Duatic DynaArm 6D Pose RL Policy")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--no-stribeck", action="store_true")
    args = parser.parse_args()

    evaluate_policy(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        label=args.label,
        use_stribeck_friction=not args.no_stribeck,
    )


if __name__ == "__main__":
    main()
