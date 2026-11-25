#!/usr/bin/env python3
"""
========================================================================================
DUATIC DYNAARM 6D POSE RL POLICY HIGH-QUALITY ANIMATION RENDERER
========================================================================================
Renders crisp 960x720 MP4 + high-quality palette-optimized GIF animations of a trained
6D pose tracking policy on the Asymmetric Stribeck plant across multiple trajectories:
  - validation : Held-out real hardware trajectory (validation_trajectory.csv)
  - figure8    : Smooth 3D figure-8 with coupled orientation sweep
  - circle     : Tilted 3D circular orbit with smooth wrist orientation tracking
  - helix      : 3D vertical spiral ascent/descent with orientation modulation
  - random     : Seeded random 6D Lissajous episode from DynaArmReachEnv

Overlays:
  - Live 3D target sphere (green) + target orientation RGB triad
  - Actual TCP marker (orange) + reference trail (cyan) & actual TCP trail (orange)
  - Live telemetry HUD (time, position error mm, orientation error deg, mean/peak)
"""

import argparse
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import mujoco

RL_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(RL_DIR)
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

from dynaarm_reach_env import DynaArmReachEnv, HOME_POSITION, rotvec_to_mat
from evaluate_policy import load_policy_checkpoint, build_validation_reference, VAL_CSV

DEFAULT_CKPT = os.path.join(RL_DIR, "checkpoints", "stribeck_pose6d_500hz", "best_policy.pt")
GIFS_DIR = os.path.join(SIM_DIR, "media", "gifs")
VIDEOS_DIR = os.path.join(SIM_DIR, "media", "videos")


def add_sphere_geom(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_capsule_segment(scene, p1, p2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    if np.linalg.norm(p2 - p1) < 1e-5:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p1, p2)
    scene.ngeom += 1


def add_frame_triad(scene, pos, R, length=0.07, radius=0.0035, alpha=0.9):
    """Draw an RGB coordinate triad showing 3D orientation of a pose (pos, R)."""
    colors = [
        [0.95, 0.20, 0.20, alpha],  # X - red
        [0.20, 0.95, 0.30, alpha],  # Y - green
        [0.25, 0.55, 1.00, alpha],  # Z - blue
    ]
    for i in range(3):
        add_capsule_segment(scene, pos, pos + length * R[:, i], radius, colors[i])


def overlay_hud(frame, title, subtitle, t_sec, err_mm, mean_err_mm, ori_deg, mean_ori_deg, a_res_norm):
    h, w, _ = frame.shape
    scale = w / 800.0
    top_h = int(56 * scale)
    bot_h = int(42 * scale)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), (18, 22, 28), -1)
    cv2.rectangle(overlay, (0, h - bot_h), (w, h), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    cv2.putText(
        frame, title, (int(16 * scale), int(23 * scale)),
        cv2.FONT_HERSHEY_DUPLEX, 0.55 * scale, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, subtitle, (int(16 * scale), int(46 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46 * scale, (80, 235, 190), 1, cv2.LINE_AA,
    )

    lx = int(w - 250 * scale)
    cv2.circle(frame, (lx, int(20 * scale)), int(5 * scale), (50, 235, 90), -1)
    cv2.putText(frame, "Target", (lx + int(12 * scale), int(24 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, (230, 240, 245), 1, cv2.LINE_AA)
    cv2.circle(frame, (lx + int(95 * scale), int(20 * scale)), int(5 * scale), (255, 130, 40), -1)
    cv2.putText(frame, "Actual TCP", (lx + int(107 * scale), int(24 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, (230, 240, 245), 1, cv2.LINE_AA)

    status = (
        f"t={t_sec:4.1f}s | Pos={err_mm:4.1f}mm (Mean {mean_err_mm:4.1f}) | "
        f"Ori={ori_deg:4.1f}deg (Mean {mean_ori_deg:4.1f}) | |a_res|={a_res_norm:3.1f}m/s^2"
    )
    cv2.putText(
        frame, status, (int(16 * scale), h - int(14 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (220, 235, 248), 1, cv2.LINE_AA,
    )
    return frame


def build_analytic_trajectory(env: DynaArmReachEnv, mode: str, duration_s: float) -> tuple[np.ndarray, list[dict], str]:
    """Generate smooth C2-continuous 6D trajectories centered well inside the hardware-feasible workspace."""
    # Mid-workspace pose with healthy elbow & wrist bend (well away from J5=0 wrist singularity)
    q0 = np.array([0.0, 0.25, 1.05, 0.0, 0.55, 0.0], dtype=np.float32)
    dq0 = np.zeros(6, dtype=np.float32)
    env.reset_to_state(q0, dq0)
    p0 = env.data.site_xpos[env.ee_site_id].copy().astype(np.float64)
    R0 = env.data.site_xmat[env.ee_site_id].reshape(3, 3).copy()

    n_steps = int(round(duration_s / env.dt_ctrl))
    dt = env.dt_ctrl
    targets = []

    for k in range(n_steps):
        t = (k + 1) * dt
        # Smooth ramp-in over first 1.2 s so initial velocity & acceleration start cleanly at zero
        ramp = min(1.0, t / 1.2)
        s = 3 * ramp**2 - 2 * ramp**3

        if mode == "figure8":
            # 100% IK-feasible 3D figure-8 in Y-Z with gentle inward X arc (well inside SAFE_LIMITS)
            w = 1.3
            ry, rz = 0.13, 0.065
            rx = -0.025  # arcs slightly inward (-x) away from elbow extension limit
            p = p0 + s * np.array([rx * (1.0 - np.cos(w * t)), ry * np.sin(w * t), rz * np.sin(2 * w * t)])
            v = s * np.array([rx * w * np.sin(w * t), ry * w * np.cos(w * t), 2 * rz * w * np.cos(2 * w * t)])
            a = s * np.array([rx * w**2 * np.cos(w * t), -ry * w**2 * np.sin(w * t), -4 * rz * w**2 * np.sin(2 * w * t)])
            axis = np.array([0.0, 0.0, 1.0])  # smooth yaw sweep staying well inside wrist limits
            amp, wo = 0.15, 1.3
            ang = s * amp * np.sin(wo * t)
            dang = s * amp * wo * np.cos(wo * t)
            ddang = -s * amp * (wo**2) * np.sin(wo * t)
            R = rotvec_to_mat(ang * axis) @ R0
            w_vec, alpha_vec = dang * axis, ddang * axis
            sub = "IK-Feasible 3D Figure-8 + Yaw Sweep | 500Hz QP-OSC + Asymmetric Stribeck"

        elif mode == "circle":
            w = 1.4
            r = 0.09
            # Tilted circle in Y-Z with gentle X tilt, centered cleanly inside workspace
            p = p0 + s * np.array([0.03 * np.sin(w * t), r * np.sin(w * t), r * (np.cos(w * t) - 1.0)])
            v = s * np.array([0.03 * w * np.cos(w * t), r * w * np.cos(w * t), -r * w * np.sin(w * t)])
            a = s * np.array([-0.03 * w**2 * np.sin(w * t), -r * w**2 * np.sin(w * t), -r * w**2 * np.cos(w * t)])
            axis = np.array([1.0, 0.0, 0.0])
            amp, wo = 0.18, 1.4
            ang = s * amp * np.sin(wo * t)
            dang = s * amp * wo * np.cos(wo * t)
            ddang = -s * amp * (wo**2) * np.sin(wo * t)
            R = rotvec_to_mat(ang * axis) @ R0
            w_vec, alpha_vec = dang * axis, ddang * axis
            sub = "Tilted 3D Circular Orbit + Roll Modulation | 500Hz OSC + Asymmetric Stribeck"

        elif mode == "helix":
            w = 1.6
            r = 0.075
            wz = 0.65
            p = p0 + s * np.array([r * (np.cos(w * t) - 1.0), r * np.sin(w * t), 0.08 * np.sin(wz * t)])
            v = s * np.array([-r * w * np.sin(w * t), r * w * np.cos(w * t), 0.08 * wz * np.cos(wz * t)])
            a = s * np.array([-r * w**2 * np.cos(w * t), -r * w**2 * np.sin(w * t), -0.08 * wz**2 * np.sin(wz * t)])
            axis = np.array([0.0, 1.0, 0.0])
            amp, wo = 0.16, 1.2
            ang = s * amp * np.sin(wo * t)
            dang = s * amp * wo * np.cos(wo * t)
            ddang = -s * amp * (wo**2) * np.sin(wo * t)
            R = rotvec_to_mat(ang * axis) @ R0
            w_vec, alpha_vec = dang * axis, ddang * axis
            sub = "3D Vertical Helix Spiral + Pitch Sweep | 500Hz OSC + Asymmetric Stribeck"
        else:
            raise ValueError(f"Unknown analytic mode: {mode}")

        targets.append({"p": p, "v": v, "a": a, "R": R, "w": w_vec, "alpha": alpha_vec})

    obs = env.reset_to_state(q0, dq0, p0, np.zeros(3), np.zeros(3), R0, np.zeros(3), np.zeros(3))
    return obs, targets, sub


def save_high_quality_gif(mp4_path: str, gif_path: str, frames: list[np.ndarray], fps: int, width: int, height: int):
    """Use 2-pass ffmpeg palettegen/paletteuse for crisp 960x720 GIFs (with PIL fallback)."""
    if shutil.which("ffmpeg"):
        palette_path = gif_path + ".palette.png"
        vf_pal = f"fps={fps},scale={width}:{height}:flags=lanczos,palettegen=max_colors=256:stats_mode=diff"
        vf_use = f"fps={fps},scale={width}:{height}:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path, "-vf", vf_pal, palette_path],
                check=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path, "-i", palette_path, "-lavfi", vf_use, gif_path],
                check=True,
            )
            if os.path.exists(palette_path):
                os.remove(palette_path)
            return
        except Exception as exc:
            print(f"[!] ffmpeg GIF encoding failed ({exc}), falling back to PIL.")

    gif_frames = [
        Image.fromarray(f).quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG)
        for f in frames
    ]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
    )


def render_animation(
    checkpoint_path: str = DEFAULT_CKPT,
    mode: str = "validation",
    output_name: str = "rl_policy_stribeck_pose6d_500hz",
    width: int = 960,
    height: int = 720,
    fps: int = 25,
    max_duration_s: float = 10.0,
    seed: int = 104,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = raw_meta.get("config", {}) if isinstance(raw_meta, dict) else {}
    action_scale = float(ckpt_cfg.get("action_scale", 8.0))
    action_scale_ori = float(ckpt_cfg.get("action_scale_ori", 12.0))
    action_lpf_alpha = float(ckpt_cfg.get("action_lpf_alpha", 0.5))
    squashed_tanh = bool(ckpt_cfg.get("squashed_tanh", True))

    env = DynaArmReachEnv(
        domain_randomization=False,
        dynamic_targets=True,
        use_stribeck_friction=True,
        action_scale=action_scale,
        action_scale_ori=action_scale_ori,
        action_lpf_alpha=action_lpf_alpha,
        max_episode_steps=int(max_duration_s * 50) + 10,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent, obs_rms, _ = load_policy_checkpoint(checkpoint_path, obs_dim, act_dim, device)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
            act = agent.get_deterministic_action(obs_t) if squashed_tanh else agent.actor_mean(obs_t)
            return act.cpu().numpy()[0]

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.46, 0.0, 1.18]
    cam.distance = 2.05
    cam.elevation = -15.0
    cam.azimuth = 132.0

    stride_render = max(1, int(round(50.0 / fps)))
    frames: list[np.ndarray] = []
    ee_trail: list[np.ndarray] = []
    tgt_trail: list[np.ndarray] = []
    errs_mm: list[float] = []
    oris_deg: list[float] = []

    if mode == "validation":
        df = pd.read_csv(VAL_CSV)
        ref = build_validation_reference(env, df)
        idx_50hz = ref["idx_50hz"][: int(max_duration_s * 50)]
        obs = env.reset_to_state(
            ref["q_cmd"][0], ref["dq_cmd"][0],
            ref["p_ref"][0], ref["v_ref"][0], ref["a_ref"][0],
            ref["R_ref"][0], ref["w_ref"][0], ref["alpha_ref"][0],
        )
        subtitle = "Held-Out 6D Hardware Trajectory (validation_trajectory.csv) | 500Hz OSC + Stribeck"
        step_iter = [
            (
                k_idx,
                {
                    "p": ref["p_ref"][k],
                    "v": ref["v_ref"][k],
                    "a": ref["a_ref"][k],
                    "R": ref["R_ref"][k],
                    "w": ref["w_ref"][k],
                    "alpha": ref["alpha_ref"][k],
                },
            )
            for k_idx, k in enumerate(idx_50hz)
        ]
    elif mode in ("figure8", "circle", "helix"):
        obs, targets, subtitle = build_analytic_trajectory(env, mode, max_duration_s)
        step_iter = list(enumerate(targets))
    else:
        obs, _ = env.reset(seed=seed)
        max_steps = min(200, int(max_duration_s * 50))
        subtitle = f"Random 6D Lissajous Trajectory (seed={seed}) | 500Hz OSC + Stribeck"
        step_iter = [(k_idx, None) for k_idx in range(max_steps)]

    title = f"Duatic DynaArm 6D Pose RL Policy ({mode})"
    trail_max = 100

    print(f"[*] [{mode}] Rendering {len(step_iter)} steps ({len(step_iter) // stride_render} frames @ {width}x{height}, {fps} fps)...")
    for step_idx, ext_target in step_iter:
        action = policy_fn(obs)
        obs, _, _, _, info = env.step(action, external_target=ext_target)

        p_ee = env.data.site_xpos[env.ee_site_id].copy()
        R_ee = env.data.site_xmat[env.ee_site_id].reshape(3, 3).copy()
        p_tgt = env.p_target.copy()
        R_tgt = env.R_target.copy()
        errs_mm.append(info["dist_mm"])
        oris_deg.append(info["ori_err_deg"])

        if step_idx % stride_render == 0:
            ee_trail.append(p_ee)
            tgt_trail.append(p_tgt)
            if len(ee_trail) > trail_max:
                ee_trail.pop(0)
                tgt_trail.pop(0)

            renderer.update_scene(env.data, camera=cam)

            for i in range(1, len(tgt_trail)):
                alpha = 0.25 + 0.65 * (i / len(tgt_trail))
                add_capsule_segment(renderer.scene, tgt_trail[i - 1], tgt_trail[i], 0.004, [0.1, 0.9, 0.6, alpha])
            for i in range(1, len(ee_trail)):
                alpha = 0.30 + 0.70 * (i / len(ee_trail))
                add_capsule_segment(renderer.scene, ee_trail[i - 1], ee_trail[i], 0.0045, [1.0, 0.45, 0.1, alpha])

            # Target sphere + 6D orientation triad
            add_sphere_geom(renderer.scene, p_tgt, 0.015, [0.15, 0.95, 0.30, 0.85])
            add_frame_triad(renderer.scene, p_tgt, R_tgt, length=0.075, radius=0.0032, alpha=0.85)
            add_sphere_geom(renderer.scene, p_ee, 0.010, [1.0, 0.45, 0.10, 0.95])

            frame = renderer.render().copy()
            t_sec = (step_idx + 1) * env.dt_ctrl
            a_res_norm = float(np.linalg.norm(action[:3] * action_scale))
            frame = overlay_hud(
                frame,
                title=title,
                subtitle=subtitle,
                t_sec=t_sec,
                err_mm=info["dist_mm"],
                mean_err_mm=float(np.mean(errs_mm)),
                ori_deg=info["ori_err_deg"],
                mean_ori_deg=float(np.mean(oris_deg)),
                a_res_norm=a_res_norm,
            )
            frames.append(frame)

    renderer.close()

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(GIFS_DIR, exist_ok=True)

    mp4_path = os.path.join(VIDEOS_DIR, f"{output_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, float(fps), (width, height))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    gif_path = os.path.join(GIFS_DIR, f"{output_name}.gif")
    save_high_quality_gif(mp4_path, gif_path, frames, fps=fps, width=width, height=height)

    print(
        f"[✓] [{mode}] Pos Mean: {np.mean(errs_mm):.2f} mm (Peak {np.max(errs_mm):.2f}) | "
        f"Ori Mean: {np.mean(oris_deg):.2f} deg (Peak {np.max(oris_deg):.2f})"
    )
    print(f"    MP4: {mp4_path}")
    print(f"    GIF: {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Render high-quality 3D MuJoCo animations of 6D pose RL policy")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["validation", "figure8", "circle", "helix"],
        choices=["validation", "figure8", "circle", "helix", "random"],
    )
    parser.add_argument("--prefix", type=str, default="rl_pose6d")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=104)
    args = parser.parse_args()

    for mode in args.modes:
        dur = 16.0 if mode == "validation" else args.max_duration
        render_animation(
            checkpoint_path=args.checkpoint,
            mode=mode,
            output_name=f"{args.prefix}_{mode}",
            width=args.width,
            height=args.height,
            fps=args.fps,
            max_duration_s=dur,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
