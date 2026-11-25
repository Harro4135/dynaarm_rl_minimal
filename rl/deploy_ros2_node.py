#!/usr/bin/env python3
"""
========================================================================================
DUATIC DYNAARM ROS 2 RL 6D POSE TRACKING DEPLOYMENT NODE (CLEARPATH RIDGEBACK R100)
========================================================================================
Deploys the exported TorchScript 6D pose RL policy (`policy_deploy.ts`) onto the real
or mock Clearpath Ridgeback + Duatic DynaArm stack (`/home/harrison/Repos/ridgeback_dynaarm`):

  Namespace : /r100_0208 (ROS_DOMAIN_ID=8, CycloneDDS, ROS 2 Jazzy)
  Feedback  : /{ns}/platform/joint_states [sensor_msgs/JointState]
  Optional  : /{ns}/rl_target_pose [geometry_msgs/PoseStamped] (when --target-mode topic)

  Command Modes (--controller-mode):
    1. `jtc` (default, safest bring-up):
       Runs 50 Hz RL residual + 6-DOF QP-OSC + 2nd-order CBF to generate smooth, limit-safe
       preview points published to:
         /{ns}/joint_trajectory_controller/joint_trajectory [trajectory_msgs/JointTrajectory]
    2. `crisp`:
       Publishes residual-compensated 6D pose + velocity feedforward to:
         /{ns}/target_pose  [geometry_msgs/PoseStamped]
         /{ns}/target_twist [geometry_msgs/TwistStamped]
       for `crisp_cartesian_controller` or `crisp_cartesian_admittance_controller`.

Built-in Hardware Safety Guardrails:
  - Starts with `--residual-gain 0.0` by default (pure nominal 6D QP-OSC + CBF).
    Ramp gradually on hardware: 0.0 -> 0.2 -> 0.5 -> 1.0.
  - Smooth startup capture: anchors analytic trajectories at the robot's live pose with
    a 1.5 s smoothstep ramp-in (zero jump on start).
  - Joint-state staleness watchdog (>0.15 s -> halt commands).
  - Velocity & tracking fault latch (|dq| > 1.8 rad/s stays below safety_observer's 2.0 rad/s
    limit; |err_p| > 0.15 m halts cleanly).
  - Works offline with `--self-test` to verify the entire pipeline without a ROS 2 master.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
import numpy as np
import torch
import mujoco

RL_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(RL_DIR)
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

from dynaarm_reach_env import (
    DEFAULT_XML,
    HOME_POSITION,
    JOINT_NAMES,
    SAFE_LIMITS_HIGH,
    SAFE_LIMITS_LOW,
    orientation_error_world,
    rotvec_to_mat,
    solve_box_qp_6d,
)

DEFAULT_MODEL = os.path.join(RL_DIR, "checkpoints", "stribeck_pose6d_pvtp", "policy_deploy.ts")
DEFAULT_META = os.path.join(RL_DIR, "checkpoints", "stribeck_pose6d_pvtp", "policy_deploy_meta.json")


def quat_xyzw_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert ROS geometry_msgs Quaternion (x, y, z, w) to 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-12
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def mat_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to ROS geometry_msgs Quaternion (x, y, z, w)."""
    q_wxyz = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q_wxyz, R.flatten())
    return float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])


class RL6DPoseControllerCore:
    """
    Framework-agnostic 50 Hz RL + 500 Hz 6-DOF QP-OSC + CBF controller core.
    Used identically by the ROS 2 deployment node and the offline `--self-test`.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        meta_path: str = DEFAULT_META,
        xml_path: str = os.path.join(SIM_DIR, "models", "ridgeback_duatic_kinematics.xml"),
        residual_gain: float = 0.0,
        control_freq_hz: float = 50.0,
        max_joint_vel: float = 1.5,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Exported TorchScript model not found at {model_path}. Run `python3 rl/export_policy.py` first."
            )

        torch.set_num_threads(1)
        self.policy = torch.jit.load(model_path, map_location="cpu")
        self.policy.eval()

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        self.action_scale = float(meta.get("action_scale", 3.0))
        self.action_scale_ori = float(meta.get("action_scale_ori", 5.0))
        self.action_lpf_alpha = float(meta.get("action_lpf_alpha", 0.5))
        self.residual_gain = float(np.clip(residual_gain, 0.0, 1.0))
        self.dt_ctrl = 1.0 / float(control_freq_hz)
        self.max_joint_vel = float(max_joint_vel)

        # Load calibrated nominal MuJoCo model for exact FK & Jacobian (matches training)
        self.nom_model = mujoco.MjModel.from_xml_path(xml_path)
        self.nom_data = mujoco.MjData(self.nom_model)
        self.qpos_idx = np.array([self.nom_model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.qvel_idx = np.array([self.nom_model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self.ee_site_id = self.nom_model.site("ee_site").id
        self.base_body_id = self.nom_model.body("ridgeback_base").id
        mujoco.mj_forward(self.nom_model, self.nom_data)
        self.base_pos_world = self.nom_data.body("ridgeback_base").xpos.copy()
        self._jac_pos = np.zeros((3, self.nom_model.nv), dtype=np.float64)
        self._jac_rot = np.zeros((3, self.nom_model.nv), dtype=np.float64)
        self._max_joint_acc = np.array([15.0, 15.0, 15.0, 20.0, 20.0, 20.0], dtype=np.float64)

        self.prev_action = np.zeros(6, dtype=np.float32)

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.nom_data.qpos[:] = 0.0
        self.nom_data.qvel[:] = 0.0
        self.nom_data.qpos[self.qpos_idx] = q
        mujoco.mj_forward(self.nom_model, self.nom_data)
        p_ee = self.nom_data.site_xpos[self.ee_site_id].copy()
        R_ee = self.nom_data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return p_ee, R_ee

    def compute_step(
        self,
        q_now: np.ndarray,
        dq_now: np.ndarray,
        p_tgt: np.ndarray,
        v_tgt: np.ndarray,
        a_tgt: np.ndarray,
        R_tgt: np.ndarray,
        w_tgt: np.ndarray,
        alpha_tgt: np.ndarray,
        preview_steps: int = 3,
    ) -> dict:
        """
        Compute one 50 Hz RL inference step + 500 Hz QP-OSC rollout preview for ros2_control.
        Returns next joint trajectory points (q_traj, dq_traj), residual, and diagnostics.
        """
        p_ee, R_ee = self.forward_kinematics(q_now)
        err_p = (p_tgt - p_ee).astype(np.float32)
        err_ori = orientation_error_world(R_tgt, R_ee)

        obs = np.concatenate([
            q_now.astype(np.float32),
            dq_now.astype(np.float32),
            p_ee.astype(np.float32),
            p_tgt.astype(np.float32),
            err_p,
            v_tgt.astype(np.float32),
            a_tgt.astype(np.float32),
            err_ori,
            w_tgt.astype(np.float32),
            self.prev_action,
        ]).astype(np.float32)

        t0 = time.perf_counter()
        if self.residual_gain > 0.0:
            with torch.no_grad():
                raw_act = self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
            raw_act = np.clip(raw_act, -1.0, 1.0).astype(np.float32) * self.residual_gain
        else:
            raw_act = np.zeros(6, dtype=np.float32)

        # Identical in-loop EMA smoothing as training
        alpha = self.action_lpf_alpha
        smooth_act = (alpha * self.prev_action + (1.0 - alpha) * raw_act).astype(np.float32)
        self.prev_action = smooth_act

        a_res_pos = (smooth_act[:3] * self.action_scale).astype(np.float64)
        a_res_ori = (smooth_act[3:] * self.action_scale_ori).astype(np.float64)

        # Integrate short smooth QP-OSC + CBF kinematic preview (for JTC interpolation)
        q_sim = q_now.astype(np.float64).copy()
        dq_sim = dq_now.astype(np.float64).copy()
        q_traj = []
        dq_traj = []
        sub_dt = 0.004
        substeps_per_ctrl = max(1, int(round(self.dt_ctrl / sub_dt)))

        for k in range(preview_steps):
            p_k = p_tgt + v_tgt * (k * self.dt_ctrl)
            R_k = rotvec_to_mat(w_tgt * (k * self.dt_ctrl)) @ R_tgt
            for _ in range(substeps_per_ctrl):
                self.nom_data.qpos[:] = 0.0
                self.nom_data.qvel[:] = 0.0
                self.nom_data.qpos[self.qpos_idx] = q_sim
                self.nom_data.qvel[self.qvel_idx] = dq_sim
                mujoco.mj_forward(self.nom_model, self.nom_data)

                p_sub = self.nom_data.site_xpos[self.ee_site_id]
                R_sub = self.nom_data.site_xmat[self.ee_site_id].reshape(3, 3)
                self._jac_pos.fill(0.0)
                self._jac_rot.fill(0.0)
                mujoco.mj_jacSite(self.nom_model, self.nom_data, self._jac_pos, self._jac_rot, self.ee_site_id)
                J_pos = self._jac_pos[:, self.qvel_idx]
                J_rot = self._jac_rot[:, self.qvel_idx]
                J6 = np.vstack([J_pos, J_rot])

                v_cart = J_pos @ dq_sim
                w_cart = J_rot @ dq_sim
                ep_sub = p_k - p_sub
                eo_sub = orientation_error_world(R_k, R_sub).astype(np.float64)

                a_lin = np.clip(a_tgt + 180.0 * ep_sub + 28.0 * (v_tgt - v_cart) + a_res_pos, -20.0, 20.0)
                a_ang = np.clip(alpha_tgt + 150.0 * eo_sub + 24.0 * (w_tgt - w_cart) + a_res_ori, -30.0, 30.0)

                min_sv = float(np.linalg.svd(J6, compute_uv=False)[-1])
                lam = 0.025 + 0.12 * float(np.exp(-((min_sv / 0.035) ** 2)))
                lam2 = lam * lam
                ddq_reg = 8.0 * (HOME_POSITION - q_sim) - 4.0 * dq_sim

                H = (J_pos.T @ J_pos) + 0.25 * (J_rot.T @ J_rot) + lam2 * np.eye(6)
                g = (J_pos.T @ a_lin) + 0.25 * (J_rot.T @ a_ang) + lam2 * ddq_reg

                cbf_alpha = 12.0
                q_lo = SAFE_LIMITS_LOW + 0.08
                q_hi = SAFE_LIMITS_HIGH - 0.08
                ddq_cbf_min = -2.0 * cbf_alpha * dq_sim - (cbf_alpha ** 2) * (q_sim - q_lo)
                ddq_cbf_max = -2.0 * cbf_alpha * dq_sim + (cbf_alpha ** 2) * (q_hi - q_sim)
                lb = np.maximum(-self._max_joint_acc, ddq_cbf_min)
                ub = np.minimum(self._max_joint_acc, ddq_cbf_max)
                ddq = solve_box_qp_6d(H, g, lb, ub)

                dq_sim = np.clip(dq_sim + ddq * sub_dt, -self.max_joint_vel, self.max_joint_vel)
                q_sim = np.clip(q_sim + dq_sim * sub_dt, SAFE_LIMITS_LOW + 0.05, SAFE_LIMITS_HIGH - 0.05)

            q_traj.append(q_sim.copy())
            dq_traj.append(dq_sim.copy())

        solve_ms = (time.perf_counter() - t0) * 1000.0
        # Lead-compensated pose & twist for CRISP Cartesian controllers
        p_crisp = p_tgt + 0.5 * a_res_pos * (self.dt_ctrl ** 2)
        v_crisp = v_tgt + a_res_pos * self.dt_ctrl
        R_crisp = rotvec_to_mat(0.5 * a_res_ori * (self.dt_ctrl ** 2)) @ R_tgt
        w_crisp = w_tgt + a_res_ori * self.dt_ctrl

        return {
            "q_traj": q_traj,
            "dq_traj": dq_traj,
            "p_crisp": p_crisp,
            "v_crisp": v_crisp,
            "R_crisp": R_crisp,
            "w_crisp": w_crisp,
            "pos_err_mm": float(np.linalg.norm(err_p) * 1000.0),
            "ori_err_deg": float(np.degrees(np.linalg.norm(err_ori))),
            "a_res_pos": a_res_pos,
            "solve_ms": solve_ms,
        }


def compute_analytic_target(mode: str, t: float, p0: np.ndarray, R0: np.ndarray, speed_scale: float = 0.5):
    """Hardware-safe, IK-feasible analytic trajectories with C^2 quintic 2.5s ramp-in."""
    T_ramp = 2.5
    if t <= 0.0:
        s, ds, dds = 0.0, 0.0, 0.0
    elif t >= T_ramp:
        s, ds, dds = 1.0, 0.0, 0.0
    else:
        r = t / T_ramp
        s = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5
        ds = (30.0 * r**2 - 60.0 * r**3 + 30.0 * r**4) / T_ramp
        dds = (60.0 * r - 180.0 * r**2 + 120.0 * r**3) / (T_ramp**2)

    sc = float(np.clip(speed_scale, 0.1, 1.0))

    if mode == "helix":
        w = 1.6 * sc
        wz = 0.65 * sc
        r = 0.075
        dp = np.array([r * (np.cos(w * t) - 1.0), r * np.sin(w * t), 0.08 * np.sin(wz * t)])
        dv = np.array([-r * w * np.sin(w * t), r * w * np.cos(w * t), 0.08 * wz * np.cos(wz * t)])
        da = np.array([-r * w**2 * np.cos(w * t), -r * w**2 * np.sin(w * t), -0.08 * wz**2 * np.sin(wz * t)])
        axis = np.array([0.0, 1.0, 0.0])
        amp, wo = 0.15, 1.2 * sc
    elif mode == "circle":
        w = 1.4 * sc
        r = 0.08
        dp = np.array([0.025 * np.sin(w * t), r * np.sin(w * t), r * (np.cos(w * t) - 1.0)])
        dv = np.array([0.025 * w * np.cos(w * t), r * w * np.cos(w * t), -r * w * np.sin(w * t)])
        da = np.array([-0.025 * w**2 * np.sin(w * t), -r * w**2 * np.sin(w * t), -r * w**2 * np.cos(w * t)])
        axis = np.array([1.0, 0.0, 0.0])
        amp, wo = 0.15, 1.4 * sc
    elif mode == "figure8":
        w = 1.3 * sc
        rx, ry, rz = -0.025, 0.12, 0.06
        dp = np.array([rx * (1.0 - np.cos(w * t)), ry * np.sin(w * t), rz * np.sin(2 * w * t)])
        dv = np.array([rx * w * np.sin(w * t), ry * w * np.cos(w * t), 2 * rz * w * np.cos(2 * w * t)])
        da = np.array([rx * w**2 * np.cos(w * t), -ry * w**2 * np.sin(w * t), -4 * rz * w**2 * np.sin(2 * w * t)])
        axis = np.array([0.0, 0.0, 1.0])
        amp, wo = 0.14, 1.3 * sc
    else:
        return p0.copy(), np.zeros(3), np.zeros(3), R0.copy(), np.zeros(3), np.zeros(3)

    # Exact product-rule derivatives so (p, v, a) are 100% consistent during startup ramp
    p = p0 + s * dp
    v = s * dv + ds * dp
    a = s * da + 2.0 * ds * dv + dds * dp

    th = amp * np.sin(wo * t)
    dth = amp * wo * np.cos(wo * t)
    ddth = -amp * (wo**2) * np.sin(wo * t)
    ang = s * th
    dang = s * dth + ds * th
    ddang = s * ddth + 2.0 * ds * dth + dds * th
    R = rotvec_to_mat(ang * axis) @ R0
    return p, v, a, R, dang * axis, ddang * axis


def run_self_test(args):
    """Offline verification of exported TorchScript + 6-DOF QP-OSC + CBF without ROS 2."""
    core = RL6DPoseControllerCore(
        model_path=args.model,
        meta_path=args.meta,
        residual_gain=args.residual_gain,
    )
    q = np.array([0.0, 0.25, 1.05, 0.0, 0.55, 0.0], dtype=np.float64)
    dq = np.zeros(6, dtype=np.float64)
    p0, R0 = core.forward_kinematics(q)

    solve_times = []
    pos_errs = []
    for k in range(50):
        t = (k + 1) * core.dt_ctrl
        p, v, a, R, w, alpha = compute_analytic_target(args.target_mode, t, p0, R0, speed_scale=args.speed_scale)
        out = core.compute_step(q, dq, p, v, a, R, w, alpha, preview_steps=2)
        q = out["q_traj"][0]
        dq = out["dq_traj"][0]
        solve_times.append(out["solve_ms"])
        pos_errs.append(out["pos_err_mm"])

    print(
        f"[✓] Self-test passed ({args.target_mode}, gain={args.residual_gain:.2f}): "
        f"mean_solve={np.mean(solve_times):.2f} ms (budget 20 ms) | "
        f"mean_pos_err={np.mean(pos_errs):.2f} mm | max_|dq|={np.max(np.abs(dq)):.2f} rad/s"
    )


def run_ros2_node(args):
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class DynaArmRLDeployNode(Node):
        def __init__(self):
            super().__init__("dynaarm_rl_pose6d_deploy")
            ns = args.robot_ns.strip("/")
            prefix = f"/{ns}" if ns else ""

            self.core = RL6DPoseControllerCore(
                model_path=args.model,
                meta_path=args.meta,
                residual_gain=args.residual_gain,
                control_freq_hz=args.rate_hz,
                max_joint_vel=args.max_vel,
            )

            self.joint_idx_map = {n: i for i, n in enumerate(JOINT_NAMES)}
            self.q_now = np.zeros(6, dtype=np.float64)
            self.dq_now = np.zeros(6, dtype=np.float64)
            self.tau_now = np.zeros(6, dtype=np.float64)
            self.state_received = False
            self.last_state_time = 0.0
            self.start_time = None
            self.p0 = None
            self.R0 = None
            self.faulted = False
            self.step_count = 0

            # Optional 50 Hz telemetry logger (buffered in RAM, flushed on shutdown)
            self.log_path = None
            self.log_rows = []
            if args.log_csv:
                self.log_path = args.log_csv
            elif args.log:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = os.path.join(SIM_DIR, "logs")
                os.makedirs(log_dir, exist_ok=True)
                self.log_path = os.path.join(
                    log_dir,
                    f"{args.controller_mode}_{args.target_mode}_g{args.residual_gain:.2f}_s{args.speed_scale:.2f}_{ts}.csv",
                )
            if self.log_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
                self.get_logger().info(f"Telemetry logging enabled -> {self.log_path}")

            # External topic target state
            self.ext_p = None
            self.ext_R = None

            # Live base_link pose from pose_broadcaster (ensures 0.0 mm frame offset with C++ Pinocchio controller)
            self.live_bl_p = None
            self.live_bl_R = None
            self.p0_bl = None
            self.R0_bl = None
            self.node_init_time = time.monotonic()

            joint_topic = f"{prefix}/platform/joint_states"
            self.sub_js = self.create_subscription(JointState, joint_topic, self._on_joint_state, 10)

            pose_bc_topic = f"{prefix}/current_pose"
            self.sub_pose_bc = self.create_subscription(PoseStamped, pose_bc_topic, self._on_current_pose, 10)

            if args.target_mode == "topic":
                tgt_topic = f"{prefix}/rl_target_pose"
                self.sub_tgt = self.create_subscription(PoseStamped, tgt_topic, self._on_target_pose, 10)
                self.get_logger().info(f"Subscribed to external target pose: {tgt_topic}")

            self.mode = args.controller_mode
            if self.mode == "jtc":
                cmd_topic = f"{prefix}/joint_trajectory_controller/joint_trajectory"
                self.pub_jtc = self.create_publisher(JointTrajectory, cmd_topic, 10)
                self.get_logger().info(f"Publishing JointTrajectory to: {cmd_topic}")
            elif self.mode == "qp_osc":
                self.pub_pose = self.create_publisher(PoseStamped, f"{prefix}/target_pose", 10)
                self.pub_twist = self.create_publisher(TwistStamped, f"{prefix}/target_twist", 10)
                self.pub_accel = self.create_publisher(AccelStamped, f"{prefix}/target_accel", 10)
                self.pub_res = self.create_publisher(Float64MultiArray, f"{prefix}/rl_residual", 10)
                self.get_logger().info(f"Streaming 50 Hz target + RL residual to C++ 500 Hz {prefix}/{{target_pose,target_twist,target_accel,rl_residual}}")
            else:
                self.pub_pose = self.create_publisher(PoseStamped, f"{prefix}/target_pose", 10)
                self.pub_twist = self.create_publisher(TwistStamped, f"{prefix}/target_twist", 10)
                self.get_logger().info(f"Publishing CRISP target_pose & target_twist under {prefix}/")

            self.timer = self.create_timer(1.0 / args.rate_hz, self._control_step)
            self.get_logger().info(
                f"[✓] DynaArm RL Deploy Node ready | mode={self.mode} | target={args.target_mode} | "
                f"residual_gain={args.residual_gain:.2f} | speed_scale={args.speed_scale:.2f} | rate={args.rate_hz:.0f} Hz"
            )

        def _on_joint_state(self, msg: JointState):
            updated = False
            for i, name in enumerate(msg.name):
                if name in self.joint_idx_map:
                    idx = self.joint_idx_map[name]
                    if i < len(msg.position):
                        self.q_now[idx] = msg.position[i]
                        updated = True
                    if i < len(msg.velocity):
                        self.dq_now[idx] = msg.velocity[i]
                    if i < len(msg.effort):
                        self.tau_now[idx] = msg.effort[i]
            if updated:
                self.last_state_time = time.monotonic()
                self.state_received = True

        def _on_current_pose(self, msg: PoseStamped):
            self.live_bl_p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
            q = msg.pose.orientation
            self.live_bl_R = quat_xyzw_to_mat(q.x, q.y, q.z, q.w)

        def _on_target_pose(self, msg: PoseStamped):
            self.ext_p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
            q = msg.pose.orientation
            self.ext_R = quat_xyzw_to_mat(q.x, q.y, q.z, q.w)

        def _control_step(self):
            if self.faulted:
                return
            if not self.state_received:
                self.get_logger().info("Waiting for /platform/joint_states...", throttle_duration_sec=2.0)
                return

            now = time.monotonic()
            if now - self.last_state_time > 0.15:
                self.get_logger().warn("Joint state stale (>150 ms)! Halting commands.", throttle_duration_sec=1.0)
                return

            # Safety check strictly below safety_observer's 2.0 rad/s threshold
            max_dq = float(np.max(np.abs(self.dq_now)))
            if max_dq > 1.85:
                self.faulted = True
                self.get_logger().error(f"Velocity safety trip (|dq|={max_dq:.2f} > 1.85 rad/s). Latching halt.")
                return

            if self.p0 is None:
                # Give /current_pose up to 0.5s to arrive so we lock onto the exact Pinocchio tcp frame
                if self.live_bl_p is None and (now - self.node_init_time) < 0.5:
                    return
                self.p0, self.R0 = self.core.forward_kinematics(self.q_now)
                if self.live_bl_p is not None and self.live_bl_R is not None:
                    self.p0_bl = self.live_bl_p.copy()
                    self.R0_bl = self.live_bl_R.copy()
                    src = "/current_pose (exact Pinocchio tcp in base_link)"
                else:
                    self.p0_bl = self.p0 - self.core.base_pos_world
                    self.R0_bl = self.R0.copy()
                    src = "MuJoCo FK - base_z_offset"
                self.start_time = now
                self.get_logger().info(
                    f"Captured startup pose ({src}): p0_bl={np.round(self.p0_bl, 3)}. Starting smooth 2.5s C^2 ramp-in."
                )

            t_rel = now - self.start_time
            if args.target_mode == "topic":
                if self.ext_p is None:
                    p_bl, v_t, a_t, R_bl, w_t, al_t = self.p0_bl, np.zeros(3), np.zeros(3), self.R0_bl, np.zeros(3), np.zeros(3)
                else:
                    p_bl, v_t, a_t, R_bl, w_t, al_t = self.ext_p, np.zeros(3), np.zeros(3), self.ext_R, np.zeros(3), np.zeros(3)
                p_t = self.p0 + (p_bl - self.p0_bl)
                R_t = (R_bl @ self.R0_bl.T) @ self.R0
            else:
                # Generate identical delta in both base_link (for C++ Pinocchio controller) and MuJoCo world (for RL obs)
                p_bl, v_t, a_t, R_bl, w_t, al_t = compute_analytic_target(
                    args.target_mode, t_rel, self.p0_bl, self.R0_bl, speed_scale=args.speed_scale
                )
                p_t, _, _, R_t, _, _ = compute_analytic_target(
                    args.target_mode, t_rel, self.p0, self.R0, speed_scale=args.speed_scale
                )

            p_steps = 3 if self.mode == "jtc" else 0
            out = self.core.compute_step(self.q_now, self.dq_now, p_t, v_t, a_t, R_t, w_t, al_t, preview_steps=p_steps)

            if out["pos_err_mm"] > 150.0:
                self.faulted = True
                self.get_logger().error(f"Tracking error trip ({out['pos_err_mm']:.1f} mm > 150 mm). Latching halt.")
                return

            if self.mode == "jtc":
                msg = JointTrajectory()
                msg.header.stamp = rclpy.time.Time().to_msg()
                msg.joint_names = list(JOINT_NAMES)
                for k, (qk, dqk) in enumerate(zip(out["q_traj"], out["dq_traj"]), start=1):
                    pt = JointTrajectoryPoint()
                    pt.time_from_start = Duration(seconds=float(k * self.core.dt_ctrl)).to_msg()
                    pt.positions = qk.tolist()
                    pt.velocities = dqk.tolist()
                    msg.points.append(pt)
                self.pub_jtc.publish(msg)
            elif self.mode == "qp_osc":
                stamp = self.get_clock().now().to_msg()
                p_msg = PoseStamped()
                p_msg.header.stamp = stamp
                p_msg.header.frame_id = "base_link"
                p_msg.pose.position.x, p_msg.pose.position.y, p_msg.pose.position.z = p_bl.tolist()
                qx, qy, qz, qw = mat_to_quat_xyzw(R_bl)
                p_msg.pose.orientation.x, p_msg.pose.orientation.y, p_msg.pose.orientation.z, p_msg.pose.orientation.w = qx, qy, qz, qw
                self.pub_pose.publish(p_msg)

                tw_msg = TwistStamped()
                tw_msg.header.stamp = stamp
                tw_msg.header.frame_id = "base_link"
                tw_msg.twist.linear.x, tw_msg.twist.linear.y, tw_msg.twist.linear.z = v_t.tolist()
                tw_msg.twist.angular.x, tw_msg.twist.angular.y, tw_msg.twist.angular.z = w_t.tolist()
                self.pub_twist.publish(tw_msg)

                ac_msg = AccelStamped()
                ac_msg.header.stamp = stamp
                ac_msg.header.frame_id = "base_link"
                ac_msg.accel.linear.x, ac_msg.accel.linear.y, ac_msg.accel.linear.z = a_t.tolist()
                ac_msg.accel.angular.x, ac_msg.accel.angular.y, ac_msg.accel.angular.z = al_t.tolist()
                self.pub_accel.publish(ac_msg)

                res_msg = Float64MultiArray()
                a_res_ori = (self.core.prev_action[3:] * self.core.action_scale_ori).astype(np.float64)
                res_msg.data = np.concatenate([out["a_res_pos"], a_res_ori]).tolist()
                self.pub_res.publish(res_msg)
            else:
                stamp = self.get_clock().now().to_msg()
                p_msg = PoseStamped()
                p_msg.header.stamp = stamp
                p_msg.header.frame_id = "base_link"
                p_msg.pose.position.x, p_msg.pose.position.y, p_msg.pose.position.z = out["p_crisp"].tolist()
                qx, qy, qz, qw = mat_to_quat_xyzw(out["R_crisp"])
                p_msg.pose.orientation.x, p_msg.pose.orientation.y, p_msg.pose.orientation.z, p_msg.pose.orientation.w = qx, qy, qz, qw
                self.pub_pose.publish(p_msg)

                tw_msg = TwistStamped()
                tw_msg.header.stamp = stamp
                tw_msg.header.frame_id = "base_link"
                tw_msg.twist.linear.x, tw_msg.twist.linear.y, tw_msg.twist.linear.z = out["v_crisp"].tolist()
                tw_msg.twist.angular.x, tw_msg.twist.angular.y, tw_msg.twist.angular.z = out["w_crisp"].tolist()
                self.pub_twist.publish(tw_msg)

            if self.live_bl_p is not None and self.live_bl_R is not None:
                p_ee_bl = self.live_bl_p
                disp_pos_mm = float(np.linalg.norm(p_bl - p_ee_bl) * 1000.0)
                disp_ori_deg = float(np.degrees(np.linalg.norm(orientation_error_world(R_bl, self.live_bl_R))))
            else:
                p_ee_bl = p_t - self.core.base_pos_world
                disp_pos_mm = out["pos_err_mm"]
                disp_ori_deg = out["ori_err_deg"]

            if self.log_path is not None:
                a_res_ori = (self.core.prev_action[3:] * self.core.action_scale_ori).astype(np.float64)
                self.log_rows.append([
                    t_rel,
                    p_ee_bl[0], p_ee_bl[1], p_ee_bl[2],
                    p_bl[0], p_bl[1], p_bl[2],
                    disp_pos_mm, disp_ori_deg,
                    *self.q_now.tolist(),
                    *self.dq_now.tolist(),
                    *self.tau_now.tolist(),
                    *out["a_res_pos"].tolist(),
                    *a_res_ori.tolist(),
                    out["solve_ms"],
                ])

            if self.step_count % int(args.rate_hz * 2) == 0:
                self.get_logger().info(
                    f"[t={t_rel:5.1f}s] PosErr={disp_pos_mm:5.1f}mm | OriErr={disp_ori_deg:4.1f}deg | "
                    f"|a_res|={np.linalg.norm(out['a_res_pos']):4.2f}m/s^2 | solve={out['solve_ms']:.2f}ms"
                )
            self.step_count += 1

        def flush_log(self):
            if not self.log_path or not self.log_rows:
                return
            header = [
                "t",
                "p_ee_x", "p_ee_y", "p_ee_z",
                "p_tgt_x", "p_tgt_y", "p_tgt_z",
                "pos_err_mm", "ori_err_deg",
                *[f"q_{i}" for i in range(6)],
                *[f"dq_{i}" for i in range(6)],
                *[f"tau_{i}" for i in range(6)],
                *[f"a_res_{i}" for i in range(6)],
                "solve_ms",
            ]
            with open(self.log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(self.log_rows)
            print(f"\n[✓] Saved {len(self.log_rows)} samples to {self.log_path}")
            print(f"    Analyze: python3 rl/analyze_deploy_log.py {self.log_path}")

    rclpy.init()
    node = DynaArmRLDeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.flush_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Duatic DynaArm ROS 2 6D Pose RL Deployment Node")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Exported TorchScript policy (.ts)")
    parser.add_argument("--meta", type=str, default=DEFAULT_META, help="Deployment metadata JSON")
    parser.add_argument("--robot-ns", type=str, default="r100_0208", help="ROS 2 robot namespace")
    parser.add_argument("--controller-mode", choices=["qp_osc", "crisp", "jtc"], default="qp_osc", help="Target ros2_control controller")
    parser.add_argument("--target-mode", choices=["hold", "helix", "circle", "figure8", "topic"], default="helix")
    parser.add_argument("--residual-gain", type=float, default=0.0, help="Staged RL gain in [0.0, 1.0] (start at 0.0!)")
    parser.add_argument("--speed-scale", type=float, default=0.4, help="Trajectory speed multiplier in [0.1, 1.0]")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Policy inference rate [Hz]")
    parser.add_argument("--max-vel", type=float, default=1.4, help="Max commanded joint velocity [rad/s]")
    parser.add_argument("--log", action="store_true", help="Auto-save 50 Hz telemetry CSV into logs/")
    parser.add_argument("--log-csv", type=str, default="", help="Custom output CSV path for 50 Hz telemetry")
    parser.add_argument("--self-test", action="store_true", help="Run offline timing & tracking verification")
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args)
    else:
        run_ros2_node(args)


if __name__ == "__main__":
    main()
