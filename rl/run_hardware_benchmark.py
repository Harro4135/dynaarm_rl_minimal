#!/usr/bin/env python3
"""
========================================================================================
DUATIC DYNAARM HARDWARE BENCHMARK SUITE (JTC vs PVTP QP-OSC vs PVTP QP-OSC + RL)
========================================================================================
Runs an apples-to-apples multi-trajectory evaluation on the real (or mock) Ridgeback +
DynaArm robot:
  - Assumes the arm starts in `joint_trajectory_controller` (JTC) active.
  - Smoothly repositions to an identical home pose via JTC before EVERY trial.
  - Handles safe controller switching (`joint_trajectory_controller` <-> `qp_osc_rl_controller`)
    and `residual_gain` parameter reconfiguration (via unconfigured -> set -> inactive).
  - Records full 50 Hz telemetry CSVs for every (trajectory, method) pair.
  - Always returns safely to `joint_trajectory_controller` at home on completion or Ctrl+C.
  - Generates a complete Markdown comparison table + per-run jerk/chatter/FFT plots.

Usage:
  ./run_benchmark.sh
  ./run_benchmark.sh --trajectories figure8 helix circle --speed-scale 0.5 --duration 15
  ./run_benchmark.sh --methods jtc qp_osc_base qp_osc_rl --rl-gain 1.0 --no-pause
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
import numpy as np

RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(RL_DIR)
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

from analyze_deploy_log import analyze_file
from dynaarm_reach_env import HOME_POSITION, JOINT_NAMES


def run_cmd(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


class HardwareBenchmarkRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ns = args.robot_ns.strip("/")
        self.prefix = f"/{self.ns}" if self.ns else ""
        self.cm = f"{self.prefix}/controller_manager"
        self.active_controller: str | None = None
        self.current_c_residual_gain: float | None = None

        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory

        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()

        self.node = Node("dynaarm_hardware_benchmark")
        self.joint_idx_map = {n: i for i, n in enumerate(JOINT_NAMES)}
        self.q_now = np.zeros(6, dtype=np.float64)
        self.dq_now = np.zeros(6, dtype=np.float64)
        self.tau_now = np.zeros(6, dtype=np.float64)
        self.live_bl_p = None
        self.live_bl_R = None
        self.state_received = False

        self.sub_js = self.node.create_subscription(
            JointState, f"{self.prefix}/platform/joint_states", self._on_js, 10
        )
        self.sub_pose = self.node.create_subscription(
            PoseStamped, f"{self.prefix}/current_pose", self._on_pose, 10
        )
        self.pub_jtc = self.node.create_publisher(
            JointTrajectory, f"{self.prefix}/joint_trajectory_controller/joint_trajectory", 10
        )

    def _on_js(self, msg):
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
            self.state_received = True

    def _on_pose(self, msg):
        from deploy_ros2_node import quat_xyzw_to_mat
        self.live_bl_p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        q = msg.pose.orientation
        self.live_bl_R = quat_xyzw_to_mat(q.x, q.y, q.z, q.w)

    def spin_for(self, seconds: float):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def wait_for_joint_states(self, timeout_s: float = 10.0):
        print(f"[*] Waiting for {self.prefix}/platform/joint_states...")
        t_end = time.monotonic() + timeout_s
        while not self.state_received and time.monotonic() < t_end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if not self.state_received:
            raise RuntimeError(f"Timed out waiting for {self.prefix}/platform/joint_states")
        print(f"[✓] Live joint_states received: q={np.round(self.q_now, 3).tolist()}")

    def switch_to_jtc(self):
        if self.active_controller == "joint_trajectory_controller":
            return
        print("[*] Switching controllers: qp_osc_rl_controller -> joint_trajectory_controller + gravity_compensation_controller")
        run_cmd([
            "ros2", "control", "switch_controllers", "-c", self.cm,
            "--activate", "joint_trajectory_controller", "gravity_compensation_controller",
            "--deactivate", "qp_osc_rl_controller", "freeze_controller",
            "--best-effort",
        ])
        self.active_controller = "joint_trajectory_controller"
        self.spin_for(0.5)

    def ensure_qp_osc_loaded(self):
        res = run_cmd(["ros2", "control", "list_controllers", "-c", self.cm], capture=True, check=False)
        out = res.stdout or ""
        if "qp_osc_rl_controller" not in out:
            print("[*] qp_osc_rl_controller not loaded yet — loading & configuring...")
            run_cmd(["ros2", "control", "load_controller", "-c", self.cm, "qp_osc_rl_controller"])
            time.sleep(0.3)
            run_cmd(["ros2", "param", "set", f"{self.prefix}/qp_osc_rl_controller", "residual_gain", "1.0"])
            run_cmd(["ros2", "control", "set_controller_state", "-c", self.cm, "qp_osc_rl_controller", "inactive"])
            self.current_c_residual_gain = 1.0
        elif "unconfigured" in [line for line in out.splitlines() if "qp_osc_rl_controller" in line][0]:
            run_cmd(["ros2", "param", "set", f"{self.prefix}/qp_osc_rl_controller", "residual_gain", "1.0"])
            run_cmd(["ros2", "control", "set_controller_state", "-c", self.cm, "qp_osc_rl_controller", "inactive"])
            self.current_c_residual_gain = 1.0

    def configure_and_switch_to_qp_osc(self, residual_gain: float):
        # Ensure JTC + gravity comp hold the arm; keep C++ residual_gain=1.0 and gate via Python --residual-gain
        self.switch_to_jtc()
        self.ensure_qp_osc_loaded()

        print("[*] Switching controllers: joint_trajectory_controller + gravity_compensation_controller -> qp_osc_rl_controller")
        run_cmd([
            "ros2", "control", "switch_controllers", "-c", self.cm,
            "--activate", "qp_osc_rl_controller",
            "--deactivate", "joint_trajectory_controller", "gravity_compensation_controller", "freeze_controller",
            "--best-effort",
        ])
        self.active_controller = "qp_osc_rl_controller"

        # Safety standstill check: verify arm holds quietly for 1.0s with no unexpected motion
        self.spin_for(1.0)
        max_dq = float(np.max(np.abs(self.dq_now)))
        if max_dq > 0.25:
            self.switch_to_jtc()
            raise RuntimeError(f"Standstill safety trip after switching to qp_osc_rl_controller (|dq|={max_dq:.2f} rad/s)!")
        print(f"[✓] qp_osc_rl_controller active & quiet at standstill (|dq|={max_dq:.3f} rad/s)")

    def reposition_to_home_via_jtc(self, q_home: np.ndarray, duration_s: float = 4.0):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        self.switch_to_jtc()
        self.spin_for(0.1)
        err0 = float(np.max(np.abs(self.q_now - q_home)))
        if err0 < 0.015:
            print(f"[✓] Already at home pose (max |dq_err|={err0:.4f} rad).")
            self.spin_for(self.args.settle_time)
            return

        # Scale duration gently if far from home
        move_t = max(duration_s, min(8.0, err0 / 0.35))
        print(f"[*] Repositioning to home via JTC over {move_t:.1f}s (max joint delta={err0:.3f} rad)...")
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = q_home.tolist()
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(sec=int(move_t), nanosec=int((move_t - int(move_t)) * 1e9))
        msg.points.append(pt)
        self.pub_jtc.publish(msg)

        t_deadline = time.monotonic() + move_t + 3.0
        while time.monotonic() < t_deadline:
            self.spin_for(0.1)
            if float(np.max(np.abs(self.q_now - q_home))) < 0.025 and float(np.max(np.abs(self.dq_now))) < 0.05:
                break

        final_err = float(np.max(np.abs(self.q_now - q_home)))
        print(f"[✓] Reposition complete (max joint err={final_err:.4f} rad). Settling {self.args.settle_time:.1f}s...")
        self.spin_for(self.args.settle_time)

    def run_native_jtc_trial(self, traj: str, csv_path: str):
        """
        Precompute the full smooth IK joint trajectory and send ONE single JointTrajectory
        message to joint_trajectory_controller (no 50 Hz spline preemption, factory default gains!),
        while logging 50 Hz telemetry identical to deploy_ros2_node.py.
        """
        import csv
        import mujoco
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from deploy_ros2_node import compute_analytic_target
        from dynaarm_reach_env import SAFE_LIMITS_HIGH, SAFE_LIMITS_LOW, orientation_error_world

        self.switch_to_jtc()
        self.spin_for(0.2)

        xml_path = os.path.join(REPO_DIR, "models", "ridgeback_duatic_kinematics.xml")
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        qpos_idx = np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES])
        qvel_idx = np.array([model.joint(n).dofadr[0] for n in JOINT_NAMES])
        ee_site_id = model.site("ee_site").id
        jac_p = np.zeros((3, model.nv), dtype=np.float64)
        jac_r = np.zeros((3, model.nv), dtype=np.float64)

        q_sim = self.q_now.copy()
        data.qpos[:] = 0.0
        data.qpos[qpos_idx] = q_sim
        mujoco.mj_forward(model, data)
        p0_mj = data.site_xpos[ee_site_id].copy()
        R0_mj = data.site_xmat[ee_site_id].reshape(3, 3).copy()
        p0_bl = self.live_bl_p.copy() if self.live_bl_p is not None else p0_mj.copy()
        R0_bl = self.live_bl_R.copy() if self.live_bl_R is not None else R0_mj.copy()

        # Precompute smooth 500 Hz CLIK -> downsample to clean 10 Hz spline waypoints for JTC
        dt_ik = 0.002
        n_steps = int(round(self.args.duration / dt_ik))
        waypoint_stride = 50  # 0.1s (10 Hz) knots -> silky C^2 spline inside JTC
        q_lo = SAFE_LIMITS_LOW + 0.05
        q_hi = SAFE_LIMITS_HIGH - 0.05

        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        for k in range(1, n_steps + 1):
            t_k = k * dt_ik
            p_t, v_t, _, R_t, w_t, _ = compute_analytic_target(traj, t_k, p0_mj, R0_mj, speed_scale=self.args.speed_scale)
            data.qpos[:] = 0.0
            data.qpos[qpos_idx] = q_sim
            mujoco.mj_forward(model, data)
            p_c = data.site_xpos[ee_site_id].copy()
            R_c = data.site_xmat[ee_site_id].reshape(3, 3).copy()
            jac_p.fill(0.0); jac_r.fill(0.0)
            mujoco.mj_jacSite(model, data, jac_p, jac_r, ee_site_id)
            J6 = np.vstack([jac_p[:, qvel_idx], jac_r[:, qvel_idx]])
            v6 = np.concatenate([v_t + 15.0 * (p_t - p_c), w_t + 8.0 * orientation_error_world(R_t, R_c).astype(np.float64)])
            sv = float(np.linalg.svd(J6, compute_uv=False)[-1])
            lam = 0.025 + 0.12 * np.exp(-((sv / 0.035) ** 2))
            H = J6.T @ J6 + (lam ** 2) * np.eye(6)
            dq_k = np.clip(np.linalg.solve(H, J6.T @ v6 + (lam ** 2) * 4.0 * (HOME_POSITION - q_sim)), -1.5, 1.5)
            q_sim = np.clip(q_sim + dq_k * dt_ik, q_lo, q_hi)

            if k % waypoint_stride == 0 or k == n_steps:
                pt = JointTrajectoryPoint()
                pt.positions = q_sim.tolist()
                pt.velocities = (np.zeros(6) if k == n_steps else dq_k).tolist()
                pt.time_from_start = Duration(sec=int(t_k), nanosec=int((t_k - int(t_k)) * 1e9))
                msg.points.append(pt)

        if not self.args.no_pause:
            input(f"\n>>> Ready to run [{traj} | jtc (single smooth spline, {len(msg.points)} knots) | "
                  f"speed={self.args.speed_scale:.2f} | {self.args.duration:.0f}s]. "
                  f"Keep thumb on E-stop and press ENTER to start... ")

        print(f"\n[*] Publishing single smooth JointTrajectory ({len(msg.points)} knots, {self.args.duration:.1f}s) -> {csv_path}")
        t_start = time.monotonic()
        self.pub_jtc.publish(msg)

        rows = []
        next_sample = t_start
        while True:
            now = time.monotonic()
            t_rel = now - t_start
            if t_rel >= self.args.duration:
                break
            self.rclpy.spin_once(self.node, timeout_sec=0.005)
            if now >= next_sample:
                p_bl, _, _, R_bl, _, _ = compute_analytic_target(traj, t_rel, p0_bl, R0_bl, speed_scale=self.args.speed_scale)
                if self.live_bl_p is not None and self.live_bl_R is not None:
                    p_ee = self.live_bl_p
                    pos_mm = float(np.linalg.norm(p_bl - p_ee) * 1000.0)
                    ori_deg = float(np.degrees(np.linalg.norm(orientation_error_world(R_bl, self.live_bl_R))))
                else:
                    p_ee = p_bl
                    pos_mm = 0.0
                    ori_deg = 0.0
                rows.append([
                    t_rel, p_ee[0], p_ee[1], p_ee[2], p_bl[0], p_bl[1], p_bl[2],
                    pos_mm, ori_deg, *self.q_now.tolist(), *self.dq_now.tolist(), *self.tau_now.tolist(),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ])
                next_sample += 0.02

        header = [
            "t", "p_ee_x", "p_ee_y", "p_ee_z", "p_tgt_x", "p_tgt_y", "p_tgt_z",
            "pos_err_mm", "ori_err_deg",
            *[f"q_{i}" for i in range(6)], *[f"dq_{i}" for i in range(6)],
            *[f"tau_{i}" for i in range(6)], *[f"a_res_{i}" for i in range(6)], "solve_ms",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"[✓] Saved {len(rows)} native-JTC samples to {csv_path}")
        self.spin_for(0.5)

    def run_trial(self, traj: str, method: str, csv_path: str):
        if method == "jtc":
            self.run_native_jtc_trial(traj, csv_path)
            return
        elif method == "qp_osc_base":
            ctrl_mode = "qp_osc"
            py_gain = 0.0
            self.configure_and_switch_to_qp_osc(residual_gain=0.0)
        elif method == "qp_osc_rl":
            ctrl_mode = "qp_osc"
            py_gain = float(self.args.rl_gain)
            self.configure_and_switch_to_qp_osc(residual_gain=1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

        if not self.args.no_pause:
            input(f"\n>>> Ready to run [{traj} | {method} | speed={self.args.speed_scale:.2f} | {self.args.duration:.0f}s]. "
                  f"Keep thumb on E-stop and press ENTER to start (Ctrl+C to abort)... ")

        cmd = [
            sys.executable, os.path.join(RL_DIR, "deploy_ros2_node.py"),
            "--robot-ns", self.ns,
            "--controller-mode", ctrl_mode,
            "--target-mode", traj,
            "--residual-gain", f"{py_gain:.3f}",
            "--speed-scale", f"{self.args.speed_scale:.3f}",
            "--duration", f"{self.args.duration:.2f}",
            "--log-csv", csv_path,
        ]
        if self.args.model:
            cmd += ["--model", self.args.model]
        if self.args.meta:
            cmd += ["--meta", self.args.meta]

        print(f"\n[*] Running [{traj} | {method}] for {self.args.duration:.1f}s -> {csv_path}")
        run_cmd(cmd, check=True)
        self.spin_for(0.5)

    def shutdown(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self.rclpy.ok():
            self.rclpy.shutdown()


def write_summary_markdown(results: list[tuple[str, str, dict]], out_dir: str, args: argparse.Namespace):
    lines = [
        f"# DynaArm Hardware Benchmark Summary",
        f"- **Speed scale:** `{args.speed_scale}` | **Duration:** `{args.duration}s` | **RL gain:** `{args.rl_gain}`",
        "",
        "| Trajectory | Method | Pos Mean (P95 / Max) | Ori Mean (P95 / Max) | Cart Jerk RMS | Joint Jerk RMS | Startup Jerk Max | Torque dTau/dt | FFT Peak (dq / tau) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for traj, method, m in results:
        lines.append(
            f"| **{traj}** | `{method}` | "
            f"`{m['pos_mean']:.1f}` ({m['pos_p95']:.1f} / {m['pos_max']:.1f}) mm | "
            f"`{m['ori_mean']:.2f}` ({m['ori_p95']:.2f} / {m['ori_max']:.2f})° | "
            f"`{m['cart_jerk_rms']:.2f}` m/s³ | "
            f"`{m['joint_jerk_rms']:.1f}` rad/s³ | "
            f"`{m['startup_jerk_max']:.1f}` rad/s³ | "
            f"`{m['dtau_rms']:.1f}` Nm/s | "
            f"{m['f_dq']:.1f} / {m['f_tau']:.1f} Hz |"
        )
    md = "\n".join(lines) + "\n"
    md_path = os.path.join(out_dir, "benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"[✓] Saved benchmark summary -> {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Hardware Benchmark: JTC vs PVTP QP-OSC vs PVTP QP-OSC + RL")
    parser.add_argument("--robot-ns", type=str, default="r100_0208", help="ROS 2 robot namespace")
    parser.add_argument("--trajectories", nargs="+", default=["figure8", "helix", "circle"],
                        choices=["figure8", "helix", "circle", "hold"], help="Trajectories to evaluate")
    parser.add_argument("--methods", nargs="+", default=["jtc", "qp_osc_base", "qp_osc_rl"],
                        choices=["jtc", "qp_osc_base", "qp_osc_rl"], help="Control methods to compare")
    parser.add_argument("--speed-scale", type=float, default=0.5, help="Trajectory speed multiplier in [0.1, 1.0]")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration [s] per trial (includes 2.5s C^2 ramp-in)")
    parser.add_argument("--rl-gain", type=float, default=1.0, help="RL residual gain for qp_osc_rl method")
    parser.add_argument("--reposition-time", type=float, default=4.0, help="JTC reposition duration [s] between trials")
    parser.add_argument("--settle-time", type=float, default=1.0, help="Settle pause [s] at home before switching")
    parser.add_argument("--home-q", nargs=6, type=float, default=HOME_POSITION.tolist(),
                        help="Home joint angles [rad] used to reposition between trials")
    parser.add_argument("--use-current-as-home", action="store_true",
                        help="Use the robot's initial pose at script startup as the home pose instead of --home-q")
    parser.add_argument("--no-pause", action="store_true",
                        help="Run all trials back-to-back without pressing ENTER between trials")
    parser.add_argument("--model", type=str, default="", help="Optional override path to policy_deploy.ts")
    parser.add_argument("--meta", type=str, default="", help="Optional override path to policy_deploy_meta.json")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory for CSVs, plots & summary")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(REPO_DIR, "logs", f"benchmark_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    runner = HardwareBenchmarkRunner(args)
    results: list[tuple[str, str, dict]] = []

    try:
        runner.wait_for_joint_states()
        q_home = runner.q_now.copy() if args.use_current_as_home else np.asarray(args.home_q, dtype=np.float64)
        print(f"[*] Benchmark output dir : {out_dir}")
        print(f"[*] Home joint pose (rad): {np.round(q_home, 3).tolist()}")
        print(f"[*] Plan                 : trajectories={args.trajectories} x methods={args.methods}")

        # Run all JTC trials FIRST (before qp_osc_rl_controller ever modifies onboard PID gains)
        # so JTC runs with 100% untouched factory/default hardware gains!
        ordered_methods = ([m for m in args.methods if m == "jtc"] +
                           [m for m in args.methods if m != "jtc"])
        for method in ordered_methods:
            for traj in args.trajectories:
                print("\n" + "=" * 78)
                print(f"TRIAL: trajectory={traj} | method={method} | speed={args.speed_scale:.2f}")
                print("=" * 78)
                runner.reposition_to_home_via_jtc(q_home, duration_s=args.reposition_time)
                csv_path = os.path.join(out_dir, f"{traj}_{method}.csv")
                runner.run_trial(traj, method, csv_path)
                if os.path.exists(csv_path):
                    metrics = analyze_file(csv_path)
                    if metrics:
                        results.append((traj, method, metrics))

        # Return cleanly to JTC at home when done
        print("\n[*] Benchmark complete! Returning arm safely to home via JTC...")
        runner.reposition_to_home_via_jtc(q_home, duration_s=args.reposition_time)

    except KeyboardInterrupt:
        print("\n[!] Benchmark interrupted by user (Ctrl+C). Returning safely to JTC...")
        try:
            runner.switch_to_jtc()
        except Exception as e:
            print(f"[!] Could not auto-switch back to JTC: {e}")
    finally:
        if results:
            write_summary_markdown(results, out_dir, args)
        runner.shutdown()


if __name__ == "__main__":
    main()
