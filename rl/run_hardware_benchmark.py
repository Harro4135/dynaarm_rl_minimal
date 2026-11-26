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
        self.active_controller = "joint_trajectory_controller"
        self.current_c_residual_gain: float | None = None

        import rclpy
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
        self.state_received = False

        self.sub_js = self.node.create_subscription(
            JointState, f"{self.prefix}/platform/joint_states", self._on_js, 10
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
        if updated:
            self.state_received = True

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
        print("[*] Switching controllers: qp_osc_rl_controller -> joint_trajectory_controller")
        run_cmd([
            "ros2", "control", "switch_controllers", "-c", self.cm,
            "--activate", "joint_trajectory_controller",
            "--deactivate", "qp_osc_rl_controller",
            "--best-effort",
        ])
        self.active_controller = "joint_trajectory_controller"
        self.spin_for(0.5)

    def configure_and_switch_to_qp_osc(self, residual_gain: float):
        # Ensure JTC is holding the arm before reconfiguring qp_osc_rl_controller
        self.switch_to_jtc()

        if self.current_c_residual_gain is None or abs(self.current_c_residual_gain - residual_gain) > 1e-6:
            print(f"[*] Configuring qp_osc_rl_controller residual_gain={residual_gain:.2f} (while JTC holds arm)...")
            run_cmd([
                "ros2", "control", "set_controller_state", "-c", self.cm,
                "qp_osc_rl_controller", "unconfigured",
            ], check=False)
            run_cmd([
                "ros2", "param", "set", f"{self.prefix}/qp_osc_rl_controller",
                "residual_gain", f"{residual_gain:.3f}",
            ])
            res = run_cmd([
                "ros2", "param", "get", f"{self.prefix}/qp_osc_rl_controller",
                "residual_gain",
            ], capture=True)
            print(f"    Verified: {res.stdout.strip()}")
            run_cmd([
                "ros2", "control", "set_controller_state", "-c", self.cm,
                "qp_osc_rl_controller", "inactive",
            ])
            self.current_c_residual_gain = residual_gain

        print("[*] Switching controllers: joint_trajectory_controller -> qp_osc_rl_controller")
        run_cmd([
            "ros2", "control", "switch_controllers", "-c", self.cm,
            "--activate", "qp_osc_rl_controller",
            "--deactivate", "joint_trajectory_controller",
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

    def run_trial(self, traj: str, method: str, csv_path: str):
        if method == "jtc":
            ctrl_mode = "jtc"
            py_gain = 0.0
            self.switch_to_jtc()
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

        for traj in args.trajectories:
            for method in args.methods:
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
