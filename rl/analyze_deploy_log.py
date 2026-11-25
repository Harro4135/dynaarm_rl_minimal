#!/usr/bin/env python3
"""
Analyze 50 Hz real-robot telemetry CSV logs from deploy_ros2_node.py (--log).

Computes:
  - Position & orientation tracking error (mean, RMS, P95, max; full & steady-state t > 2.5s)
  - Cartesian velocity, acceleration, and JERK (m/s^3)
  - Joint velocity, acceleration, and JERK (rad/s^3)
  - Torque rate / chatter (dTau/dt in Nm/s, max |dTau| per step)
  - Startup transient metrics (t <= 3.0s)
  - Dominant oscillation frequency (FFT peak Hz in joint velocity / torque)
  - Optional multi-panel PNG plot saved next to the CSV (if matplotlib is installed)

Usage:
  python3 rl/analyze_deploy_log.py logs/qp_osc_figure8_*.csv
  python3 rl/analyze_deploy_log.py logs/run1.csv logs/run2.csv   # side-by-side comparison
"""

import csv
import os
import sys
import numpy as np


def load_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, r)) for r in reader if r]
    arr = np.asarray(rows, dtype=np.float64)
    return {col: arr[:, i] for i, col in enumerate(header)}


def smooth_diff(x: np.ndarray, t: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average pre-smoothed central difference (avoids high-freq encoder differentiation noise)."""
    if len(t) < window + 2:
        return np.gradient(x, t, axis=0)
    kernel = np.ones(window, dtype=np.float64) / window
    if x.ndim == 1:
        xs = np.convolve(np.pad(x, (window // 2, window // 2), mode="edge"), kernel, mode="valid")
    else:
        xs = np.column_stack([
            np.convolve(np.pad(x[:, c], (window // 2, window // 2), mode="edge"), kernel, mode="valid")
            for c in range(x.shape[1])
        ])
    return np.gradient(xs, t, axis=0)


def dominant_freq_hz(sig: np.ndarray, dt: float, fmin: float = 0.8) -> tuple[float, float]:
    """Return (peak_freq_hz, peak_amplitude) above fmin Hz."""
    n = len(sig)
    if n < 32:
        return 0.0, 0.0
    s = sig - np.mean(sig)
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(np.fft.rfft(s * win)) * (2.0 / np.sum(win))
    mask = freqs >= fmin
    if not np.any(mask):
        return 0.0, 0.0
    idx = np.argmax(mag[mask])
    return float(freqs[mask][idx]), float(mag[mask][idx])


def analyze_file(path: str):
    d = load_csv(path)
    t = d["t"]
    if len(t) < 10:
        print(f"[!] Too few samples ({len(t)}) in {path}")
        return None

    dt = float(np.median(np.diff(t)))
    p_ee = np.column_stack([d["p_ee_x"], d["p_ee_y"], d["p_ee_z"]])
    p_tgt = np.column_stack([d["p_tgt_x"], d["p_tgt_y"], d["p_tgt_z"]])
    pos_err = d["pos_err_mm"]
    ori_err = d["ori_err_deg"]
    q = np.column_stack([d[f"q_{i}"] for i in range(6)])
    dq = np.column_stack([d[f"dq_{i}"] for i in range(6)])
    tau = np.column_stack([d[f"tau_{i}"] for i in range(6)])
    a_res = np.column_stack([d[f"a_res_{i}"] for i in range(6)])

    # Cartesian kinematics & jerk
    v_ee = smooth_diff(p_ee, t, window=5)
    a_ee = smooth_diff(v_ee, t, window=5)
    j_ee = smooth_diff(a_ee, t, window=5)
    j_ee_norm = np.linalg.norm(j_ee, axis=1)

    # Joint acceleration & jerk (from measured dq)
    ddq = smooth_diff(dq, t, window=5)
    dddq = smooth_diff(ddq, t, window=5)
    dddq_norm = np.linalg.norm(dddq, axis=1)

    # Torque rate / chatter
    dtau_step = np.abs(np.diff(tau, axis=0))
    dtau_dt = np.abs(np.gradient(tau, t, axis=0))

    ss = t >= 2.5
    start = t <= 3.0
    if not np.any(ss):
        ss = np.ones_like(t, dtype=bool)

    f_dq, amp_dq = dominant_freq_hz(np.linalg.norm(dq[ss], axis=1), dt, fmin=1.0)
    f_tau, amp_tau = dominant_freq_hz(np.linalg.norm(tau[ss], axis=1), dt, fmin=1.0)

    print("=" * 78)
    print(f"FILE: {os.path.basename(path)}  ({len(t)} samples, {t[-1]:.1f}s @ {1.0/dt:.1f} Hz)")
    print("-" * 78)
    print(f"  Tracking (steady t>=2.5s): PosErr mean={np.mean(pos_err[ss]):5.1f} mm | P95={np.percentile(pos_err[ss],95):5.1f} mm | max={np.max(pos_err[ss]):5.1f} mm")
    print(f"                             OriErr mean={np.mean(ori_err[ss]):5.1f} deg| P95={np.percentile(ori_err[ss],95):5.1f} deg| max={np.max(ori_err[ss]):5.1f} deg")
    print(f"  Startup  (t<=3.0s)       : PosErr max={np.max(pos_err[start]):5.1f} mm | CartJerk max={np.max(j_ee_norm[start]):6.2f} m/s^3 | JointJerk max={np.max(dddq_norm[start]):6.1f} rad/s^3")
    print(f"  Cartesian Jerk (m/s^3)   : RMS={np.sqrt(np.mean(j_ee_norm[ss]**2)):6.2f} | P95={np.percentile(j_ee_norm[ss],95):6.2f} | max={np.max(j_ee_norm[ss]):6.2f}")
    print(f"  Joint Jerk     (rad/s^3) : RMS={np.sqrt(np.mean(dddq_norm[ss]**2)):6.2f} | P95={np.percentile(dddq_norm[ss],95):6.2f} | max={np.max(dddq_norm[ss]):6.2f}")
    print(f"    Per-joint RMS jerk     : {np.round(np.sqrt(np.mean(dddq[ss]**2, axis=0)), 2).tolist()}")
    print(f"  Torque Chatter           : RMS dTau/dt={np.sqrt(np.mean(dtau_dt[ss]**2)):5.1f} Nm/s | max |dTau/step|={np.max(dtau_step):5.2f} Nm")
    print(f"  RL Residual |a_res_pos|  : mean={np.mean(np.linalg.norm(a_res[ss,:3], axis=1)):4.2f} m/s^2 | max={np.max(np.linalg.norm(a_res[ss,:3], axis=1)):4.2f} m/s^2")
    print(f"  Dominant Osc (>1 Hz)     : |dq| peak @ {f_dq:4.2f} Hz (amp={amp_dq:.3f}) | |tau| peak @ {f_tau:4.2f} Hz (amp={amp_tau:.2f} Nm)")
    print("=" * 78)

    # Optional plot if matplotlib is installed
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(t, pos_err, label="PosErr (mm)", lw=1.4)
        ax0r = axes[0].twinx()
        ax0r.plot(t, ori_err, color="tab:orange", label="OriErr (deg)", lw=1.2, alpha=0.85)
        axes[0].set_ylabel("PosErr (mm)")
        ax0r.set_ylabel("OriErr (deg)", color="tab:orange")
        axes[0].set_title(os.path.basename(path))
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, j_ee_norm, label="Cartesian Jerk ||j_ee|| (m/s^3)", color="tab:red", lw=1.2)
        axes[1].set_ylabel("Cart Jerk (m/s^3)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, dddq_norm, label="Joint Jerk ||dddq|| (rad/s^3)", color="tab:purple", lw=1.2)
        axes[2].set_ylabel("Joint Jerk (rad/s^3)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        for j in range(6):
            axes[3].plot(t, tau[:, j], label=f"J{j+1}", lw=1.0, alpha=0.85)
        axes[3].set_ylabel("Measured Torque (Nm)")
        axes[3].set_xlabel("Time (s)")
        axes[3].legend(loc="upper right", ncol=6, fontsize=8)
        axes[3].grid(True, alpha=0.3)

        out_png = os.path.splitext(path)[0] + "_analysis.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  [✓] Saved plot: {out_png}")
    except ImportError:
        pass

    return {"pos_rms": float(np.sqrt(np.mean(pos_err[ss]**2))), "cart_jerk_rms": float(np.sqrt(np.mean(j_ee_norm[ss]**2)))}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rl/analyze_deploy_log.py <log1.csv> [log2.csv ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze_file(path)


if __name__ == "__main__":
    main()
