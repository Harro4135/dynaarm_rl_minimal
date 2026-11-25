#!/usr/bin/env python3
"""
========================================================================================
DUATIC DYNAARM GYMNASIUM 6D POSE TRACKING ENVIRONMENT (NO-LOAD, 500 Hz OSC + STRIBECK)
========================================================================================
Clean, reproducible Gymnasium environment for full 6D Cartesian pose (3D position + SO(3)
orientation) tracking on the calibrated Duatic DynaArm (no external payload):
  - Simulated plant: Calibrated MuJoCo model (`models/ridgeback_duatic.xml`) augmented
    with the learned Nov 19 Asymmetric Stribeck friction model from
    `datasets/calibration_real/sysid_parameters.json` (symmetric arm friction/damping
    zeroed on the plant to prevent double-counting).
  - 500 Hz Inner Loop: Separate unperturbed nominal MuJoCo model computes 6D Operational
    Space Control (6x6 Jacobian) + inverse dynamics feedforward at EVERY 2 ms substep,
    eliminating 50 Hz zero-order-hold torque chatter and matching DuaDrive hardware.
  - 50 Hz RL Residual: Policy outputs a bounded 6D Cartesian acceleration residual
    (3D linear + 3D angular) passed through an in-env EMA filter (`action_lpf_alpha`)
    so training and deployment share identical smooth dynamics.

Observation (dim=39):
  - q (6), dq (6)
  - p_ee (3), p_target (3), err_p (3), v_target (3), a_target (3)
  - err_ori (3), w_target (3)
  - prev_action (6)

Action (dim=6):
  - [a_res_pos (3), a_res_ori (3)] in [-1, 1], scaled by (8.0 m/s^2, 12.0 rad/s^2).
"""

import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

RL_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(RL_DIR)
DEFAULT_XML = os.path.join(SIM_DIR, "models", "ridgeback_duatic.xml")
SYSID_JSON = os.path.join(SIM_DIR, "datasets", "calibration_real", "sysid_parameters.json")

JOINT_NAMES = [
    "shoulder_rotation",
    "shoulder_flexion",
    "elbow_flexion",
    "forearm_rotation",
    "wrist_flexion",
    "wrist_rotation",
]

SAFE_LIMITS_LOW = np.array([-1.50, -0.60, 0.15, -2.80, -1.20, -1.35], dtype=np.float32)
SAFE_LIMITS_HIGH = np.array([1.50, 0.75, 1.75, 2.80, 1.20, 2.80], dtype=np.float32)
HOME_POSITION = np.array([0.0, 0.15, 0.90, 0.0, 0.30, 0.0], dtype=np.float32)


def solve_box_qp_6d(
    H: np.ndarray,
    g: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: int = 12,
) -> np.ndarray:
    """
    Exact fast primal active-set solver for a 6-DOF strictly convex box-constrained QP:
        min_x  0.5 * x^T H x - g^T x    s.t.    lb <= x <= ub
    Returns immediately in 1 solve when unconstrained optimum is feasible (95%+ of steps).
    """
    lb = np.minimum(lb, ub)
    x = np.linalg.solve(H, g)
    if np.all((x >= lb) & (x <= ub)):
        return x

    x = np.clip(x, lb, ub)
    for _ in range(max_iter):
        grad = H @ x - g
        low_act = (x <= lb + 1e-9) & (grad > 0.0)
        hi_act = (x >= ub - 1e-9) & (grad < 0.0)
        free = ~(low_act | hi_act)
        if not np.any(free):
            break

        idx_f = np.flatnonzero(free)
        idx_a = np.flatnonzero(~free)
        rhs = g[idx_f]
        if idx_a.size > 0:
            rhs = rhs - H[np.ix_(idx_f, idx_a)] @ x[idx_a]

        x_f_new = np.linalg.solve(H[np.ix_(idx_f, idx_f)], rhs)
        dx_f = x_f_new - x[idx_f]

        # Step length to first blocking bound along feasible direction
        alpha = 1.0
        for i_local, i_glob in enumerate(idx_f):
            d = dx_f[i_local]
            if d > 1e-12:
                alpha = min(alpha, (ub[i_glob] - x[i_glob]) / d)
            elif d < -1e-12:
                alpha = min(alpha, (lb[i_glob] - x[i_glob]) / d)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        x[idx_f] = np.clip(x[idx_f] + alpha * dx_f, lb[idx_f], ub[idx_f])
        if alpha >= 1.0 - 1e-9:
            # Check KKT optimality on active bounds
            grad_post = H @ x - g
            if np.all(grad_post[low_act] >= -1e-6) and np.all(grad_post[hi_act] <= 1e-6):
                break

    return np.clip(x, lb, ub)


def rotvec_to_mat(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues formula: axis-angle vector (3,) -> 3x3 rotation matrix."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = rotvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def orientation_error_world(R_target: np.ndarray, R_ee: np.ndarray) -> np.ndarray:
    """
    Exact SO(3) orientation error vector in world frame (rad), pointing from R_ee to R_target.
    """
    R_err = R_target @ R_ee.T
    cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    v = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ], dtype=np.float64)
    sin_theta = float(np.linalg.norm(v))
    if sin_theta < 1e-7:
        return v.astype(np.float32)
    return ((theta / sin_theta) * v).astype(np.float32)


class AsymmetricStribeckFriction:
    """Smooth directional Asymmetric Stribeck friction model loaded from SysID JSON."""

    def __init__(self, sysid_path: str = SYSID_JSON, v_eps: float = 0.01):
        if not os.path.exists(sysid_path):
            raise FileNotFoundError(f"SysID parameters not found at {sysid_path}")
        with open(sysid_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        asym = params.get("asymmetric_stribeck_friction")
        if asym is None:
            raise KeyError(f"'asymmetric_stribeck_friction' not found in {sysid_path}")

        self.v_eps = float(v_eps)
        self.nom_fc_pos = np.array([asym[j]["fc_pos"] for j in JOINT_NAMES], dtype=np.float64)
        self.nom_fc_neg = np.array([asym[j]["fc_neg"] for j in JOINT_NAMES], dtype=np.float64)
        self.nom_fv_pos = np.array([asym[j]["fv_pos"] for j in JOINT_NAMES], dtype=np.float64)
        self.nom_fv_neg = np.array([asym[j]["fv_neg"] for j in JOINT_NAMES], dtype=np.float64)
        self.nom_delta_fs = np.array([asym[j]["delta_fs"] for j in JOINT_NAMES], dtype=np.float64)
        self.nom_tau_offset = np.array([asym[j]["tau_offset"] for j in JOINT_NAMES], dtype=np.float64)
        self.v_stribeck = np.array([asym[j]["v_stribeck"] for j in JOINT_NAMES], dtype=np.float64)
        self.reset_nominal()

    def reset_nominal(self):
        self.fc_pos = self.nom_fc_pos.copy()
        self.fc_neg = self.nom_fc_neg.copy()
        self.fv_pos = self.nom_fv_pos.copy()
        self.fv_neg = self.nom_fv_neg.copy()
        self.delta_fs = self.nom_delta_fs.copy()
        self.tau_offset = self.nom_tau_offset.copy()

    def randomize(self, rng: np.random.Generator, scale_low: float = 0.90, scale_high: float = 1.10):
        fc_scale = rng.uniform(scale_low, scale_high, size=6)
        fv_scale = rng.uniform(scale_low, scale_high, size=6)
        fs_scale = rng.uniform(scale_low, scale_high, size=6)
        self.fc_pos = self.nom_fc_pos * fc_scale
        self.fc_neg = self.nom_fc_neg * fc_scale
        self.fv_pos = self.nom_fv_pos * fv_scale
        self.fv_neg = self.nom_fv_neg * fv_scale
        self.delta_fs = self.nom_delta_fs * fs_scale
        self.tau_offset = self.nom_tau_offset.copy()

    def compute_torque(self, dq: np.ndarray) -> np.ndarray:
        s = np.tanh(dq / self.v_eps)
        w_pos = 0.5 * (1.0 + s)
        w_neg = 1.0 - w_pos
        fc_eff = w_pos * self.fc_pos + w_neg * self.fc_neg
        fv_eff = w_pos * self.fv_pos + w_neg * self.fv_neg
        strib = self.delta_fs * s * np.exp(-((dq / self.v_stribeck) ** 2))
        return self.tau_offset + fc_eff * s + fv_eff * dq + strib


class DynaArmReachEnv(gym.Env):
    """
    Gymnasium environment for no-load 6D Cartesian pose tracking with:
      - 500 Hz nominal 6D OSC + inverse dynamics inner loop (no ZOH chatter)
      - 50 Hz smoothed RL 6D acceleration residual
      - Asymmetric Stribeck friction on the simulated plant
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        sysid_path: str = SYSID_JSON,
        max_episode_steps: int = 200,
        control_freq_hz: float = 50.0,
        domain_randomization: bool = True,
        dynamic_targets: bool = True,
        use_stribeck_friction: bool = True,
        action_scale: float = 3.0,
        action_scale_ori: float = 5.0,
        action_lpf_alpha: float = 0.5,
        simulate_latency: bool = True,
    ):
        super().__init__()
        self.xml_path = xml_path
        self.sysid_path = sysid_path
        self.max_steps = max_episode_steps
        self.dt_ctrl = 1.0 / control_freq_hz
        self.dr_enabled = domain_randomization
        self.dynamic_targets = dynamic_targets
        self.use_stribeck_friction = use_stribeck_friction
        self.action_scale = float(action_scale)
        self.action_scale_ori = float(action_scale_ori)
        self.action_lpf_alpha = float(np.clip(action_lpf_alpha, 0.0, 0.95))
        self.simulate_latency = bool(simulate_latency)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo model not found at {xml_path}")

        # 1. Simulated plant model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 2. Separate nominal controller model (never mutated by DR -> zero oracle leakage)
        self.nom_model = mujoco.MjModel.from_xml_path(xml_path)
        self.nom_data = mujoco.MjData(self.nom_model)

        self.dt_sim = self.model.opt.timestep
        self.substeps = max(1, int(round(self.dt_ctrl / self.dt_sim)))

        self.qpos_idx = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.qvel_idx = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self.act_idx = np.array([self.model.actuator(f"dynaarm_tau_{i+1}").id for i in range(6)])
        self.ee_site_id = self.model.site("ee_site").id

        # Strictly no external payload mass
        try:
            payload_id = self.model.body("payload_body").id
            self.model.body_mass[payload_id] = 1e-4
            self.nom_model.body_mass[payload_id] = 1e-4
        except KeyError:
            pass

        if self.use_stribeck_friction:
            self.stribeck = AsymmetricStribeckFriction(sysid_path=sysid_path)
            self.model.dof_frictionloss[self.qvel_idx] = 0.0
            self.model.dof_damping[self.qvel_idx] = 0.0
        else:
            self.stribeck = None

        self.nom_body_masses = self.model.body_mass.copy()
        self.nom_friction = self.model.dof_frictionloss.copy()
        self.nom_damping = self.model.dof_damping.copy()

        # Pre-allocated Jacobian & limit buffers
        self._jac_pos = np.zeros((3, self.nom_model.nv), dtype=np.float64)
        self._jac_rot = np.zeros((3, self.nom_model.nv), dtype=np.float64)
        self._max_joint_acc = np.array([25.0, 25.0, 25.0, 35.0, 35.0, 35.0], dtype=np.float64)
        self._max_tau = np.array([120.0, 120.0, 120.0, 40.0, 40.0, 40.0], dtype=np.float32)
        self._dls_eye6 = (0.03 ** 2) * np.eye(6, dtype=np.float64)

        # Action Space (dim=6): [a_res_pos (3), a_res_ori (3)] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation Space (dim=39):
        # q(6), dq(6), p_ee(3), p_tgt(3), err_p(3), v_tgt(3), a_tgt(3), err_ori(3), w_tgt(3), prev_action(6)
        obs_dim = 6 + 6 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.prev_raw_action = np.zeros(6, dtype=np.float32)
        self.p_target = np.zeros(3, dtype=np.float32)
        self.v_target = np.zeros(3, dtype=np.float32)
        self.a_target = np.zeros(3, dtype=np.float32)
        self.R_target = np.eye(3, dtype=np.float64)
        self.w_target = np.zeros(3, dtype=np.float32)
        self.alpha_target = np.zeros(3, dtype=np.float32)

        self.target_center = np.zeros(3, dtype=np.float32)
        self.target_phase = 0.0
        self.target_omega = 0.0
        self.rx = self.ry = self.rz = 0.0
        self.R_center = np.eye(3, dtype=np.float64)
        self.ori_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.ori_amp = 0.0
        self.ori_omega = 0.0
        self.ori_phase = 0.0

        self._ep_dists_mm: list[float] = []
        self._ep_ori_deg: list[float] = []

        # RL residual transport latency buffer (in 500 Hz substeps: 4-16 ms)
        self._max_delay_substeps = 10
        self._rl_delay_substeps = 4 if self.simulate_latency else 0
        self._res_queue = np.zeros((self._max_delay_substeps + 1, 6), dtype=np.float64)

    def _sample_target(self, p_ee_init: np.ndarray, R_ee_init: np.ndarray):
        """
        Samples a reachable 6D pose trajectory anchored near (p_ee_init, R_ee_init)
        so episodes focus on smooth continuous 6D tracking.
        """
        rng = self.np_random
        if self.dynamic_targets:
            self.target_omega = float(rng.uniform(0.8, 2.8))
            self.rx = float(rng.uniform(0.03, 0.09))
            self.ry = float(rng.uniform(0.05, 0.14))
            self.rz = float(rng.uniform(0.03, 0.08))
            self.ori_omega = float(rng.uniform(0.6, 2.0))
            self.ori_amp = float(rng.uniform(0.08, 0.25))
            axis = rng.normal(size=3)
            self.ori_axis = axis / (np.linalg.norm(axis) + 1e-8)
        else:
            self.target_omega = 0.0
            self.rx = self.ry = self.rz = 0.0
            self.ori_omega = 0.0
            self.ori_amp = 0.0

        self.target_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        self.ori_phase = float(rng.uniform(0.0, 2.0 * np.pi))

        offset_0 = np.array([
            self.rx * np.cos(self.target_phase),
            self.ry * np.sin(self.target_phase),
            self.rz * np.sin(2.0 * self.target_phase),
        ], dtype=np.float32)
        jitter = rng.uniform(-0.015, 0.015, size=3).astype(np.float32)
        self.target_center = np.clip(
            p_ee_init - offset_0 + jitter,
            np.array([0.65, -0.18, 1.25], dtype=np.float32),
            np.array([0.96, 0.18, 1.56], dtype=np.float32),
        )

        # Anchor R_center so R_target(0) starts smoothly at R_ee_init
        ang0 = self.ori_amp * np.sin(self.ori_phase)
        self.R_center = rotvec_to_mat(-ang0 * self.ori_axis) @ R_ee_init
        self._update_target(0.0)

    def _update_target(self, t: float):
        """Updates smooth 6D pose target (position, orientation, velocities, accelerations)."""
        if self.dynamic_targets and self.target_omega > 0.0:
            theta = self.target_omega * t + self.target_phase
            w = self.target_omega
            w2 = w * w
            self.p_target = self.target_center + np.array([
                self.rx * np.cos(theta),
                self.ry * np.sin(theta),
                self.rz * np.sin(2.0 * theta),
            ], dtype=np.float32)
            self.v_target = np.array([
                -self.rx * w * np.sin(theta),
                self.ry * w * np.cos(theta),
                2.0 * self.rz * w * np.cos(2.0 * theta),
            ], dtype=np.float32)
            self.a_target = np.array([
                -self.rx * w2 * np.cos(theta),
                -self.ry * w2 * np.sin(theta),
                -4.0 * self.rz * w2 * np.sin(2.0 * theta),
            ], dtype=np.float32)

            phi = self.ori_omega * t + self.ori_phase
            ang = self.ori_amp * np.sin(phi)
            dang = self.ori_amp * self.ori_omega * np.cos(phi)
            ddang = -self.ori_amp * (self.ori_omega ** 2) * np.sin(phi)
            self.R_target = rotvec_to_mat(ang * self.ori_axis) @ self.R_center
            self.w_target = (dang * self.ori_axis).astype(np.float32)
            self.alpha_target = (ddang * self.ori_axis).astype(np.float32)
        else:
            self.p_target = self.target_center.copy()
            self.v_target = np.zeros(3, dtype=np.float32)
            self.a_target = np.zeros(3, dtype=np.float32)
            self.R_target = self.R_center.copy()
            self.w_target = np.zeros(3, dtype=np.float32)
            self.alpha_target = np.zeros(3, dtype=np.float32)

    def _apply_domain_randomization(self):
        rng = self.np_random
        mass_scales = rng.uniform(0.95, 1.05, size=len(self.nom_body_masses))
        self.model.body_mass[:] = self.nom_body_masses * mass_scales

        if self.use_stribeck_friction and self.stribeck is not None:
            self.stribeck.randomize(rng, scale_low=0.90, scale_high=1.10)
        else:
            fric_scales = rng.uniform(0.90, 1.10, size=len(self.nom_friction))
            damp_scales = rng.uniform(0.90, 1.10, size=len(self.nom_damping))
            self.model.dof_frictionloss[:] = self.nom_friction * fric_scales
            self.model.dof_damping[:] = self.nom_damping * damp_scales

    def reset_to_state(
        self,
        q_init: np.ndarray,
        dq_init: np.ndarray,
        p_target: np.ndarray | None = None,
        v_target: np.ndarray | None = None,
        a_target: np.ndarray | None = None,
        R_target: np.ndarray | None = None,
        w_target: np.ndarray | None = None,
        alpha_target: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reset plant to an explicit joint state and 6D target (used for held-out eval)."""
        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.prev_raw_action = np.zeros(6, dtype=np.float32)
        self._ep_dists_mm = []
        self._ep_ori_deg = []
        self._res_queue.fill(0.0)
        self._rl_delay_substeps = 4 if self.simulate_latency else 0

        self.model.body_mass[:] = self.nom_body_masses
        self.model.dof_frictionloss[:] = self.nom_friction
        self.model.dof_damping[:] = self.nom_damping
        if self.stribeck is not None:
            self.stribeck.reset_nominal()

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.qpos[self.qpos_idx] = q_init
        self.data.qvel[self.qvel_idx] = dq_init
        mujoco.mj_forward(self.model, self.data)

        self.p_target = (
            np.asarray(p_target, dtype=np.float32).copy()
            if p_target is not None
            else self.data.site_xpos[self.ee_site_id].copy().astype(np.float32)
        )
        self.v_target = np.zeros(3, dtype=np.float32) if v_target is None else np.asarray(v_target, dtype=np.float32).copy()
        self.a_target = np.zeros(3, dtype=np.float32) if a_target is None else np.asarray(a_target, dtype=np.float32).copy()
        self.R_target = (
            np.asarray(R_target, dtype=np.float64).copy()
            if R_target is not None
            else self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        )
        self.w_target = np.zeros(3, dtype=np.float32) if w_target is None else np.asarray(w_target, dtype=np.float32).copy()
        self.alpha_target = (
            np.zeros(3, dtype=np.float32)
            if alpha_target is None
            else np.asarray(alpha_target, dtype=np.float32).copy()
        )
        return self._get_obs()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.prev_raw_action = np.zeros(6, dtype=np.float32)
        self._ep_dists_mm = []
        self._ep_ori_deg = []
        self._res_queue.fill(0.0)
        self._rl_delay_substeps = 4 if self.simulate_latency else 0

        if self.dr_enabled:
            self._apply_domain_randomization()
            self._rl_delay_substeps = int(self.np_random.integers(2, 9)) if self.simulate_latency else 0
        else:
            self.model.body_mass[:] = self.nom_body_masses
            self.model.dof_frictionloss[:] = self.nom_friction
            self.model.dof_damping[:] = self.nom_damping
            if self.stribeck is not None:
                self.stribeck.reset_nominal()
            self._rl_delay_substeps = 4 if self.simulate_latency else 0
        self._res_queue.fill(0.0)

        rng = self.np_random
        q_init = HOME_POSITION + rng.uniform(-0.22, 0.22, size=6).astype(np.float32)
        q_init = np.clip(q_init, SAFE_LIMITS_LOW + 0.08, SAFE_LIMITS_HIGH - 0.08)
        dq_init = rng.uniform(-0.02, 0.02, size=6).astype(np.float32)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.qpos[self.qpos_idx] = q_init
        self.data.qvel[self.qvel_idx] = dq_init
        mujoco.mj_forward(self.model, self.data)

        p_ee_init = self.data.site_xpos[self.ee_site_id].copy().astype(np.float32)
        R_ee_init = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        self._sample_target(p_ee_init, R_ee_init)
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_idx].copy().astype(np.float32)
        dq = self.data.qvel[self.qvel_idx].copy().astype(np.float32)
        # Inject realistic encoder & velocity noise during domain-randomized training
        if self.dr_enabled:
            q = q + self.np_random.normal(0.0, 0.0005, size=6).astype(np.float32)
            dq = dq + self.np_random.normal(0.0, 0.02, size=6).astype(np.float32)
        p_ee = self.data.site_xpos[self.ee_site_id].copy().astype(np.float32)
        R_ee = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        err_p = (self.p_target - p_ee).astype(np.float32)
        err_ori = orientation_error_world(self.R_target, R_ee)

        return np.concatenate([
            q,
            dq,
            p_ee,
            self.p_target,
            err_p,
            self.v_target,
            self.a_target,
            err_ori,
            self.w_target,
            self.prev_action,
        ]).astype(np.float32)

    def compute_control_torque_6d(
        self,
        q_now: np.ndarray,
        dq_now: np.ndarray,
        p_ee: np.ndarray,
        R_ee: np.ndarray,
        a_res_pos: np.ndarray,
        a_res_ori: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        500 Hz nominal 6D Pose OSC + inverse dynamics feedforward + smoothed RL residual.
        Runs exclusively on `self.nom_model` / `self.nom_data` (zero oracle DR leakage).
        """
        err_p = (self.p_target - p_ee).astype(np.float64)
        err_ori = orientation_error_world(self.R_target, R_ee).astype(np.float64)

        self.nom_data.qpos[:] = 0.0
        self.nom_data.qvel[:] = 0.0
        self.nom_data.qacc[:] = 0.0
        self.nom_data.qpos[self.qpos_idx] = q_now
        self.nom_data.qvel[self.qvel_idx] = dq_now
        mujoco.mj_forward(self.nom_model, self.nom_data)

        self._jac_pos.fill(0.0)
        self._jac_rot.fill(0.0)
        mujoco.mj_jacSite(self.nom_model, self.nom_data, self._jac_pos, self._jac_rot, self.ee_site_id)
        J_pos = self._jac_pos[:, self.qvel_idx]  # (3, 6)
        J_rot = self._jac_rot[:, self.qvel_idx]  # (3, 6)
        J6 = np.vstack([J_pos, J_rot])           # (6, 6)

        v_cart = J_pos @ dq_now
        w_cart = J_rot @ dq_now

        a_ff = np.clip(self.a_target, -15.0, 15.0)
        alpha_ff = np.clip(self.alpha_target, -20.0, 20.0)

        a_lin = np.clip(a_ff + 180.0 * err_p + 28.0 * (self.v_target - v_cart) + a_res_pos, -25.0, 25.0)
        a_ang = np.clip(alpha_ff + 150.0 * err_ori + 24.0 * (self.w_target - w_cart) + a_res_ori, -35.0, 35.0)
        a_6d = np.concatenate([a_lin, a_ang])

        # Singularity-robust 6-DOF Box-Constrained QP-OSC with 2nd-Order CBF Joint Limits
        # Prioritizes 3D position (w_pos=1.0) over orientation (w_ori=0.25) whenever a joint
        # approaches SAFE_LIMITS, eliminating joint-limit transients smoothly.
        min_sv = float(np.linalg.svd(J6, compute_uv=False)[-1])
        lam = 0.025 + 0.12 * float(np.exp(-((min_sv / 0.035) ** 2)))
        lam2 = lam * lam
        ddq_reg = 8.0 * (HOME_POSITION - q_now) - 4.0 * dq_now

        w_pos = 1.0
        w_ori = 0.25
        H = w_pos * (J_pos.T @ J_pos) + w_ori * (J_rot.T @ J_rot) + lam2 * np.eye(6)
        g = w_pos * (J_pos.T @ a_lin) + w_ori * (J_rot.T @ a_ang) + lam2 * ddq_reg

        # 2nd-order Exponential CBF box bounds (with 0.05 rad inner safety cushion)
        cbf_alpha = 12.0
        q_lo = SAFE_LIMITS_LOW + 0.05
        q_hi = SAFE_LIMITS_HIGH - 0.05
        ddq_cbf_min = -2.0 * cbf_alpha * dq_now - (cbf_alpha ** 2) * (q_now - q_lo)
        ddq_cbf_max = -2.0 * cbf_alpha * dq_now + (cbf_alpha ** 2) * (q_hi - q_now)
        lb = np.maximum(-self._max_joint_acc, ddq_cbf_min)
        ub = np.minimum(self._max_joint_acc, ddq_cbf_max)
        ddq = solve_box_qp_6d(H, g, lb, ub)

        self.nom_data.qacc[self.qvel_idx] = ddq
        mujoco.mj_inverse(self.nom_model, self.nom_data)
        tau_cmd = self.nom_data.qfrc_inverse[self.qvel_idx].copy().astype(np.float32)
        tau_cmd = np.clip(tau_cmd, -self._max_tau, self._max_tau)

        # Soft proximity margin (0.15 rad) reported for RL reward awareness (no torque spring)
        dist_to_low = np.maximum(0.0, (SAFE_LIMITS_LOW + 0.15) - q_now)
        dist_to_high = np.maximum(0.0, q_now - (SAFE_LIMITS_HIGH - 0.15))
        return tau_cmd, dist_to_low, dist_to_high

    def step(
        self,
        action: np.ndarray,
        external_target: dict | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        raw_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # In-env EMA action smoothing (identical in training and deployment)
        alpha = self.action_lpf_alpha
        smooth_action = (alpha * self.prev_action + (1.0 - alpha) * raw_action).astype(np.float32)

        self.step_count += 1
        if external_target is not None:
            self.p_target = np.asarray(external_target["p"], dtype=np.float32)
            self.v_target = np.asarray(external_target["v"], dtype=np.float32)
            self.a_target = np.asarray(external_target.get("a", np.zeros(3)), dtype=np.float32)
            self.R_target = np.asarray(external_target["R"], dtype=np.float64)
            self.w_target = np.asarray(external_target.get("w", np.zeros(3)), dtype=np.float32)
            self.alpha_target = np.asarray(external_target.get("alpha", np.zeros(3)), dtype=np.float32)
        else:
            t_sim = self.step_count * self.dt_ctrl
            self._update_target(t_sim)

        a_res_new = np.concatenate([
            (smooth_action[:3] * self.action_scale).astype(np.float64),
            (smooth_action[3:] * self.action_scale_ori).astype(np.float64),
        ])

        # 500 Hz inner loop: recompute nominal 6D QP-OSC + Stribeck friction every 2 ms substep
        dist_to_low = np.zeros(6, dtype=np.float32)
        dist_to_high = np.zeros(6, dtype=np.float32)
        delay = self._rl_delay_substeps
        for _ in range(self.substeps):
            # Shift RL residual delay queue (models 4-16 ms policy transport latency)
            if delay > 0:
                self._res_queue[1 : delay + 1] = self._res_queue[0:delay]
                self._res_queue[0] = a_res_new
                a_res_delayed = self._res_queue[delay]
            else:
                a_res_delayed = a_res_new

            q_sub = self.data.qpos[self.qpos_idx]
            dq_sub = self.data.qvel[self.qvel_idx]
            p_sub = self.data.site_xpos[self.ee_site_id]
            R_sub = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

            tau_cmd, dist_to_low, dist_to_high = self.compute_control_torque_6d(
                q_sub, dq_sub, p_sub, R_sub, a_res_delayed[:3], a_res_delayed[3:]
            )
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self.act_idx] = tau_cmd

            if self.use_stribeck_friction and self.stribeck is not None:
                tau_fric = self.stribeck.compute_torque(dq_sub)
                self.data.qfrc_applied[self.qvel_idx] = -tau_fric

            mujoco.mj_step(self.model, self.data)

        self.data.qfrc_applied[self.qvel_idx] = 0.0

        # Post-step 6D pose tracking error
        p_ee_post = self.data.site_xpos[self.ee_site_id].copy()
        R_ee_post = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        dist = float(np.linalg.norm(self.p_target - p_ee_post))
        dist_mm = dist * 1000.0
        err_ori_rad = float(np.linalg.norm(orientation_error_world(self.R_target, R_ee_post)))
        err_ori_deg = float(np.degrees(err_ori_rad))

        self._ep_dists_mm.append(dist_mm)
        self._ep_ori_deg.append(err_ori_deg)

        # Smooth 6D pose tracking reward + stronger smoothness penalty
        r_pos = 2.0 * float(np.exp(-15.0 * dist)) + 1.0 * float(np.exp(-60.0 * dist)) - 2.0 * dist
        r_ori = 1.0 * float(np.exp(-4.0 * err_ori_rad)) + 0.5 * float(np.exp(-15.0 * err_ori_rad)) - 0.5 * err_ori_rad

        action_diff = raw_action - self.prev_raw_action
        r_reg = (
            -0.02 * float(np.sum(smooth_action ** 2))
            - 0.15 * float(np.sum(action_diff ** 2))
            - 5.0 * float(np.sum(dist_to_low ** 2) + np.sum(dist_to_high ** 2))
        )
        reward = r_pos + r_ori + r_reg

        self.prev_action = smooth_action
        self.prev_raw_action = raw_action

        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "dist_mm": dist_mm,
            "ori_err_deg": err_ori_deg,
            "ep_mean_dist_mm": float(np.mean(self._ep_dists_mm)),
            "ep_rms_dist_mm": float(np.sqrt(np.mean(np.square(self._ep_dists_mm)))),
            "ep_peak_dist_mm": float(np.max(self._ep_dists_mm)),
            "ep_mean_ori_deg": float(np.mean(self._ep_ori_deg)),
            "success": dist < 0.015 and err_ori_deg < 5.0,
        }
        return obs, float(reward), terminated, truncated, info
