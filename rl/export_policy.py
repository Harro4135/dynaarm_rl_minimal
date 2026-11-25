#!/usr/bin/env python3
"""
========================================================================================
EXPORT TRAINED DYNAARM 6D POSE RL POLICY TO STANDALONE TORCHSCRIPT (+ OPTIONAL ONNX)
========================================================================================
Bakes observation normalization (`obs_rms.mean`, `obs_rms.var`) + `actor_mean` + `tanh`
into a single self-contained deployment module so training and real-robot inference can
never diverge on observation scaling.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

RL_DIR = os.path.dirname(os.path.abspath(__file__))
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

from evaluate_policy import load_policy_checkpoint

DEFAULT_CKPT = os.path.join(RL_DIR, "checkpoints", "stribeck_pose6d_qp", "best_policy.pt")


class DeployableActor(nn.Module):
    """Self-contained deterministic actor with baked-in observation normalization."""

    def __init__(
        self,
        actor_mean: nn.Module,
        obs_mean: np.ndarray,
        obs_var: np.ndarray,
        obs_clip: float = 10.0,
        squashed_tanh: bool = True,
    ):
        super().__init__()
        self.actor_mean = actor_mean
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_inv_std", torch.as_tensor(1.0 / np.sqrt(obs_var + 1e-8), dtype=torch.float32))
        self.obs_clip = float(obs_clip)
        self.squashed_tanh = bool(squashed_tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) * self.obs_inv_std
        x = torch.clamp(x, -self.obs_clip, self.obs_clip)
        out = self.actor_mean(x)
        if self.squashed_tanh:
            return torch.tanh(out)
        return torch.clamp(out, -1.0, 1.0)


def export_policy(checkpoint_path: str = DEFAULT_CKPT, out_dir: str | None = None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = raw.get("config", {}) if isinstance(raw, dict) else {}
    obs_dim = int(cfg.get("obs_dim", 39))
    act_dim = int(cfg.get("action_dim", 6))
    squashed_tanh = bool(cfg.get("squashed_tanh", True))

    agent, obs_rms, _ = load_policy_checkpoint(checkpoint_path, obs_dim, act_dim, device)
    obs_mean = obs_rms.mean if obs_rms is not None else np.zeros(obs_dim, dtype=np.float64)
    obs_var = obs_rms.var if obs_rms is not None else np.ones(obs_dim, dtype=np.float64)

    deploy_mod = DeployableActor(
        actor_mean=agent.actor_mean,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_clip=10.0,
        squashed_tanh=squashed_tanh,
    )
    deploy_mod.eval()

    out_dir = out_dir or os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(out_dir, exist_ok=True)
    ts_path = os.path.join(out_dir, "policy_deploy.ts")
    meta_path = os.path.join(out_dir, "policy_deploy_meta.json")

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    scripted = torch.jit.trace(deploy_mod, dummy_obs)
    scripted.save(ts_path)

    # Verify numerical equivalence on random inputs
    loaded = torch.jit.load(ts_path, map_location="cpu")
    test_obs = torch.randn(16, obs_dim, dtype=torch.float32) * 0.5
    with torch.no_grad():
        ref_out = deploy_mod(test_obs)
        ts_out = loaded(test_obs)
    max_diff = float(torch.max(torch.abs(ref_out - ts_out)).item())
    assert max_diff < 1e-6, f"TorchScript mismatch: {max_diff}"

    meta = {
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "torchscript_path": os.path.abspath(ts_path),
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "action_scale": float(cfg.get("action_scale", 3.0)),
        "action_scale_ori": float(cfg.get("action_scale_ori", 5.0)),
        "action_lpf_alpha": float(cfg.get("action_lpf_alpha", 0.5)),
        "control_freq_hz": 50.0,
        "squashed_tanh": squashed_tanh,
        "max_export_diff": max_diff,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[✓] Exported TorchScript policy : {ts_path} (max_diff={max_diff:.2e})")
    print(f"[✓] Saved deployment metadata   : {meta_path}")

    # Optional ONNX export if onnx is available
    try:
        import onnx  # noqa: F401
        onnx_path = os.path.join(out_dir, "policy_deploy.onnx")
        torch.onnx.export(
            deploy_mod,
            dummy_obs,
            onnx_path,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
        print(f"[✓] Exported ONNX policy        : {onnx_path}")
    except Exception:
        print("[i] ONNX package not installed; TorchScript (.ts) export is ready for deployment.")

    return ts_path, meta_path


def main():
    parser = argparse.ArgumentParser(description="Export DynaArm 6D Pose RL Policy for Deployment")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    export_policy(checkpoint_path=args.checkpoint, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
