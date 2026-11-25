#!/usr/bin/env python3
"""
========================================================================================
DUATIC DYNAARM PPO TRACKING TRAINER (ROBUST, SQUASHED-TANH, OBS-NORMALIZED)
========================================================================================
Vectorized PPO training pipeline for no-load 3D Cartesian tracking with Asymmetric
Stribeck friction.

Key robustness fixes:
  - Squashed Tanh-Gaussian policy: actions strictly in [-1, 1] with exact Jacobian
    log-prob correction (no action-clipping gradient mismatch).
  - Running observation normalization (`RunningMeanStd`) saved inside checkpoints.
  - Gymnasium 1.x `SAME_STEP` autoreset with exact time-limit truncation bootstrapping
    in GAE (`final_obs`) and working `best_policy.pt` checkpointing.
  - Unclipped critic loss (avoids ±0.2 clipping bottleneck on multi-hundred returns)
    + target-KL early stopping.
"""

import argparse
import os
import random
import signal
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from dynaarm_reach_env import DynaArmReachEnv

RL_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(RL_DIR, "checkpoints")
LOGS_DIR = os.path.join(RL_DIR, "logs", "tensorboard")


class RunningMeanStd:
    """Welford running mean/variance tracker for observation normalization."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        norm = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(norm, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict):
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Squashed Tanh-Gaussian Actor-Critic.
    Pre-squash actions u ~ Normal(mean, std) are mapped via a = tanh(u) in (-1, 1)
    with exact change-of-variables log-probability correction.
    """

    def __init__(self, obs_dim: int = 39, action_dim: int = 6):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        # Start near zero residual with gentle exploration std (~0.22 -> ~2.2 m/s^2 at scale=10)
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -1.5))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(x))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(x)
        logstd = torch.clamp(self.actor_logstd, -4.0, 0.5).expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if action is None:
            u = dist.rsample()
            action = torch.tanh(u)
        else:
            # Invert tanh safely for stored squashed actions
            action_clamped = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            u = 0.5 * (torch.log1p(action_clamped) - torch.log1p(-action_clamped))

        # Exact squashed log_prob: log p(u) - sum log(1 - tanh(u)^2)
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x)
        return action, log_prob, entropy, value


def save_checkpoint(path: str, agent: ActorCritic, obs_rms: RunningMeanStd, config: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "obs_rms": obs_rms.state_dict(),
            "config": config,
        },
        path,
    )


def make_env(
    xml_path: str | None = None,
    dynamic_targets: bool = True,
    dr_enabled: bool = True,
    use_stribeck_friction: bool = True,
    action_scale: float = 3.0,
    action_scale_ori: float = 5.0,
    action_lpf_alpha: float = 0.5,
):
    def thunk():
        kwargs = {
            "domain_randomization": dr_enabled,
            "dynamic_targets": dynamic_targets,
            "use_stribeck_friction": use_stribeck_friction,
            "action_scale": action_scale,
            "action_scale_ori": action_scale_ori,
            "action_lpf_alpha": action_lpf_alpha,
        }
        if xml_path is not None:
            kwargs["xml_path"] = xml_path
        env = DynaArmReachEnv(**kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def train(
    total_timesteps: int = 3_000_000,
    num_envs: int = 12,
    num_steps: int = 1024,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    num_minibatches: int = 8,
    update_epochs: int = 10,
    clip_coef: float = 0.2,
    ent_coef: float = 0.002,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.03,
    xml_path: str | None = None,
    dynamic_targets: bool = True,
    dr_enabled: bool = True,
    use_stribeck_friction: bool = True,
    action_scale: float = 3.0,
    action_scale_ori: float = 5.0,
    action_lpf_alpha: float = 0.5,
    seed: int = 42,
    exp_name: str = "dynaarm_stribeck_tracking",
    dry_run: bool = False,
):
    if dry_run:
        total_timesteps = 4096
        num_envs = 4
        num_steps = 256
        print("[!] Running in DRY-RUN mode for verification.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("      DUATIC DYNAARM PPO TRACKING TRAINER (NO-LOAD, ASYMMETRIC STRIBECK)")
    print(f"  Experiment       : {exp_name}")
    print(f"  Device           : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Total Steps      : {total_timesteps:,} | Parallel Envs: {num_envs} | Rollout: {num_steps}")
    print(f"  Batch Size       : {num_envs * num_steps} | Minibatches: {num_minibatches}")
    print(f"  Stribeck Friction: {use_stribeck_friction} | Domain Rand: {dr_enabled} | Dynamic: {dynamic_targets}")
    print(f"  Action Scale     : {action_scale:.1f} m/s^2 | Seed: {seed}")
    print("=" * 78)

    exp_ckpt_dir = os.path.join(CHECKPOINTS_DIR, exp_name)
    os.makedirs(exp_ckpt_dir, exist_ok=True)
    run_log_dir = os.path.join(LOGS_DIR, f"{exp_name}_{int(time.time())}")
    writer = SummaryWriter(run_log_dir)

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                xml_path=xml_path,
                dynamic_targets=dynamic_targets,
                dr_enabled=dr_enabled,
                use_stribeck_friction=use_stribeck_friction,
                action_scale=action_scale,
                action_scale_ori=action_scale_ori,
                action_lpf_alpha=action_lpf_alpha,
            )
            for _ in range(num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    agent = ActorCritic(obs_dim=obs_dim, action_dim=act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    ckpt_config = {
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "action_scale": action_scale,
        "action_scale_ori": action_scale_ori,
        "action_lpf_alpha": action_lpf_alpha,
        "use_stribeck_friction": use_stribeck_friction,
        "dynamic_targets": dynamic_targets,
        "dr_enabled": dr_enabled,
        "xml_path": xml_path,
        "squashed_tanh": True,
    }

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = max(1, total_timesteps // batch_size)

    obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    global_step = 0
    start_time = time.time()
    raw_next_obs, _ = envs.reset(seed=seed)
    obs_rms.update(raw_next_obs)
    next_obs = torch.as_tensor(obs_rms.normalize(raw_next_obs), dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)

    best_mean_reward = -np.inf

    def handle_signal(sig, frame):
        print(f"\n[!] Caught signal {sig}. Saving interrupted checkpoint...")
        save_path = os.path.join(exp_ckpt_dir, "interrupted_policy.pt")
        save_checkpoint(save_path, agent, obs_rms, ckpt_config)
        envs.close()
        writer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\n[*] Starting PPO Optimization Loop...\n")

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        lr_now = max(1e-5, frac * learning_rate)
        optimizer.param_groups[0]["lr"] = lr_now

        recent_ep_returns: list[float] = []
        recent_ep_dists: list[float] = []

        # ----------------------------------------------------------------------
        # 1. Rollout Phase
        # ----------------------------------------------------------------------
        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            step_action = action.cpu().numpy()
            raw_next_obs, reward, terminations, truncations, infos = envs.step(step_action)
            done = np.logical_or(terminations, truncations)

            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device)

            # Exact time-limit truncation bootstrapping using final_obs (SAME_STEP mode)
            if np.any(truncations) and "final_obs" in infos:
                final_obs_arr = infos["final_obs"]
                for i in range(num_envs):
                    if truncations[i] and not terminations[i] and final_obs_arr[i] is not None:
                        f_obs_norm = obs_rms.normalize(np.asarray(final_obs_arr[i], dtype=np.float32))
                        with torch.no_grad():
                            v_term = agent.get_value(
                                torch.as_tensor(f_obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
                            ).item()
                        reward_t[i] += gamma * v_term

            rewards[step] = reward_t

            # Record completed episodes (Gymnasium 1.x SAME_STEP format)
            if "final_info" in infos:
                finfos = infos["final_info"]
                fmask = infos.get("_final_info", np.ones(num_envs, dtype=bool))
                if isinstance(finfos, dict) and "episode" in finfos:
                    ep_mask = finfos.get("_episode", fmask)
                    for i in range(num_envs):
                        if ep_mask[i]:
                            ep_r = float(finfos["episode"]["r"][i])
                            ep_l = int(finfos["episode"]["l"][i])
                            recent_ep_returns.append(ep_r)
                            writer.add_scalar("charts/episodic_return", ep_r, global_step)
                            writer.add_scalar("charts/episodic_length", ep_l, global_step)
                            if "ep_mean_dist_mm" in finfos:
                                d_mm = float(finfos["ep_mean_dist_mm"][i])
                                recent_ep_dists.append(d_mm)
                                writer.add_scalar("charts/ep_mean_dist_mm", d_mm, global_step)
                elif isinstance(finfos, (list, tuple, np.ndarray)):
                    for i, info in enumerate(finfos):
                        if fmask[i] and info and "episode" in info:
                            ep_r = float(info["episode"]["r"])
                            recent_ep_returns.append(ep_r)
                            writer.add_scalar("charts/episodic_return", ep_r, global_step)
                            writer.add_scalar("charts/episodic_length", int(info["episode"]["l"]), global_step)
                            if "ep_mean_dist_mm" in info:
                                d_mm = float(info["ep_mean_dist_mm"])
                                recent_ep_dists.append(d_mm)
                                writer.add_scalar("charts/ep_mean_dist_mm", d_mm, global_step)
            elif "episode" in infos and "_episode" in infos:
                ep_mask = infos["_episode"]
                for i in range(num_envs):
                    if ep_mask[i]:
                        ep_r = float(infos["episode"]["r"][i])
                        recent_ep_returns.append(ep_r)
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", int(infos["episode"]["l"][i]), global_step)
                        if "ep_mean_dist_mm" in infos:
                            d_mm = float(infos["ep_mean_dist_mm"][i])
                            recent_ep_dists.append(d_mm)
                            writer.add_scalar("charts/ep_mean_dist_mm", d_mm, global_step)

            obs_rms.update(raw_next_obs)
            next_obs = torch.as_tensor(obs_rms.normalize(raw_next_obs), dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

        # ----------------------------------------------------------------------
        # 2. Generalized Advantage Estimation (GAE)
        # ----------------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, act_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # ----------------------------------------------------------------------
        # 3. PPO Optimization Epochs (with target-KL early stopping)
        # ----------------------------------------------------------------------
        b_inds = np.arange(batch_size)
        clipfracs = []
        early_stopped = False
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                if target_kl is not None and approx_kl.item() > 1.5 * target_kl:
                    early_stopped = True
                    break

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Unclipped value loss so critic can freely track return scale
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if early_stopped:
                break

        # Telemetry
        elapsed = max(1e-6, time.time() - start_time)
        sps = int(global_step / elapsed)
        elapsed_min = elapsed / 60.0
        remaining_min = ((total_timesteps - global_step) / max(1, sps)) / 60.0

        mean_ret = float(np.mean(recent_ep_returns)) if recent_ep_returns else float("nan")
        mean_dst = float(np.mean(recent_ep_dists)) if recent_ep_dists else float("nan")

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", lr_now, global_step)

        if update % 5 == 0 or update == 1 or update == num_updates:
            print(
                f"[Update {update:4d}/{num_updates}] Step {global_step:9,d}/{total_timesteps:,d} | "
                f"SPS: {sps:4d} | Ret: {mean_ret:6.1f} | Mean Err: {mean_dst:5.2f} mm | "
                f"KL: {approx_kl.item():.4f} | Elapsed: {elapsed_min:4.1f}m | ETA: {remaining_min:4.1f}m"
            )

        # Save best policy
        min_steps_for_best = 0 if dry_run else 50_000
        if recent_ep_returns and mean_ret > best_mean_reward and global_step >= min_steps_for_best:
            best_mean_reward = mean_ret
            best_path = os.path.join(exp_ckpt_dir, "best_policy.pt")
            save_checkpoint(best_path, agent, obs_rms, ckpt_config)
            save_checkpoint(os.path.join(CHECKPOINTS_DIR, "best_policy.pt"), agent, obs_rms, ckpt_config)
            print(f"  [★ NEW BEST] Ret: {mean_ret:.1f} | Mean Err: {mean_dst:.2f} mm -> {best_path}")

        if not dry_run and global_step % 500_000 < batch_size:
            ckpt_path = os.path.join(exp_ckpt_dir, f"policy_step_{global_step // 1000}k.pt")
            save_checkpoint(ckpt_path, agent, obs_rms, ckpt_config)

    final_path = os.path.join(exp_ckpt_dir, "final_policy.pt")
    save_checkpoint(final_path, agent, obs_rms, ckpt_config)
    save_checkpoint(os.path.join(CHECKPOINTS_DIR, "final_policy.pt"), agent, obs_rms, ckpt_config)
    print(f"\n[✓] Training complete! Final policy saved to: {final_path}")

    envs.close()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Duatic DynaArm PPO Tracking Trainer (No-Load, Stribeck)")
    parser.add_argument("--total-steps", type=int, default=3_000_000, help="Total timesteps")
    parser.add_argument("--num-envs", type=int, default=12, help="Parallel environments")
    parser.add_argument("--num-steps", type=int, default=1024, help="Rollout steps per env")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--action-scale", type=float, default=3.0, help="Linear residual acceleration scale [m/s^2]")
    parser.add_argument("--action-scale-ori", type=float, default=5.0, help="Angular residual acceleration scale [rad/s^2]")
    parser.add_argument("--action-lpf-alpha", type=float, default=0.5, help="In-env EMA action smoothing alpha")
    parser.add_argument("--xml-path", type=str, default=None, help="Path to MuJoCo XML model")
    parser.add_argument("--no-dr", action="store_true", help="Disable domain randomization")
    parser.add_argument("--no-stribeck", action="store_true", help="Use symmetric XML friction instead of Stribeck")
    parser.add_argument("--static-targets", action="store_true", help="Use static setpoints instead of dynamic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run 4k steps to verify pipeline")
    parser.add_argument("--exp-name", type=str, default="dynaarm_stribeck_tracking", help="Experiment name")
    args = parser.parse_args()

    train(
        total_timesteps=args.total_steps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        action_scale=args.action_scale,
        action_scale_ori=args.action_scale_ori,
        action_lpf_alpha=args.action_lpf_alpha,
        xml_path=args.xml_path,
        dr_enabled=not args.no_dr,
        use_stribeck_friction=not args.no_stribeck,
        dynamic_targets=not args.static_targets,
        seed=args.seed,
        dry_run=args.dry_run,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    main()
