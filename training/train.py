"""
Training script for RoboCup Humanoid Soccer RL.

Phase 1: Single-robot skills (stand → walk → dribble → shoot)
Phase 2: Multi-robot match with self-play

Uses Genesis for training, logs to Weights & Biases.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (CHECKPOINTS_DIR, LOGS_DIR, VIDEOS_DIR,
                            Phase1Config, Phase2Config, ProjectConfig)


def setup_logger(config: ProjectConfig, phase: str, run_name: str = None,
                 use_wandb: bool = False):
    """Initialise the unified TensorBoard + (optional) W&B logger.

    TensorBoard is always on if installed; W&B requires `--wandb` AND the
    package to be installed. Returns a TrainingLogger instance — the
    trainer calls .log_scalar / .log_video on it.
    """
    from training.logger import TrainingLogger

    return TrainingLogger(
        run_name=run_name or f"{phase}_{time.strftime('%Y%m%d_%H%M%S')}",
        log_root=str(LOGS_DIR),
        use_wandb=use_wandb,
        wandb_project=config.wandb.project,
        wandb_entity=config.wandb.entity,
        wandb_tags=config.wandb.tags + [phase],
        config={
            "phase": phase,
            "seed": config.seed,
            "robot": config.robot.__dict__,
            "phase1": (config.phase1.__dict__ if phase == "phase1"
                       else config.phase2.__dict__),
        },
    )


def log_metrics(logger, metrics: dict, step: int, video_path: str = None):
    """Push a metrics dict (and optional video file) to the logger."""
    if logger is None:
        return
    prefixed = {(k if "/" in k else f"train/{k}"): v
                for k, v in metrics.items()}
    logger.log_scalars(prefixed, step)
    if video_path and os.path.exists(video_path):
        try:
            import imageio.v2 as imageio
            frames = imageio.mimread(video_path)
            logger.log_video("train/rollout", frames, step=step, fps=30)
        except Exception as e:
            print(f"[train] failed to attach video {video_path}: {e}")


def save_checkpoint(
    policy, optimizer, step: int, phase: str, stage: str = None, path: str = None,
    obs_norm=None, ret_norm=None,
):
    """Save model checkpoint.

    obs_norm / ret_norm are the optional running-stat normalizers — saved
    so that eval (and Phase 2 fine-tuning) feed inputs with the same
    scaling the policy was trained on. Skipping this is the #1 cause of
    "checkpoint that worked in training does nothing in eval".
    """
    try:
        import torch

        if path is None:
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            suffix = f"_{stage}" if stage else ""
            path = os.path.join(CHECKPOINTS_DIR, f"{phase}{suffix}_step{step}.pt")
        checkpoint = {
            "step": step,
            "phase": phase,
            "stage": stage,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if obs_norm is not None and hasattr(obs_norm, "state_dict"):
            checkpoint["obs_norm"] = obs_norm.state_dict()
        if ret_norm is not None and hasattr(ret_norm, "state_dict"):
            checkpoint["ret_norm"] = ret_norm.state_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        return path
    except ImportError:
        print("PyTorch not available for checkpoint saving")
        return None


def load_checkpoint(path: str, policy, optimizer=None):
    """Load model checkpoint."""
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        policy.load_state_dict(checkpoint["policy_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint: {path} (step {checkpoint.get('step', '?')})")
        return checkpoint
    except ImportError:
        print("PyTorch not available")
        return None


# ═══════════════════════════════════════════════════════════════════
#  PPO Policy Network
# ═══════════════════════════════════════════════════════════════════


def create_policy(obs_dim: int, act_dim: int, hidden_dims=(512, 256, 128)):
    """Create actor-critic policy network."""
    try:
        import torch
        import torch.nn as nn

        class ActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                # Shared feature extractor
                layers = []
                in_dim = obs_dim
                for h_dim in hidden_dims:
                    layers.extend(
                        [
                            nn.Linear(in_dim, h_dim),
                            nn.LayerNorm(h_dim),
                            nn.ELU(),
                        ]
                    )
                    in_dim = h_dim
                self.features = nn.Sequential(*layers)

                # Actor head (mean + log_std)
                self.actor_mean = nn.Linear(hidden_dims[-1], act_dim)
                self.actor_log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

                # Critic head
                self.critic = nn.Sequential(
                    nn.Linear(hidden_dims[-1], 128),
                    nn.ELU(),
                    nn.Linear(128, 1),
                )

                # Initialize
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight, gain=0.01)
                        nn.init.zeros_(m.bias)

            def forward(self, obs):
                features = self.features(obs)
                return features

            def get_action(self, obs, deterministic=False):
                features = self.forward(obs)
                mean = self.actor_mean(features)
                std = self.actor_log_std.exp()

                if deterministic:
                    return mean, None, None

                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                return action, log_prob, dist.entropy().sum(-1)

            def get_value(self, obs):
                features = self.forward(obs)
                return self.critic(features).squeeze(-1)

            def evaluate_actions(self, obs, actions):
                features = self.forward(obs)
                mean = self.actor_mean(features)
                std = self.actor_log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(actions).sum(-1)
                entropy = dist.entropy().sum(-1)
                value = self.critic(features).squeeze(-1)
                return value, log_prob, entropy

        return ActorCritic()
    except ImportError:
        print("PyTorch not available")
        return None


# ═══════════════════════════════════════════════════════════════════
#  Vectorised PPO Training Loop
# ═══════════════════════════════════════════════════════════════════


def train_ppo_vec(env, policy, config, logger=None,
                  phase="phase1", curriculum_stage=None):
    """PPO loop for a Genesis n_envs-style vectorised env.

    Differences from the single-env loop:
      * Rollout buffer is (n_steps, n_envs, ...) so each iteration
        consumes n_steps*n_envs interactions.
      * Done-flag handling is per-env; the env auto-resets finished envs
        so we never need to call env.reset() inside the rollout.
      * Observation / return normalisation is shared across envs.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("PyTorch required for training.")
        return

    from training.normalizers import (
        RunningMeanStd, ReturnNormalizer, linear_std_schedule,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)

    n_envs = env.num_envs
    n_steps = config.n_steps
    obs_dim = config.obs_dim
    act_dim = config.act_dim

    obs_norm = RunningMeanStd(shape=(obs_dim,))
    ret_norm = ReturnNormalizer(gamma=config.gamma)

    ep_rewards = deque(maxlen=200)
    ep_lengths = deque(maxlen=200)
    running_ep_r = np.zeros(n_envs, dtype=np.float32)
    running_ep_len = np.zeros(n_envs, dtype=np.int64)
    # Rolling batch-mean component metrics (updated every step)
    comp_queues: dict = {}

    total_steps = 0
    # Each iteration produces n_steps*n_envs samples
    num_iterations = max(1, config.total_timesteps // (n_steps * n_envs))

    print(f"\n{'='*60}")
    print(f" Phase: {phase} | Stage: {curriculum_stage or 'full'}")
    print(f" Vectorised | n_envs={n_envs} | n_steps={n_steps}")
    print(f" Total steps target: {config.total_timesteps:,}")
    print(f" Device: {device}")
    print(f"{'='*60}\n")

    obs = env.reset()
    obs_norm.update(obs)
    obs_t = torch.as_tensor(obs_norm.normalize(obs), device=device)

    for iteration in range(num_iterations):
        obs_buf = torch.zeros(n_steps, n_envs, obs_dim, device=device)
        act_buf = torch.zeros(n_steps, n_envs, act_dim, device=device)
        logp_buf = torch.zeros(n_steps, n_envs, device=device)
        rew_buf = torch.zeros(n_steps, n_envs, device=device)
        val_buf = torch.zeros(n_steps, n_envs, device=device)
        done_buf = torch.zeros(n_steps, n_envs, device=device)

        for step in range(n_steps):
            with torch.no_grad():
                action, log_prob, _ = policy.get_action(obs_t)
                value = policy.get_value(obs_t)

            action_np = action.cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)

            obs_buf[step] = obs_t
            act_buf[step] = action
            logp_buf[step] = log_prob
            rew_buf[step] = torch.as_tensor(reward, device=device)
            val_buf[step] = value
            done_buf[step] = torch.as_tensor(done.astype(np.float32),
                                              device=device)

            running_ep_r += reward
            running_ep_len += 1
            total_steps += n_envs

            # Episode bookkeeping for any envs that finished THIS step
            if done.any():
                for i in np.where(done)[0]:
                    ep_rewards.append(float(running_ep_r[i]))
                    ep_lengths.append(int(running_ep_len[i]))
                    running_ep_r[i] = 0.0
                    running_ep_len[i] = 0

            # Batch-mean reward components from vec env info
            for k, v in info.get("reward_components", {}).items():
                comp_queues.setdefault(k, deque(maxlen=200)).append(float(v))

            obs_norm.update(next_obs)
            obs_t = torch.as_tensor(obs_norm.normalize(next_obs), device=device)

        # Reward normalisation
        rew_np = rew_buf.detach().cpu().numpy().reshape(-1)
        done_np = done_buf.detach().cpu().numpy().reshape(-1)
        ret_norm.update(rew_np, done_np)
        rew_buf = rew_buf / max(1e-8, float(ret_norm.rms.std))

        # GAE — vectorised over envs
        with torch.no_grad():
            next_value = policy.get_value(obs_t)

        advantages = torch.zeros_like(rew_buf)
        gae = torch.zeros(n_envs, device=device)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
                next_done = torch.zeros(n_envs, device=device)
            else:
                next_val = val_buf[t + 1]
                next_done = done_buf[t + 1]
            delta = (rew_buf[t] + config.gamma * next_val
                     * (1 - next_done) - val_buf[t])
            gae = delta + config.gamma * config.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        returns = advantages + val_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Action log_std schedule
        with torch.no_grad():
            prog = iteration / max(1, num_iterations - 1)
            policy.actor_log_std.data.fill_(
                linear_std_schedule(initial=-0.5, final=-1.5, progress=prog)
            )

        # Flatten (n_steps, n_envs, ...) → (n_steps*n_envs, ...) for mini-batch
        b_obs = obs_buf.reshape(n_steps * n_envs, obs_dim)
        b_act = act_buf.reshape(n_steps * n_envs, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        n_samples = b_obs.shape[0]
        mb_size = max(1, n_samples // 4)

        for epoch in range(config.n_epochs):
            indices = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, mb_size):
                end = start + mb_size
                mb_idx = indices[start:end]
                mb_obs = b_obs[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logp = b_logp[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]

                values, new_log_prob, entropy = policy.evaluate_actions(
                    mb_obs, mb_act,
                )
                ratio = (new_log_prob - mb_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - config.clip_range,
                                     1 + config.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = (policy_loss
                        + config.vf_coef * value_loss
                        + config.entropy_coef * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),
                                         config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_prob - mb_logp
                    approx_kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
                    clip_frac = float((log_ratio.abs() > math.log(1 + config.clip_range)).float().mean())

        # Explained variance
        with torch.no_grad():
            y_true = b_ret.cpu().numpy()
            y_pred = val_buf.reshape(-1).cpu().numpy()
            var_y = float(np.var(y_true))
            explained_var = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))

        if iteration % 10 == 0:
            mr = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            ml = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            sr = float(np.std(ep_rewards)) if len(ep_rewards) > 1 else 0.0
            mean_robot_z = float(obs_buf[:, :, 2].mean().item())
            print(f"[{phase}|{curriculum_stage or 'full'}|vec] "
                  f"Iter {iteration:5d} | Steps {total_steps:12,d} | "
                  f"R̄={mr:7.2f} (σ={sr:.2f}) | L̄={ml:6.1f} | "
                  f"z̄={mean_robot_z:.3f} | "
                  f"π={policy_loss.item():.4f} V={value_loss.item():.4f} "
                  f"KL={approx_kl:.4f} ev={explained_var:.3f}")
            metrics = {
                "mean_reward": mr, "std_reward": sr, "mean_length": ml,
                "mean_robot_z": mean_robot_z,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": -entropy_loss.item(),
                "explained_variance": explained_var,
                "approx_kl": approx_kl,
                "clip_fraction": clip_frac,
            }
            for k, q in comp_queues.items():
                if q:
                    metrics[f"rewards/{k}"] = float(np.mean(q))
            log_metrics(logger, metrics, total_steps)

        if iteration % 100 == 0 and iteration > 0:
            save_checkpoint(policy, optimizer, total_steps, phase,
                            curriculum_stage, obs_norm=obs_norm,
                            ret_norm=ret_norm)

    save_checkpoint(policy, optimizer, total_steps, phase, curriculum_stage,
                    obs_norm=obs_norm, ret_norm=ret_norm)
    return policy


# ═══════════════════════════════════════════════════════════════════
#  PPO Training Loop (single-env)
# ═══════════════════════════════════════════════════════════════════


def train_ppo(
    env, policy, config, logger=None, phase="phase1", curriculum_stage=None
):
    """
    PPO training loop.

    Args:
        env: Training environment (Phase1 or Phase2).
        policy: ActorCritic network.
        config: Phase1Config or Phase2Config.
        logger: TrainingLogger (TensorBoard + optional W&B). None = quiet.
        phase: "phase1" or "phase2".
        curriculum_stage: Current curriculum stage (Phase 1 only).
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("PyTorch required for training. Install with: pip install torch")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)

    # Running observation / return normalizers — pre-condition the
    # network with well-scaled inputs for ~2x faster convergence on PPO.
    from training.normalizers import (
        RunningMeanStd, ReturnNormalizer, linear_std_schedule,
    )
    obs_norm = RunningMeanStd(shape=(config.obs_dim,))
    ret_norm = ReturnNormalizer(gamma=config.gamma)

    # Rolling stats
    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    # Per-reward-component rolling queues (populated on episode end)
    comp_queues: dict = {}
    ep_components: dict = {}  # accumulates within the current episode

    # Rollout buffer
    n_steps = config.n_steps
    obs_dim = config.obs_dim
    act_dim = config.act_dim

    total_steps = 0
    num_iterations = config.total_timesteps // n_steps

    print(f"\n{'='*60}")
    print(f" Phase: {phase} | Stage: {curriculum_stage or 'full'}")
    print(f" Total steps: {config.total_timesteps:,}")
    print(f" Device: {device}")
    print(f"{'='*60}\n")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs[0]  # multi-agent: use first agent for single training
    obs_norm.update(obs)
    obs_tensor = torch.FloatTensor(obs_norm.normalize(obs)).to(device)

    episode_reward = 0.0
    episode_length = 0

    for iteration in range(num_iterations):
        # ── Collect rollout ──
        obs_buf = torch.zeros(n_steps, obs_dim, device=device)
        act_buf = torch.zeros(n_steps, act_dim, device=device)
        logp_buf = torch.zeros(n_steps, device=device)
        rew_buf = torch.zeros(n_steps, device=device)
        val_buf = torch.zeros(n_steps, device=device)
        done_buf = torch.zeros(n_steps, device=device)

        for step in range(n_steps):
            with torch.no_grad():
                action, log_prob, _ = policy.get_action(obs_tensor.unsqueeze(0))
                value = policy.get_value(obs_tensor.unsqueeze(0))

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)
            if isinstance(next_obs, dict):
                next_obs = next_obs[0]

            obs_buf[step] = obs_tensor
            act_buf[step] = action.squeeze(0)
            logp_buf[step] = log_prob.squeeze(0)
            rew_buf[step] = float(reward)
            val_buf[step] = value.squeeze(0)
            done_buf[step] = float(done)

            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Accumulate per-component rewards for this episode
            for k, v in info.get("reward_components", {}).items():
                if k != "total":
                    ep_components[k] = ep_components.get(k, 0.0) + float(v)

            if done:
                ep_rewards.append(episode_reward)
                ep_lengths.append(episode_length)
                ep_len = max(1, episode_length)
                for k, v in ep_components.items():
                    comp_queues.setdefault(k, deque(maxlen=100)).append(v / ep_len)
                ep_components = {}
                episode_reward = 0.0
                episode_length = 0
                next_obs = env.reset()
                if isinstance(next_obs, dict):
                    next_obs = next_obs[0]

            # Update observation running stats and feed normalised obs to net
            obs_norm.update(next_obs)
            obs_tensor = torch.FloatTensor(obs_norm.normalize(next_obs)).to(device)

        # Reward normalisation: track running std of discounted returns,
        # divide rewards by it so the value loss has stable scale.
        rew_np = rew_buf.detach().cpu().numpy()
        done_np = done_buf.detach().cpu().numpy()
        ret_norm.update(rew_np, done_np)
        norm_factor = float(ret_norm.rms.std + 1e-8)
        rew_buf = rew_buf / norm_factor

        # ── Compute GAE ──
        with torch.no_grad():
            next_value = policy.get_value(obs_tensor.unsqueeze(0)).squeeze(0)

        advantages = torch.zeros(n_steps, device=device)
        gae = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
                next_done = 0.0
            else:
                next_val = val_buf[t + 1]
                next_done = done_buf[t + 1]

            delta = rew_buf[t] + config.gamma * next_val * (1 - next_done) - val_buf[t]
            gae = delta + config.gamma * config.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        returns = advantages + val_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Explained variance (before update — how well V̂ predicts returns)
        with torch.no_grad():
            y_true = returns.reshape(-1).cpu().numpy()
            y_pred = val_buf.reshape(-1).cpu().numpy()
            var_y = float(np.var(y_true))
            explained_var = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))

        # Linear decay of action log_std so exploration shrinks over training
        with torch.no_grad():
            prog = iteration / max(1, num_iterations - 1)
            target_log_std = linear_std_schedule(initial=-0.5, final=-1.5,
                                                 progress=prog)
            policy.actor_log_std.data.fill_(target_log_std)

        # ── PPO Update ──
        approx_kl = 0.0
        clip_frac = 0.0
        for epoch in range(config.n_epochs):
            # Mini-batch indices
            indices = torch.randperm(n_steps, device=device)
            mb_size = max(1, n_steps // 4)

            for start in range(0, n_steps, mb_size):
                end = start + mb_size
                mb_idx = indices[start:end]

                mb_obs = obs_buf[mb_idx]
                mb_act = act_buf[mb_idx]
                mb_logp = logp_buf[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                values, new_log_prob, entropy = policy.evaluate_actions(mb_obs, mb_act)

                # Policy loss (clipped)
                ratio = (new_log_prob - mb_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range)
                    * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + config.vf_coef * value_loss
                    + config.entropy_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_prob - mb_logp
                    approx_kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
                    clip_frac = float((log_ratio.abs() > math.log(1 + config.clip_range)).float().mean())

        # ── Logging ──
        if iteration % 10 == 0 or len(ep_rewards) > 0:
            mr = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            ml = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            sr = float(np.std(ep_rewards)) if len(ep_rewards) > 1 else 0.0
            mean_robot_z = float(obs_buf[:, 2].mean().item())
            metrics = {
                "iteration": iteration,
                "total_steps": total_steps,
                "mean_reward": mr,
                "std_reward": sr,
                "mean_length": ml,
                "mean_robot_z": mean_robot_z,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": -entropy_loss.item(),
                "explained_variance": explained_var,
                "approx_kl": approx_kl,
                "clip_fraction": clip_frac,
            }
            # Per-component reward breakdown
            for k, q in comp_queues.items():
                if q:
                    metrics[f"rewards/{k}"] = float(np.mean(q))

            if iteration % 10 == 0:
                print(
                    f"[{phase}|{curriculum_stage or 'full'}] "
                    f"Iter {iteration:5d} | "
                    f"Steps {total_steps:10,d} | "
                    f"R̄={mr:7.2f} (σ={sr:.2f}) | "
                    f"L̄={ml:5.1f} | z̄={mean_robot_z:.3f} | "
                    f"π={policy_loss.item():.4f} V={value_loss.item():.4f} "
                    f"KL={approx_kl:.4f} ev={explained_var:.3f}"
                )

            log_metrics(logger, metrics, total_steps)

        # ── Checkpointing ──
        if iteration % 100 == 0 and iteration > 0:
            save_checkpoint(policy, optimizer, total_steps, phase, curriculum_stage,
                            obs_norm=obs_norm, ret_norm=ret_norm)

        # ── Video recording ──
        if (
            logger is not None
            and iteration % config.__dict__.get("video_frequency", 200) == 0
            and iteration > 0
        ):
            video_path = record_eval_video(env, policy, device, phase, total_steps)
            if video_path:
                log_metrics(logger, {}, total_steps, video_path=video_path)

    # Final checkpoint
    save_checkpoint(policy, optimizer, total_steps, phase, curriculum_stage)
    return policy


def record_eval_video(
    env, policy, device, phase: str, step: int, max_frames: int = 300
) -> Optional[str]:
    """Record a short evaluation video."""
    try:
        import torch

        os.makedirs(VIDEOS_DIR, exist_ok=True)
        frames = []

        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs[0]

        for _ in range(max_frames):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_t, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()

            obs, _, done, _ = env.step(action_np)
            if isinstance(obs, dict):
                obs = obs[0]

            frame = env.render_frame()
            if frame is not None:
                frames.append(frame)

            if done:
                break

        if frames:
            video_path = os.path.join(VIDEOS_DIR, f"{phase}_step{step}.mp4")
            _save_video(frames, video_path)
            return video_path
    except Exception as e:
        print(f"Video recording failed: {e}")
    return None


def _save_video(frames: list, path: str, fps: int = 30):
    """Save frames as MP4 video."""
    try:
        import imageio

        imageio.mimwrite(path, frames, fps=fps)
    except ImportError:
        try:
            import cv2

            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except ImportError:
            print("Install imageio or opencv-python for video recording")


# ═══════════════════════════════════════════════════════════════════
#  Main Entry Points
# ═══════════════════════════════════════════════════════════════════


def train_phase1(config: ProjectConfig, resume_from: str = None,
                 use_wandb: bool = False):
    """
    Phase 1: Train single-robot skills with curriculum learning.

    Curriculum stages: stand → standup → walk → dribble → shoot → full
    """
    from envs.phase1_dribble_shoot import K1DribbleShootEnv

    logger = setup_logger(config, "phase1", use_wandb=use_wandb)

    policy = create_policy(config.phase1.obs_dim, config.phase1.act_dim)
    if policy is None:
        print("Cannot create policy (PyTorch missing)")
        return

    if resume_from:
        import torch

        optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.phase1.learning_rate
        )
        load_checkpoint(resume_from, policy, optimizer)

    use_vec = getattr(config.phase1, "use_vec_env", False)
    if use_vec:
        from envs.phase1_vec import K1DribbleShootVecEnv

        def _make_env(stage):
            return K1DribbleShootVecEnv(
                num_envs=config.phase1.vec_num_envs,
                cfg=config.phase1,
                robot_cfg=config.robot,
                curriculum_stage=stage,
            )
        train_fn = train_ppo_vec
    else:
        def _make_env(stage):
            return K1DribbleShootEnv(
                cfg=config.phase1,
                robot_cfg=config.robot,
                curriculum_stage=stage,
            )
        train_fn = train_ppo

    if config.phase1.use_curriculum:
        stages = config.phase1.curriculum_stages
        steps_per_stage = config.phase1.total_timesteps // len(stages)

        for stage in stages:
            print(f"\n>>> Curriculum Stage: {stage.upper()}")
            env = _make_env(stage)
            stage_config = Phase1Config(**config.phase1.__dict__)
            stage_config.total_timesteps = steps_per_stage
            policy = train_fn(env, policy, stage_config, logger,
                              "phase1", stage)
            env.close()
    else:
        env = _make_env("full")
        policy = train_fn(env, policy, config.phase1, logger, "phase1")
        env.close()

    logger.close()
    return policy


def train_phase2(config: ProjectConfig, phase1_checkpoint: str = None,
                 use_wandb: bool = False):
    """
    Phase 2: Fine-tune in multi-robot match environment.

    Uses self-play: trains home team policy while periodically
    updating the opponent pool with past versions.
    """
    from envs.phase2_match import K1SoccerMatchEnv

    logger = setup_logger(config, "phase2", use_wandb=use_wandb)

    # Create policy and load Phase 1 weights
    policy = create_policy(config.phase2.obs_dim, config.phase2.act_dim)
    if policy is None:
        return

    if phase1_checkpoint:
        # Load Phase 1 weights (obs_dim may differ, partial load)
        try:
            import torch

            ckpt = torch.load(phase1_checkpoint, map_location="cpu")
            # Try partial loading
            state_dict = ckpt.get("policy_state_dict", {})
            model_dict = policy.state_dict()
            compatible = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(compatible)
            policy.load_state_dict(model_dict)
            print(f"Loaded {len(compatible)}/{len(state_dict)} layers from Phase 1")
        except Exception as e:
            print(f"Could not load Phase 1 checkpoint: {e}")

    # Opponent policy (clone for self-play)
    opponent_policy = create_policy(config.phase2.obs_dim, config.phase2.act_dim)
    opponent_pool = []

    env = K1SoccerMatchEnv(
        cfg=config.phase2,
        robot_cfg=config.robot,
    )

    print(
        f"\n>>> Phase 2: Match Training ({config.phase2.players_per_team}v"
        f"{config.phase2.players_per_team})"
    )

    policy = train_ppo(env, policy, config.phase2, logger, "phase2")

    env.close()
    logger.close()
    return policy


def main():
    parser = argparse.ArgumentParser(description="RoboCup Humanoid Soccer RL Training")
    parser.add_argument(
        "--phase",
        choices=["1", "2", "both"],
        default="both",
        help="Training phase (1=skills, 2=match, both)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--phase1-ckpt",
        type=str,
        default=None,
        help="Phase 1 checkpoint for Phase 2 fine-tuning",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb", action="store_true",
        help="Also log to Weights & Biases (TensorBoard is always on)",
    )
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Override W&B project name (implies --wandb)")
    parser.add_argument(
        "--aggressiveness", type=float, default=0.3, help="Aggressiveness level 0.0-1.0"
    )
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--vec-num-envs", type=int, default=None,
                        help="Use vectorised Genesis env with this many parallel envs")
    args = parser.parse_args()

    config = ProjectConfig()
    config.seed = args.seed

    use_wandb = bool(args.wandb or args.wandb_project)
    if args.wandb_project:
        config.wandb.project = args.wandb_project
    if args.no_curriculum:
        config.phase1.use_curriculum = False
    if args.aggressiveness:
        config.phase1.reward.scale_aggressiveness(args.aggressiveness)
        config.phase2.reward.scale_aggressiveness(args.aggressiveness)
    if args.vec_num_envs:
        config.phase1.use_vec_env = True
        config.phase1.vec_num_envs = args.vec_num_envs

    np.random.seed(config.seed)

    phase1_ckpt = args.phase1_ckpt

    if args.phase in ("1", "both"):
        policy = train_phase1(config, resume_from=args.resume,
                              use_wandb=use_wandb)
        # Find latest checkpoint
        if phase1_ckpt is None and os.path.exists(CHECKPOINTS_DIR):
            ckpts = sorted(Path(CHECKPOINTS_DIR).glob("phase1_*.pt"))
            if ckpts:
                phase1_ckpt = str(ckpts[-1])

    if args.phase in ("2", "both"):
        train_phase2(config, phase1_checkpoint=phase1_ckpt,
                     use_wandb=use_wandb)


if __name__ == "__main__":
    main()
