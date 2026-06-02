"""Improved PPO trainer (rsl_rl-style).

Key differences vs the old training/train.py loop:
  * Separate actor / critic trunks (more stable on long PPO horizons).
  * Optional value-loss clipping (default ON).
  * Adaptive KL early stopping: if mean KL across mini-batches exceeds
    a threshold this iteration, we break out of remaining update epochs.
  * Adaptive LR: rsl_rl multiplies LR by 1.5 if KL is too high, divides
    by 1.5 if too low. Bounded to [1e-5, 1e-2].
  * Vectorised rollout supports Genesis n_envs natively.
  * Per-component reward metrics threaded through to the logger.
  * Diagnostic metrics: explained_variance, approx_kl, clip_fraction,
    mean trunk z (so we can see if the robot is learning to stand).

The single-env path is kept for backwards compatibility (and the unit
tests that don't have GPU Genesis available) — it shares the same actor-
critic class. For real training, use the vec env path.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from training.algorithms.networks import PPOActorCritic
from training.normalizers import RunningMeanStd, ReturnNormalizer


# ─── checkpoint I/O ────────────────────────────────────────────────────


def save_ppo_checkpoint(policy, optimizer, step, phase, stage=None, path=None,
                       obs_norm=None, ret_norm=None,
                       checkpoint_dir: str = "checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if path is None:
        suffix = f"_{stage}" if stage else ""
        path = os.path.join(checkpoint_dir, f"{phase}{suffix}_step{step}.pt")
    ckpt = {
        "step": step, "phase": phase, "stage": stage,
        "algorithm": "ppo",
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if obs_norm is not None and hasattr(obs_norm, "state_dict"):
        ckpt["obs_norm"] = obs_norm.state_dict()
    if ret_norm is not None and hasattr(ret_norm, "state_dict"):
        ckpt["ret_norm"] = ret_norm.state_dict()
    torch.save(ckpt, path)
    print(f"[ppo] checkpoint → {path}")
    return path


# ─── Vec-env PPO (the fast path) ───────────────────────────────────────


def train_ppo_vec(env, policy, config, logger=None,
                  phase="phase1", curriculum_stage=None,
                  checkpoint_dir: str = "checkpoints",
                  desired_kl: float = 0.01,
                  use_value_clipping: bool = True,
                  adaptive_lr: bool = True,
                  min_lr: float = 1e-5,
                  max_lr: float = 1e-2,
                  video_frequency: int = 50,
                  video_n_frames: int = 300,
                  video_fps: int = 30,
                  device: Optional[torch.device] = None):
    """PPO for vectorised Genesis env.

    Args:
        env: Vec env with `.num_envs`, `.reset()`, `.step()` returning
             (obs[N,O], rew[N], done[N], info).
        policy: PPOActorCritic instance, or None to build one.
        config: a duck-typed config object exposing `obs_dim`, `act_dim`,
                `learning_rate`, `gamma`, `gae_lambda`, `clip_range`,
                `entropy_coef`, `vf_coef`, `max_grad_norm`, `n_epochs`,
                `n_steps`, `total_timesteps`. The per-skill configs
                (WalkConfig / StandupConfig / DribbleConfig / ShootConfig)
                and `OrchestratorConfig` all satisfy this contract.
        desired_kl: Target KL per iteration for adaptive LR + early stop.
        use_value_clipping: Apply PPO-style value clipping.
        adaptive_lr: rsl_rl-style LR adjustment based on KL.
        device: torch device. If None, auto-detect CUDA.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if policy is None:
        policy = PPOActorCritic(config.obs_dim, config.act_dim)
    policy = policy.to(device)
    lr = float(config.learning_rate)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    n_envs = env.num_envs
    n_steps = int(config.n_steps)
    obs_dim = int(config.obs_dim)
    act_dim = int(config.act_dim)

    obs_norm = RunningMeanStd(shape=(obs_dim,))
    ret_norm = ReturnNormalizer(gamma=config.gamma)

    ep_rewards = deque(maxlen=200)
    ep_lengths = deque(maxlen=200)
    running_ep_r = np.zeros(n_envs, dtype=np.float32)
    running_ep_len = np.zeros(n_envs, dtype=np.int64)
    comp_queues: dict = {}

    total_steps = 0
    num_iterations = max(1, int(config.total_timesteps) // (n_steps * n_envs))

    # Video capture state. We piggy-back on the rollout loop — when
    # active, every env step in env 0 captures one frame. Once we have
    # `video_n_frames` we push to the logger and stop until the next
    # video_frequency-th iteration.
    video_recording = False
    video_frames: list = []
    video_supported = hasattr(env, "render_frame")
    if logger is None or video_frequency <= 0:
        video_supported = False

    print(f"\n{'='*60}")
    print(f" [PPO] phase={phase} stage={curriculum_stage or 'full'}")
    print(f"   n_envs={n_envs}  n_steps={n_steps}  iters={num_iterations}")
    print(f"   total_steps target={config.total_timesteps:,}  device={device}")
    print(f"   desired_kl={desired_kl}  val_clip={use_value_clipping}"
          f"  adaptive_lr={adaptive_lr}")
    print(f"   video: {'every ' + str(video_frequency) + ' iters, ' + str(video_n_frames) + ' frames' if video_supported else 'disabled'}")
    print(f"{'='*60}\n")

    obs = env.reset()
    obs_norm.update(obs)
    obs_t = torch.as_tensor(obs_norm.normalize(obs),
                            dtype=torch.float32, device=device)

    # Persistent rollout buffer — re-used each iteration
    obs_buf = torch.zeros(n_steps, n_envs, obs_dim, device=device)
    act_buf = torch.zeros(n_steps, n_envs, act_dim, device=device)
    logp_buf = torch.zeros(n_steps, n_envs, device=device)
    rew_buf = torch.zeros(n_steps, n_envs, device=device)
    val_buf = torch.zeros(n_steps, n_envs, device=device)
    done_buf = torch.zeros(n_steps, n_envs, device=device)
    # Episode-end split for correct GAE: term_buf = true terminal (no
    # bootstrap), trunc_buf = time-limit (bootstrap from terminal-obs value).
    term_buf = torch.zeros(n_steps, n_envs, device=device)
    trunc_buf = torch.zeros(n_steps, n_envs, device=device)
    truncval_buf = torch.zeros(n_steps, n_envs, device=device)

    for iteration in range(num_iterations):
        # Kick off a new video clip every `video_frequency` iterations.
        # Skip iter 0 — first-iteration footage is uninformative noise.
        if (video_supported and not video_recording
                and iteration > 0
                and iteration % video_frequency == 0):
            video_recording = True
            video_frames = []

        # ── Collect rollout ──
        with torch.no_grad():
            for step in range(n_steps):
                action, log_prob, _ = policy.act(obs_t)
                value = policy.get_value(obs_t)

                action_np = action.cpu().numpy()
                next_obs, reward, done, info = env.step(action_np)

                obs_buf[step] = obs_t
                act_buf[step] = action
                logp_buf[step] = log_prob
                rew_buf[step] = torch.as_tensor(reward, device=device,
                                                dtype=torch.float32)
                val_buf[step] = value
                done_buf[step] = torch.as_tensor(
                    np.asarray(done, dtype=np.float32), device=device)

                # Time-limit bookkeeping. Fall back to treating every done as
                # a true terminal if the env doesn't provide the split.
                terminated = np.asarray(
                    info.get("terminated", done), dtype=np.float32)
                truncated = np.asarray(
                    info.get("truncated", np.zeros_like(terminated)),
                    dtype=np.float32)
                term_buf[step] = torch.as_tensor(terminated, device=device)
                trunc_buf[step] = torch.as_tensor(truncated, device=device)
                # On a truncation the returned next_obs is a reset state, so
                # the bootstrap value must come from the TERMINAL obs.
                if truncated.any() and "terminal_obs" in info:
                    tobs = obs_norm.normalize(np.asarray(info["terminal_obs"]))
                    truncval_buf[step] = policy.get_value(
                        torch.as_tensor(tobs, dtype=torch.float32,
                                        device=device))
                else:
                    truncval_buf[step] = 0.0

                running_ep_r += reward
                running_ep_len += 1
                total_steps += n_envs

                # Episode bookkeeping
                if np.any(done):
                    for i in np.where(done)[0]:
                        ep_rewards.append(float(running_ep_r[i]))
                        ep_lengths.append(int(running_ep_len[i]))
                        running_ep_r[i] = 0.0
                        running_ep_len[i] = 0

                for k, v in info.get("reward_components", {}).items():
                    comp_queues.setdefault(k, deque(maxlen=200)).append(float(v))

                # Capture a frame for video logging if active
                if video_recording:
                    frame = env.render_frame()
                    if frame is not None:
                        video_frames.append(frame)
                    if len(video_frames) >= video_n_frames:
                        try:
                            logger.log_video("train/rollout", video_frames,
                                             step=total_steps, fps=video_fps)
                            print(f"[ppo] logged {len(video_frames)}-frame video "
                                  f"at step {total_steps:,}")
                        except Exception as e:
                            print(f"[ppo] video log failed: {e}")
                        video_recording = False
                        video_frames = []

                obs_norm.update(next_obs)
                obs_t = torch.as_tensor(obs_norm.normalize(next_obs),
                                        dtype=torch.float32, device=device)

        # ── Reward normalisation ──
        rew_np = rew_buf.cpu().numpy().reshape(-1)
        done_np = done_buf.cpu().numpy().reshape(-1)
        ret_norm.update(rew_np, done_np)
        rew_norm_buf = rew_buf / max(1e-8, float(ret_norm.rms.std))

        # ── GAE ──
        with torch.no_grad():
            next_value = policy.get_value(obs_t)

        advantages = torch.zeros_like(rew_norm_buf)
        gae = torch.zeros(n_envs, device=device)
        for t in reversed(range(n_steps)):
            next_val = next_value if t == n_steps - 1 else val_buf[t + 1]
            # On truncation the stored next state is a reset obs; bootstrap
            # from the terminal-obs value instead.
            boot = torch.where(trunc_buf[t] > 0, truncval_buf[t], next_val)
            # Zero the bootstrap ONLY on a true terminal; reset the GAE trace
            # on ANY episode end (uses the CURRENT step's flags — done_buf[t]
            # marks that THIS step ended the episode).
            nonterminal = 1.0 - term_buf[t]
            notdone = 1.0 - done_buf[t]
            delta = (rew_norm_buf[t]
                     + config.gamma * boot * nonterminal
                     - val_buf[t])
            gae = delta + config.gamma * config.gae_lambda * notdone * gae
            advantages[t] = gae
        returns = advantages + val_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Explained variance BEFORE the update (a cleaner signal than after)
        with torch.no_grad():
            y_true = returns.reshape(-1).cpu().numpy()
            y_pred = val_buf.reshape(-1).cpu().numpy()
            var_y = float(np.var(y_true))
            explained_var = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))

        # Exploration std is the learnable `actor_log_std`, driven by the
        # entropy bonus + policy gradient (NOT a fixed wall-clock schedule —
        # that overwrite made entropy_coef a no-op).

        # ── Flatten for mini-batching ──
        b_obs = obs_buf.reshape(n_steps * n_envs, obs_dim)
        b_act = act_buf.reshape(n_steps * n_envs, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        n_samples = b_obs.shape[0]
        mb_size = max(1, n_samples // 4)

        approx_kl = 0.0
        clip_frac = 0.0
        policy_loss_val = 0.0
        value_loss_val = 0.0
        entropy_val = 0.0
        kl_too_big = False

        for epoch in range(config.n_epochs):
            indices = torch.randperm(n_samples, device=device)
            epoch_kls = []
            for start in range(0, n_samples, mb_size):
                end = start + mb_size
                mb_idx = indices[start:end]
                mb_obs = b_obs[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logp = b_logp[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]
                mb_val_old = b_val[mb_idx]

                values, new_log_prob, entropy = policy.evaluate(mb_obs, mb_act)

                ratio = (new_log_prob - mb_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - config.clip_range,
                                    1 + config.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if use_value_clipping:
                    v_clip = mb_val_old + (values - mb_val_old).clamp(
                        -config.clip_range, config.clip_range
                    )
                    v_loss_unclipped = (values - mb_ret).pow(2)
                    v_loss_clipped = (v_clip - mb_ret).pow(2)
                    value_loss = 0.5 * torch.max(
                        v_loss_unclipped, v_loss_clipped).mean()
                else:
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
                    # k3 estimator: low-variance unbiased KL
                    kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                    approx_kl = float(kl)
                    clip_frac = float((log_ratio.abs()
                                       > math.log(1 + config.clip_range)
                                       ).float().mean())
                    epoch_kls.append(approx_kl)

                policy_loss_val = float(policy_loss.detach())
                value_loss_val = float(value_loss.detach())
                entropy_val = float(-entropy_loss.detach())

            # Early stop if KL ran away this epoch
            mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if desired_kl > 0 and mean_kl > 2.0 * desired_kl:
                kl_too_big = True
                break

        # ── Adaptive LR (rsl_rl) ──
        if adaptive_lr and desired_kl > 0:
            if approx_kl > 2.0 * desired_kl:
                lr = max(min_lr, lr / 1.5)
            elif approx_kl < 0.5 * desired_kl:
                lr = min(max_lr, lr * 1.5)
            for g in optimizer.param_groups:
                g["lr"] = lr

        # ── Logging ──
        if iteration % 10 == 0 or len(ep_rewards) > 0:
            mr = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            ml = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            sr = float(np.std(ep_rewards)) if len(ep_rewards) > 1 else 0.0
            # obs_buf holds NORMALIZED obs; un-normalize the root-height
            # channel (index 0) for a human-readable metric.
            mean_robot_z = (float(obs_buf[:, :, 0].mean())
                            * float(obs_norm.std[0])
                            + float(obs_norm.mean[0]))
            metrics = {
                "mean_reward": mr, "std_reward": sr, "mean_length": ml,
                "mean_robot_z": mean_robot_z,
                "policy_loss": policy_loss_val,
                "value_loss": value_loss_val,
                "entropy": entropy_val,
                "explained_variance": explained_var,
                "approx_kl": approx_kl,
                "clip_fraction": clip_frac,
                "learning_rate": lr,
                "kl_early_stop": float(kl_too_big),
            }
            for k, q in comp_queues.items():
                if q:
                    metrics[f"rewards/{k}"] = float(np.mean(q))

            print(f"[ppo|{curriculum_stage or 'full'}|vec] "
                  f"Iter {iteration:5d} | Steps {total_steps:12,d} | "
                  f"R̄={mr:7.2f} (σ={sr:.2f}) | L̄={ml:6.1f} | "
                  f"z̄={mean_robot_z:.3f} | "
                  f"π={policy_loss_val:.4f} V={value_loss_val:.4f} "
                  f"KL={approx_kl:.4f} lr={lr:.1e} ev={explained_var:.3f}"
                  + ("  [KL-stop]" if kl_too_big else ""))
            _log_metrics(logger, metrics, total_steps)

        if iteration % 100 == 0 and iteration > 0:
            save_ppo_checkpoint(policy, optimizer, total_steps, phase,
                                curriculum_stage,
                                checkpoint_dir=checkpoint_dir,
                                obs_norm=obs_norm, ret_norm=ret_norm)

    save_ppo_checkpoint(policy, optimizer, total_steps, phase,
                        curriculum_stage, checkpoint_dir=checkpoint_dir,
                        obs_norm=obs_norm, ret_norm=ret_norm)
    return policy


# ─── Multi-critic vec PPO (HoST-style) ─────────────────────────────────


def train_ppo_multicritic_vec(env, policy, config, logger=None,
                              phase="phase1", curriculum_stage=None,
                              checkpoint_dir: str = "checkpoints",
                              group_weights=None,
                              desired_kl: float = 0.01,
                              use_value_clipping: bool = True,
                              adaptive_lr: bool = True,
                              min_lr: float = 1e-5,
                              max_lr: float = 1e-2,
                              video_frequency: int = 50,
                              video_n_frames: int = 300,
                              video_fps: int = 30,
                              device: Optional[torch.device] = None):
    """PPO with one critic per reward GROUP (HoST, arXiv:2502.08378).

    Differs from `train_ppo_vec` only in the value/advantage path:
      * The env returns `info["group_rewards"]` shaped (N, G); `policy`
        must be a `PPOActorCritic(..., n_critics=G)`.
      * Each group gets its OWN return-normalizer, GAE, and value head,
        so a critic only ever fits a homogeneous-scale return (the whole
        point — a single critic over the mixed reward fails per HoST).
      * Advantages are normalized PER GROUP, then aggregated as a
        `group_weights`-weighted sum into the scalar advantage that drives
        the (unchanged) clipped policy surrogate.
      * Value loss is the mean over (sample, group) of each head's
        (optionally clipped) MSE to its own group return.

    The scalar `reward` from `env.step` is still used for episode-return
    bookkeeping/logging, so dashboards stay comparable to single-critic.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group_names = tuple(getattr(env, "CRITIC_GROUP_NAMES", ()))
    n_groups = len(group_names)
    if n_groups < 2:
        raise ValueError(
            "train_ppo_multicritic_vec needs env.CRITIC_GROUP_NAMES with "
            f"≥2 groups; got {group_names!r}. Use train_ppo_vec instead.")

    if policy is None:
        policy = PPOActorCritic(config.obs_dim, config.act_dim,
                                n_critics=n_groups)
    if getattr(policy, "n_critics", 1) != n_groups:
        raise ValueError(
            f"policy.n_critics={getattr(policy, 'n_critics', 1)} != "
            f"{n_groups} reward groups {group_names}")
    policy = policy.to(device)

    if group_weights is None:
        group_weights = [1.0] * n_groups
    gw = torch.as_tensor(group_weights, dtype=torch.float32, device=device)
    assert gw.shape == (n_groups,), f"group_weights must be {n_groups}-dim"

    lr = float(config.learning_rate)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    n_envs = env.num_envs
    n_steps = int(config.n_steps)
    obs_dim = int(config.obs_dim)
    act_dim = int(config.act_dim)

    obs_norm = RunningMeanStd(shape=(obs_dim,))
    # One return-normalizer per group — each group's return has its own scale.
    ret_norms = [ReturnNormalizer(gamma=config.gamma) for _ in range(n_groups)]

    ep_rewards = deque(maxlen=200)
    ep_lengths = deque(maxlen=200)
    running_ep_r = np.zeros(n_envs, dtype=np.float32)
    running_ep_len = np.zeros(n_envs, dtype=np.int64)
    comp_queues: dict = {}

    total_steps = 0
    num_iterations = max(1, int(config.total_timesteps) // (n_steps * n_envs))

    video_recording = False
    video_frames: list = []
    video_supported = hasattr(env, "render_frame")
    if logger is None or video_frequency <= 0:
        video_supported = False

    print(f"\n{'='*60}")
    print(f" [PPO|multi-critic] phase={phase} stage={curriculum_stage or 'full'}")
    print(f"   groups={group_names}  weights={[float(x) for x in gw]}")
    print(f"   n_envs={n_envs}  n_steps={n_steps}  iters={num_iterations}")
    print(f"   total_steps target={config.total_timesteps:,}  device={device}")
    print(f"{'='*60}\n")

    obs = env.reset()
    obs_norm.update(obs)
    obs_t = torch.as_tensor(obs_norm.normalize(obs),
                            dtype=torch.float32, device=device)

    obs_buf = torch.zeros(n_steps, n_envs, obs_dim, device=device)
    act_buf = torch.zeros(n_steps, n_envs, act_dim, device=device)
    logp_buf = torch.zeros(n_steps, n_envs, device=device)
    rew_buf = torch.zeros(n_steps, n_envs, n_groups, device=device)
    val_buf = torch.zeros(n_steps, n_envs, n_groups, device=device)
    done_buf = torch.zeros(n_steps, n_envs, device=device)
    # Episode-end split for correct per-group GAE (see train_ppo_vec).
    term_buf = torch.zeros(n_steps, n_envs, device=device)
    trunc_buf = torch.zeros(n_steps, n_envs, device=device)
    truncval_buf = torch.zeros(n_steps, n_envs, n_groups, device=device)

    for iteration in range(num_iterations):
        if (video_supported and not video_recording
                and iteration > 0
                and iteration % video_frequency == 0):
            video_recording = True
            video_frames = []

        # ── Collect rollout ──
        with torch.no_grad():
            for step in range(n_steps):
                action, log_prob, _ = policy.act(obs_t)
                value = policy.get_value(obs_t)  # (N, G)

                action_np = action.cpu().numpy()
                next_obs, reward, done, info = env.step(action_np)

                group_rew = info.get("group_rewards", None)
                if group_rew is None:
                    raise RuntimeError(
                        "multi-critic PPO requires info['group_rewards'] "
                        "(N,G); env returned None. Did the skill populate "
                        "self._group_rewards?")

                obs_buf[step] = obs_t
                act_buf[step] = action
                logp_buf[step] = log_prob
                rew_buf[step] = torch.as_tensor(np.asarray(group_rew),
                                                device=device,
                                                dtype=torch.float32)
                val_buf[step] = value
                done_buf[step] = torch.as_tensor(
                    np.asarray(done, dtype=np.float32), device=device)

                terminated = np.asarray(
                    info.get("terminated", done), dtype=np.float32)
                truncated = np.asarray(
                    info.get("truncated", np.zeros_like(terminated)),
                    dtype=np.float32)
                term_buf[step] = torch.as_tensor(terminated, device=device)
                trunc_buf[step] = torch.as_tensor(truncated, device=device)
                if truncated.any() and "terminal_obs" in info:
                    tobs = obs_norm.normalize(np.asarray(info["terminal_obs"]))
                    truncval_buf[step] = policy.get_value(
                        torch.as_tensor(tobs, dtype=torch.float32,
                                        device=device))  # (N, G)
                else:
                    truncval_buf[step] = 0.0

                running_ep_r += reward
                running_ep_len += 1
                total_steps += n_envs

                if np.any(done):
                    for i in np.where(done)[0]:
                        ep_rewards.append(float(running_ep_r[i]))
                        ep_lengths.append(int(running_ep_len[i]))
                        running_ep_r[i] = 0.0
                        running_ep_len[i] = 0

                for k, v in info.get("reward_components", {}).items():
                    comp_queues.setdefault(k, deque(maxlen=200)).append(float(v))

                if video_recording:
                    frame = env.render_frame()
                    if frame is not None:
                        video_frames.append(frame)
                    if len(video_frames) >= video_n_frames:
                        try:
                            logger.log_video("train/rollout", video_frames,
                                             step=total_steps, fps=video_fps)
                        except Exception as e:
                            print(f"[ppo-mc] video log failed: {e}")
                        video_recording = False
                        video_frames = []

                obs_norm.update(next_obs)
                obs_t = torch.as_tensor(obs_norm.normalize(next_obs),
                                        dtype=torch.float32, device=device)

        # ── Per-group reward normalisation ──
        rew_norm_buf = torch.empty_like(rew_buf)
        for g in range(n_groups):
            rg = rew_buf[..., g]
            rn = ret_norms[g]
            rn.update(rg.reshape(-1).cpu().numpy(),
                      done_buf.reshape(-1).cpu().numpy())
            rew_norm_buf[..., g] = rg / max(1e-8, float(rn.rms.std))

        # ── Per-group GAE ──
        with torch.no_grad():
            next_value = policy.get_value(obs_t)  # (N, G)

        advantages = torch.zeros_like(rew_norm_buf)  # (T, N, G)
        gae = torch.zeros(n_envs, n_groups, device=device)
        for t in reversed(range(n_steps)):
            next_val = next_value if t == n_steps - 1 else val_buf[t + 1]
            boot = torch.where((trunc_buf[t] > 0).unsqueeze(-1),
                               truncval_buf[t], next_val)
            nonterminal = (1.0 - term_buf[t]).unsqueeze(-1)  # (N, 1)
            notdone = (1.0 - done_buf[t]).unsqueeze(-1)
            delta = (rew_norm_buf[t]
                     + config.gamma * boot * nonterminal
                     - val_buf[t])
            gae = delta + config.gamma * config.gae_lambda * notdone * gae
            advantages[t] = gae
        returns = advantages + val_buf  # (T, N, G) per-group value targets

        # Per-group advantage normalisation, then weighted aggregation into
        # the scalar advantage that drives the policy surrogate.
        adv_flat = advantages.reshape(-1, n_groups)
        adv_norm = (adv_flat - adv_flat.mean(0, keepdim=True)) \
            / (adv_flat.std(0, keepdim=True) + 1e-8)
        agg_adv = (adv_norm * gw).sum(-1)  # (T*N,)
        # Re-standardize the aggregated advantage so its scale doesn't depend
        # on n_groups / group_weights (a weighted sum of unit-std groups has
        # std ~sqrt(Σw²)); keeps the effective policy step on par with
        # single-critic PPO, which normalizes its advantage the same way.
        agg_adv = (agg_adv - agg_adv.mean()) / (agg_adv.std() + 1e-8)

        # Per-group explained variance (diagnostic).
        with torch.no_grad():
            ev_per_group = []
            yt = returns.reshape(-1, n_groups).cpu().numpy()
            yp = val_buf.reshape(-1, n_groups).cpu().numpy()
            for g in range(n_groups):
                var_y = float(np.var(yt[:, g]))
                ev_per_group.append(
                    float(1.0 - np.var(yt[:, g] - yp[:, g]) / (var_y + 1e-8)))

        # Exploration std = learnable actor_log_std (entropy bonus + PG),
        # not a fixed wall-clock schedule.

        # ── Flatten ──
        b_obs = obs_buf.reshape(n_steps * n_envs, obs_dim)
        b_act = act_buf.reshape(n_steps * n_envs, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = agg_adv
        b_ret = returns.reshape(n_steps * n_envs, n_groups)
        b_val = val_buf.reshape(n_steps * n_envs, n_groups)

        n_samples = b_obs.shape[0]
        mb_size = max(1, n_samples // 4)

        approx_kl = 0.0
        clip_frac = 0.0
        policy_loss_val = 0.0
        value_loss_val = 0.0
        entropy_val = 0.0
        kl_too_big = False

        for epoch in range(config.n_epochs):
            indices = torch.randperm(n_samples, device=device)
            epoch_kls = []
            for start in range(0, n_samples, mb_size):
                end = start + mb_size
                mb_idx = indices[start:end]
                mb_obs = b_obs[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logp = b_logp[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]          # (mb, G)
                mb_val_old = b_val[mb_idx]      # (mb, G)

                values, new_log_prob, entropy = policy.evaluate(mb_obs, mb_act)

                ratio = (new_log_prob - mb_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - config.clip_range,
                                    1 + config.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if use_value_clipping:
                    v_clip = mb_val_old + (values - mb_val_old).clamp(
                        -config.clip_range, config.clip_range
                    )
                    v_loss_unclipped = (values - mb_ret).pow(2)
                    v_loss_clipped = (v_clip - mb_ret).pow(2)
                    value_loss = 0.5 * torch.max(
                        v_loss_unclipped, v_loss_clipped).mean()
                else:
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
                    kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                    approx_kl = float(kl)
                    clip_frac = float((log_ratio.abs()
                                       > math.log(1 + config.clip_range)
                                       ).float().mean())
                    epoch_kls.append(approx_kl)
                policy_loss_val = float(policy_loss.detach())
                value_loss_val = float(value_loss.detach())
                entropy_val = float(-entropy_loss.detach())

            mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if desired_kl > 0 and mean_kl > 2.0 * desired_kl:
                kl_too_big = True
                break

        if adaptive_lr and desired_kl > 0:
            if approx_kl > 2.0 * desired_kl:
                lr = max(min_lr, lr / 1.5)
            elif approx_kl < 0.5 * desired_kl:
                lr = min(max_lr, lr * 1.5)
            for grp in optimizer.param_groups:
                grp["lr"] = lr

        # ── Logging ──
        if iteration % 10 == 0 or len(ep_rewards) > 0:
            mr = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            ml = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            sr = float(np.std(ep_rewards)) if len(ep_rewards) > 1 else 0.0
            mean_robot_z = (float(obs_buf[:, :, 0].mean())
                            * float(obs_norm.std[0])
                            + float(obs_norm.mean[0]))
            metrics = {
                "mean_reward": mr, "std_reward": sr, "mean_length": ml,
                "mean_robot_z": mean_robot_z,
                "policy_loss": policy_loss_val,
                "value_loss": value_loss_val,
                "entropy": entropy_val,
                "approx_kl": approx_kl,
                "clip_fraction": clip_frac,
                "learning_rate": lr,
                "kl_early_stop": float(kl_too_big),
            }
            for g, name in enumerate(group_names):
                metrics[f"explained_variance_{name}"] = ev_per_group[g]
            for k, q in comp_queues.items():
                if q:
                    metrics[f"rewards/{k}"] = float(np.mean(q))

            ev_str = " ".join(f"{n}={ev_per_group[g]:.2f}"
                              for g, n in enumerate(group_names))
            print(f"[ppo-mc|{curriculum_stage or 'full'}] "
                  f"Iter {iteration:5d} | Steps {total_steps:12,d} | "
                  f"R̄={mr:7.2f} (σ={sr:.2f}) | L̄={ml:6.1f} | "
                  f"z̄={mean_robot_z:.3f} | π={policy_loss_val:.4f} "
                  f"V={value_loss_val:.4f} KL={approx_kl:.4f} lr={lr:.1e} | "
                  f"ev[{ev_str}]" + ("  [KL-stop]" if kl_too_big else ""))
            _log_metrics(logger, metrics, total_steps)

        if iteration % 100 == 0 and iteration > 0:
            save_ppo_checkpoint(policy, optimizer, total_steps, phase,
                                curriculum_stage,
                                checkpoint_dir=checkpoint_dir,
                                obs_norm=obs_norm, ret_norm=ret_norms[0])

    save_ppo_checkpoint(policy, optimizer, total_steps, phase,
                        curriculum_stage, checkpoint_dir=checkpoint_dir,
                        obs_norm=obs_norm, ret_norm=ret_norms[0])
    return policy


# ─── Single-env PPO (debug / portability path) ─────────────────────────


def train_ppo(env, policy, config, logger=None, phase="phase1",
              curriculum_stage=None, checkpoint_dir: str = "checkpoints",
              desired_kl: float = 0.01,
              use_value_clipping: bool = True,
              adaptive_lr: bool = True,
              device: Optional[torch.device] = None):
    """Single-env PPO. Calls vec-loop logic with num_envs=1 wrapper-style."""
    # The vec version assumes batched obs. For single-env we re-implement
    # the loop locally to avoid forcing an env wrapper. This is the slow
    # debug path; for real training use the vec env (see train_ppo_vec).
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if policy is None:
        policy = PPOActorCritic(config.obs_dim, config.act_dim)
    policy = policy.to(device)
    lr = float(config.learning_rate)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    obs_norm = RunningMeanStd(shape=(config.obs_dim,))
    ret_norm = ReturnNormalizer(gamma=config.gamma)

    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    comp_queues: dict = {}
    ep_components: dict = {}

    n_steps = int(config.n_steps)
    obs_dim = int(config.obs_dim)
    act_dim = int(config.act_dim)
    total_steps = 0
    num_iterations = int(config.total_timesteps) // n_steps

    print(f"\n[PPO|single-env] phase={phase} stage={curriculum_stage or 'full'}"
          f"  total={config.total_timesteps:,}  device={device}")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs[0]
    obs_norm.update(obs)
    obs_t = torch.as_tensor(obs_norm.normalize(obs),
                            dtype=torch.float32, device=device)

    episode_reward = 0.0
    episode_length = 0

    for iteration in range(num_iterations):
        obs_buf = torch.zeros(n_steps, obs_dim, device=device)
        act_buf = torch.zeros(n_steps, act_dim, device=device)
        logp_buf = torch.zeros(n_steps, device=device)
        rew_buf = torch.zeros(n_steps, device=device)
        val_buf = torch.zeros(n_steps, device=device)
        done_buf = torch.zeros(n_steps, device=device)
        term_buf = torch.zeros(n_steps, device=device)
        trunc_buf = torch.zeros(n_steps, device=device)
        truncval_buf = torch.zeros(n_steps, device=device)

        with torch.no_grad():
            for step in range(n_steps):
                action, log_prob, _ = policy.act(obs_t.unsqueeze(0))
                value = policy.get_value(obs_t.unsqueeze(0))

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, done, info = env.step(action_np)
                if isinstance(next_obs, dict):
                    next_obs = next_obs[0]

                obs_buf[step] = obs_t
                act_buf[step] = action.squeeze(0)
                logp_buf[step] = log_prob.squeeze(0)
                rew_buf[step] = float(reward)
                val_buf[step] = value.squeeze(0)
                done_buf[step] = float(done)

                # Time-limit split for correct bootstrapping (see vec path).
                terminated = float(np.asarray(info.get("terminated", done)).any())
                truncated = float(np.asarray(info.get("truncated", 0.0)).any())
                term_buf[step] = terminated
                trunc_buf[step] = truncated
                if truncated and "terminal_obs" in info:
                    tobs = np.asarray(info["terminal_obs"])
                    if tobs.ndim > 1:
                        tobs = tobs[0]
                    tnorm = obs_norm.normalize(tobs)
                    truncval_buf[step] = policy.get_value(
                        torch.as_tensor(tnorm, dtype=torch.float32,
                                        device=device).unsqueeze(0)).squeeze(0)

                episode_reward += float(reward)
                episode_length += 1
                total_steps += 1

                for k, v in info.get("reward_components", {}).items():
                    if k != "total":
                        ep_components[k] = ep_components.get(k, 0.0) + float(v)

                if done:
                    ep_rewards.append(episode_reward)
                    ep_lengths.append(episode_length)
                    L = max(1, episode_length)
                    for k, v in ep_components.items():
                        comp_queues.setdefault(k, deque(maxlen=100)).append(v / L)
                    ep_components = {}
                    episode_reward = 0.0
                    episode_length = 0
                    # NOTE: env.step() already auto-resets internally and
                    # returns the fresh obs, so we do NOT reset again here.

                obs_norm.update(next_obs)
                obs_t = torch.as_tensor(obs_norm.normalize(next_obs),
                                        dtype=torch.float32, device=device)

        # Reward normalisation
        rew_np = rew_buf.cpu().numpy()
        done_np = done_buf.cpu().numpy()
        ret_norm.update(rew_np, done_np)
        rew_buf_n = rew_buf / max(1e-8, float(ret_norm.rms.std))

        with torch.no_grad():
            next_value = policy.get_value(obs_t.unsqueeze(0)).squeeze(0)

        advantages = torch.zeros(n_steps, device=device)
        gae = 0.0
        for t in reversed(range(n_steps)):
            next_val = next_value if t == n_steps - 1 else val_buf[t + 1]
            # Bootstrap from the terminal-obs value on truncation; mask only
            # on a true terminal; reset the trace on any episode end. Uses the
            # CURRENT step's flags (off-by-one fix).
            boot = truncval_buf[t] if float(trunc_buf[t]) > 0 else next_val
            nonterminal = 1.0 - float(term_buf[t])
            notdone = 1.0 - float(done_buf[t])
            delta = (rew_buf_n[t] + config.gamma * boot * nonterminal
                     - val_buf[t])
            gae = delta + config.gamma * config.gae_lambda * notdone * gae
            advantages[t] = gae
        returns = advantages + val_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            y_true = returns.cpu().numpy()
            y_pred = val_buf.cpu().numpy()
            var_y = float(np.var(y_true))
            explained_var = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))

        # Exploration std = learnable actor_log_std (entropy bonus + PG).

        approx_kl = 0.0
        clip_frac = 0.0
        policy_loss_val = 0.0
        value_loss_val = 0.0
        entropy_val = 0.0
        kl_too_big = False

        for epoch in range(config.n_epochs):
            indices = torch.randperm(n_steps, device=device)
            mb_size = max(1, n_steps // 4)
            epoch_kls = []
            for start in range(0, n_steps, mb_size):
                mb_idx = indices[start:start + mb_size]
                mb_obs = obs_buf[mb_idx]
                mb_act = act_buf[mb_idx]
                mb_logp = logp_buf[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_val_old = val_buf[mb_idx]

                values, new_log_prob, entropy = policy.evaluate(mb_obs, mb_act)
                ratio = (new_log_prob - mb_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - config.clip_range,
                                    1 + config.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                if use_value_clipping:
                    v_clip = mb_val_old + (values - mb_val_old).clamp(
                        -config.clip_range, config.clip_range
                    )
                    value_loss = 0.5 * torch.max(
                        (values - mb_ret).pow(2),
                        (v_clip - mb_ret).pow(2)).mean()
                else:
                    value_loss = 0.5 * (values - mb_ret).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = (policy_loss + config.vf_coef * value_loss
                        + config.entropy_coef * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),
                                         config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_prob - mb_logp
                    kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                    approx_kl = float(kl)
                    clip_frac = float((log_ratio.abs()
                                       > math.log(1 + config.clip_range)
                                       ).float().mean())
                    epoch_kls.append(approx_kl)
                policy_loss_val = float(policy_loss.detach())
                value_loss_val = float(value_loss.detach())
                entropy_val = float(-entropy_loss.detach())

            mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if desired_kl > 0 and mean_kl > 2.0 * desired_kl:
                kl_too_big = True
                break

        if adaptive_lr and desired_kl > 0:
            if approx_kl > 2.0 * desired_kl:
                lr = max(1e-5, lr / 1.5)
            elif approx_kl < 0.5 * desired_kl:
                lr = min(1e-2, lr * 1.5)
            for g in optimizer.param_groups:
                g["lr"] = lr

        if iteration % 10 == 0 or len(ep_rewards) > 0:
            mr = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            ml = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            sr = float(np.std(ep_rewards)) if len(ep_rewards) > 1 else 0.0
            mean_robot_z = (float(obs_buf[:, 0].mean())
                            * float(obs_norm.std[0])
                            + float(obs_norm.mean[0]))
            metrics = {
                "mean_reward": mr, "std_reward": sr, "mean_length": ml,
                "mean_robot_z": mean_robot_z,
                "policy_loss": policy_loss_val,
                "value_loss": value_loss_val,
                "entropy": entropy_val,
                "explained_variance": explained_var,
                "approx_kl": approx_kl,
                "clip_fraction": clip_frac,
                "learning_rate": lr,
                "kl_early_stop": float(kl_too_big),
            }
            for k, q in comp_queues.items():
                if q:
                    metrics[f"rewards/{k}"] = float(np.mean(q))
            if iteration % 10 == 0:
                print(f"[ppo|{curriculum_stage or 'full'}] "
                      f"Iter {iteration:5d} | Steps {total_steps:10,d} | "
                      f"R̄={mr:7.2f} (σ={sr:.2f}) | L̄={ml:5.1f} | "
                      f"z̄={mean_robot_z:.3f} | "
                      f"π={policy_loss_val:.4f} KL={approx_kl:.4f} "
                      f"lr={lr:.1e} ev={explained_var:.3f}")
            _log_metrics(logger, metrics, total_steps)

        if iteration % 100 == 0 and iteration > 0:
            save_ppo_checkpoint(policy, optimizer, total_steps, phase,
                                curriculum_stage,
                                checkpoint_dir=checkpoint_dir,
                                obs_norm=obs_norm, ret_norm=ret_norm)

    save_ppo_checkpoint(policy, optimizer, total_steps, phase,
                        curriculum_stage, checkpoint_dir=checkpoint_dir,
                        obs_norm=obs_norm, ret_norm=ret_norm)
    return policy


# ─── helpers ───────────────────────────────────────────────────────────


def _log_metrics(logger, metrics: dict, step: int) -> None:
    if logger is None:
        return
    prefixed = {(k if "/" in k else f"train/{k}"): v
                for k, v in metrics.items()}
    logger.log_scalars(prefixed, step)
