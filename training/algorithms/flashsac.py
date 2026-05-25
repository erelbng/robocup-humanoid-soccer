"""FlashSAC: fast off-policy SAC for Genesis vec-env training.

This is a pragmatic FlashSAC that captures the key engineering wins of
Holiday-Robot/FlashSAC without porting the categorical distributional Q
or the bespoke UnitRMSNorm blocks (which add substantial code for
secondary gains on this task). Specifically we provide:

  * Twin Q (clipped double Q-learning) — variance reduction.
  * GPU-resident replay buffer that ingests N transitions per step from
    the Genesis vec env.
  * Squashed-Gaussian actor with proper tanh-correction in log_prob.
  * Polyak (EMA) target updates with tunable τ.
  * Automatic temperature tuning (Haarnoja et al. SAC v2): minimise
    α · (−H(π) − H_target). H_target defaults to −act_dim, the standard
    heuristic.
  * UTD (Update-To-Data) ratio: number of gradient steps per env step.
    For Genesis n_envs=256 we collect 256 transitions per env-step, so
    a single gradient step per env-step is already a healthy UTD of
    ~1/256 per transition. Configurable.
  * Per-component reward metrics threaded through to the logger so the
    user can see exactly which reward terms are doing the work.

The off-policy nature means we don't need long rollouts — we collect a
small number of steps, then do `gradient_steps` updates from the
replay buffer, then repeat. This is the inverse of PPO's collect-a-lot,
update-once cycle.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from training.algorithms.networks import (
    SACActor, TwinQNetwork, soft_update,
)
from training.algorithms.replay_buffer import GPUReplayBuffer
from training.normalizers import RunningMeanStd


# ─── checkpoint I/O ────────────────────────────────────────────────────


def save_sac_checkpoint(actor, critic, target_critic, log_alpha,
                        actor_opt, critic_opt, alpha_opt, step, phase,
                        stage=None, path=None, obs_norm=None,
                        checkpoint_dir: str = "checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if path is None:
        suffix = f"_{stage}" if stage else ""
        path = os.path.join(checkpoint_dir,
                            f"{phase}{suffix}_flashsac_step{step}.pt")
    ckpt = {
        "step": step, "phase": phase, "stage": stage,
        "algorithm": "flashsac",
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "target_critic_state_dict": target_critic.state_dict(),
        "log_alpha": log_alpha.detach().clone(),
        "actor_opt_state": actor_opt.state_dict(),
        "critic_opt_state": critic_opt.state_dict(),
        "alpha_opt_state": alpha_opt.state_dict(),
    }
    if obs_norm is not None and hasattr(obs_norm, "state_dict"):
        ckpt["obs_norm"] = obs_norm.state_dict()
    torch.save(ckpt, path)
    print(f"[flashsac] checkpoint → {path}")
    return path


# ─── main loop ─────────────────────────────────────────────────────────


def train_flashsac_vec(
    env,
    policy=None,  # not used; FlashSAC builds its own nets
    config=None,
    logger=None,
    phase: str = "phase1",
    curriculum_stage: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    # SAC-specific hyperparams (sensible defaults for Genesis humanoid):
    buffer_capacity: int = 200_000,   # 200k × 190 floats ≈ 152 MB on CPU
    batch_size: int = 512,
    actor_hidden=(512, 256, 128),
    critic_hidden=(512, 256, 128),
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    tau: float = 0.005,
    init_alpha: float = 0.2,
    target_entropy: Optional[float] = None,  # default: −act_dim
    warmup_steps: int = 1024,  # use random actions until buffer has this many
    learning_starts: int = 5_000,  # first gradient update after this many env-steps
    gradient_steps: int = 1,  # updates per env-step
    # Joint position targets only need ~1.5 rad of range from the default
    # pose to cover everything a walking humanoid does. The old default
    # of π let the policy slam joints to ±π, flipping the robot before
    # it could learn anything. Override to π for full-range tasks.
    action_scale: Optional[float] = None,  # default: 1.5 rad
    video_frequency: int = 50,  # log a video clip every N trainer iterations
    video_n_frames: int = 300,
    video_fps: int = 30,
    device: Optional[torch.device] = None,
):
    """FlashSAC trainer for a Genesis-style vec env."""
    if config is None:
        raise ValueError("train_flashsac_vec requires a config")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = int(config.obs_dim)
    act_dim = int(config.act_dim)
    n_envs = int(env.num_envs)
    if action_scale is None:
        action_scale = 1.5  # joint-range-friendly default
    if target_entropy is None:
        target_entropy = -float(act_dim)

    # Auto-scale gradient_steps with n_envs so effective per-transition
    # UTD stays roughly constant regardless of parallelism.
    # Reference: gradient_steps calibrated for n_envs=256.
    # With n_envs=4096 we collect 16× more transitions per env-step, so
    # scale up proportionally. Only increases — never reduces a value the
    # caller explicitly set higher. Cap at 20 to bound wall-clock cost.
    _auto_gs = max(1, n_envs // 256)
    if _auto_gs > gradient_steps:
        gradient_steps = min(20, _auto_gs)
        print(f"   [auto] gradient_steps → {gradient_steps} "
              f"(n_envs={n_envs} / 256)")

    # ── Build networks ──
    actor = SACActor(obs_dim, act_dim, hidden_dims=actor_hidden,
                     action_scale=action_scale).to(device)
    critic = TwinQNetwork(obs_dim, act_dim,
                          hidden_dims=critic_hidden).to(device)
    target_critic = TwinQNetwork(obs_dim, act_dim,
                                 hidden_dims=critic_hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())
    for p in target_critic.parameters():
        p.requires_grad_(False)

    # torch.compile gives 2-3× throughput on CUDA with PyTorch ≥ 2.0.
    # We compile after moving to device so the tracer sees the right dtype.
    _compiled = False
    if device.type == "cuda" and int(torch.__version__.split(".")[0]) >= 2:
        try:
            actor = torch.compile(actor)
            critic = torch.compile(critic)
            target_critic = torch.compile(target_critic)
            _compiled = True
            print("   [compile] torch.compile enabled")
        except Exception as _ce:
            print(f"   [compile] torch.compile unavailable: {_ce}")

    # Learnable log α with optimizer
    log_alpha = torch.tensor(math.log(init_alpha), dtype=torch.float32,
                             device=device, requires_grad=True)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr, eps=1e-5)
    alpha_opt = torch.optim.Adam([log_alpha], lr=alpha_lr, eps=1e-5)

    # ── Replay buffer ──
    buffer = GPUReplayBuffer(buffer_capacity, obs_dim, act_dim, device)

    # ── Obs normaliser (running stats, updated forever) ──
    # We keep a numpy RunningMeanStd (Welford) for online updates, plus
    # mirror tensors on `device` that are refreshed in-place after each
    # env step. This avoids allocating fresh torch tensors twice per
    # gradient step, which on CPU adds non-trivial Python overhead.
    obs_norm = RunningMeanStd(shape=(obs_dim,))
    obs_mean_t = torch.zeros(obs_dim, dtype=torch.float32, device=device)
    obs_std_t = torch.ones(obs_dim, dtype=torch.float32, device=device)

    def _refresh_norm_cache():
        obs_mean_t.copy_(torch.from_numpy(
            obs_norm.mean.astype(np.float32)).to(device))
        obs_std_t.copy_(torch.from_numpy(
            (np.sqrt(obs_norm.var) + 1e-8).astype(np.float32)).to(device))

    def _norm(t: torch.Tensor) -> torch.Tensor:
        return (t - obs_mean_t) / obs_std_t

    # ── Rolling metrics ──
    ep_rewards = deque(maxlen=200)
    ep_lengths = deque(maxlen=200)
    running_ep_r = np.zeros(n_envs, dtype=np.float32)
    running_ep_len = np.zeros(n_envs, dtype=np.int64)
    comp_queues: dict = {}

    total_env_steps = 0
    total_grad_steps = 0
    # FlashSAC training horizon is wall-clock-bounded by total_timesteps too
    target_env_steps = int(config.total_timesteps)

    # Video capture state — same scheme as PPO: trigger a clip every
    # `video_frequency` iterations, fill it during normal rollout steps
    # by rendering env 0, push to logger when full.
    video_recording = False
    video_frames: list = []
    video_supported = hasattr(env, "render_frame")
    if logger is None or video_frequency <= 0:
        video_supported = False

    print(f"\n{'='*60}")
    print(f" [FlashSAC] phase={phase} stage={curriculum_stage or 'full'}")
    print(f"   n_envs={n_envs}  buffer={buffer_capacity:,}  "
          f"batch={batch_size}")
    print(f"   actor_lr={actor_lr}  critic_lr={critic_lr}  τ={tau}")
    print(f"   target_ent={target_entropy:.2f}  α0={init_alpha}  "
          f"action_scale={action_scale:.3f}")
    print(f"   warmup={warmup_steps}  learn_after={learning_starts}  "
          f"grad_steps/env_step={gradient_steps}")
    print(f"   total_env_steps target={target_env_steps:,}  device={device}")
    print(f"   video: {'every ' + str(video_frequency) + ' iters, ' + str(video_n_frames) + ' frames' if video_supported else 'disabled'}")
    print(f"{'='*60}\n")

    # ── Roll the env ──
    obs = env.reset()
    obs_norm.update(obs)
    _refresh_norm_cache()

    iteration = 0
    log_interval = 10
    save_interval = 100

    while total_env_steps < target_env_steps:
        # Trigger a new video clip every `video_frequency` iterations.
        # We do this BEFORE the env step so the first frame captures the
        # state after the previous iter's reset/transition.
        if (video_supported and not video_recording
                and iteration > 0
                and iteration % video_frequency == 0):
            video_recording = True
            video_frames = []

        # ── Step the env once across all sub-envs ──
        with torch.no_grad():
            if buffer.size < warmup_steps:
                # Uniform exploration during warmup
                action = np.random.uniform(
                    -action_scale, action_scale,
                    size=(n_envs, act_dim)).astype(np.float32)
            else:
                # Use cached device tensors; one float32 array→tensor
                # copy of `obs` per env step instead of two per grad step.
                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=device)
                obs_t = _norm(obs_t)
                action_t, _ = actor(obs_t, deterministic=False)
                action = action_t.cpu().numpy()

        next_obs, reward, done, info = env.step(action)

        # Capture a frame after the env step if recording
        if video_recording:
            frame = env.render_frame()
            if frame is not None:
                video_frames.append(frame)
            if len(video_frames) >= video_n_frames:
                try:
                    logger.log_video("train/rollout", video_frames,
                                     step=total_env_steps, fps=video_fps)
                    print(f"[flashsac] logged {len(video_frames)}-frame video "
                          f"at env_step {total_env_steps:,}")
                except Exception as e:
                    print(f"[flashsac] video log failed: {e}")
                video_recording = False
                video_frames = []
        # Genesis vec env auto-resets finished envs; next_obs[i] when
        # done[i]=1 is from the freshly-reset episode. The done flag
        # zeroes the bootstrap, so storing this transition is still
        # correct (we just don't bootstrap across reset).
        buffer.add_batch(obs, action, reward, next_obs, done)

        running_ep_r += reward
        running_ep_len += 1
        total_env_steps += n_envs
        done_arr = np.asarray(done)
        if done_arr.any():
            for i in np.where(done_arr)[0]:
                ep_rewards.append(float(running_ep_r[i]))
                ep_lengths.append(int(running_ep_len[i]))
                running_ep_r[i] = 0.0
                running_ep_len[i] = 0

        for k, v in info.get("reward_components", {}).items():
            comp_queues.setdefault(k, deque(maxlen=200)).append(float(v))

        obs_norm.update(next_obs)
        # Refreshing cached device tensors every step adds two host→device
        # transfers per env step. Amortise to every 10 iterations — the
        # normaliser stats change slowly enough that stale values are fine.
        if iteration % 10 == 0:
            _refresh_norm_cache()
        obs = next_obs

        # ── Gradient updates (off-policy) ──
        critic_loss_val = 0.0
        actor_loss_val = 0.0
        alpha_loss_val = 0.0
        alpha_val = float(log_alpha.exp().detach())
        mean_q_val = 0.0
        target_q_val = 0.0
        policy_entropy = 0.0

        if (total_env_steps >= learning_starts
                and buffer.size >= batch_size):
            for _ in range(gradient_steps):
                batch = buffer.sample(batch_size)
                # Normalise via cached device tensors (no allocation).
                b_obs = _norm(batch["obs"])
                b_next = _norm(batch["next_obs"])
                b_act = batch["act"]
                b_rew = batch["rew"]
                b_done = batch["done"]

                # ── Critic update ──
                with torch.no_grad():
                    next_action, next_logp = actor(b_next,
                                                   deterministic=False)
                    target_q1, target_q2 = target_critic(b_next, next_action)
                    target_q = torch.min(target_q1, target_q2)
                    alpha = log_alpha.exp()
                    target = b_rew + (1.0 - b_done) * config.gamma * (
                        target_q - alpha * next_logp
                    )

                q1, q2 = critic(b_obs, b_act)
                critic_loss = 0.5 * ((q1 - target).pow(2).mean()
                                     + (q2 - target).pow(2).mean())
                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                critic_opt.step()

                # ── Actor update ──
                # Freeze critic params during the actor backward pass —
                # we still need the dq/d(new_action) gradient to flow
                # through, but skipping critic-param grads is ~10% faster
                # on CPU. Mirrors FlashSAC's pattern (requires_grad_(False)).
                for p in critic.parameters():
                    p.requires_grad_(False)

                new_action, new_logp = actor(b_obs, deterministic=False)
                q1_pi, q2_pi = critic(b_obs, new_action)
                q_pi = torch.min(q1_pi, q2_pi)
                alpha = log_alpha.exp().detach()
                actor_loss = (alpha * new_logp - q_pi).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                actor_opt.step()

                for p in critic.parameters():
                    p.requires_grad_(True)

                # ── Temperature (α) update ──
                # log_prob of the SAME action (re-use), detached
                with torch.no_grad():
                    logp_detached = new_logp.detach()
                alpha_loss = -(log_alpha
                               * (logp_detached + target_entropy)).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                # ── EMA target update ──
                soft_update(target_critic, critic, tau)

                total_grad_steps += 1

                critic_loss_val = float(critic_loss.detach())
                actor_loss_val = float(actor_loss.detach())
                alpha_loss_val = float(alpha_loss.detach())
                alpha_val = float(log_alpha.exp().detach())
                mean_q_val = float(q1.mean().detach())
                target_q_val = float(target.mean().detach())
                policy_entropy = float(-new_logp.mean().detach())

        # ── Logging ──
        iteration += 1
        if iteration % log_interval == 0:
            mr = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            ml = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            sr = float(np.std(ep_rewards)) if len(ep_rewards) > 1 else 0.0
            # Robot z lives in obs[2] (raw obs, not normalised)
            mean_robot_z = float(np.mean(obs[:, 2]))
            metrics = {
                "mean_reward": mr, "std_reward": sr, "mean_length": ml,
                "mean_robot_z": mean_robot_z,
                "critic_loss": critic_loss_val,
                "actor_loss": actor_loss_val,
                "alpha_loss": alpha_loss_val,
                "alpha": alpha_val,
                "policy_entropy": policy_entropy,
                "mean_q": mean_q_val,
                "target_q": target_q_val,
                "buffer_size": float(buffer.size),
                "env_steps": float(total_env_steps),
                "grad_steps": float(total_grad_steps),
            }
            for k, q in comp_queues.items():
                if q:
                    metrics[f"rewards/{k}"] = float(np.mean(q))

            print(f"[flashsac|{curriculum_stage or 'full'}|vec] "
                  f"Iter {iteration:5d} | EnvStep {total_env_steps:12,d} | "
                  f"R̄={mr:7.2f} (σ={sr:.2f}) | L̄={ml:6.1f} | "
                  f"z̄={mean_robot_z:.3f} | "
                  f"Q̂={mean_q_val:.2f} α={alpha_val:.3f} "
                  f"H={policy_entropy:.2f} buf={buffer.size:,}")
            _log_metrics(logger, metrics, total_env_steps)

        if iteration % save_interval == 0 and iteration > 0:
            save_sac_checkpoint(actor, critic, target_critic, log_alpha,
                                actor_opt, critic_opt, alpha_opt,
                                total_env_steps, phase, curriculum_stage,
                                obs_norm=obs_norm,
                                checkpoint_dir=checkpoint_dir)

    save_sac_checkpoint(actor, critic, target_critic, log_alpha,
                        actor_opt, critic_opt, alpha_opt,
                        total_env_steps, phase, curriculum_stage,
                        obs_norm=obs_norm, checkpoint_dir=checkpoint_dir)
    return actor


# ─── helpers ───────────────────────────────────────────────────────────


def _log_metrics(logger, metrics: dict, step: int) -> None:
    if logger is None:
        return
    prefixed = {(k if "/" in k else f"train/{k}"): v
                for k, v in metrics.items()}
    logger.log_scalars(prefixed, step)
