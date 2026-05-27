"""Student policy distillation from a frozen teacher (DAgger-lite).

The teacher policy is a PPO actor-critic trained with PRIVILEGED OBS
(true DR sample, ground friction, mass scaling, …). The student must
match the teacher's behaviour using only PROPRIOCEPTIVE OBS — the same
signal that a real robot has access to. This is the standard sim-to-real
recipe (RMA / Concurrent Teacher-Student / DreamWaQ).

Algorithm: behaviour cloning with on-policy rollouts (a stripped-down
DAgger):

  loop:
    1. Roll the env forward N steps. Use either the teacher's
       deterministic action OR the student's deterministic action,
       mixed with probability `beta` (DAgger β-schedule: starts at 1.0
       meaning "always teacher", decays to 0 meaning "always student").
       β=0.5 throughout is a robust default.
    2. For every step collected, evaluate BOTH:
         * Student obs (proprio only) → student forward pass.
         * Teacher obs (proprio + privileged) → teacher's mean action.
    3. Loss = MSE(student_action, teacher_action).
    4. Optimize student with Adam.

The env is the SAME skill env in both passes — we just need it to
produce *both* obs flavours. SkillEnv supports this directly: the
teacher needs an env with `include_privileged=True`; the student
network sees `obs[:, :proprio_dim]` (the env's privileged channel sits
at the end so slicing off the tail gives proprio-only).

Why behaviour cloning and not just running the teacher? Because
without DR-aware state estimation, the teacher's policy is uncalibrated
to noisy proprio. The student learns to be robust to the same DR + obs
noise the teacher saw — exactly what real-robot deployment needs.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from training.algorithms.networks import PPOActorCritic
from training.normalizers import RunningMeanStd


# ─── checkpoint I/O ────────────────────────────────────────────────────


def save_student_checkpoint(student, optimizer, step, skill,
                             obs_norm=None,
                             checkpoint_dir: str = "checkpoints",
                             path: Optional[str] = None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if path is None:
        path = os.path.join(checkpoint_dir,
                             f"student_{skill}_step{step}.pt")
    ckpt = {
        "step": step, "skill": skill,
        "algorithm": "student_bc",
        "policy_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if obs_norm is not None and hasattr(obs_norm, "state_dict"):
        ckpt["obs_norm"] = obs_norm.state_dict()
    torch.save(ckpt, path)
    print(f"[distill] checkpoint → {path}")
    return path


# ─── main loop ─────────────────────────────────────────────────────────


def train_student(
    env_teacher,                          # SkillEnv with include_privileged=True
    teacher_policy: PPOActorCritic,       # frozen, eval mode
    student_obs_dim: int,                 # proprio-only obs width
    act_dim: int,
    *,
    total_env_steps: int = 50_000_000,
    n_steps: int = 64,                    # rollout horizon per iteration
    learning_rate: float = 3e-4,
    batch_size: int = 8192,
    n_epochs: int = 3,
    beta_schedule=lambda frac: max(0.0, 1.0 - frac),  # DAgger β
    logger=None,
    skill: str = "walk",
    checkpoint_dir: str = "checkpoints",
    device: Optional[torch.device] = None,
):
    """Train a student that imitates `teacher_policy` from proprio-only obs.

    `env_teacher` is the SkillEnv with `include_privileged=True`. The
    teacher's obs = env.reset() / env.step() directly. The student's
    obs is the leading `student_obs_dim` of the same obs (the env
    appends the 8-dim privileged tail).

    Returns the trained student `PPOActorCritic` (same architecture as
    the teacher, just narrower input).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_policy = teacher_policy.to(device).eval()
    for p in teacher_policy.parameters():
        p.requires_grad_(False)

    # Student shares the actor-critic architecture but ingests proprio
    # only. We only use the actor head during distillation; the critic
    # head trains too (free freebie) so future fine-tuning with PPO can
    # pick up where BC left off.
    student = PPOActorCritic(student_obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(student.parameters(),
                                  lr=float(learning_rate), eps=1e-5)

    obs_norm = RunningMeanStd(shape=(student_obs_dim,))

    n_envs = int(env_teacher.num_envs)
    full_obs_dim = int(env_teacher.obs_dim)
    priv_dim = full_obs_dim - student_obs_dim
    assert priv_dim > 0, (
        f"distillation expects an env with privileged obs (env obs_dim "
        f"{full_obs_dim}, student dim {student_obs_dim}); did you build "
        f"the env with include_privileged=True?")

    print(f"\n[distill] skill={skill}  n_envs={n_envs}  "
          f"obs_full={full_obs_dim}  obs_student={student_obs_dim}  "
          f"priv={priv_dim}  device={device}")

    obs_full = env_teacher.reset()                  # (N, full_obs_dim)
    obs_norm.update(obs_full[:, :student_obs_dim])

    total_steps = 0
    target_iters = max(1, total_env_steps // (n_steps * n_envs))
    losses_window = deque(maxlen=100)
    ep_rewards_window = deque(maxlen=200)
    running_ep_r = np.zeros(n_envs, dtype=np.float32)

    for iteration in range(target_iters):
        frac = iteration / max(1, target_iters - 1)
        beta = float(beta_schedule(frac))

        # Rollout buffer: just (obs_student_normed, teacher_action).
        obs_buf = np.zeros((n_steps, n_envs, student_obs_dim),
                            dtype=np.float32)
        tgt_buf = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)

        with torch.no_grad():
            for t in range(n_steps):
                # Always query the teacher (we need its action as the
                # target). For ROLLOUT, choose between teacher and
                # student according to β.
                obs_full_t = torch.as_tensor(obs_full, dtype=torch.float32,
                                              device=device)
                teacher_action, _, _ = teacher_policy.act(obs_full_t,
                                                          deterministic=True)

                if beta >= 1.0:
                    act_for_step = teacher_action
                elif beta <= 0.0:
                    obs_stu = obs_norm.normalize(
                        obs_full[:, :student_obs_dim])
                    obs_stu_t = torch.as_tensor(obs_stu, dtype=torch.float32,
                                                 device=device)
                    student_action, _, _ = student.act(obs_stu_t,
                                                        deterministic=True)
                    act_for_step = student_action
                else:
                    use_teacher = (np.random.rand(n_envs) < beta)
                    obs_stu = obs_norm.normalize(
                        obs_full[:, :student_obs_dim])
                    obs_stu_t = torch.as_tensor(obs_stu, dtype=torch.float32,
                                                 device=device)
                    student_action, _, _ = student.act(obs_stu_t,
                                                        deterministic=True)
                    mask = torch.as_tensor(use_teacher.astype(np.float32),
                                            device=device).unsqueeze(-1)
                    act_for_step = teacher_action * mask \
                                   + student_action * (1.0 - mask)

                # Store student's input obs + teacher's target action.
                obs_buf[t] = obs_norm.normalize(
                    obs_full[:, :student_obs_dim])
                tgt_buf[t] = teacher_action.cpu().numpy()

                # Step env with the chosen action.
                next_obs, rew, done, _ = env_teacher.step(
                    act_for_step.cpu().numpy())

                running_ep_r += rew
                if np.any(done):
                    for i in np.where(done)[0]:
                        ep_rewards_window.append(float(running_ep_r[i]))
                        running_ep_r[i] = 0.0

                obs_full = next_obs
                obs_norm.update(obs_full[:, :student_obs_dim])
                total_steps += n_envs

        # ── BC update ─────────────────────────────────────────────
        flat_obs = obs_buf.reshape(-1, student_obs_dim)
        flat_tgt = tgt_buf.reshape(-1, act_dim)
        idx = np.arange(flat_obs.shape[0])
        n_samples = idx.shape[0]
        loss_avg = 0.0
        for epoch in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n_samples, batch_size):
                mb = idx[start:start + batch_size]
                mb_obs = torch.as_tensor(flat_obs[mb], dtype=torch.float32,
                                          device=device)
                mb_tgt = torch.as_tensor(flat_tgt[mb], dtype=torch.float32,
                                          device=device)
                pred, _, _ = student.act(mb_obs, deterministic=True)
                loss = nn.functional.mse_loss(pred, mb_tgt)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                loss_avg = float(loss.detach())
        losses_window.append(loss_avg)

        # ── logging ───────────────────────────────────────────────
        if iteration % 10 == 0 or len(ep_rewards_window) > 0:
            mean_loss = float(np.mean(losses_window)) if losses_window else 0.0
            mean_r = (float(np.mean(ep_rewards_window))
                       if ep_rewards_window else 0.0)
            print(f"[distill|{skill}] iter {iteration:5d} | "
                  f"steps {total_steps:12,d} | β={beta:.2f} | "
                  f"bc_loss={mean_loss:.5f} | R̄={mean_r:7.2f}")
            if logger is not None:
                logger.log_scalars({
                    "distill/bc_loss": mean_loss,
                    "distill/beta": beta,
                    "distill/teacher_mean_reward": mean_r,
                }, step=total_steps)

        if iteration % 100 == 0 and iteration > 0:
            save_student_checkpoint(student, optimizer, total_steps,
                                     skill, obs_norm=obs_norm,
                                     checkpoint_dir=checkpoint_dir)

    save_student_checkpoint(student, optimizer, total_steps, skill,
                             obs_norm=obs_norm,
                             checkpoint_dir=checkpoint_dir)
    return student
