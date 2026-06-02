"""Adversarial Motion Priors (AMP) for the walk skill.

Implements AMP (Peng et al. 2021, arXiv:2104.02180) on top of the existing
PPO loop. The idea: a discriminator is trained to tell apart short motion
transitions produced by the policy from transitions in a *reference* motion
dataset. The policy gets a "style" reward for fooling the discriminator
(looking like the reference), added to the task reward (velocity tracking).
The policy thus learns to do the task WHILE moving like the reference data —
no per-frame phase alignment needed (unlike DeepMimic).

Pieces here:
  * `build_amp_obs`        — the discriminator's feature view of one state.
  * `AMPDiscriminator`     — MLP on a (s, s') transition → real/fake logit,
                             least-squares loss + gradient penalty (AMP-style).
  * `MotionDataset`        — holds reference (amp_obs, amp_obs') transitions,
                             samples minibatches.
  * `parametric_walk_dataset` — synthesizes a walking reference in K1 joint
                             space (no mocap needed) → MotionDataset.
  * `train_ppo_amp_vec`    — PPO + AMP training loop (mirrors the corrected
                             train_ppo_vec, with the style reward + disc update).

The reference data source is swappable: `parametric_walk_dataset` now,
retargeted mocap later — both just produce a MotionDataset of amp-obs
transitions, so nothing downstream changes.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from training.algorithms.networks import make_mlp
from training.algorithms.ppo import save_ppo_checkpoint, _log_metrics, _mean_trunk_height
from training.normalizers import RunningMeanStd, ReturnNormalizer


# ─── AMP observation (the discriminator's feature view of a state) ──────────
#
# Root-invariant proprio features so the discriminator judges MOTION STYLE,
# not world position/heading: root height, projected gravity (orientation),
# body-frame linear+angular velocity, joint positions, joint velocities.
# Identical layout for policy states and reference states (that's the whole
# point — they must be comparable).

AMP_NUM_DOF = 22
# NOTE: root LINEAR velocity is deliberately EXCLUDED. The discriminator should
# judge gait STYLE (pose, orientation, joint motion), not forward speed — speed
# is the task reward's job. Including it let the discriminator separate policy
# from reference on velocity alone (the always-moving reference vs a slow
# policy → trivial win, no style gradient). Angular velocity stays (turning is
# part of style).
#
# foot_clear (2): per-foot CLEARANCE above the ground (height − ground level),
# ≈0 when planted, positive in swing. CRITICAL for stepping: without it the
# discriminator can't tell a glide (feet down) from a step (feet up) — joint
# angles look similar — so it never forces foot-lift and the policy converges
# to a stable glide. It's a CLEARANCE (relative to each sim's standing foot
# height), not raw z, so the reference (MuJoCo FK) and policy (Genesis) align
# despite differing foot-link-frame offsets between simulators.
AMP_OBS_DIM = 1 + 3 + 3 + AMP_NUM_DOF + AMP_NUM_DOF + 2   # = 53


def build_amp_obs(root_height, proj_g, ang_vel_body,
                  dof_pos, dof_vel, foot_clear) -> np.ndarray:
    """Assemble the (N, AMP_OBS_DIM) AMP feature vector. All inputs are numpy
    arrays batched on axis 0 (root_height is (N,) or (N,1)). `foot_clear` is
    (N,2) per-foot height above ground. Root linear velocity is omitted."""
    root_height = np.asarray(root_height, dtype=np.float32).reshape(-1, 1)
    parts = [root_height,
             np.asarray(proj_g, np.float32),
             np.asarray(ang_vel_body, np.float32),
             np.asarray(dof_pos, np.float32),
             np.asarray(dof_vel, np.float32),
             np.asarray(foot_clear, np.float32).reshape(-1, 2)]
    return np.concatenate(parts, axis=1).astype(np.float32)


# ─── Discriminator ─────────────────────────────────────────────────────────


class AMPDiscriminator(nn.Module):
    """Takes a transition (amp_obs_t, amp_obs_{t+1}) → scalar logit. Trained
    least-squares (LSGAN, as in the AMP paper): reference→+1, policy→−1."""

    def __init__(self, amp_obs_dim: int = AMP_OBS_DIM,
                 hidden=(256, 128)):
        super().__init__()
        self.amp_obs_dim = amp_obs_dim
        # Plain MLP (no LayerNorm) on the (s,s') transition; the gradient
        # penalty is the regularizer that keeps the discriminator smooth.
        hidden = (512, 256)   # mid capacity: (1024,512) saturated, (256,128) went random
        self.trunk = make_mlp(2 * amp_obs_dim, list(hidden),
                              layernorm=False, activation="relu")
        self.head = nn.Linear(hidden[-1], 1)
        # Observation normalizer state for amp-obs (set externally).
        self.register_buffer("obs_mean", torch.zeros(amp_obs_dim))
        self.register_buffer("obs_std", torch.ones(amp_obs_dim))

    def normalize(self, transition: torch.Tensor) -> torch.Tensor:
        # transition is (B, 2*amp_obs_dim) = [obs_t | obs_{t+1}]; normalize
        # each half by the same per-feature stats.
        m = torch.cat([self.obs_mean, self.obs_mean])
        s = torch.cat([self.obs_std, self.obs_std])
        return (transition - m) / (s + 1e-5)

    def logit(self, transition: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.normalize(transition)))

    @torch.no_grad()
    def style_reward(self, transition: torch.Tensor) -> torch.Tensor:
        """AMP style reward: r = max(0, 1 − 0.25·(D−1)²), in [0,1]. 1 when the
        discriminator is sure the transition is 'reference-like'."""
        d = self.logit(transition)
        r = 1.0 - 0.25 * (d - 1.0) ** 2
        return torch.clamp(r, min=0.0).squeeze(-1)

    def set_obs_norm(self, mean: np.ndarray, std: np.ndarray):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32,
                                             device=self.obs_mean.device))
        self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32,
                                            device=self.obs_std.device))


def discriminator_loss(disc: AMPDiscriminator,
                       policy_tr: torch.Tensor, ref_tr: torch.Tensor,
                       grad_penalty_coef: float = 5.0,
                       weight_decay_coef: float = 1.0e-4):
    """LSGAN loss + zero-centered gradient penalty on reference transitions
    (the AMP regularizer that keeps the discriminator smooth) + output-weight
    decay. Returns (total_loss, dict of components for logging)."""
    d_ref = disc.logit(ref_tr)
    d_pol = disc.logit(policy_tr)
    loss_ref = ((d_ref - 1.0) ** 2).mean()      # reference → +1
    loss_pol = ((d_pol + 1.0) ** 2).mean()      # policy    → −1
    pred_loss = 0.5 * (loss_ref + loss_pol)

    # Gradient penalty: E[||∂D/∂x||²] on (normalized) reference inputs.
    ref = ref_tr.clone().requires_grad_(True)
    d = disc.logit(ref)
    grad = torch.autograd.grad(d.sum(), ref, create_graph=True)[0]
    gp = (grad.norm(2, dim=-1) ** 2).mean()

    wd = sum((p ** 2).sum() for p in disc.head.parameters())

    total = pred_loss + grad_penalty_coef * gp + weight_decay_coef * wd
    with torch.no_grad():
        acc = ((d_ref > 0).float().mean() + (d_pol < 0).float().mean()) * 0.5
    return total, {
        "disc_loss": float(pred_loss.detach()),
        "disc_grad_pen": float(gp.detach()),
        "disc_acc": float(acc),
        "disc_logit_ref": float(d_ref.mean().detach()),
        "disc_logit_pol": float(d_pol.mean().detach()),
    }


# ─── Reference motion dataset ───────────────────────────────────────────────


class MotionDataset:
    """Holds reference AMP-obs transitions as a flat (M, 2*amp_obs_dim) tensor
    and samples minibatches. Source-agnostic: parametric now, mocap later."""

    def __init__(self, transitions: np.ndarray, device):
        t = torch.as_tensor(np.asarray(transitions, np.float32), device=device)
        assert t.ndim == 2, "transitions must be (M, 2*amp_obs_dim)"
        self.data = t
        self.n = t.shape[0]
        self.device = device

    def sample(self, batch: int) -> torch.Tensor:
        idx = torch.randint(0, self.n, (batch,), device=self.device)
        return self.data[idx]

    @property
    def amp_obs_dim(self) -> int:
        return self.data.shape[1] // 2

    def feature_stats(self):
        """Per-feature mean/std over single amp-obs (both halves stacked) —
        used to seed the discriminator's input normalizer."""
        d = self.amp_obs_dim
        both = torch.cat([self.data[:, :d], self.data[:, d:]], dim=0)
        # Floor the std at 0.1: the parametric reference holds several features
        # CONSTANT (upright gravity, zero lateral/vertical/angular velocity),
        # so their true std≈0. Dividing by ~0 would explode normalization and
        # swamp the discriminator; the floor keeps constant features strongly
        # discriminative (policy must match them) without numerical blowup.
        return (both.mean(0).cpu().numpy(),
                both.std(0).clamp(min=0.1).cpu().numpy())


# ─── Parametric walking reference (no mocap) ────────────────────────────────
#
# Synthesize a recognizable bipedal walk in K1 joint space: sinusoidal
# hip/knee/ankle oscillation (legs antiphase) + arm counter-swing, around the
# default crouch pose. The discriminator only needs the DISTRIBUTION of
# (pose, pose') transitions to read as "walking", so exact biomechanics don't
# matter — feet lift, legs swing rhythmically, arms counter-swing.

# K1 joint indices (see configs/config.K1RobotConfig.joint_names).
_L_SH_P, _R_SH_P = 2, 6
_L_HIP_P, _L_KNEE, _L_ANK_P = 10, 13, 14
_R_HIP_P, _R_KNEE, _R_ANK_P = 16, 19, 20


def _walk_pose(phase: np.ndarray, default: np.ndarray) -> np.ndarray:
    """Joint positions (len-22) for each phase in [0,1). `phase` is (K,).
    Returns (K, 22)."""
    K = phase.shape[0]
    q = np.tile(default[None, :].astype(np.float32), (K, 1))
    pL = 2 * np.pi * phase
    pR = 2 * np.pi * (phase + 0.5)
    A_hip, A_knee, A_ank, A_arm = 0.35, 0.5, 0.15, 0.30
    # legs (deltas around default crouch)
    q[:, _L_HIP_P] += A_hip * np.sin(pL)
    q[:, _R_HIP_P] += A_hip * np.sin(pR)
    q[:, _L_KNEE] += A_knee * (1.0 - np.cos(pL)) * 0.5   # bend in swing
    q[:, _R_KNEE] += A_knee * (1.0 - np.cos(pR)) * 0.5
    q[:, _L_ANK_P] += A_ank * np.sin(pL)
    q[:, _R_ANK_P] += A_ank * np.sin(pR)
    # arm counter-swing (left arm with right leg, and vice versa)
    q[:, _L_SH_P] += A_arm * np.sin(pR)
    q[:, _R_SH_P] += A_arm * np.sin(pL)
    return q.astype(np.float32)


def load_motion_dataset(npz_path: str, device) -> MotionDataset:
    """Load a reference-motion AMP dataset from a .npz of (M, 2*AMP_OBS_DIM)
    transitions — e.g. produced by retargeting human mocap (LAFAN1) to K1 via
    GMR and converting to build_amp_obs features. This is the REAL-motion
    reference (vs the synthetic parametric_walk_dataset); the rest of the AMP
    pipeline is identical."""
    data = np.load(npz_path)
    trans = data["transitions"]
    exp = 2 * AMP_OBS_DIM
    if trans.shape[1] != exp:
        raise ValueError(f"motion file {npz_path}: transitions dim "
                         f"{trans.shape[1]} != expected {exp} (2*AMP_OBS_DIM). "
                         "Was it built with the current build_amp_obs layout?")
    print(f"[amp] loaded {trans.shape[0]} reference transitions from {npz_path}")
    return MotionDataset(trans, device)


def parametric_walk_dataset(default_joint_pos, device,
                            n_samples: int = 8192,
                            dt: float = 0.02,
                            gait_freq_range=(1.0, 2.0),
                            speed_range=(0.2, 0.9),
                            stand_height: float = 0.51) -> MotionDataset:
    """Build a MotionDataset of (amp_obs_t, amp_obs_{t+1}) transitions from the
    parametric walk, sampled over random phase / gait-frequency / speed."""
    default = np.asarray(default_joint_pos, dtype=np.float32)
    rng = np.random.default_rng(0)
    phase = rng.uniform(0.0, 1.0, n_samples).astype(np.float32)
    gfreq = rng.uniform(*gait_freq_range, n_samples).astype(np.float32)
    speed = rng.uniform(*speed_range, n_samples).astype(np.float32)
    dphi = gfreq * dt                                  # phase advance per step

    # three consecutive poses to get pos + finite-diff vel at t and t+1
    q0 = _walk_pose(phase, default)
    q1 = _walk_pose(phase + dphi, default)
    q2 = _walk_pose(phase + 2 * dphi, default)
    v0 = (q1 - q0) / dt
    v1 = (q2 - q1) / dt

    def feats(q, v, ph):
        N = q.shape[0]
        proj_g = np.tile(np.array([0, 0, -1], np.float32), (N, 1))
        ang = np.zeros((N, 3), np.float32)
        # slight vertical bob at 2× gait freq
        h = (stand_height + 0.01 * np.cos(4 * np.pi * ph)).astype(np.float32)
        return build_amp_obs(h, proj_g, ang, q, v)

    amp_t = feats(q0, v0, phase)
    amp_tp1 = feats(q1, v1, phase + dphi)
    transitions = np.concatenate([amp_t, amp_tp1], axis=1).astype(np.float32)
    return MotionDataset(transitions, device)


# ─── PPO + AMP training loop ────────────────────────────────────────────────


def train_ppo_amp_vec(env, policy, config, motion_dataset: MotionDataset,
                      logger=None, phase="skill_walk", curriculum_stage=None,
                      checkpoint_dir: str = "checkpoints",
                      task_reward_coef: float = 0.5,
                      style_reward_coef: float = 0.5,
                      disc_lr: float = 6.0e-5,
                      disc_updates_per_iter: int = 1,
                      disc_batch: int = 4096,
                      grad_penalty_coef: float = 6.0,
                      desired_kl: float = 0.01,
                      use_value_clipping: bool = True,
                      adaptive_lr: bool = True,
                      min_lr: float = 1e-5, max_lr: float = 1e-2,
                      device: Optional[torch.device] = None):
    """PPO with an AMP style reward. Mirrors train_ppo_vec (incl. the corrected
    GAE / terminal-obs / truncation bootstrap), adding: per-step amp-obs
    collection, a style reward from the discriminator combined with the env's
    task reward, and an interleaved discriminator update each iteration.

    The env must expose `amp_observation() -> (N, AMP_OBS_DIM)`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert hasattr(env, "amp_observation"), \
        "AMP training needs env.amp_observation()"

    policy = policy.to(device)
    lr = float(config.learning_rate)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    amp_dim = motion_dataset.amp_obs_dim
    disc = AMPDiscriminator(amp_dim).to(device)
    disc.set_obs_norm(*motion_dataset.feature_stats())
    disc_opt = torch.optim.Adam(disc.parameters(), lr=disc_lr)

    n_envs = env.num_envs
    n_steps = int(config.n_steps)
    obs_dim = int(config.obs_dim)
    act_dim = int(config.act_dim)

    obs_norm = RunningMeanStd(shape=(obs_dim,))
    ret_norm = ReturnNormalizer(gamma=config.gamma)

    ep_rewards, ep_lengths = deque(maxlen=200), deque(maxlen=200)
    running_ep_r = np.zeros(n_envs, dtype=np.float32)
    running_ep_len = np.zeros(n_envs, dtype=np.int64)
    comp_queues: dict = {}
    style_q = deque(maxlen=200)

    total_steps = 0
    num_iterations = max(1, int(config.total_timesteps) // (n_steps * n_envs))

    print(f"\n{'='*60}")
    print(f" [PPO+AMP] phase={phase}  n_envs={n_envs} n_steps={n_steps} "
          f"iters={num_iterations}")
    print(f"   amp_obs_dim={amp_dim}  ref_transitions={motion_dataset.n}")
    print(f"   reward = {task_reward_coef}·task + {style_reward_coef}·style")
    print(f"   device={device}")
    print(f"{'='*60}\n")

    obs = env.reset()
    obs_norm.update(obs)
    obs_t = torch.as_tensor(obs_norm.normalize(obs), dtype=torch.float32,
                            device=device)
    amp_obs = env.amp_observation()                    # (N, amp_dim) raw

    obs_buf = torch.zeros(n_steps, n_envs, obs_dim, device=device)
    act_buf = torch.zeros(n_steps, n_envs, act_dim, device=device)
    logp_buf = torch.zeros(n_steps, n_envs, device=device)
    rew_buf = torch.zeros(n_steps, n_envs, device=device)
    val_buf = torch.zeros(n_steps, n_envs, device=device)
    done_buf = torch.zeros(n_steps, n_envs, device=device)
    term_buf = torch.zeros(n_steps, n_envs, device=device)
    trunc_buf = torch.zeros(n_steps, n_envs, device=device)
    term_obs_buf = torch.zeros(n_steps, n_envs, obs_dim, device=device)
    # AMP transitions for THIS rollout (policy samples for the discriminator).
    amp_tr_buf = torch.zeros(n_steps, n_envs, 2 * amp_dim, device=device)

    for iteration in range(num_iterations):
        # ── rollout ──
        with torch.no_grad():
            for step in range(n_steps):
                action, log_prob, _ = policy.act(obs_t)
                value = policy.get_value(obs_t)
                action_np = action.cpu().numpy()
                next_obs, reward, done, info = env.step(action_np)
                next_amp_obs = env.amp_observation()

                obs_buf[step] = obs_t
                act_buf[step] = action
                logp_buf[step] = log_prob
                rew_buf[step] = torch.as_tensor(reward, device=device,
                                                dtype=torch.float32)
                val_buf[step] = value
                done_buf[step] = torch.as_tensor(
                    np.asarray(done, np.float32), device=device)
                truncated = info.get("truncated")
                terminated = info.get("terminated")
                if truncated is None:
                    truncated = np.zeros(n_envs, np.float32)
                    terminated = np.asarray(done, np.float32)
                trunc_buf[step] = torch.as_tensor(np.asarray(truncated, np.float32),
                                                  device=device)
                term_buf[step] = torch.as_tensor(np.asarray(terminated, np.float32),
                                                 device=device)
                term_obs_buf[step] = torch.as_tensor(
                    obs_norm.normalize(info.get("terminal_obs", next_obs)),
                    dtype=torch.float32, device=device)
                # AMP transition (s_t, s_{t+1}); on a reset the next amp-obs is
                # the post-reset frame — fine, it's a rare boundary sample.
                amp_tr_buf[step] = torch.as_tensor(
                    np.concatenate([amp_obs, next_amp_obs], axis=1),
                    dtype=torch.float32, device=device)

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

                obs_norm.update(next_obs)
                obs_t = torch.as_tensor(obs_norm.normalize(next_obs),
                                        dtype=torch.float32, device=device)
                amp_obs = next_amp_obs

        # ── style reward + combine with task reward ──
        flat_tr = amp_tr_buf.reshape(n_steps * n_envs, 2 * amp_dim)
        style_r = disc.style_reward(flat_tr).reshape(n_steps, n_envs)
        style_q.append(float(style_r.mean()))
        combined = task_reward_coef * rew_buf + style_reward_coef * style_r

        # ── reward normalisation (on the COMBINED reward) ──
        ret_norm.update(combined.cpu().numpy().reshape(-1),
                        done_buf.cpu().numpy().reshape(-1))
        rew_norm_buf = combined / max(1e-8, float(ret_norm.rms.std))

        # ── GAE (corrected: current-step mask, truncation bootstrap) ──
        with torch.no_grad():
            next_value = policy.get_value(obs_t)
            term_values = policy.get_value(
                term_obs_buf.reshape(n_steps * n_envs, obs_dim)
            ).reshape(n_steps, n_envs)
        advantages = torch.zeros_like(rew_norm_buf)
        gae = torch.zeros(n_envs, device=device)
        for t in reversed(range(n_steps)):
            next_val_default = next_value if t == n_steps - 1 else val_buf[t + 1]
            boot = torch.where(trunc_buf[t].bool(), term_values[t], next_val_default)
            nonterminal = 1.0 - term_buf[t]
            cont = 1.0 - done_buf[t]
            delta = rew_norm_buf[t] + config.gamma * boot * nonterminal - val_buf[t]
            gae = delta + config.gamma * config.gae_lambda * cont * gae
            advantages[t] = gae
        returns = advantages + val_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            y_true = returns.reshape(-1).cpu().numpy()
            y_pred = val_buf.reshape(-1).cpu().numpy()
            explained_var = float(
                1.0 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8))

        # ── discriminator update ──
        disc_metrics = {}
        for _ in range(disc_updates_per_iter):
            pol_tr = flat_tr[torch.randint(0, flat_tr.shape[0], (disc_batch,),
                                           device=device)]
            ref_tr = motion_dataset.sample(disc_batch)
            d_loss, disc_metrics = discriminator_loss(
                disc, pol_tr, ref_tr, grad_penalty_coef=grad_penalty_coef)
            disc_opt.zero_grad()
            d_loss.backward()
            disc_opt.step()

        # ── PPO update ──
        b_obs = obs_buf.reshape(n_steps * n_envs, obs_dim)
        b_act = act_buf.reshape(n_steps * n_envs, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)
        n_samples = b_obs.shape[0]
        mb_size = max(1, n_samples // 4)

        approx_kl = clip_frac = 0.0
        policy_loss_val = value_loss_val = entropy_val = 0.0
        kl_too_big = False
        for epoch in range(config.n_epochs):
            indices = torch.randperm(n_samples, device=device)
            epoch_kls = []
            for start in range(0, n_samples, mb_size):
                mb = indices[start:start + mb_size]
                values, new_log_prob, entropy = policy.evaluate(b_obs[mb], b_act[mb])
                ratio = (new_log_prob - b_logp[mb]).exp()
                surr1 = ratio * b_adv[mb]
                surr2 = torch.clamp(ratio, 1 - config.clip_range,
                                    1 + config.clip_range) * b_adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                if use_value_clipping:
                    v_clip = b_val[mb] + (values - b_val[mb]).clamp(
                        -config.clip_range, config.clip_range)
                    value_loss = 0.5 * torch.max((values - b_ret[mb]) ** 2,
                                                 (v_clip - b_ret[mb]) ** 2).mean()
                else:
                    value_loss = 0.5 * ((values - b_ret[mb]) ** 2).mean()
                entropy_loss = -entropy.mean()
                loss = (policy_loss + config.vf_coef * value_loss
                        + config.entropy_coef * entropy_loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    log_ratio = new_log_prob - b_logp[mb]
                    approx_kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
                    clip_frac = float((log_ratio.abs()
                                       > math.log(1 + config.clip_range)).float().mean())
                    epoch_kls.append(approx_kl)
                policy_loss_val = float(policy_loss.detach())
                value_loss_val = float(value_loss.detach())
                entropy_val = float(-entropy_loss.detach())
            if desired_kl > 0 and np.mean(epoch_kls) > 2.0 * desired_kl:
                kl_too_big = True
                break

        if adaptive_lr and desired_kl > 0:
            if approx_kl > 2.0 * desired_kl:
                lr = max(min_lr, lr / 1.5)
            elif approx_kl < 0.5 * desired_kl:
                lr = min(max_lr, lr * 1.5)
            for g in optimizer.param_groups:
                g["lr"] = lr

        # ── logging ──
        if iteration % 10 == 0 or len(ep_rewards) > 0:
            mr = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            ml = float(np.mean(ep_lengths)) if ep_lengths else 0.0
            metrics = {
                "mean_reward": mr, "mean_length": ml,
                "mean_robot_z": _mean_trunk_height(obs_buf, obs_norm),
                "policy_loss": policy_loss_val, "value_loss": value_loss_val,
                "entropy": entropy_val, "explained_variance": explained_var,
                "approx_kl": approx_kl, "clip_fraction": clip_frac,
                "learning_rate": lr, "kl_early_stop": float(kl_too_big),
                "amp/style_reward": float(np.mean(style_q)) if style_q else 0.0,
            }
            for k, v in disc_metrics.items():
                metrics[f"amp/{k}"] = v
            for k, q in comp_queues.items():
                if q:
                    metrics[f"rewards/{k}"] = float(np.mean(q))
            print(f"[ppo+amp] Iter {iteration:5d} | Steps {total_steps:12,d} | "
                  f"R̄={mr:7.2f} | L̄={ml:6.1f} | style={metrics['amp/style_reward']:.3f} "
                  f"| d_acc={disc_metrics.get('disc_acc', 0):.2f} "
                  f"| KL={approx_kl:.4f}" + ("  [KL-stop]" if kl_too_big else ""))
            _log_metrics(logger, metrics, total_steps)

        if iteration % 100 == 0 and iteration > 0:
            save_ppo_checkpoint(policy, optimizer, total_steps, phase,
                                curriculum_stage, checkpoint_dir=checkpoint_dir,
                                obs_norm=obs_norm, ret_norm=ret_norm)

    save_ppo_checkpoint(policy, optimizer, total_steps, phase, curriculum_stage,
                        checkpoint_dir=checkpoint_dir,
                        obs_norm=obs_norm, ret_norm=ret_norm)
    return policy
