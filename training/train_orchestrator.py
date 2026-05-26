"""Train the Phase-2 orchestrator policy with 4v4 self-play.

Usage:
    python -m training.train_orchestrator \
        [--num-envs 256] [--device {auto,cpu,gpu}] \
        [--total-timesteps 100000000] [--wandb]

The orchestrator (`orchestrator/policy.py`) emits a hybrid (skill_idx,
cmd_vec_7d) per agent. The 4 frozen skill policies live in
`checkpoints/skill_<name>/skill_<name>_<step>.pt` (best is picked by
mtime). The trainer:

  * Builds `K1MatchEnv` with a `SkillRouter` over the frozen skills.
  * Runs PPO on team 0 only — team 1 is controlled by a snapshot from
    the opponent pool.
  * Snapshots the current policy into the pool every
    `opponent_update_freq` iterations.

This script is structured around a custom rollout loop (rather than
reusing `train_ppo_vec` directly) because the action splitting between
current/opponent policy doesn't fit the single-policy buffer interface.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (CHECKPOINTS_DIR, FIELD_DIR, FIELD_JSON, LOGS_DIR,
                            K1RobotConfig, ProjectConfig)
from orchestrator.config import (NUM_SKILLS, ORCHESTRATOR_CMD_DIM,
                                  OrchestratorConfig, SKILL_ORDER)
from orchestrator.env import K1MatchEnv
from orchestrator.policy import OrchestratorActorCritic
from orchestrator.self_play import OpponentPool, split_team_action
from orchestrator.skill_router import SkillRouter, load_frozen_skill
from training.common import (default_run_name, load_checkpoint, resolve_device,
                             setup_logger)
from training.normalizers import RunningMeanStd, ReturnNormalizer


# ─── skill checkpoint discovery ────────────────────────────────────────


def _find_latest_skill_ckpt(skill_name: str) -> str:
    """Return the most recent checkpoint for `skill_<name>` (by mtime),
    or "" if none exist. PPO checkpoints land under
    checkpoints/skill_<name>/skill_<name>_step<N>.pt"""
    d = Path(CHECKPOINTS_DIR) / f"skill_{skill_name}"
    if not d.exists():
        return ""
    ckpts = sorted(d.glob(f"skill_{skill_name}_step*.pt"),
                   key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else ""


def _build_skill_env_for_dims(skill_name: str):
    """Tiny disposable env instance used only to read obs/act/cmd dims
    when populating the skill router. Genesis isn't initialized."""
    if skill_name == "standup":
        from skills.standup.env import K1StandupEnv
        from skills.standup.config import StandupConfig
        return K1StandupEnv(cfg=StandupConfig(num_envs=1))
    if skill_name == "walk":
        from skills.walk.env import K1WalkEnv
        from skills.walk.config import WalkConfig
        return K1WalkEnv(cfg=WalkConfig(num_envs=1))
    if skill_name == "dribble":
        from skills.dribble.env import K1DribbleEnv
        from skills.dribble.config import DribbleConfig
        return K1DribbleEnv(cfg=DribbleConfig(num_envs=1))
    if skill_name == "shoot":
        from skills.shoot.env import K1ShootEnv
        from skills.shoot.config import ShootConfig
        return K1ShootEnv(cfg=ShootConfig(num_envs=1))
    raise ValueError(f"unknown skill {skill_name!r}")


def _build_skill_router(device: torch.device) -> SkillRouter:
    frozen = {}
    for name in SKILL_ORDER:
        env_for_dims = _build_skill_env_for_dims(name)
        ckpt = _find_latest_skill_ckpt(name)
        if not ckpt:
            print(f"[train_orch] WARNING: no checkpoint found for skill "
                  f"{name!r} under checkpoints/skill_{name}/. "
                  "Orchestrator training will be uninformative until you "
                  f"train it: `python -m training.train_skill --skill {name}`.")
        frozen[name] = load_frozen_skill(
            name=name, checkpoint_path=ckpt,
            env_for_dims=env_for_dims, device=device,
            addon_builder=None,   # match env wires real builders at step time
        )
    return SkillRouter(frozen, device=device)


# ─── PPO update on team-0 transitions ──────────────────────────────────


def _ppo_update(policy: OrchestratorActorCritic,
                optimizer: torch.optim.Optimizer,
                obs_buf: torch.Tensor,
                act_buf: torch.Tensor,
                logp_buf: torch.Tensor,
                adv_buf: torch.Tensor,
                ret_buf: torch.Tensor,
                val_buf: torch.Tensor,
                cfg: OrchestratorConfig,
                desired_kl: float = 0.01,
                ) -> dict:
    """One iteration of mini-batched PPO updates. Returns scalars for
    logging. Adapted from training.algorithms.ppo (the orchestrator's
    hybrid policy is fully compatible — `evaluate()` returns the same
    (value, log_prob, entropy) tuple)."""
    n_samples = obs_buf.shape[0]
    mb_size = max(1, n_samples // 4)

    approx_kl = 0.0
    clip_frac = 0.0
    pol_loss_val = 0.0
    val_loss_val = 0.0
    ent_val = 0.0

    for epoch in range(cfg.n_epochs):
        indices = torch.randperm(n_samples, device=obs_buf.device)
        epoch_kls = []
        for start in range(0, n_samples, mb_size):
            mb = indices[start:start + mb_size]
            mb_obs = obs_buf[mb]; mb_act = act_buf[mb]
            mb_logp = logp_buf[mb]; mb_adv = adv_buf[mb]
            mb_ret = ret_buf[mb]; mb_val_old = val_buf[mb]

            values, new_log_prob, entropy = policy.evaluate(mb_obs, mb_act)

            ratio = (new_log_prob - mb_logp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_range,
                                1 + cfg.clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_clip = mb_val_old + (values - mb_val_old).clamp(
                -cfg.clip_range, cfg.clip_range)
            value_loss = 0.5 * torch.max(
                (values - mb_ret).pow(2), (v_clip - mb_ret).pow(2)).mean()
            entropy_loss = -entropy.mean()
            loss = (policy_loss + cfg.vf_coef * value_loss
                    + cfg.entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_prob - mb_logp
                kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                approx_kl = float(kl)
                clip_frac = float((log_ratio.abs()
                                   > math.log(1 + cfg.clip_range)
                                   ).float().mean())
                epoch_kls.append(approx_kl)
            pol_loss_val = float(policy_loss.detach())
            val_loss_val = float(value_loss.detach())
            ent_val = float(-entropy_loss.detach())

        mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
        if desired_kl > 0 and mean_kl > 2.0 * desired_kl:
            break

    return {"policy_loss": pol_loss_val, "value_loss": val_loss_val,
            "entropy": ent_val, "approx_kl": approx_kl,
            "clip_fraction": clip_frac}


# ─── main rollout loop ─────────────────────────────────────────────────


def train_orchestrator(cfg: OrchestratorConfig, device: torch.device,
                       logger, use_wandb: bool = False,
                       resume_from: Optional[str] = None) -> OrchestratorActorCritic:
    K = int(cfg.players_per_team)
    obs_dim = int(cfg.obs_layout.total)
    act_dim = 1 + ORCHESTRATOR_CMD_DIM
    n_envs = int(cfg.num_envs)
    n_agents = 2 * K
    n_team0 = n_envs * K

    # ── skill router + match env ─────────────────────────────────
    print("[train_orch] loading frozen skill policies …")
    router = _build_skill_router(device)
    print(f"[train_orch] router has {len(router.skills)} skills: "
          f"{[s.name for s in router.skills]}")

    env = K1MatchEnv(cfg=cfg, robot_cfg=K1RobotConfig(),
                     skill_router=router)
    env.reset()
    print(f"[train_orch] env: n_envs={n_envs} n_agents={n_agents} "
          f"obs_dim={env.obs_dim} act_dim={env.act_dim}")

    # ── policy + optimizer ───────────────────────────────────────
    policy = OrchestratorActorCritic(obs_dim=obs_dim).to(device)
    if resume_from:
        load_checkpoint(resume_from, policy)
    optimizer = torch.optim.Adam(policy.parameters(),
                                 lr=float(cfg.learning_rate), eps=1e-5)

    # Opponent (team 1) — separate frozen instance; weights swapped in
    # from the pool each iteration.
    opp_policy = OrchestratorActorCritic(obs_dim=obs_dim).to(device).eval()
    for p in opp_policy.parameters():
        p.requires_grad_(False)
    opp_pool = OpponentPool(capacity=cfg.opponent_pool_size,
                            latest_prob=cfg.opponent_latest_prob,
                            seed=42)

    # ── normalizers (team-0 obs only — opponent doesn't update them) ──
    obs_norm = RunningMeanStd(shape=(obs_dim,))
    ret_norm = ReturnNormalizer(gamma=cfg.gamma)

    # ── rollout buffer for team-0 transitions ─────────────────────
    n_steps = int(cfg.n_steps)
    obs_buf_b = torch.zeros(n_steps, n_team0, obs_dim, device=device)
    act_buf_b = torch.zeros(n_steps, n_team0, act_dim, device=device)
    logp_buf_b = torch.zeros(n_steps, n_team0, device=device)
    rew_buf_b = torch.zeros(n_steps, n_team0, device=device)
    val_buf_b = torch.zeros(n_steps, n_team0, device=device)
    done_buf_b = torch.zeros(n_steps, n_team0, device=device)

    # Episode bookkeeping per team-0 agent
    ep_rewards = deque(maxlen=200)
    running_ep_r = np.zeros(n_team0, dtype=np.float32)

    total_steps = 0
    num_iterations = max(1, int(cfg.total_timesteps) // (n_steps * n_team0))

    obs_np = env.reset()                                       # (N, A, O)
    obs_team0 = obs_np[:, :K].reshape(n_team0, obs_dim)
    obs_norm.update(obs_team0)

    print(f"\n{'='*60}\n [orchestrator] PPO + self-play  iters={num_iterations} "
          f"steps_per_iter={n_steps * n_team0:,}\n{'='*60}\n")

    for iteration in range(num_iterations):
        # Refresh opponent at the start of each iteration.
        opp_pool.load_into(opp_policy)
        opp = opp_policy if len(opp_pool) > 0 else None

        # ── rollout ────────────────────────────────────────────
        with torch.no_grad():
            for step in range(n_steps):
                # team-0 acts; team-1 acts via opponent.
                obs_team0_norm = obs_norm.normalize(obs_team0)
                obs_t = torch.as_tensor(obs_team0_norm,
                                        dtype=torch.float32, device=device)
                action0, log_prob0, _ = policy.act(obs_t)
                value0 = policy.get_value(obs_t)

                # Team 1 obs
                obs_team1 = obs_np[:, K:].reshape(n_team0, obs_dim)
                obs_team1_t = torch.as_tensor(obs_team1,
                                              dtype=torch.float32,
                                              device=device)
                if opp is not None:
                    action1, _, _ = opp.act(obs_team1_t, deterministic=False)
                else:
                    action1, _, _ = policy.act(obs_team1_t,
                                               deterministic=False)

                # Pack into (N, 2K, act_dim)
                a0 = action0.cpu().numpy().reshape(n_envs, K, act_dim)
                a1 = action1.cpu().numpy().reshape(n_envs, K, act_dim)
                full_action = np.concatenate([a0, a1], axis=1)

                next_obs, reward, done, info = env.step(full_action)
                rew_team0 = reward[:, :K].reshape(n_team0)
                done_team0 = np.broadcast_to(
                    done[:, None], (n_envs, K)).reshape(n_team0)

                obs_buf_b[step] = obs_t
                act_buf_b[step] = action0
                logp_buf_b[step] = log_prob0
                rew_buf_b[step] = torch.as_tensor(rew_team0, device=device)
                val_buf_b[step] = value0
                done_buf_b[step] = torch.as_tensor(
                    done_team0.astype(np.float32), device=device)

                running_ep_r += rew_team0
                total_steps += n_team0
                if np.any(done_team0):
                    for j in np.where(done_team0)[0]:
                        ep_rewards.append(float(running_ep_r[j]))
                        running_ep_r[j] = 0.0

                obs_np = next_obs
                obs_team0 = obs_np[:, :K].reshape(n_team0, obs_dim)
                obs_norm.update(obs_team0)

        # ── reward normalisation + GAE ─────────────────────────
        rew_np = rew_buf_b.cpu().numpy().reshape(-1)
        done_np = done_buf_b.cpu().numpy().reshape(-1)
        ret_norm.update(rew_np, done_np)
        rew_norm = rew_buf_b / max(1e-8, float(ret_norm.rms.std))

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_norm.normalize(obs_team0),
                                    dtype=torch.float32, device=device)
            next_value = policy.get_value(obs_t)

        advantages = torch.zeros_like(rew_norm)
        gae = torch.zeros(n_team0, device=device)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
                next_done = torch.zeros(n_team0, device=device)
            else:
                next_val = val_buf_b[t + 1]
                next_done = done_buf_b[t + 1]
            delta = (rew_norm[t]
                     + cfg.gamma * next_val * (1 - next_done)
                     - val_buf_b[t])
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
        returns = advantages + val_buf_b
        advantages = ((advantages - advantages.mean())
                      / (advantages.std() + 1e-8))

        flat_obs = obs_buf_b.reshape(-1, obs_dim)
        flat_act = act_buf_b.reshape(-1, act_dim)
        flat_logp = logp_buf_b.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_val = val_buf_b.reshape(-1)

        stats = _ppo_update(policy, optimizer,
                            flat_obs, flat_act, flat_logp,
                            flat_adv, flat_ret, flat_val, cfg)

        # ── self-play snapshot ─────────────────────────────────
        if (iteration + 1) % cfg.opponent_update_freq == 0:
            opp_pool.snapshot(policy)
            print(f"  [self-play] snapshot saved; pool size = {len(opp_pool)}")

        # ── logging ────────────────────────────────────────────
        mr = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        metrics = {**stats, "mean_reward": mr,
                   "pool_size": float(len(opp_pool))}
        print(f"[orch] iter {iteration:5d} | steps {total_steps:12,d} | "
              f"R̄={mr:7.2f} | π={stats['policy_loss']:.4f} "
              f"V={stats['value_loss']:.4f} KL={stats['approx_kl']:.4f}")
        if logger is not None:
            logger.log_scalars(
                {(k if "/" in k else f"train/{k}"): v
                 for k, v in metrics.items()},
                step=total_steps)

        # ── checkpoint ─────────────────────────────────────────
        if (iteration + 1) % 100 == 0:
            ckpt_dir = Path(CHECKPOINTS_DIR) / "orchestrator"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"orchestrator_step{total_steps}.pt"
            torch.save({
                "step": total_steps, "phase": "orchestrator",
                "algorithm": "ppo",
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, path)
            print(f"  [ckpt] → {path}")

    env.close()
    return policy


# ─── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train the Phase-2 orchestrator (4v4 self-play).")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override OrchestratorConfig.num_envs.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"],
                        default="auto")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume orchestrator from checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    # Ensure field generator output is present.
    builder = Path(FIELD_DIR) / "field_genesis_builder.py"
    if not builder.exists():
        print("[train_orch] field_genesis_builder.py missing; regenerating...")
        try:
            from models.field_generator import generate_field_assets
            generate_field_assets(str(FIELD_JSON), str(FIELD_DIR))
        except Exception as e:
            print(f"[train_orch] field regeneration failed: {e}")

    device, _preset, device_kind = resolve_device(args.device)
    if device_kind == "gpu":
        torch.set_float32_matmul_precision("high")

    cfg = OrchestratorConfig()
    if args.num_envs is not None:
        cfg.num_envs = int(args.num_envs)
    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project = ProjectConfig()
    use_wandb = bool(args.wandb or args.wandb_project)
    wandb_project = args.wandb_project or project.wandb.project
    logger = setup_logger(
        run_name=default_run_name("orchestrator", "ppo"),
        log_root=str(LOGS_DIR),
        wandb_project=wandb_project,
        wandb_entity=project.wandb.entity,
        wandb_tags=list(project.wandb.tags) + ["orchestrator", "phase2", "ppo"],
        use_wandb=use_wandb,
        config={
            "phase": "orchestrator", "algorithm": "ppo",
            "num_envs": cfg.num_envs,
            "obs_dim": cfg.obs_layout.total,
            "players_per_team": cfg.players_per_team,
        },
    )

    try:
        train_orchestrator(cfg, device, logger,
                           use_wandb=use_wandb,
                           resume_from=args.resume)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
