"""Train a single skill policy.

Usage:
    python -m training.train_skill --skill {walk,standup,...} \
        [--algorithm {ppo,flashsac,sac}] [--resume CKPT] [--init-from CKPT] \
        [--vec-num-envs N] [--device {auto,cpu,gpu}] \
        [--total-timesteps N] [--wandb]

Each skill runs in its own Python process so Genesis's GPU memory is
bounded per-skill (no kernel/scene leaks across stages — the bug that
drove the skill-library refactor).

Skills are registered in `_SKILL_REGISTRY` below; adding one is two
lines: import the env, append to the dict.

Algorithm support:
  * `ppo` (default) — on-policy, uses `PPOActorCritic`. Fast to converge
    on locomotion-style rewards, deterministic eval is trivial.
  * `flashsac` / `sac` — off-policy SAC with twin Q + GPU replay
    buffer. Better sample efficiency at the cost of more wall-clock
    per step; useful for skills where the reward landscape is sparser
    (e.g. shoot).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (CHECKPOINTS_DIR, FIELD_DIR, FIELD_JSON, LOGS_DIR,
                            ProjectConfig)
from training.common import (create_policy, default_run_name, load_checkpoint,
                             resolve_device, setup_logger)


# ─── skill registry ────────────────────────────────────────────────────


def _build_walk(num_envs):
    from skills.walk.env import K1WalkEnv
    from skills.walk.config import WalkConfig
    cfg = WalkConfig()
    if num_envs is not None:
        cfg.num_envs = num_envs
    return K1WalkEnv(cfg=cfg), cfg


def _build_standup(num_envs):
    from skills.standup.env import K1StandupEnv
    from skills.standup.config import StandupConfig
    cfg = StandupConfig()
    if num_envs is not None:
        cfg.num_envs = num_envs
    return K1StandupEnv(cfg=cfg), cfg


def _build_dribble(num_envs):
    from skills.dribble.env import K1DribbleEnv
    from skills.dribble.config import DribbleConfig
    cfg = DribbleConfig()
    if num_envs is not None:
        cfg.num_envs = num_envs
    return K1DribbleEnv(cfg=cfg), cfg


def _build_shoot(num_envs):
    from skills.shoot.env import K1ShootEnv
    from skills.shoot.config import ShootConfig
    cfg = ShootConfig()
    if num_envs is not None:
        cfg.num_envs = num_envs
    return K1ShootEnv(cfg=cfg), cfg


# Maps `--skill <name>` to a builder returning (env, cfg). cfg must
# expose the PPO/FlashSAC hyperparam fields read by the trainers.
_SKILL_REGISTRY = {
    "walk": _build_walk,
    "standup": _build_standup,
    "dribble": _build_dribble,
    "shoot": _build_shoot,
}


# ─── ensure required assets exist ──────────────────────────────────────


def _ensure_field_builder():
    builder = os.path.join(str(FIELD_DIR), "field_genesis_builder.py")
    if os.path.exists(builder):
        return
    print("[train_skill] field_genesis_builder.py missing; regenerating...")
    try:
        from models.field_generator import generate_field_assets
        generate_field_assets(str(FIELD_JSON), str(FIELD_DIR))
    except Exception as e:
        print(f"[train_skill] field regeneration failed: {e}")


# ─── algorithm dispatch ────────────────────────────────────────────────


def _train_with_algorithm(algorithm: str, env, cfg, logger, device,
                          checkpoint_dir: str, resume: str = None,
                          init_from: str = None,
                          algo_kwargs: dict = None):
    """Dispatch to PPO or FlashSAC. Returns the trained network (PPO
    actor-critic or SAC actor)."""
    algo_kwargs = dict(algo_kwargs or {})

    if algorithm == "ppo":
        policy = create_policy(env.obs_dim, env.act_dim)
        if init_from:
            load_checkpoint(init_from, policy)
        elif resume:
            load_checkpoint(resume, policy)
        from training.algorithms.ppo import train_ppo_vec
        return train_ppo_vec(
            env, policy, cfg, logger,
            phase=f"skill_{env.SKILL_NAME}",
            curriculum_stage=None,
            checkpoint_dir=checkpoint_dir,
            device=device,
            **algo_kwargs,
        )

    if algorithm in ("flashsac", "sac"):
        # FlashSAC builds its own actor/critic; no preset PPO policy.
        # Warm-start support deferred — partial-load from PPO checkpoint
        # would need a separate path because the network shapes differ.
        if resume or init_from:
            print("[train_skill] WARNING: --resume / --init-from with "
                  "FlashSAC is not yet supported; building fresh nets.")
        from training.algorithms.flashsac import train_flashsac_vec
        return train_flashsac_vec(
            env, policy=None, config=cfg, logger=logger,
            phase=f"skill_{env.SKILL_NAME}",
            curriculum_stage=None,
            checkpoint_dir=checkpoint_dir,
            device=device,
            **algo_kwargs,
        )

    raise ValueError(f"unknown algorithm: {algorithm!r}")


# ─── main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train a single skill policy (PPO or FlashSAC)")
    parser.add_argument("--skill", required=True,
                        choices=sorted(_SKILL_REGISTRY.keys()),
                        help="Which skill to train.")
    parser.add_argument("--algorithm", choices=["ppo", "flashsac", "sac"],
                        default="ppo",
                        help="RL algorithm. 'sac' is an alias for 'flashsac'.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (full state load).")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Warm-start policy weights from another "
                             "checkpoint (partial load — shape-mismatched "
                             "layers are skipped). PPO only.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"],
                        default="auto")
    parser.add_argument("--vec-num-envs", type=int, default=None,
                        help="Override the skill's default num_envs.")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override the skill's default total_timesteps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    algorithm = "flashsac" if args.algorithm == "sac" else args.algorithm

    _ensure_field_builder()

    device, preset, device_kind = resolve_device(args.device)

    # TF32 free throughput on Ampere+ GPUs.
    if device_kind == "gpu":
        import torch
        torch.set_float32_matmul_precision("high")

    # Pick num_envs: CLI > device preset > skill default.
    num_envs = args.vec_num_envs
    if num_envs is None:
        num_envs = preset["vec_num_envs"]

    builder = _SKILL_REGISTRY[args.skill]
    env, cfg = builder(num_envs)

    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)

    np.random.seed(args.seed)

    project = ProjectConfig()
    project.seed = args.seed
    wandb_project = args.wandb_project or project.wandb.project
    use_wandb = bool(args.wandb or args.wandb_project)

    logger = setup_logger(
        run_name=default_run_name(f"skill_{args.skill}", algorithm),
        log_root=str(LOGS_DIR),
        wandb_project=wandb_project,
        wandb_entity=project.wandb.entity,
        wandb_tags=list(project.wandb.tags) + ["skill", args.skill, algorithm],
        use_wandb=use_wandb,
        config={
            "skill": args.skill,
            "algorithm": algorithm,
            "seed": args.seed,
            "num_envs": num_envs,
            "obs_dim": env.obs_dim,
            "act_dim": env.act_dim,
            "command_dim": env.command_spec.dim,
            "command_names": list(env.command_spec.names),
        },
    )

    # Per-skill checkpoint sub-directory keeps `skill_walk_step123.pt`
    # and `skill_standup_step123.pt` from overlapping.
    ckpt_dir = os.path.join(str(CHECKPOINTS_DIR), f"skill_{args.skill}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Forward algorithm-specific kwargs from the device preset (replay
    # buffer / batch size for FlashSAC, none for PPO).
    algo_kwargs = dict(preset.get(algorithm, {}))

    print(f"\n[train_skill] {args.skill} via {algorithm}  "
          f"num_envs={num_envs}  obs_dim={env.obs_dim}  "
          f"act_dim={env.act_dim}  cmd={env.command_spec.dim}d  "
          f"device={device}  algo_kwargs={algo_kwargs or '{}'}\n")

    try:
        _train_with_algorithm(
            algorithm=algorithm, env=env, cfg=cfg, logger=logger,
            device=device, checkpoint_dir=ckpt_dir,
            resume=args.resume, init_from=args.init_from,
            algo_kwargs=algo_kwargs,
        )
    finally:
        env.close()
        logger.close()


if __name__ == "__main__":
    main()
