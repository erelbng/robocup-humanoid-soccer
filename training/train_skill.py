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


def _build_walk(num_envs, include_privileged=False):
    from skills.walk.env import K1WalkEnv
    from skills.walk.config import WalkConfig
    cfg = WalkConfig()
    if num_envs is not None:
        cfg.num_envs = num_envs
    env = K1WalkEnv(cfg=cfg, include_privileged=include_privileged)
    return env, cfg


def _build_standup(num_envs, include_privileged=False):
    from skills.standup.env import K1StandupEnv
    from skills.standup.config import StandupConfig
    cfg = StandupConfig()
    if num_envs is not None:
        cfg.num_envs = num_envs
    env = K1StandupEnv(cfg=cfg, include_privileged=include_privileged)
    return env, cfg


def _build_dribble(num_envs, include_privileged=False):
    from skills.dribble.env import K1DribbleEnv
    from skills.dribble.config import DribbleConfig
    cfg = DribbleConfig()
    if num_envs is not None:
        cfg.num_envs = num_envs
    env = K1DribbleEnv(cfg=cfg, include_privileged=include_privileged)
    return env, cfg


def _build_shoot(num_envs, include_privileged=False):
    from skills.shoot.env import K1ShootEnv
    from skills.shoot.config import ShootConfig
    cfg = ShootConfig()
    if num_envs is not None:
        cfg.num_envs = num_envs
    env = K1ShootEnv(cfg=cfg, include_privileged=include_privileged)
    return env, cfg


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
        # Multi-critic path (HoST): only when the env decomposes its reward
        # into ≥2 groups AND the config opts in. The other skills expose no
        # groups, so they transparently fall through to single-critic PPO.
        group_names = tuple(getattr(env, "CRITIC_GROUP_NAMES", ()))
        use_multi = bool(getattr(cfg, "use_multi_critic", False)) \
            and len(group_names) >= 2

        if use_multi:
            policy = create_policy(env.obs_dim, env.act_dim,
                                   n_critics=len(group_names))
            if init_from:
                # Warm-start: actor weights load (names match); the
                # per-group critics start fresh (names differ from the
                # single-critic checkpoint) — exactly what we want when
                # warm-starting a discovery run into multi-critic.
                load_checkpoint(init_from, policy)
            elif resume:
                load_checkpoint(resume, policy)
            from training.algorithms.ppo import train_ppo_multicritic_vec
            return train_ppo_multicritic_vec(
                env, policy, cfg, logger,
                phase=f"skill_{env.SKILL_NAME}",
                curriculum_stage=None,
                checkpoint_dir=checkpoint_dir,
                group_weights=getattr(cfg, "critic_group_weights", None),
                device=device,
                **algo_kwargs,
            )

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
    parser.add_argument(
        "--mode", choices=["single", "teacher", "student"], default="single",
        help="Training mode. 'single' = legacy proprio-only PPO/FlashSAC. "
             "'teacher' = PPO with privileged DR obs appended to inputs "
             "(used as the oracle for student distillation). "
             "'student' = behaviour-clone a frozen teacher checkpoint "
             "using proprio-only obs (sim-to-real path).")
    parser.add_argument(
        "--teacher-ckpt", type=str, default=None,
        help="Required for --mode student: path to a teacher checkpoint "
             "(produced by `--mode teacher`). The student inherits the "
             "same actor architecture but with proprio-only input width.")
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

    # Teacher needs privileged obs appended to its input; student does
    # not (it must be deployable on the real robot, which has no oracle
    # access to friction / mass / motor scales).
    include_privileged = (args.mode in ("teacher", "student"))
    builder = _SKILL_REGISTRY[args.skill]
    env, cfg = builder(num_envs, include_privileged=include_privileged)

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

    print(f"\n[train_skill] {args.skill} via {algorithm}  mode={args.mode}  "
          f"num_envs={num_envs}  obs_dim={env.obs_dim}  "
          f"act_dim={env.act_dim}  cmd={env.command_spec.dim}d  "
          f"privileged_dim={env.privileged_dim}  "
          f"device={device}  algo_kwargs={algo_kwargs or '{}'}\n")

    try:
        if args.mode == "student":
            # Behaviour-clone a frozen teacher into a proprio-only
            # student. The env is built with include_privileged=True
            # so the same env produces both flavours (proprio is the
            # leading slice; privileged is the trailing 8 dims).
            if not args.teacher_ckpt:
                raise SystemExit(
                    "--mode student requires --teacher-ckpt PATH "
                    "(checkpoint from a prior `--mode teacher` run).")
            import torch
            from training.algorithms.distillation import train_student
            # Use non_deployable_dim so skills with sim-only addons (e.g.
            # standup's contact obs) strip those too — not just the DR
            # tail. Falls back to privileged_dim for skills that don't
            # override it.
            non_deployable = getattr(env, "non_deployable_dim",
                                      env.privileged_dim)
            student_obs_dim = env.obs_dim - non_deployable
            teacher = create_policy(env.obs_dim, env.act_dim).to(device)
            load_checkpoint(args.teacher_ckpt, teacher)
            train_student(
                env_teacher=env, teacher_policy=teacher,
                student_obs_dim=student_obs_dim, act_dim=env.act_dim,
                total_env_steps=int(cfg.total_timesteps),
                n_steps=int(cfg.n_steps),
                learning_rate=float(cfg.learning_rate),
                logger=logger, skill=args.skill,
                checkpoint_dir=ckpt_dir, device=device,
            )
        else:
            # 'single' (legacy) and 'teacher' both use the standard PPO
            # path — the only difference is whether privileged obs are
            # appended (controlled by include_privileged on env build).
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
