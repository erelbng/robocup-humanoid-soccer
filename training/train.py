"""Training entry point for RoboCup Humanoid Soccer RL.

Phase 1: Single-robot skills (stand → walk → dribble → shoot)
Phase 2: Multi-robot match with self-play

Algorithm selection: --algorithm ppo (default) | flashsac

The algorithm logic itself lives in `training/algorithms/`. This module
is responsible for: argparse, building the env, instantiating the
logger, and dispatching to the chosen trainer.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root for sibling imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (CHECKPOINTS_DIR, LOGS_DIR, VIDEOS_DIR,
                            Phase1Config, Phase2Config, ProjectConfig)


# ─── logger setup ──────────────────────────────────────────────────────


def setup_logger(config: ProjectConfig, phase: str, run_name: str = None,
                 use_wandb: bool = False, algorithm: str = "ppo"):
    """Create a TensorBoard (+ optional W&B) logger."""
    from training.logger import TrainingLogger

    return TrainingLogger(
        run_name=run_name or f"{phase}_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}",
        log_root=str(LOGS_DIR),
        use_wandb=use_wandb,
        wandb_project=config.wandb.project,
        wandb_entity=config.wandb.entity,
        wandb_tags=config.wandb.tags + [phase, algorithm],
        config={
            "phase": phase, "algorithm": algorithm, "seed": config.seed,
            "robot": config.robot.__dict__,
            "phase1": (config.phase1.__dict__ if phase == "phase1"
                       else config.phase2.__dict__),
        },
    )


# ─── checkpoint loading (algorithm-agnostic) ───────────────────────────


def load_checkpoint(path: str, policy, optimizer=None):
    """Load a checkpoint into `policy`. Works for both PPO and SAC actors
    by checking which state_dict key is present.
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not available")
        return None
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("policy_state_dict") or ckpt.get("actor_state_dict")
    if sd is None:
        print(f"[load] checkpoint {path} has no policy/actor state_dict")
        return None
    try:
        policy.load_state_dict(sd)
    except RuntimeError as e:
        # Partial load: only the keys whose shapes match. Useful when
        # transferring from Phase 1 (obs_dim 83) into Phase 2 (156).
        model_sd = policy.state_dict()
        ok = {k: v for k, v in sd.items()
              if k in model_sd and v.shape == model_sd[k].shape}
        model_sd.update(ok)
        policy.load_state_dict(model_sd)
        print(f"[load] partial load: {len(ok)}/{len(sd)} layers matched ({e})")
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[load] optimizer state mismatched, skipping: {e}")
    print(f"[load] {path}  step={ckpt.get('step', '?')}  "
          f"algo={ckpt.get('algorithm', '?')}")
    return ckpt


# ─── video recording ───────────────────────────────────────────────────


def record_eval_video(env, policy, device, phase: str, step: int,
                      max_frames: int = 300) -> Optional[str]:
    """Record a short evaluation video. Works for both PPO and SAC actors."""
    try:
        import torch
    except ImportError:
        return None
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    frames = []

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs[0]

    for _ in range(max_frames):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device
                                ).unsqueeze(0)
        with torch.no_grad():
            # Both PPOActorCritic and SACActor expose a deterministic
            # acting interface — `act(obs, deterministic=True)` for PPO,
            # `(obs, deterministic=True)` for SAC.
            if hasattr(policy, "act"):
                action, _, _ = policy.act(obs_t, deterministic=True)
            else:
                action, _ = policy(obs_t, deterministic=True)
        action_np = action.squeeze(0).cpu().numpy()
        obs, _, done, _ = env.step(action_np)
        if isinstance(obs, dict):
            obs = obs[0]
        frame = env.render_frame() if hasattr(env, "render_frame") else None
        if frame is not None:
            frames.append(frame)
        if isinstance(done, bool) and done:
            break
        if isinstance(done, np.ndarray) and done.any():
            break

    if frames:
        path = os.path.join(VIDEOS_DIR, f"{phase}_step{step}.mp4")
        try:
            import imageio
            imageio.mimwrite(path, frames, fps=30)
            return path
        except Exception as e:
            print(f"[video] write failed: {e}")
    return None


# ─── algorithm dispatch ────────────────────────────────────────────────


def _train_phase(config, env_factory, phase_cfg, logger, phase: str,
                 algorithm: str, curriculum_stage: Optional[str],
                 resume_from: Optional[str] = None,
                 init_actor=None,
                 device=None,
                 algo_kwargs: Optional[dict] = None):
    """Build an env + policy, run one stage of training, return the
    trained policy (or actor).

    `device`: torch.device (forced if not None; trainers auto-detect otherwise).
    `algo_kwargs`: per-algorithm overrides forwarded to the trainer (e.g.
    buffer_capacity / batch_size for FlashSAC).
    """
    algo_kwargs = dict(algo_kwargs or {})
    env = env_factory(curriculum_stage)

    if algorithm == "ppo":
        from training.algorithms.ppo import (
            train_ppo, train_ppo_vec,
        )
        from training.algorithms.networks import PPOActorCritic
        policy = init_actor or PPOActorCritic(phase_cfg.obs_dim,
                                              phase_cfg.act_dim)
        if resume_from:
            load_checkpoint(resume_from, policy)
        if getattr(env, "num_envs", None):
            policy = train_ppo_vec(env, policy, phase_cfg, logger,
                                   phase, curriculum_stage,
                                   checkpoint_dir=str(CHECKPOINTS_DIR),
                                   device=device, **algo_kwargs)
        else:
            policy = train_ppo(env, policy, phase_cfg, logger,
                               phase, curriculum_stage,
                               checkpoint_dir=str(CHECKPOINTS_DIR),
                               device=device, **algo_kwargs)
        env.close()
        return policy

    elif algorithm == "flashsac":
        from training.algorithms.flashsac import train_flashsac_vec
        if not getattr(env, "num_envs", None):
            raise RuntimeError(
                "FlashSAC requires a vectorised env "
                "(--vec-num-envs N). It is off-policy with a replay "
                "buffer and trains far too slowly with a single env."
            )
        actor = train_flashsac_vec(env, None, phase_cfg, logger, phase,
                                   curriculum_stage,
                                   checkpoint_dir=str(CHECKPOINTS_DIR),
                                   device=device, **algo_kwargs)
        env.close()
        return actor

    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")


def train_phase1(config: ProjectConfig, resume_from: str = None,
                 use_wandb: bool = False, algorithm: str = "ppo",
                 device=None, algo_kwargs: Optional[dict] = None):
    """Phase 1 entry: curriculum-stage loop."""
    logger = setup_logger(config, "phase1", use_wandb=use_wandb,
                          algorithm=algorithm)

    use_vec = getattr(config.phase1, "use_vec_env", False)
    if use_vec:
        from envs.phase1_vec import K1DribbleShootVecEnv
        def _env_factory(stage):
            return K1DribbleShootVecEnv(
                num_envs=config.phase1.vec_num_envs,
                cfg=config.phase1, robot_cfg=config.robot,
                curriculum_stage=stage,
            )
    else:
        from envs.phase1_dribble_shoot import K1DribbleShootEnv
        def _env_factory(stage):
            return K1DribbleShootEnv(
                cfg=config.phase1, robot_cfg=config.robot,
                curriculum_stage=stage,
            )

    policy = None
    if config.phase1.use_curriculum:
        stages = config.phase1.curriculum_stages
        steps_per_stage = config.phase1.total_timesteps // len(stages)
        for stage in stages:
            print(f"\n>>> Curriculum Stage: {stage.upper()}")
            stage_cfg = Phase1Config(**config.phase1.__dict__)
            stage_cfg.total_timesteps = steps_per_stage
            policy = _train_phase(
                config, _env_factory, stage_cfg, logger,
                "phase1", algorithm, stage,
                resume_from=resume_from if stage == stages[0] else None,
                init_actor=policy,
                device=device, algo_kwargs=algo_kwargs,
            )
    else:
        policy = _train_phase(config, _env_factory, config.phase1, logger,
                              "phase1", algorithm, "full",
                              resume_from=resume_from,
                              device=device, algo_kwargs=algo_kwargs)

    logger.close()
    return policy


def train_phase2(config: ProjectConfig, phase1_checkpoint: str = None,
                 use_wandb: bool = False, algorithm: str = "ppo",
                 device=None, algo_kwargs: Optional[dict] = None):
    """Phase 2 entry: 4v4 match training, optionally seeded from Phase 1."""
    from envs.phase2_match import K1SoccerMatchEnv

    logger = setup_logger(config, "phase2", use_wandb=use_wandb,
                          algorithm=algorithm)

    def _env_factory(_stage):
        return K1SoccerMatchEnv(cfg=config.phase2, robot_cfg=config.robot)

    print(f"\n>>> Phase 2: Match Training "
          f"({config.phase2.players_per_team}v{config.phase2.players_per_team})  "
          f"algorithm={algorithm}")

    policy = _train_phase(config, _env_factory, config.phase2, logger,
                          "phase2", algorithm, None,
                          resume_from=phase1_checkpoint,
                          device=device, algo_kwargs=algo_kwargs)
    logger.close()
    return policy


# ─── CLI ───────────────────────────────────────────────────────────────


# Hyperparameter presets applied when the user picks `--device`.
# Anything the user passes explicitly on the CLI overrides the preset.
_CPU_PRESET = {
    "vec_num_envs": 64,
    "flashsac": {"buffer_capacity": 200_000, "batch_size": 512,
                 "gradient_steps": 1},
    "ppo": {},   # PPO scales naturally with vec_num_envs
}
_GPU_PRESET = {
    "vec_num_envs": 1024,
    # buffer_capacity: 2M keeps ~488 env-steps of history with n_envs=4096
    # (vs 244 with 1M). Fits in ~1.5 GB VRAM for obs_dim=83.
    # batch_size: 2048 better saturates GPU matrix ops vs 1024.
    # gradient_steps: 4 baseline for n_envs=1024; auto-scaled up inside
    # train_flashsac_vec when n_envs is larger (e.g. → 16 for n_envs=4096).
    "flashsac": {"buffer_capacity": 2_000_000, "batch_size": 2048,
                 "gradient_steps": 4},
    "ppo": {},
}


def _resolve_device(device_flag: str):
    """Resolve --device {auto,cpu,gpu} into a (torch.device, preset) pair.

    `auto` detects CUDA. Explicit `gpu` warns and falls back to CPU if
    CUDA isn't available — we don't want a silent runtime failure inside
    Genesis hours into a queued job.
    """
    import torch
    has_cuda = torch.cuda.is_available()
    if device_flag == "auto":
        kind = "gpu" if has_cuda else "cpu"
    elif device_flag == "gpu":
        if not has_cuda:
            print("[device] --device gpu requested but CUDA isn't available; "
                  "falling back to CPU. Set CUDA_VISIBLE_DEVICES or check "
                  "your torch install if this is unexpected.")
            kind = "cpu"
        else:
            kind = "gpu"
    else:
        kind = "cpu"

    device = torch.device("cuda" if kind == "gpu" else "cpu")
    preset = _GPU_PRESET if kind == "gpu" else _CPU_PRESET
    return device, preset, kind


def main():
    parser = argparse.ArgumentParser(
        description="RoboCup Humanoid Soccer RL Training")
    parser.add_argument(
        "--phase", choices=["1", "2", "both"], default="both",
        help="Training phase (1=skills, 2=match, both)",
    )
    parser.add_argument(
        "--algorithm", choices=["ppo", "flashsac", "sac"], default="ppo",
        help="RL algorithm. 'sac' is an alias for 'flashsac'.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "gpu"], default="auto",
        help="Hardware preset. 'auto' detects CUDA. Sets the torch device "
             "and sensible defaults for vec_num_envs, replay buffer, and "
             "batch size. CLI flags override the preset.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--phase1-ckpt", type=str, default=None,
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
        "--aggressiveness", type=float, default=0.3,
        help="Aggressiveness level 0.0-1.0",
    )
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--vec-num-envs", type=int, default=None,
        help="Use vectorised Genesis env with this many parallel envs. "
             "FlashSAC REQUIRES this; PPO works with either.",
    )
    args = parser.parse_args()

    algorithm = "flashsac" if args.algorithm == "sac" else args.algorithm

    # Resolve --device first — its preset becomes the baseline that any
    # explicit CLI flag overrides.
    device, preset, device_kind = _resolve_device(args.device)
    algo_kwargs = dict(preset.get(algorithm, {}))
    print(f"[device] resolved: kind={device_kind}  torch={device}  "
          f"vec_num_envs={preset['vec_num_envs']}  "
          f"algo_kwargs={algo_kwargs or '{}'}")

    config = ProjectConfig()
    config.seed = args.seed
    # Apply device preset to the env config; --vec-num-envs below
    # overrides this if the user passed it explicitly.
    config.phase1.use_vec_env = True
    config.phase1.vec_num_envs = preset["vec_num_envs"]

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

    # FlashSAC always needs a vec env (already enabled above by the device
    # preset, but keep this as a defensive double-check).
    if algorithm == "flashsac" and not config.phase1.use_vec_env:
        print("[train] FlashSAC requires a vectorised env; "
              f"enabling with vec_num_envs={preset['vec_num_envs']}")
        config.phase1.use_vec_env = True

    np.random.seed(config.seed)

    phase1_ckpt = args.phase1_ckpt
    if args.phase in ("1", "both"):
        train_phase1(config, resume_from=args.resume,
                     use_wandb=use_wandb, algorithm=algorithm,
                     device=device, algo_kwargs=algo_kwargs)
        if phase1_ckpt is None and os.path.exists(CHECKPOINTS_DIR):
            ckpts = sorted(Path(CHECKPOINTS_DIR).glob("phase1_*.pt"))
            if ckpts:
                phase1_ckpt = str(ckpts[-1])

    if args.phase in ("2", "both"):
        train_phase2(config, phase1_checkpoint=phase1_ckpt,
                     use_wandb=use_wandb, algorithm=algorithm,
                     device=device, algo_kwargs=algo_kwargs)


if __name__ == "__main__":
    main()
