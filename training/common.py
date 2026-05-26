"""Shared helpers for training entry points.

Anything reusable across `train.py` (legacy phase 1/2 dispatch) and the
upcoming `train_skill.py` / `train_orchestrator.py` lives here:

* `setup_logger`          — TensorBoard (+ optional W&B) logger.
* `load_checkpoint`       — algorithm-agnostic state-dict loader with
                            partial-load fallback (Phase 1 → Phase 2,
                            standup → walk warm-start, …).
* `create_policy`         — default PPOActorCritic builder.
* `resolve_device`        — `--device {auto,cpu,gpu}` → (device, preset).
* `record_eval_video`     — short rollout clip in a video file.
* `DEVICE_PRESETS`        — env / algo defaults per device kind.

`train.py` keeps the legacy curriculum + phase dispatch and re-exports
these for backwards compatibility with `evaluation/evaluate.py`. The new
per-skill training scripts import from here directly.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np


# ─── device presets ────────────────────────────────────────────────────


DEVICE_PRESETS = {
    "cpu": {
        "vec_num_envs": 64,
        "flashsac": {"buffer_capacity": 200_000, "batch_size": 512,
                     "gradient_steps": 1},
        "ppo": {},
    },
    "gpu": {
        "vec_num_envs": 1024,
        # buffer_capacity: 2M keeps ~488 env-steps of history with
        # n_envs=4096 (vs 244 with 1M). Fits in ~1.5 GB VRAM for
        # obs_dim=83. batch_size: 2048 better saturates GPU matrix ops
        # vs 1024. gradient_steps: 4 baseline for n_envs=1024; auto-
        # scaled up inside train_flashsac_vec when n_envs is larger.
        "flashsac": {"buffer_capacity": 2_000_000, "batch_size": 2048,
                     "gradient_steps": 4},
        "ppo": {},
    },
}


def resolve_device(device_flag: str):
    """Resolve --device {auto,cpu,gpu} into (torch.device, preset, kind).

    `auto` detects CUDA. Explicit `gpu` warns and falls back to CPU if
    CUDA isn't available — silent runtime failures inside Genesis hours
    into a queued job are the worst possible outcome.
    """
    import torch
    has_cuda = torch.cuda.is_available()
    if device_flag == "auto":
        kind = "gpu" if has_cuda else "cpu"
    elif device_flag == "gpu":
        if not has_cuda:
            print("[device] --device gpu requested but CUDA isn't "
                  "available; falling back to CPU. Set "
                  "CUDA_VISIBLE_DEVICES or check your torch install if "
                  "this is unexpected.")
            kind = "cpu"
        else:
            kind = "gpu"
    else:
        kind = "cpu"

    device = torch.device("cuda" if kind == "gpu" else "cpu")
    preset = DEVICE_PRESETS[kind]
    return device, preset, kind


# ─── logger ────────────────────────────────────────────────────────────


def setup_logger(run_name: str, log_root: str, wandb_project: str,
                 wandb_entity: Optional[str] = None,
                 wandb_tags: Optional[list] = None,
                 use_wandb: bool = False,
                 config: Optional[dict] = None):
    """TensorBoard (+ optional W&B) logger.

    `run_name` defines the on-disk subdirectory and the W&B run name.
    Callers usually want `f"{phase_or_skill}_{algorithm}_{timestamp}"`.
    """
    from training.logger import TrainingLogger
    return TrainingLogger(
        run_name=run_name,
        log_root=log_root,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_tags=list(wandb_tags or []),
        config=config or {},
    )


def default_run_name(name: str, algorithm: str = "ppo") -> str:
    return f"{name}_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}"


# ─── policy factory + checkpoint loading ───────────────────────────────


def create_policy(obs_dim: int, act_dim: int, **kwargs):
    """Default PPO actor-critic builder.

    Used by training entry points and by `evaluation/evaluate.py`.
    Returns None if torch isn't importable (CPU-only smoke environments).
    """
    try:
        from training.algorithms.networks import PPOActorCritic
    except ImportError:
        return None
    return PPOActorCritic(obs_dim, act_dim, **kwargs)


def load_checkpoint(path: str, policy, optimizer=None):
    """Load a checkpoint into `policy` (PPO or SAC actor).

    Falls back to a partial state-dict load when shapes don't match,
    useful when transferring across skill obs/act dims (e.g. walk →
    dribble after the obs add-on layer grows).
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
        # Partial load: only the keys whose shapes match.
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


# ─── eval video ────────────────────────────────────────────────────────


def record_eval_video(env, policy, device, name: str, step: int,
                      out_dir: str, max_frames: int = 300,
                      fps: int = 30) -> Optional[str]:
    """Record a short deterministic rollout to MP4.

    Works for both PPOActorCritic (`.act(obs, deterministic=True)`) and
    SACActor (`policy(obs, deterministic=True)`). Returns the output
    path or None if the env can't render / encoder is missing.
    """
    try:
        import torch
    except ImportError:
        return None
    os.makedirs(out_dir, exist_ok=True)
    frames = []

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs[0]

    for _ in range(max_frames):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device
                                ).unsqueeze(0)
        with torch.no_grad():
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

    if not frames:
        return None
    path = os.path.join(out_dir, f"{name}_step{step}.mp4")
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps)
        return path
    except Exception as e:
        print(f"[video] write failed: {e}")
        return None
