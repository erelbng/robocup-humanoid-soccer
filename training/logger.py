"""
Unified training logger.

Wraps TensorBoard (default) and W&B (opt-in via --wandb) behind a single
interface so the trainer just calls `logger.log_scalar(name, val, step)`
and `logger.log_video(name, frames, step)` without caring which backends
are active.

Design notes:
  * TensorBoard is always-on if the package is installed. We default to
    logs/tb/<run_name>/. No external services required.
  * W&B is opt-in. If the caller passes `use_wandb=True` but the package
    is missing, we warn and continue with TB only — never crash the run
    over telemetry.
  * Videos are normalised to (T, H, W, 3) uint8 numpy and passed to both
    backends in their preferred format (TB needs (T, 3, H, W); W&B takes
    a file path or ndarray).

The logger owns the run name + log directory so the trainer doesn't have
to pass them around to W&B and TB separately.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype.kind == "f":
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


class TrainingLogger:
    """Combined TensorBoard + (optional) W&B logger.

    Usage:
        logger = TrainingLogger(run_name="phase1_walk",
                                log_root="logs",
                                use_wandb=False,
                                config={"phase": "phase1", "stage": "walk"})
        logger.log_scalar("train/reward", 1.23, step=42)
        logger.log_video("eval/rollout", frames_uint8, step=42, fps=30)
        logger.close()
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        log_root: str = "logs",
        use_wandb: bool = False,
        wandb_project: str = "robocup-humanoid-soccer",
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[Iterable[str]] = None,
        config: Optional[dict] = None,
    ):
        if run_name is None:
            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_name = run_name
        self.log_dir = Path(log_root) / "tb" / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = Path(log_root) / "videos" / run_name
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # ── TensorBoard ─────────────────────────────────────────────
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=str(self.log_dir))
            print(f"[logger] TensorBoard → {self.log_dir} "
                  f"(`tensorboard --logdir {self.log_dir.parent}`)")
        except ImportError:
            print("[logger] tensorboard not installed; only console logging")

        # ── W&B ────────────────────────────────────────────────────
        self._wandb = None
        self._wandb_run = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    tags=list(wandb_tags) if wandb_tags else None,
                    config=config or {},
                    save_code=True,
                    dir=str(Path(log_root) / "wandb"),
                )
                print(f"[logger] W&B run: {self._wandb_run.name} "
                      f"({self._wandb_run.url})")
            except ImportError:
                print("[logger] wandb not installed; "
                      "install with `pip install -e .[wandb]`")
            except Exception as e:
                print(f"[logger] wandb init failed: {e} — continuing without")

        # Save config to disk too so a TB-only run has full context
        if config is not None:
            self._dump_config(config)

    # ── scalar / dict / histogram ──────────────────────────────────

    def log_scalar(self, name: str, value: float, step: int):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if self._tb is not None:
            self._tb.add_scalar(name, value, step)
        if self._wandb is not None and self._wandb_run is not None:
            self._wandb.log({name: value}, step=step)

    def log_scalars(self, metrics: dict, step: int):
        """Log many scalars in one call (one TB / one W&B write)."""
        clean = {}
        for k, v in metrics.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                continue
        if self._tb is not None:
            for k, v in clean.items():
                self._tb.add_scalar(k, v, step)
        if self._wandb is not None and self._wandb_run is not None:
            self._wandb.log(clean, step=step)

    def log_hist(self, name: str, values, step: int):
        if self._tb is not None:
            try:
                self._tb.add_histogram(name, np.asarray(values), step)
            except Exception:
                pass
        if self._wandb is not None and self._wandb_run is not None:
            try:
                self._wandb.log({name: self._wandb.Histogram(np.asarray(values))},
                                step=step)
            except Exception:
                pass

    # ── videos ─────────────────────────────────────────────────────

    def log_video(self, name: str, frames, step: int, fps: int = 30):
        """Log a video clip.

        `frames` is anything coercible to a numpy array shaped
        (T, H, W, 3) — accepts uint8 or float-in-[0,1]. The clip is saved
        to disk as MP4 (so it's available outside TB/W&B too) and pushed
        to both backends.
        """
        try:
            arr = np.stack([np.asarray(f) for f in frames]) if isinstance(
                frames, list
            ) else np.asarray(frames)
        except Exception as e:
            print(f"[logger] log_video({name}): bad frames ({e})")
            return
        if arr.ndim != 4 or arr.shape[-1] != 3:
            print(f"[logger] log_video({name}): expected (T,H,W,3), "
                  f"got {arr.shape}")
            return
        arr = _to_uint8(arr)

        # Save to disk first
        safe = name.replace("/", "_")
        path = self.video_dir / f"{safe}_step{step}.mp4"
        try:
            import imageio.v2 as imageio
            imageio.mimwrite(str(path), arr, fps=fps,
                             codec="libx264", quality=8)
        except Exception as e:
            path = None
            print(f"[logger] video write failed: {e}")

        # TensorBoard wants (N, T, C, H, W) float
        if self._tb is not None:
            try:
                t = arr.transpose(0, 3, 1, 2)[None, ...]   # (1, T, 3, H, W)
                self._tb.add_video(name, t, global_step=step, fps=fps)
            except Exception as e:
                print(f"[logger] TB add_video failed: {e}")

        # W&B accepts a path
        if (self._wandb is not None and self._wandb_run is not None
                and path is not None):
            try:
                self._wandb.log(
                    {name: self._wandb.Video(str(path), fps=fps, format="mp4")},
                    step=step,
                )
            except Exception as e:
                print(f"[logger] W&B log_video failed: {e}")

    # ── lifecycle ──────────────────────────────────────────────────

    def close(self):
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
        if self._wandb is not None and self._wandb_run is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass

    # ── internals ──────────────────────────────────────────────────

    def _dump_config(self, config: dict):
        import json
        out = self.log_dir / "config.json"
        try:
            with open(out, "w") as f:
                json.dump(_jsonable(config), f, indent=2, default=str)
        except Exception:
            pass


def _jsonable(o: Any):
    """Recursive cast so dataclasses / numpy / paths survive json.dump."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "__dataclass_fields__"):
        return {k: _jsonable(getattr(o, k))
                for k in o.__dataclass_fields__}
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o
