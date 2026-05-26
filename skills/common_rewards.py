"""Reward helpers shared across skills.

Each skill keeps its own composite `compute_*_reward` (they have
different shaping mixtures and different obs add-ons), but pieces that
are skill-agnostic — like tracking a head-look command from the vision
system — live here so the math has exactly one home.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def head_tracking_reward(
    joint_pos: np.ndarray,                 # (N, num_dofs) — full joint vector
    head_joint_indices: Tuple[int, ...],   # positions of head_yaw/head_pitch
                                            # in `joint_pos`; for K1 = (0, 1)
    head_commands: np.ndarray,             # (N, len(head_joint_indices))
    sigma: float = 0.20,                   # rad; ~11° at e^(-1)
) -> np.ndarray:
    """exp-shaped reward on how close the head joints sit to their target.

    Returns (N,) ∈ [0, 1]. We compare the actual joint positions
    (already in radians, no normalization) against the commanded
    targets (also radians). Sigma is the rms error at which reward is
    ≈ 0.37 — chosen so "roughly looking the right way" gets credit but
    pin-point tracking is what maxes it out.
    """
    if head_commands.size == 0 or len(head_joint_indices) == 0:
        return np.zeros(joint_pos.shape[0], dtype=np.float32)
    idx = list(head_joint_indices)
    actual = joint_pos[:, idx]                            # (N, K)
    err2 = np.sum((actual - head_commands) ** 2, axis=1)
    return np.exp(-err2 / (sigma ** 2)).astype(np.float32)


def joint_pose_deviation(
    joint_pos: np.ndarray,                 # (N, num_dofs)
    joint_indices: Tuple[int, ...],        # which DOFs to regularize
    target_pose: np.ndarray,               # (num_dofs,) full default pose
) -> np.ndarray:
    """Sum-of-squares deviation of `joint_indices` from `target_pose`.

    The standard legged_gym `dof_pos` regulariser. Used here to keep
    the arms hanging naturally at the sides — we pass
    `robot_cfg.arm_joint_indices` and `robot_cfg.default_joint_pos`, so
    any drift away from the rest pose (shoulders flailing, elbows
    splayed) accrues a small cost. Weight small so this doesn't fight
    velocity/foot-clearance tracking — it only takes over when those
    are tied.

    Returns (N,) float32, sum of squared per-joint errors over the
    selected indices.
    """
    if len(joint_indices) == 0:
        return np.zeros(joint_pos.shape[0], dtype=np.float32)
    idx = list(joint_indices)
    target = np.asarray(target_pose, dtype=np.float32)[idx]   # (K,)
    actual = joint_pos[:, idx]                                 # (N, K)
    err2 = np.sum((actual - target[None, :]) ** 2, axis=1)
    return err2.astype(np.float32)


__all__ = ["head_tracking_reward", "joint_pose_deviation"]
