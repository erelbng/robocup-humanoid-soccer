"""Batched standup reward.

Single-purpose: get the trunk upright and at standing height starting
from a fallen pose. No velocity tracking, no gait shaping. The reward
is dominated by upright + height; smoothness/energy are soft regulariz-
ers; the success bonus is a one-shot terminal pulse to make "complete
the task" >> "stay close to it forever".
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from skills.common_obs import projected_gravity


def upright_signal(quat: np.ndarray) -> np.ndarray:
    """1 when fully upright, 0 sideways, −1 fully inverted.

    Uses the body-frame world-Z direction. When the body is upright,
    its z-axis aligns with world z → −g_z = 1. When upside-down,
    −g_z = −1. Linear in the cosine of the trunk tilt — smoother than
    quat-magnitude formulas near small tilts.
    """
    g = projected_gravity(quat)
    return (-g[:, 2]).astype(np.float32)


def height_signal(root_z: np.ndarray, target: float = 0.55,
                  sigma: float = 0.15) -> np.ndarray:
    err = root_z - target
    return np.exp(-(err ** 2) / (sigma ** 2)).astype(np.float32)


def success_mask(quat: np.ndarray, root_z: np.ndarray,
                 target_h: float = 0.55,
                 upright_threshold: float = 0.92) -> np.ndarray:
    return (upright_signal(quat) > upright_threshold) \
           & (root_z > target_h - 0.10)


def compute_standup_reward(
    *,
    root_pos: np.ndarray, root_quat: np.ndarray,
    joint_vel: np.ndarray,
    action: np.ndarray, prev_action: np.ndarray,
    weights,
    target_height: float = 0.55,
    upright_threshold: float = 0.92,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute the batched reward.

    Returns (reward[N], success_mask[N] bool, components_dict). The
    `success_mask` is also exposed because the env uses it as a terminal
    condition (one-shot bonus paid AND episode ends).
    """
    up = upright_signal(root_quat)
    h = height_signal(root_pos[:, 2], target=target_height)
    energy = np.sum(joint_vel ** 2, axis=1).astype(np.float32)
    smooth = np.sum((action - prev_action) ** 2, axis=1).astype(np.float32)
    success = success_mask(root_quat, root_pos[:, 2],
                           target_h=target_height,
                           upright_threshold=upright_threshold)

    # Clamp upright to [0, 1] for the reward (negative contribution is
    # implicit — being further from upright just yields less reward).
    up_pos = np.clip(up, 0.0, 1.0)

    w = weights
    r = (w.upright * up_pos
         + w.height * h
         - w.energy * energy
         - w.action_smoothness * smooth
         + w.success_bonus * success.astype(np.float32)
         ).astype(np.float32)

    components = {
        "upright": float(np.mean(up_pos)),
        "upright_raw": float(np.mean(up)),
        "height": float(np.mean(h)),
        "energy": float(np.mean(energy)),
        "action_smooth": float(np.mean(smooth)),
        "success_rate": float(np.mean(success)),
        "mean_robot_z": float(np.mean(root_pos[:, 2])),
        "mean_reward": float(np.mean(r)),
    }
    return r, success, components
