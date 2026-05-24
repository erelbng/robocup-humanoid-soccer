"""
Gait-quality reward components for Phase 1.

These supplement the high-level dribble/shoot rewards with low-level
locomotion shaping: keep the trunk upright, swing feet at the right
height, alternate left/right contact, don't drift sideways. They are
written as pure functions taking numpy arrays so they can be called
from either the single-env or batched env path.

All components return positive numbers ∈ [0, ~1] (or larger if explicitly
documented). Multiply by weights at the call site rather than baking
magnitudes in here — that's what keeps `aggressiveness` tunable.

Smooth, low-magnitude shaping rewards are intentionally preferred over
sparse bonuses: they make convergence faster and more stable on PPO.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def upright_orientation(robot_quat: np.ndarray) -> float:
    """Cosine of trunk roll/pitch. 1.0 = perfectly upright.

    Uses the trunk Z-axis projection onto world Z, which is robust to yaw
    and well-behaved across the entire fall arc.
    """
    w, x, y, z = robot_quat
    return float(1.0 - 2.0 * (x * x + y * y))


def trunk_height_reward(robot_z: float, target_h: float = 0.55,
                        sigma: float = 0.12) -> float:
    """Gaussian on trunk z. Same shape used in standup but with a
    tighter sigma so the policy gets stronger gradients near target.
    """
    err = robot_z - target_h
    return math.exp(-(err ** 2) / (sigma ** 2))


def foot_alternation_reward(foot_contacts: np.ndarray,
                            prev_contacts: np.ndarray) -> float:
    """Reward when exactly one foot is on the ground (single-support phase)
    AND the active foot has changed since the previous step — i.e. the
    robot is alternating feet, not skating or shuffling.
    """
    if foot_contacts.shape[0] < 2:
        return 0.0
    single = int(foot_contacts[0] > 0.1) + int(foot_contacts[1] > 0.1)
    # Single support is what we want most of the time during walking;
    # double support is fine but no extra reward.
    base = 0.5 if single == 1 else (0.2 if single == 2 else 0.0)
    # Alternation bonus
    swapped = (foot_contacts[0] > 0.1) != (prev_contacts[0] > 0.1)
    return base + (0.3 if swapped and single == 1 else 0.0)


def feet_clearance_reward(feet_heights: np.ndarray, target_h: float = 0.04,
                          sigma: float = 0.02) -> float:
    """Reward the SWING foot for reaching ~target_h above the ground while
    leaving the stance foot alone. Assumes both feet roughly at z>=0.
    """
    if feet_heights.shape[0] < 2:
        return 0.0
    swing_h = float(feet_heights.max())
    err = swing_h - target_h
    return math.exp(-(err ** 2) / (sigma ** 2))


def angular_velocity_penalty(angvel: np.ndarray) -> float:
    """Penalise excessive trunk rotation. Returns POSITIVE magnitude — the
    caller subtracts."""
    return float(np.sum(angvel * angvel))


def joint_velocity_penalty(joint_vel: np.ndarray) -> float:
    """Sum of squared joint velocities (energy proxy)."""
    return float(np.sum(joint_vel * joint_vel))


def action_smoothness_penalty(action: np.ndarray,
                              prev_action: np.ndarray) -> float:
    """L2 of action delta. Smoother policies sim2real-transfer better and
    train faster on PPO."""
    return float(np.sum((action - prev_action) ** 2))


def lateral_drift_penalty(robot_vel: np.ndarray, robot_quat: np.ndarray,
                          commanded_vy: float) -> float:
    """Penalise lateral velocity in the body frame that does NOT match
    the command. Helps keep walking straight.
    """
    w, x, y, z = robot_quat
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(-yaw), math.sin(-yaw)
    vy_body = s * robot_vel[0] + c * robot_vel[1]
    return float((vy_body - commanded_vy) ** 2)


def joint_limit_penalty(joint_pos: np.ndarray,
                        lower: np.ndarray,
                        upper: np.ndarray,
                        margin: float = 0.1) -> float:
    """Soft penalty when any joint approaches its limit. Linear ramp from
    `margin` rad inside the limit to the limit itself.
    """
    near_lower = np.clip(lower - joint_pos + margin, 0.0, margin) / margin
    near_upper = np.clip(joint_pos - upper + margin, 0.0, margin) / margin
    return float(np.sum(near_lower * near_lower + near_upper * near_upper))


def compute_gait_shaping(
    *,
    robot_quat: np.ndarray,
    robot_z: float,
    robot_vel: np.ndarray,
    robot_angvel: np.ndarray,
    joint_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray,
    foot_contacts: np.ndarray,
    prev_foot_contacts: np.ndarray,
    feet_heights: np.ndarray,
    commanded_vy: float = 0.0,
    weights: dict = None,
) -> Tuple[float, dict]:
    """Combine all gait shaping into a single scalar.

    Default weights tuned for a small but meaningful shaping signal on
    top of the high-level task reward. Override by passing `weights=`.
    """
    w = {
        "upright": 0.8,
        "height": 0.5,
        "alt": 0.6,
        "clearance": 0.3,
        "angvel": 0.02,
        "joint_vel": 0.001,
        "smooth": 0.05,
        "lateral": 0.10,
        "joint_lim": 0.5,
    }
    if weights:
        w.update(weights)

    parts = {
        "gait/upright": upright_orientation(robot_quat),
        "gait/height": trunk_height_reward(robot_z),
        "gait/alt": foot_alternation_reward(foot_contacts, prev_foot_contacts),
        "gait/clearance": feet_clearance_reward(feet_heights),
        "gait/angvel": angular_velocity_penalty(robot_angvel),
        "gait/joint_vel": joint_velocity_penalty(joint_vel),
        "gait/smooth": action_smoothness_penalty(action, prev_action),
        "gait/lateral": lateral_drift_penalty(robot_vel, robot_quat,
                                              commanded_vy),
        "gait/joint_lim": joint_limit_penalty(joint_pos, joint_limits_lower,
                                              joint_limits_upper),
    }
    r = (w["upright"] * parts["gait/upright"]
         + w["height"] * parts["gait/height"]
         + w["alt"] * parts["gait/alt"]
         + w["clearance"] * parts["gait/clearance"]
         - w["angvel"] * parts["gait/angvel"]
         - w["joint_vel"] * parts["gait/joint_vel"]
         - w["smooth"] * parts["gait/smooth"]
         - w["lateral"] * parts["gait/lateral"]
         - w["joint_lim"] * parts["gait/joint_lim"])
    return r, parts
