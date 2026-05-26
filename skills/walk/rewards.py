"""Walk reward function.

Decomposed into single-purpose helpers that operate on batched numpy
arrays. Each helper returns a (N,) reward in [0, 1] (for tracking
terms) or an unbounded penalty (for regularizers). The composer in
`compute_walk_reward` weights them per `WalkRewardWeights`.

We deliberately follow rsl_rl / Isaac Lab's locomotion reward shape:

* `exp(−err²/σ²)` shaping on velocity tracking — gives smooth gradients
  even when the policy is far from the commanded vel.
* Negative regularizers on dof_acc / torque / base motion — small
  weights so they don't dominate. These are the terms most likely to
  collapse training if mis-scaled.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from skills.common_obs import body_frame_velocity, projected_gravity
from skills.common_rewards import head_tracking_reward, joint_pose_deviation


# ─── primary tracking ──────────────────────────────────────────────────


def track_lin_vel(quat: np.ndarray, vel_world: np.ndarray,
                  cmd_vx: np.ndarray, cmd_vy: np.ndarray,
                  sigma: float = 0.25) -> np.ndarray:
    """exp-shaped reward on body-frame (vx, vy) tracking error."""
    v_body = body_frame_velocity(quat, vel_world)
    err = (v_body[:, 0] - cmd_vx) ** 2 + (v_body[:, 1] - cmd_vy) ** 2
    return np.exp(-err / (sigma ** 2)).astype(np.float32)


def track_ang_vel(ang_vel: np.ndarray, cmd_vyaw: np.ndarray,
                  sigma: float = 0.25) -> np.ndarray:
    """exp-shaped reward on yaw-rate tracking error."""
    err = (ang_vel[:, 2] - cmd_vyaw) ** 2
    return np.exp(-err / (sigma ** 2)).astype(np.float32)


# ─── posture ──────────────────────────────────────────────────────────


def upright_reward(quat: np.ndarray) -> np.ndarray:
    """1 when fully upright, 0 when on the side. We use the magnitude of
    the body-frame gravity z component as the upright signal — it's
    smoother than quat-based formulas near small tilts."""
    g = projected_gravity(quat)
    # |g_z| = 1 when upright (g points along body -z), 0 when sideways.
    return np.clip(-g[:, 2], 0.0, 1.0).astype(np.float32)


def height_reward(root_pos: np.ndarray, target_h: float = 0.55,
                  sigma: float = 0.12) -> np.ndarray:
    """Gaussian-shaped reward on trunk height."""
    err = root_pos[:, 2] - target_h
    return np.exp(-(err ** 2) / (sigma ** 2)).astype(np.float32)


# ─── gait shaping ──────────────────────────────────────────────────────


def foot_clearance_reward(feet_z: np.ndarray, cmd_clearance: np.ndarray,
                          contact_mask: np.ndarray,
                          sigma: float = 0.03) -> np.ndarray:
    """For swing feet (not in contact), reward matching the commanded
    swing height. `feet_z`: (N, 2) world-frame foot heights. Both feet
    in contact → 0 (no swing to evaluate)."""
    swing_mask = (~contact_mask).astype(np.float32)  # (N, 2)
    err = (feet_z - cmd_clearance[:, None]) ** 2     # (N, 2)
    per_foot = np.exp(-err / (sigma ** 2)) * swing_mask
    denom = swing_mask.sum(axis=1).clip(min=1e-6)
    return (per_foot.sum(axis=1) / denom).astype(np.float32)


def feet_air_time_reward(air_time_left: np.ndarray,
                         air_time_right: np.ndarray,
                         contact_just_now: np.ndarray,
                         target_air_time: float = 0.5) -> np.ndarray:
    """rsl_rl-style: reward longer air-times at the moment a foot touches
    down, capped at `target_air_time`. Encourages non-shuffling gaits.

    `air_time_*`: (N,) seconds since last contact.
    `contact_just_now`: (N, 2) bool — True for a foot that JUST made
    contact this step.
    """
    bonus_left = np.clip(air_time_left - 0.2, 0.0, target_air_time)
    bonus_right = np.clip(air_time_right - 0.2, 0.0, target_air_time)
    return (bonus_left * contact_just_now[:, 0].astype(np.float32)
            + bonus_right * contact_just_now[:, 1].astype(np.float32)
            ).astype(np.float32)


# ─── regularizers ─────────────────────────────────────────────────────


def action_smoothness(action: np.ndarray, prev_action: np.ndarray
                      ) -> np.ndarray:
    """Sum-of-squares change in action across consecutive steps."""
    return np.sum((action - prev_action) ** 2, axis=1).astype(np.float32)


def dof_acc_penalty(jvel: np.ndarray, prev_jvel: np.ndarray,
                    dt: float = 0.02) -> np.ndarray:
    """L2 of joint acceleration. dt converts from velocity to acc."""
    acc = (jvel - prev_jvel) / max(1e-6, dt)
    return np.sum(acc ** 2, axis=1).astype(np.float32)


def torque_penalty(applied_torque: np.ndarray) -> np.ndarray:
    return np.sum(applied_torque ** 2, axis=1).astype(np.float32)


def base_motion_penalty(ang_vel: np.ndarray, vel_world: np.ndarray,
                        quat: np.ndarray) -> np.ndarray:
    """Penalize excessive roll/pitch rate AND vertical velocity.

    Yaw rate is part of the command, so we exclude it.
    """
    rp_rate = ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2
    v_body = body_frame_velocity(quat, vel_world)
    vz = v_body[:, 2] ** 2
    return (rp_rate + vz).astype(np.float32)


def energy_penalty(jvel: np.ndarray, applied_torque: np.ndarray
                   ) -> np.ndarray:
    """Sum of |τ · q̇|, a proxy for instantaneous mechanical power."""
    return np.sum(np.abs(applied_torque * jvel), axis=1).astype(np.float32)


# ─── composite ─────────────────────────────────────────────────────────


def compute_walk_reward(
    *,
    root_pos: np.ndarray, root_quat: np.ndarray,
    root_lin_vel: np.ndarray, root_ang_vel: np.ndarray,
    jpos: np.ndarray, jvel: np.ndarray, prev_jvel: np.ndarray,
    action: np.ndarray, prev_action: np.ndarray,
    applied_torque: np.ndarray,
    feet_z: np.ndarray, contact_mask: np.ndarray,
    commands: np.ndarray,  # (N, 5) — vx, vy, vyaw, foot_clearance, step_freq
    weights,               # WalkRewardWeights
    head_commands: np.ndarray = None,    # (N, 2) target yaw/pitch — optional
    head_joint_indices: tuple = (),      # K1 → (0, 1) for AAHead_yaw/Head_pitch
    arm_joint_indices: tuple = (),       # K1 → (2..9) for shoulder/elbow joints
    default_joint_pos: np.ndarray = None,
    dt: float = 0.02,
) -> Tuple[np.ndarray, dict]:
    """Aggregate walk reward.

    Returns (reward[N], components_dict). Components are batch-averaged.
    """
    n = root_pos.shape[0]

    cmd_vx = commands[:, 0]
    cmd_vy = commands[:, 1]
    cmd_vyaw = commands[:, 2]
    cmd_clearance = commands[:, 3]

    lin = track_lin_vel(root_quat, root_lin_vel, cmd_vx, cmd_vy)
    ang = track_ang_vel(root_ang_vel, cmd_vyaw)
    up = upright_reward(root_quat)
    h = height_reward(root_pos)
    clearance = foot_clearance_reward(feet_z, cmd_clearance, contact_mask)

    smooth = action_smoothness(action, prev_action)
    acc = dof_acc_penalty(jvel, prev_jvel, dt)
    torq = torque_penalty(applied_torque)
    base = base_motion_penalty(root_ang_vel, root_lin_vel, root_quat)
    energy = energy_penalty(jvel, applied_torque)

    fallen = (root_pos[:, 2] < 0.30).astype(np.float32)
    alive = 1.0 - fallen

    # Head-look tracking — optional (only if both head_commands and the
    # joint indices were supplied). Zero contribution otherwise.
    head = np.zeros(root_pos.shape[0], dtype=np.float32)
    if head_commands is not None and head_commands.size > 0 \
            and len(head_joint_indices) > 0:
        head = head_tracking_reward(jpos, head_joint_indices, head_commands)

    # Arm-pose deviation — keeps arms near the default rest pose so the
    # policy doesn't learn to flail. Pure penalty.
    arm_dev = np.zeros(root_pos.shape[0], dtype=np.float32)
    if len(arm_joint_indices) > 0 and default_joint_pos is not None:
        arm_dev = joint_pose_deviation(jpos, arm_joint_indices,
                                        default_joint_pos)

    w = weights
    r = (
        w.track_lin_vel * lin
        + w.track_ang_vel * ang
        + w.upright * up
        + w.height * h
        + w.foot_clearance * clearance
        + w.alive * alive
        + getattr(w, "head_tracking", 0.0) * head
        - w.action_smoothness * smooth
        - w.dof_acc * acc
        - w.torque * torq
        - w.base_motion * base
        - w.energy * energy
        - getattr(w, "arm_pose", 0.0) * arm_dev
        + w.fall * fallen
    ).astype(np.float32)

    components = {
        "track_lin_vel": float(np.mean(lin)),
        "track_ang_vel": float(np.mean(ang)),
        "upright": float(np.mean(up)),
        "height": float(np.mean(h)),
        "foot_clearance": float(np.mean(clearance)),
        "head_tracking": float(np.mean(head)),
        "arm_pose_dev": float(np.mean(arm_dev)),
        "action_smooth": float(np.mean(smooth)),
        "dof_acc": float(np.mean(acc)),
        "torque": float(np.mean(torq)),
        "base_motion": float(np.mean(base)),
        "energy": float(np.mean(energy)),
        "alive_rate": float(np.mean(alive)),
        "fall_rate": float(np.mean(fallen)),
        "mean_robot_z": float(np.mean(root_pos[:, 2])),
        "mean_reward": float(np.mean(r)),
    }
    return r, components
