"""Dribble reward = walk reward + ball-tracking terms.

Composition strategy: reuse the walk reward helpers (so the robot keeps
caring about velocity tracking, posture, gait) and add ball-specific
terms on top. The ball terms use the LAST TWO command dims
(ball_off_x, ball_off_y) as the target relative to the robot, in body
frame.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from skills.common_obs import body_frame_velocity
from skills.common_rewards import head_tracking_reward, joint_pose_deviation
from skills.walk.rewards import (action_smoothness, base_motion_penalty,
                                  dof_acc_penalty, energy_penalty,
                                  foot_clearance_reward, height_reward,
                                  torque_penalty, track_ang_vel,
                                  track_lin_vel, upright_reward)


# ─── ball-specific helpers ─────────────────────────────────────────────


def ball_offset_reward(ball_pos_body: np.ndarray,
                       target_off: np.ndarray,
                       sigma: float = 0.20) -> np.ndarray:
    """exp-shaped reward on the 2D distance between the ball's body-frame
    position and the commanded target offset.

    Args:
        ball_pos_body: (N, 2) body-frame ball xy.
        target_off: (N, 2) commanded ball offset (body frame).
    """
    err = np.sum((ball_pos_body - target_off) ** 2, axis=1)
    return np.exp(-err / (sigma ** 2)).astype(np.float32)


def ball_velocity_reward(ball_vel_body: np.ndarray,
                         cmd_vx: np.ndarray, cmd_vy: np.ndarray,
                         sigma: float = 0.6) -> np.ndarray:
    """Reward ball moving in the commanded direction. Sigma is wider
    than the velocity-tracking reward because the ball's velocity is
    noisier (contact impulses, friction)."""
    err = (ball_vel_body[:, 0] - cmd_vx) ** 2 \
          + (ball_vel_body[:, 1] - cmd_vy) ** 2
    return np.exp(-err / (sigma ** 2)).astype(np.float32)


def ball_lost_mask(ball_pos_body: np.ndarray, max_distance: float
                   ) -> np.ndarray:
    return (np.linalg.norm(ball_pos_body, axis=1)
            > float(max_distance)).astype(np.float32)


# ─── helpers used by the env to project ball state into body frame ────


def ball_state_body_frame(robot_pos: np.ndarray, robot_quat: np.ndarray,
                          ball_pos: np.ndarray, ball_vel: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Transform world-frame ball pos/vel into the robot's yaw-aligned
    body frame. Returns ((N, 3), (N, 3)).

    We use yaw-only rotation so the ball-relative position doesn't flip
    sign when the robot pitches forward — same convention as the velocity
    tracking reward.
    """
    rel_pos_world = ball_pos - robot_pos
    pos_body = body_frame_velocity(robot_quat, rel_pos_world)
    vel_body = body_frame_velocity(robot_quat, ball_vel)
    return pos_body, vel_body


# ─── composite reward ──────────────────────────────────────────────────


def compute_dribble_reward(
    *,
    root_pos: np.ndarray, root_quat: np.ndarray,
    root_lin_vel: np.ndarray, root_ang_vel: np.ndarray,
    jpos: np.ndarray, jvel: np.ndarray, prev_jvel: np.ndarray,
    action: np.ndarray, prev_action: np.ndarray,
    applied_torque: np.ndarray,
    feet_z: np.ndarray, contact_mask: np.ndarray,
    ball_pos: np.ndarray, ball_vel: np.ndarray,
    commands: np.ndarray,    # (N, 7) — walk(5) + ball_off(2)
    weights,                  # DribbleRewardWeights
    head_commands: np.ndarray = None,    # (N, 2) optional
    head_joint_indices: tuple = (),
    arm_joint_indices: tuple = (),
    default_joint_pos: np.ndarray = None,
    ball_lost_distance: float = 2.0,
    dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compose walk-style reward + ball terms.

    Returns (reward[N], ball_lost_now[N] bool, components_dict). The env
    consumes `ball_lost_now` to terminate the episode.
    """
    cmd_vx = commands[:, 0]
    cmd_vy = commands[:, 1]
    cmd_vyaw = commands[:, 2]
    cmd_clearance = commands[:, 3]
    cmd_ball_off = commands[:, 5:7]  # (N, 2)

    # ── walk-style components ────────────────────────────────────
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

    # ── ball components ──────────────────────────────────────────
    ball_pos_body, ball_vel_body = ball_state_body_frame(
        root_pos, root_quat, ball_pos, ball_vel)
    ball_off_r = ball_offset_reward(ball_pos_body[:, :2], cmd_ball_off)
    ball_vel_r = ball_velocity_reward(ball_vel_body, cmd_vx, cmd_vy)
    lost = ball_lost_mask(ball_pos_body[:, :2], ball_lost_distance)

    # Head-look tracking — optional.
    head = np.zeros(root_pos.shape[0], dtype=np.float32)
    if head_commands is not None and head_commands.size > 0 \
            and len(head_joint_indices) > 0:
        head = head_tracking_reward(jpos, head_joint_indices, head_commands)

    # Arm-pose deviation — keeps arms hanging naturally at the sides.
    arm_dev = np.zeros(root_pos.shape[0], dtype=np.float32)
    if len(arm_joint_indices) > 0 and default_joint_pos is not None:
        arm_dev = joint_pose_deviation(jpos, arm_joint_indices,
                                        default_joint_pos)

    # ── compose ──────────────────────────────────────────────────
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
        + w.ball_offset * ball_off_r
        + w.ball_velocity * ball_vel_r
        + w.ball_lost * lost
    ).astype(np.float32)

    components = {
        # walk-style
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
        # ball
        "ball_offset_r": float(np.mean(ball_off_r)),
        "ball_velocity_r": float(np.mean(ball_vel_r)),
        "ball_lost_rate": float(np.mean(lost)),
        "ball_dist_mean": float(np.mean(
            np.linalg.norm(ball_pos_body[:, :2], axis=1))),
        "mean_robot_z": float(np.mean(root_pos[:, 2])),
        "mean_reward": float(np.mean(r)),
    }
    return r, lost.astype(bool), components
