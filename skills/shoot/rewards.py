"""Shoot reward — sparse + dense composition.

Three layers:

1. **Dense approach shaping** — robot near ball, posture maintained.
   Keeps gradients informative when no kick has happened yet.

2. **Ball-to-target velocity term** — once the ball is moving, reward
   the velocity component pointing at the target. Dense and
   discriminative.

3. **Sparse kick pulse** — large one-shot bonus when the ball speed
   exceeds `kick_speed_threshold` AND the velocity is roughly toward
   the target. This is the terminal "you did it" signal. The env
   ends the episode on this event so the policy doesn't farm the
   approach reward forever.

`power_match` and `aim_accuracy` are gated on the kick event so the
reward landscape isn't dominated by aim/power signals during the
approach.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from skills.common_obs import body_frame_velocity, projected_gravity
from skills.common_rewards import head_tracking_reward, joint_pose_deviation


def _safe_unit(v: np.ndarray, axis: int = -1, eps: float = 1e-6
               ) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


# ─── helpers ──────────────────────────────────────────────────────────


def approach_reward(robot_pos: np.ndarray, ball_pos: np.ndarray,
                    sigma: float = 0.5) -> np.ndarray:
    d2 = np.sum((ball_pos[:, :2] - robot_pos[:, :2]) ** 2, axis=1)
    return np.exp(-d2 / (sigma ** 2)).astype(np.float32)


def upright_reward(quat: np.ndarray) -> np.ndarray:
    g = projected_gravity(quat)
    return np.clip(-g[:, 2], 0.0, 1.0).astype(np.float32)


def height_reward(root_pos: np.ndarray, target: float = 0.55,
                  sigma: float = 0.15) -> np.ndarray:
    err = root_pos[:, 2] - target
    return np.exp(-(err ** 2) / (sigma ** 2)).astype(np.float32)


def ball_to_target_velocity(ball_pos: np.ndarray, ball_vel: np.ndarray,
                            target_world: np.ndarray) -> np.ndarray:
    """Dot product of ball velocity (xy) with unit-direction-to-target.

    Positive when ball is moving toward target, negative when away.
    Magnitude scales with ball speed — that's intentional, faster shots
    in the right direction get more reward.
    """
    to_target = target_world[:, :2] - ball_pos[:, :2]
    unit = _safe_unit(to_target, axis=1)
    return np.sum(ball_vel[:, :2] * unit, axis=1).astype(np.float32)


def aim_error(ball_vel: np.ndarray, ball_pos: np.ndarray,
              target_world: np.ndarray) -> np.ndarray:
    """Angle between ball velocity direction (xy) and direction to
    target (xy). Returns radians ∈ [0, π]. Used only when |ball_vel|
    is non-trivial.
    """
    vel_unit = _safe_unit(ball_vel[:, :2], axis=1)
    to_target = _safe_unit(target_world[:, :2] - ball_pos[:, :2], axis=1)
    cos = np.clip(np.sum(vel_unit * to_target, axis=1), -1.0, 1.0)
    return np.arccos(cos).astype(np.float32)


def power_error(ball_vel: np.ndarray, cmd_power: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(ball_vel[:, :2], axis=1)
    return (speed - cmd_power).astype(np.float32)


# ─── composite ─────────────────────────────────────────────────────────


def compute_shoot_reward(
    *,
    root_pos: np.ndarray, root_quat: np.ndarray,
    root_lin_vel: np.ndarray, root_ang_vel: np.ndarray,
    jpos: np.ndarray, jvel: np.ndarray, prev_jvel: np.ndarray,
    action: np.ndarray, prev_action: np.ndarray,
    ball_pos: np.ndarray, ball_vel: np.ndarray,
    target_world: np.ndarray,           # (N, 3)
    commands: np.ndarray,                # (N, 3) — aim_angle, power, foot
    weights,                              # ShootRewardWeights
    head_commands: np.ndarray = None,    # (N, 2) optional
    head_joint_indices: tuple = (),
    arm_joint_indices: tuple = (),
    default_joint_pos: np.ndarray = None,
    kick_speed_threshold: float = 1.5,
    ball_lost_distance: float = 3.0,
    dt: float = 0.02,
    already_kicked: np.ndarray = None,   # (N,) bool — has the env already
                                         # collected the kick bonus?
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute shoot reward.

    Returns (reward[N], kick_event[N] bool, ball_lost[N] bool, components).

    `kick_event` is True for envs that JUST registered a clean kick (ball
    speed crosses threshold + moving toward target). The env terminates
    the episode on this event to avoid reward farming.
    """
    if already_kicked is None:
        already_kicked = np.zeros(root_pos.shape[0], dtype=bool)
    cmd_power = commands[:, 1]

    # ── dense components ────────────────────────────────────────
    approach = approach_reward(root_pos, ball_pos)
    up = upright_reward(root_quat)
    h = height_reward(root_pos)
    btt = ball_to_target_velocity(ball_pos, ball_vel, target_world)

    smooth = np.sum((action - prev_action) ** 2, axis=1).astype(np.float32)
    acc = (np.sum(((jvel - prev_jvel) / max(1e-6, dt)) ** 2, axis=1)
           .astype(np.float32))
    rp_rate = (root_ang_vel[:, 0] ** 2 + root_ang_vel[:, 1] ** 2
               ).astype(np.float32)

    fallen = (root_pos[:, 2] < 0.30).astype(np.float32)
    alive = 1.0 - fallen

    # ── kick detection ──────────────────────────────────────────
    ball_speed = np.linalg.norm(ball_vel[:, :2], axis=1)
    aim_err = aim_error(ball_vel, ball_pos, target_world)
    # Clean kick = high speed AND moving (mostly) toward target.
    clean_dir = aim_err < np.pi / 4   # within 45°
    kick_now = (ball_speed > kick_speed_threshold) & clean_dir \
               & (~already_kicked)

    # Power / aim only count on the step the kick fires (sparse pulse).
    pulse_mask = kick_now.astype(np.float32)
    pow_err = power_error(ball_vel, cmd_power)
    power_match_r = np.exp(-(pow_err ** 2) / (1.0 ** 2)).astype(np.float32)
    aim_acc_r = np.exp(-(aim_err ** 2) / ((np.pi / 6) ** 2)
                       ).astype(np.float32)

    # ── ball lost (too far away — never gonna make it) ─────────
    dist = np.linalg.norm(ball_pos[:, :2] - root_pos[:, :2], axis=1)
    ball_lost = (dist > float(ball_lost_distance))

    # Head-look tracking — optional.
    head = np.zeros(root_pos.shape[0], dtype=np.float32)
    if head_commands is not None and head_commands.size > 0 \
            and len(head_joint_indices) > 0:
        head = head_tracking_reward(jpos, head_joint_indices, head_commands)

    # Arm-pose deviation — keep arms tucked even during the kick.
    arm_dev = np.zeros(root_pos.shape[0], dtype=np.float32)
    if len(arm_joint_indices) > 0 and default_joint_pos is not None:
        arm_dev = joint_pose_deviation(jpos, arm_joint_indices,
                                        default_joint_pos)

    # ── compose ────────────────────────────────────────────────
    w = weights
    r = (
        w.approach_ball * approach
        + w.ball_to_target * btt
        + w.upright * up
        + w.height * h
        + w.alive * alive
        + getattr(w, "head_tracking", 0.0) * head
        - w.action_smoothness * smooth
        - w.dof_acc * acc
        - w.base_motion * rp_rate
        - getattr(w, "arm_pose", 0.0) * arm_dev
        + w.kick_event * pulse_mask
        + w.power_match * power_match_r * pulse_mask
        + w.aim_accuracy * aim_acc_r * pulse_mask
        + w.fall * fallen
        + w.ball_lost * ball_lost.astype(np.float32)
    ).astype(np.float32)

    components = {
        "approach": float(np.mean(approach)),
        "ball_to_target": float(np.mean(btt)),
        "upright": float(np.mean(up)),
        "height": float(np.mean(h)),
        "head_tracking": float(np.mean(head)),
        "arm_pose_dev": float(np.mean(arm_dev)),
        "ball_speed": float(np.mean(ball_speed)),
        "aim_err_deg": float(np.mean(np.rad2deg(aim_err))),
        "power_err": float(np.mean(np.abs(pow_err))),
        "kick_rate": float(np.mean(kick_now)),
        "fall_rate": float(np.mean(fallen)),
        "ball_lost_rate": float(np.mean(ball_lost)),
        "alive_rate": float(np.mean(alive)),
        "mean_reward": float(np.mean(r)),
    }
    return r, kick_now, ball_lost, components
