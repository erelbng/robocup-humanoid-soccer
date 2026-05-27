"""Batched standup reward — speed + stability-heavy composition.

Design goals:
  * One signal for speed (dense time penalty + time-scaled terminal bonus
    paid at sustained-success). PPO sees both a per-step gradient AND a
    big terminal pulse, so "finish fast" dominates.
  * Multiple stability terms — sway, jitter, drift, quiet limbs, clean
    upright — that all vanish at the standing equilibrium so they don't
    pull the optimum away from "upright + still".
  * Phase-gated drift / joint-vel penalties: only active once the robot
    is near upright. We don't want to punish needed motion during the
    recovery itself, only twitching once the standup is essentially done.
  * All bounded positive terms in [0, 1]; quadratic penalties → smooth
    gradient everywhere; no step functions in the per-step reward.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from skills.common_obs import projected_gravity


# ─── primary signals ──────────────────────────────────────────────────


def upright_signal(quat: np.ndarray) -> np.ndarray:
    """1 = fully upright, 0 = sideways, -1 = inverted."""
    g = projected_gravity(quat)
    return (-g[:, 2]).astype(np.float32)


def height_signal(root_z: np.ndarray, target: float = 0.55,
                  sigma: float = 0.15) -> np.ndarray:
    err = root_z - target
    return np.exp(-(err ** 2) / (sigma ** 2)).astype(np.float32)


def gravity_horizontal_penalty(quat: np.ndarray) -> np.ndarray:
    """gx² + gy² — symmetric tilt-from-vertical measure independent of
    yaw. Zero at upright, grows quadratically with tilt. Cleaner than
    the upright signal near small tilts (no asymmetry from the clamp)."""
    g = projected_gravity(quat)
    return (g[:, 0] ** 2 + g[:, 1] ** 2).astype(np.float32)


# ─── stability penalties ──────────────────────────────────────────────


def base_ang_vel_sway(ang_vel: np.ndarray) -> np.ndarray:
    """ωx² + ωy² — roll/pitch rate. Direct penalty on front/back +
    left/right sway. Yaw rate is intentionally ignored — robot may need
    to spin during the recovery."""
    return (ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2).astype(np.float32)


def base_lin_vel_drift(lin_vel: np.ndarray) -> np.ndarray:
    """||v||² of trunk linear vel. Phase-gated by the caller."""
    return np.sum(lin_vel ** 2, axis=1).astype(np.float32)


def joint_vel_quiet(joint_vel: np.ndarray) -> np.ndarray:
    """Σ q̇² — total joint kinetic activity. Phase-gated by the caller
    so it only fires near upright (we don't want to punish the recovery
    motion itself)."""
    return np.sum(joint_vel ** 2, axis=1).astype(np.float32)


def action_smoothness(action: np.ndarray,
                      prev_action: np.ndarray) -> np.ndarray:
    """(a_t - a_{t-1})² — first derivative."""
    return np.sum((action - prev_action) ** 2, axis=1).astype(np.float32)


def action_jerk(action: np.ndarray, prev_action: np.ndarray,
                prev_prev_action: np.ndarray) -> np.ndarray:
    """(a_t - 2 a_{t-1} + a_{t-2})² — second derivative. Penalises
    'stutter' in the command stream specifically. Smoothness only
    catches step-to-step magnitude; jerk catches direction-flipping."""
    second_diff = action - 2.0 * prev_action + prev_prev_action
    return np.sum(second_diff ** 2, axis=1).astype(np.float32)


# ─── phase gating ─────────────────────────────────────────────────────


def near_upright_gate(upright: np.ndarray,
                      lo: float = 0.5, hi: float = 0.8) -> np.ndarray:
    """Linear ramp in [lo, hi] of the upright signal, clipped to [0, 1].
    Returns the soft 'how near upright are we' mask used to phase-gate
    drift / quiet-joint penalties — smooth, no discontinuity."""
    return np.clip((upright - lo) / max(hi - lo, 1e-6), 0.0, 1.0
                   ).astype(np.float32)


# ─── success detector ────────────────────────────────────────────────


def success_frame_mask(quat: np.ndarray, root_z: np.ndarray,
                       target_h: float = 0.55,
                       upright_threshold: float = 0.92) -> np.ndarray:
    """Per-frame 'looks standing' boolean. The env composes this with
    a streak counter to require sustained success."""
    return ((upright_signal(quat) > upright_threshold)
            & (root_z > target_h - 0.10))


# ─── composite ────────────────────────────────────────────────────────


def compute_standup_reward(
    *,
    root_pos: np.ndarray,            # (N, 3)
    root_quat: np.ndarray,           # (N, 4) wxyz
    root_lin_vel: np.ndarray,        # (N, 3) world frame
    root_ang_vel: np.ndarray,        # (N, 3) body frame
    joint_vel: np.ndarray,           # (N, n_dof)
    action: np.ndarray,              # (N, n_dof)
    prev_action: np.ndarray,         # (N, n_dof)
    prev_prev_action: np.ndarray,    # (N, n_dof)
    success_streak: np.ndarray,      # (N,) int — consecutive success-frames
    sustained_now: np.ndarray,       # (N,) bool — streak reached threshold this step
    step_count: np.ndarray,          # (N,) int — control steps since reset
    weights,
    target_height: float = 0.55,
    upright_threshold: float = 0.92,
    hold_steps: int = 30,
    time_to_stand_tau_steps: float = 100.0,
    control_dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Returns (reward[N], frame_success[N] bool, components_dict).

    `frame_success` is the per-frame "looks standing" mask — the env
    uses it to advance its streak counter. The env owns termination
    (sustained_now arrives back as input here so we can pay the bonus
    on the exact frame the streak completes)."""

    up = upright_signal(root_quat)
    h = height_signal(root_pos[:, 2], target=target_height)

    # Positive shaping (bounded [0, 1])
    up_pos = np.clip(up, 0.0, 1.0)
    grav_h_pen = gravity_horizontal_penalty(root_quat)

    # Phase gate — most drift/quietness penalties only meaningful near upright.
    gate = near_upright_gate(up)

    ang_sway = base_ang_vel_sway(root_ang_vel)
    lin_drift = base_lin_vel_drift(root_lin_vel) * gate
    q_quiet = joint_vel_quiet(joint_vel) * gate

    smooth = action_smoothness(action, prev_action)
    jerk = action_jerk(action, prev_action, prev_prev_action)

    # Per-frame "looks standing" — env will count consecutive frames.
    frame_success = success_frame_mask(
        root_quat, root_pos[:, 2],
        target_h=target_height,
        upright_threshold=upright_threshold,
    )

    # Hold-window persistence: pay while currently in the streak but not
    # yet completed (streak ∈ [1, hold_steps - 1]). Doesn't double-pay
    # the terminal bonus on the completion frame.
    in_hold = (success_streak >= 1) & (success_streak < hold_steps)
    persistence = in_hold.astype(np.float32)

    # Speed signal — dense penalty until the standup is complete.
    not_yet_done = (success_streak < hold_steps).astype(np.float32)
    time_pen = not_yet_done  # constant 1 per step until done

    # Terminal time-scaled bonus, paid once on the streak-completion frame.
    # `step_count` at completion ≈ time_to_first_success + hold_steps.
    # We back out the first-success time so the bonus reflects raw speed.
    t_first = np.maximum(step_count - (hold_steps - 1), 0)
    time_bonus_scale = np.exp(
        -t_first.astype(np.float32) / max(time_to_stand_tau_steps, 1e-6))
    sustained_bonus = sustained_now.astype(np.float32) * time_bonus_scale

    w = weights
    r = (
        w.upright * up_pos
        + w.height * h
        - w.gravity_horizontal * grav_h_pen
        - w.base_ang_vel_sway * ang_sway
        - w.base_lin_vel_drift * lin_drift
        - w.joint_vel_quiet * q_quiet
        - w.action_smoothness * smooth
        - w.action_jerk * jerk
        + w.success_persistence * persistence
        - w.time_penalty * time_pen
        + w.success_bonus * sustained_bonus
    ).astype(np.float32)

    components = {
        "upright": float(np.mean(up_pos)),
        "upright_raw": float(np.mean(up)),
        "height": float(np.mean(h)),
        "grav_horizontal": float(np.mean(grav_h_pen)),
        "ang_vel_sway": float(np.mean(ang_sway)),
        "lin_vel_drift": float(np.mean(lin_drift)),
        "joint_vel_quiet": float(np.mean(q_quiet)),
        "action_smooth": float(np.mean(smooth)),
        "action_jerk": float(np.mean(jerk)),
        "near_upright_gate": float(np.mean(gate)),
        "hold_streak_mean": float(np.mean(success_streak)),
        "frame_success_rate": float(np.mean(frame_success)),
        "sustained_rate": float(np.mean(sustained_now)),
        "time_bonus_mean": float(np.mean(time_bonus_scale * sustained_now)),
        "mean_robot_z": float(np.mean(root_pos[:, 2])),
        "mean_reward": float(np.mean(r)),
    }
    return r, frame_success, components
