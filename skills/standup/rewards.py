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
from skills.common_rewards import joint_pose_deviation


# ─── critic groups (multi-critic PPO) ─────────────────────────────────
#
# HoST (arXiv:2502.08378) found a SINGLE critic over the full
# heterogeneous reward achieves ~zero success — the value net can't fit
# returns that mix a +400 terminal pulse, dense [0,1] shaping, and small
# quadratic penalties. The fix is one critic PER reward GROUP, each fed a
# homogeneous-scale return, with normalized-advantage aggregation. These
# names index the per-env group-reward array `compute_standup_reward`
# returns (column order = this tuple) and the per-group critic heads.
#
#   task    — "get upright + tall + on your feet" dense shaping
#   reg     — motion-quality penalties (zeroed in the discovery stage)
#   success — sparse/terminal: hold-window, speed, success bonus, stay-up
STANDUP_CRITIC_GROUPS = ("task", "reg", "success")


# ─── primary signals ──────────────────────────────────────────────────


def upright_signal(quat: np.ndarray) -> np.ndarray:
    """1 = fully upright, 0 = sideways, -1 = inverted."""
    g = projected_gravity(quat)
    return (-g[:, 2]).astype(np.float32)


def height_signal(root_z: np.ndarray, target: float = 0.55,
                  sigma: float = 0.3) -> np.ndarray:
    """Gaussian on trunk height. σ=0.3 keeps the gradient meaningful
    even from a fallen pose (z≈0.15 still gives ~0.17 of signal); the
    previous σ=0.15 was numerically flat for any z below ~0.4 — the
    policy got no height gradient until it was already half-up."""
    err = root_z - target
    return np.exp(-(err ** 2) / (sigma ** 2)).astype(np.float32)


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


# ─── anti-gaming "stand on feet" signal ───────────────────────────────


def _feet_grounded_score(foot_z: np.ndarray,
                          foot_max_z: float) -> np.ndarray:
    """Multiplicative AND of per-foot ground proximity in [0, 1]."""
    foot_l = np.clip(1.0 - foot_z[:, 0] / max(foot_max_z, 1e-6),
                     0.0, 1.0)
    foot_r = np.clip(1.0 - foot_z[:, 1] / max(foot_max_z, 1e-6),
                     0.0, 1.0)
    return foot_l * foot_r


def _upright_factor(upright: np.ndarray) -> np.ndarray:
    """Orientation gate in [0, 1] for the stand-on-feet rewards.

    Without this gate, both `foot_grounded_up` and `standing_tall` would
    pay full reward to a SIDE-PLANK pose — feet on the floor, trunk
    elevated by a propping arm, but trunk horizontal rather than
    vertical. The terms ONLY check scalar trunk z; they have no notion
    of orientation. Multiplying by max(0, cos(tilt)) makes "stand on
    feet" rewards proportional to how vertical the trunk is, so a
    side-plank (up≈0.5) earns half credit and a full stand (up=1.0)
    earns full credit.
    """
    return np.clip(upright, 0.0, 1.0).astype(np.float32)


def foot_grounded_up_signal(foot_z: np.ndarray, trunk_z: np.ndarray,
                             upright: np.ndarray,
                             foot_max_z: float = 0.10,
                             trunk_min_z: float = 0.30) -> np.ndarray:
    """Smooth multiplicative gate in [0, 1] that pays only for the joint
    condition "both feet on the floor AND trunk lifted AND trunk
    vertical". Forces the policy to stand on its feet — bridge /
    shoulder-stand / sprawled / side-plank poses (which can game
    upright + height by lifting the trunk off the ground without
    putting feet down OR by lying horizontally while propped up) all
    evaluate to ~0 here.

    Smooth ramps instead of hard step functions so PPO gets a gradient
    toward the threshold even before satisfying it. Saturates at
    trunk_z ≥ trunk_min_z — see `standing_tall_signal` for the term
    that continues past saturation.
    """
    feet = _feet_grounded_score(foot_z, foot_max_z)
    # Trunk: 0 at z=0.10 (still essentially flat), 1 at z=trunk_min_z.
    trunk = np.clip((trunk_z - 0.10) / max(trunk_min_z - 0.10, 1e-6),
                    0.0, 1.0)
    return (feet * trunk * _upright_factor(upright)).astype(np.float32)


def standing_tall_signal(foot_z: np.ndarray, trunk_z: np.ndarray,
                          upright: np.ndarray,
                          foot_max_z: float = 0.10,
                          trunk_min_z: float = 0.30,
                          trunk_max_z: float = 0.55) -> np.ndarray:
    """Picks up where `foot_grounded_up` saturates: same feet-grounded
    gate × trunk ramp in [trunk_min_z, trunk_max_z] × upright_factor. 0
    at squat (trunk_z = trunk_min_z), 1.0 at full standing height.
    Stacks additively on top of foot_grounded_up so the squat reward
    isn't altered, but full extension pays significantly more — pulls
    the policy out of the squat local optimum without destabilising it.
    Like `foot_grounded_up`, gated by trunk orientation so a side-plank
    pose can't game it."""
    feet = _feet_grounded_score(foot_z, foot_max_z)
    trunk = np.clip((trunk_z - trunk_min_z)
                    / max(trunk_max_z - trunk_min_z, 1e-6),
                    0.0, 1.0)
    return (feet * trunk * _upright_factor(upright)).astype(np.float32)


# ─── phase gating ─────────────────────────────────────────────────────


def near_upright_gate(upright: np.ndarray,
                      lo: float = 0.7, hi: float = 0.95) -> np.ndarray:
    """Linear ramp in [lo, hi] of the upright signal, clipped to [0, 1].
    Returns the soft 'how near upright are we' mask used to phase-gate
    motion penalties — smooth, no discontinuity. Defaults [0.7, 0.95]
    deliberately push the activation zone into the final balancing range
    (cos(tilt) ∈ [0.7, 0.95] = tilt 18°–45°): the policy gets a fully
    motion-free recovery, and stability shaping only kicks in once it's
    essentially balancing in place."""
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
    joint_pos: np.ndarray,           # (N, n_dof)
    joint_vel: np.ndarray,           # (N, n_dof)
    action: np.ndarray,              # (N, n_dof)
    prev_action: np.ndarray,         # (N, n_dof)
    prev_prev_action: np.ndarray,    # (N, n_dof)
    prev_upright: np.ndarray,        # (N,) upright_signal from last step
    success_streak: np.ndarray,      # (N,) int — consecutive success-frames
    sustained_now: np.ndarray,       # (N,) bool — streak reached threshold this step
    achieved_sustained: np.ndarray,  # (N,) bool — streak has reached threshold at some point this episode
    step_count: np.ndarray,          # (N,) int — control steps since reset
    foot_z: np.ndarray,              # (N, 2) world-frame z of left/right foot
    weights,
    arm_joint_indices: tuple = (),   # arm dofs (K1: 2..9)
    default_joint_pos: np.ndarray = None,
    target_height: float = 0.55,
    upright_threshold: float = 0.92,
    hold_steps: int = 30,
    time_to_stand_tau_steps: float = 100.0,
    foot_grounded_max_z: float = 0.10,
    trunk_up_min_z: float = 0.30,
    standing_tall_min_z: float = 0.30,
    standing_tall_max_z: float = 0.55,
    control_dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """Returns (reward[N], frame_success[N] bool, components, group_rewards).

    `frame_success` is the per-frame "looks standing" mask — the env
    uses it to advance its streak counter. The env owns termination
    (sustained_now arrives back as input here so we can pay the bonus
    on the exact frame the streak completes).

    `group_rewards` maps each name in `STANDUP_CRITIC_GROUPS` to its
    per-env (N,) reward contribution; the three sum EXACTLY to `reward`.
    Single-critic training ignores it; multi-critic training feeds one
    critic per group (see `STANDUP_CRITIC_GROUPS`)."""

    up = upright_signal(root_quat)
    h = height_signal(root_pos[:, 2], target=target_height)

    # Smooth orientation reward in [0, 1] EVERYWHERE — un-clipped so the
    # policy gets a monotonic gradient from upside-down → sideways →
    # upright. The previous `clip(up, 0, 1)` made the reward flat for
    # any pose worse than horizontal and produced a "stay still" local
    # optimum the policy couldn't escape.
    up_pos = ((up + 1.0) * 0.5).astype(np.float32)

    # Phase gate — ALL motion penalties are gated on near-upright. During
    # deep recovery the policy MUST be free to make big motions to stand
    # up; once near upright, jitter/sway becomes meaningful and the gate
    # opens. (gravity_horizontal was dropped — it rewarded upside-down
    # over sideways since g_x² + g_y² is 0 at both upright AND inverted.)
    gate = near_upright_gate(up)

    ang_sway = base_ang_vel_sway(root_ang_vel) * gate
    lin_drift = base_lin_vel_drift(root_lin_vel) * gate
    q_quiet = joint_vel_quiet(joint_vel) * gate

    smooth = action_smoothness(action, prev_action) * gate
    jerk = action_jerk(action, prev_action, prev_prev_action) * gate

    # Progress shaping — pay the policy for *active* uprightening, not
    # just for being-in-state. Without this, side-plank (up≈0.7) is a
    # stable basin worth +1.9/step; the marginal gradient toward standing
    # is too weak for PPO to commit to the risky transition. The positive-
    # only clip avoids paying for backsliding being undone — credit is
    # only awarded for genuine forward progress.
    progress = np.maximum(0.0, up - prev_upright).astype(np.float32)

    # Arm-pose deviation — drives the final standing pose to "arms
    # hanging at the sides" (the corrected K1 default with shoulder
    # rolls at ±π/2). Phase-gated on [0.5, 0.85] so arms stay free
    # through the entire recovery (up<0.5) and the penalty only ramps
    # in as the robot approaches its final pose. Many standup motions
    # need arm push-off through up≈0.3–0.5; the gate must stay open
    # there.
    arm_gate = near_upright_gate(up, lo=0.5, hi=0.85)
    arm_dev = np.zeros(root_pos.shape[0], dtype=np.float32)
    if len(arm_joint_indices) > 0 and default_joint_pos is not None:
        arm_dev = joint_pose_deviation(joint_pos, arm_joint_indices,
                                        default_joint_pos) * arm_gate

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

    # Post-success standing reward — only credited AFTER the robot has
    # achieved a sustained success in this episode AND is still upright
    # this frame. The episode runs to MAX_EPISODE_STEPS so this dominates
    # the return for a fast-and-stable standup: a 1.5 s standup on a 5 s
    # episode earns ~175 frames × w.post_success_standing of standing
    # reward, which is much larger than the terminal speed bonus. A
    # fast-but-unstable standup that collapses immediately after success
    # forfeits all of it — exactly the "fast AND stable" pressure we want.
    post_success = (achieved_sustained & frame_success).astype(np.float32)

    # Anti-gaming "stand on feet" reward — only pays for the joint
    # condition (both feet grounded AND trunk lifted). Closes the bridge
    # / shoulder-stand / sprawled-on-back local optima where the policy
    # gets partial upright + height credit without putting its feet on
    # the floor. Smooth multiplicative gate provides gradient toward the
    # threshold even before satisfying it.
    foot_up = foot_grounded_up_signal(
        foot_z, root_pos[:, 2], up,
        foot_max_z=foot_grounded_max_z,
        trunk_min_z=trunk_up_min_z,
    )
    tall = standing_tall_signal(
        foot_z, root_pos[:, 2], up,
        foot_max_z=foot_grounded_max_z,
        trunk_min_z=standing_tall_min_z,
        trunk_max_z=standing_tall_max_z,
    )

    w = weights

    # ── per-group decomposition (multi-critic PPO) ──
    # Each group is a homogeneous-scale return so a per-group critic can
    # actually fit it. The three sum EXACTLY to the single-critic reward
    # `r` below, so single-critic training is unchanged.
    r_task = (
        w.upright * up_pos
        + w.height * h
        + w.upright_progress * progress
        + w.foot_grounded_up * foot_up
        + w.standing_tall * tall
    ).astype(np.float32)
    r_reg = (
        - w.arm_pose_dev * arm_dev
        - w.base_ang_vel_sway * ang_sway
        - w.base_lin_vel_drift * lin_drift
        - w.joint_vel_quiet * q_quiet
        - w.action_smoothness * smooth
        - w.action_jerk * jerk
    ).astype(np.float32)
    r_success = (
        w.success_persistence * persistence
        - w.time_penalty * time_pen
        + w.success_bonus * sustained_bonus
        + w.post_success_standing * post_success
    ).astype(np.float32)

    r = (r_task + r_reg + r_success).astype(np.float32)
    group_rewards = {"task": r_task, "reg": r_reg, "success": r_success}

    components = {
        "upright": float(np.mean(up_pos)),
        "upright_raw": float(np.mean(up)),
        "upright_progress": float(np.mean(progress)),
        "arm_pose_dev": float(np.mean(arm_dev)),
        "arm_gate": float(np.mean(arm_gate)),
        "height": float(np.mean(h)),
        "ang_vel_sway": float(np.mean(ang_sway)),
        "lin_vel_drift": float(np.mean(lin_drift)),
        "joint_vel_quiet": float(np.mean(q_quiet)),
        "action_smooth": float(np.mean(smooth)),
        "action_jerk": float(np.mean(jerk)),
        "near_upright_gate": float(np.mean(gate)),
        "hold_streak_mean": float(np.mean(success_streak)),
        "frame_success_rate": float(np.mean(frame_success)),
        "sustained_rate": float(np.mean(sustained_now)),
        "achieved_sustained_rate": float(np.mean(achieved_sustained)),
        "post_success_standing": float(np.mean(post_success)),
        "foot_grounded_up": float(np.mean(foot_up)),
        "standing_tall": float(np.mean(tall)),
        "mean_foot_z": float(np.mean(foot_z)),
        "time_bonus_mean": float(np.mean(time_bonus_scale * sustained_now)),
        "mean_robot_z": float(np.mean(root_pos[:, 2])),
        "mean_reward": float(np.mean(r)),
        "group_task": float(np.mean(r_task)),
        "group_reg": float(np.mean(r_reg)),
        "group_success": float(np.mean(r_success)),
    }
    return r, frame_success, components, group_rewards
