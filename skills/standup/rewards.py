"""Standup reward — HoST-faithful, plain-RL design (arXiv:2502.08378).

Rewritten from the dense-shaping version to mirror HoST's structure:

  * task   — a MINIMAL, *bounded* objective: orientation + rise-height, each via
             a saturating `tolerance()` kernel (NOT a dense linear up/height ramp,
             and NO progress ratchet). This is what stops the "throw the trunk
             up" exploit at the source.
  * regu   — a *whisper* of action-rate / jerk / joint-velocity penalties
             (HoST's regu group weight is only 0.1).
  * style  — height-gated motion-shape terms (feet under base, feet flat / ground
             parallel, feet not splayed, low roll/pitch rate) that switch on once
             the trunk is off the ground.
  * post   — height-gated "now hold a clean, quiet stand" terms (flat orientation,
             target base height, stillness, arms to the stand pose) plus the
             sparse success bonus / hold persistence.

Groups sum EXACTLY to the scalar reward, so single-critic training is unchanged;
multi-critic training (optional) feeds one critic per group (see
`STANDUP_CRITIC_GROUPS`).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from skills.common_obs import projected_gravity
from skills.common_rewards import joint_pose_deviation

# HoST groups: task / regu(larization) / style / post(-task). Group WEIGHTS for
# the optional multi-critic path live in StandupConfig.critic_group_weights
# (HoST uses [2.5, 0.1, 1, 1]). Single-critic just sums the four.
STANDUP_CRITIC_GROUPS = ("task", "regu", "style", "post")


# ─── kernels ──────────────────────────────────────────────────────────────


def tolerance(
    x: np.ndarray,
    lo: float,
    hi: float = np.inf,
    margin: float = 1.0,
    value_at_margin: float = 0.1,
) -> np.ndarray:
    """dm_control-style saturating reward: 1.0 inside [lo, hi], with a Gaussian
    falloff outside that reaches `value_at_margin` at distance `margin` from the
    nearest bound. Bounded in (0, 1] — the policy cannot farm it by overshooting.
    """
    x = np.asarray(x, dtype=np.float32)
    in_bounds = (x >= lo) & (x <= hi)
    below = np.maximum(lo - x, 0.0)
    above = np.maximum(x - hi, 0.0) if np.isfinite(hi) else np.zeros_like(x)
    d = below + above
    vam = float(np.clip(value_at_margin, 1e-6, 1 - 1e-6))
    scale = float(margin) / np.sqrt(-2.0 * np.log(vam))
    val = np.exp(-0.5 * (d / max(scale, 1e-9)) ** 2)
    return np.where(in_bounds, 1.0, val).astype(np.float32)


def upright_signal(quat: np.ndarray) -> np.ndarray:
    """+1 when the trunk Z-axis aligns with world up, -1 when inverted."""
    g = projected_gravity(quat)
    return (-g[:, 2]).astype(np.float32)


# ─── geometry helpers (kept: imported by env.py) ────────────────────────────


def feet_under_base_score(
    foot_xy: np.ndarray,
    base_xy: np.ndarray,
    d_max: float = 0.40,
    plateau_d: float = 0.0,
) -> np.ndarray:
    """1 when both feet sit under the base (xy), ramping to 0 at d_max."""
    d = np.linalg.norm(foot_xy - base_xy[:, None, :], axis=2)  # (N, 2)
    excess = np.clip(d - max(plateau_d, 0.0), 0.0, None)
    span = max(d_max - max(plateau_d, 0.0), 1e-6)
    score = np.clip(1.0 - excess / span, 0.0, 1.0)
    return (score[:, 0] * score[:, 1]).astype(np.float32)


def standing_on_feet_mask(
    foot_z: np.ndarray,
    foot_xy: np.ndarray,
    base_xy: np.ndarray,
    foot_max_z: float = 0.12,
    under_base_max_d: float = 0.25,
) -> np.ndarray:
    """Hard gate: both feet grounded AND under the base — the success precondition."""
    grounded = (foot_z[:, 0] < foot_max_z) & (foot_z[:, 1] < foot_max_z)
    d = np.linalg.norm(foot_xy - base_xy[:, None, :], axis=2)  # (N, 2)
    under = (d[:, 0] < under_base_max_d) & (d[:, 1] < under_base_max_d)
    return grounded & under


def success_frame_mask(
    quat: np.ndarray,
    root_z: np.ndarray,
    target_h: float = 0.55,
    upright_threshold: float = 0.92,
    feet_ok: np.ndarray = None,
) -> np.ndarray:
    """Per-frame 'looks standing': upright + near target height + feet ok."""
    mask = (upright_signal(quat) > upright_threshold) & (root_z > target_h - 0.10)
    if feet_ok is not None:
        mask = mask & feet_ok
    return mask


# ─── reward ─────────────────────────────────────────────────────────────────


def compute_standup_reward(
    *,
    root_pos: np.ndarray,        # (N, 3)
    root_quat: np.ndarray,       # (N, 4) wxyz
    root_lin_vel: np.ndarray,    # (N, 3) world frame
    root_ang_vel: np.ndarray,    # (N, 3) body frame
    joint_pos: np.ndarray,       # (N, n_dof)
    joint_vel: np.ndarray,       # (N, n_dof)
    action: np.ndarray,          # (N, n_dof) raw policy delta this step
    prev_action: np.ndarray,     # (N, n_dof)
    prev_prev_action: np.ndarray,  # (N, n_dof)
    foot_z: np.ndarray,          # (N, 2) world-frame z of left/right foot
    foot_xy: np.ndarray,         # (N, 2, 2) world-frame xy of left/right foot
    feet_ok: np.ndarray,         # (N,) bool — standing_on_feet_mask
    success_streak: np.ndarray,  # (N,) int — consecutive success frames
    sustained_now: np.ndarray,   # (N,) bool — streak hit threshold this step
    achieved_sustained: np.ndarray,  # (N,) bool — streak hit threshold this episode
    step_count: np.ndarray,      # (N,) int — control steps since reset
    arm_joint_indices: tuple,
    default_joint_pos: np.ndarray,
    stand_target_pose: np.ndarray = None,  # default w/ optional hip abduction
    weights=None,
    # ── HoST kernel/stage params (K1-adapted; see StandupConfig) ──
    rise_target: float = 0.45,       # head/trunk rise above feet for "standing"
    rise_margin: float = 0.45,
    orientation_threshold: float = 0.95,
    orientation_margin: float = 1.0,
    style_stage_rise: float = 0.30,  # trunk-off-ground gate for style terms
    post_stage_rise: float = 0.45,   # near-standing gate for post terms
    target_height: float = 0.55,
    upright_threshold: float = 0.92,
    feet_under_base_d: float = 0.40,
    feet_distance_max: float = 0.45,
    hold_steps: int = 30,
    time_to_stand_tau_steps: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """Returns (reward[N], frame_success[N] bool, components, group_rewards)."""
    w = weights
    N = root_pos.shape[0]
    z = root_pos[:, 2]
    mean_foot_z = foot_z.mean(axis=1)
    rise = (z - mean_foot_z).astype(np.float32)  # head/trunk height above feet
    up = upright_signal(root_quat)
    g = projected_gravity(root_quat)

    # ── task (bounded, saturating) ───────────────────────────────────────
    # Both task terms use BROAD gradients that give signal across the WHOLE
    # fallen→standing range. Narrow tolerance() kernels are flat in the tail
    # (≈no gradient far from the goal): orientation's flat tail froze runs 2/3,
    # and rise's flat tail froze run 4 at an upright SIT (z≈0.13, rise<0) —
    # verticalized but with nothing pulling it to actually rise.
    #
    # orient: monotonic inverted(0)→upright(1).  rise_signal: clipped-linear
    # ramp 0→1 over [≈floor, rise_target] so the gradient is constant (never
    # vanishes) until it saturates at the target — bounded, can't be overshot.
    orient = (((up + 1.0) * 0.5) ** 2).astype(np.float32)
    rise_signal = np.clip(
        (rise + 0.15) / (rise_target + 0.15), 0.0, 1.0
    ).astype(np.float32)
    # Anti-farm coupling (deadlock-free now that BOTH signals are broad):
    #  • orientation gated on rise (0.2 floor) → can't sit upright on the floor
    #    for full orientation credit (run 1 / run 4).
    #  • rise gated on orientation (× orient)  → can't loft the trunk upside-down
    #    for height credit (run 2).
    # Full credit needs BOTH = standing.
    orient_rise_gate = (
        0.2 + 0.8 * np.clip(rise / max(style_stage_rise, 1e-6), 0.0, 1.0)
    ).astype(np.float32)
    task_orient = (orient * orient_rise_gate).astype(np.float32)
    task_rise = (rise_signal * orient).astype(np.float32)

    # ── regularization (a whisper) ───────────────────────────────────────
    action_rate = np.sum((action - prev_action) ** 2, axis=1).astype(np.float32)
    action_jerk = np.sum(
        (action - 2.0 * prev_action + prev_prev_action) ** 2, axis=1
    ).astype(np.float32)
    dof_vel = np.sum(joint_vel ** 2, axis=1).astype(np.float32)

    # ── style (gated on trunk off the ground) ────────────────────────────
    gate_style = (rise > style_stage_rise).astype(np.float32)
    fub = feet_under_base_score(foot_xy, root_pos[:, :2], d_max=feet_under_base_d)
    foot_z_var = foot_z.var(axis=1).astype(np.float32)
    ground_parallel = np.exp(-foot_z_var * 200.0).astype(np.float32)  # feet at same height
    feet_sep = np.linalg.norm(foot_xy[:, 0, :] - foot_xy[:, 1, :], axis=1)
    feet_distance_pen = np.clip(feet_sep - feet_distance_max, 0.0, None).astype(np.float32)
    ang_xy = root_ang_vel[:, 0] ** 2 + root_ang_vel[:, 1] ** 2
    style_low_angvel = np.exp(-ang_xy * 2.0).astype(np.float32)

    # ── post-task (gated near standing) + success ────────────────────────
    gate_post = (rise > post_stage_rise).astype(np.float32)
    post_orient = np.exp(-(g[:, 0] ** 2 + g[:, 1] ** 2) * 5.0).astype(np.float32)
    post_base_h = np.exp(-np.abs(z - target_height) * 20.0).astype(np.float32)
    lin_xy = root_lin_vel[:, 0] ** 2 + root_lin_vel[:, 1] ** 2
    post_still = np.exp(-(lin_xy * 5.0 + ang_xy * 2.0)).astype(np.float32)
    _target = stand_target_pose if stand_target_pose is not None else default_joint_pos
    if len(arm_joint_indices) > 0 and _target is not None:
        arm_dev = joint_pose_deviation(joint_pos, arm_joint_indices, _target)
        post_upper = np.exp(-arm_dev * 0.5).astype(np.float32)
    else:
        post_upper = np.zeros(N, dtype=np.float32)

    # Sparse success / hold (curriculum-independent: env owns the streak).
    frame_success = success_frame_mask(
        root_quat, z, target_h=target_height,
        upright_threshold=upright_threshold, feet_ok=feet_ok,
    )
    in_hold = (success_streak >= 1) & (success_streak < hold_steps)
    persistence = in_hold.astype(np.float32)
    not_yet_done = (success_streak < hold_steps).astype(np.float32)
    time_pen = not_yet_done
    t_first = np.maximum(step_count - (hold_steps - 1), 0)
    time_bonus_scale = np.exp(
        -t_first.astype(np.float32) / max(time_to_stand_tau_steps, 1e-6)
    )
    sustained_bonus = sustained_now.astype(np.float32) * time_bonus_scale
    post_success = (achieved_sustained & frame_success).astype(np.float32)

    # ── group assembly ───────────────────────────────────────────────────
    r_task = (w.task_orientation * task_orient + w.task_rise * task_rise).astype(np.float32)
    r_regu = -(
        w.regu_action_rate * action_rate
        + w.regu_action_jerk * action_jerk
        + w.regu_dof_vel * dof_vel
    ).astype(np.float32)
    r_style = (
        gate_style
        * (
            w.style_feet_under_base * fub
            + w.style_ground_parallel * ground_parallel
            - w.style_feet_distance * feet_distance_pen
            + w.style_ang_vel * style_low_angvel
        )
    ).astype(np.float32)
    r_post = (
        gate_post
        * (
            w.post_orientation * post_orient
            + w.post_base_height * post_base_h
            + w.post_stillness * post_still
            + w.post_upper_pose * post_upper
        )
        + w.success_persistence * persistence
        - w.time_penalty * time_pen
        + w.success_bonus * sustained_bonus
        + w.post_success_standing * post_success
    ).astype(np.float32)

    r = (r_task + r_regu + r_style + r_post).astype(np.float32)
    group_rewards = {"task": r_task, "regu": r_regu, "style": r_style, "post": r_post}

    components = {
        "task_orientation": float(np.mean(task_orient)),
        "task_rise": float(np.mean(task_rise)),
        "upright_raw": float(np.mean(up)),
        "mean_rise": float(np.mean(rise)),
        "action_rate": float(np.mean(action_rate)),
        "action_jerk": float(np.mean(action_jerk)),
        "dof_vel": float(np.mean(dof_vel)),
        "style_feet_under_base": float(np.mean(fub)),
        "style_ground_parallel": float(np.mean(ground_parallel)),
        "style_feet_distance_pen": float(np.mean(feet_distance_pen)),
        "style_low_angvel": float(np.mean(style_low_angvel)),
        "post_orientation": float(np.mean(post_orient)),
        "post_base_height": float(np.mean(post_base_h)),
        "post_stillness": float(np.mean(post_still)),
        "post_upper_pose": float(np.mean(post_upper)),
        "gate_style": float(np.mean(gate_style)),
        "gate_post": float(np.mean(gate_post)),
        "frame_success_rate": float(np.mean(frame_success)),
        "sustained_rate": float(np.mean(sustained_now)),
        "post_success_standing": float(np.mean(post_success)),
        "time_bonus_mean": float(np.mean(time_bonus_scale * sustained_now)),
        "mean_robot_z": float(np.mean(z)),
        "mean_foot_z": float(np.mean(foot_z)),
        "mean_reward": float(np.mean(r)),
        "group_task": float(np.mean(r_task)),
        "group_regu": float(np.mean(r_regu)),
        "group_style": float(np.mean(r_style)),
        "group_post": float(np.mean(r_post)),
    }
    return r, frame_success, components, group_rewards
