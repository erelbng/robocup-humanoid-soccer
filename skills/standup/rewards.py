from __future__ import annotations

from typing import Tuple

import numpy as np
from skills.common_obs import projected_gravity
from skills.common_rewards import joint_pose_deviation

STANDUP_CRITIC_GROUPS = ("task", "reg", "success")
#   task    — "get upright + tall + on your feet" dense shaping
#   reg     — motion-quality penalties (zeroed in the discovery stage)
#   success — sparse/terminal: hold-window, speed, success bonus, stay-up


def upright_signal(quat: np.ndarray) -> np.ndarray:
    g = projected_gravity(quat)
    return (-g[:, 2]).astype(np.float32)


def supine_situp_progress(
    root_z: np.ndarray,
    quat: np.ndarray,
    best_metric: np.ndarray,
    start_supine: np.ndarray,
) -> np.ndarray:

    g = projected_gravity(quat)

    backness = np.clip(g[:, 0], 0.0, 1.0)

    metric = 0.8 * (1.0 - backness) + 0.2 * np.clip(root_z / 0.35, 0.0, 1.0)

    progress = np.maximum(0.0, metric - best_metric)

    return progress * start_supine.astype(np.float32)


def height_signal(
    root_z: np.ndarray, target: float = 0.55, sigma: float = 0.3
) -> np.ndarray:
    err = root_z - target
    return np.exp(-(err**2) / (sigma**2)).astype(np.float32)


def explosive_rise_signal(
    root_lin_vel: np.ndarray,
    root_z: np.ndarray,
    upright: np.ndarray,
    target_h: float = 0.55,
    v_cap: float = 0.8,
) -> np.ndarray:
    rise = np.clip(root_lin_vel[:, 2], 0.0, v_cap) / max(v_cap, 1e-6)
    deficit = np.clip((target_h - root_z) / max(target_h, 1e-6), 0.0, 1.0)
    early = 1.0 - near_upright_gate(upright)
    return (rise * deficit * early).astype(np.float32)


def base_ang_vel_sway(ang_vel: np.ndarray) -> np.ndarray:
    return (ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2).astype(np.float32)


def base_lin_vel_drift(lin_vel: np.ndarray) -> np.ndarray:
    return np.sum(lin_vel**2, axis=1).astype(np.float32)


def joint_vel_quiet(joint_vel: np.ndarray) -> np.ndarray:
    return np.sum(joint_vel**2, axis=1).astype(np.float32)


def action_smoothness(action: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    return np.sum((action - prev_action) ** 2, axis=1).astype(np.float32)


def action_jerk(
    action: np.ndarray, prev_action: np.ndarray, prev_prev_action: np.ndarray
) -> np.ndarray:
    second_diff = action - 2.0 * prev_action + prev_prev_action
    return np.sum(second_diff**2, axis=1).astype(np.float32)


def _feet_grounded_score(foot_z: np.ndarray, foot_max_z: float) -> np.ndarray:
    foot_l = np.clip(1.0 - foot_z[:, 0] / max(foot_max_z, 1e-6), 0.0, 1.0)
    foot_r = np.clip(1.0 - foot_z[:, 1] / max(foot_max_z, 1e-6), 0.0, 1.0)
    return foot_l * foot_r


def feet_under_base_score(
    foot_xy: np.ndarray,
    base_xy: np.ndarray,
    d_max: float = 0.40,
    plateau_d: float = 0.0,
) -> np.ndarray:
    # foot_xy: (N, 2, 2) [foot, xy]; base_xy: (N, 2)
    d = np.linalg.norm(foot_xy - base_xy[:, None, :], axis=2)  # (N, 2)
    excess = np.clip(d - max(plateau_d, 0.0), 0.0, None)
    span = max(d_max - max(plateau_d, 0.0), 1e-6)
    score = np.clip(1.0 - excess / span, 0.0, 1.0)
    return (score[:, 0] * score[:, 1]).astype(np.float32)


def feet_tuck_signal(
    foot_z: np.ndarray,
    foot_xy: np.ndarray,
    base_xy: np.ndarray,
    foot_max_z: float = 0.12,
    d_max: float = 0.40,
) -> np.ndarray:
    grounded = _feet_grounded_score(foot_z, foot_max_z)
    under = feet_under_base_score(foot_xy, base_xy, d_max=d_max)
    return (grounded * under).astype(np.float32)


def standing_on_feet_mask(
    foot_z: np.ndarray,
    foot_xy: np.ndarray,
    base_xy: np.ndarray,
    foot_max_z: float = 0.12,
    under_base_max_d: float = 0.25,
) -> np.ndarray:
    grounded = (foot_z[:, 0] < foot_max_z) & (foot_z[:, 1] < foot_max_z)
    d = np.linalg.norm(foot_xy - base_xy[:, None, :], axis=2)  # (N, 2)
    under = (d[:, 0] < under_base_max_d) & (d[:, 1] < under_base_max_d)
    return grounded & under


def _upright_factor(upright: np.ndarray) -> np.ndarray:
    return np.clip(upright, 0.0, 1.0).astype(np.float32)


def foot_grounded_up_signal(
    foot_z: np.ndarray,
    trunk_z: np.ndarray,
    upright: np.ndarray,
    foot_max_z: float = 0.10,
    trunk_min_z: float = 0.30,
) -> np.ndarray:
    feet = _feet_grounded_score(foot_z, foot_max_z)
    # Trunk: 0 at z=0.10 (still essentially flat), 1 at z=trunk_min_z.
    trunk = np.clip((trunk_z - 0.10) / max(trunk_min_z - 0.10, 1e-6), 0.0, 1.0)
    return (feet * trunk * _upright_factor(upright)).astype(np.float32)


def standing_tall_signal(
    foot_z: np.ndarray,
    trunk_z: np.ndarray,
    upright: np.ndarray,
    foot_max_z: float = 0.10,
    trunk_min_z: float = 0.30,
    trunk_max_z: float = 0.55,
) -> np.ndarray:
    feet = _feet_grounded_score(foot_z, foot_max_z)
    trunk = np.clip(
        (trunk_z - trunk_min_z) / max(trunk_max_z - trunk_min_z, 1e-6), 0.0, 1.0
    )
    return (feet * trunk * _upright_factor(upright)).astype(np.float32)


def stand_pose_signal(
    joint_pos: np.ndarray,
    pose_joint_indices: tuple,
    target_pose: np.ndarray,
    upright: np.ndarray,
    dev_scale: float = 1.0,
) -> np.ndarray:
    if len(pose_joint_indices) == 0 or target_pose is None:
        return np.zeros(joint_pos.shape[0], dtype=np.float32)
    dev = joint_pose_deviation(joint_pos, pose_joint_indices, target_pose)
    pose = np.exp(-dev / max(dev_scale, 1e-6))
    return (pose * near_upright_gate(upright)).astype(np.float32)


def near_upright_gate(
    upright: np.ndarray, lo: float = 0.7, hi: float = 0.95
) -> np.ndarray:
    return np.clip((upright - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def success_frame_mask(
    quat: np.ndarray,
    root_z: np.ndarray,
    target_h: float = 0.55,
    upright_threshold: float = 0.92,
    feet_ok: np.ndarray = None,
) -> np.ndarray:
    mask = (upright_signal(quat) > upright_threshold) & (root_z > target_h - 0.10)
    if feet_ok is not None:
        mask = mask & feet_ok
    return mask


def compute_standup_reward(
    *,
    root_pos: np.ndarray,  # (N, 3)
    root_quat: np.ndarray,  # (N, 4) wxyz
    root_lin_vel: np.ndarray,  # (N, 3) world frame
    root_ang_vel: np.ndarray,  # (N, 3) body frame
    joint_pos: np.ndarray,  # (N, n_dof)
    joint_vel: np.ndarray,  # (N, n_dof)
    action: np.ndarray,  # (N, n_dof)
    prev_action: np.ndarray,  # (N, n_dof)
    prev_prev_action: np.ndarray,  # (N, n_dof)
    prev_upright: np.ndarray,  # (N,) upright_signal from last step
    prev_supine: np.ndarray,
    success_streak: np.ndarray,  # (N,) int — consecutive success-frames
    sustained_now: np.ndarray,  # (N,) bool — streak reached threshold this step
    achieved_sustained: np.ndarray,  # (N,) bool — streak has reached threshold at some point this episode
    step_count: np.ndarray,  # (N,) int — control steps since reset
    foot_z: np.ndarray,  # (N, 2) world-frame z of left/right foot
    foot_xy: np.ndarray = None,  # (N, 2, 2) world-frame xy of left/right foot
    feet_ok: np.ndarray = None,  # (N,) bool — standing_on_feet_mask (success gate)
    start_supine: np.ndarray = None,  # (N,) bool — episode started on the back
    weights=None,
    arm_joint_indices: tuple = (),  # arm dofs (K1: 2..9)
    pose_joint_indices: tuple = (),  # joints shaped to the target stand pose
    default_joint_pos: np.ndarray = None,
    stand_target_pose: np.ndarray = None,  # default w/ hips abducted (shoulder-wide)
    stand_pose_dev_scale: float = 1.0,
    success_ema: float = 0.0,  # running frame-success rate (0..1)
    stand_pose_success_ref: float = 0.5,
    start_xy: np.ndarray = None,  # (N, 2) base xy captured at reset
    on_spot_tol: float = 0.6,
    post_success_still_jv_scale: float = 3.0,
    post_success_still_v_scale: float = 0.2,
    feet_under_base_plateau_d: float = 0.0,
    max_upright: np.ndarray = None,  # (N,) episode high-water mark of upright
    progress_ratchet: bool = True,
    reg_success_ramp: bool = False,
    style_scale: float = None,
    trunk_contact_force: np.ndarray = None,  # (N,) net Trunk contact-force mag
    trunk_contact_force_thresh: float = 280.0,
    trunk_contact_force_scale: float = 196.0,
    knee_contact_force: np.ndarray = None,  # (N, 2) shank contact-force mags
    knee_contact_force_thresh: float = 20.0,
    knee_support_min_z: float = 0.20,
    knee_support_max_z: float = 0.45,
    target_height: float = 0.55,
    upright_threshold: float = 0.92,
    hold_steps: int = 30,
    time_to_stand_tau_steps: float = 100.0,
    foot_grounded_max_z: float = 0.10,
    trunk_up_min_z: float = 0.30,
    standing_tall_min_z: float = 0.30,
    standing_tall_max_z: float = 0.55,
    feet_under_base_soft_d: float = 0.40,
    explosive_rise_v_cap: float = 0.8,
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

    up_pos = ((up + 1.0) * 0.5).astype(np.float32)
    gate = near_upright_gate(up)

    ang_sway = base_ang_vel_sway(root_ang_vel) * gate
    lin_drift = base_lin_vel_drift(root_lin_vel) * gate
    q_quiet = joint_vel_quiet(joint_vel) * gate

    smooth = action_smoothness(action, prev_action) * gate
    jerk = action_jerk(action, prev_action, prev_prev_action) * gate

    # Progress shaping
    _prog_ref = (
        max_upright if (progress_ratchet and max_upright is not None) else prev_upright
    )
    progress = np.maximum(0.0, up - _prog_ref).astype(np.float32)

    situp_progress = supine_situp_progress(
        root_z=root_pos[:, 2],
        quat=root_quat,
        best_metric=prev_supine,
        start_supine=(
            start_supine
            if start_supine is not None
            else np.zeros(root_pos.shape[0], dtype=bool)
        ),
    ).astype(np.float32)

    # Explosive-rise shaping
    rise = explosive_rise_signal(
        root_lin_vel,
        root_pos[:, 2],
        up,
        target_h=target_height,
        v_cap=explosive_rise_v_cap,
    )

    # Arm-pose deviation
    arm_gate = near_upright_gate(up, lo=0.5, hi=0.85)
    arm_dev = np.zeros(root_pos.shape[0], dtype=np.float32)
    if len(arm_joint_indices) > 0 and default_joint_pos is not None:
        arm_dev = (
            joint_pose_deviation(joint_pos, arm_joint_indices, default_joint_pos)
            * arm_gate
        )

    # Target standing-pose reward
    _stand_target = (
        stand_target_pose if stand_target_pose is not None else default_joint_pos
    )
    pose = stand_pose_signal(
        joint_pos, pose_joint_indices, _stand_target, up, dev_scale=stand_pose_dev_scale
    )
    pose_scale = (
        float(style_scale)
        if style_scale is not None
        else float(np.clip(success_ema / max(stand_pose_success_ref, 1e-6), 0.0, 1.0))
    )
    # stand_pose is gated by near_upright_gate INSIDE stand_pose_signal (it only
    # fires when up ∈ [0.7, 0.95]), so it is inherently discovery-safe and does
    # NOT need the style_scale stage gate. Decoupling it from style_scale lets
    # the defined end-pose be shaped during every rise from the start of
    # training, instead of only after the pose curriculum reaches its final
    # level — which is why the policy used to land in a "funny" unshaped pose.
    # (pose_scale is still used for on_spot / trunk_force below.)
    stand_pose = pose.astype(np.float32)

    # Post-success STILLNESS
    still_score = np.exp(
        -(
            joint_vel_quiet(joint_vel) / max(post_success_still_jv_scale, 1e-6)
            + base_lin_vel_drift(root_lin_vel) / max(post_success_still_v_scale, 1e-6)
        )
    ).astype(np.float32)

    # Stand "on the spot"
    if start_xy is not None:
        disp = np.linalg.norm(root_pos[:, :2] - start_xy, axis=1)
        on_spot_excess = np.clip(disp - on_spot_tol, 0.0, None)
        on_spot_pen = (on_spot_excess**2 * pose_scale).astype(np.float32)
    else:
        disp = np.zeros(root_pos.shape[0], dtype=np.float32)
        on_spot_pen = np.zeros(root_pos.shape[0], dtype=np.float32)

    # Trunk contact-force penalty (anti-slam)
    if trunk_contact_force is not None:
        tf_excess = np.clip(
            (trunk_contact_force - trunk_contact_force_thresh)
            / max(trunk_contact_force_scale, 1e-6),
            0.0,
            None,
        )
        trunk_force_pen = (tf_excess**2 * pose_scale).astype(np.float32)
    else:
        trunk_force_pen = np.zeros(root_pos.shape[0], dtype=np.float32)

    # Knee/shin support credit (encourage usage of knees/shins to get up)
    if knee_contact_force is not None:
        knee_contact = (knee_contact_force > knee_contact_force_thresh).astype(
            np.float32
        )
        knee_grounded = knee_contact.mean(axis=1)  # 0, 0.5 or 1
        z = root_pos[:, 2]
        z_ramp = np.clip(
            (z - knee_support_min_z) / max(0.10, 1e-6), 0.0, 1.0
        )  # in above min
        z_fade = 1.0 - np.clip(
            (z - knee_support_max_z) / max(0.10, 1e-6), 0.0, 1.0
        )  # out above max
        knee_support = (knee_grounded * z_ramp * z_fade * _upright_factor(up)).astype(
            np.float32
        )
    else:
        knee_support = np.zeros(root_pos.shape[0], dtype=np.float32)

    # Per-frame "looks standing" — env will count consecutive frames.
    # Gated by feet_ok (feet grounded + under base) so the post-success
    # standing reward below can't be farmed from an assist-propped cobra.
    frame_success = success_frame_mask(
        root_quat,
        root_pos[:, 2],
        target_h=target_height,
        upright_threshold=upright_threshold,
        feet_ok=feet_ok,
    )

    # Hold-window persistence
    in_hold = (success_streak >= 1) & (success_streak < hold_steps)
    persistence = in_hold.astype(np.float32)

    # Speed signal — dense penalty until the standup is complete.
    not_yet_done = (success_streak < hold_steps).astype(np.float32)
    time_pen = not_yet_done  # constant 1 per step until done

    # Terminal time-scaled bonus
    t_first = np.maximum(step_count - (hold_steps - 1), 0)
    time_bonus_scale = np.exp(
        -t_first.astype(np.float32) / max(time_to_stand_tau_steps, 1e-6)
    )
    sustained_bonus = sustained_now.astype(np.float32) * time_bonus_scale

    # Post-success standing reward
    post_success = (achieved_sustained & frame_success).astype(np.float32)

    # Post-success stillness
    post_success_still = (post_success * still_score).astype(np.float32)

    # Anti-gaming "stand on feet" reward
    if foot_xy is not None:
        under_base = feet_under_base_score(
            foot_xy,
            root_pos[:, :2],
            d_max=feet_under_base_soft_d,
            plateau_d=feet_under_base_plateau_d,
        )
    else:
        under_base = np.ones(root_pos.shape[0], dtype=np.float32)

    foot_up = (
        foot_grounded_up_signal(
            foot_z,
            root_pos[:, 2],
            up,
            foot_max_z=foot_grounded_max_z,
            trunk_min_z=trunk_up_min_z,
        )
        * under_base
    )
    tall = (
        standing_tall_signal(
            foot_z,
            root_pos[:, 2],
            up,
            foot_max_z=foot_grounded_max_z,
            trunk_min_z=standing_tall_min_z,
            trunk_max_z=standing_tall_max_z,
        )
        * under_base
    )

    # Feet-tuck — dense, trunk-pose-UNGATED reward for both feet grounded AND
    # under the base (the squat-ready stance)
    if foot_xy is not None:
        tuck = feet_tuck_signal(
            foot_z,
            foot_xy,
            root_pos[:, :2],
            foot_max_z=foot_grounded_max_z,
            d_max=feet_under_base_soft_d,
        )
    else:
        tuck = np.zeros(root_pos.shape[0], dtype=np.float32)

    # Anti-detour penalty for back (supine) starts — discourage rolling
    # face-DOWN (toward prone) on the way up
    flip = np.maximum(0.0, -projected_gravity(root_quat)[:, 0]).astype(np.float32)
    if start_supine is not None:
        flip = flip * start_supine.astype(np.float32)
    supine_flip_pen = flip

    w = weights

    # ── per-group decomposition (multi-critic PPO) ──
    # Each group is a homogeneous-scale return so a per-group critic can
    # actually fit it. The three sum EXACTLY to the single-critic reward
    # `r` below, so single-critic training is unchanged.
    r_task = (
        w.upright * up_pos
        + w.height * h
        + w.upright_progress * progress
        + w.supine_situp_progress * situp_progress
        + w.explosive_rise * rise
        + w.feet_tuck * tuck
        + w.foot_grounded_up * foot_up
        + w.standing_tall * tall
        + w.stand_pose * stand_pose
        + w.knee_support * knee_support
        - w.on_spot * on_spot_pen
        - w.trunk_contact_force * trunk_force_pen
        - w.supine_anti_flip * supine_flip_pen
    ).astype(np.float32)
    r_reg = (
        -w.arm_pose_dev * arm_dev
        - w.base_ang_vel_sway * ang_sway
        - w.base_lin_vel_drift * lin_drift
        - w.joint_vel_quiet * q_quiet
        - w.action_smoothness * smooth
        - w.action_jerk * jerk
    ).astype(np.float32)

    if reg_success_ramp:
        r_reg = (r_reg * pose_scale).astype(np.float32)
    r_success = (
        w.success_persistence * persistence
        - w.time_penalty * time_pen
        + w.success_bonus * sustained_bonus
        + w.post_success_standing * post_success
        + w.post_success_still * post_success_still
    ).astype(np.float32)

    r = (r_task + r_reg + r_success).astype(np.float32)
    group_rewards = {"task": r_task, "reg": r_reg, "success": r_success}

    components = {
        "upright": float(np.mean(up_pos)),
        "upright_raw": float(np.mean(up)),
        "upright_progress": float(np.mean(progress)),
        "supine_situp_progress": float(np.mean(situp_progress)),
        "supine_anti_flip": float(np.mean(supine_flip_pen)),
        "explosive_rise": float(np.mean(rise)),
        "mean_rise_vel_z": float(np.mean(root_lin_vel[:, 2])),
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
        "feet_tuck": float(np.mean(tuck)),
        "foot_grounded_up": float(np.mean(foot_up)),
        "standing_tall": float(np.mean(tall)),
        "stand_pose": float(np.mean(stand_pose)),
        "post_success_still": float(np.mean(post_success_still)),
        "on_spot_pen": float(np.mean(on_spot_pen)),
        "mean_base_disp": float(np.mean(disp)),
        "knee_support": float(np.mean(knee_support)),
        "trunk_force_pen": float(np.mean(trunk_force_pen)),
        "mean_trunk_force": (
            float(np.mean(trunk_contact_force))
            if trunk_contact_force is not None
            else 0.0
        ),
        "stand_pose_scale": float(pose_scale),
        "supine_anti_flip": float(np.mean(supine_flip_pen)),
        "feet_under_base": float(np.mean(under_base)),
        "mean_foot_z": float(np.mean(foot_z)),
        "time_bonus_mean": float(np.mean(time_bonus_scale * sustained_now)),
        "mean_robot_z": float(np.mean(root_pos[:, 2])),
        "mean_reward": float(np.mean(r)),
        "group_task": float(np.mean(r_task)),
        "group_reg": float(np.mean(r_reg)),
        "group_success": float(np.mean(r_success)),
    }
    return r, frame_success, components, group_rewards
