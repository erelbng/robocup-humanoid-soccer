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


def explosive_rise_signal(root_lin_vel: np.ndarray, root_z: np.ndarray,
                          upright: np.ndarray,
                          target_h: float = 0.55,
                          v_cap: float = 0.8) -> np.ndarray:
    """Reward in [0, 1] for moving the trunk UPWARD fast WHILE still low.

    This is the direct "explosive rise" carrot the rest of the reward was
    missing: `upright_progress` only pays for orientation change (rate-capped,
    speed-independent total) and the speed signal is indirect (opportunity
    cost + a terminal time bonus). This term pays per-step for a high positive
    vertical trunk velocity during the recovery, which is exactly the snappy
    hip/knee extension in the reference fast-standup.

    Three multiplicative gates keep it honest:
      * `rise`    — clip(v_z, 0, v_cap)/v_cap: upward only (no credit for
                    falling), and CAPPED so a destabilising ballistic launch
                    can't out-earn a controlled fast push.
      * `deficit` — (target − z)/target: 1 when fully fallen, 0 at standing
                    height, so it can't be farmed by bouncing once already up.
      * `early`   — 1 − near_upright_gate(up): fades out as the robot nears
                    upright, leaving the final balance to the stability terms.
    """
    rise = np.clip(root_lin_vel[:, 2], 0.0, v_cap) / max(v_cap, 1e-6)
    deficit = np.clip((target_h - root_z) / max(target_h, 1e-6), 0.0, 1.0)
    early = 1.0 - near_upright_gate(upright)
    return (rise * deficit * early).astype(np.float32)


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


def feet_under_base_score(foot_xy: np.ndarray, base_xy: np.ndarray,
                           d_max: float = 0.40,
                           plateau_d: float = 0.0) -> np.ndarray:
    """Multiplicative AND in [0, 1] of how close each foot is to being
    (roughly) UNDER the base in the world horizontal plane.

    This is the signal that `_feet_grounded_score` is missing: that score
    rewards feet being LOW (z≈0), which a prone / cobra / push-up / L-sit
    pose satisfies trivially with its legs splayed FLAT on the floor — the
    feet are on the ground, but the robot is lying on them, not standing on
    them. When the robot actually stands, both feet sit roughly beneath the
    base; when the legs are extended flat they are ~0.4–0.6 m away.

    `plateau_d` is a FREE-ZONE radius: any foot within `plateau_d` of the base
    gets full credit, and the ramp to 0 runs over [plateau_d, d_max]. With the
    default 0 it peaks only at d=0 (pulling the feet together). Set it to the
    desired standing stance half-width so a SHOULDER-wide stance is NOT
    penalised, while a cobra / sprawl (d ≥ d_max) still scores 0.
    """
    # foot_xy: (N, 2, 2) [foot, xy]; base_xy: (N, 2)
    d = np.linalg.norm(foot_xy - base_xy[:, None, :], axis=2)  # (N, 2)
    excess = np.clip(d - max(plateau_d, 0.0), 0.0, None)
    span = max(d_max - max(plateau_d, 0.0), 1e-6)
    score = np.clip(1.0 - excess / span, 0.0, 1.0)
    return (score[:, 0] * score[:, 1]).astype(np.float32)


def feet_tuck_signal(foot_z: np.ndarray, foot_xy: np.ndarray,
                     base_xy: np.ndarray,
                     foot_max_z: float = 0.12,
                     d_max: float = 0.40) -> np.ndarray:
    """Dense [0, 1] reward for BOTH feet grounded AND tucked under the base —
    the squat-ready stance, paid regardless of trunk height/orientation.

    This is the missing motor primitive behind the stubborn ~0.30 m plateau
    (runs #1-3): the policy raises its torso but leaves its legs splayed flat,
    so its feet never come under its body and it can never push to standing.
    `feet_under_base_score` exists but is only used to GATE the end-of-motion
    stand rewards (multiplied by trunk-up / upright factors), so it gives ~0
    gradient while the robot is still low — exactly when the tuck must happen.
    This term is UNGATED by trunk pose: a sprawled robot with feet on the floor
    gets a smooth gradient that pulls its grounded feet inward toward under the
    hips, building the squat stance from which the other terms (height,
    standing_tall, explosive_rise) then drive the push-up.

    grounded (both feet near floor) × under_base (both feet under the base xy).
    """
    grounded = _feet_grounded_score(foot_z, foot_max_z)
    under = feet_under_base_score(foot_xy, base_xy, d_max=d_max)
    return (grounded * under).astype(np.float32)


def standing_on_feet_mask(foot_z: np.ndarray, foot_xy: np.ndarray,
                           base_xy: np.ndarray,
                           foot_max_z: float = 0.12,
                           under_base_max_d: float = 0.25) -> np.ndarray:
    """Hard boolean: both feet on the ground AND both feet under the base.

    Used to GATE success detection so the success bonus / post-success
    standing reward can only be farmed from a genuine feet-under-body stand,
    never from an assist-propped cobra / L-sit that games upright + height
    while the legs lie flat and splayed."""
    grounded = (foot_z[:, 0] < foot_max_z) & (foot_z[:, 1] < foot_max_z)
    d = np.linalg.norm(foot_xy - base_xy[:, None, :], axis=2)  # (N, 2)
    under = (d[:, 0] < under_base_max_d) & (d[:, 1] < under_base_max_d)
    return (grounded & under)


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


# ─── target standing-pose reward ──────────────────────────────────────


def stand_pose_signal(joint_pos: np.ndarray, pose_joint_indices: tuple,
                      target_pose: np.ndarray, upright: np.ndarray,
                      dev_scale: float = 1.0) -> np.ndarray:
    """Positive [0, 1] reward for holding the desired standing pose — a little
    bent knees, SHOULDER-wide feet (hips abducted), arms at the sides — ready
    to transition to walking. `target_pose` is the standup-specific target
    (the K1 default pose with the hip-roll joints abducted; see
    StandupConfig.stand_target_hip_abduction).

    `exp(-Σ(q - q_target)² / dev_scale)` over the shaped joints (arms + legs),
    peaking at 1 at the target pose. Multiplied by `near_upright_gate(up)` so
    it is 0 through the whole recovery (the policy keeps full freedom to stand
    up) and only shapes the FINAL pose. The caller additionally ramps the whole
    term with the success-rate EMA, so it stays off until the robot reliably
    stands and only then sculpts the end pose."""
    if len(pose_joint_indices) == 0 or target_pose is None:
        return np.zeros(joint_pos.shape[0], dtype=np.float32)
    dev = joint_pose_deviation(joint_pos, pose_joint_indices, target_pose)
    pose = np.exp(-dev / max(dev_scale, 1e-6))
    return (pose * near_upright_gate(upright)).astype(np.float32)


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
                       upright_threshold: float = 0.92,
                       feet_ok: np.ndarray = None) -> np.ndarray:
    """Per-frame 'looks standing' boolean. The env composes this with
    a streak counter to require sustained success.

    `feet_ok` (optional) is the hard `standing_on_feet_mask` — when
    provided, the frame only counts as success if the robot is ALSO
    standing on its feet (feet grounded + under the base). This stops the
    upright+height conditions from being farmed by an assist-propped
    cobra / L-sit, which would otherwise collect the +400 success bonus
    and the post-success standing reward without ever standing up."""
    mask = ((upright_signal(quat) > upright_threshold)
            & (root_z > target_h - 0.10))
    if feet_ok is not None:
        mask = mask & feet_ok
    return mask


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
    foot_xy: np.ndarray = None,      # (N, 2, 2) world-frame xy of left/right foot
    feet_ok: np.ndarray = None,      # (N,) bool — standing_on_feet_mask (success gate)
    start_supine: np.ndarray = None, # (N,) bool — episode started on the back
    weights=None,
    arm_joint_indices: tuple = (),   # arm dofs (K1: 2..9)
    pose_joint_indices: tuple = (),  # joints shaped to the target stand pose
    default_joint_pos: np.ndarray = None,
    stand_target_pose: np.ndarray = None,  # default w/ hips abducted (shoulder-wide)
    stand_pose_dev_scale: float = 1.0,
    success_ema: float = 0.0,         # running frame-success rate (0..1)
    stand_pose_success_ref: float = 0.5,
    start_xy: np.ndarray = None,      # (N, 2) base xy captured at reset
    on_spot_tol: float = 0.6,
    post_success_still_jv_scale: float = 3.0,
    post_success_still_v_scale: float = 0.2,
    feet_under_base_plateau_d: float = 0.0,
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

    # Explosive-rise shaping — direct per-step reward for a fast upward trunk
    # velocity while still low. Drives the snappy hip/knee extension of the
    # reference fast-standup that the (rate-capped, speed-independent)
    # `progress` term doesn't pay for. Gated to the recovery phase + capped so
    # it can't be farmed by bouncing or a destabilising ballistic launch.
    rise = explosive_rise_signal(
        root_lin_vel, root_pos[:, 2], up,
        target_h=target_height, v_cap=explosive_rise_v_cap)

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

    # Target standing-pose reward — positive, near-upright-gated reward that
    # pulls the final pose to the nominal stand (bent knees, hip-wide feet,
    # arms at the sides), ready for a walk transition. Ramped by the success-
    # rate EMA (`pose_scale`): ~0 while the policy is still learning to stand,
    # rising to full strength only once it reliably succeeds — so it never
    # disturbs the proven get-up, it only sculpts the end pose afterwards.
    _stand_target = (stand_target_pose if stand_target_pose is not None
                     else default_joint_pos)
    pose = stand_pose_signal(joint_pos, pose_joint_indices,
                             _stand_target, up,
                             dev_scale=stand_pose_dev_scale)
    pose_scale = float(np.clip(success_ema / max(stand_pose_success_ref, 1e-6),
                               0.0, 1.0))
    stand_pose = (pose * pose_scale).astype(np.float32)

    # Post-success STILLNESS — highly reward being motionless once a sustained
    # stand is achieved AND still held. Raw (ungated) joint + base kinetic
    # energy → exp() so it peaks at 1 when perfectly still. Gated to
    # (achieved_sustained & frame_success) so it ONLY ever rewards an
    # already-standing robot; it cannot touch the get-up.
    still_score = np.exp(
        -(joint_vel_quiet(joint_vel) / max(post_success_still_jv_scale, 1e-6)
          + base_lin_vel_drift(root_lin_vel)
          / max(post_success_still_v_scale, 1e-6))).astype(np.float32)

    # Stand "on the spot" — quadratic penalty on horizontal base travel from
    # the spawn xy beyond a tolerance (an in-place get-up incl. a short roll
    # pays ~0; jiggling across the field is taxed hard).
    if start_xy is not None:
        disp = np.linalg.norm(root_pos[:, :2] - start_xy, axis=1)
        on_spot_excess = np.clip(disp - on_spot_tol, 0.0, None)
        on_spot_pen = (on_spot_excess ** 2).astype(np.float32)
    else:
        disp = np.zeros(root_pos.shape[0], dtype=np.float32)
        on_spot_pen = np.zeros(root_pos.shape[0], dtype=np.float32)

    # Per-frame "looks standing" — env will count consecutive frames.
    # Gated by feet_ok (feet grounded + under base) so the post-success
    # standing reward below can't be farmed from an assist-propped cobra.
    frame_success = success_frame_mask(
        root_quat, root_pos[:, 2],
        target_h=target_height,
        upright_threshold=upright_threshold,
        feet_ok=feet_ok,
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

    # Post-success stillness — `still_score` gated to the same post-success
    # frames, so staying motionless in the held stand is highly rewarded.
    post_success_still = (post_success * still_score).astype(np.float32)

    # Anti-gaming "stand on feet" reward — only pays for the joint
    # condition (both feet grounded AND trunk lifted). Closes the bridge
    # / shoulder-stand / sprawled-on-back local optima where the policy
    # gets partial upright + height credit without putting its feet on
    # the floor. Smooth multiplicative gate provides gradient toward the
    # threshold even before satisfying it.
    # Feet-under-base gate: the missing discriminator between "standing on
    # its feet" and "lying with its feet flat". Multiplied into BOTH stand-
    # on-feet rewards so a cobra / push-up / L-sit (legs splayed flat, feet
    # at z≈0 but ~0.5 m behind the base) earns ~0 here even though
    # `_feet_grounded_score` alone would pay it full credit. Smooth ramp so
    # the policy still gets a gradient that pulls its feet inward. Defaults
    # to 1.0 (no-op) when foot_xy is unavailable.
    if foot_xy is not None:
        under_base = feet_under_base_score(
            foot_xy, root_pos[:, :2], d_max=feet_under_base_soft_d,
            plateau_d=feet_under_base_plateau_d)
    else:
        under_base = np.ones(root_pos.shape[0], dtype=np.float32)

    foot_up = foot_grounded_up_signal(
        foot_z, root_pos[:, 2], up,
        foot_max_z=foot_grounded_max_z,
        trunk_min_z=trunk_up_min_z,
    ) * under_base
    tall = standing_tall_signal(
        foot_z, root_pos[:, 2], up,
        foot_max_z=foot_grounded_max_z,
        trunk_min_z=standing_tall_min_z,
        trunk_max_z=standing_tall_max_z,
    ) * under_base

    # Feet-tuck — dense, trunk-pose-UNGATED reward for both feet grounded AND
    # under the base (the squat-ready stance). Teaches the missing primitive
    # behind the sprawled ~0.30 m plateau: pull the grounded feet under the
    # hips so the other terms can then drive the push to standing.
    if foot_xy is not None:
        tuck = feet_tuck_signal(
            foot_z, foot_xy, root_pos[:, :2],
            foot_max_z=foot_grounded_max_z, d_max=feet_under_base_soft_d)
    else:
        tuck = np.zeros(root_pos.shape[0], dtype=np.float32)

    # Anti-detour penalty for back (supine) starts — discourage rolling
    # face-DOWN (toward prone) on the way up. Body-frame gravity-x is +1
    # supine (on the back), -1 prone (belly-down), ~0 upright/side, so
    # max(0, -proj_g_x) is >0 only once a back-start robot has flipped
    # belly-down (the roll-to-prone / cobra detour). A clean sit-up/roll-up
    # keeps proj_g_x ≥ 0 → zero penalty. Gated to supine-start envs so prone
    # recovery is untouched.
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
        + w.explosive_rise * rise
        + w.feet_tuck * tuck
        + w.foot_grounded_up * foot_up
        + w.standing_tall * tall
        + w.stand_pose * stand_pose
        - w.on_spot * on_spot_pen
        - w.supine_anti_flip * supine_flip_pen
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
        + w.post_success_still * post_success_still
    ).astype(np.float32)

    r = (r_task + r_reg + r_success).astype(np.float32)
    group_rewards = {"task": r_task, "reg": r_reg, "success": r_success}

    components = {
        "upright": float(np.mean(up_pos)),
        "upright_raw": float(np.mean(up)),
        "upright_progress": float(np.mean(progress)),
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
