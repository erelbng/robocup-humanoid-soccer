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


def forward_progress_reward(quat: np.ndarray, vel_world: np.ndarray,
                            cmd_vx: np.ndarray, cmd_vy: np.ndarray
                            ) -> np.ndarray:
    """Fraction of the COMMANDED planar velocity actually achieved in the
    commanded direction, ∈[0,1]. Linear → CONSTANT gradient "go faster toward
    the command", unlike exp tracking whose gradient vanishes far from target.

    This is the fix for march-in-place: exp tracking barely rewarded creeping
    forward, so the policy lifted its feet (feet_swing) without translating.
    A linear projection reward `(v·cmd)/|cmd|²` clamped to [0,1] pulls actual
    forward speed up monotonically. Zero when the command is ~stationary.
    """
    v = body_frame_velocity(quat, vel_world)            # (N,3) body frame
    cmd_sq = cmd_vx ** 2 + cmd_vy ** 2                   # |cmd|²
    proj = v[:, 0] * cmd_vx + v[:, 1] * cmd_vy           # v·cmd
    frac = np.where(cmd_sq > 0.0025,
                    np.clip(proj / (cmd_sq + 1e-6), 0.0, 1.0), 0.0)
    return frac.astype(np.float32)


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


def feet_swing_reward(gait_phase: np.ndarray, contact_mask: np.ndarray,
                      swing_period: float = 0.2) -> np.ndarray:
    """Booster T1's proven anti-shuffle term (booster_gym/envs/t1.py).

    Reward +1 for each foot that is AIRBORNE during its scheduled swing
    window — left foot around gait phase 0.25, right foot around 0.75, each
    window `swing_period` wide. With a DOMINANT weight (Booster uses 3.0, vs
    1.0 for velocity tracking) this makes lifting the right foot at the right
    time the single most rewarding thing the policy can do, so it can't sit in
    the shuffle optimum. The gait phase is also in the obs (clock sin/cos), so
    the policy can time it. Returns (N,) ∈ {0,1,2}.
    """
    p = gait_phase
    half = 0.5 * swing_period
    left_swing = np.abs(p - 0.25) < half
    right_swing = np.abs(p - 0.75) < half
    nc = ~contact_mask                                   # (N,2) airborne
    return ((left_swing & nc[:, 0]).astype(np.float32)
            + (right_swing & nc[:, 1]).astype(np.float32))


def swing_height_reward(feet_z: np.ndarray, desired_contact: np.ndarray,
                        stand_z: float = 0.065, target_clear: float = 0.06
                        ) -> np.ndarray:
    """Phase-conditioned foot-LIFT reward — the missing initiation gradient.

    When the gait clock says a foot should be in SWING (`desired_contact`≈0),
    reward that foot for being OFF the ground, ramping linearly with how far
    the foot-link sits above its standing height up to `target_clear`. Unlike
    foot_clearance (gated on `~contact`, so zero until the foot already left
    the ground) and feet_air_time (paid only after a completed swing), this is
    active WHILE the foot is still planted — so the gradient points UP and the
    policy can discover lifting from a flat-footed shuffle. THIS is what breaks
    the shuffle local optimum; the other gait terms only shape it once stepping.

    `feet_z` (N,2) foot-link heights; `desired_contact` (N,2)∈[0,1] stance
    target. Returns (N,) mean over feet ∈ [0,1].
    """
    lift = np.clip((feet_z - stand_z) / target_clear, 0.0, 1.0)   # (N,2)
    swing_desire = 1.0 - desired_contact                          # (N,2)
    return (lift * swing_desire).mean(axis=1).astype(np.float32)


def feet_air_time_reward(air_time: np.ndarray,
                         contact_just_now: np.ndarray,
                         target_air_time: float = 0.4) -> np.ndarray:
    """rsl_rl / legged_gym-style: reward a foot's accumulated swing (air)
    time AT the moment it touches down. This is THE classic anti-shuffle
    term — a foot that never leaves the ground has air_time≈0 and earns
    nothing, so the policy is pushed to lift-swing-place real steps.

    `air_time`: (N, 2) seconds each foot has been airborne (read at the
    touchdown step, before it's zeroed). `contact_just_now`: (N, 2) bool —
    True for a foot that JUST landed this step. Bonus is `air_time - 0.2`
    (so only swings longer than a shuffle-flick count), capped at
    `target_air_time`, and only paid on the landing step.
    """
    bonus = np.clip(air_time - 0.2, 0.0, target_air_time)          # (N, 2)
    return np.sum(bonus * contact_just_now.astype(np.float32),
                  axis=1).astype(np.float32)


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


def gait_contact_reward(contact_mask: np.ndarray,
                        desired_contact: np.ndarray) -> np.ndarray:
    """Reward actual foot contact matching the clock-phased desired pattern.

    `contact_mask` (N,2) bool — feet actually in contact; `desired_contact`
    (N,2) ∈ [0,1] — the smooth alternating stance/swing target from the gait
    phase. Returns (N,) ∈ [0,1]: per-foot match averaged. A proper step
    (stance foot down, swing foot up) → ~1.0; a SHUFFLE (both feet always
    down) only matches the stance foot → ~0.5. This is the anti-shuffle term.
    """
    c = contact_mask.astype(np.float32)
    return (1.0 - np.abs(c - desired_contact)).mean(axis=1).astype(np.float32)


def feet_slip_penalty(contact_mask: np.ndarray,
                      foot_horiz_speed: np.ndarray) -> np.ndarray:
    """Penalty Σ (horizontal foot speed)² over feet that are IN CONTACT —
    i.e. punish sliding/skating a planted foot. (N,2)→(N,)."""
    c = contact_mask.astype(np.float32)
    return np.sum(c * (foot_horiz_speed ** 2), axis=1).astype(np.float32)


def arm_swing_reward(jvel: np.ndarray, l_shoulder: int, r_shoulder: int,
                     cmd_speed: np.ndarray) -> np.ndarray:
    """Encourage a little natural armswing: reward the two shoulder-pitch
    joints moving in OPPOSITE directions (the human anti-phase swing), only
    while the robot is commanded to move.

    `-(vL·vR)` is positive exactly when the shoulders rotate opposite ways
    and zero when arms are static — so it can't be gamed by freezing the
    arms in an offset pose (that's what the relaxed arm_pose penalty allows).
    `tanh` bounds it to [0,1); the speed gate stops it rewarding arm-flapping
    while standing still. Phase-vs-legs is left emergent (no sign assumption).
    """
    vL = jvel[:, l_shoulder]
    vR = jvel[:, r_shoulder]
    anti = np.clip(-(vL * vR), 0.0, None)            # opposite-direction product
    gate = np.clip(np.abs(cmd_speed), 0.0, 1.0)      # only while moving
    return (np.tanh(anti / 2.0) * gate).astype(np.float32)


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
    desired_contact: np.ndarray = None,   # (N,2) clock-phased stance/swing target
    foot_horiz_speed: np.ndarray = None,  # (N,2) m/s horizontal foot speed
    air_time: np.ndarray = None,          # (N,2) s each foot airborne (pre-zero)
    contact_just_now: np.ndarray = None,  # (N,2) bool — foot landed this step
    gait_phase: np.ndarray = None,        # (N,) gait clock ∈[0,1) for feet_swing
    head_commands: np.ndarray = None,    # (N, 2) target yaw/pitch — optional
    head_joint_indices: tuple = (),      # K1 → (0, 1) for AAHead_yaw/Head_pitch
    arm_joint_indices: tuple = (),       # K1 → (2..9) for shoulder/elbow joints
    shoulder_pitch_indices: tuple = (),  # K1 → (2, 6) left/right shoulder pitch
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
    ang = track_ang_vel(root_ang_vel, cmd_vyaw,
                        sigma=getattr(weights, "ang_tracking_sigma", 0.4))
    fwd = forward_progress_reward(root_quat, root_lin_vel, cmd_vx, cmd_vy)
    up = upright_reward(root_quat)
    h = height_reward(root_pos, target_h=0.5)   # K1 stands ≈0.5 m
    clearance = foot_clearance_reward(feet_z, cmd_clearance, contact_mask)

    # Gait-contact (anti-shuffle) + foot-slip — the natural-gait terms.
    gait_c = np.zeros(n, dtype=np.float32)
    swing_h = np.zeros(n, dtype=np.float32)
    if desired_contact is not None:
        gait_c = gait_contact_reward(contact_mask, desired_contact)
        # Phase-conditioned lift gradient — pulls the swing foot off the ground.
        swing_h = swing_height_reward(feet_z, desired_contact)

    # Booster T1's dominant anti-shuffle term: foot airborne in its swing window.
    feet_sw = np.zeros(n, dtype=np.float32)
    if gait_phase is not None:
        sp = getattr(weights, "swing_period", 0.2)
        feet_sw = feet_swing_reward(gait_phase, contact_mask, swing_period=sp)
    slip = np.zeros(n, dtype=np.float32)
    if foot_horiz_speed is not None:
        slip = feet_slip_penalty(contact_mask, foot_horiz_speed)

    # Feet air-time — reward real swings (foot lifted) at touchdown.
    air = np.zeros(n, dtype=np.float32)
    if air_time is not None and contact_just_now is not None:
        air = feet_air_time_reward(air_time, contact_just_now)

    # Armswing — anti-phase shoulder-pitch motion while moving.
    arm_sw = np.zeros(n, dtype=np.float32)
    if len(shoulder_pitch_indices) == 2:
        cmd_speed = np.abs(cmd_vx) + np.abs(cmd_vy)
        arm_sw = arm_swing_reward(jvel, shoulder_pitch_indices[0],
                                  shoulder_pitch_indices[1], cmd_speed)

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
    # Posture as PENALTIES, not bonuses (Booster recipe): `up`,`h`∈[0,1] are 1
    # when perfectly upright / at target height, so (up-1) and (h-1) are ≤0 —
    # standing earns ZERO from posture (not +2.0). This is the fix for the
    # stand-still local optimum: previously big positive upright+height made
    # doing nothing more rewarding than risking a step. Now the only way to
    # score positively is to track the command + step (feet_swing).
    r = (
        w.track_lin_vel * lin
        + getattr(w, "forward_progress", 0.0) * fwd
        + w.track_ang_vel * ang
        + w.upright * (up - 1.0)
        + w.height * (h - 1.0)
        + w.foot_clearance * clearance
        + getattr(w, "gait_contact", 0.0) * gait_c
        + getattr(w, "swing_height", 0.0) * swing_h
        + getattr(w, "feet_swing", 0.0) * feet_sw
        - getattr(w, "feet_slip", 0.0) * slip
        + getattr(w, "feet_air_time", 0.0) * air
        + getattr(w, "arm_swing", 0.0) * arm_sw
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
        "forward_progress": float(np.mean(fwd)),
        "track_ang_vel": float(np.mean(ang)),
        "upright": float(np.mean(up)),
        "height": float(np.mean(h)),
        "foot_clearance": float(np.mean(clearance)),
        "gait_contact": float(np.mean(gait_c)),
        "swing_height": float(np.mean(swing_h)),
        "feet_swing": float(np.mean(feet_sw)),
        "feet_slip": float(np.mean(slip)),
        "feet_air_time": float(np.mean(air)),
        "arm_swing": float(np.mean(arm_sw)),
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
