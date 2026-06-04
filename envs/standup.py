"""
Standup curriculum stage for Phase 1.

Resets the robot in one of several "fallen" poses (supine, prone, side-left,
side-right) and rewards getting the trunk upright above a target height.
This stage is meant to run AFTER `stand` but BEFORE `walk`, so the policy
can recover from a fall during later stages instead of terminating the
episode every time.

We deliberately keep this independent from the dribble/walk reward — it
is its own reward function. The trainer can also OPT to call this on a
sub-episode basis whenever a fall is detected during walk/dribble training
(see `should_trigger_during_episode`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# Joint-name → angle (rad). Use the SAME names as configs.config.K1RobotConfig
# so the lookup matches the URDF.

# Lying on back: hips flat, knees bent only slightly, arms by side. Robot
# starts with trunk pitch ≈ -π/2 (face up).
_POSE_SUPINE = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0, "Left_Shoulder_Roll": -0.1,
    "Left_Elbow_Pitch": 0.0, "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.0, "Right_Shoulder_Roll": 0.1,
    "Right_Elbow_Pitch": 0.0, "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": -0.2, "Left_Hip_Roll": 0.0, "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.4, "Left_Ankle_Pitch": -0.2, "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.2, "Right_Hip_Roll": 0.0, "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.4, "Right_Ankle_Pitch": -0.2, "Right_Ankle_Roll": 0.0,
}

# Lying face down: arms in push-up-ready position, knees slightly bent.
#
# ELBOW PENETRATION ANALYSIS (prone: body +X → world -Z, i.e. DOWN):
#   In prone the trunk rests at z ≈ 0.13 m. Shoulder pitch controls how far
#   the arm swings toward the floor (body +X = world -Z). The elbow hangs at:
#       elbow_z ≈ trunk_z − upper_arm_len × sin(shoulder_pitch)
#   At trunk_z = 0.13 m, upper_arm ≈ 0.25 m:
#       pitch = 0.80 → elbow_z ≈ −0.049 m  (UNDERGROUND by 5 cm!)
#       pitch = 0.65 → elbow_z ≈ −0.021 m  (still underground)
#       pitch = 0.50 → elbow_z ≈ +0.010 m  (just above floor — safe reference)
#   The pool penetration filter (eps=0.02) rejects the underground states, but
#   the reference pose itself should be safe so that even after reset-time
#   joint jitter (±0.10 rad → worst-case pitch = 0.60 → elbow ≈ −0.011 m,
#   just above the −0.02 rejection threshold) the filter keeps the state.
#   Using 0.50 rad keeps the reference elbow firmly above the floor.
_POSE_PRONE = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.1,
    "ALeft_Shoulder_Pitch": 0.5, "Left_Shoulder_Roll": 0.0,
    "Left_Elbow_Pitch": -0.4, "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.5, "Right_Shoulder_Roll": 0.0,
    "Right_Elbow_Pitch": -0.4, "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": 0.2, "Left_Hip_Roll": 0.0, "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": -0.3, "Left_Ankle_Pitch": 0.1, "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": 0.2, "Right_Hip_Roll": 0.0, "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": -0.3, "Right_Ankle_Pitch": 0.1, "Right_Ankle_Roll": 0.0,
}


# Trunk orientation (quaternion w,x,y,z) for each pose. Genesis URDF
# spawns aligned with world; we rotate around the X axis (pitch) to lay
# the robot down.
def _quat_from_axis_angle(axis: Tuple[float, float, float], angle: float):
    half = angle / 2.0
    s = math.sin(half)
    n = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) or 1.0
    return (math.cos(half),
            axis[0] / n * s, axis[1] / n * s, axis[2] / n * s)


@dataclass
class StandupPose:
    name: str
    joint_targets: dict
    trunk_quat: Tuple[float, float, float, float]
    trunk_height: float  # initial Z of trunk so it's roughly resting on
                         # the carpet (trunk thickness ~0.10m)
    # Vertical clearance (m) ADDED to trunk_height at spawn, before the
    # robot free-falls and settles. A limb that hangs more than this below
    # the trunk centre spawns INSIDE the floor — the PD then holds it
    # embedded through the settle and the snapshot captures a limb-in-ground
    # state (which, replayed at reset, pins the limb and corrupts the contact
    # reward). SIDE poses splay the down-arm far below the trunk → 0.30.
    # BACK/FRONT poses don't splay an arm, but the large pool quat/joint
    # noise (±17°) can still rotate a foot well below the trunk centre, so a
    # tiny 0.05 clearance is NOT safe — use ≥0.12 (default below) and let the
    # pose-pool penetration filter reject any residual buried-limb snapshots.
    spawn_clearance: float = 0.13
    # When True, the pool build overrides the arm joint TARGETS with wide
    # uniform-random values (within URDF joint limits) per env per round,
    # producing diverse arm configurations (crossed / one-up-one-down / bent /
    # twisted). The settle physics + penetration/orientation filters cull any
    # invalid result. The arm entries in `joint_targets` then act only as a
    # harmless fallback. Used by prone (see prone()).
    arm_random: bool = False


def supine() -> StandupPose:
    # Face up (lying on the back): rotate -π/2 about Y axis. Verified via
    # body-frame gravity — this orientation gives g_x = -1, which the env's
    # orientation test (`g[:,0] < -0.5`) classifies as on-the-back. Paired
    # with the arms-by-side joint preset (natural supine rest pose).
    q = _quat_from_axis_angle((0, 1, 0), -math.pi / 2)
    return StandupPose("supine", _POSE_SUPINE, q, trunk_height=0.13)


def prone() -> StandupPose:
    # Face down (lying on the belly): rotate +π/2 about Y axis. Gives
    # body-frame g_x = +1 (front faces the floor). Paired with the
    # arms-forward joint preset — the push-up-ready prone start.
    q = _quat_from_axis_angle((0, 1, 0), math.pi / 2)
    # arm_random: the pool build samples the 8 arm joints uniformly within
    # their limits so prone starts show many arm configurations (crossed,
    # one up one down, bent, twisted) rather than the single _POSE_PRONE arms.
    return StandupPose("prone", _POSE_PRONE, q, trunk_height=0.13,
                       arm_random=True)


# ── Side-pose joint targets ──────────────────────────────────────────────
#
# ROTATION ANALYSIS (determines which arm/leg is the "down" limb):
#
#   side_left  quat = rot(X, +π/2):  body -Y → world -Z (FLOOR).
#     ∴ RIGHT side on floor: RIGHT arm is DOWN arm, RIGHT leg is BOTTOM leg.
#
#   side_right quat = rot(X, -π/2):  body +Y → world -Z (FLOOR).
#     ∴ LEFT side on floor: LEFT arm is DOWN arm, LEFT leg is BOTTOM leg.
#
# SHOULDER ROLL convention (verified by code + shell):
#   Right arm roll > 0 → arm toward body -Y → floor-ward in side_left  ← USE for DOWN arm brace
#   Left  arm roll > 0 → arm toward body +Y → floor-ward in side_right ← USE for DOWN arm brace
#   (Same sign lifts the UP arm away from the floor in the other pose variant)
#
# HIP ROLL convention (assumed same pattern):
#   Right hip roll > 0 → right leg toward body -Y → floor-ward in side_left
#   Left  hip roll > 0 → left  leg toward body +Y → floor-ward in side_right
#
# STABILISATION STRATEGY (floor-brace tripod):
#   The robot must stay on its side through the settle physics. Without a brace
#   the PD joint torques roll the trunk within ~60 steps of landing.
#   Three contact points create a stable tripod:
#     1) Torso/shoulder against floor (natural)
#     2) DOWN ARM ELBOW on the floor — the key stabiliser. The arm is pitched
#        strongly forward (1.2 rad) so the upper-arm end (elbow) just clears the
#        floor (+0.04 m at trunk_z=0.13 m), then rolled floor-ward (0.4 rad) so
#        it settles flush. Large elbow-bend (1.1 rad) makes the elbow joint — NOT
#        the hand — the forward contact point.
#     3) BOTTOM FOOT on the floor. Hip roll floors-ward pushes the foot down.
#        Slight knee bend (0.5 rad) and ankle compensate for the body rotation.
#   The UP arm rests on/above the body (sky-ward roll); top leg is relaxed.
#
# ELBOW GEOMETRY for DOWN arm (side_left, trunk at 0.13 m):
#   Arm neutral = body -Z (hanging down in upright). Roll > 0 rotates toward
#   body -Y (floor in side_left) for right arm; toward body +Y (floor in
#   side_right) for left arm.
#   arm_body = (sin(P), ∓sin(R)·cos(P), -cos(R)·cos(P))   [∓ = right/left]
#   In world (side_left, R_bw·(x,y,z)=(x,-z,y)):
#     arm_world_z = -sin(R)·cos(P)
#     elbow_world_z = trunk_z + arm_world_z · upper_arm_len
#
#   P=0.6, R=0.65 → elbow_z = 0.13 - sin(0.65)·cos(0.6)·0.25 = +0.005 m ✓
#   Elbow just 5 mm above floor at settled height → contacts floor after settle.
#   P=0.6, R=0.50 → elbow_z = +0.031 m (floats; less stable, avoids penetration).
#   At SPAWN (trunk_z = 0.13+0.45 = 0.58 m): elbow_spawn_z = 0.455 m → safe ✓.
#   Large Elbow_Pitch (1.1 rad) bends forearm back so ELBOW JOINT = contact point.

# side_left: RIGHT arm is DOWN arm, RIGHT leg is BOTTOM leg.
_POSE_SIDE_LEFT = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    # DOWN arm (right) — FLOOR BRACE.
    # Pitch 0.6 + Roll 0.65 places elbow +0.005 m above floor → settles to contact.
    # Elbow_Pitch 1.1 makes the elbow joint (not the hand) the brace contact.
    # This tripod (torso + elbow + foot) resists rolling toward belly OR back.
    "ARight_Shoulder_Pitch": 0.6, "Right_Shoulder_Roll": 0.65,
    "Right_Elbow_Pitch": 1.1,    "Right_Elbow_Yaw": 0.0,
    # UP arm (left) — Roll −π/2 = the HANGING value (arms-at-sides default).
    # The URDF joint-zero is the T-POSE (arm straight out to the side), so
    # Roll 0.0 made the left arm point along body +Y; in side_left body +Y →
    # world +Z, i.e. straight to the SKY (the bug). The hanging value −1.5708
    # lays the upper arm along the body (world-horizontal); gravity then sags
    # the forearm down so the whole arm rests laid-down beside the torso.
    "ALeft_Shoulder_Pitch": 0.2, "Left_Shoulder_Roll": -1.5708,
    "Left_Elbow_Pitch": 0.3,     "Left_Elbow_Yaw": 0.0,
    # BOTTOM leg (right) — Hip_Roll 0.3 floor-ward (body -Y = world floor).
    # Brings foot toward the ground; knee 0.5 for compact stable stance;
    # ankle compensates body rotation to improve foot contact.
    "Right_Hip_Pitch": 0.2, "Right_Hip_Roll": 0.3, "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.5, "Right_Ankle_Pitch": -0.2, "Right_Ankle_Roll": -0.2,
    # TOP leg (left) — slight opposite roll lifts leg away; relaxed.
    "Left_Hip_Pitch": 0.1, "Left_Hip_Roll": 0.15, "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.3, "Left_Ankle_Pitch": -0.1, "Left_Ankle_Roll": 0.1,
}

# side_right: LEFT arm is DOWN arm, LEFT leg is BOTTOM leg (mirror of side_left).
# By symmetry: same pitch/roll magnitudes, sign of Roll flips floor direction
# (left arm roll > 0 → body +Y = world floor in side_right) → same brace result.
_POSE_SIDE_RIGHT = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    # DOWN arm (left) — FLOOR BRACE (mirror geometry, same elbow_z = +0.005 m).
    "ALeft_Shoulder_Pitch": 0.6, "Left_Shoulder_Roll": 0.65,
    "Left_Elbow_Pitch": 1.1,    "Left_Elbow_Yaw": 0.0,
    # UP arm (right) — Roll +π/2 = the HANGING value (mirror of side_left).
    # Roll 0.0 = T-pose → right arm along body −Y → world +Z (sky) in
    # side_right. The hanging value +1.5708 lays it along the body
    # (world-horizontal); gravity sags it down into a laid-down rest pose.
    "ARight_Shoulder_Pitch": 0.2, "Right_Shoulder_Roll": 1.5708,
    "Right_Elbow_Pitch": 0.3,    "Right_Elbow_Yaw": 0.0,
    # BOTTOM leg (left) — Left Hip_Roll 0.3 floor-ward (body +Y = world floor).
    "Left_Hip_Pitch": 0.2, "Left_Hip_Roll": 0.3, "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.5, "Left_Ankle_Pitch": -0.2, "Left_Ankle_Roll": 0.2,
    # TOP leg (right) — slight roll lifts leg away; relaxed.
    "Right_Hip_Pitch": 0.1, "Right_Hip_Roll": 0.15, "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.3, "Right_Ankle_Pitch": -0.1, "Right_Ankle_Roll": -0.1,
}


def side_left() -> StandupPose:
    q = _quat_from_axis_angle((1, 0, 0), math.pi / 2)
    # Side poses need generous spawn clearance: the down-arm extends below
    # the trunk centre, so a small clearance buries it at spawn and the PD
    # holds it embedded through the settle. 0.45 m ensures the whole body
    # (including the down shoulder/elbow) spawns well above the floor.
    return StandupPose("side_left", _POSE_SIDE_LEFT, q, trunk_height=0.13,
                       spawn_clearance=0.45)


def side_right() -> StandupPose:
    q = _quat_from_axis_angle((1, 0, 0), -math.pi / 2)
    return StandupPose("side_right", _POSE_SIDE_RIGHT, q, trunk_height=0.13,
                       spawn_clearance=0.45)


def all_poses() -> List[StandupPose]:
    return [supine(), prone(), side_left(), side_right()]


# ─── Upright crouch/squat poses (reverse start-state get-up curriculum) ──────
#
# These are NOT fallen poses — they are stable, UPRIGHT crouches at descending
# heights used by the reverse-height curriculum: the policy first learns to
# finish standing from a shallow crouch (easy), then from progressively deeper
# squats, before finally tackling the full fallen→stand get-up. Built from the
# robot's DEFAULT standing pose plus a flexion delta on hip-pitch / knee /
# ankle-pitch (scaled per stage), torso kept upright (identity quaternion).
# Deriving from the known default (rather than hand-tuned absolute angles) keeps
# the squat geometrically consistent with the standing pose across joint conventions.


def make_crouch_pose(name: str,
                     default_jpos,
                     joint_names,
                     bend_scale: float,
                     trunk_height: float,
                     d_hip: float = -0.6,
                     d_knee: float = 0.9,
                     d_ankle: float = -0.5) -> StandupPose:
    """Build an upright crouch `StandupPose` = default standing pose + a
    `bend_scale`-scaled flexion delta on the leg pitch joints.

    Sign convention (from K1 default standing hip=-0.2, knee=+0.4, ankle=-0.25):
    deeper squat = more-negative hip-pitch, more-positive knee, more-negative
    ankle-pitch — so the defaults d_hip<0, d_knee>0, d_ankle<0 deepen the squat
    while keeping the feet flat under the torso. Non-leg joints (arms/head) are
    left at the default (not added to the dict → env keeps them at default).
    `trunk_quat` is identity (upright). Physics settling resolves the exact
    resting height; `trunk_height` is just the spawn height.
    """
    targets = {}
    for i, jn in enumerate(joint_names):
        if i >= len(default_jpos):
            break
        lo = jn.lower()
        base = float(default_jpos[i])
        if "hip" in lo and "pitch" in lo:
            targets[jn] = base + bend_scale * d_hip
        elif "knee" in lo:
            targets[jn] = base + bend_scale * d_knee
        elif "ankle" in lo and "pitch" in lo:
            targets[jn] = base + bend_scale * d_ankle
    return StandupPose(name, targets, (1.0, 0.0, 0.0, 0.0), trunk_height)


# ─── Reward components specific to standup ──────────────────────────


def upright_reward(robot_quat: np.ndarray) -> float:
    """1 when trunk Z-axis aligns with world Z (perfectly upright),
    -1 when fully inverted. Stable signal across the whole standup arc.
    """
    w, x, y, z = robot_quat
    # The trunk-frame Z axis expressed in world coordinates is the third
    # column of the rotation matrix derived from the quaternion. Its
    # world-Z component:
    z_axis_world_z = 1.0 - 2.0 * (x * x + y * y)
    return float(z_axis_world_z)


def height_reward(robot_z: float, target_h: float = 0.55,
                  sigma: float = 0.15) -> float:
    """Gaussian on trunk height — saturates near the target standing
    height so the policy isn't rewarded for jumping above it."""
    err = robot_z - target_h
    return float(math.exp(-(err ** 2) / (sigma ** 2)))


def standup_success(robot_quat: np.ndarray, robot_z: float,
                    *, target_h: float = 0.55,
                    upright_threshold: float = 0.92) -> bool:
    """True when the robot is upright AND at standing height. Used as a
    terminal reward bonus and as a curriculum-advance criterion."""
    return upright_reward(robot_quat) > upright_threshold and \
        robot_z > target_h - 0.10


def compute_standup_reward(
    robot_quat: np.ndarray,
    robot_z: float,
    joint_vel: np.ndarray,
    actions: np.ndarray,
    prev_actions: np.ndarray,
    *,
    upright_weight: float = 5.0,
    height_weight: float = 3.0,
    energy_weight: float = 0.005,
    action_smoothness_weight: float = 0.1,
    success_bonus: float = 50.0,
) -> Tuple[float, dict]:
    """Reward = upright + height + smoothness penalties + success bonus.

    Returns (scalar reward, components dict for logging).
    """
    up = upright_reward(robot_quat)
    h = height_reward(robot_z)
    energy = float(np.sum(np.square(joint_vel)))
    smooth = float(np.sum(np.square(actions - prev_actions)))
    success = standup_success(robot_quat, robot_z)

    r = (upright_weight * up
         + height_weight * h
         - energy_weight * energy
         - action_smoothness_weight * smooth)
    if success:
        r += success_bonus

    return r, {
        "standup/upright": up,
        "standup/height": h,
        "standup/energy": energy,
        "standup/smooth": smooth,
        "standup/success": float(success),
    }


# ─── Trigger logic for in-episode standup ───────────────────────────


def should_trigger_during_episode(robot_z: float, robot_quat: np.ndarray,
                                  *, fallen_z: float = 0.30,
                                  fallen_upright: float = 0.3) -> bool:
    """Decide if a walk/dribble episode should switch to "standup mode"
    rather than terminate immediately. Lets the policy practice recovery
    inline."""
    return robot_z < fallen_z or upright_reward(robot_quat) < fallen_upright
