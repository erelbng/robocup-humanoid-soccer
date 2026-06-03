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

# Lying face down: arms forward, knees slightly bent.
_POSE_PRONE = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.1,
    "ALeft_Shoulder_Pitch": 0.8, "Left_Shoulder_Roll": 0.0,
    "Left_Elbow_Pitch": -0.5, "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.8, "Right_Shoulder_Roll": 0.0,
    "Right_Elbow_Pitch": -0.5, "Right_Elbow_Yaw": 0.0,
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
    # robot free-falls and settles. 0.05 is fine for back/front lying where
    # the trunk thickness is the lowest point. SIDE poses splay the down-arm
    # well below the trunk centre, so spawning at +0.05 buries that arm in
    # the floor — the PD then holds it embedded through the settle and the
    # snapshot captures an arm-in-ground state. Side poses raise this so the
    # whole body (arm included) starts above the floor and settles cleanly.
    spawn_clearance: float = 0.05


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
    return StandupPose("prone", _POSE_PRONE, q, trunk_height=0.13)


# Lying on LEFT side (right side up). Left arm (ground contact) angled forward
# and bent for push-up leverage; right arm (airborne) relaxed against body.
# Physics settling resolves the final contact configuration.
_POSE_SIDE_LEFT = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.5,  "Left_Shoulder_Roll": -0.5,
    "Left_Elbow_Pitch": -0.5,     "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": -0.2, "Right_Shoulder_Roll": 0.3,
    "Right_Elbow_Pitch": 0.0,     "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": -0.1,  "Left_Hip_Roll": 0.0,  "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.3,  "Left_Ankle_Pitch": -0.1, "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.1, "Right_Hip_Roll": 0.0, "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.3, "Right_Ankle_Pitch": -0.1, "Right_Ankle_Roll": 0.0,
}

# Lying on RIGHT side — mirror of _POSE_SIDE_LEFT.
_POSE_SIDE_RIGHT = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": -0.2, "Left_Shoulder_Roll": -0.3,
    "Left_Elbow_Pitch": 0.0,      "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.5, "Right_Shoulder_Roll": 0.5,
    "Right_Elbow_Pitch": -0.5,    "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": -0.1,  "Left_Hip_Roll": 0.0,  "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.3,  "Left_Ankle_Pitch": -0.1, "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.1, "Right_Hip_Roll": 0.0, "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.3, "Right_Ankle_Pitch": -0.1, "Right_Ankle_Roll": 0.0,
}


def side_left() -> StandupPose:
    q = _quat_from_axis_angle((1, 0, 0), math.pi / 2)
    # Larger spawn clearance: on its side the splayed down-arm reaches far
    # below the trunk centre, so it needs ~0.30 m to clear the floor at spawn.
    return StandupPose("side_left", _POSE_SIDE_LEFT, q, trunk_height=0.13,
                       spawn_clearance=0.30)


def side_right() -> StandupPose:
    q = _quat_from_axis_angle((1, 0, 0), -math.pi / 2)
    # See side_left: the down-arm needs extra clearance to avoid spawning
    # inside the floor.
    return StandupPose("side_right", _POSE_SIDE_RIGHT, q, trunk_height=0.13,
                       spawn_clearance=0.30)


def all_poses() -> List[StandupPose]:
    return [supine(), prone(), side_left(), side_right()]


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
