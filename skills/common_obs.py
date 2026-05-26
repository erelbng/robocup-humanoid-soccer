"""Shared base observation builder for all skills.

Every skill policy sees the same 78-dim proprioceptive base obs, plus an
optional command vector (skill-specific dim) and any skill-specific
add-ons (e.g. ball pos/vel for dribble, goal target for shoot).

Layout (float32, shape (N, 78)):

    [0  : 1 ]  root height (z)
    [1  : 4 ]  projected gravity in body frame  (orientation proxy)
    [4  : 7 ]  body-frame linear velocity
    [7  :10 ]  body-frame angular velocity
    [10 :32 ]  joint pos − default_joint_pos
    [32 :54 ]  joint vel
    [54 :76 ]  last action
    [76 :78 ]  clock signal (sin, cos) for periodic-gait conditioning

We deliberately exclude world-frame XY position — policies should be
translation-invariant. Orientation is encoded as projected gravity
rather than raw quat to avoid the quaternion sign ambiguity (q and −q
represent the same rotation, but the value swings between rollouts).
"""

from __future__ import annotations

import numpy as np


SKILL_BASE_OBS_DIM = 78


# ─── orientation helpers ───────────────────────────────────────────────


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Gravity vector (0, 0, −1) expressed in body frame.

    Args:
        quat: (N, 4) array in (w, x, y, z) order — world→body rotation.

    Returns:
        (N, 3) float32 array. When the robot stands upright with quat
        (1,0,0,0), output is (0, 0, −1).
    """
    w = quat[:, 0]; x = quat[:, 1]; y = quat[:, 2]; z = quat[:, 3]
    # R = world→body rotation matrix. Body-frame world-z direction is
    # the third column of R^T = third row of R. With g_world = (0,0,-1):
    # g_body = R · g_world = -R[:, 2]
    gx = -2.0 * (x * z + w * y)
    gy = -2.0 * (y * z - w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.stack([gx, gy, gz], axis=-1).astype(np.float32)


def body_frame_velocity(quat: np.ndarray, vel_world: np.ndarray
                        ) -> np.ndarray:
    """Rotate a world-frame velocity into body frame.

    Uses only yaw rotation — locomotion policies should be yaw-invariant
    but care about pitch/roll relative to the world (encoded in the
    projected gravity term).
    """
    w = quat[:, 0]; x = quat[:, 1]; y = quat[:, 2]; z = quat[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y),
                     1.0 - 2.0 * (y * y + z * z))
    c = np.cos(-yaw); s = np.sin(-yaw)
    vx = c * vel_world[:, 0] - s * vel_world[:, 1]
    vy = s * vel_world[:, 0] + c * vel_world[:, 1]
    vz = vel_world[:, 2]
    return np.stack([vx, vy, vz], axis=-1).astype(np.float32)


# ─── obs builder ───────────────────────────────────────────────────────


def _to_np(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def compute_common_obs(
    *,
    root_pos: np.ndarray,        # (N, 3)
    root_quat: np.ndarray,       # (N, 4) w,x,y,z
    root_lin_vel: np.ndarray,    # (N, 3) world frame
    root_ang_vel: np.ndarray,    # (N, 3) body frame (Genesis already returns this)
    joint_pos: np.ndarray,       # (N, num_dofs)
    joint_vel: np.ndarray,       # (N, num_dofs)
    last_action: np.ndarray,     # (N, num_dofs)
    step_count: np.ndarray,      # (N,) int — control steps since reset
    default_joint_pos: np.ndarray,  # (num_dofs,) reference pose for jpos centering
    control_dt: float = 0.02,
    gait_freq_hz: float = 1.5,   # default conditioning frequency for clock
) -> np.ndarray:
    """Build the (N, SKILL_BASE_OBS_DIM=78) common obs.

    All inputs are batched numpy arrays. Caller is responsible for
    converting Genesis tensors via the entity getters.
    """
    n = root_pos.shape[0]

    height = root_pos[:, 2:3].astype(np.float32)                # (N, 1)
    proj_g = projected_gravity(root_quat)                       # (N, 3)
    v_body = body_frame_velocity(root_quat, root_lin_vel)       # (N, 3)
    w_body = root_ang_vel.astype(np.float32)                    # (N, 3)
    jpos = (joint_pos - default_joint_pos[None, :]).astype(np.float32)
    jvel = joint_vel.astype(np.float32)
    last_act = last_action.astype(np.float32)

    t = (step_count.astype(np.float32) * float(control_dt))
    phase = 2.0 * np.pi * float(gait_freq_hz) * t
    clock = np.stack([np.sin(phase), np.cos(phase)], axis=-1
                     ).astype(np.float32)                       # (N, 2)

    out = np.concatenate(
        [height, proj_g, v_body, w_body, jpos, jvel, last_act, clock],
        axis=1,
    )
    assert out.shape == (n, SKILL_BASE_OBS_DIM), (
        f"common obs shape {out.shape} != ({n}, {SKILL_BASE_OBS_DIM})"
    )
    return out


def read_robot_state(robot, joint_indices):
    """Pull (root_pos, root_quat, root_lin_vel, root_ang_vel, jpos, jvel)
    from a Genesis robot entity into numpy. Convenience for skill envs."""
    return (
        _to_np(robot.get_pos()),
        _to_np(robot.get_quat()),
        _to_np(robot.get_vel()),
        _to_np(robot.get_ang()),
        _to_np(robot.get_dofs_position(joint_indices)),
        _to_np(robot.get_dofs_velocity(joint_indices)),
    )
