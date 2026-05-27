"""Standup skill env — recover from a fallen pose to upright standing.

Subclasses `SkillEnv` with:
  * No command vec (empty CommandSpec).
  * `_reset_robot_pose`: pick one of four fallen templates per env and
    apply random orientation + joint jitter so the policy generalizes
    across a continuous distribution of fallen starts.
  * Reward: speed (dense time penalty + time-scaled terminal bonus) +
    several stability terms (sway, jerk, drift, quiet, gravity, smooth).
  * Termination: SUSTAINED success — N consecutive 'looks standing'
    frames before episode ends. Forces the policy to land in a stable
    pose, not just touch it. Also terminates on timeout.

Pose templates live in `envs/standup.py`; this file only adds the
per-reset jitter on top.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from envs.standup import all_poses
from skills.base import CommandSpec, SkillEnv
from skills.common_obs import _to_np
from skills.standup.config import StandupConfig
from skills.standup.rewards import compute_standup_reward


# ─── quaternion helpers (local — only used by the pose jitter) ────────


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product. Both (N, 4) in (w, x, y, z). Returns (N, 4)."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1).astype(np.float32)


def _quat_from_axis_angle_batched(axis: np.ndarray,
                                   angle: np.ndarray) -> np.ndarray:
    """Build (N, 4) quats from (N, 3) axes + (N,) angles."""
    n = np.linalg.norm(axis, axis=1, keepdims=True).clip(1e-8)
    axis = axis / n
    half = angle * 0.5
    s = np.sin(half)
    w = np.cos(half)
    return np.stack([w, axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s],
                    axis=-1).astype(np.float32)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
    return (q / n).astype(np.float32)


class K1StandupEnv(SkillEnv):

    SKILL_NAME = "standup"
    SKILL_OBS_ADDONS = 0
    # Standup STARTS below this threshold — don't terminate on it.
    FALL_TERMINATE_Z = -1.0  # disable height termination

    def __init__(self, cfg: StandupConfig = None, **kwargs):
        self.cfg = cfg or StandupConfig()
        kwargs.setdefault("num_envs", self.cfg.num_envs)
        kwargs.setdefault("dt", self.cfg.dt)
        kwargs.setdefault("sim_dt", self.cfg.sim_dt)
        kwargs.setdefault("gait_freq_hz", self.cfg.gait_freq_hz)
        super().__init__(**kwargs)
        self.MAX_EPISODE_STEPS = self.cfg.max_episode_steps

        all_pose_objs = {p.name: p for p in all_poses()}
        self._poses = [all_pose_objs[n] for n in self.cfg.poses
                       if n in all_pose_objs]
        if not self._poses:
            raise ValueError(f"no standup poses matched cfg.poses="
                             f"{self.cfg.poses}; known: {list(all_pose_objs)}")

        # Pre-build per-template joint-target arrays in robot_cfg order so
        # the reset path is fully vectorised.
        joint_order = list(self.robot_cfg.joint_names)
        self._template_joint_targets = np.zeros(
            (len(self._poses), self.act_dim), dtype=np.float32)
        self._template_quats = np.zeros((len(self._poses), 4),
                                         dtype=np.float32)
        self._template_heights = np.zeros(len(self._poses), dtype=np.float32)
        for i, p in enumerate(self._poses):
            arr = np.array(self._default_action, dtype=np.float32).copy()
            for name, val in p.joint_targets.items():
                if name in joint_order:
                    arr[joint_order.index(name)] = float(val)
            self._template_joint_targets[i] = arr
            self._template_quats[i] = np.asarray(p.trunk_quat,
                                                  dtype=np.float32)
            self._template_heights[i] = float(p.trunk_height)

        # Per-env reward / termination state.
        self._frame_success = np.zeros(self.num_envs, dtype=bool)
        self._success_streak = np.zeros(self.num_envs, dtype=np.int32)
        self._sustained_now = np.zeros(self.num_envs, dtype=bool)
        self._prev_prev_action = np.tile(self._default_action,
                                          (self.num_envs, 1))

        self.cfg.obs_dim = self.obs_dim
        self.cfg.act_dim = self.act_dim

    # ── command spec (none for standup) ───────────────────────────

    def _make_command_spec(self) -> CommandSpec:
        return CommandSpec.empty()

    # ── no extra scene entities ───────────────────────────────────

    def _add_scene_extras(self, scene) -> None:
        return

    # ── fallen-pose reset with jitter ─────────────────────────────

    def _sample_initial_poses(self, n: int
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (pos[n,3], quat[n,4], joint_targets[n,act_dim]).

        Each env: pick a template uniformly, then compose a random small
        rotation + random yaw with the template quat, jitter joint
        targets, and jitter trunk height slightly.
        """
        c = self.cfg
        pose_choices = self.rng.integers(0, len(self._poses), size=n)

        # Base from templates
        base_quat = self._template_quats[pose_choices]              # (n, 4)
        base_targets = self._template_joint_targets[pose_choices]   # (n, A)
        base_height = self._template_heights[pose_choices]          # (n,)

        # Random small rotation (axis-angle) — keeps fall pose "near"
        # the template but covers a continuous distribution.
        axes = self.rng.standard_normal((n, 3)).astype(np.float32)
        angles = self.rng.uniform(-c.orient_jitter_rad,
                                   c.orient_jitter_rad,
                                   size=n).astype(np.float32)
        q_jitter = _quat_from_axis_angle_batched(axes, angles)

        # Random yaw on top — robot has no preferred world heading.
        yaw_axis = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                           (n, 1))
        yaws = self.rng.uniform(-c.yaw_jitter_rad,
                                 c.yaw_jitter_rad,
                                 size=n).astype(np.float32)
        q_yaw = _quat_from_axis_angle_batched(yaw_axis, yaws)

        # Compose: world ← yaw ← jitter ← template.
        quat = _quat_normalize(_quat_mul(q_yaw,
                                          _quat_mul(q_jitter, base_quat)))

        # Joint target jitter — per-joint Gaussian, no clipping (joint
        # limits enforced by the URDF + PD controller).
        joint_noise = (self.rng.standard_normal((n, self.act_dim))
                       .astype(np.float32)) * c.joint_jitter_rad
        targets = base_targets + joint_noise

        # Height jitter
        h_noise = self.rng.uniform(-c.height_jitter_m, c.height_jitter_m,
                                    size=n).astype(np.float32)
        pos = np.zeros((n, 3), dtype=np.float32)
        pos[:, 2] = np.clip(base_height + h_noise, 0.05, None)

        return pos, quat, targets

    def _reset_robot_pose(self, envs_idx: np.ndarray) -> None:
        n = envs_idx.shape[0]
        pos, quat, targets = self._sample_initial_poses(n)
        try:
            self.robot.set_pos(pos, envs_idx=envs_idx)
            self.robot.set_quat(quat, envs_idx=envs_idx)
            self.robot.set_dofs_position(targets, self.dof_indices,
                                         envs_idx=envs_idx,
                                         zero_velocity=True)
        except Exception as e:
            print(f"[standup] _reset_robot_pose failed: {e}")

    # ── reward + sustained-success bookkeeping ────────────────────

    def _compute_skill_reward(self, action: np.ndarray):
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            root_lin_vel = _to_np(self.robot.get_vel())
            root_ang_vel = _to_np(self.robot.get_ang())
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32), {}

        # Advance streak from prior state. `_frame_success` from the
        # last reward computation; updated below before next step.
        prev_streak = self._success_streak.copy()
        # We pass the BEFORE-update streak so the reward function can
        # decide whether persistence applies (streak in [1, hold-1]) or
        # the terminal bonus fires (streak completes this step).
        # First compute new frame_success, then derive streak/sustained.

        # We need the frame mask for *this* step before composing reward,
        # since the streak/sustained signals feed into the reward. The
        # cleanest split is: compute frame mask inline, update streak,
        # then hand all of it to compute_standup_reward.
        from skills.standup.rewards import success_frame_mask
        frame_now = success_frame_mask(
            root_quat, root_pos[:, 2],
            target_h=self.cfg.target_height,
            upright_threshold=self.cfg.upright_threshold,
        )

        # Streak: increment if frame_now, reset to 0 otherwise.
        new_streak = np.where(frame_now, prev_streak + 1, 0).astype(np.int32)
        # Sustained-now: the streak completes (crosses the threshold)
        # *this* step. Only fires once per episode.
        sustained_now = (new_streak == self.cfg.success_hold_steps) \
                        & (prev_streak < self.cfg.success_hold_steps)

        reward, _frame_success, components = compute_standup_reward(
            root_pos=root_pos, root_quat=root_quat,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            joint_vel=jvel,
            action=action,
            prev_action=self._last_action,
            prev_prev_action=self._prev_prev_action,
            success_streak=new_streak,
            sustained_now=sustained_now,
            step_count=self.step_count,
            weights=self.cfg.rewards,
            target_height=self.cfg.target_height,
            upright_threshold=self.cfg.upright_threshold,
            hold_steps=self.cfg.success_hold_steps,
            time_to_stand_tau_steps=self.cfg.time_to_stand_tau_steps,
            control_dt=self.dt,
        )

        # Commit state for next step / termination check.
        self._success_streak = new_streak
        self._sustained_now = sustained_now
        self._frame_success = frame_now
        self._prev_prev_action = self._last_action.copy()
        return reward, components

    # ── termination: sustained success OR timeout ─────────────────

    def _check_skill_done(self) -> np.ndarray:
        timeout = self.step_count >= self.MAX_EPISODE_STEPS
        return (timeout | self._sustained_now)

    # ── reset hook clears all per-env reward state ────────────────

    def _reset_skill_state(self, envs_idx: np.ndarray) -> None:
        self._frame_success[envs_idx] = False
        self._success_streak[envs_idx] = 0
        self._sustained_now[envs_idx] = False
        self._prev_prev_action[envs_idx] = self._default_action
