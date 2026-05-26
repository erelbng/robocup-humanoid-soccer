"""Dribble skill env — walk + ball control.

Subclasses `SkillEnv` with:
  * 7-dim command vector: walk vec (5) + ball offset (2).
  * Ball entity added in `_add_scene_extras`.
  * Ball spawned in front of the robot on every reset.
  * 6-dim obs add-ons: ball pos (xyz) + ball velocity (xyz), both in
    the robot's body frame.
  * Reward composes walk shaping with ball-tracking terms.
  * Episode terminates on success-less timeout, fall, OR if the ball
    drifts more than `ball_lost_distance` from the robot.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

from skills.base import CommandSpec, SkillEnv
from skills.common_obs import _to_np
from skills.dribble.config import DribbleConfig
from skills.dribble.rewards import (ball_state_body_frame,
                                     compute_dribble_reward)


_FOOT_LINK_NAMES = ("left_foot_link", "right_foot_link")
_CONTACT_Z = 0.04


class K1DribbleEnv(SkillEnv):

    SKILL_NAME = "dribble"
    SKILL_OBS_ADDONS = 6   # ball pos (3) + ball vel (3) in body frame
    FALL_TERMINATE_Z = 0.20

    def __init__(self, cfg: DribbleConfig = None, **kwargs):
        self.cfg = cfg or DribbleConfig()
        kwargs.setdefault("num_envs", self.cfg.num_envs)
        kwargs.setdefault("dt", self.cfg.dt)
        kwargs.setdefault("sim_dt", self.cfg.sim_dt)
        kwargs.setdefault("gait_freq_hz", self.cfg.gait_freq_hz)
        super().__init__(**kwargs)
        self.MAX_EPISODE_STEPS = self.cfg.max_episode_steps

        # Per-env state shared with walk-style rewards
        self._prev_jvel = np.zeros((self.num_envs, self.act_dim),
                                   dtype=np.float32)
        self._foot_links = None
        self._prev_contact = np.zeros((self.num_envs, 2), dtype=bool)
        self.ball = None
        self._ball_lost_now = np.zeros(self.num_envs, dtype=bool)

        self.cfg.obs_dim = self.obs_dim
        self.cfg.act_dim = self.act_dim

    # ── command spec ──────────────────────────────────────────────

    def _make_command_spec(self) -> CommandSpec:
        c = self.cfg if hasattr(self, "cfg") and self.cfg is not None else DribbleConfig()
        return CommandSpec(
            dim=7,
            low=np.array([c.vx_range[0], c.vy_range[0], c.vyaw_range[0],
                          c.foot_clearance_range[0], c.step_freq_range[0],
                          c.ball_off_x_range[0], c.ball_off_y_range[0]],
                         dtype=np.float32),
            high=np.array([c.vx_range[1], c.vy_range[1], c.vyaw_range[1],
                           c.foot_clearance_range[1], c.step_freq_range[1],
                           c.ball_off_x_range[1], c.ball_off_y_range[1]],
                          dtype=np.float32),
            names=("vx", "vy", "vyaw", "foot_clearance", "step_freq",
                   "ball_off_x", "ball_off_y"),
        )

    def _make_head_command_spec(self) -> CommandSpec:
        c = self.cfg if hasattr(self, "cfg") and self.cfg is not None else DribbleConfig()
        return CommandSpec(
            dim=2,
            low=np.array([c.head_yaw_range[0], c.head_pitch_range[0]],
                         dtype=np.float32),
            high=np.array([c.head_yaw_range[1], c.head_pitch_range[1]],
                          dtype=np.float32),
            names=("head_yaw", "head_pitch"),
        )

    # ── scene extras: add the ball ────────────────────────────────

    def _add_scene_extras(self, scene) -> None:
        if gs is None:
            return
        # Sphere collider. Genesis Sphere supports `collision=True`;
        # mass comes from material density × volume by default. To get
        # closer to our 0.4kg ball-mass spec we'd need a custom material;
        # in practice the default is light enough that K1 can push it.
        self.ball = scene.add_entity(
            gs.morphs.Sphere(radius=self.cfg.ball_radius,
                             pos=(0.5, 0.0, self.cfg.ball_radius),
                             collision=True),
        )

    # ── reset: place the ball ─────────────────────────────────────

    def _reset_skill_state(self, envs_idx: np.ndarray) -> None:
        self._prev_jvel[envs_idx] = 0.0
        self._prev_contact[envs_idx] = False
        self._ball_lost_now[envs_idx] = False
        if self.ball is None:
            return
        n = envs_idx.shape[0]
        bpos = np.zeros((n, 3), dtype=np.float32)
        bpos[:, 0] = self.rng.uniform(*self.cfg.ball_spawn_range_x, size=n)
        bpos[:, 1] = self.rng.uniform(*self.cfg.ball_spawn_range_y, size=n)
        bpos[:, 2] = self.cfg.ball_radius
        try:
            self.ball.set_pos(bpos, envs_idx=envs_idx)
            # Zero ball velocity by setting it twice (some Genesis builds
            # require an explicit velocity reset).
            try:
                self.ball.zero_all_dofs_velocity(envs_idx=envs_idx)
            except Exception:
                pass
        except Exception as e:
            print(f"[dribble] ball reset failed: {e}")

    # ── foot state (same pattern as walk) ─────────────────────────

    def _ensure_foot_links(self) -> None:
        if self._foot_links is not None:
            return
        links_by_name = {ln.name: ln for ln in getattr(self.robot, "links", [])}
        found = [links_by_name.get(n) for n in _FOOT_LINK_NAMES]
        if all(found):
            self._foot_links = found
        else:
            self._foot_links = None
            print(f"[dribble] foot links not found; foot_clearance zeroed.")

    def _read_foot_state(self):
        N = self.num_envs
        if self._foot_links is None:
            self._ensure_foot_links()
        if self._foot_links is None:
            return (np.zeros((N, 2), dtype=np.float32),
                    np.zeros((N, 2), dtype=bool))
        try:
            l = _to_np(self._foot_links[0].get_pos())
            r = _to_np(self._foot_links[1].get_pos())
            feet_z = np.stack([l[:, 2], r[:, 2]], axis=1).astype(np.float32)
        except Exception:
            feet_z = np.zeros((N, 2), dtype=np.float32)
        contact = feet_z < _CONTACT_Z
        return feet_z, contact

    # ── obs add-ons: ball in body frame ───────────────────────────

    def _skill_obs_addons(self) -> np.ndarray:
        N = self.num_envs
        if self.ball is None:
            return np.zeros((N, self.SKILL_OBS_ADDONS), dtype=np.float32)
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            bpos = _to_np(self.ball.get_pos())
            bvel = _to_np(self.ball.get_vel())
        except Exception:
            return np.zeros((N, self.SKILL_OBS_ADDONS), dtype=np.float32)
        pos_body, vel_body = ball_state_body_frame(
            root_pos, root_quat, bpos, bvel)
        return np.concatenate([pos_body, vel_body], axis=1).astype(np.float32)

    # ── reward ────────────────────────────────────────────────────

    def _compute_skill_reward(self, action: np.ndarray):
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            root_lin_vel = _to_np(self.robot.get_vel())
            root_ang_vel = _to_np(self.robot.get_ang())
            jpos = _to_np(self.robot.get_dofs_position(self.dof_indices))
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
            bpos = _to_np(self.ball.get_pos())
            bvel = _to_np(self.ball.get_vel())
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32), {}

        try:
            applied_torque = _to_np(
                self.robot.get_dofs_force(self.dof_indices))
        except Exception:
            applied_torque = np.zeros_like(jvel)

        feet_z, contact = self._read_foot_state()

        reward, lost, components = compute_dribble_reward(
            root_pos=root_pos, root_quat=root_quat,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            jpos=jpos, jvel=jvel, prev_jvel=self._prev_jvel,
            action=action, prev_action=self._last_action,
            applied_torque=applied_torque,
            feet_z=feet_z, contact_mask=contact,
            ball_pos=bpos, ball_vel=bvel,
            commands=self.commands,
            weights=self.cfg.rewards,
            head_commands=self.head_commands,
            head_joint_indices=self.robot_cfg.head_joint_indices,
            arm_joint_indices=self.robot_cfg.arm_joint_indices,
            default_joint_pos=self._default_action,
            ball_lost_distance=self.cfg.ball_lost_distance,
            dt=self.dt,
        )

        self._ball_lost_now = lost
        self._prev_jvel = jvel
        return reward, components

    # ── termination: walk-style + ball-lost ───────────────────────

    def _check_skill_done(self) -> np.ndarray:
        base_done = super()._check_skill_done()  # timeout + fall
        return base_done | self._ball_lost_now
