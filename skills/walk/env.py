"""Walk skill env — K1 locomotion with commanded body-frame velocity.

Subclasses `SkillEnv`. The robot spawns upright, a 5-dim command is
sampled (vx, vy, vyaw, foot_clearance, step_freq), and the policy is
rewarded for tracking that command without falling.

No ball, no goal, no opponents — purely locomotion. The ball physics
get added in the dribble skill (step 5).
"""

from __future__ import annotations

import numpy as np

from skills.base import CommandSpec, SkillEnv
from skills.common_obs import _to_np
from skills.walk.config import WalkConfig
from skills.walk.rewards import compute_walk_reward


# Soft contact threshold — foot_z below this is treated as in-contact.
# Real Genesis contact queries would be more accurate; this proxy is
# good enough for shaping rewards and matches typical foot-link mesh
# offsets (~0.02 m above ground for K1).
_CONTACT_Z = 0.04
# Foot links exposed by K1_22dof.urdf — used to read foot poses for the
# gait reward.
_FOOT_LINK_NAMES = ("left_foot_link", "right_foot_link")


class K1WalkEnv(SkillEnv):
    """Phase-1 walk skill. Trains a velocity-tracking locomotion
    controller; no ball, no field-of-play constraints beyond the field
    plane itself.
    """

    SKILL_NAME = "walk"
    SKILL_OBS_ADDONS = 0           # all extras live in the command vec
    FALL_TERMINATE_Z = 0.20

    def __init__(self, cfg: WalkConfig = None, **kwargs):
        self.cfg = cfg or WalkConfig()
        # SkillEnv reads from these kwargs; cfg-driven defaults applied
        # only when the caller didn't override.
        kwargs.setdefault("num_envs", self.cfg.num_envs)
        kwargs.setdefault("dt", self.cfg.dt)
        kwargs.setdefault("sim_dt", self.cfg.sim_dt)
        kwargs.setdefault("gait_freq_hz", self.cfg.gait_freq_hz)
        super().__init__(**kwargs)
        self.MAX_EPISODE_STEPS = self.cfg.max_episode_steps

        # Per-env state we maintain for the reward:
        self._prev_jvel = np.zeros((self.num_envs, self.act_dim),
                                   dtype=np.float32)
        self._foot_links = None
        self._prev_contact = np.zeros((self.num_envs, 2), dtype=bool)

        # Mutate the cfg with the actual obs_dim so the trainer's buffers
        # are sized correctly.
        self.cfg.obs_dim = self.obs_dim
        self.cfg.act_dim = self.act_dim

    # ── command spec ──────────────────────────────────────────────

    def _make_command_spec(self) -> CommandSpec:
        c = self.cfg if hasattr(self, "cfg") and self.cfg is not None else WalkConfig()
        return CommandSpec(
            dim=5,
            low=np.array([c.vx_range[0], c.vy_range[0], c.vyaw_range[0],
                          c.foot_clearance_range[0], c.step_freq_range[0]],
                         dtype=np.float32),
            high=np.array([c.vx_range[1], c.vy_range[1], c.vyaw_range[1],
                           c.foot_clearance_range[1], c.step_freq_range[1]],
                          dtype=np.float32),
            names=("vx", "vy", "vyaw", "foot_clearance", "step_freq"),
        )

    # ── scene extras: none for plain walk ──────────────────────────

    def _add_scene_extras(self, scene) -> None:
        return

    # ── foot link lookup (after scene.build) ───────────────────────

    def _ensure_foot_links(self) -> None:
        if self._foot_links is not None:
            return
        links_by_name = {ln.name: ln for ln in getattr(self.robot, "links", [])}
        found = []
        for name in _FOOT_LINK_NAMES:
            link = links_by_name.get(name)
            if link is not None:
                found.append(link)
        self._foot_links = found if len(found) == 2 else None
        if self._foot_links is None:
            print(f"[walk] foot links {_FOOT_LINK_NAMES} not found in robot; "
                  "foot_clearance reward will be zeroed.")

    def _read_foot_state(self):
        """Returns (feet_z (N,2), contact_mask (N,2) bool, contact_just_now (N,2) bool)."""
        N = self.num_envs
        if self._foot_links is None:
            self._ensure_foot_links()
        if self._foot_links is None:
            feet_z = np.zeros((N, 2), dtype=np.float32)
            contact = np.zeros((N, 2), dtype=bool)
            return feet_z, contact, np.zeros((N, 2), dtype=bool)
        try:
            l = _to_np(self._foot_links[0].get_pos())
            r = _to_np(self._foot_links[1].get_pos())
            feet_z = np.stack([l[:, 2], r[:, 2]], axis=1).astype(np.float32)
        except Exception:
            feet_z = np.zeros((N, 2), dtype=np.float32)
        contact = feet_z < _CONTACT_Z
        contact_just_now = contact & (~self._prev_contact)
        self._prev_contact = contact
        return feet_z, contact, contact_just_now

    # ── reward ─────────────────────────────────────────────────────

    def _compute_skill_reward(self, action: np.ndarray):
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            root_lin_vel = _to_np(self.robot.get_vel())
            root_ang_vel = _to_np(self.robot.get_ang())
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32), {}

        try:
            applied_torque = _to_np(
                self.robot.get_dofs_force(self.dof_indices))
        except Exception:
            applied_torque = np.zeros_like(jvel)

        feet_z, contact, _ = self._read_foot_state()

        reward, components = compute_walk_reward(
            root_pos=root_pos, root_quat=root_quat,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            jvel=jvel, prev_jvel=self._prev_jvel,
            action=action, prev_action=self._last_action,
            applied_torque=applied_torque,
            feet_z=feet_z, contact_mask=contact,
            commands=self.commands,
            weights=self.cfg.rewards,
            dt=self.dt,
        )

        self._prev_jvel = jvel
        return reward, components

    # ── reset hook: also reset the per-env reward state ───────────

    def _reset_skill_state(self, envs_idx: np.ndarray) -> None:
        self._prev_jvel[envs_idx] = 0.0
        self._prev_contact[envs_idx] = False
