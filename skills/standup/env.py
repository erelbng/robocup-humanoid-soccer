"""Standup skill env — recover from a fallen pose to upright standing.

Subclasses `SkillEnv` with:
  * No command vec (empty CommandSpec).
  * Custom `_reset_robot_pose` that spawns each env in one of four
    "fallen" poses (supine, prone, side-left, side-right) — same set
    used by the legacy `envs/standup.py`.
  * Standup reward: upright + height + smoothness + success bonus.
  * Episode terminates on success (caught upright + at height) OR
    timeout. Does NOT terminate on low trunk height — the whole point
    is starting low and getting up.

The fallen poses are sampled fresh on every reset, so the policy must
generalize across all four starting configurations.
"""

from __future__ import annotations

import numpy as np

from envs.standup import all_poses          # reuse pose definitions
from skills.base import CommandSpec, SkillEnv
from skills.common_obs import _to_np
from skills.standup.config import StandupConfig
from skills.standup.rewards import compute_standup_reward


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

        # Pose definitions (joint targets, trunk quat, trunk z).
        all_pose_objs = {p.name: p for p in all_poses()}
        self._poses = [all_pose_objs[n] for n in self.cfg.poses
                       if n in all_pose_objs]
        if not self._poses:
            raise ValueError(f"no standup poses matched cfg.poses="
                             f"{self.cfg.poses}; known: {list(all_pose_objs)}")

        # Joint-name → target array (in robot_cfg joint order).
        # Built lazily after dof_indices is known.
        self._pose_joint_targets_cache: dict = {}

        # Per-env: tracks success this step so we can terminate on it.
        self._success_now = np.zeros(self.num_envs, dtype=bool)

        # Pre-build joint-target arrays in robot_cfg order so reset is fast.
        joint_order = list(self.robot_cfg.joint_names)
        for p in self._poses:
            arr = np.array(self._default_action, dtype=np.float32).copy()
            for name, val in p.joint_targets.items():
                if name in joint_order:
                    arr[joint_order.index(name)] = float(val)
            self._pose_joint_targets_cache[p.name] = arr

        self.cfg.obs_dim = self.obs_dim
        self.cfg.act_dim = self.act_dim

    # ── command spec (none for standup) ───────────────────────────

    def _make_command_spec(self) -> CommandSpec:
        return CommandSpec.empty()

    # ── no extra scene entities ───────────────────────────────────

    def _add_scene_extras(self, scene) -> None:
        return

    # ── fallen-pose reset ─────────────────────────────────────────

    def _reset_robot_pose(self, envs_idx: np.ndarray) -> None:
        n = envs_idx.shape[0]
        # Sample one pose per env (uniformly among the configured set).
        pose_choices = self.rng.integers(0, len(self._poses), size=n)
        pos = np.zeros((n, 3), dtype=np.float32)
        quat = np.zeros((n, 4), dtype=np.float32)
        targets = np.zeros((n, self.act_dim), dtype=np.float32)
        for i, pi in enumerate(pose_choices):
            p = self._poses[pi]
            pos[i, 2] = p.trunk_height
            quat[i] = np.asarray(p.trunk_quat, dtype=np.float32)
            targets[i] = self._pose_joint_targets_cache[p.name]
        try:
            self.robot.set_pos(pos, envs_idx=envs_idx)
            self.robot.set_quat(quat, envs_idx=envs_idx)
            self.robot.set_dofs_position(targets, self.dof_indices,
                                         envs_idx=envs_idx,
                                         zero_velocity=True)
        except Exception as e:
            print(f"[standup] _reset_robot_pose failed: {e}")

    # ── reward ────────────────────────────────────────────────────

    def _compute_skill_reward(self, action: np.ndarray):
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32), {}

        reward, success, components = compute_standup_reward(
            root_pos=root_pos, root_quat=root_quat,
            joint_vel=jvel,
            action=action, prev_action=self._last_action,
            weights=self.cfg.rewards,
            target_height=self.cfg.target_height,
            upright_threshold=self.cfg.upright_threshold,
        )
        self._success_now = success
        return reward, components

    # ── termination: success OR timeout ───────────────────────────

    def _check_skill_done(self) -> np.ndarray:
        timeout = self.step_count >= self.MAX_EPISODE_STEPS
        return (timeout | self._success_now)

    # ── reset hook clears the success flag ────────────────────────

    def _reset_skill_state(self, envs_idx: np.ndarray) -> None:
        self._success_now[envs_idx] = False
