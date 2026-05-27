"""Standup skill env — recover from a physically-realistic fallen pose
to upright standing.

Subclasses `SkillEnv` with:
  * No command vec (empty CommandSpec).
  * `_reset_robot_pose`: lazily builds a SETTLE POOL on first reset by
    spawning all envs in the air with random orientations, stepping
    physics until they settle into the ground, and snapshotting the
    resulting per-env states. The pool is filtered to keep only
    actually-fallen poses (so trivial "landed upright" states never
    enter the training distribution). Every subsequent reset samples
    uniformly from this pool — no scene.step needed mid-rollout, which
    would desynchronise the other envs.
  * Reward: speed (dense time penalty + time-scaled terminal bonus) +
    stability cluster (sway, jerk, drift, quiet, gravity, smooth).
  * Termination: SUSTAINED success — N consecutive 'looks standing'
    frames before episode ends, OR timeout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from skills.base import CommandSpec, SkillEnv
from skills.common_obs import _to_np
from skills.standup.config import StandupConfig
from skills.standup.rewards import (compute_standup_reward, success_frame_mask,
                                     upright_signal)


# ─── quaternion helper ────────────────────────────────────────────────


def _random_unit_quat(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples on SO(3). Shoemake's method:
    u1, u2, u3 ∈ U[0,1] → unit quaternion (w, x, y, z)."""
    u = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
    s1 = np.sqrt(1.0 - u[:, 0])
    s0 = np.sqrt(u[:, 0])
    return np.stack([
        s1 * np.sin(2.0 * np.pi * u[:, 1]),
        s1 * np.cos(2.0 * np.pi * u[:, 1]),
        s0 * np.sin(2.0 * np.pi * u[:, 2]),
        s0 * np.cos(2.0 * np.pi * u[:, 2]),
    ], axis=-1).astype(np.float32)


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

        # Settle pool — built lazily on first reset (needs scene + robot).
        self._pool_pos: Optional[np.ndarray] = None
        self._pool_quat: Optional[np.ndarray] = None
        self._pool_jpos: Optional[np.ndarray] = None
        self._pool_size: int = 0

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

    # ── settle pool ───────────────────────────────────────────────

    def _build_settle_pool(self) -> None:
        """Spawn all envs in the air with random orientations, settle by
        gravity, snapshot. Repeat for `settle_pool_rounds` iterations →
        pool of (rounds × num_envs) physically-realistic fallen states.

        Filters out states the robot accidentally settled into that look
        like a successful standup already (uprightness high enough that
        the policy would get the success bonus for free). The pool is
        rebuilt from scratch if filtering drops below `num_envs` valid
        states — we'd rather pay a few extra seconds than feed the
        trainer a tiny pool.
        """
        c = self.cfg
        N = self.num_envs
        all_idx = np.arange(N)

        pool_pos, pool_quat, pool_jpos = [], [], []

        for round_idx in range(c.settle_pool_rounds):
            # Spawn high with random orientation + small joint noise.
            pos = np.zeros((N, 3), dtype=np.float32)
            pos[:, 2] = self.rng.uniform(c.spawn_height_min,
                                          c.spawn_height_max,
                                          size=N).astype(np.float32)
            quat = _random_unit_quat(N, self.rng)
            jpos_target = (self._default_action[None, :]
                           + self.rng.standard_normal((N, self.act_dim))
                              .astype(np.float32) * 0.2)

            try:
                self.robot.set_pos(pos, envs_idx=all_idx)
                self.robot.set_quat(quat, envs_idx=all_idx)
                self.robot.set_dofs_position(jpos_target, self.dof_indices,
                                              envs_idx=all_idx,
                                              zero_velocity=True)
                # Hold PD at the spawn pose during the fall — limbs land
                # roughly relaxed instead of flopping (closer to real falls
                # where the robot doesn't actively try to control mid-air).
                self.robot.control_dofs_position(jpos_target,
                                                  self.dof_indices)
            except Exception as e:
                print(f"[standup] settle spawn (round {round_idx}) "
                      f"failed: {e}")
                continue

            # Step physics until the robot has settled. We don't drive
            # the action_repeat outer loop here — this is raw physics
            # time, not control time.
            for _ in range(c.settle_steps):
                self.scene.step()

            # Snapshot.
            try:
                p = _to_np(self.robot.get_pos()).copy()
                q = _to_np(self.robot.get_quat()).copy()
                j = _to_np(self.robot.get_dofs_position(self.dof_indices)
                           ).copy()
            except Exception as e:
                print(f"[standup] settle snapshot failed: {e}")
                continue

            # Filter for "actually fallen": low trunk height AND not
            # already upright. Keep envs where BOTH conditions hold.
            up = upright_signal(q)
            ok = (p[:, 2] < c.pool_max_height) & (up < c.pool_max_upright)
            if ok.any():
                pool_pos.append(p[ok])
                pool_quat.append(q[ok])
                pool_jpos.append(j[ok])

        if not pool_pos:
            raise RuntimeError(
                "[standup] settle pool is empty after filtering — "
                "loosen pool_max_upright / pool_max_height or increase "
                "settle_pool_rounds.")

        self._pool_pos = np.concatenate(pool_pos, axis=0).astype(np.float32)
        self._pool_quat = np.concatenate(pool_quat, axis=0).astype(np.float32)
        self._pool_jpos = np.concatenate(pool_jpos, axis=0).astype(np.float32)
        # Re-center horizontally — pool snapshots have arbitrary xy drift
        # from the fall, but the env should always reset at the origin.
        self._pool_pos[:, 0:2] = 0.0
        self._pool_size = self._pool_pos.shape[0]
        print(f"[standup] settle pool built: {self._pool_size} states "
              f"(from {c.settle_pool_rounds} rounds × {N} envs, "
              f"{c.settle_steps} sim substeps each)")

    # ── reset using the pool ──────────────────────────────────────

    def _reset_robot_pose(self, envs_idx: np.ndarray) -> None:
        if self._pool_pos is None:
            self._build_settle_pool()

        n = envs_idx.shape[0]
        idx = self.rng.integers(0, self._pool_size, size=n)
        pos = self._pool_pos[idx].copy()
        quat = self._pool_quat[idx].copy()
        jpos = self._pool_jpos[idx].copy()

        # Small Gaussian joint jitter on top of the pool sample — adds
        # continuous variation across (pool_size × ∞) effective starts.
        if self.cfg.joint_jitter_rad > 0:
            jpos = jpos + (self.rng.standard_normal(jpos.shape)
                           .astype(np.float32)
                           * self.cfg.joint_jitter_rad)

        try:
            self.robot.set_pos(pos, envs_idx=envs_idx)
            self.robot.set_quat(quat, envs_idx=envs_idx)
            self.robot.set_dofs_position(jpos, self.dof_indices,
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

        prev_streak = self._success_streak.copy()
        frame_now = success_frame_mask(
            root_quat, root_pos[:, 2],
            target_h=self.cfg.target_height,
            upright_threshold=self.cfg.upright_threshold,
        )
        new_streak = np.where(frame_now, prev_streak + 1, 0).astype(np.int32)
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
