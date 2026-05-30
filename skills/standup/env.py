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
  * Reward: speed (dense time penalty + time-scaled terminal bonus on
    streak completion) + post-success standing reward (paid for every
    later frame the robot stays upright) + stability cluster (sway,
    jerk, drift, quiet, smooth) — all gated near upright.
  * Termination: timeout ONLY. Episodes run the full MAX_EPISODE_STEPS
    so the policy must prove sustained stability AFTER the success
    bonus is paid; collapsing right after success forfeits the entire
    post-success standing reward.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from skills.base import CommandSpec, SkillEnv
from skills.common_obs import _to_np
from skills.standup.config import StandupConfig, discovery_weights
from skills.standup.rewards import (compute_standup_reward, success_frame_mask,
                                     upright_signal)


# Contact-link names (K1_22dof.urdf). Feet + hands cover the four main
# standup support modes: push-from-hands, kneel-into-stand, sit-up-and-
# rise, balance-on-feet. Without these in obs the policy can't reason
# about its support polygon during the transition.
_FOOT_LINK_NAMES = ("left_foot_link", "right_foot_link")
_HAND_LINK_NAMES = ("left_hand_link", "right_hand_link")
# Soft contact threshold — link z below this is treated as in-contact.
# 0.05 m matches typical K1 mesh offsets (feet ~0.02, hands a bit higher).
_CONTACT_Z = 0.05


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


def _small_tilt_quat(n: int, max_angle: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Quat for a small rotation (≤ max_angle radians) around a random
    horizontal axis — used to add a small initial tilt to a standing
    robot without completely re-orienting it."""
    angles = rng.uniform(0.0, max_angle, size=n).astype(np.float32)
    # Axis chosen in the xy plane (no yaw component) → pitch/roll only.
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
    ax = np.cos(theta)
    ay = np.sin(theta)
    half = angles * 0.5
    sin_h = np.sin(half)
    cos_h = np.cos(half)
    return np.stack([cos_h, ax * sin_h, ay * sin_h,
                     np.zeros_like(cos_h)], axis=-1).astype(np.float32)


class K1StandupEnv(SkillEnv):

    SKILL_NAME = "standup"
    # Addon dims are decided per-instance from `cfg.proprio_only` below:
    # 8 when contact obs is enabled (fast sim training), 0 when stripped
    # for sim2real-deployable training.
    SKILL_OBS_ADDONS = 8
    # Standup STARTS below this threshold — don't terminate on it.
    FALL_TERMINATE_Z = -1.0  # disable height termination

    def __init__(self, cfg: StandupConfig = None, **kwargs):
        self.cfg = cfg or StandupConfig()
        # Shadow the class attr at instance scope. 8 addon dims when
        # contact obs is enabled: [lfoot_z, rfoot_z, lhand_z, rhand_z,
        # lfoot_contact, rfoot_contact, lhand_contact, rhand_contact].
        # 0 when `proprio_only` — policy sees only what the real robot
        # can measure.
        self.SKILL_OBS_ADDONS = 0 if self.cfg.proprio_only else 8
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

        # Easy pool — near-standing starts for the reverse curriculum.
        self._easy_pool_pos: Optional[np.ndarray] = None
        self._easy_pool_quat: Optional[np.ndarray] = None
        self._easy_pool_jpos: Optional[np.ndarray] = None
        self._easy_pool_size: int = 0

        # Per-env reward / termination state.
        self._frame_success = np.zeros(self.num_envs, dtype=bool)
        self._success_streak = np.zeros(self.num_envs, dtype=np.int32)
        self._sustained_now = np.zeros(self.num_envs, dtype=bool)
        # Latches True once sustained_now fires; stays True until the
        # episode resets. Gates the post-success standing reward so the
        # robot only earns it AFTER it has proven it can hit the hold
        # window — preventing accidental "stood for a few frames then
        # collapsed" trajectories from collecting the same reward.
        self._achieved_sustained = np.zeros(self.num_envs, dtype=bool)
        self._prev_prev_action = np.zeros((self.num_envs, self.act_dim),
                                          dtype=np.float32)
        # Cumulative env-steps seen by the policy (sum over all parallel
        # envs of every `step()` call). Drives the hold_steps curriculum.
        self._total_env_steps_seen: int = 0
        # EMA of frame_success_rate — used to gate curriculum advancement.
        # The easy-start curriculum only advances when the policy shows
        # real performance, preventing time-only advancement while stuck.
        self._success_rate_ema: float = 0.0
        self._success_ema_alpha: float = 0.005  # ~200-step window
        # Upright signal from the previous step — used by the progress
        # reward term. Initialised to -1 (fully inverted) so the first
        # step from any fallen pose produces a positive Δup credit.
        self._prev_upright = -np.ones(self.num_envs, dtype=np.float32)

        # Mean assist force applied last step (N) — for logging.
        self._last_assist_force_mean: float = 0.0

        # Contact-link cache — populated lazily after scene.build().
        self._foot_links = None
        self._hand_links = None

        # Reward weights for the active stage. "discovery" zeroes the motion
        # regularizers so the policy can find ANY standup; "deploy" uses the
        # full set for a smooth, deployable motion.
        if self.cfg.reward_stage == "discovery":
            self._reward_weights = discovery_weights(self.cfg.rewards)
            print("[standup] reward_stage=discovery — motion regularizers "
                  "zeroed (upright/height/progress/feet/speed only)")
        else:
            self._reward_weights = self.cfg.rewards
            print("[standup] reward_stage=deploy — full reward weight set")

        if self.cfg.assist_force_enabled:
            print(f"[standup] assist-force curriculum ON: up to "
                  f"{self.cfg.assist_force_max:.0f} N, decaying over "
                  f"{self.cfg.assist_curriculum_env_steps:,} env-steps")

        self.cfg.obs_dim = self.obs_dim
        self.cfg.act_dim = self.act_dim

    # ── command spec (none for standup) ───────────────────────────

    def _make_command_spec(self) -> CommandSpec:
        return CommandSpec.empty()

    # ── deployability accounting (used by distillation) ──────────

    @property
    def non_deployable_dim(self) -> int:
        """Trailing obs dims to strip when building a sim2real student.

        Standup adds 8 contact dims (foot/hand z + contact bool) that
        require absolute floor position — privileged in the sim2real
        sense. They sit AFTER the base proprio and BEFORE the optional
        DR-privileged tail, but standup has no command/head_cmd so the
        two non-deployable blocks are contiguous at the tail of the obs
        and a single trailing-slice removes both."""
        return self.SKILL_OBS_ADDONS + self.privileged_dim

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

    def _build_easy_pool(self) -> None:
        """Build pool of near-standing initial poses for the reverse
        curriculum. Spawn at standing default pose with small joint
        jitter and a small initial tilt, brief settle so physics relaxes,
        snapshot. The policy uses these starts early in training to
        learn 'what standing looks like' before recovering from harder
        fallen poses."""
        c = self.cfg
        N = self.num_envs
        all_idx = np.arange(N)

        pos = np.zeros((N, 3), dtype=np.float32)
        pos[:, 2] = c.easy_pool_height
        quat = _small_tilt_quat(N, c.easy_pool_tilt_max, self.rng)
        # If tilt_max is zero, the helper returns identity quats — but
        # _small_tilt_quat with max_angle=0 gives [1,0,0,0] for all,
        # which is what we want.
        jpos_target = (self._default_action[None, :]
                       + self.rng.standard_normal((N, self.act_dim))
                          .astype(np.float32) * c.easy_pool_joint_jitter)

        try:
            self.robot.set_pos(pos, envs_idx=all_idx)
            self.robot.set_quat(quat, envs_idx=all_idx)
            self.robot.set_dofs_position(jpos_target, self.dof_indices,
                                          envs_idx=all_idx,
                                          zero_velocity=True)
            # Hold PD at the spawn pose during settle so the robot
            # actively maintains standing posture (instead of flopping).
            self.robot.control_dofs_position(jpos_target, self.dof_indices)
        except Exception as e:
            print(f"[standup] easy pool spawn failed: {e}")
            self._easy_pool_size = 0
            return

        for _ in range(c.easy_pool_settle_steps):
            self.scene.step()

        try:
            p = _to_np(self.robot.get_pos()).copy()
            q = _to_np(self.robot.get_quat()).copy()
            j = _to_np(self.robot.get_dofs_position(self.dof_indices)).copy()
        except Exception as e:
            print(f"[standup] easy pool snapshot failed: {e}")
            self._easy_pool_size = 0
            return

        # Filter: keep only states that are still standing-ish (robot
        # didn't fall during the brief settle).
        up = upright_signal(q)
        ok = (p[:, 2] > c.easy_pool_min_height) & (up > c.easy_pool_min_upright)
        # Re-center horizontally so spawn is always at the origin.
        p[:, 0:2] = 0.0
        self._easy_pool_pos = p[ok].astype(np.float32)
        self._easy_pool_quat = q[ok].astype(np.float32)
        self._easy_pool_jpos = j[ok].astype(np.float32)
        self._easy_pool_size = int(ok.sum())
        print(f"[standup] easy pool built: {self._easy_pool_size} states "
              f"(of {N} attempts) — used for reverse curriculum")

    # ── contact-link lookup (after scene.build) ───────────────────

    def _ensure_contact_links(self) -> None:
        if self._foot_links is not None and self._hand_links is not None:
            return
        links_by_name = {ln.name: ln
                         for ln in getattr(self.robot, "links", [])}
        foot = [links_by_name.get(n) for n in _FOOT_LINK_NAMES]
        hand = [links_by_name.get(n) for n in _HAND_LINK_NAMES]
        self._foot_links = foot if all(l is not None for l in foot) else None
        self._hand_links = hand if all(l is not None for l in hand) else None
        if self._foot_links is None:
            print(f"[standup] foot links {_FOOT_LINK_NAMES} not found — "
                  "contact obs will be zeroed.")
        if self._hand_links is None:
            print(f"[standup] hand links {_HAND_LINK_NAMES} not found — "
                  "contact obs will be zeroed.")

    def _read_contact_state(self) -> np.ndarray:
        """Returns (N, 8) addon obs:
        [lf_z, rf_z, lh_z, rh_z, lf_contact, rf_contact, lh_contact, rh_contact].
        Heights are world-frame; contact is a binary derived from z-threshold."""
        N = self.num_envs
        if self._foot_links is None or self._hand_links is None:
            self._ensure_contact_links()
        out = np.zeros((N, 8), dtype=np.float32)
        try:
            if self._foot_links is not None:
                lf = _to_np(self._foot_links[0].get_pos())
                rf = _to_np(self._foot_links[1].get_pos())
                out[:, 0] = lf[:, 2]
                out[:, 1] = rf[:, 2]
                out[:, 4] = (lf[:, 2] < _CONTACT_Z).astype(np.float32)
                out[:, 5] = (rf[:, 2] < _CONTACT_Z).astype(np.float32)
            if self._hand_links is not None:
                lh = _to_np(self._hand_links[0].get_pos())
                rh = _to_np(self._hand_links[1].get_pos())
                out[:, 2] = lh[:, 2]
                out[:, 3] = rh[:, 2]
                out[:, 6] = (lh[:, 2] < _CONTACT_Z).astype(np.float32)
                out[:, 7] = (rh[:, 2] < _CONTACT_Z).astype(np.float32)
        except Exception as e:
            print(f"[standup] contact read failed: {e}")
        return out

    def _skill_obs_addons(self) -> np.ndarray:
        # `proprio_only` → no contact obs; base class checks
        # SKILL_OBS_ADDONS > 0 before calling this, so the empty-array
        # return is a defensive fallback.
        if self.SKILL_OBS_ADDONS == 0:
            return np.zeros((self.num_envs, 0), dtype=np.float32)
        return self._read_contact_state()

    def _read_foot_z(self) -> np.ndarray:
        """Return (N, 2) world-frame z of left and right foot. Used by
        the `foot_grounded_up` reward term regardless of `proprio_only`
        — reward signals can use privileged data; only the policy obs
        respects the proprio constraint."""
        N = self.num_envs
        if self._foot_links is None:
            self._ensure_contact_links()
        out = np.zeros((N, 2), dtype=np.float32)
        if self._foot_links is None:
            return out
        try:
            out[:, 0] = _to_np(self._foot_links[0].get_pos())[:, 2]
            out[:, 1] = _to_np(self._foot_links[1].get_pos())[:, 2]
        except Exception as e:
            print(f"[standup] foot z read failed: {e}")
        return out

    # ── reset using the pool ──────────────────────────────────────

    def _reset_robot_pose(self, envs_idx: np.ndarray) -> None:
        # Build pools on first reset (lazy — needs scene + robot ready).
        # IMPORTANT: build the easy pool FIRST. Both builders set state
        # across all envs and step physics, so the second build overwrites
        # the first build's state — but only the FINAL state matters here
        # because the snapshots have already been taken and stored.
        if self._pool_pos is None:
            if self.cfg.easy_pool_enabled:
                self._build_easy_pool()
            self._build_settle_pool()

        n = envs_idx.shape[0]
        # Reverse curriculum: pick a fraction of envs to start from the
        # easy (near-standing) pool. Fraction ramps from 1.0 → 0.0 as
        # training progresses; once curriculum ends, all starts are
        # fallen (same as before).
        easy_frac = self._current_easy_fraction()
        use_easy = (self.rng.uniform(0.0, 1.0, size=n) < easy_frac) \
                   & (self._easy_pool_size > 0)
        n_easy = int(use_easy.sum())
        n_hard = n - n_easy

        pos = np.empty((n, 3), dtype=np.float32)
        quat = np.empty((n, 4), dtype=np.float32)
        jpos = np.empty((n, self.act_dim), dtype=np.float32)

        if n_easy > 0:
            ei = self.rng.integers(0, self._easy_pool_size, size=n_easy)
            pos[use_easy] = self._easy_pool_pos[ei]
            quat[use_easy] = self._easy_pool_quat[ei]
            jpos[use_easy] = self._easy_pool_jpos[ei]
        if n_hard > 0:
            hi = self.rng.integers(0, self._pool_size, size=n_hard)
            pos[~use_easy] = self._pool_pos[hi]
            quat[~use_easy] = self._pool_quat[hi]
            jpos[~use_easy] = self._pool_jpos[hi]

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

    # ── curricula ─────────────────────────────────────────────────
    #
    # Three independent linear ramps from `_start` → final values over
    # their respective horizon, all driven by `_total_env_steps_seen`.
    # Together they let the policy discover a partial standup at loose
    # criteria, then tighten toward deployment quality without ever
    # losing the gradient.

    def _curriculum_progress(self, horizon_env_steps: int) -> float:
        return min(self._total_env_steps_seen / max(int(horizon_env_steps), 1),
                   1.0)

    def _current_hold_steps(self) -> int:
        c = self.cfg
        p = self._curriculum_progress(c.hold_curriculum_env_steps)
        return int(round(c.success_hold_steps_start
                         + (c.success_hold_steps - c.success_hold_steps_start)
                           * p))

    def _current_upright_threshold(self) -> float:
        c = self.cfg
        p = self._curriculum_progress(c.threshold_curriculum_env_steps)
        return float(c.upright_threshold_start
                     + (c.upright_threshold - c.upright_threshold_start) * p)

    def _current_target_height(self) -> float:
        c = self.cfg
        p = self._curriculum_progress(c.threshold_curriculum_env_steps)
        return float(c.target_height_start
                     + (c.target_height - c.target_height_start) * p)

    def _current_easy_fraction(self) -> float:
        """Fraction of episodes starting from the EASY (near-standing) pool.

        Combines time-based decay with a performance gate: if the policy
        hasn't demonstrated minimum performance (`start_curriculum_min_success`
        EMA success rate), the easy fraction is held at its current level and
        only allowed to decay further once performance recovers. This prevents
        the curriculum from advancing during a stuck phase — the previous
        purely-time-based advancement left the policy on 0% easy starts with
        0% success rate by step 25M."""
        if not self.cfg.easy_pool_enabled or self._easy_pool_size == 0:
            return 0.0
        p = self._curriculum_progress(self.cfg.start_curriculum_env_steps)
        time_frac = float(1.0 - p)
        # Only allow further decay if the policy is meeting minimum performance.
        # If below the threshold, clamp to the minimum easy fraction that
        # keeps training viable.
        if self._success_rate_ema < self.cfg.start_curriculum_min_success:
            return max(time_frac, self.cfg.start_curriculum_min_easy_frac)
        return time_frac

    def _current_assist_fraction(self) -> float:
        """Fraction (1.0 → 0.0) of the peak assist force currently applied.

        Time-based linear decay over `assist_curriculum_env_steps`, gated
        on performance: while the EMA frame-success rate is below
        `assist_min_success`, the fraction is held at `assist_min_frac` so
        the support isn't pulled out from under a still-failing policy.
        Mirrors `_current_easy_fraction`'s gate."""
        if not self.cfg.assist_force_enabled:
            return 0.0
        p = self._curriculum_progress(self.cfg.assist_curriculum_env_steps)
        time_frac = float(1.0 - p)
        if self._success_rate_ema < self.cfg.assist_min_success:
            return max(time_frac, self.cfg.assist_min_frac)
        return time_frac

    # ── assistive upward force (force curriculum) ─────────────────
    #
    # Decaying world-frame upward force on the trunk — the HoST "help an
    # infant stand" trick. Summed onto the push-DR wrench by the base
    # `step()` via the `_assist_wrench` hook, applied BEFORE scene.step.

    def _assist_wrench(self):
        zeros = np.zeros((self.num_envs, 3), dtype=np.float32)
        frac = self._current_assist_fraction()
        if frac <= 0.0:
            self._last_assist_force_mean = 0.0
            return zeros, zeros

        try:
            z = _to_np(self.robot.get_pos())[:, 2]
        except Exception:
            self._last_assist_force_mean = 0.0
            return zeros, zeros

        peak = self.cfg.assist_force_max
        if self.cfg.assist_spring_shape:
            # Strongest when fully fallen, releasing to ~0 near standing.
            target = self.cfg.target_height
            deficit = np.clip((target - z) / max(target, 1e-6), 0.0, 1.0)
            fz = frac * peak * deficit
        else:
            # Flat support that simply decays with the curriculum.
            fz = np.full(self.num_envs, frac * peak, dtype=np.float32)

        force = zeros.copy()
        force[:, 2] = fz.astype(np.float32)  # world +z, upward only
        self._last_assist_force_mean = float(np.mean(fz))
        return force, zeros

    # ── reward + sustained-success bookkeeping ────────────────────

    def _compute_skill_reward(self, action: np.ndarray):
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            root_lin_vel = _to_np(self.robot.get_vel())
            root_ang_vel = _to_np(self.robot.get_ang())
            jpos = _to_np(self.robot.get_dofs_position(self.dof_indices))
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
            foot_z = self._read_foot_z()
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32), {}

        self._total_env_steps_seen += self.num_envs
        hold_steps = self._current_hold_steps()
        upright_thresh = self._current_upright_threshold()
        target_h = self._current_target_height()

        prev_streak = self._success_streak.copy()
        frame_now = success_frame_mask(
            root_quat, root_pos[:, 2],
            target_h=target_h,
            upright_threshold=upright_thresh,
        )
        new_streak = np.where(frame_now, prev_streak + 1, 0).astype(np.int32)
        sustained_now = (new_streak == hold_steps) \
                        & (prev_streak < hold_steps)
        # Latch: once an env has hit sustained success this episode, the
        # post-success standing reward is unlocked for every later frame
        # the robot stays upright. Cleared by _reset_skill_state.
        achieved_sustained = self._achieved_sustained | sustained_now

        reward, _frame_success, components = compute_standup_reward(
            root_pos=root_pos, root_quat=root_quat,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            joint_pos=jpos, joint_vel=jvel,
            action=action,
            prev_action=self._last_action,
            prev_prev_action=self._prev_prev_action,
            prev_upright=self._prev_upright,
            success_streak=new_streak,
            sustained_now=sustained_now,
            achieved_sustained=achieved_sustained,
            step_count=self.step_count,
            foot_z=foot_z,
            weights=self._reward_weights,
            arm_joint_indices=self.robot_cfg.arm_joint_indices,
            default_joint_pos=self._default_action,
            target_height=target_h,
            upright_threshold=upright_thresh,
            hold_steps=hold_steps,
            time_to_stand_tau_steps=self.cfg.time_to_stand_tau_steps,
            foot_grounded_max_z=self.cfg.foot_grounded_max_z,
            trunk_up_min_z=self.cfg.trunk_up_min_z,
            standing_tall_min_z=self.cfg.standing_tall_min_z,
            standing_tall_max_z=self.cfg.standing_tall_max_z,
            control_dt=self.dt,
        )

        components["hold_steps_current"] = float(hold_steps)
        components["upright_threshold_current"] = float(upright_thresh)
        components["target_height_current"] = float(target_h)
        components["easy_start_fraction"] = float(self._current_easy_fraction())
        components["assist_fraction"] = float(self._current_assist_fraction())
        components["assist_force_mean"] = float(self._last_assist_force_mean)

        # Update EMA of frame success rate — used to gate curriculum.
        current_success_rate = float(np.mean(frame_now))
        self._success_rate_ema = (
            (1.0 - self._success_ema_alpha) * self._success_rate_ema
            + self._success_ema_alpha * current_success_rate
        )
        components["success_rate_ema"] = self._success_rate_ema

        self._success_streak = new_streak
        self._sustained_now = sustained_now
        self._achieved_sustained = achieved_sustained
        self._frame_success = frame_now
        self._prev_prev_action = self._last_action.copy()
        self._prev_upright = upright_signal(root_quat).astype(np.float32)
        return reward, components

    # ── termination: timeout only ─────────────────────────────────
    #
    # The episode does NOT end at sustained success. Letting it run to
    # MAX_EPISODE_STEPS forces the policy to demonstrate that it can
    # KEEP standing after the bonus is paid — a fast-but-unstable
    # standup that collapses immediately gives up all of the
    # `post_success_standing` reward for the remaining frames. With a
    # 5 s episode and a 1.5 s standup, that's ~175 frames × 10 = 1750
    # of opportunity cost, dwarfing every other term.

    def _check_skill_done(self) -> np.ndarray:
        timeout = self.step_count >= self.MAX_EPISODE_STEPS
        return timeout

    # ── reset hook clears all per-env reward state ────────────────

    def _reset_skill_state(self, envs_idx: np.ndarray) -> None:
        self._frame_success[envs_idx] = False
        self._success_streak[envs_idx] = 0
        self._sustained_now[envs_idx] = False
        self._achieved_sustained[envs_idx] = False
        self._prev_prev_action[envs_idx] = 0.0  # zero delta = hold default pose
        # Initialise prev_upright from the just-reset robot's actual
        # orientation, so the first-step progress reward is 0 instead of
        # the +(up+1) freebie we'd get by comparing against a fake -1
        # baseline. Progress credit kicks in genuinely from step 2 onward.
        try:
            quat = _to_np(self.robot.get_quat())
            self._prev_upright[envs_idx] = upright_signal(
                quat[envs_idx]).astype(np.float32)
        except Exception:
            self._prev_upright[envs_idx] = -1.0
