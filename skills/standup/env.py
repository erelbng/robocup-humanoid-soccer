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
from skills.standup.rewards import (STANDUP_CRITIC_GROUPS,
                                     compute_standup_reward,
                                     feet_under_base_score,
                                     standing_on_feet_mask, success_frame_mask,
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


def _quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of two (N, 4) wxyz quaternion arrays.

    Result = q * r, i.e. apply r first then q (standard quaternion composition).
    Both inputs must be unit quaternions; output is unit by algebra."""
    w0, x0, y0, z0 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w1, x1, y1, z1 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    return np.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], axis=-1).astype(np.float32)


class K1StandupEnv(SkillEnv):

    SKILL_NAME = "standup"
    # Critic groups for multi-critic PPO (see rewards.STANDUP_CRITIC_GROUPS).
    CRITIC_GROUP_NAMES = STANDUP_CRITIC_GROUPS
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
        # The pose curriculum only advances when the policy shows real
        # performance, preventing time-only advancement while stuck.
        self._success_rate_ema: float = 0.0
        self._success_ema_alpha: float = 0.005  # ~200-step window
        # Upright signal from the previous step — used by the progress
        # reward term. Initialised to -1 (fully inverted) so the first
        # step from any fallen pose produces a positive Δup credit.
        self._prev_upright = -np.ones(self.num_envs, dtype=np.float32)

        # Mean assist force applied last step (N) — for logging.
        self._last_assist_force_mean: float = 0.0

        # ── Pose difficulty curriculum (L0–L3) ────────────────────────────
        self._pose_level: int = self.cfg.pose_curriculum_start_level
        self._pose_level_sustain_steps: int = 0
        # Named-pose pools built lazily alongside the settle pool.
        # {pose_name: {"pos": (M,3), "quat": (M,4), "jpos": (M,22), "size": int}}
        self._named_pools: dict = {}

        # ── Reverse-height get-up curriculum (R0..R_final) ────────────────
        # Outer curriculum: start near standing (upright crouch) and move the
        # start progressively more fallen. Stages 0..K-1 sample from crouch
        # pools; the final stage K hands off to the fallen-pose L0-L3 curriculum.
        self._recovery_final_stage: int = len(self.cfg.recovery_crouch_heights)
        self._recovery_stage: int = (
            self.cfg.recovery_start_stage
            if self.cfg.recovery_curriculum_enabled
            else self._recovery_final_stage)
        self._recovery_sustain_steps: int = 0
        # Crouch start pools built lazily in _build_all_pools. {stage: pool}.
        self._crouch_pools: dict = {}

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

    # ── K1-accurate frequency-based leg gains ──────────────────────────
    def _build_per_joint_gains(self):
        """Per-joint LEG gains matching Booster's real K1 Isaac config
        (booster_train): kp = armature·(2π·f_n)², kd = 2·ζ·armature·(2π·f_n),
        f_n in Hz. Computed from the (K1-accurate) per-joint armatures, so the
        legs get K1's true gains (~kp hip-pitch 30 / knee 60 / ankle 36) instead
        of the wrong flat kp=40 (runs #1-5) or T1's too-stiff kp=200 (run #6).
        Arms/head keep the K1RobotConfig group gains (via super())."""
        kp, kd = super()._build_per_joint_gains()
        if not getattr(self.cfg, "use_frequency_gains", False):
            return kp, kd
        arm = self._build_armature()
        w = 2.0 * np.pi * float(self.cfg.gain_natural_freq_hz)
        shown = {}
        for i, name in enumerate(self.robot_cfg.joint_names):
            if i >= len(kp):
                break
            lo = name.lower()
            if not (("hip" in lo) or ("knee" in lo) or ("ankle" in lo)):
                continue
            zeta = (self.cfg.gain_damping_ratio_knee if "knee" in lo
                    else self.cfg.gain_damping_ratio_leg)
            kp[i] = float(arm[i] * w * w)
            kd[i] = float(2.0 * zeta * arm[i] * w)
            if name.startswith("Left_"):
                shown[name.replace("Left_", "")] = (round(kp[i], 1),
                                                    round(kd[i], 2))
        print(f"[standup] K1 frequency-based leg gains "
              f"(f_n={self.cfg.gain_natural_freq_hz} Hz, ζ leg/knee="
              f"{self.cfg.gain_damping_ratio_leg}/{self.cfg.gain_damping_ratio_knee}) "
              f"kp,kd = {shown}")
        return kp, kd

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

    def _build_pose_pool(self, pose, keep_upright: bool = False,
                         quat_noise_rad: float = None,
                         joint_jitter_rad: float = None,
                         settle_steps: int = None) -> dict:
        """Spawn all envs at a StandupPose, briefly settle, snapshot.

        Orientation noise (σ = cfg.pose_pool_quat_noise_rad) is composed on top
        of the reference quat so each sample is a physically distinct variant.
        Returns {"pos", "quat", "jpos", "size"}. Returns size=0 on failure
        (does NOT raise) so the caller can fall back gracefully.

        `keep_upright=False` (default) keeps only FALLEN settled states (for the
        supine/prone/side fallen pools). `keep_upright=True` inverts the filter
        to keep only UPRIGHT, off-ground settled states — used to build the
        reverse-height curriculum's crouch/squat start pools, which are
        deliberately upright (so the standard fallen filter would discard them).
        """
        c = self.cfg
        N = self.num_envs
        all_idx = np.arange(N)

        # Per-pool noise/settle overrides (crouch pools pass tighter values).
        qn = c.pose_pool_quat_noise_rad if quat_noise_rad is None else quat_noise_rad
        jj = c.pose_pool_joint_jitter_rad if joint_jitter_rad is None else joint_jitter_rad
        ss = c.pose_pool_settle_steps if settle_steps is None else settle_steps

        _empty = {"pos": np.zeros((0, 3), dtype=np.float32),
                  "quat": np.zeros((0, 4), dtype=np.float32),
                  "jpos": np.zeros((0, self.act_dim), dtype=np.float32),
                  "size": 0}

        # Build reference joint target from the pose's joint dict.
        name_to_idx = {name: i
                       for i, name in enumerate(self.robot_cfg.joint_names)}
        jpos_ref = self._default_action.copy()
        for jname, angle in pose.joint_targets.items():
            idx = name_to_idx.get(jname)
            if idx is not None:
                jpos_ref[idx] = float(angle)

        # Base orientation tiled to (N, 4).
        base_quat = np.tile(
            np.array([pose.trunk_quat], dtype=np.float32), (N, 1))

        pool_pos, pool_quat, pool_jpos = [], [], []

        for round_idx in range(c.pose_pool_rounds):
            # Compose base quat with small random tilt noise.
            noise = _small_tilt_quat(N, qn, self.rng)
            quat = _quat_mul(noise, base_quat)  # noise on top of base pose

            pos = np.zeros((N, 3), dtype=np.float32)
            pos[:, 2] = pose.trunk_height + 0.05  # small clearance above floor

            # Small joint jitter so pool entries aren't rigidly identical.
            jpos_target = (np.tile(jpos_ref, (N, 1))
                           + self.rng.standard_normal((N, self.act_dim))
                              .astype(np.float32)
                              * jj)

            try:
                self.robot.set_pos(pos, envs_idx=all_idx)
                self.robot.set_quat(quat, envs_idx=all_idx)
                self.robot.set_dofs_position(jpos_target, self.dof_indices,
                                              envs_idx=all_idx,
                                              zero_velocity=True)
                self.robot.control_dofs_position(jpos_target, self.dof_indices)
            except Exception as e:
                print(f"[standup] pose pool '{pose.name}' spawn "
                      f"(round {round_idx}) failed: {e}")
                continue

            for _ in range(ss):
                self.scene.step()

            try:
                p = _to_np(self.robot.get_pos()).copy()
                q = _to_np(self.robot.get_quat()).copy()
                j = _to_np(self.robot.get_dofs_position(
                    self.dof_indices)).copy()
            except Exception as e:
                print(f"[standup] pose pool '{pose.name}' snapshot failed: {e}")
                continue

            up = upright_signal(q)
            if keep_upright:
                # Crouch pools: keep clearly-UPRIGHT, off-ground settled states
                # (a stable squat), discard any that toppled during settling.
                ok = (up > 0.7) & (p[:, 2] > 0.12)
            else:
                max_z = pose.trunk_height + c.pose_pool_max_height_margin
                ok = (p[:, 2] < max_z) & (up < c.pool_max_upright)
            if ok.any():
                pp = p[ok].copy()
                pp[:, 0:2] = 0.0  # re-centre xy
                pool_pos.append(pp)
                pool_quat.append(q[ok].copy())
                pool_jpos.append(j[ok].copy())

        if not pool_pos:
            print(f"[standup] WARNING: pose pool '{pose.name}' empty after "
                  f"filtering — will fall back to settle pool at this level.")
            return _empty

        pos_arr = np.concatenate(pool_pos).astype(np.float32)
        quat_arr = np.concatenate(pool_quat).astype(np.float32)
        jpos_arr = np.concatenate(pool_jpos).astype(np.float32)
        return {"pos": pos_arr, "quat": quat_arr,
                "jpos": jpos_arr, "size": pos_arr.shape[0]}

    def _build_all_pools(self) -> None:
        """Build all named-pose pools and the settle pool.

        Order matters only for the settle pool (goes last so it leaves envs
        in the most neutral state for training).
        """
        from envs.standup import all_poses, make_crouch_pose
        for pose in all_poses():
            pool = self._build_pose_pool(pose)
            self._named_pools[pose.name] = pool
            print(f"[standup] pose pool '{pose.name}': {pool['size']} states")

        # Reverse-height curriculum: upright crouch/squat start pools (R0..R_K-1).
        self._crouch_pools = {}
        if self.cfg.recovery_curriculum_enabled:
            heights = self.cfg.recovery_crouch_heights
            scales = self.cfg.recovery_bend_scales
            for s in range(len(heights)):
                cpose = make_crouch_pose(
                    f"crouch{s}", self._default_action,
                    self.robot_cfg.joint_names,
                    bend_scale=float(scales[s]),
                    trunk_height=float(heights[s]),
                    d_hip=self.cfg.recovery_crouch_delta_hip,
                    d_knee=self.cfg.recovery_crouch_delta_knee,
                    d_ankle=self.cfg.recovery_crouch_delta_ankle)
                pool = self._build_pose_pool(
                    cpose, keep_upright=True,
                    quat_noise_rad=self.cfg.recovery_crouch_quat_noise_rad,
                    joint_jitter_rad=self.cfg.recovery_crouch_joint_jitter_rad,
                    settle_steps=self.cfg.recovery_crouch_settle_steps)
                self._crouch_pools[s] = pool
                print(f"[standup] crouch pool R{s} "
                      f"(spawn_h={heights[s]}, bend={scales[s]}): "
                      f"{pool['size']} states")

        self._build_settle_pool()  # leaves envs in fallen state

    # ── pool sampling helpers ─────────────────────────────────────

    def _sample_from_pool(self, pool_name: str, n: int) -> tuple:
        """Return (pos, quat, jpos) of n random states from the named pool.

        Falls back to the settle pool with a one-time warning if the requested
        pool is empty or missing.
        """
        if pool_name == "random":
            idx = self.rng.integers(0, self._pool_size, size=n)
            return (self._pool_pos[idx].copy(),
                    self._pool_quat[idx].copy(),
                    self._pool_jpos[idx].copy())

        pool = self._named_pools.get(pool_name)
        if pool is None or pool["size"] == 0:
            warn_attr = f"_pool_warn_{pool_name}"
            if not getattr(self, warn_attr, False):
                print(f"[standup] WARNING: pool '{pool_name}' empty, "
                      f"falling back to settle pool")
                setattr(self, warn_attr, True)
            idx = self.rng.integers(0, self._pool_size, size=n)
            return (self._pool_pos[idx].copy(),
                    self._pool_quat[idx].copy(),
                    self._pool_jpos[idx].copy())

        idx = self.rng.integers(0, pool["size"], size=n)
        return (pool["pos"][idx].copy(),
                pool["quat"][idx].copy(),
                pool["jpos"][idx].copy())

    def _gather_by_choice(self, name_mask_pairs: list, n: int) -> tuple:
        """Assemble (pos, quat, jpos) from multiple pools based on per-env masks."""
        pos = np.empty((n, 3), dtype=np.float32)
        quat = np.empty((n, 4), dtype=np.float32)
        jpos = np.empty((n, self.act_dim), dtype=np.float32)
        for pool_name, mask in name_mask_pairs:
            count = int(mask.sum())
            if count > 0:
                p, q, j = self._sample_from_pool(pool_name, count)
                pos[mask] = p
                quat[mask] = q
                jpos[mask] = j
        return pos, quat, jpos

    def _sample_reset(self, n: int) -> tuple:
        """Top-level reset sampler honoring the reverse-height curriculum.

        While in a crouch stage (R < R_final), sample from that stage's upright
        crouch pool; at the final stage, hand off to the fallen-pose L0-L3
        curriculum. Falls back to the fallen sampler if a crouch pool is empty.
        """
        if (self.cfg.recovery_curriculum_enabled
                and self._recovery_stage < self._recovery_final_stage):
            pool = self._crouch_pools.get(self._recovery_stage)
            if pool is not None and pool.get("size", 0) > 0:
                idx = self.rng.integers(0, pool["size"], size=n)
                return (pool["pos"][idx].copy(),
                        pool["quat"][idx].copy(),
                        pool["jpos"][idx].copy())
            warn_attr = f"_crouch_warn_{self._recovery_stage}"
            if not getattr(self, warn_attr, False):
                print(f"[standup] WARNING: crouch pool R{self._recovery_stage} "
                      f"empty, falling back to fallen-pose sampling")
                setattr(self, warn_attr, True)
        return self._sample_reset_from_level(n)

    def _sample_reset_from_level(self, n: int) -> tuple:
        """Return (pos, quat, jpos) for n envs based on the current pose level.

        L0 → supine only              — easiest single entry pose
        L1 → supine + prone (50/50)   — add the harder front recovery
        L2 → all 4 named poses equally — + side_left + side_right
        L3 → named (1-frac) + random fallen (frac) — full robustness

        Supine and prone are *different motor strategies* (supine: roll/sit
        up, tuck knees; prone: arm push-up → tuck → stand) and prone also
        pulls toward the cobra. Mixing both 50/50 from step 0 averages the
        gradients so the policy masters neither early. L0 isolates supine so
        the policy builds a foundation on one pose before prone is added —
        and each stage gets its own reachable EMA gate instead of one combined
        rate that the lagging pose drags below threshold.
        """
        level = self._pose_level
        names = ["supine", "prone", "side_left", "side_right"]

        if level <= 0:
            return self._sample_from_pool("supine", n)

        if level == 1:
            choice = self.rng.integers(0, 2, size=n)
            return self._gather_by_choice(
                [("supine", choice == 0), ("prone", choice == 1)], n)

        if level == 2:
            choice = self.rng.integers(0, 4, size=n)
            return self._gather_by_choice(
                [(nm, choice == i) for i, nm in enumerate(names)], n)

        # L3+: named poses + random fallen mix
        n_random = int(round(n * self.cfg.pose_mix_random_frac))
        n_named = n - n_random
        pos = np.empty((n, 3), dtype=np.float32)
        quat = np.empty((n, 4), dtype=np.float32)
        jpos = np.empty((n, self.act_dim), dtype=np.float32)
        if n_named > 0:
            choice = self.rng.integers(0, 4, size=n_named)
            pn, qn, jn = self._gather_by_choice(
                [(nm, choice == i) for i, nm in enumerate(names)], n_named)
            pos[:n_named] = pn
            quat[:n_named] = qn
            jpos[:n_named] = jn
        if n_random > 0:
            pr, qr, jr = self._sample_from_pool("random", n_random)
            pos[n_named:] = pr
            quat[n_named:] = qr
            jpos[n_named:] = jr
        # Shuffle to avoid positional bias across parallel envs.
        perm = self.rng.permutation(n)
        return pos[perm], quat[perm], jpos[perm]

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

    def _read_foot_pos(self) -> np.ndarray:
        """Return (N, 2, 3) world-frame position of left and right foot.
        Used by the stand-on-feet reward terms regardless of
        `proprio_only` — reward signals can use privileged data; only the
        policy obs respects the proprio constraint. The z column feeds the
        feet-grounded score; the xy columns feed the feet-under-base
        anti-cobra gate."""
        N = self.num_envs
        if self._foot_links is None:
            self._ensure_contact_links()
        out = np.zeros((N, 2, 3), dtype=np.float32)
        if self._foot_links is None:
            return out
        try:
            out[:, 0, :] = _to_np(self._foot_links[0].get_pos())
            out[:, 1, :] = _to_np(self._foot_links[1].get_pos())
        except Exception as e:
            print(f"[standup] foot pos read failed: {e}")
        return out

    # ── reset using the pool ──────────────────────────────────────

    def _reset_robot_pose(self, envs_idx: np.ndarray) -> None:
        # Build all pools on first reset (lazy — needs scene + robot ready).
        # _pool_pos is None until _build_settle_pool (called last) sets it.
        if self._pool_pos is None:
            self._build_all_pools()

        n = envs_idx.shape[0]
        pos, quat, jpos = self._sample_reset(n)

        # Small Gaussian joint jitter adds continuous variation on top of
        # the discrete pool — effectively infinite unique starting states.
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
            # `zero_velocity=True` only zeros the actuated joint DOFs we just
            # set; the 6 free-base DOFs keep whatever linear/angular velocity
            # the trunk had at episode end. Zero ALL DOFs so the new episode
            # starts from rest with no residual-impulse carryover from the
            # previous fall.
            self.robot.zero_all_dofs_velocity(envs_idx=envs_idx)
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

    def _current_assist_fraction(self) -> float:
        """Fraction (1.0 → 0.0) of the peak assist force currently applied.

        PERFORMANCE-COUPLED: the assist fades as the policy gets better,
        tied directly to the success EMA —
            success_frac = clip(1 - success_ema / assist_success_target, 0, 1)
        so it's full at zero competence and ~0 once success reaches the
        target. This auto-couples the assist to the pose curriculum: a
        level-up introduces a harder pose, the success EMA drops, and the
        assist rises back to help.

        A multiplicative TIME-DECAY backstop (1.0 → 0.0 over
        `assist_curriculum_env_steps`) guarantees weaning even if the policy
        plateaus below the target and would otherwise lean forever."""
        if not self.cfg.assist_force_enabled:
            return 0.0
        target = max(self.cfg.assist_success_target, 1e-6)
        success_frac = float(
            np.clip(1.0 - self._success_rate_ema / target, 0.0, 1.0))
        time_frac = float(
            1.0 - self._curriculum_progress(self.cfg.assist_curriculum_env_steps))
        return success_frac * time_frac

    # ── assistive upward force (force curriculum) ─────────────────
    #
    # Decaying world-frame upward force on the trunk — the HoST "help an
    # infant stand" trick. Summed onto the push-DR wrench by the base
    # `step()` via the `_assist_wrench` hook, applied BEFORE scene.step.

    def _assist_wrench(self):
        zeros = np.zeros((self.num_envs, 3), dtype=np.float32)
        force = zeros.copy()
        torque = zeros.copy()

        # ── upward assist force (decaying curriculum) ──────────────
        frac = self._current_assist_fraction()
        if frac > 0.0:
            try:
                root_pos = _to_np(self.robot.get_pos())
                z = root_pos[:, 2]
                peak = self.cfg.assist_force_max
                if self.cfg.assist_spring_shape:
                    target = self.cfg.target_height
                    deficit = np.clip(
                        (target - z) / max(target, 1e-6), 0.0, 1.0)
                    fz = frac * peak * deficit
                else:
                    fz = np.full(self.num_envs, frac * peak, dtype=np.float32)
                # Cobra-specific gate. NOT "feet under base" alone — that
                # also zeroes the assist for a freshly fallen robot (feet
                # naturally splayed), killing the bootstrap. The cobra marker
                # is trunk LIFTED *and* feet BEHIND: only then throttle. A
                # fallen robot (trunk down) keeps full support.
                #   cobra_factor = 1 - trunk_lifted · (1 - feet_under_base)
                if self.cfg.assist_cobra_gate:
                    foot_xy = self._read_foot_pos()[:, :, :2]   # (N, 2, 2)
                    under_base_soft = feet_under_base_score(
                        foot_xy, root_pos[:, :2],
                        d_max=self.cfg.assist_under_base_soft_d)
                    z_low = self.cfg.assist_cobra_z_low
                    z_high = self.cfg.assist_cobra_z_high
                    trunk_lifted = np.clip(
                        (z - z_low) / max(z_high - z_low, 1e-6), 0.0, 1.0)
                    cobra_factor = 1.0 - trunk_lifted * (1.0 - under_base_soft)
                    fz = fz * cobra_factor.astype(np.float32)
                force[:, 2] = fz.astype(np.float32)
                self._last_assist_force_mean = float(np.mean(fz))
            except Exception:
                self._last_assist_force_mean = 0.0
        else:
            self._last_assist_force_mean = 0.0

        return force, torque

    # ── pose curriculum advancement ───────────────────────────────

    def _maybe_advance_recovery_stage(self) -> None:
        """Advance the reverse-height curriculum stage R→R+1 once the success
        EMA has held above the stage threshold for recovery_advance_sustain_steps
        cumulative env-steps. Never regresses; caps at the final (fallen) stage.
        """
        if not self.cfg.recovery_curriculum_enabled:
            return
        if self._recovery_stage >= self._recovery_final_stage:
            return
        thresholds = self.cfg.recovery_stage_thresholds
        threshold = (thresholds[self._recovery_stage]
                     if self._recovery_stage < len(thresholds)
                     else thresholds[-1])
        if self._success_rate_ema >= threshold:
            self._recovery_sustain_steps += self.num_envs
            if self._recovery_sustain_steps >= self.cfg.recovery_advance_sustain_steps:
                old = self._recovery_stage
                self._recovery_stage += 1
                self._recovery_sustain_steps = 0
                tag = (f"R{self._recovery_stage}"
                       if self._recovery_stage < self._recovery_final_stage
                       else "R_final (fallen poses)")
                print(f"[standup] recovery curriculum: R{old} → {tag} "
                      f"(EMA={self._success_rate_ema:.3f})")
        else:
            self._recovery_sustain_steps = 0

    def _maybe_advance_level(self) -> None:
        """Advance pose curriculum level when EMA has been above threshold
        for pose_advance_sustain_steps cumulative env-steps. Never regresses.
        Caps at the final level (len(thresholds) = num_levels - 1)."""
        if not self.cfg.pose_curriculum_enabled:
            return
        thresholds = self.cfg.pose_level_thresholds
        if self._pose_level >= len(thresholds):
            return
        threshold = thresholds[self._pose_level]
        if self._success_rate_ema >= threshold:
            self._pose_level_sustain_steps += self.num_envs
            if self._pose_level_sustain_steps >= self.cfg.pose_advance_sustain_steps:
                old = self._pose_level
                self._pose_level += 1
                self._pose_level_sustain_steps = 0
                print(f"[standup] pose curriculum: L{old} → L{self._pose_level} "
                      f"(EMA={self._success_rate_ema:.3f})")
        else:
            # EMA fell below threshold — reset sustain counter (no regression).
            self._pose_level_sustain_steps = 0

    # ── reward + sustained-success bookkeeping ────────────────────

    def _compute_skill_reward(self, action: np.ndarray):
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            root_lin_vel = _to_np(self.robot.get_vel())
            root_ang_vel = _to_np(self.robot.get_ang())
            jpos = _to_np(self.robot.get_dofs_position(self.dof_indices))
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
            foot_pos = self._read_foot_pos()
            foot_z = foot_pos[:, :, 2]      # (N, 2)
            foot_xy = foot_pos[:, :, :2]    # (N, 2, 2)
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32), {}

        self._total_env_steps_seen += self.num_envs
        hold_steps = self._current_hold_steps()
        upright_thresh = self._current_upright_threshold()
        target_h = self._current_target_height()

        # Hard "actually standing on its feet" gate — feet grounded AND
        # under the base. Gates success detection so the success bonus /
        # post-success reward can't be farmed from an assist-propped cobra.
        feet_ok = standing_on_feet_mask(
            foot_z, foot_xy, root_pos[:, :2],
            foot_max_z=self.cfg.success_foot_max_z,
            under_base_max_d=self.cfg.success_under_base_max_d,
        )

        prev_streak = self._success_streak.copy()
        frame_now = success_frame_mask(
            root_quat, root_pos[:, 2],
            target_h=target_h,
            upright_threshold=upright_thresh,
            feet_ok=feet_ok,
        )
        new_streak = np.where(frame_now, prev_streak + 1, 0).astype(np.int32)
        sustained_now = (new_streak == hold_steps) \
                        & (prev_streak < hold_steps)
        # Latch: once an env has hit sustained success this episode, the
        # post-success standing reward is unlocked for every later frame
        # the robot stays upright. Cleared by _reset_skill_state.
        achieved_sustained = self._achieved_sustained | sustained_now

        reward, _frame_success, components, group_rewards = compute_standup_reward(
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
            foot_xy=foot_xy,
            feet_ok=feet_ok,
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
            feet_under_base_soft_d=self.cfg.feet_under_base_soft_d,
            explosive_rise_v_cap=self.cfg.explosive_rise_v_cap,
            control_dt=self.dt,
        )

        components["hold_steps_current"] = float(hold_steps)
        components["upright_threshold_current"] = float(upright_thresh)
        components["target_height_current"] = float(target_h)
        components["assist_fraction"] = float(self._current_assist_fraction())
        components["assist_force_mean"] = float(self._last_assist_force_mean)

        # Update EMA of frame success rate — used to gate curriculum.
        current_success_rate = float(np.mean(frame_now))
        self._success_rate_ema = (
            (1.0 - self._success_ema_alpha) * self._success_rate_ema
            + self._success_ema_alpha * current_success_rate
        )
        components["success_rate_ema"] = self._success_rate_ema

        # Curriculum advancement. The reverse-height (recovery) curriculum is
        # the OUTER stage; the fallen-pose L0-L3 curriculum only starts
        # advancing once we've reached the final (fallen) recovery stage.
        self._maybe_advance_recovery_stage()
        if self._recovery_stage >= self._recovery_final_stage:
            self._maybe_advance_level()

        components["pose_curriculum_level"] = float(self._pose_level)
        components["recovery_stage"] = float(self._recovery_stage)

        self._success_streak = new_streak
        self._sustained_now = sustained_now
        self._achieved_sustained = achieved_sustained
        self._frame_success = frame_now
        self._prev_prev_action = self._last_action.copy()
        self._prev_upright = upright_signal(root_quat).astype(np.float32)

        # Per-env group rewards (N, G) in STANDUP_CRITIC_GROUPS order —
        # consumed by the multi-critic trainer via info["group_rewards"].
        # Single-critic training ignores this. Always populated (cheap).
        self._group_rewards = np.stack(
            [group_rewards[g] for g in STANDUP_CRITIC_GROUPS], axis=1
        ).astype(np.float32)
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
