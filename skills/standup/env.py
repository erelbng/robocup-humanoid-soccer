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
from skills.common_obs import _to_np, projected_gravity
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
# Trunk (torso/base) link — for the anti-slam contact-force penalty.
_TRUNK_LINK_NAME = "Trunk"
# Shank (lower-leg) links — proxy for "knee/shin on the ground" support.
_KNEE_LINK_NAMES = ("Left_Shank", "Right_Shank")
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


def _quat_rotate_body_to_world(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors to world frame using unit quaternion q.

    Args:
        q: (N, 4) float32 array in (w, x, y, z) order.
        v: (N, 3) float32 array of body-frame vectors.

    Returns:
        (N, 3) float32 world-frame vectors. Uses the standard Rodrigues
        formula: v_world = v + 2w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v)).
    """
    w = q[:, 0:1]   # (N, 1)
    xyz = q[:, 1:]  # (N, 3)
    t = 2.0 * np.cross(xyz, v)
    return (v + w * t + np.cross(xyz, t)).astype(np.float32)


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
        # Per-env "started on the back" flag (supine, incl. back-lying random
        # poses), set from the reset orientation. Gates the supine_anti_flip
        # penalty so only back-starts are taxed for rolling face-down.
        self._start_supine = np.zeros(self.num_envs, dtype=bool)
        # Base xy captured at reset — anchor for the "stand on the spot"
        # (on_spot) horizontal-travel penalty.
        self._start_xy = np.zeros((self.num_envs, 2), dtype=np.float32)
        # SHOULDER-wide standing target for the stand_pose reward: the default
        # pose with the hip-roll joints abducted (Left += a, Right -= a). Arms
        # and knees keep the default (arms at sides, slight knee bend).
        self._stand_target_pose = self._default_action.copy()
        _abd = float(self.cfg.stand_target_hip_abduction)
        _names = self.robot_cfg.joint_names
        if "Left_Hip_Roll" in _names and "Right_Hip_Roll" in _names:
            self._stand_target_pose[_names.index("Left_Hip_Roll")] += _abd
            self._stand_target_pose[_names.index("Right_Hip_Roll")] -= _abd
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
        # Env-step count at which the CURRENT level started. All easing
        # curricula (assist, hold_steps, upright/height thresholds, pose-mix
        # bias) measure progress RELATIVE to this, not the global clock — so
        # every level-up re-grants the assist bootstrap + loosened success
        # criteria for the freshly-introduced (hardest) pose. Without this the
        # global clock expires (horizons 25–150M) long before the hard poses
        # arrive (performance-gated, ~180M+), leaving a brand-new pose facing
        # the hardest criteria with zero assist → success ≈ 0 → gate stuck.
        self._level_start_env_steps: int = 0
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
        self._trunk_link_idx = None
        self._knee_link_idx = None
        # Episode high-water mark of the upright signal — the reference for the
        # ratcheted progress reward (only NEW uprightness beyond the best so
        # far is paid, killing the down→up "pump" of the dolphin slam).
        self._max_upright = np.full(self.num_envs, -1.0, dtype=np.float32)

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
            # PLUS the vertex-based penetration gate (same as the named-pose
            # pools): reject any state where the lowest collision vertex sits
            # below the floor — a foot/shin/limb wedged into the ground would
            # otherwise leak into the L3 random distribution (50% of resets).
            up = upright_signal(q)
            min_contact_z = self._min_contact_link_z()
            ok = ((p[:, 2] < c.pool_max_height)
                  & (up < c.pool_max_upright)
                  & (min_contact_z > -c.pose_pool_penetration_eps))
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

        # Per-pool noise overrides (crouch pools pass tighter values). The
        # settle-step override is resolved further down, after `is_side` is
        # known, so an explicit override (crouch) wins over the side default.
        qn = c.pose_pool_quat_noise_rad if quat_noise_rad is None else quat_noise_rad
        jj = c.pose_pool_joint_jitter_rad if joint_jitter_rad is None else joint_jitter_rad

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
        # Nominal body-frame gravity for this pose's lying CLASS — used to
        # reject settled states that rolled into a different class (e.g. a
        # side pose tipping onto its back/belly). supine≈(+1,0,0),
        # prone≈(-1,0,0), side_left≈(0,+1,0), side_right≈(0,-1,0).
        g_expected = projected_gravity(
            np.array([pose.trunk_quat], dtype=np.float32))[0]

        pool_pos, pool_quat, pool_jpos = [], [], []

        # Side poses use a much shorter settle than supine/prone.
        # Humanoid side-lying is NOT a stable equilibrium: PD joint torques
        # roll the trunk to prone/supine within ~60 physics steps of landing.
        # The full 1000-step settle always produces back/belly snapshots that
        # the orientation filter rejects → empty pool → fallback to generic
        # settle pool (no true side poses). Snapshot after only the landing
        # phase (~250 steps) while the orientation is still close to the
        # target side class. Higher rejection rate → more rounds to compensate.
        is_side = pose.name.startswith("side_")
        n_rounds = c.pose_pool_side_rounds if is_side else c.pose_pool_rounds
        # Settle length: an explicit caller override (crouch pools, which need
        # a SHORT settle to stay upright) wins; otherwise side poses settle
        # briefly (they roll out of side-lying) and the rest use the default.
        if settle_steps is None:
            settle_steps = (c.pose_pool_side_settle_steps
                            if is_side else c.pose_pool_settle_steps)

        for round_idx in range(n_rounds):
            # Compose base quat with small random tilt noise.
            noise = _small_tilt_quat(N, qn, self.rng)
            quat = _quat_mul(noise, base_quat)  # noise on top of base pose

            pos = np.zeros((N, 3), dtype=np.float32)
            # Per-pose spawn clearance (side poses need more — see StandupPose).
            pos[:, 2] = pose.trunk_height + pose.spawn_clearance

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

            if is_side:
                # GUARANTEED side orientation. Humanoid side-lying is an
                # UNSTABLE equilibrium — PD joint torques + gravity roll the
                # trunk onto its back/belly within ~60 physics steps of landing.
                # An external restoring torque (the previous approach) is damped
                # out by the solver and failed to hold the pose (eval showed all
                # side states settling supine/prone).
                #
                # Instead we KINEMATICALLY PIN the trunk orientation: every
                # physics step we re-set the base quaternion to the exact side
                # target, so the torso physically cannot roll. The EXTREMITY
                # joints still integrate FREELY under their own PD targets +
                # gravity + floor contact (we never touch their dofs), which
                # realises the user's spec: "the torso is definitely turned to
                # the side and only the joints of the extremities are changing".
                #
                #   * set_quat(zero_velocity=False) snaps the orientation back
                #     WITHOUT freezing the joint dofs, so the limbs keep settling.
                #   * we zero ONLY the base ANGULAR velocity (free-base dofs 3-5)
                #     so roll momentum can't accumulate and fight the snap; the
                #     base LINEAR velocity (dofs 0-2) is left intact so the trunk
                #     still free-falls and rests at its natural side-lying height.
                side_quat = quat.copy()                 # per-env (noised) side quat
                base_ang_dofs = [3, 4, 5]
                zero_ang = np.zeros((N, 3), dtype=np.float32)
                for _ in range(settle_steps):
                    self.scene.step()
                    try:
                        self.robot.set_quat(side_quat, envs_idx=all_idx,
                                            zero_velocity=False)
                    except Exception:
                        self.robot.set_quat(side_quat, envs_idx=all_idx)
                    try:
                        self.robot.set_dofs_velocity(
                            zero_ang, base_ang_dofs, envs_idx=all_idx)
                    except Exception:
                        pass
            else:
                for _ in range(settle_steps):
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
            # Penetration gate (applies to BOTH pool kinds): the lowest
            # COLLISION-MESH VERTEX of the robot must sit above the floor
            # (within -eps). Large quat/joint noise can rotate a limb under the
            # spawn clearance and the PD holds it embedded through the settle;
            # replaying such a snapshot pins the limb in the ground and feeds
            # negative contact-z into the reward. The vertex-based measure
            # catches a buried sole/shin/knee/elbow that the old link-origin
            # check missed (origin = ankle/knee joint, well above the floor).
            min_contact_z = self._min_contact_link_z()
            if keep_upright:
                # Crouch/squat pools (reverse-height recovery curriculum):
                # keep clearly-UPRIGHT, off-ground settled states (a stable
                # squat) and discard any that toppled during settling. The
                # fallen-specific orientation/height gates below DON'T apply —
                # they would reject exactly the upright states we want.
                ok = ((up > 0.7)
                      & (p[:, 2] > 0.12)
                      & (min_contact_z > -c.pose_pool_penetration_eps))
            else:
                max_z = pose.trunk_height + c.pose_pool_max_height_margin
                # Orientation-class gate: settled gravity must still align with
                # the pose's nominal direction, else the robot rolled into a
                # different lying class (side→back/belly drift) → drop it.
                class_dot = projected_gravity(q) @ g_expected     # (N,)
                ok = ((p[:, 2] < max_z)
                      & (up < c.pool_max_upright)
                      & (class_dot > c.pose_pool_orient_dot_min)
                      & (min_contact_z > -c.pose_pool_penetration_eps))
                # Side-pose height guards.
                # MIN: a robot that rolled to supine/prone despite the arm-brace
                # settles at trunk_z ≈ 0.06–0.09 m — reject anything below
                # pose_pool_side_min_trunk_z (0.10 m) even if the orientation
                # filter barely passed (it can pass up to ~37° off-axis).
                # MAX: unusually high states (arm-propped bridge, arched back)
                # are physically unstable — they collapse at episode start.
                # Reject trunk_z > pose_pool_side_max_trunk_z (0.20 m).
                if is_side:
                    ok &= (p[:, 2] > c.pose_pool_side_min_trunk_z)
                    ok &= (p[:, 2] < c.pose_pool_side_max_trunk_z)
            if ok.any():
                pp = p[ok].copy()
                pp[:, 0:2] = 0.0  # re-centre xy
                pool_pos.append(pp)
                pool_quat.append(q[ok].copy())
                pool_jpos.append(j[ok].copy())

        if not pool_pos:
            print(f"[standup] WARNING: pose pool '{pose.name}' empty after "
                  f"filtering (settle_steps={settle_steps}, rounds={n_rounds})"
                  f" — will fall back to settle pool at this level.")
            return _empty

        pos_arr = np.concatenate(pool_pos).astype(np.float32)
        quat_arr = np.concatenate(pool_quat).astype(np.float32)
        jpos_arr = np.concatenate(pool_jpos).astype(np.float32)
        return {"pos": pos_arr, "quat": quat_arr,
                "jpos": jpos_arr, "size": pos_arr.shape[0]}

    def _extract_side_pools_from_settle(self) -> None:
        """Override side-pose pools with states extracted from the random
        settle pool.

        Rationale: humanoid side-lying is NOT a stable equilibrium when the
        robot is spawned in a forced side orientation and let to settle — PD
        joint torques roll the trunk to prone/supine within ~60 physics
        steps of landing. The random settle pool avoids this entirely: robots
        start from random orientations, fall naturally, and some arrive at
        STABLE side-lying equilibria (the ground contact prevents further
        rolling). These naturally-settled states are physically correct and
        well-distributed — exactly like ``random_065`` which is a naturally
        settled side pose.

        Filter: dot(g_body, g_expected) > pose_pool_orient_dot_min:
          side_left  → g_body = (0, +1, 0) → keep where g_body[1] > threshold
          side_right → g_body = (0, -1, 0) → keep where g_body[1] < -threshold

        This is called AFTER ``_build_settle_pool`` (which populates
        _pool_quat) and AFTER ``_build_pose_pool`` side calls (so we always
        have a pool to override into).
        """
        c = self.cfg
        # Use a slightly more lenient threshold for random-pool extraction than
        # for the forced-settle filter. These are already physically settled
        # equilibria (no rolling possible post-settle), so accepting states
        # within 45° of pure side (dot > 0.70) gives more diverse training
        # starts while still being clearly side-lying (not back/belly).
        # The forced-settle pool uses the stricter pose_pool_orient_dot_min.
        threshold = max(c.pose_pool_orient_dot_min - 0.10, 0.65)
        g = projected_gravity(self._pool_quat)  # (P, 3)

        for name, sign in [("side_left", +1.0), ("side_right", -1.0)]:
            mask = (sign * g[:, 1]) > threshold
            n_found = int(mask.sum())
            if n_found > 0:
                pool = {
                    "pos": self._pool_pos[mask].copy(),
                    "quat": self._pool_quat[mask].copy(),
                    "jpos": self._pool_jpos[mask].copy(),
                    "size": n_found,
                }
                old_size = self._named_pools.get(name, {}).get("size", 0)
                self._named_pools[name] = pool
                print(f"[standup] side pool '{name}': replaced {old_size} "
                      f"forced-settle states with {n_found} naturally-settled "
                      f"states from random pool (dot > {threshold:.2f})")
            else:
                print(f"[standup] WARNING: no '{name}' states found in random "
                      f"settle pool (g_body[1]*{sign:+.0f} > {threshold:.2f}) — "
                      f"keeping forced-settle pool ({self._named_pools.get(name, {}).get('size', 0)} states)")

    def _build_all_pools(self) -> None:
        """Build all named-pose pools and the settle pool.

        Order: named pools first (provides a fallback even if side extraction
        fails), then settle pool (leaves envs in fallen state), then side-pool
        override from the settle pool's naturally-settled side states.
        """
        from envs.standup import all_poses, make_crouch_pose
        for pose in all_poses():
            pool = self._build_pose_pool(pose)
            self._named_pools[pose.name] = pool
            print(f"[standup] pose pool '{pose.name}': {pool['size']} states "
                  f"(forced settle)")

        # Reverse-height recovery curriculum: build the upright crouch/squat
        # START pools R0..R(K-1). (This loop was dropped in the d2ee216 merge —
        # without it self._crouch_pools stays empty and _sample_reset silently
        # falls back to fallen-pose sampling, defeating the whole curriculum.)
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
        # NOTE: _extract_side_pools_from_settle() is intentionally NOT called
        # here. The forced-settle pool (arm-brace config, 500-step settle,
        # 0.80 orientation + 0.07 m min-z filters) reliably produces 700+
        # genuine side-lying states. Replacing it with the random-pool
        # extraction (dot > 0.70, no min-z guard) introduced z≈0.06 m states
        # (robot 45° tilted toward prone/supine) that visually look like back-
        # lying poses. Keep the forced-settle pool — it is better quality.

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
        """Assemble (pos, quat, jpos, is_side) from multiple pools based on per-env masks.

        Returns a 4-tuple; `is_side` is a bool array marking envs that were
        assigned a side_left or side_right pool state — used by _reset_robot_pose
        to suppress joint jitter for those envs (side-lying is metastable; jitter
        can tip the elbow/foot brace off the floor and cause an immediate fall).
        """
        pos = np.empty((n, 3), dtype=np.float32)
        quat = np.empty((n, 4), dtype=np.float32)
        jpos = np.empty((n, self.act_dim), dtype=np.float32)
        is_side = np.zeros(n, dtype=bool)
        for pool_name, mask in name_mask_pairs:
            count = int(mask.sum())
            if count > 0:
                p, q, j = self._sample_from_pool(pool_name, count)
                pos[mask] = p
                quat[mask] = q
                jpos[mask] = j
                if pool_name.startswith("side_"):
                    is_side[mask] = True
        return pos, quat, jpos, is_side

    def _sample_reset(self, n: int) -> tuple:
        """Top-level reset sampler honoring the reverse-height curriculum.

        While in a crouch stage (R < R_final), sample from that stage's upright
        crouch pool; at the final stage, hand off to the fallen-pose L0-L3
        curriculum. Falls back to the fallen sampler if a crouch pool is empty.

        Returns (pos, quat, jpos, is_side) — see _gather_by_choice for is_side.
        """
        if (self.cfg.recovery_curriculum_enabled
                and self._recovery_stage < self._recovery_final_stage):
            pool = self._crouch_pools.get(self._recovery_stage)
            if pool is not None and pool.get("size", 0) > 0:
                idx = self.rng.integers(0, pool["size"], size=n)
                return (pool["pos"][idx].copy(),
                        pool["quat"][idx].copy(),
                        pool["jpos"][idx].copy(),
                        np.zeros(n, dtype=bool))
            warn_attr = f"_crouch_warn_{self._recovery_stage}"
            if not getattr(self, warn_attr, False):
                print(f"[standup] WARNING: crouch pool R{self._recovery_stage} "
                      f"empty, falling back to fallen-pose sampling")
                setattr(self, warn_attr, True)
        return self._sample_reset_from_level(n)

    def _sample_reset_from_level(self, n: int) -> tuple:
        """Return (pos, quat, jpos) for n envs based on the current pose level.

        L0 → prone only               — easiest single entry pose (belly)
        L1 → prone + supine           — add the back recovery
        L2 → all 4 named poses        — + side_left + side_right
        L3 → named + random fallen    — full robustness

        Prone and supine are *different motor strategies* (prone: arm push-up
        → tuck → stand; supine: roll/sit up, tuck knees), so they are added
        one level at a time rather than mixed 50/50 from step 0.

        On every level-up the sampler BIASES toward the just-introduced
        pose(s) and relaxes back to the level's base mix over
        `pose_mix_bias_env_steps` (per-level clock). This points the fresh
        capacity at the hard new pose and makes success_rate_ema reflect IT,
        so the advance gate measures progress on the new pose instead of
        being dominated by the already-mastered ones.
        """
        # Category order: [supine, prone, side_left, side_right, random].
        cats = ("supine", "prone", "side_left", "side_right", "random")
        level = self._pose_level

        # Base distribution + indices of the pose(s) NEWLY introduced here.
        if level <= 0:
            base = np.array([0., 1., 0., 0., 0.], dtype=np.float64)
            new = ()
        elif level == 1:
            base = np.array([0.5, 0.5, 0., 0., 0.], dtype=np.float64)
            new = (0,)                               # supine
        elif level == 2:
            base = np.array([0.25, 0.25, 0.25, 0.25, 0.], dtype=np.float64)
            new = (2, 3)                             # side_left, side_right
        else:
            rf = float(self.cfg.pose_mix_random_frac)
            nf = (1.0 - rf) / 4.0
            base = np.array([nf, nf, nf, nf, rf], dtype=np.float64)
            new = (4,)                               # random

        # Decaying bias toward the freshly-introduced pose(s).
        if new and self.cfg.pose_mix_bias_start > 0.0:
            b = self.cfg.pose_mix_bias_start * (
                1.0 - self._curriculum_progress(self.cfg.pose_mix_bias_env_steps))
            if b > 0.0:
                new_dist = np.zeros(5, dtype=np.float64)
                for i in new:
                    new_dist[i] = 1.0
                new_dist /= new_dist.sum()
                base = (1.0 - b) * base + b * new_dist

        base /= base.sum()  # guard against fp drift before sampling
        choice = self.rng.choice(5, size=n, p=base)
        # _gather_by_choice handles the "random" settle pool and skips empty
        # masks; per-env choice already avoids positional bias (no shuffle).
        # Returns (pos, quat, jpos, is_side) — is_side marks side_left/right envs.
        return self._gather_by_choice(
            [(cats[i], choice == i) for i in range(5)], n)

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
        # Trunk + knee/shank links for the anti-slam force penalty and the
        # knee-support credit. Stored as idx_local into the entity's
        # net-contact-force tensor (N, n_links, 3).
        trunk = links_by_name.get(_TRUNK_LINK_NAME)
        self._trunk_link_idx = (trunk.idx_local if trunk is not None else None)
        knees = [links_by_name.get(n) for n in _KNEE_LINK_NAMES]
        self._knee_link_idx = ([k.idx_local for k in knees]
                               if all(k is not None for k in knees) else None)
        if self._foot_links is None:
            print(f"[standup] foot links {_FOOT_LINK_NAMES} not found — "
                  "contact obs will be zeroed.")
        if self._hand_links is None:
            print(f"[standup] hand links {_HAND_LINK_NAMES} not found — "
                  "contact obs will be zeroed.")
        if self._trunk_link_idx is None:
            print(f"[standup] trunk link '{_TRUNK_LINK_NAME}' not found — "
                  "trunk contact-force penalty will be zeroed.")
        if self._knee_link_idx is None:
            print(f"[standup] knee links {_KNEE_LINK_NAMES} not found — "
                  "knee-support reward will be zeroed.")

    def _read_contact_forces(self):
        """Returns (trunk_force_mag (N,), knee_force_mag (N, 2)) — the net
        ground-contact force magnitudes on the Trunk and the two shanks. Used
        by the anti-slam penalty and the knee-support credit (privileged: a
        reward signal, never in the policy obs). Degrades to zeros if the
        Genesis contact-force API or the links are unavailable."""
        N = self.num_envs
        trunk = np.zeros(N, dtype=np.float32)
        knees = np.zeros((N, 2), dtype=np.float32)
        if self._trunk_link_idx is None and self._knee_link_idx is None:
            self._ensure_contact_links()
        try:
            f = _to_np(self.robot.get_links_net_contact_force())  # (N, L, 3)
            if f.ndim == 2:                      # (L, 3) single-env safety
                f = f[None, :, :]
            if self._trunk_link_idx is not None:
                trunk = np.linalg.norm(f[:, self._trunk_link_idx, :], axis=1)
            if self._knee_link_idx is not None:
                for i, li in enumerate(self._knee_link_idx):
                    knees[:, i] = np.linalg.norm(f[:, li, :], axis=1)
        except Exception as e:
            print(f"[standup] contact-force read failed: {e}")
        return trunk.astype(np.float32), knees.astype(np.float32)

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

    def _min_contact_link_z(self) -> np.ndarray:
        """Return (N,) world-z of the LOWEST COLLISION VERTEX of the robot per env.

        Used by the pose-pool penetration filter to reject snapshots where any
        part of the robot settled below the floor (z < 0).

        CRITICAL: this measures the actual collision-MESH geometry via Genesis
        `robot.get_verts()` (all collision vertices in world frame), NOT the
        link FRAME origins. `link.get_pos()` returns the link origin — e.g. the
        ankle joint for a foot, or the knee joint for a shin — which can sit
        several centimetres ABOVE the floor while the foot SOLE / shin SURFACE
        penetrates it. The origin-based check therefore silently passed poses
        with a leg buried in the ground. The vertex-based bound is exact: it is
        the true lowest surface point of the whole robot, so an embedded sole,
        shin, knee, elbow, or shoulder is always caught.

        A correctly settled fallen pose rests ON the floor → lowest vertex
        z ≈ 0 (within the solver's small steady-state contact penetration).
        A buried limb drives it clearly negative. The caller compares against
        `-pose_pool_penetration_eps`.

        Returns +inf on read failure so the caller's `> -eps` test leaves those
        states untouched rather than dropping them.
        """
        N = self.num_envs
        # Primary path: vertex-based lowest point of the entire collision mesh.
        try:
            verts = _to_np(self.robot.get_verts())   # (N, V, 3) batched
            if verts.ndim == 3:
                return verts[:, :, 2].min(axis=1).astype(np.float32)
            if verts.ndim == 2:                       # single-env fallback
                return np.full(N, float(verts[:, 2].min()), dtype=np.float32)
        except Exception as e:
            if not getattr(self, "_warned_getverts", False):
                print(f"[standup] robot.get_verts() unavailable "
                      f"({type(e).__name__}: {e}); falling back to per-link "
                      "AABB / origin penetration check.")
                self._warned_getverts = True

        # Fallback 1: per-link collision AABB lower bound (still geometry-aware).
        min_z = np.full(N, np.inf, dtype=np.float32)
        got_aabb = False
        try:
            for link in self.robot.links:
                try:
                    if getattr(link, "n_geoms", 0) == 0:
                        continue
                    aabb = _to_np(link.get_AABB())     # (N, 2, 3): [min, max]
                    min_z = np.minimum(min_z, aabb[:, 0, 2])
                    got_aabb = True
                except Exception:
                    pass
        except Exception:
            pass
        if got_aabb:
            return min_z

        # Fallback 2: link-origin minimum (coarse — misses geometry extent).
        min_z = np.full(N, np.inf, dtype=np.float32)
        try:
            for link in self.robot.links:
                try:
                    min_z = np.minimum(min_z, _to_np(link.get_pos())[:, 2])
                except Exception:
                    pass
        except Exception:
            self._ensure_contact_links()
            for links in (self._foot_links, self._hand_links):
                if links is None:
                    continue
                for link in links:
                    try:
                        min_z = np.minimum(min_z, _to_np(link.get_pos())[:, 2])
                    except Exception as e:
                        print(f"[standup] contact-link z read failed: {e}")
        return min_z

    # ── reset using the pool ──────────────────────────────────────

    def _reset_robot_pose(self, envs_idx: np.ndarray) -> None:
        # Build all pools on first reset (lazy — needs scene + robot ready).
        # _pool_pos is None until _build_settle_pool (called last) sets it.
        if self._pool_pos is None:
            self._build_all_pools()

        n = envs_idx.shape[0]
        pos, quat, jpos, is_side = self._sample_reset(n)

        # NO reset-time joint jitter. The pool states are already filtered to
        # be penetration-free (lowest collision vertex ≳ 0), but adding ±jitter
        # on the joint angles at reset re-introduces ground penetration — most
        # visibly the ankle/foot driven into the floor — because there is no
        # settling step after the jitter to resolve it. Pool diversity comes
        # from the settle physics (quat + joint noise + per-env DR) during pool
        # build, which IS resolved by the subsequent settle steps; that is the
        # right place for variation. (The build-time noise is unaffected.)

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
    # their respective horizon, measured on the PER-LEVEL clock (env-steps
    # since the current level started). Together they let the policy discover
    # a partial standup at loose criteria, then tighten toward deployment
    # quality — and crucially RE-loosen for each newly-introduced pose.

    def _curriculum_progress(self, horizon_env_steps: int) -> float:
        elapsed = self._total_env_steps_seen - self._level_start_env_steps
        return min(max(elapsed, 0) / max(int(horizon_env_steps), 1), 1.0)

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

    def _style_scale(self) -> float:
        """Stage gate (0..1) for the motion-quality / style reward terms — the
        single-run TWO-STAGE mechanism.

        Stage 1 (style_scale = 0): while Karl's pose curriculum is still climbing
        (pose_level below the final level), ALL style/quality terms are OFF, so
        the reward is the proven discovery set and the curriculum advances
        unhindered. Stage 2 (style_scale ramps 0→1): once the curriculum is
        COMPLETE (final level reached), ramp the style terms in by the success
        EMA — which the env reset to 0 on the final advance — so they sculpt a
        smooth, motionless, shoulder-wide hold on an ALREADY-generalising policy.

        This replaces the old global-success gate (clip(success_ema/0.5)), which
        crossed its threshold during L0 and stalled the curriculum there. With
        `style_stage_gate=False` it degrades to that legacy EMA ramp.
        """
        ref = max(self.cfg.style_success_ref, 1e-6)
        final_level = len(self.cfg.pose_level_thresholds)
        if (self.cfg.style_stage_gate and self.cfg.pose_curriculum_enabled
                and self._pose_level < final_level):
            return 0.0
        return float(np.clip(self._success_rate_ema / ref, 0.0, 1.0))

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
        plateaus below the target and would otherwise lean forever. It runs
        on the PER-LEVEL clock (see `_curriculum_progress`), so it resets to
        ~1.0 on every level-up — the assist can fully recover for each new
        pose instead of being stuck at 0 from an expired global clock."""
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
                # Restart the per-level easing clock: the new (hardest) pose
                # re-gets the assist bootstrap + loosened hold/upright/height
                # criteria + a fresh mix-bias toward itself.
                self._level_start_env_steps = self._total_env_steps_seen
                # Re-bootstrap the HoST assist for the new pose: the assist is
                # EMA-coupled, so without dropping the EMA here it would only
                # come back at ~8% (the EMA is still ≈ the advance threshold).
                # Resetting it to 0 makes the assist jump to full for the new,
                # harder pose and re-disarms the success-ramped stand_pose term
                # until the new pose is mastered.
                ema_before = self._success_rate_ema
                if self.cfg.reset_success_ema_on_level_up:
                    self._success_rate_ema = 0.0
                print(f"[standup] pose curriculum: L{old} → L{self._pose_level} "
                      f"(EMA={ema_before:.3f}); easing curricula + assist reset")
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

        # Privileged contact-force reads for the anti-slam penalty + knee credit.
        trunk_force, knee_force = self._read_contact_forces()

        # ── Single-run TWO-STAGE gate ────────────────────────────────────────
        # Stage 1 (discovery + Karl's L0-L3 curriculum): style_scale = 0, so all
        # the motion-quality terms (reg group, on_spot, stand_pose, trunk-force)
        # are OFF — the reward is the proven discovery set and the curriculum
        # runs unhindered. Stage 2 ("stand still" / style): once the curriculum
        # is COMPLETE (pose_level reached the final level), ramp style in by the
        # (now L3) success EMA so it sculpts a smooth, motionless, shoulder-wide
        # hold on an already-generalising policy. This replaces the old global-
        # success gate that fired during L0 and stalled the curriculum there.
        style_scale = self._style_scale()

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
            start_supine=self._start_supine,
            weights=self._reward_weights,
            arm_joint_indices=self.robot_cfg.arm_joint_indices,
            pose_joint_indices=(tuple(self.robot_cfg.arm_joint_indices)
                                + tuple(self.robot_cfg.leg_joint_indices)),
            default_joint_pos=self._default_action,
            stand_target_pose=self._stand_target_pose,
            stand_pose_dev_scale=self.cfg.stand_pose_dev_scale,
            success_ema=self._success_rate_ema,
            stand_pose_success_ref=self.cfg.stand_pose_success_ref,
            start_xy=self._start_xy,
            on_spot_tol=self.cfg.on_spot_tol,
            post_success_still_jv_scale=self.cfg.post_success_still_jv_scale,
            post_success_still_v_scale=self.cfg.post_success_still_v_scale,
            feet_under_base_plateau_d=self.cfg.feet_under_base_plateau_d,
            max_upright=self._max_upright,
            progress_ratchet=self.cfg.progress_ratchet,
            reg_success_ramp=self.cfg.reg_success_ramp,
            style_scale=style_scale,
            trunk_contact_force=trunk_force,
            trunk_contact_force_thresh=self.cfg.trunk_contact_force_thresh,
            trunk_contact_force_scale=self.cfg.trunk_contact_force_scale,
            knee_contact_force=knee_force,
            knee_contact_force_thresh=self.cfg.knee_contact_force_thresh,
            knee_support_min_z=self.cfg.knee_support_min_z,
            knee_support_max_z=self.cfg.knee_support_max_z,
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
        # Two-stage telemetry: stage 1 (curriculum/discovery) until style_scale
        # lifts off 0 at curriculum completion, then stage 2 (style/stand-still).
        components["style_scale"] = float(style_scale)
        components["training_stage"] = 2.0 if style_scale > 0.0 else 1.0

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

        # Report a SINGLE monotonic curriculum counter under the key the PPO
        # level-up pump watches (training/algorithms/ppo.py), so Karl's
        # log_std-pump + LR-reset (b09d769) fires on every curriculum advance —
        # entropy decays over time and jumps back up at each level-up.
        # With the recovery curriculum DISABLED (the current config) this is
        # exactly Karl's pose level L0..L3. (If the recovery curriculum is
        # re-enabled, prepend its stages R0..R(K-1) so the pump fires on those
        # advances too — they're 0 while disabled.)
        if self.cfg.recovery_curriculum_enabled:
            curriculum_level = self._recovery_stage + self._pose_level
        else:
            curriculum_level = self._pose_level
        components["pose_curriculum_level"] = float(curriculum_level)
        components["recovery_stage"] = float(self._recovery_stage)
        components["pose_level"] = float(self._pose_level)

        self._success_streak = new_streak
        self._sustained_now = sustained_now
        self._achieved_sustained = achieved_sustained
        self._frame_success = frame_now
        self._prev_prev_action = self._last_action.copy()
        self._prev_upright = upright_signal(root_quat).astype(np.float32)
        # Advance the episode high-water mark for the ratcheted progress reward.
        self._max_upright = np.maximum(self._max_upright,
                                       self._prev_upright).astype(np.float32)

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
            # High-water mark starts at the reset orientation so the ratchet
            # only pays uprightness BEYOND where the episode began.
            self._max_upright[envs_idx] = self._prev_upright[envs_idx]
            # Anchor the on-spot penalty to the post-reset base xy.
            pos = _to_np(self.robot.get_pos())
            self._start_xy[envs_idx] = pos[envs_idx, :2].astype(np.float32)
            # Body-frame gravity-x > +0.5 ⇔ lying on the back (supine). Side
            # (~0) and prone (-1) are excluded by orientation. Additionally
            # gate by a pose-level WINDOW so the anti-flip only arms where the
            # detour appears (default: L1 only) — NOT in L0 discovery, and
            # never on a back-lying RANDOM start in L2/L3. Must match the
            # rewards-side penalty (max(0, -proj_g_x)): both treat supine as
            # g_x>0 and the belly-flip as g_x<0.
            armed = (self.cfg.supine_anti_flip_min_level
                     <= self._pose_level
                     <= self.cfg.supine_anti_flip_max_level)
            g = projected_gravity(quat[envs_idx])
            self._start_supine[envs_idx] = (g[:, 0] > 0.5) & bool(armed)
        except Exception:
            self._prev_upright[envs_idx] = -1.0
            self._max_upright[envs_idx] = -1.0
            self._start_supine[envs_idx] = False
