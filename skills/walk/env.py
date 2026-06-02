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
from skills.common_obs import _to_np, projected_gravity, body_frame_velocity
from skills.walk.config import WalkConfig
from skills.walk.rewards import compute_walk_reward


# Soft contact threshold — foot_z below this is treated as in-contact.
# Real Genesis contact queries would be more accurate; this proxy is
# Foot-LINK-origin height above ground at which the sole is in contact.
# MEASURED in Genesis: standing flat, the K1 foot_link origin sits at
# z≈0.065 m (the link frame is the ankle-roll joint, ~6.5 cm above the
# sole). The old 0.04 m threshold was BELOW that, so contact was NEVER
# detected → the gait-contact + feet-slip terms were dead no-ops and the
# robot just shuffled. 0.085 = standing 0.065 + ~2 cm tolerance: a planted
# foot reads as contact, a foot lifted ≳2 cm into swing reads as swing.
_CONTACT_Z = 0.085
# Foot-link-origin z when standing flat in Genesis (measured ≈0.065). Used as
# the ground baseline for the AMP foot-clearance feature (height above this).
_FOOT_GROUND_Z = 0.065
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
        # Per-foot airborne time (s) for the feet_air_time reward (the classic
        # anti-shuffle term): accumulates while a foot is off the ground, is
        # read at touchdown, then zeroed.
        self._air_time = np.zeros((self.num_envs, 2), dtype=np.float32)
        # Gait-phase tracker (anti-shuffle): per-env phase ∈ [0,1) advancing at
        # the commanded step_freq → smooth alternating desired-contact pattern.
        # Random init so envs are de-phased. `_prev_foot_xy` is for finite-diff
        # horizontal foot speed (the anti-slip penalty).
        self._gait_phase = self.rng.uniform(0.0, 1.0, self.num_envs).astype(np.float32)
        self._prev_foot_xy = None
        # Shoulder-pitch joint indices (for the armswing reward) — derived from
        # joint names so it survives any reordering. K1 → (2, 6).
        names = self.robot_cfg.joint_names
        self._shoulder_pitch_indices = tuple(
            i for i, n in enumerate(names) if "Shoulder_Pitch" in n)

        # Speed-curriculum state (SPRINT-style): the forward-speed cap expands
        # only as the policy tracks the current speeds well. `_speed_prog`
        # accrues env-steps ONLY while the lin-vel tracking EMA clears the
        # gate, so a still-wobbly policy isn't handed sprint commands.
        self._total_env_steps_seen: int = 0
        self._speed_prog: int = 0
        self._track_ema: float = 0.0
        self._track_ema_alpha: float = 0.01
        # Independent yaw curriculum: the turning-rate cap expands only as YAW
        # tracking gets good (separate from the linear speed curriculum).
        self._yaw_prog: int = 0
        self._ang_ema: float = 0.0

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

    def _make_head_command_spec(self) -> CommandSpec:
        c = self.cfg if hasattr(self, "cfg") and self.cfg is not None else WalkConfig()
        return CommandSpec(
            dim=2,
            low=np.array([c.head_yaw_range[0], c.head_pitch_range[0]],
                         dtype=np.float32),
            high=np.array([c.head_yaw_range[1], c.head_pitch_range[1]],
                          dtype=np.float32),
            names=("head_yaw", "head_pitch"),
        )

    # ── command sampling: speed curriculum + frequency-adaptive gait ──

    def _current_vx_max(self) -> float:
        """Forward-speed cap, ramped from vx_walk_max → vx_range[1] as the
        gated speed-progress accrues (SPRINT speed curriculum)."""
        c = self.cfg
        if c.speed_curriculum_env_steps <= 0:
            return c.vx_range[1]
        p = min(self._speed_prog / float(c.speed_curriculum_env_steps), 1.0)
        return float(c.vx_walk_max + (c.vx_range[1] - c.vx_walk_max) * p)

    def _current_vyaw_max(self) -> float:
        """Turning-rate cap, ramped from vyaw_walk_max → vyaw_range[1] as the
        gated YAW-tracking progress accrues (yaw slow-start curriculum)."""
        c = self.cfg
        if c.speed_curriculum_env_steps <= 0:
            return c.vyaw_range[1]
        p = min(self._yaw_prog / float(c.speed_curriculum_env_steps), 1.0)
        return float(c.vyaw_walk_max + (c.vyaw_range[1] - c.vyaw_walk_max) * p)

    def _sample_commands(self, envs_idx: np.ndarray) -> np.ndarray:
        n = len(envs_idx)
        c, rng = self.cfg, self.rng
        vx = rng.uniform(c.vx_range[0], self._current_vx_max(), n)
        vy = rng.uniform(c.vy_range[0], c.vy_range[1], n)
        vyaw_hi = self._current_vyaw_max()        # symmetric ±cap (curriculum)
        vyaw = rng.uniform(-vyaw_hi, vyaw_hi, n)
        fc = rng.uniform(c.foot_clearance_range[0], c.foot_clearance_range[1], n)
        if getattr(c, "freq_adaptive_gait", False):
            # cadence rises with commanded speed: step_freq ≈ base + slope·|v|
            speed = np.sqrt(vx ** 2 + vy ** 2)
            sf = (c.step_freq_base + c.step_freq_per_mps * speed
                  + rng.normal(0.0, 0.1, n))
            sf = np.clip(sf, c.step_freq_range[0], c.step_freq_range[1])
        else:
            sf = rng.uniform(c.step_freq_range[0], c.step_freq_range[1], n)
        return np.stack([vx, vy, vyaw, fc, sf], axis=1).astype(np.float32)

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
        """Returns (feet_z (N,2), contact_mask (N,2) bool, foot_horiz_speed
        (N,2) m/s, air_time (N,2) s, contact_just_now (N,2) bool).

        Horizontal speed is a finite-difference of foot xy (anti-slip).
        air_time accumulates while a foot is airborne and is returned at its
        value for THIS step (read before being zeroed on contact), so the
        feet_air_time reward can pay it out on the landing step."""
        N = self.num_envs
        if self._foot_links is None:
            self._ensure_foot_links()
        if self._foot_links is None:
            z = np.zeros((N, 2), np.float32)
            return z, np.zeros((N, 2), bool), z, z, np.zeros((N, 2), bool)
        foot_xy = None
        try:
            l = _to_np(self._foot_links[0].get_pos())
            r = _to_np(self._foot_links[1].get_pos())
            feet_z = np.stack([l[:, 2], r[:, 2]], axis=1).astype(np.float32)
            foot_xy = np.stack([l[:, :2], r[:, :2]], axis=1).astype(np.float32)  # (N,2,2)
        except Exception:
            feet_z = np.zeros((N, 2), dtype=np.float32)
        contact = feet_z < _CONTACT_Z
        if (foot_xy is not None and self._prev_foot_xy is not None
                and self._prev_foot_xy.shape == foot_xy.shape):
            foot_horiz_speed = (np.linalg.norm(foot_xy - self._prev_foot_xy, axis=2)
                                / max(self.dt, 1e-6)).astype(np.float32)
        else:
            foot_horiz_speed = np.zeros((N, 2), dtype=np.float32)
        if foot_xy is not None:
            self._prev_foot_xy = foot_xy

        # Air-time bookkeeping (legged_gym order): mark feet that JUST landed,
        # accumulate airborne time, snapshot it for the reward, then zero the
        # feet now in contact.
        contact_just_now = contact & (~self._prev_contact)
        self._air_time += self.dt
        air_time = self._air_time.copy()
        self._air_time[contact] = 0.0

        self._prev_contact = contact
        return feet_z, contact, foot_horiz_speed, air_time, contact_just_now

    def _gait_desired_contact(self) -> np.ndarray:
        """Advance the gait phase at the commanded step_freq and return the
        smooth alternating desired-contact pattern (N,2) for [left, right]."""
        c = self.cfg
        sf = self.commands[:, 4]                       # step_freq command (Hz)
        self._gait_phase = np.mod(self._gait_phase + self.dt * sf, 1.0)
        k, duty = c.gait_contact_kappa, c.gait_duty

        def cdf(x):
            return 1.0 / (1.0 + np.exp(-x / k))         # logistic ≈ smooth step

        def stance(p):
            p = np.mod(p, 1.0)
            return (cdf(p) * (1.0 - cdf(p - duty))
                    + cdf(p - 1.0) * (1.0 - cdf(p - 1.0 - duty)))

        left = stance(self._gait_phase)
        right = stance(np.mod(self._gait_phase + 0.5, 1.0))   # antiphase
        return np.stack([left, right], axis=1).astype(np.float32)

    # ── AMP discriminator features ─────────────────────────────────

    def amp_observation(self) -> np.ndarray:
        """AMP feature view of the CURRENT state, in the SAME layout as the
        parametric reference (training.algorithms.amp.build_amp_obs):
        [root_height, projected_gravity(3), body ang vel(3),
        ABSOLUTE joint pos(22), joint vel(22)] = 51. Root LINEAR velocity is
        omitted so the discriminator judges gait STYLE, not forward speed."""
        from training.algorithms.amp import build_amp_obs, AMP_OBS_DIM
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            root_ang = _to_np(self.robot.get_ang())
            jpos = _to_np(self.robot.get_dofs_position(self.dof_indices))
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
        except Exception:
            return np.zeros((self.num_envs, AMP_OBS_DIM), dtype=np.float32)
        # Per-foot CLEARANCE above ground (Genesis foot-link standing z ≈0.065).
        # Relative to standing so it aligns with the reference's MuJoCo-FK
        # clearance despite the sims' differing foot-link-frame offsets.
        if self._foot_links is None:
            self._ensure_foot_links()
        if self._foot_links is not None:
            lz = _to_np(self._foot_links[0].get_pos())[:, 2]
            rz = _to_np(self._foot_links[1].get_pos())[:, 2]
            foot_clear = np.clip(np.stack([lz, rz], 1) - _FOOT_GROUND_Z,
                                 0.0, 0.5).astype(np.float32)
        else:
            foot_clear = np.zeros((self.num_envs, 2), np.float32)
        return build_amp_obs(
            root_pos[:, 2], projected_gravity(root_quat),
            body_frame_velocity(root_quat, root_ang), jpos, jvel, foot_clear)

    # ── reward ─────────────────────────────────────────────────────

    def _compute_skill_reward(self, action: np.ndarray):
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            root_lin_vel = _to_np(self.robot.get_vel())
            root_ang_vel = _to_np(self.robot.get_ang())
            jpos = _to_np(self.robot.get_dofs_position(self.dof_indices))
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32), {}

        try:
            applied_torque = _to_np(
                self.robot.get_dofs_force(self.dof_indices))
        except Exception:
            applied_torque = np.zeros_like(jvel)

        (feet_z, contact, foot_horiz_speed,
         air_time, contact_just_now) = self._read_foot_state()
        desired_contact = self._gait_desired_contact()

        reward, components = compute_walk_reward(
            root_pos=root_pos, root_quat=root_quat,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            jpos=jpos, jvel=jvel, prev_jvel=self._prev_jvel,
            action=action, prev_action=self._last_action,
            applied_torque=applied_torque,
            feet_z=feet_z, contact_mask=contact,
            desired_contact=desired_contact,
            foot_horiz_speed=foot_horiz_speed,
            air_time=air_time,
            contact_just_now=contact_just_now,
            gait_phase=self._gait_phase,
            commands=self.commands,
            weights=self.cfg.rewards,
            head_commands=self.head_commands,
            head_joint_indices=self.robot_cfg.head_joint_indices,
            arm_joint_indices=self.robot_cfg.arm_joint_indices,
            shoulder_pitch_indices=self._shoulder_pitch_indices,
            default_joint_pos=self._default_action,
            dt=self.dt,
        )

        # ── speed-curriculum bookkeeping ──
        self._total_env_steps_seen += self.num_envs
        track = float(components.get("track_lin_vel", 0.0))
        self._track_ema = ((1.0 - self._track_ema_alpha) * self._track_ema
                           + self._track_ema_alpha * track)
        # accrue progress (→ widen the speed cap) ONLY while tracking is good
        if self._track_ema >= self.cfg.speed_curriculum_min_track:
            self._speed_prog += self.num_envs
        components["vx_max_current"] = self._current_vx_max()
        components["track_ema"] = self._track_ema
        # yaw curriculum: widen the turning cap only while yaw tracking is good
        ang = float(components.get("track_ang_vel", 0.0))
        self._ang_ema = ((1.0 - self._track_ema_alpha) * self._ang_ema
                         + self._track_ema_alpha * ang)
        if self._ang_ema >= self.cfg.speed_curriculum_min_track:
            self._yaw_prog += self.num_envs
        components["vyaw_max_current"] = self._current_vyaw_max()
        components["ang_ema"] = self._ang_ema

        self._prev_jvel = jvel
        return reward, components

    # ── reset hook: also reset the per-env reward state ───────────

    def _reset_skill_state(self, envs_idx: np.ndarray) -> None:
        self._prev_jvel[envs_idx] = 0.0
        self._prev_contact[envs_idx] = False
        self._air_time[envs_idx] = 0.0
        # Re-randomise the gait phase for reset envs so they don't all march in
        # lockstep. `_prev_foot_xy` is left as-is (full-array finite diff); the
        # first post-reset step's slip term sees a one-step transient, harmless.
        self._gait_phase[envs_idx] = self.rng.uniform(
            0.0, 1.0, envs_idx.shape[0]).astype(np.float32)
