"""Shoot skill env — kick the ball toward a commanded target.

Subclasses `SkillEnv`:
  * 3-dim command vector: [aim_angle, power, foot_pref].
  * Ball entity (reused from dribble pattern).
  * Per-env world-frame target sampled at reset from the aim_angle
    command (in front of the robot at `target_distance_for_aim`).
  * 9-dim obs add-ons: ball pos body (3) + ball vel body (3) +
    target-relative pos body (3).
  * Episode terminates on kick detected, ball lost, fall, or timeout.

Single-skill training: every reset samples a new aim_angle + power, so
the policy sees the full command distribution. The orchestrator (Phase
2) overrides aim_angle / power at runtime to drive shot direction.
"""

from __future__ import annotations

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

from skills.base import CommandSpec, SkillEnv
from skills.common_obs import _to_np, body_frame_velocity
from skills.shoot.config import ShootConfig
from skills.shoot.rewards import compute_shoot_reward


class K1ShootEnv(SkillEnv):

    SKILL_NAME = "shoot"
    SKILL_OBS_ADDONS = 9    # ball pos body (3) + ball vel body (3) + target body (3)
    FALL_TERMINATE_Z = 0.20

    def __init__(self, cfg: ShootConfig = None, **kwargs):
        self.cfg = cfg or ShootConfig()
        kwargs.setdefault("num_envs", self.cfg.num_envs)
        kwargs.setdefault("dt", self.cfg.dt)
        kwargs.setdefault("sim_dt", self.cfg.sim_dt)
        kwargs.setdefault("gait_freq_hz", self.cfg.gait_freq_hz)
        super().__init__(**kwargs)
        self.MAX_EPISODE_STEPS = self.cfg.max_episode_steps

        # Per-env state
        self._prev_jvel = np.zeros((self.num_envs, self.act_dim),
                                   dtype=np.float32)
        self.ball = None
        # World-frame target sampled at reset (each env can have a
        # different aim).
        self._target_world = np.zeros((self.num_envs, 3), dtype=np.float32)
        # Track whether this env has already collected the kick bonus
        # this episode — prevents double-paying when the ball is still
        # fast on the step after the trigger.
        self._already_kicked = np.zeros(self.num_envs, dtype=bool)
        self._kick_event = np.zeros(self.num_envs, dtype=bool)
        self._ball_lost = np.zeros(self.num_envs, dtype=bool)

        self.cfg.obs_dim = self.obs_dim
        self.cfg.act_dim = self.act_dim

    # ── command spec ──────────────────────────────────────────────

    def _make_command_spec(self) -> CommandSpec:
        c = self.cfg if hasattr(self, "cfg") and self.cfg is not None else ShootConfig()
        return CommandSpec(
            dim=3,
            low=np.array([c.aim_angle_range[0], c.power_range[0],
                          c.foot_range[0]], dtype=np.float32),
            high=np.array([c.aim_angle_range[1], c.power_range[1],
                           c.foot_range[1]], dtype=np.float32),
            names=("aim_angle", "power", "foot_pref"),
        )

    # ── scene extras: ball ────────────────────────────────────────

    def _add_scene_extras(self, scene) -> None:
        if gs is None:
            return
        self.ball = scene.add_entity(
            gs.morphs.Sphere(radius=self.cfg.ball_radius,
                             pos=(0.3, 0.0, self.cfg.ball_radius),
                             collision=True),
        )

    # ── reset: place ball + sample world target from aim_angle ────

    def _reset_skill_state(self, envs_idx: np.ndarray) -> None:
        self._prev_jvel[envs_idx] = 0.0
        self._already_kicked[envs_idx] = False
        self._kick_event[envs_idx] = False
        self._ball_lost[envs_idx] = False

        n = envs_idx.shape[0]

        # Ball: in front of the robot, slightly randomized.
        if self.ball is not None:
            bpos = np.zeros((n, 3), dtype=np.float32)
            bpos[:, 0] = self.rng.uniform(*self.cfg.ball_spawn_range_x, size=n)
            bpos[:, 1] = self.rng.uniform(*self.cfg.ball_spawn_range_y, size=n)
            bpos[:, 2] = self.cfg.ball_radius
            try:
                self.ball.set_pos(bpos, envs_idx=envs_idx)
                try:
                    self.ball.zero_all_dofs_velocity(envs_idx=envs_idx)
                except Exception:
                    pass
            except Exception as e:
                print(f"[shoot] ball reset failed: {e}")

        # Target: convert command's aim_angle into a world-frame target
        # in front of the robot. Robot starts at world origin facing +x,
        # so body forward == +x world.
        aim = self.commands[envs_idx, 0]
        d = float(self.cfg.target_distance_for_aim)
        tgt = np.zeros((n, 3), dtype=np.float32)
        tgt[:, 0] = d * np.cos(aim)
        tgt[:, 1] = d * np.sin(aim)
        tgt[:, 2] = self.cfg.goal_z
        # Clamp to goal mouth so the target is actually a valid shot.
        # We allow some slack outside the mouth for variety but cap at
        # the goal width.
        tgt[:, 1] = np.clip(tgt[:, 1], -self.cfg.goal_half_width,
                            self.cfg.goal_half_width)
        self._target_world[envs_idx] = tgt

    # ── obs add-ons ───────────────────────────────────────────────

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
        ball_rel = bpos - root_pos
        target_rel = self._target_world - root_pos
        ball_pos_body = body_frame_velocity(root_quat, ball_rel)
        ball_vel_body = body_frame_velocity(root_quat, bvel)
        target_body = body_frame_velocity(root_quat, target_rel)
        return np.concatenate(
            [ball_pos_body, ball_vel_body, target_body], axis=1
        ).astype(np.float32)

    # ── reward ────────────────────────────────────────────────────

    def _compute_skill_reward(self, action: np.ndarray):
        try:
            root_pos = _to_np(self.robot.get_pos())
            root_quat = _to_np(self.robot.get_quat())
            root_lin_vel = _to_np(self.robot.get_vel())
            root_ang_vel = _to_np(self.robot.get_ang())
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
            bpos = _to_np(self.ball.get_pos())
            bvel = _to_np(self.ball.get_vel())
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32), {}

        reward, kick_now, lost, components = compute_shoot_reward(
            root_pos=root_pos, root_quat=root_quat,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            jvel=jvel, prev_jvel=self._prev_jvel,
            action=action, prev_action=self._last_action,
            ball_pos=bpos, ball_vel=bvel,
            target_world=self._target_world,
            commands=self.commands,
            weights=self.cfg.rewards,
            kick_speed_threshold=self.cfg.kick_speed_threshold,
            ball_lost_distance=self.cfg.ball_lost_distance,
            dt=self.dt,
            already_kicked=self._already_kicked,
        )

        # Latch already_kicked so the bonus only pays once. We still
        # terminate on kick_event (see _check_skill_done), so this is
        # mostly defensive — handles the case where multiple kicks
        # happen across step boundaries before reset.
        self._already_kicked = self._already_kicked | kick_now
        self._kick_event = kick_now
        self._ball_lost = lost
        self._prev_jvel = jvel
        return reward, components

    # ── termination: kick success / ball lost / fall / timeout ────

    def _check_skill_done(self) -> np.ndarray:
        base_done = super()._check_skill_done()  # timeout + fall
        return base_done | self._kick_event | self._ball_lost
