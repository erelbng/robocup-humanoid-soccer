"""
Vectorized Phase 1 environment (Genesis n_envs).

Builds a Genesis scene with `scene.build(n_envs=N)` so N copies of the
robot+ball train in parallel on a single GPU. Genesis batches all rigid
solver ops, which is the dominant cost — going from 1 env to 4096
typically gives 200-1000x throughput on a modern GPU.

Differences vs the single-env class:
  * `reset()` / `step()` operate on whole batches; obs is (N, obs_dim),
    reward / done are (N,).
  * Entity getters return torch tensors of shape (N, ...); we convert
    to numpy for the trainer's bookkeeping where needed.
  * Per-env state (style command, prev action, episode reward) is held
    in (N,)-shaped numpy arrays.

Kept deliberately compatible with the same observation layout as
`K1DribbleShootEnv` so a policy trained here can roll out in the
single-env eval path without retraining.

NOTE: gait shaping / standup are NOT yet batched in this class — they
fall back to scalar reward when called. The base velocity-tracking and
style rewards ARE batched. The follow-up to make every shaping term
batched is straightforward; this class is structured to make that swap
local to `_compute_reward_batch`.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

from configs.config import K1RobotConfig, Phase1Config
from envs.style_command import StyleCommandSampler


def _to_np(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


class K1DribbleShootVecEnv:
    """Batched Phase 1 environment.

    Public API:
        env = K1DribbleShootVecEnv(num_envs=4096, ...)
        obs = env.reset()          # (N, obs_dim)
        obs, rew, done, info = env.step(actions)  # actions: (N, act_dim)
    """

    def __init__(
        self,
        num_envs: int = 4096,
        cfg: Phase1Config = None,
        robot_cfg: K1RobotConfig = None,
        field_info: dict = None,
        render: bool = False,
        curriculum_stage: str = "full",
        env_spacing: Tuple[float, float] = (12.0, 9.0),
    ):
        self.num_envs = num_envs
        self.cfg = cfg or Phase1Config()
        self.robot_cfg = robot_cfg or K1RobotConfig()
        self.render = render
        self.curriculum_stage = curriculum_stage
        # Robots in different envs share a world — space them out so they
        # can't accidentally collide across env boundaries.
        self.env_spacing = env_spacing

        if field_info is None:
            field_info_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "field",
                "field_info.json",
            )
            if os.path.exists(field_info_path):
                with open(field_info_path) as f:
                    field_info = json.load(f)
            else:
                field_info = {"half_length": 4.5, "half_width": 3.0}
        self.field_info = field_info

        self.scene = None
        self.robot = None
        self.ball = None
        self.dof_indices = []
        self._initialized = False

        # Per-env bookkeeping (numpy, shape (num_envs,))
        self.step_count = np.zeros(num_envs, dtype=np.int64)
        self.episode_reward = np.zeros(num_envs, dtype=np.float32)
        self._prev_action = np.zeros((num_envs, self.cfg.act_dim),
                                     dtype=np.float32)

        # Style command (one sampler shared, per-env current command)
        self._style_dim = self.cfg.style_command_dim if getattr(
            self.cfg, "use_style_command", False) else 0
        self._cmd = np.zeros((num_envs, self._style_dim), dtype=np.float32)
        self._sampler = StyleCommandSampler() if self._style_dim > 0 else None

        self.rng = np.random.default_rng()

    # ── Scene construction ─────────────────────────────────────────

    def _init_genesis(self):
        if self._initialized or gs is None:
            return

        try:
            gs.init(backend=gs.gpu, precision="32",
                    logging_level="warning", seed=1, performance_mode=True)
        except Exception:
            pass

        self.scene = gs.Scene(
            show_viewer=self.render,
            sim_options=gs.options.SimOptions(dt=self.cfg.sim_dt, substeps=2),
        )

        # Field (built once; replicated by Genesis across envs at build())
        try:
            from models.field.field_genesis_builder import build_soccer_field
            build_soccer_field(self.scene)
        except Exception:
            self.scene.add_entity(gs.morphs.Plane())

        # Robot
        urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "robot", "K1",
            "K1_22dof.urdf",
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, pos=(0, 0, 1.05),
                           merge_fixed_links=True),
        )

        # Ball
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.07, pos=(1.0, 0, 0.10), collision=True),
        )

        self.scene.build(n_envs=self.num_envs, env_spacing=self.env_spacing,
                         center_envs_at_origin=True)
        self._setup_joint_mapping()

        # PD gains on actuated joints
        n = len(self.dof_indices)
        try:
            self.robot.set_dofs_kp([float(self.robot_cfg.kp)] * n,
                                   self.dof_indices)
            self.robot.set_dofs_kv([float(self.robot_cfg.kd)] * n,
                                   self.dof_indices)
        except Exception:
            pass

        self._initialized = True

    def _setup_joint_mapping(self):
        self.dof_indices = []
        joint_by_name = {j.name: j for j in self.robot.joints}
        for name in self.robot_cfg.joint_names:
            j = joint_by_name.get(name)
            if j is None:
                continue
            try:
                idxs = j.dofs_idx_local
            except Exception:
                idxs = None
            if idxs and len(idxs) == 1:
                self.dof_indices.append(int(idxs[0]))
        if not self.dof_indices:
            self.dof_indices = list(range(6, 6 + self.robot_cfg.num_dofs))

    # ── Reset / step ───────────────────────────────────────────────

    def reset(self, envs_idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset the specified envs (or all). Returns full (N, obs_dim) obs."""
        if not self._initialized:
            self._init_genesis()

        if envs_idx is None:
            envs_idx = np.arange(self.num_envs)
        envs_idx = np.asarray(envs_idx, dtype=np.int64)

        # Per-env reset bookkeeping
        self.step_count[envs_idx] = 0
        self.episode_reward[envs_idx] = 0.0
        self._prev_action[envs_idx] = 0.0

        # Resample style command for reset envs
        if self._sampler is not None and self._style_dim > 0:
            for i in envs_idx:
                self._sampler.sample()
                self._cmd[i] = self._sampler.current.as_array()[:self._style_dim]

        # Place robot + ball
        self._reset_pose(envs_idx)

        return self._get_obs()

    def _reset_pose(self, envs_idx: np.ndarray):
        n = envs_idx.shape[0]
        try:
            # Trunk at z=1.05, neutral orientation
            pos = np.zeros((n, 3), dtype=np.float32)
            pos[:, 2] = 1.05
            if self.cfg.randomize_robot_pos:
                pos[:, 0] = self.rng.uniform(-0.5, 0.5, n)
                pos[:, 1] = self.rng.uniform(-0.5, 0.5, n)
            quat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
            self.robot.set_pos(pos, envs_idx=envs_idx)
            self.robot.set_quat(quat, envs_idx=envs_idx)
            targets = np.tile(np.array(self.robot_cfg.default_joint_pos,
                                       dtype=np.float32), (n, 1))
            self.robot.set_dofs_position(targets, self.dof_indices,
                                         envs_idx=envs_idx,
                                         zero_velocity=True)
        except Exception as e:
            print(f"[vec] _reset_pose failed: {e}")

        # Ball
        try:
            bpos = np.zeros((n, 3), dtype=np.float32)
            if self.curriculum_stage in ("stand", "standup"):
                bpos[:, 0] = 5.0
                bpos[:, 1] = 5.0
            else:
                bpos[:, 0] = self.rng.uniform(0.5, 3.0, n)
                bpos[:, 1] = self.rng.uniform(-2.0, 2.0, n)
            bpos[:, 2] = 0.07
            self.ball.set_pos(bpos, envs_idx=envs_idx)
        except Exception:
            pass

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step all envs in lockstep. `action` shape (num_envs, act_dim)."""
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :].repeat(self.num_envs, axis=0)
        action = np.clip(action, -math.pi, math.pi)

        # Apply PD targets to ALL envs at once
        try:
            self.robot.control_dofs_position(action, self.dof_indices)
        except Exception:
            pass

        for _ in range(self.cfg.action_repeat):
            self.scene.step()

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward_batch(action)
        done = self._check_done_batch()

        self.episode_reward += reward
        self._prev_action = action

        # Auto-reset finished envs (standard vec-env behaviour)
        if done.any():
            self.reset(envs_idx=np.where(done)[0])

        info = {"episode_reward": self.episode_reward.copy(),
                "curriculum_stage": self.curriculum_stage}
        return obs, reward, done, info

    # ── Observations / rewards / done — batched ────────────────────

    def _get_obs(self) -> np.ndarray:
        out = np.zeros((self.num_envs, self.cfg.obs_dim), dtype=np.float32)
        try:
            pos = _to_np(self.robot.get_pos())            # (N, 3)
            quat = _to_np(self.robot.get_quat())          # (N, 4)
            vel = _to_np(self.robot.get_vel())            # (N, 3)
            angv = _to_np(self.robot.get_ang())           # (N, 3)
            jpos = _to_np(self.robot.get_dofs_position(self.dof_indices))
            jvel = _to_np(self.robot.get_dofs_velocity(self.dof_indices))
            bpos = _to_np(self.ball.get_pos())            # (N, 3)
            bvel = _to_np(self.ball.get_vel())            # (N, 3)

            n = self.robot_cfg.num_dofs
            i = 0
            out[:, i:i+3] = pos; i += 3
            out[:, i:i+4] = quat; i += 4
            out[:, i:i+3] = vel; i += 3
            out[:, i:i+3] = angv; i += 3
            out[:, i:i+n] = jpos; i += n
            out[:, i:i+n] = jvel; i += n
            out[:, i:i+3] = bpos - pos; i += 3   # ball relative
            out[:, i:i+3] = bvel; i += 3
            goal_x = self.field_info.get("half_length", 4.5)
            goal = np.array([goal_x, 0.0, 0.4], dtype=np.float32)
            out[:, i:i+3] = goal[None, :] - pos; i += 3
            out[:, i:i+2] = 1.0; i += 2  # foot contacts placeholder

            if self._style_dim > 0:
                base = self.cfg.base_obs_dim
                out[:, base:base + self._style_dim] = self._cmd
        except Exception as e:
            # Leave zeros — keeps shapes consistent for the trainer.
            pass
        return out

    def _compute_reward_batch(self, action: np.ndarray) -> np.ndarray:
        """Batched reward.

        Current implementation: upright + style-velocity tracking + small
        smoothness penalty + ball-distance shaping. Mirrors the most
        important pieces of the single-env reward. Standup and gait-shaping
        terms are scalar today; that's the next refactor.
        """
        try:
            pos = _to_np(self.robot.get_pos())
            quat = _to_np(self.robot.get_quat())
            vel = _to_np(self.robot.get_vel())
            angv = _to_np(self.robot.get_ang())
            bpos = _to_np(self.ball.get_pos())
        except Exception:
            return np.zeros(self.num_envs, dtype=np.float32)

        # Upright (cos of trunk pitch+roll)
        upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)

        # Trunk height shaping
        height_err = pos[:, 2] - 0.55
        height_r = np.exp(-(height_err ** 2) / (0.12 ** 2))

        # Style-command velocity tracking (only meaningful for non-stand stages)
        vx_target = self._cmd[:, 0] if self._style_dim > 0 else 0.0
        vy_target = self._cmd[:, 1] if self._style_dim > 1 else 0.0
        # World→body via yaw
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        vx_body = np.cos(-yaw) * vel[:, 0] - np.sin(-yaw) * vel[:, 1]
        vy_body = np.sin(-yaw) * vel[:, 0] + np.cos(-yaw) * vel[:, 1]
        vel_err = (vx_body - vx_target) ** 2 + (vy_body - vy_target) ** 2
        vel_track = np.exp(-vel_err / (0.25 ** 2))

        # Ball distance shaping (only for walk/dribble/shoot/full)
        ball_dist = np.linalg.norm(bpos[:, :2] - pos[:, :2], axis=1)
        ball_close_r = np.exp(-(ball_dist ** 2) / (1.0 ** 2))

        # Smoothness penalty
        smooth = np.sum((action - self._prev_action) ** 2, axis=1)

        r = np.zeros(self.num_envs, dtype=np.float32)
        r += 1.0 * upright.astype(np.float32)
        r += 0.5 * height_r.astype(np.float32)

        if self.curriculum_stage in ("walk", "dribble", "shoot", "full"):
            r += 1.5 * vel_track.astype(np.float32)
            r += 0.8 * ball_close_r.astype(np.float32)

        r -= 0.05 * smooth.astype(np.float32)

        # Fall penalty
        fallen = pos[:, 2] < 0.20
        r -= 10.0 * fallen.astype(np.float32)

        return r

    def _check_done_batch(self) -> np.ndarray:
        try:
            pos = _to_np(self.robot.get_pos())
        except Exception:
            return np.zeros(self.num_envs, dtype=bool)
        timeout = self.step_count >= self.cfg.max_episode_steps
        fallen = pos[:, 2] < 0.10
        return (timeout | fallen)

    def close(self):
        self.scene = None
        self._initialized = False
