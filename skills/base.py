"""Skill-library base classes.

Every locomotion / soccer skill (`standup`, `walk`, `dribble`, `shoot`)
inherits from `SkillEnv` and exposes a `CommandSpec` describing its
continuous command vector. The orchestrator (Phase 2) consumes these
specs to know which command dims to emit for each skill.

Design contract:

  * SkillEnv subclasses set `SKILL_NAME`, `SKILL_OBS_ADDONS`, and
    `MAX_EPISODE_STEPS`. They override `_make_command_spec`,
    `_add_scene_extras`, `_reset_skill_state`, `_compute_skill_reward`,
    and optionally `_skill_obs_addons` / `_check_skill_done`.
  * The base class handles Genesis init, robot + camera setup, common
    obs, command resampling on episode reset, auto-reset of finished
    envs, and PD-controlled action application.
  * `obs_dim = SKILL_BASE_OBS_DIM + command_spec.dim + SKILL_OBS_ADDONS`.

Each skill trains in its own Python process so Genesis's GPU memory
(field+kernels) is bounded per-skill and doesn't leak across stages —
the OOM that drove this refactor.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

from configs.config import K1RobotConfig
from skills.common_obs import (SKILL_BASE_OBS_DIM, compute_common_obs,
                                read_robot_state)


# ─── command spec ──────────────────────────────────────────────────────


@dataclass
class CommandSpec:
    """A continuous command vector exposed by a skill.

    The orchestrator emits a 7-dim padded command and slices the first
    `dim` entries for the active skill. `names` is for human-readable
    logging; `low`/`high` bound the uniform sampler during single-skill
    training.
    """
    dim: int
    low: np.ndarray
    high: np.ndarray
    names: Tuple[str, ...]

    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)
        assert self.low.shape == (self.dim,), f"low {self.low.shape}"
        assert self.high.shape == (self.dim,), f"high {self.high.shape}"
        assert len(self.names) == self.dim

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform sample in [low, high]. Returns (n, dim)."""
        if self.dim == 0:
            return np.zeros((n, 0), dtype=np.float32)
        return rng.uniform(self.low, self.high,
                           size=(n, self.dim)).astype(np.float32)

    @classmethod
    def empty(cls) -> "CommandSpec":
        return cls(dim=0, low=np.zeros(0), high=np.zeros(0), names=())


# ─── skill env base ────────────────────────────────────────────────────


def _to_np(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


class SkillEnv(ABC):
    """Vectorised Genesis env for a single skill.

    Concrete skills override the hooks below. Common machinery (scene
    build, obs assembly, reset/step plumbing, PD control, video render)
    lives in this class.
    """

    SKILL_NAME: str = "skill"
    SKILL_OBS_ADDONS: int = 0
    MAX_EPISODE_STEPS: int = 1000

    # Episode-termination thresholds (override in subclasses if needed).
    FALL_TERMINATE_Z: float = 0.10   # trunk below this → done
    FALL_RECOVERY_Z: float = 0.30    # below this → "fallen" but maybe not done

    def __init__(
        self,
        num_envs: int = 1024,
        robot_cfg: Optional[K1RobotConfig] = None,
        render: bool = False,
        dt: float = 0.02,             # control timestep
        sim_dt: float = 0.002,        # physics timestep
        env_spacing: Tuple[float, float] = (14.0, 11.0),
        seed: int = 0,
        gait_freq_hz: float = 1.5,
        backend: str = "gpu",
    ):
        self.num_envs = int(num_envs)
        self.robot_cfg = robot_cfg or K1RobotConfig()
        self.render = bool(render)
        self.dt = float(dt)
        self.sim_dt = float(sim_dt)
        self.action_repeat = max(1, int(round(self.dt / self.sim_dt)))
        self.env_spacing = env_spacing
        self.gait_freq_hz = float(gait_freq_hz)
        self.backend = backend

        self.scene = None
        self.robot = None
        self.camera = None
        self.dof_indices: list = []
        self._initialized = False

        # Per-env state (numpy)
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self._default_action = np.asarray(self.robot_cfg.default_joint_pos,
                                          dtype=np.float32)
        self._last_action = np.tile(self._default_action,
                                    (self.num_envs, 1))
        self._episode_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

        # Command vector — populated after _make_command_spec is called
        self.command_spec: CommandSpec = self._make_command_spec()
        self.commands = np.zeros((self.num_envs, self.command_spec.dim),
                                 dtype=np.float32)

    # ── shape properties ───────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return (SKILL_BASE_OBS_DIM
                + self.command_spec.dim
                + self.SKILL_OBS_ADDONS)

    @property
    def act_dim(self) -> int:
        return int(self.robot_cfg.num_dofs)

    # ── overrides for subclasses ───────────────────────────────────────

    @abstractmethod
    def _make_command_spec(self) -> CommandSpec:
        """Return the command vector for this skill (use CommandSpec.empty()
        for no command)."""

    def _add_scene_extras(self, scene) -> None:
        """Optional: add ball, obstacles, etc. Called once during scene
        build, before `scene.build()`. Default: nothing."""
        return

    def _reset_skill_state(self, envs_idx: np.ndarray) -> None:
        """Optional: per-env init beyond robot pose (e.g. ball placement,
        target sampling). Called from `reset()` after the robot pose is
        set. Default: nothing."""
        return

    @abstractmethod
    def _compute_skill_reward(self, action: np.ndarray
                              ) -> Tuple[np.ndarray, dict]:
        """Return (reward[N] float32, components_dict).

        Components are batch-averaged scalars used for logging.
        """

    def _check_skill_done(self) -> np.ndarray:
        """Default termination: timeout + trunk fell below FALL_TERMINATE_Z."""
        pos = _to_np(self.robot.get_pos())
        timeout = self.step_count >= self.MAX_EPISODE_STEPS
        fallen = pos[:, 2] < self.FALL_TERMINATE_Z
        return (timeout | fallen)

    def _skill_obs_addons(self) -> np.ndarray:
        """Override to append (N, SKILL_OBS_ADDONS) extras after the
        common obs + command. Default: zeros."""
        return np.zeros((self.num_envs, self.SKILL_OBS_ADDONS),
                        dtype=np.float32)

    def _reset_robot_pose(self, envs_idx: np.ndarray) -> None:
        """Default: spawn robot upright at ~0.65m with the default joint
        pose. Subclasses (e.g. standup) override to spawn fallen poses.
        """
        n = envs_idx.shape[0]
        pos = np.zeros((n, 3), dtype=np.float32)
        pos[:, 2] = 0.65
        quat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
        try:
            self.robot.set_pos(pos, envs_idx=envs_idx)
            self.robot.set_quat(quat, envs_idx=envs_idx)
            targets = np.tile(self._default_action, (n, 1))
            self.robot.set_dofs_position(targets, self.dof_indices,
                                         envs_idx=envs_idx,
                                         zero_velocity=True)
        except Exception as e:
            print(f"[{self.SKILL_NAME}] _reset_robot_pose failed: {e}")

    # ── Genesis scene setup ────────────────────────────────────────────

    def _init_genesis(self) -> None:
        if self._initialized or gs is None:
            return

        backend = gs.gpu if self.backend == "gpu" else gs.cpu
        try:
            gs.init(backend=backend, precision="32",
                    logging_level="warning", seed=int(self.rng.integers(1<<30)),
                    performance_mode=True)
        except Exception:
            pass  # already initialized in this process

        self.scene = gs.Scene(
            show_viewer=self.render,
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=2),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                ambient_light=(0.4, 0.4, 0.4),
            ),
        )

        # Field (physics-only to keep replicated-body count down).
        try:
            from models.field.field_genesis_builder import build_soccer_field
            build_soccer_field(self.scene, physics_only=True)
        except Exception as e:
            print(f"[{self.SKILL_NAME}] field builder failed ({e}); "
                  "falling back to plain green plane")
            self.scene.add_entity(
                gs.morphs.Plane(),
                surface=gs.surfaces.Default(color=(0.10, 0.55, 0.10, 1.0),
                                            roughness=0.9),
            )

        # Robot (URDF — Genesis path).
        urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "robot", "K1",
            "K1_22dof.urdf",
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, pos=(0, 0, 0.65),
                           merge_fixed_links=True),
        )

        # Skill-specific entities (ball, targets, …)
        self._add_scene_extras(self.scene)

        # Single static camera following env 0.
        try:
            self.camera = self.scene.add_camera(
                res=(640, 480), pos=(0, -6, 4),
                lookat=(0, 0, 0.5), fov=50,
            )
        except Exception as e:
            print(f"[{self.SKILL_NAME}] camera setup failed: {e}")

        self.scene.build(n_envs=self.num_envs,
                         env_spacing=self.env_spacing,
                         center_envs_at_origin=False)
        self._setup_joint_mapping()

        # PD gains
        n = len(self.dof_indices)
        try:
            self.robot.set_dofs_kp([float(self.robot_cfg.kp)] * n,
                                   self.dof_indices)
            self.robot.set_dofs_kv([float(self.robot_cfg.kd)] * n,
                                   self.dof_indices)
        except Exception:
            pass

        self._initialized = True

    def _setup_joint_mapping(self) -> None:
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
            self.dof_indices = list(
                range(6, 6 + self.robot_cfg.num_dofs))

    # ── reset / step ───────────────────────────────────────────────────

    def reset(self, envs_idx: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._initialized:
            self._init_genesis()

        if envs_idx is None:
            envs_idx = np.arange(self.num_envs)
        envs_idx = np.asarray(envs_idx, dtype=np.int64)

        self.step_count[envs_idx] = 0
        self._episode_reward[envs_idx] = 0.0
        self._last_action[envs_idx] = self._default_action

        if self.command_spec.dim > 0:
            self.commands[envs_idx] = self.command_spec.sample(
                len(envs_idx), self.rng)

        self._reset_robot_pose(envs_idx)
        self._reset_skill_state(envs_idx)

        return self._get_obs()

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :].repeat(self.num_envs, axis=0)
        action = np.clip(action, -math.pi, math.pi)

        try:
            self.robot.control_dofs_position(action, self.dof_indices)
        except Exception:
            pass

        for _ in range(self.action_repeat):
            self.scene.step()

        self.step_count += 1

        obs = self._get_obs()
        reward, components = self._compute_skill_reward(action)
        done = self._check_skill_done()

        self._episode_reward += reward
        self._last_action = action

        if done.any():
            self.reset(envs_idx=np.where(done)[0])

        info = {
            "episode_reward": self._episode_reward.copy(),
            "skill": self.SKILL_NAME,
            "reward_components": components,
        }
        return obs, reward, done, info

    # ── obs assembly ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        try:
            (root_pos, root_quat, root_lin_vel, root_ang_vel,
             jpos, jvel) = read_robot_state(self.robot, self.dof_indices)
        except Exception:
            return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        base = compute_common_obs(
            root_pos=root_pos, root_quat=root_quat,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            joint_pos=jpos, joint_vel=jvel,
            last_action=self._last_action,
            step_count=self.step_count,
            default_joint_pos=self._default_action,
            control_dt=self.dt,
            gait_freq_hz=self.gait_freq_hz,
        )
        parts = [base]
        if self.command_spec.dim > 0:
            parts.append(self.commands.astype(np.float32))
        if self.SKILL_OBS_ADDONS > 0:
            parts.append(self._skill_obs_addons())
        return np.concatenate(parts, axis=1)

    # ── rendering / cleanup ────────────────────────────────────────────

    def render_frame(self):
        if self.camera is None:
            return None
        try:
            out = self.camera.render()
            rgb = out[0] if isinstance(out, tuple) else out
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            return rgb
        except Exception as e:
            print(f"[{self.SKILL_NAME}] render_frame error: {e}")
            return None

    def close(self) -> None:
        """Drop all Genesis-owned GPU references and flush CUDA cache."""
        self.camera = None
        self.robot = None
        self.scene = None
        self._initialized = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass


# ─── policy convenience ────────────────────────────────────────────────


@dataclass
class SkillSpec:
    """Lightweight descriptor used by the orchestrator (Phase 2) to know
    the obs/act dims and command spec of each skill without importing
    the heavy SkillEnv subclass."""
    name: str
    obs_dim: int
    act_dim: int
    command_spec: CommandSpec
    checkpoint_path: str = ""


__all__ = [
    "SKILL_BASE_OBS_DIM",
    "CommandSpec",
    "SkillEnv",
    "SkillSpec",
]
