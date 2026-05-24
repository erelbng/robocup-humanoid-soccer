"""
Domain randomization for sim2real transfer.

Designed to be called from env.reset() with the entities + scene already
built. The main class `MotorRandomizer` perturbs ALL the per-joint
parameters that vary between identical hardware units in practice:

  * PD gains (kp, kv) — motor controller settings drift unit-to-unit
  * Joint damping (kinetic friction in the gearbox)
  * Joint static friction (`frictionloss`)
  * Joint armature (effective rotor inertia)
  * Torque limit (`force_range`) — hot motors can saturate earlier
  * Action latency — controller-to-actuator delay (a small ring buffer)

Plus the existing perturbations on robot/ball masses and material
friction. Each parameter is sampled at reset from a per-parameter
multiplicative range around a baseline captured at attach time.

Usage:
    dr = MotorRandomizer(rng=np.random.default_rng(0))
    env.scene.build()
    dr.attach(robot=env.robot, dof_indices=env.dof_indices,
              ball=env.ball)
    obs = env.reset()
    dr.randomize_episode()           # called every reset
    # In env.step:
    delayed_action = dr.delay_action(raw_action)
    robot.control_dofs_position(delayed_action, env.dof_indices)

What's stubbed:
  * Visual texture swaps (rebuild required) — disabled by default.
  * Per-link CoM offset / mass distribution — needs a separate API path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np


# ─── Parameter ranges ──────────────────────────────────────────────────


@dataclass
class MotorRanges:
    """Per-parameter randomisation ranges for the motors. All values are
    multiplicative around the baseline captured at `attach()` time, except
    `action_delay_steps` which is absolute (in control-rate steps)."""

    # PD gains (per joint, independently sampled)
    kp_mult: Tuple[float, float] = (0.80, 1.20)
    kv_mult: Tuple[float, float] = (0.70, 1.30)

    # Passive dynamics
    damping_mult: Tuple[float, float] = (0.5, 1.5)
    friction_loss_mult: Tuple[float, float] = (0.5, 1.5)
    armature_mult: Tuple[float, float] = (0.8, 1.2)

    # Torque saturation
    torque_limit_mult: Tuple[float, float] = (0.80, 1.10)

    # Action latency (in CONTROL steps, i.e. multiples of cfg.dt). The
    # delay is sampled per episode and applied uniformly across all
    # joints. 0 = no delay; 2 = ~40 ms at 50 Hz, realistic upper bound.
    action_delay_steps: Tuple[int, int] = (0, 2)

    # Optional: zero-mean Gaussian noise on the position target sent to
    # the PD controller (simulates encoder noise / quantisation). Std
    # in radians.
    target_noise_std: float = 0.005


@dataclass
class BodyRanges:
    """Body / contact parameters."""
    robot_link_mass_mult: Tuple[float, float] = (0.90, 1.10)
    # Note: ball physics randomisation is left to BallRanges below
    com_offset_m: float = 0.0  # placeholder for future CoM shift


@dataclass
class BallRanges:
    """Ball physical parameters — absolute (not multiplicative)."""
    mass_kg: Tuple[float, float] = (0.16, 0.24)
    friction: Tuple[float, float] = (0.6, 1.0)
    # Genesis-side restitution isn't directly exposed on Sphere; we use
    # the contact `solref` on the env side as the practical knob.


@dataclass
class DRConfig:
    motors: MotorRanges = field(default_factory=MotorRanges)
    bodies: BodyRanges = field(default_factory=BodyRanges)
    ball: BallRanges = field(default_factory=BallRanges)

    enabled: bool = True
    # If False, motor randomisation is still computed (logged) but not
    # applied — useful for ablation studies.
    apply_motors: bool = True
    apply_bodies: bool = True
    apply_ball: bool = True

    # Texture randomisation (rebuild required); off by default
    swap_textures: bool = False


# ─── The randomiser itself ─────────────────────────────────────────────


def _to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _safe_get(entity, name):
    fn = getattr(entity, name, None)
    if fn is None:
        return None
    try:
        return _to_np(fn())
    except Exception:
        return None


def _safe_get_with_idx(entity, name, idx):
    """Some Genesis getters accept dof indices; some don't. Try both."""
    fn = getattr(entity, name, None)
    if fn is None:
        return None
    try:
        return _to_np(fn(idx))
    except TypeError:
        try:
            return _to_np(fn())
        except Exception:
            return None
    except Exception:
        return None


class MotorRandomizer:
    """Per-episode randomiser for motors, body parameters, and ball
    parameters. Also maintains an action-delay buffer."""

    def __init__(self, cfg: Optional[DRConfig] = None,
                 rng: Optional[np.random.Generator] = None,
                 act_dim: int = 22):
        self.cfg = cfg or DRConfig()
        self.rng = rng or np.random.default_rng()
        self.act_dim = act_dim

        self.robot = None
        self.ball = None
        self.dof_indices: List[int] = []

        # Baseline values captured post-build
        self._base_kp = None
        self._base_kv = None
        self._base_damping = None
        self._base_friction_loss = None
        self._base_armature = None
        self._base_force_range = None  # (low, high) shape (N, 2)
        self._base_link_mass = None

        # Action delay state
        self._delay_buf: deque = deque(maxlen=8)
        self._current_delay = 0

        # Logging of last sampled values (for telemetry)
        self.last_sample: dict = {}

    # ── lifecycle ──────────────────────────────────────────────────

    def attach(self, robot, dof_indices: Iterable[int], ball=None):
        self.robot = robot
        self.dof_indices = list(dof_indices)
        self.ball = ball

        di = self.dof_indices
        self._base_kp = _safe_get_with_idx(robot, "get_dofs_kp", di)
        self._base_kv = _safe_get_with_idx(robot, "get_dofs_kv", di)
        self._base_damping = _safe_get_with_idx(robot, "get_dofs_damping", di)
        self._base_friction_loss = _safe_get_with_idx(
            robot, "get_dofs_frictionloss", di)
        self._base_armature = _safe_get_with_idx(robot, "get_dofs_armature", di)
        # force_range is shape (N, 2)
        fr = _safe_get_with_idx(robot, "get_dofs_force_range", di)
        self._base_force_range = fr
        self._base_link_mass = _safe_get(robot, "get_links_inertial_mass")

    def reset_episode(self):
        """Sample fresh parameters and push them to the simulator. Called
        from env.reset() after the scene has been reset."""
        if not self.cfg.enabled:
            return
        if self.robot is None or not self.dof_indices:
            return

        if self.cfg.apply_motors:
            self._apply_motor_randomisation()
        if self.cfg.apply_bodies:
            self._apply_body_randomisation()
        if self.cfg.apply_ball and self.ball is not None:
            self._apply_ball_randomisation()

        # Action delay
        delay_lo, delay_hi = self.cfg.motors.action_delay_steps
        self._current_delay = int(self.rng.integers(delay_lo, delay_hi + 1))
        # Buffer must hold one extra slot so we can index by delay
        self._delay_buf = deque(maxlen=max(2, self._current_delay + 1))

    # ── action-delay buffer ────────────────────────────────────────

    def delay_action(self, action: np.ndarray) -> np.ndarray:
        """Push a new action into the buffer; return the action that should
        be applied THIS step (i.e. the one from `_current_delay` steps ago).

        Also adds optional Gaussian noise to the target (encoder/quant noise).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Buffer push
        self._delay_buf.append(action.copy())
        # Pull the delayed one
        idx = max(0, len(self._delay_buf) - 1 - self._current_delay)
        delayed = self._delay_buf[idx]
        # Add target noise
        std = float(self.cfg.motors.target_noise_std)
        if std > 0.0:
            delayed = delayed + self.rng.normal(0.0, std, size=delayed.shape)
        return delayed.astype(np.float32)

    # ── apply: motors ──────────────────────────────────────────────

    def _apply_motor_randomisation(self):
        m = self.cfg.motors
        di = self.dof_indices
        n = len(di)

        def _sample_mult(rng_range):
            return self.rng.uniform(rng_range[0], rng_range[1], size=n)

        sampled = {}

        if self._base_kp is not None:
            scale = _sample_mult(m.kp_mult)
            new_kp = (self._base_kp * scale).astype(np.float32)
            try:
                self.robot.set_dofs_kp(new_kp.tolist(), di)
                sampled["kp_mean"] = float(scale.mean())
            except Exception:
                pass

        if self._base_kv is not None:
            scale = _sample_mult(m.kv_mult)
            new_kv = (self._base_kv * scale).astype(np.float32)
            try:
                self.robot.set_dofs_kv(new_kv.tolist(), di)
                sampled["kv_mean"] = float(scale.mean())
            except Exception:
                pass

        if self._base_damping is not None:
            scale = _sample_mult(m.damping_mult)
            new_d = (self._base_damping * scale).astype(np.float32)
            try:
                self.robot.set_dofs_damping(new_d.tolist(), di)
                sampled["damping_mean"] = float(scale.mean())
            except Exception:
                pass

        if self._base_friction_loss is not None:
            scale = _sample_mult(m.friction_loss_mult)
            new_f = (self._base_friction_loss * scale).astype(np.float32)
            try:
                self.robot.set_dofs_frictionloss(new_f.tolist(), di)
                sampled["fric_mean"] = float(scale.mean())
            except Exception:
                pass

        if self._base_armature is not None:
            scale = _sample_mult(m.armature_mult)
            new_a = (self._base_armature * scale).astype(np.float32)
            try:
                self.robot.set_dofs_armature(new_a.tolist(), di)
                sampled["arm_mean"] = float(scale.mean())
            except Exception:
                pass

        if self._base_force_range is not None:
            scale = _sample_mult(m.torque_limit_mult)
            # force_range can be shape (N,2) or (2,) depending on Genesis
            # version; normalise to (N,2)
            base = self._base_force_range
            if base.ndim == 1 and base.shape[0] == 2:
                base = np.tile(base, (n, 1))
            if base.ndim == 2 and base.shape == (n, 2):
                new_fr = base.copy()
                new_fr[:, 0] = base[:, 0] * scale  # lower (negative)
                new_fr[:, 1] = base[:, 1] * scale  # upper
                try:
                    self.robot.set_dofs_force_range(
                        new_fr[:, 0].tolist(), new_fr[:, 1].tolist(), di)
                    sampled["torque_mean"] = float(scale.mean())
                except Exception:
                    pass

        sampled["action_delay_steps"] = int(self._current_delay)
        self.last_sample["motors"] = sampled

    # ── apply: bodies ──────────────────────────────────────────────

    def _apply_body_randomisation(self):
        b = self.cfg.bodies
        if self._base_link_mass is None:
            return
        scale = self.rng.uniform(*b.robot_link_mass_mult,
                                 size=self._base_link_mass.shape)
        new_mass = (self._base_link_mass * scale).astype(np.float32)
        try:
            self.robot.set_links_inertial_mass(new_mass.tolist())
            self.last_sample["bodies"] = {"mass_mean": float(scale.mean())}
        except Exception:
            pass

    # ── apply: ball ────────────────────────────────────────────────

    def _apply_ball_randomisation(self):
        rng = self.cfg.ball
        try:
            m = float(self.rng.uniform(*rng.mass_kg))
            self.ball.set_mass(m)
            self.last_sample["ball"] = {"mass_kg": m}
        except Exception:
            pass

    # ── stats ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        out = {}
        for section, vals in self.last_sample.items():
            for k, v in vals.items():
                out[f"dr/{section}/{k}"] = v
        return out


# ── Backwards-compatible alias ─────────────────────────────────────────
#
# The earlier scaffold exported `DomainRandomizer`; keep that name pointing
# at the new class so existing imports keep working.
DomainRandomizer = MotorRandomizer
DRRanges = MotorRanges  # closest semantic match for the old name
