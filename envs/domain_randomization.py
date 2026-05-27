"""Domain randomization for sim-to-real transfer.

Re-introduced after step 9's deletion of the original monolith. Ranges
and structure are aligned with BoosterRobotics' T1 walking config
(`booster_gym/envs/T1.yaml`), which is their battle-tested setup for
the T1 biped — the closest published reference for a similar robot.

Two kinds of randomization live here:

  * Per-env physical parameters sampled ONCE at scene build:
    ground friction, kp/kd scaling, joint friction, base/link mass.
    These are reasonably treated as constants within an episode and
    redrawn each scene rebuild (i.e. on next training process).

  * Per-step perturbations sampled CONTINUOUSLY during training:
    base-frame push forces / torques applied at random intervals.

Observation noise (per-step additive Gaussians on dof_pos/vel, gravity,
etc.) is a separate concern handled in `skills/common_obs.py`'s noise
overlay — kept here in spirit so all DR hyperparams have one home.

DR is exposed to the teacher policy as part of its **privileged
observation**: the teacher gets the true sampled values (friction,
kp_scale, mass_scale, …) appended to its obs and learns optimal control
given oracle knowledge. The student is then distilled from the teacher
using proprio-only inputs (see `training/algorithms/distillation.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ─── config ────────────────────────────────────────────────────────────


@dataclass
class DomainRandConfig:
    """All DR knobs. Disable individual fields by setting their range to
    a degenerate `(c, c)` interval; disable the whole thing with
    `enabled=False`."""

    enabled: bool = True

    # ── physical params (per-env, set once at scene build) ────────
    ground_friction_range: Tuple[float, float] = (0.5, 1.5)
    kp_scale_range:        Tuple[float, float] = (0.95, 1.05)
    kd_scale_range:        Tuple[float, float] = (0.95, 1.05)
    joint_friction_range:  Tuple[float, float] = (0.0, 0.4)   # additive Nm/(rad/s)
    base_mass_scale_range: Tuple[float, float] = (0.8, 1.2)
    link_mass_scale_range: Tuple[float, float] = (0.98, 1.02)
    # Centre-of-mass jitter on the trunk (xyz, metres).
    com_offset_range:      Tuple[float, float] = (-0.02, 0.02)

    # ── per-step perturbations ────────────────────────────────────
    push_interval_s:       float = 5.0       # mean interval between pushes
    push_force_max:        float = 12.0      # N — peak transient force
    push_torque_max:       float = 3.0       # N·m
    push_duration_steps:   int = 5           # how many control steps the push lasts

    # ── observation noise (Gaussian σ, additive) ─────────────────
    obs_noise_root_quat: float = 0.01
    obs_noise_lin_vel:   float = 0.05
    obs_noise_ang_vel:   float = 0.10
    obs_noise_dof_pos:   float = 0.01
    obs_noise_dof_vel:   float = 0.10


# ─── sampler ───────────────────────────────────────────────────────────


@dataclass
class DRSample:
    """Per-env sampled DR values. Used as PRIVILEGED OBS for the teacher
    policy (see `SkillEnv.privileged_obs`). Layout (8 dims):

        [0] ground_friction
        [1] kp_scale
        [2] kd_scale
        [3] joint_friction
        [4] base_mass_scale
        [5..8] com_offset_xyz
    """
    ground_friction: np.ndarray   # (N,)
    kp_scale:        np.ndarray   # (N,)
    kd_scale:        np.ndarray   # (N,)
    joint_friction:  np.ndarray   # (N,)
    base_mass_scale: np.ndarray   # (N,)
    com_offset:      np.ndarray   # (N, 3)

    PRIVILEGED_DIM: int = 8

    def as_privileged_obs(self) -> np.ndarray:
        """Return (N, PRIVILEGED_DIM) float32 stacked array."""
        return np.concatenate([
            self.ground_friction[:, None],
            self.kp_scale[:, None],
            self.kd_scale[:, None],
            self.joint_friction[:, None],
            self.base_mass_scale[:, None],
            self.com_offset,
        ], axis=1).astype(np.float32)


def sample_dr(cfg: DomainRandConfig, num_envs: int,
              rng: np.random.Generator) -> DRSample:
    """Draw a fresh DR sample for `num_envs` parallel envs.

    Even with `cfg.enabled=False` this returns a sample full of
    nominal (1.0 / 0.0) values, so downstream code never has to branch
    on the flag — the privileged obs has a stable shape across runs.
    """
    if not cfg.enabled:
        return DRSample(
            ground_friction=np.ones(num_envs, dtype=np.float32),
            kp_scale=np.ones(num_envs, dtype=np.float32),
            kd_scale=np.ones(num_envs, dtype=np.float32),
            joint_friction=np.zeros(num_envs, dtype=np.float32),
            base_mass_scale=np.ones(num_envs, dtype=np.float32),
            com_offset=np.zeros((num_envs, 3), dtype=np.float32),
        )
    u = lambda a, b: rng.uniform(a, b, size=num_envs).astype(np.float32)
    return DRSample(
        ground_friction=u(*cfg.ground_friction_range),
        kp_scale=u(*cfg.kp_scale_range),
        kd_scale=u(*cfg.kd_scale_range),
        joint_friction=u(*cfg.joint_friction_range),
        base_mass_scale=u(*cfg.base_mass_scale_range),
        com_offset=rng.uniform(
            cfg.com_offset_range[0], cfg.com_offset_range[1],
            size=(num_envs, 3)).astype(np.float32),
    )


# ─── per-step push perturbations ──────────────────────────────────────


class PushScheduler:
    """Per-env scheduler that triggers random base-frame impulses.

    Each env independently rolls a Poisson timer: at every control
    step there's a probability `1/push_interval_steps` of starting a
    new push. A push lasts `push_duration_steps` control steps. The
    force/torque vector is sampled once per push and held constant for
    its duration.

    `step()` returns (N, 3) force and (N, 3) torque to apply to the
    base of each env. Envs not currently being pushed get zeros.
    """

    def __init__(self, cfg: DomainRandConfig, num_envs: int,
                 control_dt: float, rng: np.random.Generator):
        self.cfg = cfg
        self.N = int(num_envs)
        self.rng = rng
        # Probability that any given env starts a push this step.
        self._push_prob = float(control_dt) / max(cfg.push_interval_s,
                                                   control_dt)
        self._remaining_steps = np.zeros(self.N, dtype=np.int32)
        self._active_force = np.zeros((self.N, 3), dtype=np.float32)
        self._active_torque = np.zeros((self.N, 3), dtype=np.float32)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cfg.enabled:
            return (np.zeros((self.N, 3), dtype=np.float32),
                    np.zeros((self.N, 3), dtype=np.float32))

        # Decrement active push timers; clear forces when expired.
        self._remaining_steps = np.maximum(0, self._remaining_steps - 1)
        expired = self._remaining_steps == 0
        self._active_force[expired] = 0.0
        self._active_torque[expired] = 0.0

        # Roll new pushes on any env that's NOT currently being pushed.
        eligible = self._remaining_steps == 0
        if eligible.any():
            start = (self.rng.random(self.N) < self._push_prob) & eligible
            n_start = int(start.sum())
            if n_start > 0:
                # Random direction × magnitude. Uniform in a 3-ball is
                # cleaner with the (x, y, z) ~ N(0,1) normalisation trick.
                d = self.rng.standard_normal((n_start, 3))
                d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
                mag_f = self.rng.uniform(0.0, self.cfg.push_force_max,
                                          size=(n_start, 1))
                mag_t = self.rng.uniform(0.0, self.cfg.push_torque_max,
                                          size=(n_start, 1))
                idx = np.where(start)[0]
                self._active_force[idx] = (d * mag_f).astype(np.float32)
                # Independent direction for the torque
                d2 = self.rng.standard_normal((n_start, 3))
                d2 /= (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-6)
                self._active_torque[idx] = (d2 * mag_t).astype(np.float32)
                self._remaining_steps[idx] = self.cfg.push_duration_steps

        return self._active_force.copy(), self._active_torque.copy()


# ─── obs noise overlay ────────────────────────────────────────────────


def add_obs_noise(*, root_quat: np.ndarray, lin_vel: np.ndarray,
                  ang_vel: np.ndarray, dof_pos: np.ndarray,
                  dof_vel: np.ndarray, cfg: DomainRandConfig,
                  rng: np.random.Generator) -> dict:
    """Return noisified copies of each obs component. Caller substitutes
    these in place of the clean readings before calling
    `compute_common_obs`.

    We keep the noise on RAW sensor channels (quat, vels, dof state),
    not on the derived projected_gravity / body_velocity — applying
    noise at the source is closer to the real sensor model.
    """
    if not cfg.enabled:
        return dict(root_quat=root_quat, lin_vel=lin_vel, ang_vel=ang_vel,
                    dof_pos=dof_pos, dof_vel=dof_vel)
    def _n(x, sigma):
        if sigma <= 0.0:
            return x
        return (x + rng.standard_normal(x.shape).astype(np.float32) * sigma
                ).astype(np.float32)
    return dict(
        root_quat=_n(root_quat, cfg.obs_noise_root_quat),
        lin_vel=_n(lin_vel,   cfg.obs_noise_lin_vel),
        ang_vel=_n(ang_vel,   cfg.obs_noise_ang_vel),
        dof_pos=_n(dof_pos,   cfg.obs_noise_dof_pos),
        dof_vel=_n(dof_vel,   cfg.obs_noise_dof_vel),
    )


__all__ = [
    "DomainRandConfig", "DRSample", "sample_dr",
    "PushScheduler", "add_obs_noise",
]
