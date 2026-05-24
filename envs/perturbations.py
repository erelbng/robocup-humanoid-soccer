"""
External perturbations for robust Phase 1 training.

Implements the "push robot" curriculum: at random intervals during an
episode we apply an impulse (a brief horizontal force at the trunk) so the
policy learns to recover. The force schedule ramps with curriculum stage —
no pushes while learning to stand, light pushes while walking, strong
pushes once dribbling.

Genesis exposes per-link external forces via
`entity.apply_links_external_force(force, links_idx)`. This module wraps
that with a stateful scheduler so step() can just call
`pusher.maybe_push(sim_step)` without sampling logic at the call site.

Designed to be fully optional — if Genesis isn't available, or the link
lookup fails, every method is a no-op and training continues unperturbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PerturbationSchedule:
    """How hard, how often, and where to push.

    All forces are in Newtons. The K1 trunk weighs ~6.5 kg, so a 30 N
    horizontal push for 100 ms produces a ~0.5 m/s velocity change — a
    realistic "shove" from a teammate. 100 N is closer to a deliberate
    foul.
    """
    # Probability that a push fires on any given control-rate step.
    # 50Hz control → p=0.005 ≈ one push per 4s of sim time.
    push_probability: float = 0.005

    # Force magnitude bounds in Newtons.
    force_min: float = 0.0
    force_max: float = 30.0

    # How long (in PHYSICS sub-steps, not control steps) a single push
    # holds. Below ~50 phys-steps the impulse rounds to zero.
    duration_steps_min: int = 30
    duration_steps_max: int = 80

    # Vertical component allowed (set >0 to occasionally trip the robot).
    vertical_fraction: float = 0.0

    # Allow downward forces too (push down on shoulders) — used in standup
    # to simulate being pinned briefly. Off by default.
    allow_downward: bool = False

    @classmethod
    def for_stage(cls, stage: str) -> "PerturbationSchedule":
        """Default schedule per curriculum stage. Tune from training logs."""
        if stage == "stand":
            # No pushes — learn balance first.
            return cls(push_probability=0.0)
        if stage == "standup":
            # Standup training: occasional re-push to keep robot prone-ish.
            return cls(push_probability=0.001, force_min=10, force_max=40,
                       vertical_fraction=0.5, allow_downward=True)
        if stage == "walk":
            return cls(push_probability=0.003, force_min=5, force_max=25)
        if stage == "dribble":
            return cls(push_probability=0.005, force_min=10, force_max=40)
        if stage == "shoot":
            return cls(push_probability=0.004, force_min=10, force_max=35)
        # "full" — strongest pushes
        return cls(push_probability=0.006, force_min=15, force_max=50)


class RobotPusher:
    """Stateful perturbation applier bound to a single robot entity.

    Lifecycle:
        pusher = RobotPusher(schedule, rng=np.random.default_rng(seed))
        pusher.attach(robot, trunk_link_name="Trunk")    # post-build
        for _ in range(...):
            pusher.maybe_push(physics_step_index)
            scene.step()
    """

    def __init__(self, schedule: PerturbationSchedule = None,
                 rng: Optional[np.random.Generator] = None):
        self.sched = schedule or PerturbationSchedule()
        self.rng = rng or np.random.default_rng()
        self.robot = None
        self.trunk_idx: Optional[int] = None

        # Active push state
        self._remaining = 0       # phys-steps left in current push
        self._force = np.zeros(3, dtype=np.float32)

        # Stats (useful for W&B logging)
        self.total_pushes = 0
        self.total_force_impulse = 0.0  # ∑ |F| · dt over training

    def attach(self, robot, trunk_link_name: str = "Trunk") -> bool:
        """Resolve the trunk link index for force application.

        Genesis re-numbers links during build; we look up by name. Returns
        True if attachment succeeded, False otherwise (pusher becomes
        a no-op in that case).
        """
        self.robot = robot
        try:
            link = robot.get_link(trunk_link_name)
            # `link.idx_local` is the link index used by
            # apply_links_external_force.
            self.trunk_idx = int(getattr(link, "idx_local",
                                         getattr(link, "idx", None)))
            return self.trunk_idx is not None
        except Exception:
            # Common case: robot is a placeholder Sphere — no named links.
            self.trunk_idx = None
            return False

    def reset(self):
        """Clear any in-flight push (called from env.reset)."""
        self._remaining = 0
        self._force[:] = 0.0

    def set_schedule(self, schedule: PerturbationSchedule):
        """Hot-swap the schedule, e.g. mid-episode curriculum changes."""
        self.sched = schedule
        self.reset()

    def maybe_push(self, phys_dt: float = 0.002):
        """Called before each physics step. Samples or continues a push.

        `phys_dt` is the simulator timestep — used for impulse accounting.
        """
        if self.robot is None or self.trunk_idx is None:
            return
        if self.sched.push_probability <= 0.0:
            return

        if self._remaining > 0:
            self._apply_force()
            self._remaining -= 1
            self.total_force_impulse += float(
                np.linalg.norm(self._force) * phys_dt
            )
            return

        # Sample a new push?
        if self.rng.random() < self.sched.push_probability:
            self._sample_push()
            self._apply_force()
            self.total_pushes += 1

    # ── internals ──────────────────────────────────────────────────

    def _sample_push(self):
        mag = self.rng.uniform(self.sched.force_min, self.sched.force_max)
        # Random heading in the XY plane
        theta = self.rng.uniform(0, 2 * np.pi)
        fx = mag * np.cos(theta)
        fy = mag * np.sin(theta)
        fz = 0.0
        if self.sched.vertical_fraction > 0.0:
            vmax = mag * self.sched.vertical_fraction
            fz = self.rng.uniform(-vmax if self.sched.allow_downward else 0.0,
                                  vmax)
        self._force[:] = (fx, fy, fz)
        self._remaining = int(self.rng.integers(
            self.sched.duration_steps_min,
            self.sched.duration_steps_max + 1,
        ))

    def _apply_force(self):
        """Push the trunk for one phys-step using Genesis's
        link external force API. Falls back silently if unavailable.
        """
        try:
            self.robot.apply_links_external_force(
                force=self._force.reshape(1, 3),
                links_idx=[self.trunk_idx],
            )
        except Exception:
            # Old Genesis versions named this differently — try a fallback.
            try:
                self.robot.apply_links_external_force(
                    self._force.tolist(), [self.trunk_idx]
                )
            except Exception:
                pass

    # ── introspection for logging ──────────────────────────────────

    def is_pushing(self) -> bool:
        return self._remaining > 0

    def current_force(self) -> np.ndarray:
        return self._force.copy()

    def stats(self) -> dict:
        return {
            "perturb/total_pushes": self.total_pushes,
            "perturb/total_impulse_Ns": self.total_force_impulse,
            "perturb/is_pushing": float(self.is_pushing()),
        }
