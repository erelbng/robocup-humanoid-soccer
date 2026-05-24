"""
Walk-style command conditioning for Phase 1.

Concatenates a small "command vector" to the agent's observation so a
SINGLE policy can produce different gaits / behaviors at deploy time
without retraining per style. We follow the htwk-gym / Isaac-Gym pattern:
sample a command at each reset, condition the policy on it, and shape
the reward so reaching the commanded velocity / heading / aggressiveness
is what's rewarded.

Command vector layout (default 5-dim):
    [target_vx, target_vy, target_yaw_rate, aggressiveness, defensiveness]

  * target_vx / target_vy : commanded base velocity in body frame [m/s]
  * target_yaw_rate       : commanded yaw rate [rad/s]
  * aggressiveness        : 0..1 — pushes toward attacker behavior
                            (faster gait, harder kicks, ball-to-goal weight)
  * defensiveness         : 0..1 — pushes toward defender behavior
                            (cautious tracking, defensive coverage weight)

Aggressiveness and defensiveness can co-exist but the policy will learn
they trade off (you can't sprint and also play it safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class StyleCommandRanges:
    """Sampling ranges for each command dim."""
    # Forward/lateral target velocity (m/s)
    vx_range: Tuple[float, float] = (-0.4, 1.0)
    vy_range: Tuple[float, float] = (-0.4, 0.4)
    yaw_range: Tuple[float, float] = (-1.0, 1.0)
    aggressiveness_range: Tuple[float, float] = (0.0, 1.0)
    defensiveness_range: Tuple[float, float] = (0.0, 1.0)

    # Probability of resampling commands mid-episode. Higher = the policy
    # gets exposed to command transitions instead of just constant phases.
    resample_prob_per_step: float = 0.0

    # Probability that a sampled aggressiveness OR defensiveness is forced
    # to 0 (so the policy also sees neutral / balanced styles often).
    style_zero_prob: float = 0.3


@dataclass
class StyleCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    aggressiveness: float = 0.0
    defensiveness: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.yaw,
                         self.aggressiveness, self.defensiveness],
                        dtype=np.float32)

    @property
    def dim(self) -> int:
        return 5


class StyleCommandSampler:
    """Episode/step sampler. Holds the current command + the RNG."""

    def __init__(self, ranges: StyleCommandRanges = None,
                 rng: Optional[np.random.Generator] = None):
        self.ranges = ranges or StyleCommandRanges()
        self.rng = rng or np.random.default_rng()
        self.current = StyleCommand()

    def sample(self) -> StyleCommand:
        r = self.ranges
        c = StyleCommand(
            vx=float(self.rng.uniform(*r.vx_range)),
            vy=float(self.rng.uniform(*r.vy_range)),
            yaw=float(self.rng.uniform(*r.yaw_range)),
            aggressiveness=float(self.rng.uniform(*r.aggressiveness_range)),
            defensiveness=float(self.rng.uniform(*r.defensiveness_range)),
        )
        # Occasionally force balanced style
        if self.rng.random() < r.style_zero_prob:
            c.aggressiveness = 0.0
        if self.rng.random() < r.style_zero_prob:
            c.defensiveness = 0.0
        self.current = c
        return c

    def maybe_resample_step(self) -> bool:
        """Called once per env.step(). Re-samples with low probability so
        the policy sees command transitions inside one episode."""
        if self.ranges.resample_prob_per_step <= 0.0:
            return False
        if self.rng.random() < self.ranges.resample_prob_per_step:
            self.sample()
            return True
        return False

    def fix(self, **kwargs) -> StyleCommand:
        """Override the current command — used at deploy/eval time to
        pick a specific style (e.g. aggressive striker, defensive midfielder).
        """
        for k, v in kwargs.items():
            if hasattr(self.current, k):
                setattr(self.current, k, float(v))
        return self.current


def commanded_velocity_reward(
    robot_vel_world: np.ndarray,
    robot_quat: np.ndarray,
    cmd: StyleCommand,
    tracking_sigma: float = 0.25,
) -> float:
    """Reward for matching the commanded planar velocity.

    Uses a Gaussian on the velocity error so reward saturates near the
    target (avoids unstable "the faster the better" gradients). The world
    velocity is projected into the body frame via the trunk yaw — we
    only use yaw, not full orientation, so a tipped robot still gets
    credit for moving in the commanded body-frame direction.
    """
    # Extract yaw from quaternion (w, x, y, z)
    w, x, y, z = robot_quat
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    vx_body = c * robot_vel_world[0] - s * robot_vel_world[1]
    vy_body = s * robot_vel_world[0] + c * robot_vel_world[1]
    err = (vx_body - cmd.vx) ** 2 + (vy_body - cmd.vy) ** 2
    return float(np.exp(-err / (tracking_sigma ** 2)))


def commanded_yaw_rate_reward(
    angvel_world: np.ndarray,
    cmd: StyleCommand,
    tracking_sigma: float = 0.5,
) -> float:
    err = (float(angvel_world[2]) - cmd.yaw) ** 2
    return float(np.exp(-err / (tracking_sigma ** 2)))
