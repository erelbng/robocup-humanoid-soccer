"""Standup skill config — stability-heavy reward + diverse initial pose."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class StandupRewardWeights:
    """Weights for the standup composite reward.

    Speed: ONE term (`time_penalty`, dense) + ONE terminal (`success_bonus`,
    time-scaled). Stability: several terms — all quadratic deviations that
    vanish at the standing equilibrium, so the optimum stays at 'upright
    + still' and isn't pulled away by any one term.
    """
    # Primary shaping (positive, exp/clipped to [0, 1])
    upright: float = 3.0               # cos(trunk-z, world-z), clamped [0, 1]
    height: float = 2.0                # gaussian around target_height

    # Stability penalties (vanish at equilibrium)
    gravity_horizontal: float = 0.5    # gx² + gy² — symmetric tilt
    base_ang_vel_sway: float = 0.05    # ωx² + ωy² — roll/pitch rate
    base_lin_vel_drift: float = 0.5    # ||v||², phase-gated near upright
    joint_vel_quiet: float = 0.001     # Σ q̇², phase-gated near upright
    action_smoothness: float = 0.1     # (a - a_{-1})² — first derivative
    action_jerk: float = 0.1           # (a - 2 a_{-1} + a_{-2})² — second der.

    # Speed signal — exactly one dense term + one terminal pulse.
    # The bonus is steep: τ=40 steps (0.8 s) means a 0.5 s stand pays
    # ~214 (0.54 × 400), a 1 s stand pays ~115, a 2 s stand pays ~33,
    # a 3 s stand pays ~9. Sub-second standups become massively rewarded.
    time_penalty: float = 1.0          # per step until sustained-success
    success_bonus: float = 400.0       # paid on streak completion, scaled
                                       #   by exp(-t_first / tau)
    success_persistence: float = 5.0   # per step while in the hold window


@dataclass
class StandupConfig:
    # ── env ────────────────────────────────────────────────────────
    num_envs: int = 1024
    max_episode_steps: int = 250       # 5 s at 50 Hz — enough for a 3 s
                                       #   standup + 2 s margin
    dt: float = 0.02
    sim_dt: float = 0.002
    gait_freq_hz: float = 1.5          # unused but keeps obs layout uniform

    # Initial-pose templates. Each env samples uniformly, then jitters
    # orientation + joint targets per the params below.
    poses: Tuple[str, ...] = ("supine", "prone", "side_left", "side_right")

    # Pose-jitter parameters. Composed on top of the chosen template at
    # every reset → diverse fallen starts without leaving the
    # anatomically sensible region.
    orient_jitter_rad: float = math.pi / 6.0   # ±30° extra rotation
    yaw_jitter_rad: float = math.pi            # uniform yaw in [-π, π]
    joint_jitter_rad: float = 0.15             # σ on each joint target
    height_jitter_m: float = 0.02              # ±2 cm trunk height jitter

    # Sustained-success thresholds. A standup is "done" once
    # `success_hold_steps` consecutive frames satisfy both upright and
    # height conditions.
    target_height: float = 0.55
    upright_threshold: float = 0.92            # cosine ~23° tilt max
    success_hold_steps: int = 50               # 1.0 s at 50 Hz

    # Time-scaling for the terminal bonus. Bonus *= exp(-t_first / tau).
    # τ=40 steps (0.8 s) gives a steep curve so sub-second standups
    # dominate the return — see the explanatory comment on
    # `success_bonus` above for example payouts.
    time_to_stand_tau_steps: float = 40.0

    # ── PPO defaults (training.algorithms.ppo) ────────────────────
    total_timesteps: int = 50_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    n_steps: int = 64

    obs_dim: int = 0
    act_dim: int = 22

    rewards: StandupRewardWeights = field(
        default_factory=lambda: StandupRewardWeights())
