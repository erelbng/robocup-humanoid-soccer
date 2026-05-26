"""Standup skill config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class StandupConfig:
    # ── env ────────────────────────────────────────────────────────
    num_envs: int = 1024
    max_episode_steps: int = 500
    dt: float = 0.02
    sim_dt: float = 0.002
    gait_freq_hz: float = 1.5  # unused for standup but keeps obs layout uniform

    # The set of fallen poses spawned at episode start. Mix is uniform
    # over this tuple; subclassing standup_poses lets us bias later.
    poses: Tuple[str, ...] = ("supine", "prone", "side_left", "side_right")

    # Trunk height target for "successful standup" detection.
    target_height: float = 0.55
    upright_threshold: float = 0.92    # cosine, 1=perfectly upright

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

    rewards: "StandupRewardWeights" = field(
        default_factory=lambda: StandupRewardWeights())


@dataclass
class StandupRewardWeights:
    """Weights for the standup composite reward.

    Heavier on upright + height than the walk reward because there's no
    velocity-tracking term to balance against. The success bonus is a
    one-shot terminal pulse.
    """
    upright: float = 5.0           # cos(trunk-z, world-z), clamped to [0, 1]
    height: float = 3.0            # gaussian around target_height
    energy: float = 0.005
    action_smoothness: float = 0.1
    success_bonus: float = 50.0    # one-shot when upright + at height
