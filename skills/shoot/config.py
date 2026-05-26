"""Shoot skill config.

Command vector (3-dim):

    [0] aim_angle   — desired ball direction relative to robot forward,
                      radians, ±π. The world-frame target is computed
                      at reset from this + robot yaw + shot distance.
    [1] power       — desired ball speed at the moment of kick, m/s.
                      Reward exp-shaped around this value.
    [2] foot_pref   — −1 = left foot preferred, +1 = right foot. Weak
                      signal in this first cut (we don't query contact
                      to detect which foot actually kicked); kept in the
                      command so the orchestrator can express intent.

The episode succeeds (terminates with a bonus) when the ball reaches
`kick_speed_threshold` and is moving toward the target, indicating a
clean kick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ShootConfig:
    # ── env ────────────────────────────────────────────────────────
    num_envs: int = 1024
    max_episode_steps: int = 300         # shorter than dribble — focused task
    dt: float = 0.02
    sim_dt: float = 0.002
    gait_freq_hz: float = 1.5

    # ── ball ──────────────────────────────────────────────────────
    ball_radius: float = 0.07
    # Spawn the ball within striking range (front of robot, near feet).
    ball_spawn_range_x: Tuple[float, float] = (0.25, 0.40)
    ball_spawn_range_y: Tuple[float, float] = (-0.10, 0.10)

    # ── target ────────────────────────────────────────────────────
    # In single-skill training, the target is sampled inside the +x
    # goal mouth. The orchestrator will override this at runtime via
    # the aim_angle command, but during training we want shots to
    # actually go somewhere useful.
    goal_x: float = 4.5
    goal_half_width: float = 1.3         # goal_width/2 from field_info.json
    goal_z: float = 0.4                  # mid-height of goal mouth
    target_distance_for_aim: float = 3.0  # used to compute world target
                                          # from aim_angle command

    # ── command ranges ────────────────────────────────────────────
    aim_angle_range: Tuple[float, float] = (-0.5, 0.5)   # rad, ~±28°
    power_range: Tuple[float, float] = (1.0, 4.0)        # m/s
    foot_range: Tuple[float, float] = (-1.0, 1.0)

    # ── kick detection thresholds ─────────────────────────────────
    kick_speed_threshold: float = 1.5    # m/s — ball speed considered "kicked"
    ball_lost_distance: float = 3.0      # >3m → episode end (out of range)

    # ── PPO defaults ──────────────────────────────────────────────
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01           # higher entropy: kicking is sparse
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    n_steps: int = 64

    obs_dim: int = 0
    act_dim: int = 22

    rewards: "ShootRewardWeights" = field(
        default_factory=lambda: ShootRewardWeights())


@dataclass
class ShootRewardWeights:
    """Sparse-task reward weights.

    The kick_event bonus is the dominant signal — by design — so the
    policy actively seeks to kick rather than dawdle near the ball.
    """
    # Dense shaping
    approach_ball: float = 0.5      # exp-shaped on robot-ball distance
    ball_to_target: float = 1.0     # dot(ball_vel, dir_to_target_unit)
    upright: float = 0.3
    height: float = 0.3
    alive: float = 0.05

    # Regularizers
    action_smoothness: float = 0.002
    dof_acc: float = 1.0e-7
    base_motion: float = 0.3

    # Sparse pulses
    kick_event: float = 30.0        # one-shot when ball crosses threshold
    power_match: float = 5.0        # exp-shaped on (ball_speed - cmd_power)
    aim_accuracy: float = 5.0       # exp-shaped on (ball_dir - target_dir)
    fall: float = -2.0
    ball_lost: float = -5.0
