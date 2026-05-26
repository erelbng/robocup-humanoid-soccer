"""Dribble skill config.

Command vector (7-dim): walk command (5) + ball offset (2).

    [0] vx              — body-frame target velocity, m/s
    [1] vy              — body-frame target velocity, m/s
    [2] vyaw            — yaw rate, rad/s
    [3] foot_clearance  — commanded swing height, m
    [4] step_freq       — commanded step frequency, Hz
    [5] ball_off_x      — desired ball position relative to robot, body-frame x (m)
    [6] ball_off_y      — desired ball position relative to robot, body-frame y (m)

A "successful" dribble keeps the ball within a small window around
(ball_off_x, ball_off_y) while still tracking the locomotion command.
The orchestrator (Phase 2) can vary the offset to dribble around an
opponent or set up a shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DribbleConfig:
    # ── env ────────────────────────────────────────────────────────
    num_envs: int = 1024
    max_episode_steps: int = 1000
    dt: float = 0.02
    sim_dt: float = 0.002
    gait_freq_hz: float = 1.5

    # ── ball ──────────────────────────────────────────────────────
    ball_radius: float = 0.07            # FIFA size-5 ≈ 0.11 m; size-3 ≈ 0.09;
                                          # K1 has tiny feet — keep 0.07 for now.
    ball_mass: float = 0.4                # kg, light enough for K1 to push
    ball_spawn_range_x: Tuple[float, float] = (0.30, 0.80)   # forward of robot
    ball_spawn_range_y: Tuple[float, float] = (-0.20, 0.20)  # roughly centered

    # ── command vec ranges ────────────────────────────────────────
    vx_range: Tuple[float, float] = (-0.3, 0.8)
    vy_range: Tuple[float, float] = (-0.3, 0.3)
    vyaw_range: Tuple[float, float] = (-0.5, 0.5)
    foot_clearance_range: Tuple[float, float] = (0.04, 0.10)
    step_freq_range: Tuple[float, float] = (1.0, 2.2)
    ball_off_x_range: Tuple[float, float] = (0.30, 0.60)
    ball_off_y_range: Tuple[float, float] = (-0.15, 0.15)

    # ── head-look command (from the vision system) ───────────────
    # During dribbling, the head will typically look at the ball, so
    # pitch skews downward.
    head_yaw_range:   Tuple[float, float] = (-0.8, 0.8)
    head_pitch_range: Tuple[float, float] = (-0.2, 0.7)   # down-bias for ball

    # Episode-termination ball thresholds (relative to robot).
    ball_lost_distance: float = 2.0       # >2m → episode ends (out of control)

    # ── PPO defaults ──────────────────────────────────────────────
    total_timesteps: int = 200_000_000
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

    rewards: "DribbleRewardWeights" = field(
        default_factory=lambda: DribbleRewardWeights())


@dataclass
class DribbleRewardWeights:
    """Composite weights: walk-reward terms + ball-tracking terms.

    The dribble reward is the walk reward plus three ball terms:
    * `ball_offset` — exp-shaped reward on (ball - desired_offset) error
    * `ball_velocity` — encourage ball moving in commanded direction
    * `ball_lost` — large negative when the ball drifts beyond
      `ball_lost_distance`
    """
    # Walk-style terms (smaller than walk's because ball terms must
    # carry signal — total reward shouldn't blow up).
    track_lin_vel: float = 1.0
    track_ang_vel: float = 0.5
    upright: float = 0.5
    height: float = 0.5
    foot_clearance: float = 0.2
    action_smoothness: float = 0.002
    dof_acc: float = 1.0e-7
    torque: float = 1.0e-4
    base_motion: float = 0.5
    energy: float = 1.0e-4
    alive: float = 0.1
    fall: float = -2.0

    # Ball-specific terms
    ball_offset: float = 2.0              # exp-shaped on (ball - target_offset)
    ball_velocity: float = 0.5            # ball moves in commanded direction
    ball_lost: float = -10.0              # ball drifts beyond threshold

    # Head-look tracking from the vision system.
    head_tracking: float = 0.3
    # Arms-at-side regulariser (legged_gym-style).
    arm_pose: float = 0.05
