"""Walk skill config — hyperparameters and command spec ranges.

Used by both the env (for command sampling) and the trainer (for PPO
hyperparams). Kept as a dataclass for easy override from CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class WalkConfig:
    # ── env / scene ────────────────────────────────────────────────
    num_envs: int = 1024
    max_episode_steps: int = 1000
    dt: float = 0.02            # 50 Hz control
    sim_dt: float = 0.002       # 500 Hz physics
    gait_freq_hz: float = 1.5   # clock signal for periodic-gait conditioning

    # ── command vec ranges (vx, vy, vyaw, foot_clearance, step_freq) ──
    vx_range: Tuple[float, float] = (-0.5, 1.0)     # m/s — forward bias
    vy_range: Tuple[float, float] = (-0.4, 0.4)     # m/s — side-step
    vyaw_range: Tuple[float, float] = (-1.0, 1.0)   # rad/s
    foot_clearance_range: Tuple[float, float] = (0.04, 0.12)  # m
    step_freq_range: Tuple[float, float] = (1.0, 2.5)         # Hz

    # ── head-look command (from the vision system) ───────────────
    # Direct joint targets for AAHead_yaw / Head_pitch (rad). Ranges
    # are intentionally narrower than the URDF mechanical limits so
    # the policy doesn't waste samples on hyperextended postures.
    head_yaw_range:   Tuple[float, float] = (-0.8, 0.8)   # ~±46°
    head_pitch_range: Tuple[float, float] = (-0.4, 0.6)   # slight down-bias

    # ── PPO hyperparams (consumed by training.algorithms.ppo) ─────
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    n_steps: int = 64

    # Filled in at runtime — set by the env so the trainer can sanity-
    # check before allocating buffers.
    obs_dim: int = 0
    act_dim: int = 22

    # ── reward weights ────────────────────────────────────────────
    rewards: "WalkRewardWeights" = field(
        default_factory=lambda: WalkRewardWeights())


@dataclass
class WalkRewardWeights:
    """Per-component scalars used by skills.walk.rewards.

    Tuned so the dominant signal is velocity tracking + stay-alive;
    smoothness / energy are SOFT shaping terms ≪ tracking.
    """
    # primary objectives
    track_lin_vel: float = 1.5         # exp-shaped on (vx, vy) error
    track_ang_vel: float = 0.75        # exp-shaped on vyaw error
    upright: float = 0.5
    height: float = 0.5                # gaussian around 0.55 m

    # gait shaping
    foot_clearance: float = 0.3        # match commanded swing height
    feet_air_time: float = 0.2         # encourage non-zero air time

    # head-look tracking (from the vision system) — small weight so it
    # doesn't trade off against velocity tracking.
    head_tracking: float = 0.3

    # arms-at-side regulariser (legged_gym-style): squared deviation
    # of the 8 arm DOFs from the default rest pose. Small weight so it
    # only kicks in when the policy is otherwise indifferent about arm
    # posture — keeps shoulders / elbows tucked instead of flailing.
    arm_pose: float = 0.05

    # regularizers (negative)
    action_smoothness: float = 0.002
    dof_acc: float = 1.0e-7
    torque: float = 1.0e-4
    base_motion: float = 0.5           # damp roll/pitch rates
    energy: float = 1.0e-4

    # termination signals
    alive: float = 0.1
    fall: float = -2.0
