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
    # vx_range is the FINAL (sprint) range; early training is restricted to
    # `vx_walk_max` and ramps up via the speed curriculum (see below) so the
    # policy masters balance + a walking gait before being asked to sprint.
    vx_range: Tuple[float, float] = (-0.8, 2.5)     # m/s — sprint-capable fwd
    vy_range: Tuple[float, float] = (-0.5, 0.5)     # m/s — side-step
    vyaw_range: Tuple[float, float] = (-1.5, 1.5)   # rad/s
    foot_clearance_range: Tuple[float, float] = (0.04, 0.14)  # m
    step_freq_range: Tuple[float, float] = (1.0, 3.2)         # Hz

    # ── speed curriculum (SPRINT, arXiv:2605.28549) ───────────────────
    # Ramp the commanded forward-speed cap from a walk to the full sprint
    # range over training, gated on the policy actually tracking well (like
    # the standup curricula). Without this, sampling sprint speeds from step
    # 0 just makes the policy fall — it never learns the slow gait first.
    speed_curriculum_env_steps: int = 120_000_000
    vx_walk_max: float = 0.4          # SLOW start so early tracking is easy
    #                                   (learn to track at low speed first,
    #                                   then ramp to the sprint range) — also
    #                                   helps escape the "stand still" optimum
    speed_curriculum_min_track: float = 0.5   # widen the speed cap once
    #   tracking is decent. 0.6 let it ramp while tracking degraded (0.67→0.51);
    #   0.75 was too strict (froze speed at 0.43); 0.68 lets speed grow while
    #   keeping quality reasonable.
    # vyaw slow-start curriculum (SAME idea as vx, applied to turning): yaw was
    # commanded over the full ±1.5 from step 0, so the yaw error was always
    # large and the (tight-σ) reward was flat-zero → no gradient → yaw never
    # learned. Start small, ramp gated on YAW tracking so turning is learnable.
    vyaw_walk_max: float = 0.5        # rad/s — turning cap at curriculum start

    # ── frequency-adaptive gait (SPRINT's core idea) ─────────────────
    # Natural locomotion cadence rises with speed. Rather than sampling
    # step_freq independently, couple it to the commanded speed during
    # training: step_freq ≈ base + slope·|v|. The command channel still
    # exposes step_freq (the orchestrator/deploy can override), but the
    # TRAINING distribution reflects speed→cadence coupling so the learned
    # gait is sane across the speed range.
    freq_adaptive_gait: bool = True
    step_freq_base: float = 1.3       # Hz at standstill
    step_freq_per_mps: float = 0.55   # Hz added per m/s of commanded speed

    # ── gait-contact pattern (anti-shuffle) ──────────────────────────
    # A per-env gait phase advances at the commanded step_freq; it defines a
    # smooth alternating desired-contact pattern (one foot stance, one swing).
    # The gait_contact reward scores actual foot contact against it → forces
    # real stepping. duty>0.5 gives a brief double-support (a walk, not a run).
    # duty→0.5 (near-symmetric single-support, minimal double-support) and a
    # SHARPER transition (kappa 0.08→0.05) make the desired pattern read as a
    # clean "one foot down / one up" alternation. That LOWERS the score a
    # shuffle (both feet always down) can earn and RAISES the ceiling for true
    # stepping — widening the reward gap the policy was previously ignoring.
    gait_duty: float = 0.5            # stance fraction of the gait cycle
    gait_contact_kappa: float = 0.05  # transition smoothness of desired contact

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
    # primary objectives — BOOSTER T1 RECIPE (booster_gym/envs/T1.yaml).
    # The decisive lesson from 3 failed shuffle-breaking attempts: velocity
    # tracking must be MODEST, not dominant. We previously had tracking at
    # 3.0/3.0, so shuffling-to-track out-rewarded any stepping term and the
    # feet never left the ground for 75M steps. Booster keeps tracking at
    # 1.0/0.5 and makes the SWING reward dominant (3.0) — so lifting the right
    # foot at the right time is the single most rewarding thing the policy can
    # do. We mirror that balance.
    track_lin_vel: float = 2.5         # exp-shaped on (vx, vy) error. 1.0→2.0→
    #   2.5: the policy kept marching IN PLACE (feet_swing 3.0 is fully earned
    #   by stepping without translating; track stuck at 0.17). At 2.5, walking
    #   FORWARD earns both feet_swing AND tracking (~5.5) while marching earns
    #   only feet_swing+partial-track (~3.5) — a ~2.0/step margin that should
    #   finally pull real forward locomotion. Still ≤ feet_swing so no shuffle.
    forward_progress: float = 2.0      # LINEAR fraction-of-commanded-speed
    #   reward (constant gradient → "go faster"). The decisive anti-march-in-
    #   place lever: exp tracking's gradient vanished far from target so the
    #   policy stepped without translating (track stuck ~0.28 after 262M). This
    #   pulls actual forward speed up monotonically. Walking forward now earns
    #   feet_swing(3)+forward(3)+track(2.5) ≈ 8.5 vs marching's ~3.7.
    track_ang_vel: float = 0.5         # exp-shaped on vyaw error (Booster 0.5)
    ang_tracking_sigma: float = 0.4    # yaw-rate reward exp σ (usable gradient)
    # POSTURE PENALTIES (applied as (up-1)/(h-1) in compute_walk_reward, ≤0).
    # Booster uses orientation -5 / base_height -20 as PENALTIES; we mirror
    # that. CRITICAL: these must NOT be positive standing bonuses — when they
    # were (+1.0 each) the policy earned +2.0/step just standing and never
    # risked a step (feet_swing stuck at 0.0 for 27M steps). As penalties,
    # standing earns 0 and stepping is the only way to score.
    upright: float = 5.0               # penalty weight on (1−upright)
    height: float = 20.0               # penalty weight on (1−height@0.5m)

    # gait shaping — feet_swing is now the DOMINANT term (Booster's recipe).
    feet_swing: float = 3.0            # +1 per foot AIRBORNE during its swing
    #   window (left phase 0.25 / right 0.75, width swing_period). Dominant
    #   weight (> tracking) makes stepping the top priority — THE term that
    #   breaks the shuffle on Booster's own biped. Sole gait driver now.
    swing_period: float = 0.2          # width (in gait-cycle fraction) of each
    #   foot's swing window for feet_swing.
    swing_height: float = 2.0          # dense companion to feet_swing: ramps
    #   with how far the swing foot is off the ground, giving a smooth lift
    #   gradient toward the (binary) feet_swing payoff.
    gait_contact: float = 0.0          # DROPPED — feet_swing replaces the
    #   phase-matching reward (it sat dead-flat at the shuffle baseline anyway).
    feet_slip: float = 0.3             # PENALTY on horizontal foot speed while
    #   in contact — directly kills the skate/wiggle (Booster has this at 0.1).
    foot_clearance: float = 0.0        # DROPPED — redundant with feet_swing +
    #   swing_height, and it collapsed to ~0 under the old balance.
    feet_air_time: float = 0.0         # DROPPED — Booster doesn't use it; it
    #   never moved off 0 here (no gradient from a never-lift state).

    # head-look tracking (from the vision system) — small weight so it
    # doesn't trade off against velocity tracking.
    head_tracking: float = 0.3

    # arms-at-side regulariser (legged_gym-style): squared deviation
    # of the 8 arm DOFs from the default rest pose. RELAXED 0.05→0.02 so the
    # arms are free to swing a little for balance/angular-momentum instead of
    # being pinned at the side (the user asked for a bit of armswing). Still
    # nonzero so the arms don't drift into a flail.
    arm_pose: float = 0.02
    # armswing — small reward for the two shoulder pitches moving anti-phase
    # while walking (natural human swing). "A little": tanh-bounded ≤1, gated
    # on commanded speed, weight kept well below the tracking/gait terms.
    arm_swing: float = 0.2

    # regularizers (negative)
    action_smoothness: float = 0.002
    dof_acc: float = 1.0e-7
    torque: float = 1.0e-4
    base_motion: float = 0.1           # damp roll/pitch rates — REDUCED 0.5→0.1
    #   so it doesn't over-penalize the natural base motion of walking (a walk
    #   bobs/rolls a little); at 0.5 it added to the stand-still attractor.
    energy: float = 1.0e-4

    # termination signals
    alive: float = 0.1
    fall: float = -2.0
