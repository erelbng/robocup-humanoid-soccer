"""Standup skill config — stability-heavy reward + diverse initial pose."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StandupRewardWeights:
    """Weights for the standup composite reward.

    Speed: ONE term (`time_penalty`, dense) + ONE terminal (`success_bonus`,
    time-scaled). Stability: several terms — all quadratic deviations that
    vanish at the standing equilibrium, so the optimum stays at 'upright
    + still' and isn't pulled away by any one term.
    """
    # Primary shaping — positive everywhere in [0, 1], smooth monotonic
    # gradient from upside-down → sideways → upright.
    upright: float = 3.0               # (cos(trunk-z, world-z) + 1) / 2
    height: float = 2.0                # gaussian around target_height (σ=0.3)
    # Progress shaping — paid only for active uprightening (Δup > 0).
    # Without this, side-plank (up≈0.7) is a stable basin — the marginal
    # gradient toward standing is too weak to motivate PPO's risk-averse
    # exploration through the high-motion transition zone.
    upright_progress: float = 5.0      # weight on max(0, up_t - up_{t-1})

    # Arm-pose deviation penalty — drives the final standing pose to
    # arms-hanging-at-the-sides (the corrected K1 default with shoulder
    # rolls at ±π/2). Phase-gated on a [0.5, 0.85] band so arms are
    # completely free during the entire recovery (up < 0.5) and the
    # penalty only ramps in as the robot approaches its final pose.
    # Many standup motions need arm push-off through up≈0.3–0.5, so the
    # gate has to stay open through that range.
    arm_pose_dev: float = 0.2          # Σ (q_arm - q_arm_rest)² × arm_gate

    # Stability penalties — ALL phase-gated by `near_upright_gate`, so
    # they vanish during deep recovery (the policy needs full motion
    # freedom to actually stand up) and ramp in only as we approach the
    # standing pose. They vanish again at the equilibrium itself.
    base_ang_vel_sway: float = 0.05    # ωx² + ωy² — roll/pitch rate
    base_lin_vel_drift: float = 0.5    # ||v||² — trunk linear drift
    joint_vel_quiet: float = 0.001     # Σ q̇² — joint kinetic activity
    action_smoothness: float = 0.1     # (a - a_{-1})² — first derivative
    action_jerk: float = 0.1           # (a - 2 a_{-1} + a_{-2})² — jitter

    # Speed signal — exactly one dense term + one terminal pulse.
    # Default τ=150 steps (3.0 s) keeps the bonus meaningful across the
    # realistic standup time range: a 1 s stand pays ~330, a 2 s stand
    # ~150, a 3 s stand ~100, a 4 s stand ~55. Sub-second standups still
    # get the largest pulse but slow ones are no longer disqualified.
    time_penalty: float = 1.0          # per step until sustained-success
    success_bonus: float = 400.0       # paid on streak completion, scaled
                                       #   by exp(-t_first / tau)
    success_persistence: float = 5.0   # per step while in the hold window

    # Post-success standing reward. The env does NOT terminate on
    # sustained success — episodes run to the full MAX_EPISODE_STEPS so
    # the robot must prove it can keep standing after the bonus is paid.
    # Each frame after success where frame_success is still True earns
    # this reward; falling back over forfeits the rest of the episode
    # (no termination penalty, but the opportunity cost is huge — a
    # 5 s episode with a 1.5 s standup yields ~175 steps × 10 = 1750
    # of post-success reward, easily dominating any other term).
    post_success_standing: float = 10.0


@dataclass
class StandupConfig:
    # ── env ────────────────────────────────────────────────────────
    num_envs: int = 1024
    max_episode_steps: int = 250       # 5 s at 50 Hz — enough for a 3 s
                                       #   standup + 2 s margin
    dt: float = 0.02
    sim_dt: float = 0.002
    gait_freq_hz: float = 1.5          # unused but keeps obs layout uniform

    # Initial-pose generation: spawn robot in the air with random
    # orientations + random joint perturbations, let physics settle it
    # in parallel across all envs, snapshot the resulting states into a
    # pool. Each subsequent reset samples uniformly from the pool — no
    # mid-rollout physics step needed (which would desynchronize the
    # other envs).
    spawn_height_min: float = 0.8        # m
    spawn_height_max: float = 1.5        # m
    settle_steps: int = 300              # sim substeps (0.6 s at 500 Hz)
    settle_pool_rounds: int = 4          # pool_size = num_envs × rounds
    # Filter the pool: keep only states with a clearly fallen robot.
    # Avoids "robot landed upright" trivial-success starts.
    pool_max_upright: float = 0.7        # upright signal upper bound
    pool_max_height: float = 0.4         # trunk-z upper bound (m)
    # Small joint noise added on every pool-sample → effectively
    # unlimited per-reset variation on top of the discrete pool.
    joint_jitter_rad: float = 0.03

    # Sim2real flag. Contact-obs addons (foot/hand z + contact bool)
    # require knowing the absolute floor position — privileged info the
    # real robot doesn't have. Set True to remove the contact dims from
    # the policy obs: training is slower (no contact signal) but the
    # resulting policy is directly deployable. Production sim2real path:
    # leave this False and use the teacher-student pipeline (`--mode
    # teacher` → `--mode student`) so the student learns to estimate
    # contact implicitly from proprio.
    proprio_only: bool = False

    # Sustained-success thresholds. A standup is "done" once
    # `success_hold_steps` consecutive frames satisfy both upright and
    # height conditions.
    target_height: float = 0.55
    upright_threshold: float = 0.92            # cosine ~23° tilt max
    success_hold_steps: int = 50               # 1.0 s at 50 Hz — END of curriculum

    # Curriculum on the hold-window length. Starts at `hold_start` (easier
    # to register a success → policy can discover partial standups and
    # bootstrap into the full motion) and ramps linearly to
    # `success_hold_steps` over `hold_curriculum_env_steps` cumulative
    # env-steps. Without this the 1.0 s requirement is so far from the
    # initial policy's capability that the terminal bonus is effectively
    # never paid → flat learning curve.
    success_hold_steps_start: int = 15         # 0.3 s at 50 Hz
    hold_curriculum_env_steps: int = 25_000_000

    # Time-scaling for the terminal bonus. Bonus *= exp(-t_first / tau).
    # τ=150 steps (3.0 s) keeps the bonus meaningful for realistic
    # standups: a 1 s standup pays ~330, a 2 s standup ~150, a 3 s
    # standup ~100. The previous τ=40 (0.8 s) decayed so fast that
    # 3 s standups paid only ~9 — less than the side-plank attractor.
    time_to_stand_tau_steps: float = 150.0

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
