"""Standup skill config — stability-heavy reward + diverse initial pose."""

from __future__ import annotations

import dataclasses
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
    # exploration through the high-motion transition zone. Bumped from
    # 5.0 to 10.0 because the policy got stuck in a sit/kneel attractor
    # (z ≈ 0.25, up ≈ 0.85) where Δup ≈ 0; a stronger progress weight
    # amplifies the tiny gradient that pulls the policy onward.
    upright_progress: float = 10.0     # weight on max(0, up_t - up_{t-1})

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
    time_penalty: float = 3.0          # per step until sustained-success
                                       # (≥2.5 to make floor net-negative:
                                       #  upright≈0.5×3 + height≈0.3 − 3.0 < 0)
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

    # Anti-gaming term — pays only when BOTH feet are on the ground AND
    # the trunk is lifted. Closes the local-optima loophole where the
    # policy gets partial upright/height credit from bridge / shoulder-
    # stand / sprawled poses that never touch its feet to the floor.
    # Smooth multiplicative gate (left_foot_proximity × right_foot_proximity
    # × trunk_lift_score) so PPO gets gradient toward the threshold even
    # before satisfying it. At a fallen pose: 0 (trunk down). At any
    # bridge / sprawl: 0 (feet not grounded). At a squat (trunk ≥ 0.30):
    # saturated at ~1.0. Doesn't distinguish squat from full standing.
    foot_grounded_up: float = 5.0

    # Continues where `foot_grounded_up` saturates. Same feet-grounded
    # gate × trunk ramp on [0.30, 0.55] — 0 at squat, 1.0 at full stand
    # height. Stacks ADDITIVELY on top of foot_grounded_up so the squat
    # reward is unchanged (no destabilising regression), but full
    # extension pays an extra ~5/step → ~1250 over the post-squat
    # trajectory. Pulls the policy out of the squat local optimum.
    standing_tall: float = 5.0


# Regularizer weights zeroed in the "discovery" reward stage. These are
# motion-quality / deployability shaping terms — useful for a SMOOTH final
# motion, but (per HumanUP, arXiv:2502.12152) early regularization blocks
# task discovery. Stage 1 turns them OFF so the policy can find *any*
# standup; stage 2 ("deploy") turns them back on to refine the motion.
_DISCOVERY_ZEROED_WEIGHTS = (
    "arm_pose_dev",
    "base_ang_vel_sway",
    "base_lin_vel_drift",
    "joint_vel_quiet",
    "action_smoothness",
    "action_jerk",
)


def discovery_weights(base: StandupRewardWeights) -> StandupRewardWeights:
    """Return a copy of `base` with all motion-regularizer weights zeroed
    — the Stage-1 (discovery) reward set: upright + height + progress +
    feet-grounded + standing-tall + speed/success terms ONLY."""
    return dataclasses.replace(
        base, **{name: 0.0 for name in _DISCOVERY_ZEROED_WEIGHTS})


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
    settle_steps: int = 1500            # sim substeps = 3.0 s at 500 Hz. HUMANUP uses 10 s
                                       # to resolve self-collisions from randomised DOFs;
                                       # 3 s is a practical GPU compromise. Was 300 (0.6 s) —
                                       # too short to actually let the pose settle.
    settle_pool_rounds: int = 4          # pool_size = num_envs × rounds
    # Filter the pool: keep only states with a clearly fallen robot.
    # Avoids "robot landed upright" trivial-success starts.
    pool_max_upright: float = 0.7        # upright signal upper bound
    pool_max_height: float = 0.4         # trunk-z upper bound (m)
    # Joint noise added on every pool-sample → effectively unlimited
    # per-reset variation on top of the discrete pool. 0.10 rad ≈ ±6°.
    # Raised from 0.03 (±1.7°, too small to give real start diversity);
    # the pool was already settled at this magnitude so penetration risk
    # is low.
    joint_jitter_rad: float = 0.10

    # ── Assistive-force curriculum (HoST, arXiv:2502.08378) ──────────
    # The single highest-leverage exploration aid for standup: apply a
    # DECAYING upward "support" force on the trunk — like helping an
    # infant stand. Early in training the force nearly holds the robot up,
    # so the policy reliably reaches the standing pose and learns what the
    # whole fallen→upright trajectory feels like; the force then weans to
    # zero so the final policy stands unaided.
    assist_force_enabled: bool = True
    # Peak upward force (N) at full assist. Robot weighs ~20 kg (≈196 N),
    # so 160 N supports most of body weight without launching it — the
    # policy still has to do the last ~20% and get its feet underneath.
    assist_force_max: float = 300.0
    # Spring shape: force ramps with height deficit (target − z), so it's
    # strongest when fully fallen and releases to ~0 near standing height.
    # Upward-only (never pushes the trunk down). This keeps the assist from
    # fighting the robot once it's nearly up.
    assist_spring_shape: bool = True
    # PERFORMANCE-COUPLED assist (primary driver). The assist fraction is
    # tied directly to the success EMA:
    #   success_frac = clip(1 - success_ema / assist_success_target, 0, 1)
    # → full support at zero competence, fading to ~0 as the policy reaches
    # `assist_success_target`. This auto-couples the assist to the pose
    # curriculum: a level-up introduces a harder pose, the success EMA drops,
    # and the assist rises back to help — no explicit per-level logic needed.
    # Set slightly ABOVE the highest advance threshold (0.60 vs 0.55/0.60) so
    # that at the moment of advancement the assist is already LOW (but not
    # exactly 0). The old `assist_min_success`/`assist_min_frac` floor is
    # replaced by this: it pinned the assist at 0.60 right up to the 0.50
    # advance threshold, so the policy advanced *before* it was weaned.
    assist_success_target: float = 0.60
    # Time-decay BACKSTOP (multiplicative). On its own the success coupling
    # would let a policy that plateaus BELOW the target lean on the assist
    # forever. Multiply success_frac by a linear time ramp (1.0 → 0.0 over
    # this many cumulative env-steps) so support is always weaned out
    # eventually, regardless of success. Should be on the same order as the
    # discovery phase length (a good chunk of total_timesteps).
    assist_curriculum_env_steps: int = 150_000_000
    # Anti-cobra assist gate. The naive fix — multiply the assist by a "feet
    # under base" score — is WRONG: a freshly fallen robot also has its feet
    # splayed out, so that gate zeroes the assist in *every* legitimate fall,
    # killing the HoST bootstrap (≈167 N → 0 N at z≈0.13). The real cobra
    # marker is TRUNK LIFTED *and* FEET BEHIND simultaneously — a fallen
    # robot has the trunk DOWN, so it must still be fully supported. We only
    # throttle the assist once the chest is up but the feet haven't tucked:
    #   cobra_factor = 1 - trunk_lifted · (1 - feet_under_base)
    # → fallen: ~1 (full support), cobra: ~0 (cut), real stand: ~1 (deficit≈0
    #   makes the force ~0 anyway). Set False for unconditional upward assist.
    assist_cobra_gate: bool = True
    # Soft ramp width (m) for the feet-under-base term inside the cobra gate:
    # 1 at foot-under-base (d=0), → 0 at d ≥ this. Matches the reward-side
    # feet_under_base_soft_d so assist and on-feet reward agree on geometry.
    assist_under_base_soft_d: float = 0.40
    # Trunk-z band (m) for the `trunk_lifted` ramp in the cobra gate: 0 at
    # z ≤ z_low (fallen — full support), 1 at z ≥ z_high (chest clearly up —
    # cobra territory if feet still behind). Fallen prone/supine sits ≈0.13.
    assist_cobra_z_low: float = 0.15
    assist_cobra_z_high: float = 0.35

    # Reward stage. "discovery" (Stage 1) zeroes the motion regularizers so
    # the policy can find ANY standup; "deploy" (Stage 2) uses the full
    # weight set for a smooth, sim2real-ready motion. Train discovery →
    # warm-start a deploy run from its checkpoint via --init-from.
    reward_stage: str = "discovery"           # "discovery" | "deploy"

    # ── Multi-critic PPO (HoST, arXiv:2502.08378) ────────────────────
    # One value head per reward group (STANDUP_CRITIC_GROUPS: task / reg /
    # success) instead of a single critic over the full heterogeneous
    # reward. HoST found single-critic = ~zero success because the value
    # net can't fit a return mixing a +400 terminal pulse with dense [0,1]
    # shaping and small penalties. Each critic fits one homogeneous-scale
    # group; advantages are normalized per group then weighted-summed.
    # Standup-only (the other skills keep single-critic PPO).
    use_multi_critic: bool = False
    # Aggregation weights, aligned to STANDUP_CRITIC_GROUPS = (task, reg,
    # success). Bias toward `task` early to drive the get-up; `reg` is ~0
    # in the discovery stage anyway (its weights are zeroed).
    critic_group_weights: tuple = (1.0, 1.0, 1.0)

    # ── Pose difficulty curriculum (discrete, L0–L3) ──────────────────────
    # Each level presents harder starting poses. Advancement is gated on both
    # sustained EMA success and minimum time at threshold.
    #
    #   L0: supine only               — easiest single entry pose
    #   L1: supine + prone (50/50)    — add the harder front recovery
    #   L2: all 4 named poses (25% each) — + side_left + side_right
    #   L3: named 50% + random fallen 50% — full robustness
    #
    # Supine (roll/sit up) and prone (arm push-up → tuck → stand) are
    # different motor strategies, and prone additionally pulls toward the
    # cobra. Training both 50/50 from the start averages the gradients so the
    # policy learns neither cleanly, and the combined success EMA stalls below
    # the gate when one pose lags — so prone is held back to L1 and gets there
    # only after supine is solid. Recovery from fully-fallen poses is
    # bootstrapped by the decaying assist force. Set
    # pose_curriculum_start_level=3 to train directly on the full mixed
    # distribution (no curriculum ramp).
    pose_curriculum_enabled: bool = True
    pose_curriculum_start_level: int = 0

    # EMA threshold to advance FROM each level. Element i controls the
    # transition from level i → i+1. Length = (num_levels - 1) = 3.
    #   L0→L1 (supine solid before adding prone), L1→L2, L2→L3.
    # Set just below assist_success_target (0.60) so advancement requires
    # genuine, largely-unassisted competence — by the time the success EMA
    # hits 0.55 the performance-coupled assist is already low (~0.08).
    pose_level_thresholds: tuple = (0.55, 0.55, 0.60)

    # How many cumulative env-steps the EMA must CONTINUOUSLY stay above the
    # threshold before advancing. Prevents a single lucky spike triggering a
    # level jump. At 2048 envs, 1M steps ≈ 488 control steps ≈ ~10 s.
    pose_advance_sustain_steps: int = 1_000_000

    # Named-pose pool build parameters. Same settle mechanism as the main
    # settle pool but starting from the named pose's reference orientation.
    pose_pool_settle_steps: int = 1000     # physics substeps per round = 2.0 s at 500 Hz.
                                            # Was 150 (0.3 s) — too short to damp bouncing
                                            # after 0.19 s free-fall from spawn height.
    pose_pool_rounds: int = 2               # total snapshots = rounds × num_envs
    pose_pool_quat_noise_rad: float = 0.30  # Gaussian σ on orientation perturbation
    # Joint noise for named-pose pools. HUMANUP + X-Loco recommend large joint noise
    # so each pool sample is a distinct fallen configuration rather than a rigid clone
    # of the canonical pose. 0.30 rad σ ≈ ±17° per joint (was hardcoded 0.05 = ±3°).
    pose_pool_joint_jitter_rad: float = 0.30
    # Filter: keep pool entries with trunk_z < trunk_height + this margin.
    # Named poses settle to ~0.13 m; 0.30 m margin catches bounced states.
    pose_pool_max_height_margin: float = 0.30

    # L3: random-pool fraction (rest drawn equally from the 4 named poses).
    pose_mix_random_frac: float = 0.50

    # Sim2real flag. Contact-obs addons (foot/hand z + contact bool)
    # require knowing the absolute floor position — privileged info the
    # real robot doesn't have. Set True to remove the contact dims from
    # the policy obs: training is slower (no contact signal) but the
    # resulting policy is directly deployable. Production sim2real path:
    # leave this False and use the teacher-student pipeline (`--mode
    # teacher` → `--mode student`) so the student learns to estimate
    # contact implicitly from proprio.
    proprio_only: bool = False

    # Sustained-success thresholds (END of curriculum — see _start values
    # below). A standup is "done" once `success_hold_steps` consecutive
    # frames satisfy both upright and height conditions.
    target_height: float = 0.55
    upright_threshold: float = 0.92            # cosine ~23° tilt max
    success_hold_steps: int = 50               # 1.0 s at 50 Hz

    # Curricula on the success criteria. All three independently ramp
    # from their `_start` value to the final value over
    # `*_curriculum_env_steps` cumulative env-steps.
    #
    # Why three curricula:
    #   * hold_steps tightens HOW LONG you must hold (1 s = real stability)
    #   * upright_threshold tightens HOW UPRIGHT (0.92 = deployment quality)
    #   * target_height tightens HOW TALL (0.55 = K1 standing height)
    #
    # Without the threshold curricula the policy commonly gets stuck in a
    # sit/kneel attractor at z ≈ 0.25, up ≈ 0.85 that pays well from the
    # dense terms but never triggers `frame_success` (which requires
    # up > 0.92 AND z > 0.45), so the terminal bonus and post-success
    # standing reward never fire. The looser starting criteria give the
    # policy partial credit at intermediate poses, then tighten as it
    # masters the harder ones.
    success_hold_steps_start: int = 15         # 0.3 s at 50 Hz
    hold_curriculum_env_steps: int = 25_000_000

    upright_threshold_start: float = 0.80      # cosine ~37° tilt — kneel-ish
    target_height_start: float = 0.40          # frame_success at z > 0.30
    threshold_curriculum_env_steps: int = 25_000_000

    # Thresholds for the `foot_grounded_up` anti-gaming reward.
    foot_grounded_max_z: float = 0.10          # feet "on ground" when z < this
    trunk_up_min_z: float = 0.30               # trunk "lifted" when z > this

    # ── Feet-under-base (anti-cobra / anti-push-up) ──────────────────────
    # The missing discriminator between "standing on its feet" and "lying
    # with its feet flat and splayed". `_feet_grounded_score` only checks
    # foot_z, which a prone / cobra / L-sit pose satisfies trivially — so
    # the policy got ~10/step from foot_grounded_up + standing_tall while
    # keeping its legs flat on the floor and letting the assist hold the
    # trunk up. These gate both stand-on-feet rewards (soft, for gradient)
    # and success detection (hard, so the +400 bonus can't be farmed from
    # a propped pose) on the horizontal foot↔base distance.
    feet_under_base_soft_d: float = 0.40       # soft ramp: 1 at d=0, 0 at d≥this
    success_under_base_max_d: float = 0.25     # hard: feet must be within this of base
    success_foot_max_z: float = 0.12           # hard: feet must be on the ground

    # Thresholds for the `standing_tall` reward — pulls the policy from
    # squat (trunk ~0.30) to full standing (trunk ~0.55).
    standing_tall_min_z: float = 0.30          # signal starts ramping here
    standing_tall_max_z: float = 0.55          # signal saturates at K1 standing height

    # Time-scaling for the terminal bonus. Bonus *= exp(-t_first / tau).
    # τ=60 steps (1.2 s): the speed reward must have a real GRADIENT in the
    # range standups actually occur. Diagnosis (2026-05-30): with τ=150,
    # observed stands clustered ~70 steps (1.4 s) and never got faster —
    # exp(-t/150) is nearly flat there (70→35 steps only +27% bonus). At
    # τ=60, 70→35 steps is 0.31→0.56 (+82%), a strong pull toward faster.
    # NOTE: revisit after the torque-limited retrain — realizable (≤40 N·m)
    # standups are slower than the old 780-N·m ones, so the realistic
    # stand-time (and thus the ideal τ) may shift up.
    time_to_stand_tau_steps: float = 60.0

    # ── PPO defaults (training.algorithms.ppo) ────────────────────
    total_timesteps: int = 50_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.002  # was 0.01 — stops std runaway; std stays moderate
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    n_steps: int = 64

    obs_dim: int = 0
    act_dim: int = 22

    rewards: StandupRewardWeights = field(
        default_factory=lambda: StandupRewardWeights())
