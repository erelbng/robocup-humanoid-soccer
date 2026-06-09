"""Standup skill config — stability-heavy reward + diverse initial pose."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class StandupRewardWeights:
    """HoST-faithful reward weights (arXiv:2502.08378). Field names map 1:1 to
    the four groups consumed by `compute_standup_reward`. Single-critic sums all
    four; the optional multi-critic path additionally weights the GROUPS via
    `StandupConfig.critic_group_weights` (HoST: [2.5, 0.1, 1, 1])."""

    # task (bounded, saturating) — keep both at ~1; the GROUP weight scales task.
    task_orientation: float = 1.0
    task_rise: float = 1.0

    # regularization (a whisper — these are summed as a small penalty)
    regu_action_rate: float = 0.01
    regu_action_jerk: float = 0.01
    regu_dof_vel: float = 0.001

    # style (height-gated motion shape)
    style_feet_under_base: float = 1.0
    style_ground_parallel: float = 1.0
    style_feet_distance: float = 2.0
    style_ang_vel: float = 0.5

    # post-task (height-gated "hold a clean stand") + sparse success
    post_orientation: float = 1.0
    post_base_height: float = 1.0
    post_stillness: float = 1.0
    post_upper_pose: float = 1.0
    success_persistence: float = 1.0
    time_penalty: float = 1.0
    success_bonus: float = 50.0
    post_success_standing: float = 2.0


# Discovery stage zeroes the regularizers + style so the policy can first FIND a
# get-up (HumanUP two-stage). With the bounded task reward this is optional;
# reward_stage defaults to "deploy" (full set) — discovery is kept for ablations.
_DISCOVERY_ZEROED_WEIGHTS = (
    "regu_action_rate",
    "regu_action_jerk",
    "regu_dof_vel",
    "style_feet_under_base",
    "style_ground_parallel",
    "style_feet_distance",
    "style_ang_vel",
)


def discovery_weights(base: StandupRewardWeights) -> StandupRewardWeights:
    """Return a copy of `base` with regularizer + style weights zeroed — the
    Stage-1 (discovery) reward set: task + post/success terms ONLY."""
    return dataclasses.replace(
        base, **{name: 0.0 for name in _DISCOVERY_ZEROED_WEIGHTS}
    )


@dataclass
class StandupConfig:
    num_envs: int = 1024
    max_episode_steps: int = 250  # 5 s at 50 Hz — enough for a 3 s
    dt: float = 0.02
    sim_dt: float = 0.002
    gait_freq_hz: float = 1.5  # unused but keeps obs layout uniform

    # ── HoST action representation (arXiv:2502.08378) ────────────────────
    # target = current_dof_pos + beta * action (incremental / relative-to-current)
    # with beta annealed beta_start -> beta_min on success. This is the implicit
    # motion-speed bound that makes the deployed motion smooth.
    incremental_action: bool = True
    action_clip: float = 1.0        # raw policy output clipped to +/- this before beta
    beta_start: float = 1.0
    beta_min: float = 0.25
    beta_step: float = 0.02         # decrement per success-threshold crossing
    beta_success_threshold: float = 0.5   # success EMA above which beta decays
    expose_beta_in_obs: bool = True       # append beta scalar to the obs

    # ── HoST reward kernel / stage params (K1-adapted; G1 uses 0.45/0.65) ──
    rise_target: float = 0.45       # head/trunk rise above feet for "standing"
    rise_margin: float = 0.45
    orientation_threshold_kernel: float = 0.95
    orientation_margin: float = 1.0
    style_stage_rise: float = 0.30  # trunk-off-ground gate for style terms
    post_stage_rise: float = 0.45   # near-standing gate for post terms
    feet_distance_max: float = 0.45

    # ── HoST pull-force gate ─────────────────────────────────────────────
    # Fire the upward trunk force ONLY when the trunk is near-vertical
    # (projected_gravity_z < this) and wean it together with beta.
    pull_force_orient_gate: float = -0.8

    # Every reset pose is drawn from the four already-randomized named-pose
    # pools (supine/prone/side_left/side_right), each built by the
    # forced-settle + filter pipeline in _build_pose_pool. There is no
    # separate "random" settle pool.
    pool_max_upright: float = 0.7  # upright signal upper bound (fallen filter)

    joint_jitter_rad: float = 0.10

    assist_force_enabled: bool = True
    # HoST rule for porting to a new robot: pull force ≈ 60% of body weight.
    # K1 ≈ 19.67 kg → 192.9 N weight → 60% ≈ 116 N. (Single trunk link, so no
    # ×2 virtual-torso doubling like G1's URDF.) Was 300 N ≈ 155% body weight.
    assist_force_max: float = 120.0
    assist_spring_shape: bool = True
    assist_success_target: float = 0.60
    assist_curriculum_env_steps: int = 150_000_000
    assist_cobra_gate: bool = True
    assist_under_base_soft_d: float = 0.40
    assist_cobra_z_low: float = 0.15
    assist_cobra_z_high: float = 0.35

    reward_stage: str = "deploy"  # "discovery" | "deploy"

    reg_success_ramp: bool = True
    style_stage_gate: bool = True
    style_success_ref: float = 0.5

    # Tier 1 = plain single-critic (recommended start). Flip to True for the
    # optional Tier-2 multi-critic (already implemented & verified correct).
    use_multi_critic: bool = False
    critic_group_names: tuple = ("task", "regu", "style", "post")
    critic_group_weights: tuple = (2.5, 0.1, 1.0, 1.0)  # HoST group weights

    pose_curriculum_enabled: bool = True
    pose_curriculum_start_level: int = 0
    # L0→L1 (prone solid before adding supine), L1→L2 (add side poses). The
    # four named pools are each already heavily randomized, so L2 (terminal)
    # draws equally from all four — no separate "random" fallen level.
    pose_level_thresholds: tuple = (0.55, 0.55)
    supine_anti_flip_min_level: int = 1
    supine_anti_flip_max_level: int = 1
    pose_advance_sustain_steps: int = 1_000_000
    pose_pool_settle_steps: int = 1000  # physics substeps per round = 2.0 s at 500 Hz.
    pose_pool_side_settle_steps: int = 500  # 1.0 s at 500 Hz (was 250)
    # After the pinned side settle, RELEASE the trunk pin and free-step this
    # many substeps to verify the pose stays on its side (arm/leg-random side
    # poses). A config that doesn't brace rolls out of class and is culled.
    pose_pool_side_verify_steps: int = 300
    # At-rest gate: reject side snapshots whose base angular velocity exceeds
    # this (rad/s) after the unpinned verify — they're still mid-roll.
    pose_pool_side_max_ang_vel: float = 0.5
    pose_pool_side_rounds: int = 40  # compensates for higher filter rejection rate
    pose_pool_rounds: int = 2  # total snapshots = rounds × num_envs
    # Prone/supine wide arm+leg random targets raise the rejection rate, so
    # they build more rounds than pose_pool_rounds.
    pose_pool_limb_random_rounds: int = 10
    # Inset (rad) from each joint's hard URDF limit when sampling random arm/leg
    # targets, so targets don't sit exactly at the mechanical stop.
    pose_pool_arm_random_limit_margin: float = 0.10
    pose_pool_quat_noise_rad: float = 0.15
    pose_pool_joint_jitter_rad: float = 0.15
    pose_pool_max_height_margin: float = 0.30
    pose_pool_side_min_trunk_z: float = 0.10
    pose_pool_side_max_trunk_z: float = 0.20
    pose_pool_side_stabilize_torque: float = 80.0
    pose_pool_orient_dot_min: float = 0.80
    pose_pool_penetration_eps: float = 0.02
    pose_mix_bias_start: float = 0.80
    pose_mix_bias_env_steps: int = 15_000_000

    recovery_curriculum_enabled: bool = False
    recovery_start_stage: int = 0
    recovery_crouch_heights: tuple = (0.47, 0.38, 0.30)
    recovery_bend_scales: tuple = (0.5, 1.0, 1.5)
    recovery_crouch_delta_hip: float = -0.6
    recovery_crouch_delta_knee: float = 0.9
    recovery_crouch_delta_ankle: float = -0.5
    recovery_stage_thresholds: tuple = (0.60, 0.55, 0.50)
    recovery_advance_sustain_steps: int = 1_000_000
    recovery_crouch_quat_noise_rad: float = 0.05  # ≈ ±3° tilt
    recovery_crouch_joint_jitter_rad: float = 0.05  # ≈ ±3° per joint
    recovery_crouch_settle_steps: int = 500  # 1.0 s at 500 Hz

    use_frequency_gains: bool = False
    gain_natural_freq_hz: float = 4.0
    gain_damping_ratio_leg: float = 1.5
    gain_damping_ratio_knee: float = 1.0

    proprio_only: bool = False
    target_height: float = 0.55
    upright_threshold: float = 0.92
    success_hold_steps: int = 50  # 1.0 s at 50 Hz
    success_hold_steps_start: int = 15  # 0.3 s at 50 Hz
    hold_curriculum_env_steps: int = 25_000_000

    upright_threshold_start: float = 0.80  # cosine ~37° tilt — kneel-ish
    target_height_start: float = 0.40  # frame_success at z > 0.30
    threshold_curriculum_env_steps: int = 25_000_000

    foot_grounded_max_z: float = 0.10  # feet "on ground" when z < this
    trunk_up_min_z: float = 0.30  # trunk "lifted" when z > this
    feet_under_base_soft_d: float = 0.40  # soft ramp: 1 at d=0, 0 at d≥this
    success_under_base_max_d: float = 0.25  # hard: feet must be within this of base
    success_foot_max_z: float = 0.12  # hard: feet must be on the ground
    standing_tall_min_z: float = 0.30  # signal starts ramping here
    standing_tall_max_z: float = 0.55  # signal saturates at K1 standing height

    time_to_stand_tau_steps: float = 60.0

    explosive_rise_v_cap: float = 0.8

    stand_pose_dev_scale: float = 0.5  # sharper basin (was 1.0): pulls harder to
    stand_pose_success_ref: float = 0.5

    stand_target_hip_abduction: float = 0.0
    feet_under_base_plateau_d: float = 0.22

    post_success_still_jv_scale: float = 3.0  # joint kinetic-energy scale
    post_success_still_v_scale: float = 0.2  # base linear-velocity scale

    on_spot_tol: float = 0.6

    trunk_contact_force_thresh: float = 280.0
    trunk_contact_force_scale: float = 196.0

    knee_contact_force_thresh: float = 20.0
    knee_support_min_z: float = 0.20
    knee_support_max_z: float = 0.45

    progress_ratchet: bool = True

    reset_success_ema_on_level_up: bool = True

    use_amp: bool = False  # HoST uses NO AMP; get-up style comes from reward terms
    amp_motion_file: str = "data/motions/k1_standup_amp.npz"
    amp_reward_coef: float = 0.5
    amp_task_reward_coef: float = 0.5
    amp_disc_lr: float = 6e-5
    amp_disc_updates: int = 1
    amp_disc_batch: int = 4096
    amp_grad_penalty: float = 5.0
    # WHEN the AMP style (motion-prior) reward is active, in the multi-critic path:
    #   "curriculum" — gated by the env's two-stage style_scale (0 during the pose
    #                  curriculum, on at the final level) → AMP only REFINES motion
    #                  after the robot can already stand. Safe default.
    #   "always"     — style active from step 0 at full strength → AMP GUIDES
    #                  exploration toward the reference get-up the whole run.
    #   "anneal"     — style gate decays 1.0 → amp_style_floor over
    #                  amp_style_anneal_steps → strong early guidance that fades so
    #                  the task reward takes over
    amp_style_schedule: str = "curriculum"
    amp_style_anneal_steps: int = 50_000_000
    amp_style_floor: float = 0.0
    # Per-foot z (m) subtracted in amp_observation so a PLANTED foot reads ~0
    # clearance, matching the MuJoCo reference (planted foot ≈0). Genesis foot_link
    # stands at ~0.038 m; the old 0.02 left a constant ~0.018 m offset the
    # discriminator could exploit as a free shortcut.
    amp_foot_z_offset: float = 0.0377

    total_timesteps: int = 300_000_000  # two-stage run: ~L0-L3 curriculum
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01  # HoST uses 0.01 (more exploration for get-up)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    n_steps: int = 64

    # HoST init_noise_std = 0.8 -> init_log_std = ln(0.8) ~= -0.22 (we default
    # -0.5). Plumbed into create_policy via the OPTIONAL train_skill change in
    # Appendix C; harmless if that change is not applied.
    init_log_std: float = -0.22

    obs_dim: int = 0
    act_dim: int = 22

    level_up_log_std_pump: float = 0.5  # 0 disables the exploration pump
    level_up_reset_lr: bool = True
    desired_kl: float = 0.015

    rewards: StandupRewardWeights = field(
        default_factory=lambda: StandupRewardWeights()
    )
