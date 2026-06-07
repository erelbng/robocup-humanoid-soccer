"""Standup skill config — stability-heavy reward + diverse initial pose."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class StandupRewardWeights:
    upright: float = 3.0
    height: float = 2.0
    upright_progress: float = 10.0
    supine_situp_progress: float = 0
    explosive_rise: float = 0.0
    feet_tuck: float = 0.0
    arm_pose_dev: float = 0.2
    base_ang_vel_sway: float = 0.05
    base_lin_vel_drift: float = 0.5
    joint_vel_quiet: float = 0.0003
    action_smoothness: float = 0.05
    action_jerk: float = 0.05
    time_penalty: float = 3.0
    success_bonus: float = 400.0
    success_persistence: float = 5.0
    post_success_standing: float = 10.0
    foot_grounded_up: float = 5.0
    standing_tall: float = 5.0
    stand_pose: float = 3.0
    post_success_still: float = 3.0
    on_spot: float = 2.0
    supine_anti_flip: float = 0.5
    trunk_contact_force: float = 1.5
    knee_support: float = 0.0


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
        base, **{name: 0.0 for name in _DISCOVERY_ZEROED_WEIGHTS}
    )


@dataclass
class StandupConfig:
    num_envs: int = 1024
    max_episode_steps: int = 250  # 5 s at 50 Hz — enough for a 3 s
    dt: float = 0.02
    sim_dt: float = 0.002
    gait_freq_hz: float = 1.5  # unused but keeps obs layout uniform

    spawn_height_min: float = 0.8  # m
    spawn_height_max: float = 1.5  # m
    settle_steps: int = 1500  # sim substeps = 3.0 s at 500 Hz. HUMANUP uses 10 s
    settle_pool_rounds: int = 4  # pool_size = num_envs × rounds
    pool_max_upright: float = 0.7  # upright signal upper bound
    pool_max_height: float = 0.4  # trunk-z upper bound (m)

    joint_jitter_rad: float = 0.10

    assist_force_enabled: bool = True
    assist_force_max: float = 300.0
    assist_spring_shape: bool = True
    assist_success_target: float = 0.60
    assist_curriculum_env_steps: int = 150_000_000
    assist_cobra_gate: bool = True
    assist_under_base_soft_d: float = 0.40
    assist_cobra_z_low: float = 0.15
    assist_cobra_z_high: float = 0.35

    reward_stage: str = "deploy"  # "discovery" | "deploy"

    reg_success_ramp: bool = True
    style_stage_gate: bool = False
    style_success_ref: float = 0.5

    use_multi_critic: bool = True  # standup-host-recipe: ON. HoST's multi-
    critic_group_weights: tuple = (1.0, 1.0, 1.0)

    pose_curriculum_enabled: bool = True
    pose_curriculum_start_level: int = 0
    pose_level_thresholds: tuple = (0.55, 0.55, 0.60)
    supine_anti_flip_min_level: int = 1
    supine_anti_flip_max_level: int = 1
    pose_advance_sustain_steps: int = 1_000_000
    pose_pool_settle_steps: int = 1000  # physics substeps per round = 2.0 s at 500 Hz.
    pose_pool_side_settle_steps: int = 500  # 1.0 s at 500 Hz (was 250)
    pose_pool_side_rounds: int = 6  # compensates for higher filter rejection rate
    pose_pool_rounds: int = 2  # total snapshots = rounds × num_envs
    pose_pool_quat_noise_rad: float = 0.15
    pose_pool_joint_jitter_rad: float = 0.15
    pose_pool_max_height_margin: float = 0.30
    pose_pool_side_min_trunk_z: float = 0.10
    pose_pool_side_max_trunk_z: float = 0.20
    pose_pool_side_stabilize_torque: float = 80.0
    pose_pool_orient_dot_min: float = 0.80
    pose_pool_penetration_eps: float = 0.02
    pose_mix_random_frac: float = 0.50
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

    use_amp: bool = True
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
    entropy_coef: float = 0.002  # Restored 0.005→0.002 (mergefixes): at 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    n_steps: int = 64

    obs_dim: int = 0
    act_dim: int = 22

    level_up_log_std_pump: float = 0.5  # 0 disables the exploration pump
    level_up_reset_lr: bool = True
    desired_kl: float = 0.015

    rewards: StandupRewardWeights = field(
        default_factory=lambda: StandupRewardWeights()
    )
