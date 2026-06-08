"""
Project-wide configuration for RoboCup Humanoid Soccer RL Training.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─── Paths ────────────────────────────────────────────────────────────────────
# PROJECT_ROOT must point at the repo root, not this file's directory.
# `os.path.dirname(__file__)` of `configs/config.py` is `configs/`, so we go
# up one level. Every downstream asset path depends on this being correct;
# the MuJoCo evaluator silently falls back to an empty robot-less scene if
# FIELD_MJCF doesn't exist, which used to make "no robot visible" symptoms
# appear out of nowhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIELD_DIR = os.path.join(MODELS_DIR, "field")
ROBOT_DIR = os.path.join(MODELS_DIR, "robot")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")

FIELD_JSON = os.path.join(CONFIGS_DIR, "field_hsl_2026.json")
FIELD_MJCF = os.path.join(FIELD_DIR, "field_robocup.xml")
FIELD_INFO = os.path.join(FIELD_DIR, "field_info.json")

# K1 robot assets (cloned from BoosterRobotics/booster_assets)
K1_URDF = os.path.join(ROBOT_DIR, "K1", "K1_22dof.urdf")
K1_MJCF = os.path.join(ROBOT_DIR, "K1", "K1_22dof.xml")


# ─── Robot Config ────────────────────────────────────────────────────────────


@dataclass
class K1RobotConfig:
    """Booster K1 robot configuration."""

    name: str = "booster_k1"
    num_dofs: int = 22
    height: float = 0.95  # meters
    mass: float = 20.0  # kg

    # Joint names in order (from booster_assets)
    joint_names: Tuple[str, ...] = (
        "AAHead_yaw",
        "Head_pitch",
        "ALeft_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "ARight_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    )

    # Leg joint indices (for locomotion-focused training)
    leg_joint_indices: Tuple[int, ...] = (
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    )
    arm_joint_indices: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
    head_joint_indices: Tuple[int, ...] = (0, 1)

    # Default standing pose (radians). Slight hip/knee/ankle bend
    # (≈ 11/23/14 deg) for a natural standing posture; arms hanging at
    # the sides.
    #
    # IMPORTANT — the K1 URDF's joint-zero pose is the T-pose (arms
    # extended straight out to the sides), confirmed by MuJoCo FK. To
    # actually hang the arms vertically along the body, both shoulder
    # rolls have to rotate ±π/2 (left negative, right positive). Other
    # arm joints stay at 0 (no shoulder pitch swing, no elbow bend, no
    # forearm yaw). Don't put 0 here for shoulder roll unless you want
    # a T-pose default — which the obs (`jpos − default`), the PD reset
    # target, and the standup `arm_pose_dev` penalty all read from this
    # field.
    default_joint_pos: Tuple[float, ...] = (
        0.0,
        0.0,  # head: yaw, pitch
        0.0,
        -1.4,
        0.0,
        0.0,  # left arm:  shoulder pitch, roll(-π/2), elbow pitch, elbow yaw
        0.0,
        1.4,
        0.0,
        0.0,  # right arm: shoulder pitch, roll(+π/2), elbow pitch, elbow yaw
        -0.2,
        0.0,
        0.0,  # left hip: pitch, roll, yaw
        0.4,
        -0.15,
        0.0,  # left knee, ankle pitch, ankle roll
        -0.2,
        0.0,
        0.0,  # right hip
        0.4,
        -0.15,
        0.0,  # right knee, ankle pitch, ankle roll
    )

    # PD gains. T1 uses much higher gains on hip+knee (200/5) and lower
    # on ankles (50/1) since ankle joints can't physically push as hard.
    # We expose per-joint-group gains here; per-joint plumbing happens
    # in the skill env's _init_genesis() when it calls set_dofs_kp.
    kp: float = 30.0  # legacy uniform fallback
    kd: float = 1.0
    # Per-joint-group PD gains. LEG gains set to NaoHTWK's own K1 RL config
    # (github.com/NaoHTWK/htwk-gym envs/K1/Parameter_Walk.yaml) — the
    # authoritative reference for THIS robot (same 0.002 dt / decimation-10 /
    # default angles as us): stiffness {Hip:100, Knee:100, Ankle:50}, damping
    # {Hip:2, Knee:2, Ankle:1}. Replaces the earlier guesses (flat kp=40, and
    # standup's frequency-derived ~30/60/36) which were far too soft on kp — the
    # legs couldn't hold the robot up. Ankle is intentionally softer (50/1):
    # ankles have less mechanical authority. Arm/head kept as before (htwk-gym's
    # walk policy doesn't actuate them; Booster's B1 SDK example uses shoulder
    # ~40 / elbow ~20 / head ~5, so our 20/10 is in range).
    kp_hip: float = 100.0
    kp_knee: float = 100.0
    kp_ankle: float = 50.0
    kp_arm: float = 20.0
    kp_head: float = 10.0
    kd_hip: float = 2.0
    kd_knee: float = 2.0
    kd_ankle: float = 1.0
    kd_arm: float = 0.5
    kd_head: float = 0.5

    # Per-joint motor ARMATURE (reflected rotor inertia), kg·m² — from the K1
    # MJCF. CRITICAL for sim2sim/sim2real: the URDF has no <dynamics>, so
    # Genesis trained with ZERO armature while MuJoCo (and the real motors)
    # have it — and for the legs it's 0.05-0.10, i.e. 5-50× the link's own
    # inertia, so it DOMINATES the joint dynamics. A policy trained without it
    # cannot transfer (the MuJoCo joints feel far heavier). The skill env now
    # applies these via `set_dofs_armature` at scene build so Genesis joint
    # dynamics match MuJoCo.
    armature_head: float = 0.002
    armature_arm: float = 0.001
    armature_hip_pitch: float = 0.0478125
    armature_hip_roll: float = 0.0339552
    armature_hip_yaw: float = 0.0282528
    armature_knee: float = 0.095625
    armature_ankle: float = 0.0565
    armature_default: float = 0.01


# ─── Training Config ─────────────────────────────────────────────────────────


@dataclass
class RewardWeights:
    """Configurable reward weights for the multi-objective reward function."""

    # Phase 1: Single robot skills
    forward_velocity: float = 2.0
    tracking_ball: float = 1.5
    ball_to_goal: float = 3.0
    kick_reward: float = 10.0
    dribble_control: float = 2.0
    alive_bonus: float = 0.5
    upright_bonus: float = 1.0
    energy_penalty: float = -0.01
    joint_limit_penalty: float = -1.0
    fall_penalty: float = -10.0
    action_smoothness: float = -0.1
    foot_contact_reward: float = 0.5

    # Phase 2: Match tactics (used in multi-agent fine-tuning)
    team_ball_possession: float = 2.0
    goal_scored: float = 50.0
    goal_conceded: float = -50.0
    positioning: float = 1.0
    passing: float = 5.0
    defensive_coverage: float = 1.5
    offsides_penalty: float = -5.0
    aggressiveness: float = 0.0  # tuneable: >0 = more aggressive play

    def scale_aggressiveness(self, level: float):
        """Scale reward weights based on aggressiveness 0.0 - 1.0."""
        self.forward_velocity *= 1.0 + 0.5 * level
        self.kick_reward *= 1.0 + level
        self.ball_to_goal *= 1.0 + 0.5 * level
        self.defensive_coverage *= 1.0 - 0.5 * level
        self.aggressiveness = level


@dataclass
class Phase1Config:
    """Single-robot dribble and shoot training config."""

    env_name: str = "K1SoccerDribbleShoot"
    num_envs: int = 4096
    max_episode_steps: int = 1000
    dt: float = 0.02  # 50Hz control
    sim_dt: float = 0.002  # 500Hz physics
    action_repeat: int = 10  # sim_dt * action_repeat = dt

    # Observation space
    # 78 base + 5 style command = 83 by default. The env enforces this
    # automatically based on use_style_command.
    obs_dim: int = 83
    base_obs_dim: int = 78
    act_dim: int = 22  # joint position targets for all 22 DOF

    # Training
    total_timesteps: int = 200_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 32768
    n_epochs: int = 5
    n_steps: int = 64

    # Curriculum
    # Order matters: standup follows stand so the policy can recover
    # before any forward-motion stage exposes it to falls.
    use_curriculum: bool = True
    curriculum_stages: Tuple[str, ...] = (
        "stand",  # learn to stand and balance
        "standup",  # recover to upright from a fallen pose
        "walk",  # walk toward ball at commanded velocity
        "dribble",  # dribble ball forward (style-conditioned)
        "shoot",  # shoot at goal
        "full",  # combined skills
    )

    # Style command (extra obs dims appended after the 78-dim base obs).
    use_style_command: bool = True
    style_command_dim: int = 5

    # External perturbations (push robot)
    use_perturbations: bool = True

    # Motor / body / ball domain randomisation. Independent of
    # randomize_friction / randomize_robot_pos (which are env-resets).
    # Default ON because robust sim2real transfer is the priority.
    use_domain_randomization: bool = True

    # Vectorised training: when True (and CUDA is available), trainer
    # builds Genesis with n_envs > 1 for parallel rollouts.
    use_vec_env: bool = False
    # Number of parallel envs when use_vec_env. The default is small so
    # CPU machines don't OOM; bump to 1024+ on GPU.
    vec_num_envs: int = 64

    # Domain randomization
    randomize_ball_pos: bool = True
    randomize_robot_pos: bool = True
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.6, 1.2)

    reward: RewardWeights = field(default_factory=RewardWeights)


@dataclass
class Phase2Config:
    """Multi-robot match training config (fine-tuning)."""

    env_name: str = "K1SoccerMatch"
    num_envs: int = 256  # fewer envs due to multi-agent complexity
    players_per_team: int = 4
    max_episode_steps: int = 3000  # ~60 seconds at 50Hz
    dt: float = 0.02
    sim_dt: float = 0.002

    # Observation space per agent
    obs_dim: int = 156  # self + teammates + opponents + ball + field
    act_dim: int = 22

    # Training
    total_timesteps: int = 100_000_000
    learning_rate: float = 1e-4
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    entropy_coef: float = 0.003
    batch_size: int = 16384
    n_epochs: int = 3
    n_steps: int = 128

    # Self-play
    self_play: bool = True
    opponent_update_freq: int = 50  # update opponent policy every N iterations
    opponent_pool_size: int = 10

    # GameController integration
    use_game_controller: bool = True
    half_duration: float = 300.0  # 5 minutes per half (sim time)

    reward: RewardWeights = field(
        default_factory=lambda: RewardWeights(aggressiveness=0.3)
    )


@dataclass
class EvalConfig:
    """Evaluation configuration (MuJoCo)."""

    num_eval_episodes: int = 100
    record_video: bool = True
    video_fps: int = 30
    eval_in_mujoco: bool = True  # evaluate in MuJoCo for sim2sim


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    project: str = "robocup-humanoid-soccer"
    entity: Optional[str] = None
    log_frequency: int = 10
    log_video: bool = True
    video_frequency: int = 50  # log video every N iterations
    log_model: bool = True
    model_save_frequency: int = 100
    tags: List[str] = field(default_factory=lambda: ["k1", "genesis", "robocup"])


@dataclass
class ProjectConfig:
    """Top-level project configuration."""

    robot: K1RobotConfig = field(default_factory=K1RobotConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42
    device: str = "cuda"
    use_genesis: bool = True  # train in Genesis
    use_mujoco_eval: bool = True  # evaluate in MuJoCo
