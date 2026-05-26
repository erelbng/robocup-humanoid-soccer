"""Orchestrator (Phase 2) config.

Discrete-skill orchestrator over the frozen skill library. Each agent
emits (skill_idx, cmd_vec_7d) at orchestrator frequency (10 Hz by
default); the chosen skill's frozen policy then runs at 50 Hz inside
the slot, taking commands sliced from cmd_vec_7d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# Canonical skill ordering used by the discrete action head. Index in
# this tuple === categorical class index. Don't reorder — checkpoints
# rely on it.
SKILL_ORDER: Tuple[str, ...] = ("standup", "walk", "dribble", "shoot")
NUM_SKILLS: int = len(SKILL_ORDER)

# Padded command width: max command_spec.dim across all skills.
# Each skill consumes only its leading dims; the rest are ignored.
ORCHESTRATOR_CMD_DIM: int = 7
# Per-skill leading command width (must match the skill's CommandSpec.dim).
SKILL_CMD_DIMS = {"standup": 0, "walk": 5, "dribble": 7, "shoot": 3}


@dataclass
class OrchestratorObsLayout:
    """How many dims each obs block occupies. The orchestrator's policy
    sees a flat concatenation in this order.

    self_proprio (78) + ball (6) + teammates(3×6=18) + opps(4×6=24)
    + gc_state(24) + role_onehot(4) + score_time(2) = 156

    role_onehot: [GK, DEF, MID, ATK]
    score_time: [score_diff_normalized, time_remaining_normalized]
    """
    proprio: int = 78           # same shared base obs as the skills
    ball: int = 6               # ball pos body (3) + vel body (3)
    teammates: int = 18         # 3 teammates × (pos_body 3 + vel_body 3)
    opponents: int = 24         # 4 opponents × 6 dims
    gc_state: int = 24          # GameController vector
    role_onehot: int = 4
    score_time: int = 2

    @property
    def total(self) -> int:
        return (self.proprio + self.ball + self.teammates
                + self.opponents + self.gc_state + self.role_onehot
                + self.score_time)


@dataclass
class OrchestratorConfig:
    # ── match parameters ──────────────────────────────────────────
    num_envs: int = 256
    players_per_team: int = 4
    half_duration: float = 300.0      # 5 min sim time / half
    max_episode_steps: int = 3000     # ~60 s at 50 Hz inner

    # ── timing ───────────────────────────────────────────────────
    dt: float = 0.02                  # inner skill control timestep
    sim_dt: float = 0.002
    orchestrator_dt: float = 0.10     # 10 Hz orchestrator decisions
    # Number of inner skill steps between orchestrator decisions.
    inner_steps_per_decision: int = 5  # = orchestrator_dt / dt

    # ── obs / action shapes ──────────────────────────────────────
    obs_layout: OrchestratorObsLayout = field(
        default_factory=OrchestratorObsLayout)

    # ── policy / training ───────────────────────────────────────
    total_timesteps: int = 100_000_000
    learning_rate: float = 1e-4
    gamma: float = 0.998              # long horizon — half is 300 s sim
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 3
    n_steps: int = 128

    obs_dim: int = 0   # filled by env at __init__
    # PPO infers act_dim from policy output, so this stays metadata-only.

    # ── self-play ───────────────────────────────────────────────
    self_play: bool = True
    opponent_pool_size: int = 10
    opponent_update_freq: int = 50    # iterations between snapshot saves
    opponent_latest_prob: float = 0.5  # prob of sampling the latest snapshot
                                       # vs uniform over older entries

    # ── reward ───────────────────────────────────────────────────
    rewards: "MatchRewardWeights" = field(
        default_factory=lambda: MatchRewardWeights())


@dataclass
class MatchRewardWeights:
    """Team-level + agent-level reward composition.

    The orchestrator policy receives per-agent reward, but the dominant
    signals are TEAM events (goal_scored / goal_conceded). Per-agent
    shaping keeps gradients informative early.
    """
    # Team events (large, sparse)
    goal_scored: float = 50.0
    goal_conceded: float = -50.0
    out_of_bounds: float = -1.0

    # Possession / positioning (dense, modest)
    team_ball_possession: float = 0.5
    ball_toward_opp_goal: float = 0.3
    defensive_coverage: float = 0.3

    # Agent-level basics
    upright: float = 0.1
    alive: float = 0.05
    fall: float = -2.0
