"""Orchestrator policy — hybrid discrete + continuous PPO actor-critic.

Action space:
  * discrete: skill index ∈ {0, …, NUM_SKILLS−1} (categorical)
  * continuous: command vector ∈ ℝ⁷ (Gaussian with learned log_std)

These two heads are conditionally independent given the obs (standard
factorized-action assumption). log_prob and entropy are the sum across
heads, so PPO surrogate / GAE work without modification — the trainer
just sees a single scalar log_prob per timestep.

Action layout returned by `act()`:
  (N, 1 + ORCHESTRATOR_CMD_DIM) float32, where action[:, 0] is the
  skill index as float (cast to long in the env), and action[:, 1:] is
  the continuous command. We use a single tensor so the existing PPO
  rollout buffer can store it without special-casing.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn

from orchestrator.config import NUM_SKILLS, ORCHESTRATOR_CMD_DIM
from training.algorithms.networks import make_mlp, orthogonal_init


class OrchestratorActorCritic(nn.Module):
    """PPO actor-critic with hybrid (discrete + continuous) actions.

    Shape contract:
      obs    : (B, obs_dim)
      action : (B, 1 + ORCHESTRATOR_CMD_DIM)
               column 0 = skill index (long cast inside env)
               columns 1: = continuous command (float)
      log_prob, entropy: (B,) summed across heads
    """

    def __init__(
        self,
        obs_dim: int,
        actor_hidden: Sequence[int] = (512, 256, 128),
        critic_hidden: Sequence[int] = (512, 256, 128),
        init_log_std: float = -0.5,
        layernorm: bool = True,
        activation: str = "elu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_skills = int(NUM_SKILLS)
        self.cmd_dim = int(ORCHESTRATOR_CMD_DIM)

        # Actor trunk shared between the two heads — keeps net size
        # tractable. Separate trunks per head doubled parameters with
        # no measurable benefit in early experiments on simpler hybrid
        # tasks. Critic gets its own trunk (standard rsl_rl pattern).
        self.actor_trunk = make_mlp(obs_dim, actor_hidden, layernorm, activation)
        self.skill_head = nn.Linear(actor_hidden[-1], self.num_skills)
        self.cmd_mean_head = nn.Linear(actor_hidden[-1], self.cmd_dim)
        self.cmd_log_std = nn.Parameter(
            torch.full((self.cmd_dim,), float(init_log_std),
                       dtype=torch.float32)
        )

        self.critic_trunk = make_mlp(obs_dim, critic_hidden,
                                     layernorm, activation)
        self.critic_head = nn.Linear(critic_hidden[-1], 1)

        # Init
        for m in self.actor_trunk:
            orthogonal_init(m, gain=math.sqrt(2.0))
        orthogonal_init(self.skill_head, gain=0.01)
        orthogonal_init(self.cmd_mean_head, gain=0.01)
        for m in self.critic_trunk:
            orthogonal_init(m, gain=math.sqrt(2.0))
        orthogonal_init(self.critic_head, gain=1.0)

    # ── distributions ─────────────────────────────────────────────

    def _actor_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_trunk(obs)

    def _skill_dist(self, feats: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.skill_head(feats))

    def _cmd_dist(self, feats: torch.Tensor) -> torch.distributions.Normal:
        mean = self.cmd_mean_head(feats)
        std = self.cmd_log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    # ── public API matching PPOActorCritic ───────────────────────

    def act(self, obs: torch.Tensor, deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self._actor_features(obs)
        skill_d = self._skill_dist(feats)
        cmd_d = self._cmd_dist(feats)

        if deterministic:
            skill_idx = skill_d.probs.argmax(dim=-1)
            cmd = cmd_d.mean
        else:
            skill_idx = skill_d.sample()
            cmd = cmd_d.rsample()

        log_prob = (skill_d.log_prob(skill_idx)
                    + cmd_d.log_prob(cmd).sum(-1))
        entropy = skill_d.entropy() + cmd_d.entropy().sum(-1)

        action = torch.cat([skill_idx.float().unsqueeze(-1), cmd], dim=-1)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute (value, log_prob, entropy) for stored actions.

        `actions[:, 0]` is the stored skill index (float cast — we cast
        it back to long here for Categorical.log_prob).
        """
        feats = self._actor_features(obs)
        skill_d = self._skill_dist(feats)
        cmd_d = self._cmd_dist(feats)

        skill_idx = actions[:, 0].long()
        cmd = actions[:, 1:]

        log_prob = (skill_d.log_prob(skill_idx)
                    + cmd_d.log_prob(cmd).sum(-1))
        entropy = skill_d.entropy() + cmd_d.entropy().sum(-1)
        value = self.critic_head(self.critic_trunk(obs)).squeeze(-1)
        return value, log_prob, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_trunk(obs)).squeeze(-1)

    # ── compatibility shim ───────────────────────────────────────

    # PPO's actor_log_std schedule (linear_std_schedule) writes to a
    # parameter called `actor_log_std` on the policy. We expose
    # `cmd_log_std` under that alias so the existing schedule code
    # still works. The continuous head's log_std is the only one that
    # needs scheduling — discrete entropy is handled by the standard
    # PPO entropy_coef.
    @property
    def actor_log_std(self) -> nn.Parameter:
        return self.cmd_log_std
