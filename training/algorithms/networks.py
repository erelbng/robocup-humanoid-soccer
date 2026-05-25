"""Network building blocks shared by PPO and FlashSAC.

Both algorithms need an MLP feature extractor and a (Gaussian) policy
head. PPO uses unbounded Gaussian + tanh on the env side; SAC uses a
squashed Gaussian with the standard tanh-correction term in the log-prob
so the entropy bonus is computed on the actual emitted action.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn


# ─── MLP backbone ──────────────────────────────────────────────────────


def make_mlp(in_dim: int, hidden_dims: Sequence[int], layernorm: bool = True,
             activation: str = "elu") -> nn.Sequential:
    """Standard MLP with optional LayerNorm.

    rsl_rl uses ELU + no LayerNorm for the actor and critic in their
    locomotion configs. We default to LayerNorm-on (more stable on
    high-DR humanoid tasks with PPO).
    """
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
               "gelu": nn.GELU, "silu": nn.SiLU}[activation]
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_cls())
        last = h
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0),
                    final_layer_gain: float = 0.01) -> None:
    """rsl_rl-style init: orthogonal for hidden layers, smaller gain on
    the final layer of actor/critic heads to keep initial outputs near 0.
    Final-layer detection is left to the caller — call with
    `gain=final_layer_gain` for those.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)


# ─── PPO Actor-Critic ──────────────────────────────────────────────────


class PPOActorCritic(nn.Module):
    """rsl_rl-style ActorCritic.

    Separate (non-shared) trunks for actor and critic — empirically more
    stable than the shared backbone our old code used. Action distribution
    is a Gaussian with a learned PER-ACTION log_std (state-independent),
    which is the standard formulation for locomotion PPO.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 actor_hidden: Sequence[int] = (512, 256, 128),
                 critic_hidden: Sequence[int] = (512, 256, 128),
                 init_log_std: float = -0.5,
                 layernorm: bool = True,
                 activation: str = "elu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor_trunk = make_mlp(obs_dim, actor_hidden, layernorm, activation)
        self.actor_head = nn.Linear(actor_hidden[-1], act_dim)
        self.actor_log_std = nn.Parameter(
            torch.full((act_dim,), float(init_log_std), dtype=torch.float32)
        )

        self.critic_trunk = make_mlp(obs_dim, critic_hidden, layernorm, activation)
        self.critic_head = nn.Linear(critic_hidden[-1], 1)

        # Init
        for m in self.actor_trunk:
            orthogonal_init(m, gain=math.sqrt(2.0))
        orthogonal_init(self.actor_head, gain=0.01)
        for m in self.critic_trunk:
            orthogonal_init(m, gain=math.sqrt(2.0))
        orthogonal_init(self.critic_head, gain=1.0)

    def _action_dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_head(self.actor_trunk(obs))
        std = self.actor_log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._action_dist(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # reparameterised sample
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._action_dist(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic_head(self.critic_trunk(obs)).squeeze(-1)
        return value, log_prob, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_trunk(obs)).squeeze(-1)

    # Convenience: emulate the old `get_action` signature used by the
    # single-env trainer so legacy paths keep working.
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        action, log_prob, entropy = self.act(obs, deterministic=deterministic)
        return action, log_prob, entropy


# ─── SAC Actor (squashed Gaussian) ─────────────────────────────────────


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SACActor(nn.Module):
    """Squashed-Gaussian policy as in soft actor-critic.

    Action range is fixed to [-action_scale, +action_scale] via tanh. For
    joint position targets, we use action_scale = π so the policy can
    address the full revolute range. The env then clips internally.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dims: Sequence[int] = (512, 256, 128),
                 action_scale: float = math.pi,
                 layernorm: bool = True,
                 activation: str = "elu"):
        super().__init__()
        self.action_scale = float(action_scale)
        self.act_dim_size = int(act_dim)
        self.trunk = make_mlp(obs_dim, hidden_dims, layernorm, activation)
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)

        for m in self.trunk:
            orthogonal_init(m, gain=math.sqrt(2.0))
        orthogonal_init(self.mean_head, gain=0.01)
        orthogonal_init(self.log_std_head, gain=0.01)

    def forward(self, obs: torch.Tensor, deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action ∈ [-scale, scale], log_prob per-sample).

        The Jacobian correction term for the tanh squash is included in
        log_prob so the temperature loss sees the true entropy of the
        emitted action distribution.
        """
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        # Build distribution once; rsample uses the same object for log_prob.
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()

        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale

        # log π(a|s) for squashed-Gaussian. Standard SAC convention:
        # compute log-density of tanh(u), NOT of scale*tanh(u). The
        # constant log|d(scale*tanh)/d(tanh)| = D*log(scale) would
        # cancel out of every optimization term, and omitting it lets
        # target_entropy = -|A| keep its usual meaning.
        # log p(tanh u) = log N(u|μ,σ) − Σ log(1 − tanh(u)²)
        # Stable form: log(1 − tanh²) = 2(log 2 − u − softplus(−2u))
        log_prob = dist.log_prob(pre_tanh).sum(-1)
        log_prob = log_prob - (
            2.0 * (math.log(2.0) - pre_tanh
                   - torch.nn.functional.softplus(-2.0 * pre_tanh))
        ).sum(-1)
        return action, log_prob


# ─── SAC Twin Q ────────────────────────────────────────────────────────


class TwinQNetwork(nn.Module):
    """Twin Q-networks (clipped double Q-learning).

    Two independent MLPs that take (obs, action) and return a scalar Q.
    We keep them as parallel modules and stack outputs on dim 0 so the
    rest of the training loop can min/clamp over the ensemble dim.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dims: Sequence[int] = (512, 256, 128),
                 layernorm: bool = True,
                 activation: str = "elu"):
        super().__init__()
        in_dim = obs_dim + act_dim
        self.q1_trunk = make_mlp(in_dim, hidden_dims, layernorm, activation)
        self.q1_head = nn.Linear(hidden_dims[-1], 1)
        self.q2_trunk = make_mlp(in_dim, hidden_dims, layernorm, activation)
        self.q2_head = nn.Linear(hidden_dims[-1], 1)

        for trunk in (self.q1_trunk, self.q2_trunk):
            for m in trunk:
                orthogonal_init(m, gain=math.sqrt(2.0))
        orthogonal_init(self.q1_head, gain=1.0)
        orthogonal_init(self.q2_head, gain=1.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1_head(self.q1_trunk(x)).squeeze(-1)
        q2 = self.q2_head(self.q2_trunk(x)).squeeze(-1)
        return q1, q2


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Polyak / EMA target update: θ_tgt ← τ·θ_online + (1-τ)·θ_tgt.

    Done in-place under no_grad for max throughput. Matches FlashSAC's
    EMA-update intent (their version is torch.compile-d; we use plain
    Python for portability across PyTorch versions).
    """
    with torch.no_grad():
        for t_p, p in zip(target.parameters(), online.parameters()):
            t_p.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        # Copy buffers (LayerNorm running stats etc.) verbatim
        for t_b, b in zip(target.buffers(), online.buffers()):
            t_b.data.copy_(b.data)
