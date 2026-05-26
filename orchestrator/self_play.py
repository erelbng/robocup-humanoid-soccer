"""Self-play opponent pool for orchestrator training.

The trainer maintains a deque of past `OrchestratorActorCritic` snapshots
(state_dicts). Each iteration:

  * The currently-training policy controls team 0.
  * An opponent policy is sampled from the pool — with probability
    `latest_prob` we use the most recent snapshot (forces the trainer
    to keep up with itself); otherwise uniformly over older entries
    (prevents catastrophic forgetting).
  * Every `update_freq` iterations we snapshot the current policy
    into the pool, evicting the oldest entry if at capacity.

Snapshots are stored as plain `state_dict()` copies on CPU so the pool
doesn't grow GPU memory. Loading is done into a single reusable
"opponent" policy object — we never instantiate more than one.

This module is intentionally independent of the env so it can be unit-
tested in isolation.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Optional

import numpy as np
import torch


class OpponentPool:

    def __init__(self, capacity: int = 10, latest_prob: float = 0.5,
                 seed: int = 0):
        self.capacity = int(capacity)
        self.latest_prob = float(latest_prob)
        self.pool: deque = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pool)

    def snapshot(self, policy: torch.nn.Module) -> None:
        """Save a CPU copy of `policy.state_dict()`. Cheap — just
        clones tensors off-GPU. The trainer should call this after
        every `update_freq` iterations."""
        sd_cpu = {k: v.detach().to("cpu").clone()
                  for k, v in policy.state_dict().items()}
        self.pool.append(sd_cpu)

    def sample(self) -> Optional[dict]:
        """Return one snapshot state_dict, or None if empty.

        Sampling rule:
          * If pool size == 1, always return that one.
          * Otherwise with prob `latest_prob` return the most recent;
            else uniformly over the remaining older entries.
        """
        if not self.pool:
            return None
        if len(self.pool) == 1:
            return self.pool[-1]
        if self.rng.random() < self.latest_prob:
            return self.pool[-1]
        # Uniform over older entries (everything except the latest)
        idx = int(self.rng.integers(0, len(self.pool) - 1))
        return self.pool[idx]

    def load_into(self, policy: torch.nn.Module,
                  state_dict: Optional[dict] = None) -> bool:
        """Load `state_dict` (or a freshly sampled one) into `policy`.
        Returns True if a load happened, False if the pool was empty."""
        if state_dict is None:
            state_dict = self.sample()
        if state_dict is None:
            return False
        # Move to policy's device as we load.
        device = next(policy.parameters()).device
        loaded = {k: v.to(device) for k, v in state_dict.items()}
        policy.load_state_dict(loaded)
        return True


# ─── helper: split actions across two policies ─────────────────────────


def split_team_action(obs: np.ndarray, current_policy, opponent_policy,
                      n_per_team: int, device) -> np.ndarray:
    """Run team 0 obs through current_policy, team 1 obs through
    opponent_policy, return packed action of shape (N, 2K, 1+CMD).

    Used inside the training rollout loop. `obs` has shape
    (N, 2K, obs_dim).
    """
    N, A, _ = obs.shape
    K = int(n_per_team)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    team0_obs = obs_t[:, :K].reshape(N * K, -1)
    team1_obs = obs_t[:, K:].reshape(N * K, -1)

    with torch.no_grad():
        a0, _, _ = current_policy.act(team0_obs, deterministic=False)
        if opponent_policy is None:
            # No opponent loaded yet — team 1 plays with the current
            # policy too (i.e. mirror-self play). Standard cold-start
            # for the first few iterations.
            a1, _, _ = current_policy.act(team1_obs, deterministic=False)
        else:
            a1, _, _ = opponent_policy.act(team1_obs, deterministic=False)

    action = torch.zeros(N, A, a0.shape[-1], device=device)
    action[:, :K] = a0.reshape(N, K, -1)
    action[:, K:] = a1.reshape(N, K, -1)
    return action.cpu().numpy()
