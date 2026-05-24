"""GPU-resident replay buffer for off-policy training with Genesis vec env.

Genesis hands us N parallel envs every step. The buffer stores transitions
in a circular tensor of shape (capacity, obs_dim/act_dim/...) on whichever
device the trainer is on (GPU by default). Each `add()` ingests a full
batch of N transitions at once.

Storage layout is intentionally flat (not per-env) so the trainer can
draw uniform random batches across all envs. The `done` flag in the
batch is the env-reported terminal/truncation signal — the trainer
handles bootstrap masking with it.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class GPUReplayBuffer:
    """Fixed-size circular replay buffer kept entirely on `device`.

    All slots are pre-allocated up front; this avoids the latency spike
    of growing a Python list. With capacity=1e6 and obs_dim=83 the
    storage is ~500 MB — fine for GPUs but worth being aware of.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device
        self.dtype = dtype

        self.obs = torch.zeros((self.capacity, obs_dim), dtype=dtype, device=device)
        self.act = torch.zeros((self.capacity, act_dim), dtype=dtype, device=device)
        self.rew = torch.zeros((self.capacity,), dtype=dtype, device=device)
        self.next_obs = torch.zeros((self.capacity, obs_dim), dtype=dtype, device=device)
        self.done = torch.zeros((self.capacity,), dtype=dtype, device=device)

        self.ptr = 0
        self.size = 0

    @torch.no_grad()
    def add_batch(self, obs, act, rew, next_obs, done) -> None:
        """Ingest N transitions in one call.

        All args are converted to tensors on the buffer's device; this
        does NOT copy if they're already there. We wrap around the
        circular ptr cleanly even when N > remaining slots.
        """
        obs_t = _as_tensor(obs, self.dtype, self.device)
        act_t = _as_tensor(act, self.dtype, self.device)
        rew_t = _as_tensor(rew, self.dtype, self.device).reshape(-1)
        next_obs_t = _as_tensor(next_obs, self.dtype, self.device)
        done_t = _as_tensor(done, self.dtype, self.device).reshape(-1)

        n = obs_t.shape[0]
        if n == 0:
            return

        # Two-segment copy to handle wrap-around. Note: torch advanced
        # indexing with an arange wraps fine, but explicit slicing is
        # ~2x faster on CUDA.
        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.obs[sl] = obs_t
            self.act[sl] = act_t
            self.rew[sl] = rew_t
            self.next_obs[sl] = next_obs_t
            self.done[sl] = done_t
        else:
            first = self.capacity - self.ptr
            second = n - first
            self.obs[self.ptr:] = obs_t[:first]
            self.act[self.ptr:] = act_t[:first]
            self.rew[self.ptr:] = rew_t[:first]
            self.next_obs[self.ptr:] = next_obs_t[:first]
            self.done[self.ptr:] = done_t[:first]
            self.obs[:second] = obs_t[first:]
            self.act[:second] = act_t[first:]
            self.rew[:second] = rew_t[first:]
            self.next_obs[:second] = next_obs_t[first:]
            self.done[:second] = done_t[first:]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Uniform random batch. Returns tensors already on `device`."""
        if self.size == 0:
            raise RuntimeError("replay buffer is empty")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }

    def __len__(self) -> int:
        return self.size


def _as_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
