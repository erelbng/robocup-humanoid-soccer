"""
Running statistics normalizers for PPO.

Why they exist:
  * Observation normalization gives the value & policy networks
    well-scaled inputs from step 0, which is a major reason PPO papers
    report 2-5x faster convergence on locomotion benchmarks.
  * Reward normalization (running std on discounted returns) prevents
    huge reward magnitudes from blowing up the value loss early.

All three are torch-backed so they can sit on the same device as the
policy and be saved/loaded with the checkpoint.

Welford's algorithm (single-pass running mean + variance) is used so
updates are O(1) regardless of batch size.

Standard practice for PPO; see Andrychowicz et al. "What Matters in
On-Policy Reinforcement Learning?" and the Stable-Baselines3 default.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # The module is still importable; normalizers no-op


class RunningMeanStd:
    """Pure-numpy running mean / std (Welford's algorithm)."""

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # avoid div-by-zero on the first update

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == self.mean.ndim:
            x = x[None, ...]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        out = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        return np.clip(out, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(),
                "count": float(self.count)}

    def load_state_dict(self, sd: dict):
        self.mean = np.asarray(sd["mean"], dtype=np.float64)
        self.var = np.asarray(sd["var"], dtype=np.float64)
        self.count = float(sd["count"])


class ReturnNormalizer:
    """Running-std normalizer for PPO returns/advantages.

    Maintains an exponentially-discounted running estimate of the return
    variance and divides advantages (or returns) by sqrt(var). This is the
    "reward-norm-via-return-std" trick used by Stable-Baselines3.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-4):
        self.gamma = gamma
        self.running = 0.0
        self.rms = RunningMeanStd(shape=())

    def update(self, rewards: np.ndarray, dones: np.ndarray):
        """Update with a rollout of single-env rewards & dones."""
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones = np.asarray(dones, dtype=np.float64).reshape(-1)
        for r, d in zip(rewards, dones):
            self.running = self.running * self.gamma * (1.0 - d) + r
            self.rms.update(np.array([self.running]))

    def normalize(self, x):
        return np.asarray(x) / float(self.rms.std + 1e-8)

    def state_dict(self) -> dict:
        return {"gamma": self.gamma, "running": self.running,
                "rms": self.rms.state_dict()}

    def load_state_dict(self, sd: dict):
        self.gamma = float(sd["gamma"])
        self.running = float(sd["running"])
        self.rms.load_state_dict(sd["rms"])


def linear_std_schedule(initial: float = -0.5, final: float = -1.5,
                        progress: float = 0.0) -> float:
    """Linear schedule for action log_std: starts exploratory (high),
    ends exploitative (low). `progress` ∈ [0, 1].

    Returns the log_std value to set on the policy's `actor_log_std`
    Parameter. The default range maps to std=0.61 → 0.22.
    """
    progress = max(0.0, min(1.0, float(progress)))
    return float(initial + (final - initial) * progress)
