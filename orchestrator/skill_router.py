"""Skill router — loads frozen skill policies and routes orchestrator
actions through them.

Given the orchestrator's decision `(skill_idx, cmd_vec_7d)` for each
agent, this module:

  1. Slices the relevant prefix of `cmd_vec_7d` for the chosen skill
     (e.g. walk uses [0:5], shoot uses [0:3]).
  2. Builds the per-skill obs vector: shared base obs + sliced command
     + skill-specific add-ons.
  3. Groups agents by skill choice and runs each frozen policy as a
     single batched forward pass.
  4. Scatters the resulting joint targets back into the original agent
     ordering.

Frozen policies are pure inference — `policy.act(obs, deterministic=
True)` then `.cpu().numpy()`. We use the deterministic path because
exploration happens at the orchestrator level, not inside the skills.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from orchestrator.config import SKILL_CMD_DIMS, SKILL_ORDER
from skills.base import SkillSpec
from training.algorithms.networks import PPOActorCritic
from training.common import load_checkpoint


@dataclass
class FrozenSkill:
    """Wraps a frozen PPO actor-critic + the SkillSpec metadata.

    `addon_builder` produces the skill-specific obs extras (e.g. ball
    state, goal target) — None for skills with no add-ons (standup, walk).
    """
    name: str
    spec: SkillSpec
    policy: torch.nn.Module
    # callable(agent_indices: (M,) int64, env) -> (M, addon_dim) float32
    # passed at runtime so dribble/shoot can pull live ball state from
    # the match env. None for skills with SKILL_OBS_ADDONS == 0.
    addon_builder: Optional[callable] = None


def build_skill_spec(name: str, env_for_dims) -> SkillSpec:
    """Construct a SkillSpec from a one-shot env instance.

    `env_for_dims` should be a single-process construction of the
    skill's env (not initialised — we just need obs_dim/act_dim/
    command_spec). The env is discarded after metadata extraction.
    """
    return SkillSpec(
        name=name,
        obs_dim=int(env_for_dims.obs_dim),
        act_dim=int(env_for_dims.act_dim),
        command_spec=env_for_dims.command_spec,
    )


def load_frozen_skill(name: str, checkpoint_path: str,
                      env_for_dims, device: torch.device,
                      addon_builder=None) -> FrozenSkill:
    """Build a PPOActorCritic of the right shape, load the checkpoint,
    freeze all params, and return a FrozenSkill.
    """
    spec = build_skill_spec(name, env_for_dims)
    policy = PPOActorCritic(spec.obs_dim, spec.act_dim)
    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, policy)
    else:
        print(f"[skill_router] WARNING: no checkpoint for skill {name!r} "
              f"(path={checkpoint_path!r}); using freshly-init weights. "
              "Orchestrator training will be uninformative until skill "
              "checkpoints are populated.")
    policy = policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    spec.checkpoint_path = checkpoint_path or ""
    return FrozenSkill(name=name, spec=spec, policy=policy,
                       addon_builder=addon_builder)


class SkillRouter:
    """Group-by-skill batched inference over frozen skill policies.

    Maintains references to all 4 skills, each as a FrozenSkill. Given
    a batch of agents — each with their own base obs, skill choice,
    and 7-dim command — produces joint targets per agent.
    """

    def __init__(self, frozen_skills: Dict[str, FrozenSkill],
                 device: torch.device):
        # Order skills to match SKILL_ORDER so the discrete index ↔
        # skill name mapping is fixed.
        self.skills: list = [frozen_skills[name] for name in SKILL_ORDER]
        self.device = device
        self.num_skills = len(self.skills)
        # Sanity: every skill must declare its command_spec.dim
        # consistent with SKILL_CMD_DIMS.
        for sk in self.skills:
            expected = SKILL_CMD_DIMS[sk.name]
            actual = int(sk.spec.command_spec.dim)
            assert actual == expected, (
                f"command_spec.dim mismatch for {sk.name!r}: "
                f"spec={actual} vs SKILL_CMD_DIMS={expected}")

    @torch.no_grad()
    def route(self,
              base_obs: np.ndarray,           # (B, 78) — shared common obs per agent
              skill_idx: np.ndarray,          # (B,) int64 — chosen skill per agent
              cmd_vec: np.ndarray,            # (B, 7) — orchestrator cmd, masked per skill
              addon_inputs: dict = None,
              ) -> np.ndarray:
        """Return joint targets (B, act_dim) for every agent.

        `addon_inputs` is forwarded to each skill's `addon_builder`
        when present; it carries whatever live state the addon builder
        needs (e.g. ball pos, target world). Keys per skill name.
        """
        addon_inputs = addon_inputs or {}
        B = base_obs.shape[0]
        # Allocate output; PPOActorCritic.act returns float32.
        act_dim = self.skills[0].spec.act_dim
        out = np.zeros((B, act_dim), dtype=np.float32)

        for k, sk in enumerate(self.skills):
            mask = (skill_idx == k)
            if not np.any(mask):
                continue
            idx_k = np.where(mask)[0]
            base_k = base_obs[idx_k]
            # Slice command to this skill's dim (may be 0).
            cmd_dim_k = int(sk.spec.command_spec.dim)
            cmd_k = cmd_vec[idx_k, :cmd_dim_k] if cmd_dim_k > 0 else None

            # Optional skill-specific addons (dribble, shoot).
            addon_k = None
            if sk.addon_builder is not None and sk.name in addon_inputs:
                addon_k = sk.addon_builder(idx_k, addon_inputs[sk.name])

            parts = [base_k]
            if cmd_k is not None:
                parts.append(cmd_k.astype(np.float32))
            if addon_k is not None:
                parts.append(addon_k.astype(np.float32))
            obs_k = np.concatenate(parts, axis=1)

            assert obs_k.shape[1] == sk.spec.obs_dim, (
                f"{sk.name}: built obs dim {obs_k.shape[1]} vs spec "
                f"{sk.spec.obs_dim}")

            obs_t = torch.as_tensor(obs_k, dtype=torch.float32,
                                    device=self.device)
            action, _, _ = sk.policy.act(obs_t, deterministic=True)
            out[idx_k] = action.cpu().numpy()

        return out
