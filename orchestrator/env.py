"""Phase-2 4v4 match env driven by the discrete-skill orchestrator.

This env exposes a hybrid action space (skill_idx, cmd_vec_7d) per
agent. Each call to `step()`:

  1. Splits the action into (skill_idx, cmd_vec) per agent.
  2. Builds the shared 78-dim base obs for every agent (one per robot
     across both teams across all parallel envs).
  3. Routes through `SkillRouter` to obtain (B, 22) joint targets.
  4. Holds the orchestrator decision constant for
     `inner_steps_per_decision` inner physics steps — orchestrator
     decisions are at 10 Hz, skill control loop runs at 50 Hz.
  5. Computes per-agent 156-dim orchestrator obs.
  6. Computes per-agent team-aware reward via `compute_match_reward`.

Scene layout: 2K robot entities + 1 ball + field. `scene.build(
n_envs=N, env_spacing=...)` replicates everything N times. Each
`robots[a].get_pos()` then returns (N, 3) — one row per parallel env.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

from configs.config import K1RobotConfig
from gamecontroller.sim_game_controller import SimulatedGameController
from orchestrator.config import (NUM_SKILLS, ORCHESTRATOR_CMD_DIM,
                                  OrchestratorConfig, SKILL_ORDER)
from orchestrator.rewards import compute_match_reward
from orchestrator.skill_router import SkillRouter
from skills.common_obs import compute_common_obs, body_frame_velocity


# ─── obs construction helpers (testable in isolation) ─────────────────


def _to_np(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _ball_state_in_body_frame(robot_pos, robot_quat, ball_pos, ball_vel):
    """Ball pos & vel relative to a single agent, in body frame.
    All inputs (N, ...) per env; returns ((N, 3), (N, 3))."""
    rel = ball_pos - robot_pos
    return (body_frame_velocity(robot_quat, rel),
            body_frame_velocity(robot_quat, ball_vel))


def _other_robots_in_body_frame(self_pos, self_quat, others_pos, others_vel):
    """Project N_envs × M others into the agent's body frame.

    Args:
        self_pos:    (N, 3)
        self_quat:   (N, 4)
        others_pos:  (N, M, 3)
        others_vel:  (N, M, 3)
    Returns:
        (N, M*6) — for each of M others: 3-dim pos body + 3-dim vel body.
    """
    N = self_pos.shape[0]
    M = others_pos.shape[1]
    out = np.zeros((N, M * 6), dtype=np.float32)
    for j in range(M):
        rel = others_pos[:, j] - self_pos
        pos_b = body_frame_velocity(self_quat, rel)
        vel_b = body_frame_velocity(self_quat, others_vel[:, j])
        out[:, j * 6:j * 6 + 3] = pos_b
        out[:, j * 6 + 3:j * 6 + 6] = vel_b
    return out


def build_agent_orchestrator_obs(
    *,
    self_base_obs: np.ndarray,             # (N, 78)
    self_pos: np.ndarray, self_quat: np.ndarray,
    ball_pos: np.ndarray, ball_vel: np.ndarray,
    teammates_pos: np.ndarray, teammates_vel: np.ndarray,  # (N, 3, 3) — 3 mates
    opponents_pos: np.ndarray, opponents_vel: np.ndarray,  # (N, 4, 3)
    gc_state: np.ndarray,                  # (N, 24)
    role_onehot: np.ndarray,               # (N, 4) — GK/DEF/MID/ATK
    score_diff: np.ndarray,                # (N,)
    time_remaining: np.ndarray,            # (N,) in [0, 1]
) -> np.ndarray:
    """Build the (N, 156) orchestrator obs for a single AGENT INDEX
    across all parallel envs. The match env calls this once per agent
    per orchestrator step.
    """
    ball_pos_b, ball_vel_b = _ball_state_in_body_frame(
        self_pos, self_quat, ball_pos, ball_vel)
    mates = _other_robots_in_body_frame(
        self_pos, self_quat, teammates_pos, teammates_vel)
    opps = _other_robots_in_body_frame(
        self_pos, self_quat, opponents_pos, opponents_vel)
    score_time = np.stack([score_diff.astype(np.float32),
                           time_remaining.astype(np.float32)], axis=1)
    return np.concatenate(
        [self_base_obs.astype(np.float32),
         ball_pos_b, ball_vel_b,
         mates, opps,
         gc_state.astype(np.float32),
         role_onehot.astype(np.float32),
         score_time],
        axis=1).astype(np.float32)


# ─── match env ─────────────────────────────────────────────────────────


class K1MatchEnv:
    """4v4 match env with hybrid orchestrator action.

    Public API (mirrors SkillEnv pattern):
        env = K1MatchEnv(cfg, skill_router, ...)
        obs = env.reset()                  # (N, n_agents, obs_dim)
        obs, rew, done, info = env.step(action)
            action: (N, n_agents, 1 + ORCHESTRATOR_CMD_DIM) — packed by
                    OrchestratorActorCritic.act().
            rew:    (N, n_agents) — per-agent reward.
            done:   (N,) — env-level termination (half/full time, etc.)
    """

    def __init__(
        self,
        cfg: Optional[OrchestratorConfig] = None,
        robot_cfg: Optional[K1RobotConfig] = None,
        skill_router: Optional[SkillRouter] = None,
        field_info: Optional[dict] = None,
        render: bool = False,
        seed: int = 0,
    ):
        self.cfg = cfg or OrchestratorConfig()
        self.robot_cfg = robot_cfg or K1RobotConfig()
        self.render = render
        self.K = int(self.cfg.players_per_team)
        self.n_agents = 2 * self.K
        self.num_envs = int(self.cfg.num_envs)
        self.rng = np.random.default_rng(seed)

        self.skill_router = skill_router
        if skill_router is None:
            print("[match_env] WARNING: no SkillRouter provided; step() will "
                  "fail until one is attached. (This env still builds + "
                  "exposes its obs/action contract for testing.)")

        # Field info
        if field_info is None:
            field_info_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "field",
                "field_info.json")
            if os.path.exists(field_info_path):
                with open(field_info_path) as f:
                    field_info = json.load(f)
            else:
                field_info = {"half_length": 4.5, "half_width": 3.0,
                              "length": 9.0, "width": 6.0,
                              "goal_width": 2.6, "goal_height": 0.8}
        self.field_info = field_info

        # GameController per env. The Python GC isn't currently vec-
        # batched, so each parallel env gets its own GC instance. For
        # very large num_envs this becomes a Python-side bottleneck;
        # batching is a follow-up.
        self.gcs = [
            SimulatedGameController(
                players_per_team=self.K,
                half_duration=self.cfg.half_duration,
                fast_forward=10.0)
            for _ in range(self.num_envs)
        ]

        # Genesis scene state — built lazily.
        self.scene = None
        self.robots: list = []   # length 2K; each entry is a robot entity
        self.ball = None
        self.camera = None
        self.dof_indices: list = []
        self._initialized = False

        # Per-env / per-agent bookkeeping
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self.orch_substep = np.zeros(self.num_envs, dtype=np.int64)
        # Latched orchestrator decisions across an inner-step window:
        self._latched_skill_idx = np.zeros((self.num_envs, self.n_agents),
                                            dtype=np.int64)
        self._latched_cmd = np.zeros(
            (self.num_envs, self.n_agents, ORCHESTRATOR_CMD_DIM),
            dtype=np.float32)

        # Roles (one-hot). Default assignment: per team, agent 0 = GK,
        # agent 1 = DEF, agent 2 = MID, agent 3 = ATK. Configurable.
        # role_onehot shape: (n_agents, 4)
        roles = np.zeros((self.n_agents, 4), dtype=np.float32)
        for k in range(self.K):
            role = min(k, 3)
            roles[k, role] = 1.0
            roles[self.K + k, role] = 1.0
        self._role_onehot = roles

        # Per-agent default joint pose for the common-obs centering.
        self._default_action = np.asarray(self.robot_cfg.default_joint_pos,
                                          dtype=np.float32)
        self._last_action_per_agent = np.tile(
            self._default_action, (self.num_envs, self.n_agents, 1))

        # Computed obs_dim used for buffer sizing.
        self.cfg.obs_dim = int(self.cfg.obs_layout.total)
        self.obs_dim = self.cfg.obs_dim
        self.act_dim = 1 + ORCHESTRATOR_CMD_DIM     # discrete + continuous

    # ── shape properties ───────────────────────────────────────────

    @property
    def total_agents(self) -> int:
        return self.num_envs * self.n_agents

    # ── Genesis scene setup ────────────────────────────────────────

    def _init_genesis(self) -> None:
        """Build the multi-robot Genesis scene.

        Same per-robot add-entity + scene.build(n_envs=N) pattern as
        the phase1 vec env; for each of 2K agent slots we add one URDF
        entity and Genesis replicates everything N times under the
        hood. `robots[a].get_pos()` then returns (N, 3).
        """
        if self._initialized or gs is None:
            return
        try:
            gs.init(backend=gs.gpu, precision="32",
                    logging_level="warning",
                    seed=int(self.rng.integers(1 << 30)),
                    performance_mode=True)
        except Exception:
            pass  # already initialised in this process

        self.scene = gs.Scene(
            show_viewer=self.render,
            sim_options=gs.options.SimOptions(dt=self.cfg.sim_dt,
                                              substeps=2),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                ambient_light=(0.4, 0.4, 0.4),
            ),
        )

        # Field — physics-only (skip the ~80 visual-only line/circle
        # entities) to keep per-env body count manageable. With 2K
        # robots × 256 envs we're already at ~2K bodies; a fancy
        # field would inflate that 10×.
        try:
            from models.field.field_genesis_builder import build_soccer_field
            build_soccer_field(self.scene, physics_only=True)
        except Exception as e:
            print(f"[match_env] field builder failed ({e}); using plane.")
            self.scene.add_entity(
                gs.morphs.Plane(),
                surface=gs.surfaces.Default(color=(0.10, 0.55, 0.10, 1.0),
                                            roughness=0.9),
            )

        # Robots — 2K of them, one entity per agent slot.
        urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "robot", "K1",
            "K1_22dof.urdf",
        )
        positions = self._get_starting_positions()
        self.robots = []
        for agent_id, (x, y, heading) in enumerate(positions):
            robot = self.scene.add_entity(
                gs.morphs.URDF(file=urdf_path,
                               pos=(x, y, 0.65),
                               euler=(0, 0, heading),
                               merge_fixed_links=True),
            )
            self.robots.append(robot)
        self._start_positions = np.asarray(positions, dtype=np.float32)

        # Ball — placed at kickoff (center) by default.
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.07, pos=(0, 0, 0.07),
                             collision=True),
        )

        # Static camera that follows env 0's centre point.
        try:
            self.camera = self.scene.add_camera(
                res=(640, 480), pos=(0, -7, 5),
                lookat=(0, 0, 0.5), fov=60,
            )
        except Exception as e:
            print(f"[match_env] camera setup failed: {e}")
            self.camera = None

        # Spacing must be wide enough to contain the field (9×6) plus
        # a safety border so envs don't collide across boundaries.
        self.scene.build(n_envs=self.num_envs,
                         env_spacing=(12.0, 9.0),
                         center_envs_at_origin=False)
        self._setup_joint_mapping()

        # PD gains on every robot's actuated joints.
        n_dof = len(self.dof_indices)
        for robot in self.robots:
            try:
                robot.set_dofs_kp([float(self.robot_cfg.kp)] * n_dof,
                                  self.dof_indices)
                robot.set_dofs_kv([float(self.robot_cfg.kd)] * n_dof,
                                  self.dof_indices)
            except Exception:
                pass

        self._initialized = True

    def _setup_joint_mapping(self) -> None:
        """Detect DOF indices by joint name. All robots share the URDF
        so we only need to look at robots[0]."""
        self.dof_indices = []
        joint_by_name = {j.name: j for j in self.robots[0].joints}
        for name in self.robot_cfg.joint_names:
            j = joint_by_name.get(name)
            if j is None:
                continue
            try:
                idxs = j.dofs_idx_local
            except Exception:
                idxs = None
            if idxs and len(idxs) == 1:
                self.dof_indices.append(int(idxs[0]))
        if not self.dof_indices:
            self.dof_indices = list(
                range(6, 6 + self.robot_cfg.num_dofs))

    def _get_starting_positions(self) -> List[Tuple[float, float, float]]:
        """Kickoff positions for all 2K agents: GK + DEF + MID + ATK
        per team. Returns a flat list of (x, y, heading_rad). Indices
        0..K-1 are team 0 (defending −x goal); K..2K-1 are team 1.
        """
        hl = float(self.field_info["half_length"])

        # Per-team formation (home = team 0, attacking +x).
        # Heading 0 = facing +x; team 1 mirrors and flips by π.
        K = self.K
        formations = {
            1: [(-hl + 0.5, 0.0)],
            2: [(-hl + 0.5, 0.0), (-1.5, 0.0)],
            3: [(-hl + 0.5, 0.0), (-2.0, +1.0), (-0.8, 0.0)],
            4: [(-hl + 0.5, 0.0), (-2.0, +1.0), (-2.0, -1.0), (-0.8, 0.0)],
        }
        layout = formations.get(min(K, 4), formations[4])
        # Pad if K>4
        while len(layout) < K:
            i = len(layout)
            y = (i - 3) * 0.5 * (1 if i % 2 else -1)
            layout.append((-hl / 3, y))

        positions: list = []
        for x, y in layout[:K]:
            positions.append((x, y, 0.0))
        for x, y in layout[:K]:
            positions.append((-x, -y, math.pi))
        return positions

    # ── reset / step (contract; Genesis-driven internals come w/ step 8) ──

    def reset(self, envs_idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset specified envs (or all). Returns (num_envs, n_agents, obs_dim)."""
        if not self._initialized:
            self._init_genesis()

        if envs_idx is None:
            envs_idx = np.arange(self.num_envs)
        envs_idx = np.asarray(envs_idx, dtype=np.int64)
        n = envs_idx.shape[0]

        self.step_count[envs_idx] = 0
        self.orch_substep[envs_idx] = 0
        self._last_action_per_agent[envs_idx] = self._default_action

        for i in envs_idx:
            self.gcs[int(i)].reset(kick_off_team=int(self.rng.integers(0, 2)))

        # Place every robot at its formation slot.
        if self.robots and len(self.robots) == self.n_agents:
            for a, robot in enumerate(self.robots):
                pos = np.zeros((n, 3), dtype=np.float32)
                pos[:, 0] = self._start_positions[a, 0]
                pos[:, 1] = self._start_positions[a, 1]
                pos[:, 2] = 0.65
                # Heading → quaternion (yaw only).
                yaw = float(self._start_positions[a, 2])
                half = yaw / 2.0
                quat = np.tile(np.array(
                    [math.cos(half), 0.0, 0.0, math.sin(half)],
                    dtype=np.float32), (n, 1))
                targets = np.tile(self._default_action, (n, 1))
                try:
                    robot.set_pos(pos, envs_idx=envs_idx)
                    robot.set_quat(quat, envs_idx=envs_idx)
                    robot.set_dofs_position(targets, self.dof_indices,
                                            envs_idx=envs_idx,
                                            zero_velocity=True)
                except Exception as e:
                    print(f"[match_env] reset robot {a} failed: {e}")

        # Centre the ball at kickoff.
        if self.ball is not None:
            bpos = np.zeros((n, 3), dtype=np.float32)
            bpos[:, 2] = 0.07
            try:
                self.ball.set_pos(bpos, envs_idx=envs_idx)
                try:
                    self.ball.zero_all_dofs_velocity(envs_idx=envs_idx)
                except Exception:
                    pass
            except Exception as e:
                print(f"[match_env] ball reset failed: {e}")

        return self._get_obs_all_agents()

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step the env.

        Args:
            action: (num_envs, n_agents, 1 + ORCHESTRATOR_CMD_DIM) —
                    packed by OrchestratorActorCritic. Column 0 of the
                    last axis is the skill index (float cast); the
                    remaining 7 are the continuous command.
        Returns:
            obs: (num_envs, n_agents, obs_dim)
            rew: (num_envs, n_agents)
            done: (num_envs,)
            info: dict with reward_components, gc state per env, etc.
        """
        action = np.asarray(action)
        assert action.shape == (self.num_envs, self.n_agents,
                                1 + ORCHESTRATOR_CMD_DIM), (
            f"orchestrator action shape {action.shape} != "
            f"({self.num_envs}, {self.n_agents}, {1 + ORCHESTRATOR_CMD_DIM})"
        )
        # Decision update: every `inner_steps_per_decision` calls,
        # latch fresh decisions; otherwise reuse the latched one. This
        # is what enforces the orchestrator's lower decision frequency.
        is_decision_step = (self.orch_substep == 0)
        # In vectorized envs we want to allow some envs to be at
        # different substeps (e.g. after an env-level reset), so the
        # latch is per-env. The trainer can choose to call act() only
        # on the decision steps to save compute.
        if np.any(is_decision_step):
            d_idx = np.where(is_decision_step)[0]
            self._latched_skill_idx[d_idx] = (
                action[d_idx, :, 0].astype(np.int64))
            self._latched_cmd[d_idx] = action[d_idx, :, 1:]

        joint_targets = self._route_actions_to_joint_targets()

        self._apply_and_step_physics(joint_targets)

        self.step_count += 1
        self.orch_substep = (self.orch_substep + 1) % \
            self.cfg.inner_steps_per_decision

        obs = self._get_obs_all_agents()
        reward, components = self._compute_reward()
        done = self._check_done()

        self._last_action_per_agent = joint_targets.reshape(
            self.num_envs, self.n_agents, -1)

        if done.any():
            self._reset_envs(np.where(done)[0])

        info = {
            "reward_components": components,
            "skill_idx": self._latched_skill_idx.copy(),
        }
        return obs, reward, done, info

    # ── inner steps (Genesis-dependent; step-8 implements) ─────────

    def _route_actions_to_joint_targets(self) -> np.ndarray:
        """Build per-agent base obs, call SkillRouter, return
        (num_envs*n_agents, 22) joint targets in flat order
        agent-major (env 0 agents 0..2K-1, env 1, …)."""
        assert self.skill_router is not None, \
            "SkillRouter required to route orchestrator actions."

        base_obs_all = self._get_base_obs_all_agents()        # (E, A, 78)
        E, A, P = base_obs_all.shape
        base_flat = base_obs_all.reshape(E * A, P)
        skill_idx_flat = self._latched_skill_idx.reshape(E * A)
        cmd_flat = self._latched_cmd.reshape(E * A, ORCHESTRATOR_CMD_DIM)

        addon_inputs = self._build_skill_addon_inputs()
        return self.skill_router.route(
            base_obs=base_flat, skill_idx=skill_idx_flat,
            cmd_vec=cmd_flat, addon_inputs=addon_inputs)

    def _build_skill_addon_inputs(self) -> dict:
        """Provide live state to each skill's `addon_builder`.

        Dribble's addon expects (M, 6) = ball pos body + vel body for
        the agents that chose dribble. Shoot's addon expects (M, 9) =
        ball pos body + vel body + target body. We compute these per-
        agent flat arrays here and let the router slice them per skill.

        The flat layout matches the row-major (env, agent) ordering used
        elsewhere (env 0 agents 0..A-1, env 1, …).
        """
        if self.ball is None or not self.robots:
            return {}

        # Read state once per step.
        ball_pos = _to_np(self.ball.get_pos())              # (N, 3)
        ball_vel = _to_np(self.ball.get_vel())              # (N, 3)

        # Per-agent self pos/quat: stack across agents.
        self_pos = np.stack(
            [_to_np(r.get_pos()) for r in self.robots], axis=1)   # (N, A, 3)
        self_quat = np.stack(
            [_to_np(r.get_quat()) for r in self.robots], axis=1)  # (N, A, 4)

        # Flatten so addon builders see one row per agent across all envs.
        N, A = self_pos.shape[0], self_pos.shape[1]
        EA = N * A

        bp = np.broadcast_to(ball_pos[:, None, :], (N, A, 3)).reshape(EA, 3)
        bv = np.broadcast_to(ball_vel[:, None, :], (N, A, 3)).reshape(EA, 3)
        sp = self_pos.reshape(EA, 3)
        sq = self_quat.reshape(EA, 4)

        ball_pos_body = body_frame_velocity(sq, bp - sp)
        ball_vel_body = body_frame_velocity(sq, bv)

        # Per-agent "target world": for dribble there's no target; for
        # shoot we point each agent at the opponent goal centre.
        # Team 0 agents (rows 0..K-1 inside each env) → +x goal.
        hl = float(self.field_info["half_length"])
        target_world = np.zeros((N, A, 3), dtype=np.float32)
        target_world[:, :self.K, 0] = +hl
        target_world[:, self.K:, 0] = -hl
        target_world[:, :, 2] = 0.4
        target_rel = target_world.reshape(EA, 3) - sp
        target_body = body_frame_velocity(sq, target_rel)

        # Addon for dribble: ball pos body (3) + ball vel body (3) = 6.
        dribble_addon_all = np.concatenate(
            [ball_pos_body, ball_vel_body], axis=1).astype(np.float32)
        # Addon for shoot: ball pos body (3) + ball vel body (3) +
        # target body (3) = 9.
        shoot_addon_all = np.concatenate(
            [ball_pos_body, ball_vel_body, target_body],
            axis=1).astype(np.float32)

        # Builders that slice the pre-computed flat arrays by the
        # router-provided agent indices.
        def _dribble_builder(idx, _inputs):
            return dribble_addon_all[idx]

        def _shoot_builder(idx, _inputs):
            return shoot_addon_all[idx]

        # Attach the builders to the router's frozen skill objects in
        # place. (Set once; subsequent calls overwrite — cheap.)
        if self.skill_router is not None:
            for sk in self.skill_router.skills:
                if sk.name == "dribble":
                    sk.addon_builder = _dribble_builder
                elif sk.name == "shoot":
                    sk.addon_builder = _shoot_builder
        return {"dribble": None, "shoot": None}

    def _apply_and_step_physics(self, joint_targets: np.ndarray) -> None:
        """Apply per-agent joint targets and step Genesis."""
        # joint_targets: (num_envs * n_agents, act_dim) flat,
        # agent-major within each env (env 0: a0..aA-1; env 1: a0..).
        N, A = self.num_envs, self.n_agents
        targets_2d = joint_targets.reshape(N, A, -1)
        clipped = np.clip(targets_2d, -math.pi, math.pi)
        for a, robot in enumerate(self.robots):
            try:
                robot.control_dofs_position(clipped[:, a, :],
                                            self.dof_indices)
            except Exception:
                pass
        # Inner action repeat — convert orchestrator dt back to
        # physics steps. We rely on the SkillEnv-style action_repeat
        # (dt / sim_dt) since every PD command targets the next
        # control tick.
        action_repeat = max(1, int(round(self.cfg.dt / self.cfg.sim_dt)))
        for _ in range(action_repeat):
            self.scene.step()

    # ── observations ───────────────────────────────────────────────

    def _read_per_agent_state(self):
        """Pull every robot's state once. Returns a dict of (N, A, *)
        arrays so the obs / reward helpers don't each re-query Genesis."""
        N, A = self.num_envs, self.n_agents
        pos = np.stack([_to_np(r.get_pos()) for r in self.robots], axis=1)
        quat = np.stack([_to_np(r.get_quat()) for r in self.robots], axis=1)
        lvel = np.stack([_to_np(r.get_vel()) for r in self.robots], axis=1)
        avel = np.stack([_to_np(r.get_ang()) for r in self.robots], axis=1)
        jpos = np.stack(
            [_to_np(r.get_dofs_position(self.dof_indices))
             for r in self.robots], axis=1)
        jvel = np.stack(
            [_to_np(r.get_dofs_velocity(self.dof_indices))
             for r in self.robots], axis=1)
        return dict(pos=pos, quat=quat, lvel=lvel, avel=avel,
                    jpos=jpos, jvel=jvel)

    def _get_base_obs_all_agents(self) -> np.ndarray:
        """Build the shared 78-dim base obs for every agent. (N, A, 78)."""
        if not self._initialized or not self.robots:
            return np.zeros((self.num_envs, self.n_agents, 78),
                            dtype=np.float32)
        st = self._read_per_agent_state()
        N, A = self.num_envs, self.n_agents
        # Compute per-agent in a loop — common_obs is small enough that
        # the loop cost is dominated by the underlying Genesis getters.
        out = np.zeros((N, A, 78), dtype=np.float32)
        for a in range(A):
            out[:, a, :] = compute_common_obs(
                root_pos=st["pos"][:, a],
                root_quat=st["quat"][:, a],
                root_lin_vel=st["lvel"][:, a],
                root_ang_vel=st["avel"][:, a],
                joint_pos=st["jpos"][:, a],
                joint_vel=st["jvel"][:, a],
                last_action=self._last_action_per_agent[:, a],
                step_count=self.step_count,
                default_joint_pos=self._default_action,
                control_dt=self.cfg.dt,
            )
        return out

    def _get_obs_all_agents(self) -> np.ndarray:
        """Full per-agent orchestrator obs (N, A, 156)."""
        if not self._initialized or self.ball is None:
            return np.zeros(
                (self.num_envs, self.n_agents, self.obs_dim),
                dtype=np.float32)

        base_all = self._get_base_obs_all_agents()      # (N, A, 78)
        ball_pos = _to_np(self.ball.get_pos())          # (N, 3)
        ball_vel = _to_np(self.ball.get_vel())          # (N, 3)

        # Read per-agent self pos/quat fresh — could reuse from
        # _get_base_obs_all_agents, but the second call is cached
        # by Genesis internally.
        self_pos = np.stack(
            [_to_np(r.get_pos()) for r in self.robots], axis=1)   # (N, A, 3)
        self_quat = np.stack(
            [_to_np(r.get_quat()) for r in self.robots], axis=1)  # (N, A, 4)
        self_lvel = np.stack(
            [_to_np(r.get_vel()) for r in self.robots], axis=1)   # (N, A, 3)

        # GameController state — Python loop, one row per env.
        # The Python GC exposes `get_state_vector()` (variable length).
        # We pad/truncate to fit the 24-dim slot.
        gc_state = np.zeros((self.num_envs, 24), dtype=np.float32)
        score_diff = np.zeros(self.num_envs, dtype=np.float32)
        time_remaining = np.zeros(self.num_envs, dtype=np.float32)
        for i, gc in enumerate(self.gcs):
            try:
                vec = gc.get_state_vector()
                vec = np.asarray(vec, dtype=np.float32)
                m = min(24, vec.shape[0])
                gc_state[i, :m] = vec[:m]
            except Exception:
                pass
            try:
                team0_score, team1_score = gc.score
                score_diff[i] = float(team0_score - team1_score)
                # secs_remaining → fraction of a half remaining.
                time_remaining[i] = max(0.0, min(1.0, float(
                    gc.data.secs_remaining
                    / max(1e-6, self.cfg.half_duration))))
            except Exception:
                pass

        obs = np.zeros((self.num_envs, self.n_agents, self.obs_dim),
                       dtype=np.float32)

        # Build agent obs per slot. The teammate/opponent indexing
        # excludes the self agent and groups team-mates together
        # (always K-1 mates + K opponents per agent).
        for a in range(self.n_agents):
            team = 0 if a < self.K else 1
            team_mates_idx = [
                j for j in (range(self.K) if team == 0
                            else range(self.K, 2 * self.K))
                if j != a
            ]
            opp_idx = list(range(self.K, 2 * self.K)) if team == 0 \
                else list(range(0, self.K))

            # Pad to 3 mates / 4 opps for fixed obs width.
            def _pad(idx_list, n):
                arr = list(idx_list)
                while len(arr) < n:
                    arr.append(a)  # self-fill — same position, zero rel
                return arr[:n]
            mates_list = _pad(team_mates_idx, 3)
            opps_list = _pad(opp_idx, 4)

            mates_pos = np.stack(
                [self_pos[:, j] for j in mates_list], axis=1)    # (N, 3, 3)
            mates_vel = np.stack(
                [self_lvel[:, j] for j in mates_list], axis=1)
            opps_pos = np.stack(
                [self_pos[:, j] for j in opps_list], axis=1)     # (N, 4, 3)
            opps_vel = np.stack(
                [self_lvel[:, j] for j in opps_list], axis=1)

            role = np.broadcast_to(self._role_onehot[a],
                                   (self.num_envs, 4)).astype(np.float32)
            # Score from the agent's team's perspective.
            sd = score_diff if team == 0 else -score_diff

            obs[:, a, :] = build_agent_orchestrator_obs(
                self_base_obs=base_all[:, a],
                self_pos=self_pos[:, a],
                self_quat=self_quat[:, a],
                ball_pos=ball_pos, ball_vel=ball_vel,
                teammates_pos=mates_pos, teammates_vel=mates_vel,
                opponents_pos=opps_pos, opponents_vel=opps_vel,
                gc_state=gc_state, role_onehot=role,
                score_diff=sd, time_remaining=time_remaining,
            )
        return obs

    # ── reward / done ──────────────────────────────────────────────

    def _compute_reward(self):
        """Per-agent reward from current Genesis + GC state."""
        if not self._initialized:
            return (np.zeros((self.num_envs, self.n_agents),
                             dtype=np.float32), {})
        st = self._read_per_agent_state()
        ball_pos = _to_np(self.ball.get_pos())
        ball_vel = _to_np(self.ball.get_vel())

        # Step the GameController per env so its events are up to date.
        # The Python GC is not vectorised, so this loops in Python — a
        # known bottleneck for very large num_envs (mitigation: cap at
        # ~256 envs for Phase 2 training, as configured).
        # Attribution: the GC's `goal_just_scored` is a single-step
        # pulse. We look at score-delta from the previous step to tell
        # which team scored.
        goal0 = np.zeros(self.num_envs, dtype=bool)
        goal1 = np.zeros(self.num_envs, dtype=bool)
        oob = np.zeros(self.num_envs, dtype=bool)
        if not hasattr(self, "_prev_scores"):
            self._prev_scores = np.zeros((self.num_envs, 2), dtype=np.int64)
        for i, gc in enumerate(self.gcs):
            try:
                player_positions = [
                    (float(st["pos"][i, a, 0]),
                     float(st["pos"][i, a, 1]),
                     0.0)
                    for a in range(self.n_agents)
                ]
                gc.step(self.cfg.dt,
                        (float(ball_pos[i, 0]),
                         float(ball_pos[i, 1]),
                         float(ball_pos[i, 2])),
                        player_positions, self.field_info)
                t0, t1 = gc.score
                if t0 > int(self._prev_scores[i, 0]):
                    goal0[i] = True
                if t1 > int(self._prev_scores[i, 1]):
                    goal1[i] = True
                self._prev_scores[i, 0] = t0
                self._prev_scores[i, 1] = t1
                oob[i] = bool(getattr(gc, "_out_of_bounds", False))
            except Exception:
                pass

        reward, components = compute_match_reward(
            robot_pos=st["pos"], robot_quat=st["quat"],
            ball_pos=ball_pos, ball_vel=ball_vel,
            n_per_team=self.K,
            goal_for_team0=goal0, goal_for_team1=goal1,
            out_of_bounds=oob,
            field_half_length=float(self.field_info["half_length"]),
            weights=self.cfg.rewards,
        )
        return reward, components

    def _check_done(self) -> np.ndarray:
        timeout = self.step_count >= self.cfg.max_episode_steps
        return timeout

    def _reset_envs(self, envs_idx: np.ndarray) -> None:
        # Re-run the full reset so robot poses and ball get re-placed.
        self.reset(envs_idx=envs_idx)
        if hasattr(self, "_prev_scores"):
            self._prev_scores[envs_idx] = 0

    # ── rendering / cleanup ────────────────────────────────────────

    def render_frame(self):
        if self.camera is None:
            return None
        try:
            out = self.camera.render()
            rgb = out[0] if isinstance(out, tuple) else out
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            return rgb
        except Exception as e:
            print(f"[match_env] render_frame error: {e}")
            return None

    def close(self) -> None:
        self.camera = None
        self.robots = []
        self.ball = None
        self.scene = None
        self._initialized = False
        import gc as _gc
        _gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
