"""Match-level rewards for the orchestrator (Phase 2).

Combines:
  * sparse team events (goals, OOB) — provided by the env from
    GameController state.
  * dense possession / positioning — computed each step from positions.
  * agent-level basics (upright, alive, fall) — same shape as the walk
    reward but with smaller weights since the dominant signal is the
    team match outcome.

Returned reward is per-AGENT: agents on the same team share team
events; agent-specific terms (upright/alive) come from that agent's
own state. This is the standard multi-agent credit-assignment shape
that PPO handles natively (each agent's GAE uses its own reward).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from skills.common_obs import projected_gravity


def upright_reward(quat: np.ndarray) -> np.ndarray:
    g = projected_gravity(quat)
    return np.clip(-g[:, 2], 0.0, 1.0).astype(np.float32)


def ball_possession_signal(
        ball_pos: np.ndarray,                # (N, 3) per env
        team_robot_pos: np.ndarray,          # (N, n_per_team, 3)
        opp_robot_pos: np.ndarray,           # (N, n_per_team, 3)
        proximity_radius: float = 0.6,
) -> np.ndarray:
    """1 if any teammate is within `proximity_radius` of the ball AND
    is closer than the closest opponent. 0 otherwise. Computed per env.

    Returns (N,) float32 — same value is given to every agent on that
    team (team-level credit).
    """
    # (N, n_per_team) distances
    team_d = np.linalg.norm(team_robot_pos - ball_pos[:, None, :], axis=2)
    opp_d = np.linalg.norm(opp_robot_pos - ball_pos[:, None, :], axis=2)

    team_min = team_d.min(axis=1)
    opp_min = opp_d.min(axis=1)
    return ((team_min < proximity_radius) & (team_min < opp_min)
            ).astype(np.float32)


def ball_toward_goal_signal(
        ball_pos: np.ndarray,                # (N, 3)
        ball_vel: np.ndarray,                # (N, 3)
        goal_x: float,                        # opponent goal x in world frame
) -> np.ndarray:
    """Dot product of ball velocity (xy) with direction-to-goal (xy).

    Positive when the ball is moving toward the opponent goal. We
    normalize direction-to-goal so the magnitude reflects ball speed
    only — otherwise close-to-goal possession would inflate this term.
    """
    to_goal = np.zeros_like(ball_pos[:, :2])
    to_goal[:, 0] = goal_x - ball_pos[:, 0]
    to_goal[:, 1] = -ball_pos[:, 1]
    norm = np.linalg.norm(to_goal, axis=1, keepdims=True).clip(min=1e-6)
    unit = to_goal / norm
    return np.sum(ball_vel[:, :2] * unit, axis=1).astype(np.float32)


def defensive_coverage_signal(
        team_robot_pos: np.ndarray,          # (N, n_per_team, 3)
        ball_pos: np.ndarray,                # (N, 3)
        own_goal_x: float,
) -> np.ndarray:
    """Reward at least one teammate being between the ball and own goal.

    Coarse: counts as defended if the closest teammate is on the goal
    side of the ball (own_goal_x sign-aware). Returns (N,) ∈ [0, 1].
    """
    # Sign telling us which side own_goal is on (+x or −x).
    sign = 1.0 if own_goal_x > 0 else -1.0

    # For each teammate, check if its x is BETWEEN the ball and own
    # goal (i.e. on the defending side).
    # "Between" means sign·(ball_x − robot_x) > 0 AND robot is past the
    # ball toward own goal: sign·(own_goal_x − robot_x) > 0.
    rx = team_robot_pos[:, :, 0]            # (N, n_per_team)
    bx = ball_pos[:, 0:1]                   # (N, 1)
    own = own_goal_x

    on_defending_side = (sign * (bx - rx) > 0) & (sign * (own - rx) > 0)
    return (on_defending_side.any(axis=1)).astype(np.float32)


def compute_match_reward(
    *,
    # state — agent-centric arrays. n_per_team = K, n_envs = N.
    robot_pos: np.ndarray,                  # (N, 2K, 3)
    robot_quat: np.ndarray,                 # (N, 2K, 4)
    ball_pos: np.ndarray,                   # (N, 3)
    ball_vel: np.ndarray,                   # (N, 3)
    # team indexing: 0..K-1 are team 0; K..2K-1 are team 1.
    n_per_team: int,
    # event flags from the GameController (1 step pulse). (N,) bool each.
    goal_for_team0: np.ndarray,
    goal_for_team1: np.ndarray,
    out_of_bounds: np.ndarray,
    # geometry
    field_half_length: float,
    weights,
) -> Tuple[np.ndarray, dict]:
    """Returns per-agent reward (N, 2K) and a batch-averaged component
    dict for logging.

    Team 0 attacks +x goal; team 1 attacks −x goal.
    """
    N = robot_pos.shape[0]
    K = int(n_per_team)
    assert robot_pos.shape == (N, 2 * K, 3)

    team0_pos = robot_pos[:, :K]
    team1_pos = robot_pos[:, K:]
    team0_quat = robot_quat[:, :K]
    team1_quat = robot_quat[:, K:]

    # ── team-level dense signals ─────────────────────────────────
    poss0 = ball_possession_signal(ball_pos, team0_pos, team1_pos)
    poss1 = ball_possession_signal(ball_pos, team1_pos, team0_pos)

    btg_team0 = ball_toward_goal_signal(ball_pos, ball_vel,
                                         goal_x=+field_half_length)
    btg_team1 = ball_toward_goal_signal(ball_pos, ball_vel,
                                         goal_x=-field_half_length)

    cov0 = defensive_coverage_signal(team0_pos, ball_pos,
                                      own_goal_x=-field_half_length)
    cov1 = defensive_coverage_signal(team1_pos, ball_pos,
                                      own_goal_x=+field_half_length)

    # ── per-agent posture ───────────────────────────────────────
    # Reshape to (N*2K, 4) for projected gravity then back.
    up_all = upright_reward(robot_quat.reshape(-1, 4)).reshape(N, 2 * K)
    fallen = (robot_pos[:, :, 2] < 0.30).astype(np.float32)
    alive = 1.0 - fallen

    w = weights

    # ── compose per-team reward, broadcast to agents ────────────
    team0_r = (
        w.goal_scored * goal_for_team0.astype(np.float32)
        + w.goal_conceded * goal_for_team1.astype(np.float32)
        + w.out_of_bounds * out_of_bounds.astype(np.float32)
        + w.team_ball_possession * poss0
        + w.ball_toward_opp_goal * btg_team0
        + w.defensive_coverage * cov0
    ).astype(np.float32)
    team1_r = (
        w.goal_scored * goal_for_team1.astype(np.float32)
        + w.goal_conceded * goal_for_team0.astype(np.float32)
        + w.out_of_bounds * out_of_bounds.astype(np.float32)
        + w.team_ball_possession * poss1
        + w.ball_toward_opp_goal * btg_team1
        + w.defensive_coverage * cov1
    ).astype(np.float32)

    # Broadcast team scalar to each agent on that team.
    team_r_per_agent = np.concatenate(
        [np.broadcast_to(team0_r[:, None], (N, K)),
         np.broadcast_to(team1_r[:, None], (N, K))],
        axis=1).astype(np.float32)

    agent_r = (
        w.upright * up_all
        + w.alive * alive
        + w.fall * fallen
    )

    reward = team_r_per_agent + agent_r

    components = {
        "team0_possession": float(np.mean(poss0)),
        "team1_possession": float(np.mean(poss1)),
        "team0_ball_to_goal": float(np.mean(btg_team0)),
        "team1_ball_to_goal": float(np.mean(btg_team1)),
        "team0_defensive_coverage": float(np.mean(cov0)),
        "team1_defensive_coverage": float(np.mean(cov1)),
        "goals_for_team0": float(np.sum(goal_for_team0)),
        "goals_for_team1": float(np.sum(goal_for_team1)),
        "fall_rate": float(np.mean(fallen)),
        "upright_mean": float(np.mean(up_all)),
        "mean_reward": float(np.mean(reward)),
    }
    return reward, components
