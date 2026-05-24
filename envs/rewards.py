"""
Multi-objective reward function for humanoid soccer training.

Implements a hierarchical reward system with:
  - Phase 1 rewards: locomotion, ball tracking, dribbling, shooting
  - Phase 2 rewards: tactical positioning, passing, match play
  - Configurable aggressiveness scaling
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Tracks individual reward components for logging."""
    total: float = 0.0
    alive: float = 0.0
    upright: float = 0.0
    forward_vel: float = 0.0
    ball_tracking: float = 0.0
    ball_to_goal: float = 0.0
    kick: float = 0.0
    dribble: float = 0.0
    energy: float = 0.0
    smoothness: float = 0.0
    fall: float = 0.0
    joint_limit: float = 0.0
    foot_contact: float = 0.0
    # Phase 2
    possession: float = 0.0
    goal_scored: float = 0.0
    goal_conceded: float = 0.0
    positioning: float = 0.0
    passing: float = 0.0
    defensive: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


class SoccerRewardFunction:
    """
    Computes multi-objective rewards for humanoid soccer.

    The reward function is designed as a weighted sum of components,
    each targeting a specific behavior. Weights are configurable to
    allow tuning the play style (aggressive vs defensive, etc.).
    """

    def __init__(self, weights, field_info: dict):
        """
        Args:
            weights: RewardWeights from config.
            field_info: Dictionary with field dimensions.
        """
        self.w = weights
        self.fi = field_info
        self._prev_ball_dist_to_goal = None
        self._prev_robot_to_ball_dist = None
        self._prev_actions = None
        self._kick_cooldown = 0

    def reset(self):
        """Reset episode-level state."""
        self._prev_ball_dist_to_goal = None
        self._prev_robot_to_ball_dist = None
        self._prev_actions = None
        self._kick_cooldown = 0

    # ── Phase 1: Single Robot Rewards ─────────────────────────────

    def compute_phase1(
        self,
        robot_pos: np.ndarray,        # (3,) x,y,z
        robot_quat: np.ndarray,       # (4,) w,x,y,z
        robot_vel: np.ndarray,        # (3,) linear velocity
        robot_angvel: np.ndarray,     # (3,) angular velocity
        joint_pos: np.ndarray,        # (n_joints,) current joint positions
        joint_vel: np.ndarray,        # (n_joints,) current joint velocities
        joint_limits_lower: np.ndarray,
        joint_limits_upper: np.ndarray,
        actions: np.ndarray,          # (n_joints,) action applied
        ball_pos: np.ndarray,         # (3,)
        ball_vel: np.ndarray,         # (3,)
        target_goal_pos: np.ndarray,  # (3,) center of opponent goal
        foot_contacts: np.ndarray,    # (2,) left/right foot contact force
        is_fallen: bool,
        dt: float,
    ) -> Tuple[float, RewardComponents]:
        """Compute Phase 1 reward for single-robot skills training."""
        rc = RewardComponents()

        # ── Alive bonus ──
        rc.alive = self.w.alive_bonus if not is_fallen else 0.0

        # ── Upright bonus ──
        # Reward for keeping torso vertical (z-axis of body aligned with world z)
        # Using quaternion to get the "up" direction of the robot
        up_vec = self._quat_rotate_vec(robot_quat, np.array([0, 0, 1]))
        uprightness = up_vec[2]  # dot product with world z
        rc.upright = self.w.upright_bonus * max(0, uprightness)

        # ── Forward velocity toward ball ──
        robot_to_ball = ball_pos[:2] - robot_pos[:2]
        dist_to_ball = np.linalg.norm(robot_to_ball)
        if dist_to_ball > 0.01:
            dir_to_ball = robot_to_ball / dist_to_ball
            vel_toward_ball = np.dot(robot_vel[:2], dir_to_ball)
            rc.forward_vel = self.w.forward_velocity * max(0, vel_toward_ball)
        else:
            rc.forward_vel = self.w.forward_velocity * 0.5  # close to ball

        # ── Ball tracking (getting closer to ball) ──
        if self._prev_robot_to_ball_dist is not None:
            dist_delta = self._prev_robot_to_ball_dist - dist_to_ball
            rc.ball_tracking = self.w.tracking_ball * dist_delta * 10.0
        self._prev_robot_to_ball_dist = dist_to_ball

        # ── Ball to goal (ball moving toward target goal) ──
        ball_to_goal = target_goal_pos[:2] - ball_pos[:2]
        dist_ball_to_goal = np.linalg.norm(ball_to_goal)
        if self._prev_ball_dist_to_goal is not None:
            goal_delta = self._prev_ball_dist_to_goal - dist_ball_to_goal
            rc.ball_to_goal = self.w.ball_to_goal * max(0, goal_delta) * 5.0
        self._prev_ball_dist_to_goal = dist_ball_to_goal

        # ── Kick reward (ball velocity spike when robot is close) ──
        ball_speed = np.linalg.norm(ball_vel[:2])
        if self._kick_cooldown <= 0:
            if dist_to_ball < 0.3 and ball_speed > 1.0:
                # Check if ball is heading toward goal
                if dist_ball_to_goal > 0.01:
                    ball_dir = ball_vel[:2] / max(ball_speed, 1e-6)
                    goal_dir = ball_to_goal / dist_ball_to_goal
                    alignment = np.dot(ball_dir, goal_dir)
                    if alignment > 0.3:
                        rc.kick = self.w.kick_reward * alignment * ball_speed
                        self._kick_cooldown = 20  # cooldown steps
        else:
            self._kick_cooldown -= 1

        # ── Dribble control (ball near robot and moving with it) ──
        if dist_to_ball < 0.5 and ball_speed > 0.1:
            robot_speed = np.linalg.norm(robot_vel[:2])
            if robot_speed > 0.1:
                speed_match = 1.0 - min(1.0,
                    abs(ball_speed - robot_speed) / max(robot_speed, 0.1))
                rc.dribble = self.w.dribble_control * speed_match

        # ── Foot contact (at least one foot on ground) ──
        has_contact = np.any(foot_contacts > 0.1)
        rc.foot_contact = self.w.foot_contact_reward if has_contact else 0.0

        # ── Energy penalty (penalize large joint velocities/torques) ──
        energy = np.sum(np.abs(actions * joint_vel))
        rc.energy = self.w.energy_penalty * energy

        # ── Action smoothness ──
        if self._prev_actions is not None:
            action_diff = np.sum((actions - self._prev_actions) ** 2)
            rc.smoothness = self.w.action_smoothness * action_diff
        self._prev_actions = actions.copy()

        # ── Joint limit penalty ──
        lower_violation = np.sum(np.maximum(0,
                                 joint_limits_lower - joint_pos) ** 2)
        upper_violation = np.sum(np.maximum(0,
                                 joint_pos - joint_limits_upper) ** 2)
        rc.joint_limit = self.w.joint_limit_penalty * (
            lower_violation + upper_violation)

        # ── Fall penalty ──
        rc.fall = self.w.fall_penalty if is_fallen else 0.0

        # ── Total ──
        rc.total = (rc.alive + rc.upright + rc.forward_vel + rc.ball_tracking
                    + rc.ball_to_goal + rc.kick + rc.dribble + rc.foot_contact
                    + rc.energy + rc.smoothness + rc.joint_limit + rc.fall)

        return rc.total, rc

    # ── Phase 2: Match Rewards ────────────────────────────────────

    def compute_phase2(
        self,
        agent_idx: int,
        team_idx: int,
        robot_states: Dict,   # {team_idx: [{pos, vel, quat, ...}, ...]}
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        game_state: dict,
        phase1_reward: float,
        phase1_components: RewardComponents,
    ) -> Tuple[float, RewardComponents]:
        """
        Compute Phase 2 match reward on top of Phase 1 individual reward.

        Args:
            agent_idx: Index of this agent within its team.
            team_idx: 0 (home) or 1 (away).
            robot_states: All robot states indexed by team and player.
            ball_pos: Ball position.
            ball_vel: Ball velocity.
            game_state: GameController state dict.
            phase1_reward: The individual reward from Phase 1.
            phase1_components: The component breakdown from Phase 1.
        """
        rc = RewardComponents(**phase1_components.__dict__)

        opp_team = 1 - team_idx
        my_team = robot_states.get(team_idx, [])
        opp_team_states = robot_states.get(opp_team, [])

        if not my_team or agent_idx >= len(my_team):
            rc.total = phase1_reward
            return rc.total, rc

        me = my_team[agent_idx]
        my_pos = np.array(me["pos"][:2])

        # ── Possession reward ──
        # Team closest to ball gets possession reward
        my_team_dists = [np.linalg.norm(np.array(p["pos"][:2]) - ball_pos[:2])
                         for p in my_team]
        opp_dists = [np.linalg.norm(np.array(p["pos"][:2]) - ball_pos[:2])
                     for p in opp_team_states] if opp_team_states else [999]
        my_closest = min(my_team_dists) if my_team_dists else 999
        opp_closest = min(opp_dists) if opp_dists else 999
        if my_closest < opp_closest:
            rc.possession = self.w.team_ball_possession * 0.5
            if min(my_team_dists) == my_team_dists[agent_idx]:
                rc.possession += self.w.team_ball_possession * 0.5

        # ── Goal scored / conceded ──
        if game_state.get("goal_just_scored"):
            if game_state.get("scoring_team") == team_idx:
                rc.goal_scored = self.w.goal_scored
            else:
                rc.goal_conceded = self.w.goal_conceded

        # ── Tactical positioning ──
        # Reward spreading out on the field and maintaining formation
        rc.positioning = self._compute_positioning_reward(
            agent_idx, my_team, ball_pos, team_idx)

        # ── Defensive coverage ──
        # Reward being between opponents and own goal
        own_goal_x = -self.fi["half_length"] if team_idx == 0 \
                     else self.fi["half_length"]
        rc.defensive = self._compute_defensive_reward(
            my_pos, opp_team_states, ball_pos, own_goal_x)

        # ── Total ──
        rc.total = (phase1_reward + rc.possession + rc.goal_scored
                    + rc.goal_conceded + rc.positioning + rc.defensive)

        return rc.total, rc

    def _compute_positioning_reward(
        self, agent_idx, team_states, ball_pos, team_idx
    ) -> float:
        """Reward good field positioning relative to teammates."""
        if len(team_states) < 2:
            return 0.0

        my_pos = np.array(team_states[agent_idx]["pos"][:2])
        attack_dir = 1.0 if team_idx == 0 else -1.0
        hl = self.fi["half_length"]

        # Avoid clustering with teammates
        min_teammate_dist = float("inf")
        for i, t in enumerate(team_states):
            if i != agent_idx:
                d = np.linalg.norm(my_pos - np.array(t["pos"][:2]))
                min_teammate_dist = min(min_teammate_dist, d)

        spread_reward = min(1.0, min_teammate_dist / 2.0)

        # Goalkeeper should stay near own goal
        is_gk = (agent_idx == 0)
        if is_gk:
            own_goal_x = -attack_dir * hl
            dist_from_goal = abs(my_pos[0] - own_goal_x)
            gk_reward = max(0, 1.0 - dist_from_goal / 2.0)
            return self.w.positioning * gk_reward

        return self.w.positioning * spread_reward * 0.5

    def _compute_defensive_reward(
        self, my_pos, opponents, ball_pos, own_goal_x
    ) -> float:
        """Reward positioning between opponent ball-carrier and own goal."""
        if not opponents:
            return 0.0

        # Find opponent closest to ball
        opp_dists = [(np.linalg.norm(np.array(o["pos"][:2]) - ball_pos[:2]),
                      np.array(o["pos"][:2]))
                     for o in opponents]
        _, closest_opp_pos = min(opp_dists, key=lambda x: x[0])

        # Am I between the opponent and our goal?
        goal_pos = np.array([own_goal_x, 0])
        opp_to_goal = goal_pos - closest_opp_pos
        opp_to_me = my_pos - closest_opp_pos

        if np.linalg.norm(opp_to_goal) > 0.01:
            opp_to_goal_norm = opp_to_goal / np.linalg.norm(opp_to_goal)
            projection = np.dot(opp_to_me, opp_to_goal_norm)
            if 0 < projection < np.linalg.norm(opp_to_goal):
                # I'm between opponent and goal
                lateral_dist = np.linalg.norm(
                    opp_to_me - projection * opp_to_goal_norm)
                if lateral_dist < 1.5:
                    return self.w.defensive_coverage * (
                        1.0 - lateral_dist / 1.5)

        return 0.0

    # ── Curriculum Rewards ────────────────────────────────────────

    def compute_curriculum(
        self, stage: str, **kwargs
    ) -> Tuple[float, RewardComponents]:
        """Compute reward for a specific curriculum stage."""
        if stage == "stand":
            return self._reward_stand(**kwargs)
        elif stage == "walk":
            return self._reward_walk(**kwargs)
        elif stage == "dribble":
            return self._reward_dribble(**kwargs)
        elif stage == "shoot":
            return self._reward_shoot(**kwargs)
        else:  # "full"
            return self.compute_phase1(**kwargs)

    def _reward_stand(self, robot_quat, robot_pos, joint_pos, joint_vel,
                      foot_contacts, is_fallen, **_):
        """Standing balance reward."""
        rc = RewardComponents()
        up_vec = self._quat_rotate_vec(robot_quat, np.array([0, 0, 1]))
        rc.upright = 2.0 * max(0, up_vec[2])
        rc.alive = 1.0 if not is_fallen else 0.0
        rc.fall = -10.0 if is_fallen else 0.0
        # Penalize excessive joint velocities
        rc.energy = -0.01 * np.sum(joint_vel ** 2)
        # Reward foot contact
        rc.foot_contact = 0.5 if np.any(foot_contacts > 0.1) else 0.0
        rc.total = rc.upright + rc.alive + rc.fall + rc.energy + rc.foot_contact
        return rc.total, rc

    def _reward_walk(self, robot_vel, robot_quat, ball_pos, robot_pos,
                     is_fallen, foot_contacts, **_):
        """Walking toward ball reward."""
        rc = RewardComponents()
        # Upright
        up_vec = self._quat_rotate_vec(robot_quat, np.array([0, 0, 1]))
        rc.upright = 1.0 * max(0, up_vec[2])
        # Walk toward ball
        to_ball = ball_pos[:2] - robot_pos[:2]
        dist = np.linalg.norm(to_ball)
        if dist > 0.01:
            vel_toward = np.dot(robot_vel[:2], to_ball / dist)
            rc.forward_vel = 2.0 * max(0, vel_toward)
        rc.alive = 0.5 if not is_fallen else 0.0
        rc.fall = -10.0 if is_fallen else 0.0
        rc.foot_contact = 0.3 if np.any(foot_contacts > 0.1) else 0.0
        rc.total = (rc.upright + rc.forward_vel + rc.alive
                    + rc.fall + rc.foot_contact)
        return rc.total, rc

    def _reward_dribble(self, **kwargs):
        """Dribbling reward - uses Phase 1 with boosted dribble weight."""
        old_dribble_w = self.w.dribble_control
        self.w.dribble_control *= 2.0
        result = self.compute_phase1(**kwargs)
        self.w.dribble_control = old_dribble_w
        return result

    def _reward_shoot(self, **kwargs):
        """Shooting reward - uses Phase 1 with boosted kick weight."""
        old_kick_w = self.w.kick_reward
        self.w.kick_reward *= 2.0
        result = self.compute_phase1(**kwargs)
        self.w.kick_reward = old_kick_w
        return result

    # ── Utilities ─────────────────────────────────────────────────

    @staticmethod
    def _quat_rotate_vec(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion (w,x,y,z convention)."""
        w, x, y, z = quat
        # Quaternion rotation: q * v * q_conj
        t = 2.0 * np.cross(np.array([x, y, z]), vec)
        return vec + w * t + np.cross(np.array([x, y, z]), t)
