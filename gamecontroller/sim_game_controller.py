"""
Simulated GameController for RoboCup HSL match simulation.

Implements a simplified version of the RoboCup HSL GameController protocol
(https://github.com/RoboCup-HumanoidSoccerLeague/GameController) for use
in reinforcement learning environments.

The real GameController is a Rust application with a Tauri frontend that
broadcasts UDP messages (port 3838) in the RoboCupGameControlData format.
This module simulates that logic internally for training, without networking.
"""

import enum
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math


class GameState(enum.IntEnum):
    """Mirrors game_controller_core game states."""
    INITIAL = 0
    READY = 1
    SET = 2
    PLAYING = 3
    FINISHED = 4


class SetPlay(enum.IntEnum):
    """Set play types from the GameController."""
    NONE = 0
    GOAL_KICK = 1
    PUSHING_FREE_KICK = 2
    CORNER_KICK = 3
    KICK_IN = 4
    PENALTY_KICK = 5


class Penalty(enum.IntEnum):
    """Player penalty types."""
    NONE = 0
    SUBSTITUTE = 1
    MANUAL = 2
    PICKUP = 3
    PUSHING = 4
    FALLEN_INACTIVE = 5
    LOCAL_GAME_STUCK = 6
    BALL_HOLDING = 7
    PLAYER_STANCE = 8
    MOTION_IN_SET = 9


class Half(enum.IntEnum):
    FIRST = 0
    SECOND = 1


@dataclass
class PlayerState:
    """State of a single player."""
    player_number: int
    penalty: Penalty = Penalty.NONE
    penalty_timer: float = 0.0
    position: Tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    is_goalkeeper: bool = False
    fallen: bool = False
    active: bool = True


@dataclass
class TeamState:
    """State of a team."""
    team_number: int
    team_color: int  # 0=blue, 1=red
    score: int = 0
    penalty_shot_count: int = 0
    message_budget: int = 1200
    players: List[PlayerState] = field(default_factory=list)

    @property
    def active_players(self) -> List[PlayerState]:
        return [p for p in self.players if p.active
                and p.penalty == Penalty.NONE]


@dataclass
class GameControlData:
    """
    Mirrors the RoboCupGameControlData struct from
    game_controller_msgs/headers/RoboCupGameControlData.h

    The actual struct uses C-style packed format with header magic 'RGme'.
    Here we model it as a Python dataclass for RL integration.
    """
    # Game timing
    game_state: GameState = GameState.INITIAL
    half: Half = Half.FIRST
    secs_remaining: float = 600.0  # 10 minutes per half
    secondary_time: float = 0.0
    kick_off_team: int = 0  # 0 or 1

    # Set plays
    set_play: SetPlay = SetPlay.NONE
    set_play_team: int = 0

    # Teams
    teams: List[TeamState] = field(default_factory=list)

    # Ball
    ball_in_play: bool = False
    last_ball_contact_team: int = -1


class SimulatedGameController:
    """
    Simulates the RoboCup HSL GameController for training environments.

    The real GameController (written in Rust) manages the game via UDP:
    - Broadcasts control messages at 2 Hz on port 3838
    - Receives status messages from robots on port 3939
    - Manages penalties, substitutions, timeouts, etc.

    This simulator implements the core state machine and timing logic
    without networking, directly callable from the RL environment.
    """

    # Timing constants (from config/competition/params.yaml)
    HALF_DURATION = 600.0        # 10 minutes per half
    READY_DURATION = 30.0        # seconds in READY state
    SET_TIMEOUT = 15.0           # seconds in SET before whistle
    PENALTY_DURATION = 30.0      # seconds for standard penalty
    KICKOFF_DURATION = 10.0      # seconds for kick-off
    GOAL_CELEBRATION = 15.0      # seconds after goal

    def __init__(self, players_per_team: int = 4,
                 half_duration: float = None,
                 fast_forward: float = 1.0):
        """
        Args:
            players_per_team: Number of players per side.
            half_duration: Override half duration (for shorter training matches).
            fast_forward: Speed multiplier for game clock.
        """
        self.players_per_team = players_per_team
        self.fast_forward = fast_forward

        if half_duration is not None:
            self.HALF_DURATION = half_duration

        self.data = GameControlData()
        self._state_timer = 0.0
        self._goal_scored_this_step = False
        self._out_of_bounds = False
        self.reset()

    def reset(self, kick_off_team: int = 0):
        """Reset to initial state for a new match."""
        self.data = GameControlData(
            game_state=GameState.INITIAL,
            half=Half.FIRST,
            secs_remaining=self.HALF_DURATION,
            kick_off_team=kick_off_team,
            teams=[
                TeamState(
                    team_number=0, team_color=0,
                    players=[PlayerState(player_number=i+1,
                                        is_goalkeeper=(i == 0))
                             for i in range(self.players_per_team)],
                ),
                TeamState(
                    team_number=1, team_color=1,
                    players=[PlayerState(player_number=i+1,
                                        is_goalkeeper=(i == 0))
                             for i in range(self.players_per_team)],
                ),
            ],
        )
        self._state_timer = 0.0
        self._goal_scored_this_step = False

    def step(self, dt: float, ball_pos: Tuple[float, float, float],
             player_positions: Dict[int, List[Tuple[float, float]]],
             field_info: dict) -> GameControlData:
        """
        Advance the game controller by one time step.

        Args:
            dt: Time step in seconds.
            ball_pos: (x, y, z) ball position.
            player_positions: {team_idx: [(x,y), ...]} player positions.
            field_info: Field dimension dict.

        Returns:
            Updated GameControlData.
        """
        dt_scaled = dt * self.fast_forward
        self._state_timer += dt_scaled
        self._goal_scored_this_step = False

        if self.data.game_state == GameState.INITIAL:
            self._handle_initial()
        elif self.data.game_state == GameState.READY:
            self._handle_ready()
        elif self.data.game_state == GameState.SET:
            self._handle_set()
        elif self.data.game_state == GameState.PLAYING:
            self._handle_playing(dt_scaled, ball_pos, field_info)
        elif self.data.game_state == GameState.FINISHED:
            self._handle_finished()

        # Update penalty timers
        self._update_penalties(dt_scaled)

        # Update remaining time
        if self.data.game_state == GameState.PLAYING:
            self.data.secs_remaining -= dt_scaled
            if self.data.secs_remaining <= 0:
                self.data.secs_remaining = 0
                self._transition(GameState.FINISHED)

        return self.data

    def _handle_initial(self):
        """Auto-transition from INITIAL to READY."""
        if self._state_timer >= 2.0:
            self._transition(GameState.READY)

    def _handle_ready(self):
        """READY state: robots walk to positions."""
        if self._state_timer >= self.READY_DURATION:
            self._transition(GameState.SET)

    def _handle_set(self):
        """SET state: wait for whistle."""
        if self._state_timer >= 3.0:  # Short SET for training
            self._transition(GameState.PLAYING)
            self.data.ball_in_play = True

    def _handle_playing(self, dt: float, ball_pos: Tuple[float, float, float],
                        field_info: dict):
        """PLAYING: check for goals, out-of-bounds, etc."""
        hl = field_info["half_length"]
        hw = field_info["half_width"]
        gw = field_info["goal_width"] / 2
        bx, by, bz = ball_pos

        # Check for goals
        if abs(bx) > hl and abs(by) < gw and bz < field_info["goal_height"]:
            if bx > hl:
                # Ball crossed positive goal line → home team scores
                self.data.teams[0].score += 1
                self.data.kick_off_team = 1
            else:
                # Ball crossed negative goal line → away team scores
                self.data.teams[1].score += 1
                self.data.kick_off_team = 0
            self._goal_scored_this_step = True
            self._transition(GameState.READY)
            return

        # Check out of bounds
        if abs(bx) > hl + 0.5 or abs(by) > hw + 0.5:
            self._out_of_bounds = True
            # Determine set play type
            if abs(bx) > hl:
                # Goal kick or corner kick
                self.data.set_play = SetPlay.GOAL_KICK
            else:
                self.data.set_play = SetPlay.KICK_IN
            self._transition(GameState.READY)

    def _handle_finished(self):
        """Handle end of half or match."""
        if self.data.half == Half.FIRST:
            if self._state_timer >= 5.0:
                # Switch to second half
                self.data.half = Half.SECOND
                self.data.secs_remaining = self.HALF_DURATION
                self.data.kick_off_team = 1 - self.data.kick_off_team
                self._transition(GameState.READY)

    def _transition(self, new_state: GameState):
        """Transition to a new game state."""
        self.data.game_state = new_state
        self._state_timer = 0.0
        if new_state != GameState.PLAYING:
            self.data.ball_in_play = False
            self.data.set_play = SetPlay.NONE

    def _update_penalties(self, dt: float):
        """Decrement penalty timers."""
        for team in self.data.teams:
            for player in team.players:
                if player.penalty != Penalty.NONE and player.penalty_timer > 0:
                    player.penalty_timer -= dt
                    if player.penalty_timer <= 0:
                        player.penalty = Penalty.NONE
                        player.penalty_timer = 0.0

    def apply_penalty(self, team_idx: int, player_idx: int,
                      penalty: Penalty, duration: float = None):
        """Apply a penalty to a player."""
        if duration is None:
            duration = self.PENALTY_DURATION
        player = self.data.teams[team_idx].players[player_idx]
        player.penalty = penalty
        player.penalty_timer = duration

    @property
    def is_playing(self) -> bool:
        return self.data.game_state == GameState.PLAYING

    @property
    def goal_just_scored(self) -> bool:
        return self._goal_scored_this_step

    @property
    def score(self) -> Tuple[int, int]:
        return (self.data.teams[0].score, self.data.teams[1].score)

    @property
    def is_match_over(self) -> bool:
        return (self.data.game_state == GameState.FINISHED
                and self.data.half == Half.SECOND)

    def get_state_vector(self) -> list:
        """Return a flat vector representation for RL observations."""
        state = [
            float(self.data.game_state),
            float(self.data.half),
            self.data.secs_remaining / self.HALF_DURATION,
            float(self.data.kick_off_team),
            float(self.data.set_play),
            float(self.data.ball_in_play),
            float(self.data.teams[0].score),
            float(self.data.teams[1].score),
        ]
        return state

    def get_info_dict(self) -> dict:
        """Return info dict for logging."""
        return {
            "game_state": self.data.game_state.name,
            "half": self.data.half.name,
            "score": f"{self.data.teams[0].score}-{self.data.teams[1].score}",
            "secs_remaining": self.data.secs_remaining,
            "ball_in_play": self.data.ball_in_play,
        }
