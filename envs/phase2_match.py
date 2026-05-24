"""
Phase 2: Multi-robot soccer match environment (Genesis simulator).

Fine-tunes pre-trained Phase 1 policies in full match settings using:
  - Multiple K1 robots per team (4v4 default)
  - Simulated RoboCup HSL GameController protocol
  - Self-play training with opponent pool
  - Tactical reward shaping
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

from configs.config import K1RobotConfig, Phase2Config
from gamecontroller.sim_game_controller import (GameControlData, GameState,
                                                SimulatedGameController)

from envs.rewards import RewardComponents, SoccerRewardFunction


class K1SoccerMatchEnv:
    """
    Multi-agent soccer match environment using Genesis.

    Each agent observes:
      - Own state (pos, quat, vel, angvel, joints) = 35
      - Ball state relative to self (pos, vel) = 6
      - Teammate states relative to self (N-1 * 5) = 15 (3 teammates x 5)
      - Opponent states relative to self (N * 5) = 20 (4 opponents x 5)
      - Goal positions relative to self = 6 (own + opp)
      - GameController state = 8
      - Previous action summary = varies
      Total ≈ 156 depending on team size

    Actions: 22 joint position targets per robot.
    """

    def __init__(
        self,
        cfg: Phase2Config = None,
        robot_cfg: K1RobotConfig = None,
        field_info: dict = None,
        render: bool = False,
    ):
        self.cfg = cfg or Phase2Config()
        self.robot_cfg = robot_cfg or K1RobotConfig()
        self.render = render
        self.n_per_team = self.cfg.players_per_team
        self.n_agents = 2 * self.n_per_team

        # Load field info
        if field_info is None:
            field_info_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "field", "field_info.json"
            )
            if os.path.exists(field_info_path):
                with open(field_info_path) as f:
                    field_info = json.load(f)
            else:
                field_info = {
                    "length": 9.0,
                    "width": 6.0,
                    "half_length": 4.5,
                    "half_width": 3.0,
                    "goal_width": 2.6,
                    "goal_height": 0.8,
                    "goal_depth": 0.6,
                    "penalty_area_length": 1.0,
                    "penalty_area_width": 3.0,
                    "penalty_mark_distance": 1.5,
                    "center_circle_radius": 0.75,
                    "total_length": 11.0,
                    "total_width": 8.0,
                    "border_strip_width": 1.0,
                }
        self.field_info = field_info

        # GameController
        self.gc = SimulatedGameController(
            players_per_team=self.n_per_team,
            half_duration=self.cfg.half_duration,
            fast_forward=10.0,  # speed up game clock for training
        )

        # Reward functions per agent
        self.reward_fns = {}
        for team in range(2):
            for i in range(self.n_per_team):
                agent_id = team * self.n_per_team + i
                self.reward_fns[agent_id] = SoccerRewardFunction(
                    self.cfg.reward, field_info
                )

        # Simulation objects
        self.scene = None
        self.robots = {}  # {agent_id: robot_entity}
        self.ball = None
        self._initialized = False

        # State tracking
        self.step_count = 0
        self.episode_rewards = {i: 0.0 for i in range(self.n_agents)}

    def _init_genesis(self):
        """Initialize the Genesis scene with all robots and field."""
        if self._initialized or gs is None:
            return

        try:
            gs.init(
                backend=gs.gpu,
                precision="32",
                logging_level="warning",
                seed=1,
                performance_mode=True,
            )
        except:
            print("Genesis already initialized")

        self.scene = gs.Scene(
            show_viewer=self.render,
            sim_options=gs.options.SimOptions(
                dt=self.cfg.sim_dt,
                substeps=2,
            ),
            viewer_options=(
                gs.options.ViewerOptions(
                    res=(1280, 720),
                    camera_pos=(0, 0, 12),
                    camera_lookat=(0, 0, 0),
                    camera_fov=60,
                    max_FPS=30,
                )
                if self.render
                else None
            ),
        )

        # Ground
        self.scene.add_entity(gs.morphs.Plane())

        # Build field elements (simplified for multi-agent performance)
        self._build_field()

        # Load robots
        self._load_robots()

        # Ball
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.07, pos=(0, 0, 0.07)),
        )

        # Camera for recording
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=(0, 0, 12),
            lookat=(0, 0, 0),
            fov=60,
        )

        self.scene.build()
        self._initialized = True

    def _build_field(self):
        """Add simplified field markings to the Genesis scene."""
        hl = self.field_info["half_length"]
        hw = self.field_info["half_width"]
        tl = self.field_info["total_length"]
        tw = self.field_info["total_width"]

        # Green carpet
        self.scene.add_entity(
            gs.morphs.Box(
                size=(tl, tw, 0.02),
                pos=(0, 0, -0.01),
                fixed=True,
            )
        )

        # Goal structures (simplified)
        for sign in [1, -1]:
            gx = sign * hl
            gw = self.field_info["goal_width"] / 2
            gh = self.field_info["goal_height"]
            # Posts
            for y_sign in [1, -1]:
                self.scene.add_entity(
                    gs.morphs.Cylinder(
                        radius=0.05,
                        height=gh,
                        pos=(gx, y_sign * gw, gh / 2),
                        fixed=True,
                    )
                )
            # Crossbar
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.05, gw * 2, 0.05),
                    pos=(gx, 0, gh),
                    fixed=True,
                )
            )

    def _load_robots(self):
        """Load K1 robots for both teams."""
        positions = self._get_starting_positions()

        urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "robot", "K1", "K1_22dof.urdf"
        )

        for team in range(2):
            for i in range(self.n_per_team):
                agent_id = team * self.n_per_team + i
                x, y, heading = positions[team][i]

                if os.path.exists(urdf_path):
                    robot = self.scene.add_entity(
                        gs.morphs.URDF(
                            file=urdf_path,
                            pos=(x, y, 0.55),
                            euler=(0, 0, heading),
                        ),
                    )
                else:
                    # Placeholder
                    robot = self.scene.add_entity(
                        gs.morphs.Sphere(
                            radius=0.15,
                            pos=(x, y, 0.55),
                        ),
                    )
                self.robots[agent_id] = robot

    def _get_starting_positions(self) -> Dict[int, List[Tuple]]:
        """Get kickoff positions for all players."""
        hl = self.field_info["half_length"]
        hw = self.field_info["half_width"]
        n = self.n_per_team

        home = []
        away = []

        # Standard formation
        formations = {
            1: [(-hl + 0.3, 0, 0)],  # GK only
            2: [(-hl + 0.3, 0, 0), (-1.0, 0, 0)],
            3: [(-hl + 0.3, 0, 0), (-2.0, 1.0, 0), (-0.5, 0, 0)],
            4: [(-hl + 0.3, 0, 0), (-2.0, 1.0, 0), (-2.0, -1.0, 0), (-0.5, 0, 0)],
        }

        n_clamp = min(n, 4)
        for x, y, h in formations.get(n_clamp, formations[4]):
            home.append((x, y, h))
            away.append((-x, -y, h + math.pi))

        # Extra players
        for i in range(n_clamp, n):
            y_off = (i - 3) * 0.5 * (1 if i % 2 else -1)
            home.append((-hl / 3, y_off, 0))
            away.append((hl / 3, -y_off, math.pi))

        return {0: home, 1: away}

    def reset(self, seed: int = None) -> Dict[int, np.ndarray]:
        """Reset for a new match."""
        if not self._initialized:
            self._init_genesis()

        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.episode_rewards = {i: 0.0 for i in range(self.n_agents)}

        # Reset GameController
        self.gc.reset(kick_off_team=np.random.randint(0, 2))

        # Reset reward functions
        for rf in self.reward_fns.values():
            rf.reset()

        # Reset scene
        if self.scene is not None and hasattr(self.scene, "reset"):
            self.scene.reset()

        return self._get_all_observations()

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Step all agents simultaneously.

        Args:
            actions: {agent_id: (22,) action array}

        Returns:
            observations, rewards, dones, infos (all dicts keyed by agent_id)
        """
        # Apply actions
        for agent_id, action in actions.items():
            action = np.clip(action, -math.pi, math.pi)
            robot = self.robots.get(agent_id)
            if robot is not None and hasattr(robot, "set_dofs_position"):
                dof_idx = list(range(self.robot_cfg.num_dofs))
                robot.set_dofs_position(action, dof_idx)

        # Step simulation
        for _ in range(10):  # action repeat
            if self.scene is not None:
                self.scene.step()

        self.step_count += 1

        # Get ball state
        ball_pos = self._get_ball_pos()
        ball_vel = self._get_ball_vel()

        # Get all player positions for GameController
        player_positions = self._get_all_player_positions()

        # Step GameController
        gc_data = self.gc.step(
            self.cfg.dt, tuple(ball_pos), player_positions, self.field_info
        )

        # Compute observations, rewards, dones
        observations = self._get_all_observations()
        rewards = {}
        dones = {}
        infos = {}

        robot_states = self._get_robot_states()

        for agent_id in range(self.n_agents):
            team_idx = agent_id // self.n_per_team
            player_idx = agent_id % self.n_per_team

            # Phase 1 reward (individual skills)
            p1_reward, p1_components = self._compute_individual_reward(
                agent_id, actions.get(agent_id, np.zeros(self.robot_cfg.num_dofs))
            )

            # Phase 2 reward (tactical, team)
            game_state = {
                "goal_just_scored": self.gc.goal_just_scored,
                "scoring_team": (
                    0 if self.gc.goal_just_scored and gc_data.kick_off_team == 1 else 1
                ),
                **self.gc.get_info_dict(),
            }

            reward, components = self.reward_fns[agent_id].compute_phase2(
                agent_idx=player_idx,
                team_idx=team_idx,
                robot_states=robot_states,
                ball_pos=ball_pos,
                ball_vel=ball_vel,
                game_state=game_state,
                phase1_reward=p1_reward,
                phase1_components=p1_components,
            )

            rewards[agent_id] = reward
            self.episode_rewards[agent_id] += reward

            # Termination
            done = (
                self.step_count >= self.cfg.max_episode_steps or self.gc.is_match_over
            )
            dones[agent_id] = done

            infos[agent_id] = {
                "reward_components": components.to_dict(),
                "game_controller": self.gc.get_info_dict(),
                "team": team_idx,
                "player": player_idx,
                "episode_reward": self.episode_rewards[agent_id],
            }

        return observations, rewards, dones, infos

    def _get_all_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents."""
        obs = {}
        for agent_id in range(self.n_agents):
            obs[agent_id] = self._get_single_observation(agent_id)
        return obs

    def _get_single_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a single agent."""
        obs = np.zeros(self.cfg.obs_dim, dtype=np.float32)

        try:
            robot = self.robots[agent_id]
            team_idx = agent_id // self.n_per_team
            player_idx = agent_id % self.n_per_team

            # Self state
            pos = (
                np.array(robot.get_pos()) if hasattr(robot, "get_pos") else np.zeros(3)
            )
            quat = (
                np.array(robot.get_quat())
                if hasattr(robot, "get_quat")
                else np.array([1, 0, 0, 0])
            )
            vel = (
                np.array(robot.get_vel()) if hasattr(robot, "get_vel") else np.zeros(3)
            )
            angvel = (
                np.array(robot.get_ang()) if hasattr(robot, "get_ang") else np.zeros(3)
            )
            jpos = (
                np.array(robot.get_dofs_position(list(range(self.robot_cfg.num_dofs))))
                if hasattr(robot, "get_dofs_position")
                else np.zeros(self.robot_cfg.num_dofs)
            )
            jvel = (
                np.array(robot.get_dofs_velocity(list(range(self.robot_cfg.num_dofs))))
                if hasattr(robot, "get_dofs_velocity")
                else np.zeros(self.robot_cfg.num_dofs)
            )

            ball_pos = self._get_ball_pos()
            ball_vel = self._get_ball_vel()

            idx = 0
            # Self state (35)
            obs[idx : idx + 3] = pos
            idx += 3
            obs[idx : idx + 4] = quat
            idx += 4
            obs[idx : idx + 3] = vel
            idx += 3
            obs[idx : idx + 3] = angvel
            idx += 3
            obs[idx : idx + 22] = jpos
            idx += 22

            # Ball relative (6)
            obs[idx : idx + 3] = ball_pos - pos
            idx += 3
            obs[idx : idx + 3] = ball_vel
            idx += 3

            # Teammate relative positions (15 = 3 teammates * 5)
            for i in range(self.n_per_team):
                if i != player_idx:
                    teammate_id = team_idx * self.n_per_team + i
                    t_robot = self.robots[teammate_id]
                    t_pos = (
                        np.array(t_robot.get_pos())
                        if hasattr(t_robot, "get_pos")
                        else np.zeros(3)
                    )
                    t_vel = (
                        np.array(t_robot.get_vel())
                        if hasattr(t_robot, "get_vel")
                        else np.zeros(3)
                    )
                    obs[idx : idx + 3] = t_pos - pos
                    idx += 3
                    obs[idx : idx + 2] = t_vel[:2]
                    idx += 2

            # Opponent relative positions (20 = 4 opponents * 5)
            opp_team = 1 - team_idx
            for i in range(self.n_per_team):
                opp_id = opp_team * self.n_per_team + i
                o_robot = self.robots[opp_id]
                o_pos = (
                    np.array(o_robot.get_pos())
                    if hasattr(o_robot, "get_pos")
                    else np.zeros(3)
                )
                o_vel = (
                    np.array(o_robot.get_vel())
                    if hasattr(o_robot, "get_vel")
                    else np.zeros(3)
                )
                obs[idx : idx + 3] = o_pos - pos
                idx += 3
                obs[idx : idx + 2] = o_vel[:2]
                idx += 2

            # Goal positions relative (6)
            hl = self.field_info["half_length"]
            own_goal = np.array([-hl if team_idx == 0 else hl, 0, 0.4])
            opp_goal = np.array([hl if team_idx == 0 else -hl, 0, 0.4])
            obs[idx : idx + 3] = own_goal - pos
            idx += 3
            obs[idx : idx + 3] = opp_goal - pos
            idx += 3

            # GameController state (8)
            gc_vec = self.gc.get_state_vector()
            obs[idx : idx + 8] = gc_vec
            idx += 8

        except Exception:
            pass

        return obs

    def _compute_individual_reward(
        self, agent_id: int, action: np.ndarray
    ) -> Tuple[float, RewardComponents]:
        """Compute Phase 1 individual reward for an agent."""
        try:
            robot = self.robots[agent_id]
            team_idx = agent_id // self.n_per_team

            pos = (
                np.array(robot.get_pos())
                if hasattr(robot, "get_pos")
                else np.array([0, 0, 0.55])
            )
            quat = (
                np.array(robot.get_quat())
                if hasattr(robot, "get_quat")
                else np.array([1, 0, 0, 0])
            )
            vel = (
                np.array(robot.get_vel()) if hasattr(robot, "get_vel") else np.zeros(3)
            )
            angvel = (
                np.array(robot.get_ang()) if hasattr(robot, "get_ang") else np.zeros(3)
            )
            jpos = np.zeros(self.robot_cfg.num_dofs)
            jvel = np.zeros(self.robot_cfg.num_dofs)
            ball_pos = self._get_ball_pos()
            ball_vel = self._get_ball_vel()

            hl = self.field_info["half_length"]
            target_goal = np.array([hl if team_idx == 0 else -hl, 0, 0.4])

            return self.reward_fns[agent_id].compute_phase1(
                robot_pos=pos,
                robot_quat=quat,
                robot_vel=vel,
                robot_angvel=angvel,
                joint_pos=jpos,
                joint_vel=jvel,
                joint_limits_lower=np.full(22, -math.pi),
                joint_limits_upper=np.full(22, math.pi),
                actions=action,
                ball_pos=ball_pos,
                ball_vel=ball_vel,
                target_goal_pos=target_goal,
                foot_contacts=np.array([1.0, 1.0]),
                is_fallen=pos[2] < 0.3,
                dt=self.cfg.dt,
            )
        except Exception:
            return 0.0, RewardComponents()

    def _get_ball_pos(self) -> np.ndarray:
        if self.ball is not None and hasattr(self.ball, "get_pos"):
            return np.array(self.ball.get_pos())
        return np.array([0, 0, 0.07])

    def _get_ball_vel(self) -> np.ndarray:
        if self.ball is not None and hasattr(self.ball, "get_vel"):
            return np.array(self.ball.get_vel())
        return np.zeros(3)

    def _get_all_player_positions(self) -> Dict:
        positions = {0: [], 1: []}
        for agent_id, robot in self.robots.items():
            team = agent_id // self.n_per_team
            pos = (
                np.array(robot.get_pos()) if hasattr(robot, "get_pos") else np.zeros(3)
            )
            positions[team].append(tuple(pos[:2]))
        return positions

    def _get_robot_states(self) -> Dict:
        states = {0: [], 1: []}
        for agent_id, robot in self.robots.items():
            team = agent_id // self.n_per_team
            pos = (
                np.array(robot.get_pos()) if hasattr(robot, "get_pos") else np.zeros(3)
            )
            vel = (
                np.array(robot.get_vel()) if hasattr(robot, "get_vel") else np.zeros(3)
            )
            states[team].append({"pos": pos.tolist(), "vel": vel.tolist()})
        return states

    def render_frame(self) -> Optional[np.ndarray]:
        """Capture a frame for video recording."""
        if self.camera is not None:
            try:
                render_out = self.camera.render()

                # Genesis returns a tuple of (rgb, depth, segmentation, normal)
                # We only want the first element (RGB)
                if isinstance(render_out, tuple):
                    rgb_frame = render_out[0]
                else:
                    rgb_frame = render_out

                # Ensure it's a numpy array
                if hasattr(rgb_frame, "cpu"):
                    rgb_frame = rgb_frame.cpu().numpy()

                return rgb_frame
            except Exception as e:
                print(f"Render error: {e}")
        return None

    def close(self):
        if self.scene is not None:
            del self.scene
            self.scene = None
        self._initialized = False
