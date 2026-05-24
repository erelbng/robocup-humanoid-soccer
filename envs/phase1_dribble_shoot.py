"""
Phase 1: Single-robot dribble & shoot environment (Genesis simulator).

Trains a Booster K1 robot to:
  1. Stand and balance (curriculum stage 1)
  2. Walk toward a ball (stage 2)
  3. Dribble the ball (stage 3)
  4. Shoot at goal (stage 4)
  5. Combined skills (stage 5)
"""

import json
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None

from configs.config import K1RobotConfig, Phase1Config, RewardWeights

from envs.rewards import RewardComponents, SoccerRewardFunction
from envs.perturbations import PerturbationSchedule, RobotPusher
from envs.style_command import (StyleCommandSampler, commanded_velocity_reward,
                                commanded_yaw_rate_reward)
from envs import standup as standup_mod
from envs.gait_rewards import compute_gait_shaping
from envs.domain_randomization import MotorRandomizer, DRConfig


class K1DribbleShootEnv:
    """
    Genesis-based environment for single-robot soccer skill training.

    Observation space (78-dim):
        - Robot base position (3): x, y, z
        - Robot base orientation quaternion (4): w, x, y, z
        - Robot base linear velocity (3): vx, vy, vz
        - Robot base angular velocity (3): wx, wy, wz
        - Joint positions (22): all DoFs
        - Joint velocities (22): all DoFs
        - Ball position relative to robot (3): dx, dy, dz
        - Ball velocity (3): vx, vy, vz
        - Goal position relative to robot (3): dx, dy, dz
        - Foot contact forces (2): left, right
        - Previous action (10 → compressed): key joint targets

    Action space (22-dim):
        - Joint position targets for all 22 DoFs
    """

    def __init__(
        self,
        cfg: Phase1Config = None,
        robot_cfg: K1RobotConfig = None,
        field_info: dict = None,
        render: bool = False,
        curriculum_stage: str = "full",
    ):
        self.cfg = cfg or Phase1Config()
        self.robot_cfg = robot_cfg or K1RobotConfig()
        self.render = render
        self.curriculum_stage = curriculum_stage

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

        # Reward function
        self.reward_fn = SoccerRewardFunction(self.cfg.reward, field_info)

        # State
        self.step_count = 0
        self.episode_reward = 0.0
        self.scene = None
        self.robot = None
        self.ball = None
        self.camera = None
        self._initialized = False

        # Auxiliaries (created on demand based on cfg flags)
        self._pusher: Optional[RobotPusher] = None
        self._style: Optional[StyleCommandSampler] = None
        self._dr: Optional[MotorRandomizer] = None
        self._prev_action = np.zeros(self.cfg.act_dim, dtype=np.float32)
        self._prev_foot_contacts = np.zeros(2, dtype=np.float32)

        # Set of poses to choose from when stage == "standup"
        self._standup_poses = standup_mod.all_poses()

    def _init_genesis(self):
        """Initialize Genesis scene (called once, lazily)."""
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
        except Exception:
            # Genesis is a process-global singleton; re-init raises but we
            # don't care if it's already up.
            pass

        self.scene = gs.Scene(
            show_viewer=self.render,
            sim_options=gs.options.SimOptions(
                dt=self.cfg.sim_dt,
                substeps=2,
            ),
            viewer_options=(
                gs.options.ViewerOptions(
                    res=(1280, 720),
                    camera_pos=(0, -8, 5),
                    camera_lookat=(0, 0, 0.5),
                    camera_fov=50,
                    max_FPS=60,
                )
                if self.render
                else None
            ),
            vis_options=(
                gs.options.VisOptions(
                    show_world_frame=False,
                    ambient_light=(0.4, 0.4, 0.4),
                )
                if self.render
                else None
            ),
        )

        # Soccer field (carpet + lines + goals). Loaded from the
        # auto-generated builder so dimensions stay in sync with
        # configs/field_hsl_2026.json. Falls back to a plain plane if the
        # builder is missing.
        field_added = False
        try:
            from models.field.field_genesis_builder import build_soccer_field
            build_soccer_field(self.scene)
            field_added = True
        except Exception as e:
            print(f"[phase1] field builder unavailable ({e}); using plain Plane")
        if not field_added:
            self.scene.add_entity(gs.morphs.Plane())

        # Load K1 robot
        urdf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "robot",
            "K1",
            "K1_22dof.urdf",
        )

        # The K1 trunk needs to start above its hip-to-foot height (~0.95m
        # total). Spawning lower (e.g. 0.55m) puts the feet under the ground
        # and the robot pops on the first step. We add a small clearance so
        # the robot has room to settle.
        spawn_z = 1.05

        if os.path.exists(urdf_path):
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file=urdf_path, pos=(0, 0, spawn_z),
                               merge_fixed_links=True),
            )
        else:
            mjcf_path = urdf_path.replace(".urdf", ".xml")
            if os.path.exists(mjcf_path):
                self.robot = self.scene.add_entity(
                    gs.morphs.MJCF(file=mjcf_path, pos=(0, 0, spawn_z)),
                )
            else:
                print(f"WARNING: Robot model not found at {urdf_path} or {mjcf_path}")
                print("Using placeholder robot (sphere for testing)")
                self.robot = self.scene.add_entity(
                    gs.morphs.Sphere(radius=0.15, pos=(0, 0, spawn_z)),
                )

        # Ball — drop slightly above carpet so it settles to z=0.07.
        # Try to use the procedurally-generated telstar texture; fall back
        # to plain white if it hasn't been generated yet (the texture is
        # produced by `python -m models.textures.make_ball_texture`).
        ball_tex_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "textures", "ball.png"
        )
        if os.path.exists(ball_tex_path):
            # `color` and `diffuse_texture` are mutually exclusive in
            # Genesis (the texture supplies the diffuse channel).
            ball_surface = gs.surfaces.Default(
                roughness=0.6,
                diffuse_texture=gs.textures.ImageTexture(
                    image_path=ball_tex_path
                ),
            )
        else:
            ball_surface = gs.surfaces.Default(
                color=(0.95, 0.95, 0.95, 1.0), roughness=0.6,
            )
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.07, pos=(1.0, 0, 0.10), collision=True),
            surface=ball_surface,
        )

        # Camera for video recording
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=(0, -6, 4),
            lookat=(0, 0, 0),
            fov=50,
        )

        self.scene.build()

        # Get joint indices
        self._setup_joint_mapping()

        # Configure PD gains on actuated joints so control_dofs_position()
        # acts as a position-controlled servo.
        try:
            n = len(self.dof_indices)
            kp = [float(self.robot_cfg.kp)] * n
            kd = [float(self.robot_cfg.kd)] * n
            self.robot.set_dofs_kp(kp, self.dof_indices)
            self.robot.set_dofs_kv(kd, self.dof_indices)
        except Exception as e:
            print(f"[phase1] could not set PD gains: {e}")

        # Drive robot to default joint pose so it starts standing-ish.
        try:
            target = list(self.robot_cfg.default_joint_pos)
            if len(target) == len(self.dof_indices):
                self.robot.set_dofs_position(target, self.dof_indices,
                                             zero_velocity=True)
        except Exception:
            pass

        # Push-robot perturbations
        if getattr(self.cfg, "use_perturbations", False):
            self._pusher = RobotPusher(
                schedule=PerturbationSchedule.for_stage(self.curriculum_stage),
            )
            self._pusher.attach(self.robot, trunk_link_name="Trunk")

        # Style command sampler
        if getattr(self.cfg, "use_style_command", False):
            self._style = StyleCommandSampler()

        # Motor / body / ball domain randomisation. Captures baseline
        # values RIGHT AFTER we set PD gains so multiplicative scaling
        # works around the intended defaults rather than whatever URDF
        # ships with.
        if getattr(self.cfg, "use_domain_randomization", False):
            self._dr = MotorRandomizer(act_dim=self.cfg.act_dim)
            self._dr.attach(self.robot, self.dof_indices, ball=self.ball)

        self._initialized = True

    def _setup_joint_mapping(self):
        """Resolve actuated joint DOF indices once after scene.build().

        Genesis's per-joint `dofs_idx_local` is a *list* (a free root joint
        spans 6 dofs, revolute joints span 1). We pick out the single-DoF
        revolute joints in URDF-declaration order and store a flat list of
        ints, which is what `control_dofs_position(targets, dofs_idx)`
        expects.
        """
        self.dof_indices = []
        joint_by_name = {}
        for j in getattr(self.robot, "joints", []):
            joint_by_name[j.name] = j

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
            # Fallback: assume DOF layout is "6 free root + N revolute"
            print("[phase1] joint name lookup failed; falling back to "
                  "sequential indices [6..6+num_dofs)")
            self.dof_indices = list(range(6, 6 + self.robot_cfg.num_dofs))

    def reset(self, seed: int = None) -> np.ndarray:
        """Reset environment for a new episode."""
        if not self._initialized:
            self._init_genesis()

        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.episode_reward = 0.0
        self.reward_fn.reset()
        self._prev_action[:] = 0.0

        # Reset scene
        if self.scene is not None and hasattr(self.scene, "reset"):
            self.scene.reset()

        # Stage-specific reset: pose the robot
        self._reset_robot_pose()

        # Ball placement (randomised per stage)
        self._reset_ball()

        # Resample style command for this episode
        if self._style is not None:
            self._style.sample()

        # Reset / re-schedule pusher
        if self._pusher is not None:
            self._pusher.set_schedule(
                PerturbationSchedule.for_stage(self.curriculum_stage)
            )

        # Sample fresh motor / body / ball parameters for this episode
        if self._dr is not None:
            self._dr.reset_episode()

        return self._get_observation()

    def _reset_robot_pose(self):
        """Place the robot in the stage-appropriate starting pose."""
        try:
            if self.curriculum_stage == "standup":
                # Pick a random fallen pose
                pose = self._standup_poses[
                    np.random.randint(len(self._standup_poses))
                ]
                # Position the trunk at fallen height + small XY jitter
                rx = np.random.uniform(-0.3, 0.3) if self.cfg.randomize_robot_pos else 0.0
                ry = np.random.uniform(-0.3, 0.3) if self.cfg.randomize_robot_pos else 0.0
                self.robot.set_pos([rx, ry, pose.trunk_height])
                self.robot.set_quat(list(pose.trunk_quat))
                # Drive actuated joints to the pose's targets
                targets = [pose.joint_targets.get(name, 0.0)
                           for name in self.robot_cfg.joint_names]
                self.robot.set_dofs_position(targets, self.dof_indices,
                                             zero_velocity=True)
            else:
                rx = np.random.uniform(-0.5, 0.5) if self.cfg.randomize_robot_pos else 0.0
                ry = np.random.uniform(-0.5, 0.5) if self.cfg.randomize_robot_pos else 0.0
                self.robot.set_pos([rx, ry, 1.05])
                self.robot.set_quat([1.0, 0.0, 0.0, 0.0])
                self.robot.set_dofs_position(
                    list(self.robot_cfg.default_joint_pos),
                    self.dof_indices, zero_velocity=True,
                )
        except Exception as e:
            print(f"[phase1] _reset_robot_pose failed: {e}")

    def _reset_ball(self):
        if self.ball is None or not hasattr(self.ball, "set_pos"):
            return
        # No ball during stand / standup — keep it far away so it doesn't
        # interfere with balance training.
        if self.curriculum_stage in ("stand", "standup"):
            bx, by, bz = 5.0, 5.0, 0.07
        elif self.cfg.randomize_ball_pos:
            bx = float(np.random.uniform(0.5, 3.0))
            by = float(np.random.uniform(-2.0, 2.0))
            bz = 0.07
        else:
            bx, by, bz = 1.0, 0.0, 0.07
        try:
            self.ball.set_pos([bx, by, bz])
        except Exception:
            pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: (22,) joint position targets for actuated joints.
                The base (free root) is NOT touched.

        Returns:
            obs, reward, done, info
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -math.pi, math.pi)

        # Truncate / pad action to match the number of actuated dofs
        n = len(self.dof_indices)
        if action.shape[0] != n:
            if action.shape[0] > n:
                action = action[:n]
            else:
                pad = np.zeros(n - action.shape[0], dtype=action.dtype)
                action = np.concatenate([action, pad])

        # Action delay + target noise (motor DR). Falls through to the
        # raw action if DR isn't active.
        applied = action
        if self._dr is not None:
            applied = self._dr.delay_action(action)

        # Apply as PD position targets — DO NOT use set_dofs_position which
        # is a hard state set and bypasses physics integration.
        if hasattr(self.robot, "control_dofs_position"):
            try:
                self.robot.control_dofs_position(
                    applied.tolist(), self.dof_indices
                )
            except Exception:
                pass

        # Optionally resample style command mid-episode (low-probability,
        # off by default but available).
        if self._style is not None:
            self._style.maybe_resample_step()

        # Step simulation, applying any active push at every phys-step
        for _ in range(self.cfg.action_repeat):
            if self._pusher is not None:
                self._pusher.maybe_push(phys_dt=self.cfg.sim_dt)
            if self.scene is not None:
                self.scene.step()

        self.step_count += 1

        # Get state
        obs = self._get_observation()

        # Compute reward
        reward, components = self._compute_reward(action)
        self.episode_reward += reward

        # Check termination
        done = self._check_done()

        info = {
            "step": self.step_count,
            "episode_reward": self.episode_reward,
            "reward_components": components.to_dict(),
            "curriculum_stage": self.curriculum_stage,
        }

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector.

        Layout: [base 78-dim physical obs | optional 5-dim style command].
        Width is exactly `self.cfg.obs_dim` regardless of stage so policy
        input dim stays constant.
        """
        # Placeholder when Genesis isn't available
        obs = np.zeros(self.cfg.obs_dim, dtype=np.float32)

        try:
            if hasattr(self.robot, "get_pos"):
                robot_pos = np.array(self.robot.get_pos())
                robot_quat = np.array(self.robot.get_quat())
                robot_vel = np.array(self.robot.get_vel())
                robot_angvel = np.array(self.robot.get_ang())
            else:
                robot_pos = np.zeros(3)
                robot_quat = np.array([1, 0, 0, 0])
                robot_vel = np.zeros(3)
                robot_angvel = np.zeros(3)

            if hasattr(self.robot, "get_dofs_position"):
                joint_pos = np.array(self.robot.get_dofs_position(self.dof_indices))
                joint_vel = np.array(self.robot.get_dofs_velocity(self.dof_indices))
            else:
                joint_pos = np.zeros(self.robot_cfg.num_dofs)
                joint_vel = np.zeros(self.robot_cfg.num_dofs)

            if hasattr(self.ball, "get_pos"):
                ball_pos = np.array(self.ball.get_pos())
                ball_vel = np.array(self.ball.get_vel())
            else:
                ball_pos = np.array([1.0, 0.0, 0.07])
                ball_vel = np.zeros(3)

            # Target goal (positive-x end)
            goal_pos = np.array([self.field_info["half_length"], 0, 0.4])

            # Relative positions
            ball_rel = ball_pos - robot_pos
            goal_rel = goal_pos - robot_pos

            # Pack observation
            idx = 0
            obs[idx : idx + 3] = robot_pos
            idx += 3
            obs[idx : idx + 4] = robot_quat
            idx += 4
            obs[idx : idx + 3] = robot_vel
            idx += 3
            obs[idx : idx + 3] = robot_angvel
            idx += 3
            obs[idx : idx + self.robot_cfg.num_dofs] = joint_pos
            idx += self.robot_cfg.num_dofs
            obs[idx : idx + self.robot_cfg.num_dofs] = joint_vel
            idx += self.robot_cfg.num_dofs
            obs[idx : idx + 3] = ball_rel
            idx += 3
            obs[idx : idx + 3] = ball_vel
            idx += 3
            obs[idx : idx + 3] = goal_rel
            idx += 3
            # Foot contacts (approximate)
            obs[idx : idx + 2] = [1.0, 1.0]  # placeholder
            idx += 2

            # Append style command if enabled. Width controlled by
            # cfg.style_command_dim so obs_dim arithmetic stays consistent.
            if self._style is not None:
                cmd = self._style.current.as_array()
                cmd_len = min(len(cmd),
                              max(0, self.cfg.obs_dim - self.cfg.base_obs_dim))
                base_end = self.cfg.base_obs_dim
                if cmd_len > 0:
                    obs[base_end : base_end + cmd_len] = cmd[:cmd_len]
        except Exception:
            pass

        return obs

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, RewardComponents]:
        """Compute reward based on current curriculum stage."""
        try:
            robot_pos = (
                np.array(self.robot.get_pos())
                if hasattr(self.robot, "get_pos")
                else np.zeros(3)
            )
            robot_quat = (
                np.array(self.robot.get_quat())
                if hasattr(self.robot, "get_quat")
                else np.array([1, 0, 0, 0])
            )
            robot_vel = (
                np.array(self.robot.get_vel())
                if hasattr(self.robot, "get_vel")
                else np.zeros(3)
            )
            robot_angvel = (
                np.array(self.robot.get_ang())
                if hasattr(self.robot, "get_ang")
                else np.zeros(3)
            )
            ball_pos = (
                np.array(self.ball.get_pos())
                if hasattr(self.ball, "get_pos")
                else np.array([1, 0, 0.07])
            )
            ball_vel = (
                np.array(self.ball.get_vel())
                if hasattr(self.ball, "get_vel")
                else np.zeros(3)
            )
            joint_pos = (
                np.array(self.robot.get_dofs_position(self.dof_indices))
                if hasattr(self.robot, "get_dofs_position")
                else np.zeros(self.robot_cfg.num_dofs)
            )
            joint_vel = (
                np.array(self.robot.get_dofs_velocity(self.dof_indices))
                if hasattr(self.robot, "get_dofs_velocity")
                else np.zeros(self.robot_cfg.num_dofs)
            )
        except Exception:
            return 0.0, RewardComponents()

        # Fall detection
        is_fallen = robot_pos[2] < 0.3

        # Joint limits (approximate)
        jl_lower = np.full(self.robot_cfg.num_dofs, -math.pi)
        jl_upper = np.full(self.robot_cfg.num_dofs, math.pi)

        target_goal = np.array([self.field_info["half_length"], 0.0, 0.4])
        foot_contacts = np.array([1.0, 1.0])  # placeholder

        kwargs = dict(
            robot_pos=robot_pos,
            robot_quat=robot_quat,
            robot_vel=robot_vel,
            robot_angvel=robot_angvel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_limits_lower=jl_lower,
            joint_limits_upper=jl_upper,
            actions=action,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            target_goal_pos=target_goal,
            foot_contacts=foot_contacts,
            is_fallen=is_fallen,
            dt=self.cfg.dt,
        )

        # Standup stage: use dedicated standup reward
        if self.curriculum_stage == "standup":
            r, comps = standup_mod.compute_standup_reward(
                robot_quat=robot_quat,
                robot_z=float(robot_pos[2]),
                joint_vel=joint_vel,
                actions=action,
                prev_actions=self._prev_action,
            )
            # Style-command rewards don't apply in standup
            self._prev_action = action.astype(np.float32)
            return r, RewardComponents()  # log via info dict downstream

        # Base curriculum reward
        if self.curriculum_stage in ("stand", "walk", "dribble", "shoot"):
            r, comps = self.reward_fn.compute_curriculum(
                self.curriculum_stage, **kwargs
            )
        else:
            r, comps = self.reward_fn.compute_phase1(**kwargs)

        # Add style-command tracking rewards for walk/dribble/shoot/full
        commanded_vy = 0.0
        if self._style is not None and self.curriculum_stage in (
            "walk", "dribble", "shoot", "full"
        ):
            cmd = self._style.current
            r += 1.5 * commanded_velocity_reward(robot_vel, robot_quat, cmd)
            r += 0.5 * commanded_yaw_rate_reward(robot_angvel, cmd)
            r += 0.5 * cmd.aggressiveness * commanded_velocity_reward(
                robot_vel, robot_quat, cmd, tracking_sigma=0.18,
            )
            commanded_vy = cmd.vy

        # Low-level gait shaping (small but consistent signal — fastest
        # convergence comes from this rather than the high-level rewards)
        if self.curriculum_stage in ("walk", "dribble", "shoot", "full"):
            try:
                feet_h = self._get_feet_heights()
                gait_r, gait_parts = compute_gait_shaping(
                    robot_quat=robot_quat,
                    robot_z=float(robot_pos[2]),
                    robot_vel=robot_vel,
                    robot_angvel=robot_angvel,
                    joint_vel=joint_vel,
                    joint_pos=joint_pos,
                    joint_limits_lower=jl_lower,
                    joint_limits_upper=jl_upper,
                    action=action,
                    prev_action=self._prev_action,
                    foot_contacts=foot_contacts,
                    prev_foot_contacts=self._prev_foot_contacts,
                    feet_heights=feet_h,
                    commanded_vy=commanded_vy,
                )
                r += gait_r
            except Exception:
                pass

        self._prev_action = action.astype(np.float32)
        self._prev_foot_contacts = foot_contacts.astype(np.float32)
        return r, comps

    def _get_feet_heights(self) -> np.ndarray:
        """Best-effort feet z lookup. Returns zeros if foot links not found."""
        if self.robot is None:
            return np.zeros(2, dtype=np.float32)
        out = np.zeros(2, dtype=np.float32)
        for i, name in enumerate(("left_foot_link", "right_foot_link")):
            try:
                link = self.robot.get_link(name)
                pos = link.get_pos()
                if hasattr(pos, "cpu"):
                    pos = pos.cpu().numpy()
                out[i] = float(np.atleast_1d(pos).flatten()[2])
            except Exception:
                pass
        return out

    def _check_done(self) -> bool:
        """Check if episode should terminate.

        For `standup`: terminate on success (upright at standing height) so
        the policy gets the success bonus and a fresh episode. Otherwise
        the only timeout terminates the episode.

        For walk/dribble/shoot/full: a fall used to be an immediate
        terminate. We now keep the episode alive long enough that the
        policy can attempt recovery — only terminate if the robot has
        been "deeply fallen" (trunk almost flat on the ground) for too
        many consecutive steps.
        """
        if self.step_count >= self.cfg.max_episode_steps:
            return True

        try:
            robot_pos = np.array(self.robot.get_pos()).flatten()
            robot_quat = np.array(self.robot.get_quat()).flatten()
        except Exception:
            return False

        # Standup stage: terminate on success
        if self.curriculum_stage == "standup":
            return standup_mod.standup_success(robot_quat, float(robot_pos[2]))

        # Other stages: only terminate on a deep fall (trunk z very low)
        if float(robot_pos[2]) < 0.10:
            return True

        # Ball out of bounds (only relevant when ball is in play)
        if self.curriculum_stage not in ("stand", "standup"):
            try:
                ball_pos = np.array(self.ball.get_pos()).flatten()
                hl = self.field_info["half_length"] + 1.0
                hw = self.field_info["half_width"] + 1.0
                if abs(ball_pos[0]) > hl or abs(ball_pos[1]) > hw:
                    return True
            except Exception:
                pass

        return False

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
        """Clean up resources."""
        if self.scene is not None:
            del self.scene
            self.scene = None
        self._initialized = False
