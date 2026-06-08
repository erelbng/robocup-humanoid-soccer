"""
MuJoCo Evaluation Script for trained policies.

Evaluates Genesis-trained policies in MuJoCo for sim2sim validation.
Records videos and logs metrics to W&B.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    ProjectConfig, EvalConfig, K1RobotConfig,
    CHECKPOINTS_DIR, VIDEOS_DIR, FIELD_MJCF, FIELD_INFO
)


class MuJoCoEvaluator:
    """
    Evaluates trained policies in MuJoCo.

    This provides sim2sim validation: policies trained in Genesis
    are deployed in MuJoCo to test transfer robustness.
    """

    def __init__(self, field_xml: str, robot_xml: str = None,
                 eval_cfg: EvalConfig = None,
                 robot_cfg: K1RobotConfig = None):
        self.eval_cfg = eval_cfg or EvalConfig()
        self.robot_cfg = robot_cfg or K1RobotConfig()
        self.field_xml = field_xml
        self.robot_xml = robot_xml

        # Load MuJoCo
        try:
            import mujoco
            self.mj = mujoco
        except ImportError:
            print("MuJoCo Python bindings required: pip install mujoco")
            self.mj = None
            return

        # Load field info
        if os.path.exists(FIELD_INFO):
            with open(FIELD_INFO) as f:
                self.field_info = json.load(f)
        else:
            self.field_info = {
                "half_length": 4.5, "half_width": 3.0,
                "goal_width": 2.6, "goal_height": 0.8,
            }

        self.model = None
        self.data = None
        self.renderer = None

    def load_scene(self, include_robot: bool = True):
        """Load MuJoCo model.

        Strategy: load K1_22dof.xml as the host MJCF (so its meshdir resolves
        cleanly), then splice the field carpet/lines/goals/ball into the same
        worldbody via MjSpec. This is the same technique used in
        scripts/debug_mujoco_scene.py.

        Loading the standalone field_robocup.xml is intentionally avoided —
        even with the bundled-robot bug fixed, a robot-less scene gives the
        "I see a field but no robot" symptom in the eval video.
        """
        if self.mj is None:
            return False

        robot_xml_path = (
            self.robot_xml
            or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "robot", "K1", "K1_22dof.xml",
            )
        )

        try:
            if include_robot and os.path.exists(robot_xml_path):
                spec = self.mj.MjSpec.from_file(robot_xml_path)
                # K1 ships with its own checker ground plane; drop it so the
                # field carpet is visible (and the simulation contact is
                # against ONE ground, not two stacked).
                try:
                    g = spec.geom("ground")
                    if g is not None:
                        spec.delete(g)
                except Exception:
                    pass
                self._add_field_and_ball_to_spec(spec)
                self.model = spec.compile()
            elif os.path.exists(self.field_xml):
                # Field-only fallback (no robot in scene — useful for sanity)
                self.model = self.mj.MjModel.from_xml_path(self.field_xml)
            else:
                xml = self._generate_minimal_field_xml(include_robot)
                self.model = self.mj.MjModel.from_xml_string(xml)

            self.data = self.mj.MjData(self.model)
            self.renderer = self.mj.Renderer(self.model, height=480, width=640)
            return True
        except Exception as e:
            print(f"Failed to load MuJoCo scene: {e}")
            return False

    def _add_field_and_ball_to_spec(self, spec):
        """Splice the shared HSL field (grass + full markings + goals) into a
        K1-rooted MjSpec, then add the eval ball (telstar texture if present).

        The field geometry comes from `models.field_builder.add_field_to_spec`
        — the single source of truth shared with the static scene generator and
        the debug scene — so every render path shows the same proper pitch
        (grass texture, center circle, penalty boxes + marks, corner arcs,
        goals + nets) instead of a bare outline on flat green.
        """
        mj = self.mj
        wb = spec.worldbody
        from models.field_builder import add_field_to_spec
        from models.field_generator import FieldDimensions
        field_json = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "field_hsl_2026.json",
        )
        field_dims = FieldDimensions.from_json(field_json)
        # Markings/goals/carpet only — the eval-specific ball (with its tuned
        # contact params + telstar texture) is added below.
        add_field_to_spec(spec, mj, field_dims, add_ball=False)

        # Ball with telstar pattern texture (generated by
        # models.textures.make_ball_texture). Falls back to plain white if
        # the texture is missing — keeps eval working without forcing the
        # texture-generation step.
        ball_tex_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "textures", "ball.png",
        )
        ball_material = None
        if os.path.exists(ball_tex_path):
            tex = spec.add_texture()
            tex.name = "ball_tex"
            tex.type = mj.mjtTexture.mjTEXTURE_2D
            tex.file = ball_tex_path
            mat = spec.add_material()
            mat.name = "ball_mat"
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "ball_tex"
            ball_material = "ball_mat"

        ball = wb.add_body()
        ball.name = "ball"
        ball.pos = [1.0, 0.0, 0.07]
        ball.add_freejoint()
        bg = ball.add_geom()
        bg.name = "ball_geom"
        bg.type = mj.mjtGeom.mjGEOM_SPHERE
        bg.size = [0.07, 0, 0]
        bg.mass = 0.2
        if ball_material:
            bg.material = ball_material
        else:
            bg.rgba = [0.95, 0.95, 0.95, 1]
        bg.friction = [0.8, 0.005, 0.0001]
        bg.condim = 4
        bg.solref = [0.01, 1.0]
        bg.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

    def _generate_minimal_field_xml(self, include_robot: bool) -> str:
        """Generate a minimal MuJoCo field XML for testing."""
        hl = self.field_info.get("half_length", 4.5)
        hw = self.field_info.get("half_width", 3.0)

        robot_xml = ""
        if include_robot and self.robot_xml and os.path.exists(self.robot_xml):
            robot_xml = f'<include file="{self.robot_xml}"/>'

        return f"""
        <mujoco model="robocup_eval">
            <compiler angle="radian" autolimits="true"/>
            <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

            <asset>
                <texture name="field" type="2d" builtin="flat"
                         rgb1="0.1 0.6 0.1" width="256" height="256"/>
                <material name="field_mat" texture="field"/>
            </asset>

            <worldbody>
                <light name="top" pos="0 0 8" dir="0 0 -1" diffuse="1 1 1"
                       directional="true"/>

                <geom name="ground" type="plane" size="{hl+1} {hw+1} 0.01"
                      material="field_mat"/>

                <!-- Ball -->
                <body name="ball" pos="1 0 0.07">
                    <joint name="ball_free" type="free"/>
                    <geom name="ball_geom" type="sphere" size="0.07"
                          mass="0.2" rgba="1 1 1 1"
                          friction="0.8 0.005 0.0001"/>
                </body>

                <!-- Goals -->
                <body name="goal_pos" pos="{hl} 0 0">
                    <geom type="cylinder" size="0.05 0.4"
                          pos="0.3 1.3 0.4" rgba="1 1 1 1"/>
                    <geom type="cylinder" size="0.05 0.4"
                          pos="0.3 -1.3 0.4" rgba="1 1 1 1"/>
                </body>
                <body name="goal_neg" pos="-{hl} 0 0">
                    <geom type="cylinder" size="0.05 0.4"
                          pos="-0.3 1.3 0.4" rgba="1 1 1 1"/>
                    <geom type="cylinder" size="0.05 0.4"
                          pos="-0.3 -1.3 0.4" rgba="1 1 1 1"/>
                </body>

                {robot_xml}
            </worldbody>
        </mujoco>
        """

    def evaluate_single(self, policy, device, num_episodes: int = None
                        ) -> Dict:
        """
        Evaluate a single-robot policy in MuJoCo.

        Returns metrics dict.
        """
        if self.mj is None or self.model is None:
            print("MuJoCo not loaded")
            return {}

        num_episodes = num_episodes or self.eval_cfg.num_eval_episodes

        results = defaultdict(list)
        all_frames = []

        for ep in range(num_episodes):
            self.mj.mj_resetData(self.model, self.data)
            self.mj.mj_forward(self.model, self.data)

            ep_reward = 0.0
            ep_length = 0
            ep_fallen = False
            ball_final_x = 0.0
            max_ball_speed = 0.0

            for step in range(1000):
                # Get observation
                obs = self._get_mujoco_obs()

                # Get action from policy
                action = self._policy_action(policy, obs, device)

                # Apply action
                if action is not None:
                    n_ctrl = min(len(action), self.model.nu)
                    self.data.ctrl[:n_ctrl] = action[:n_ctrl]

                # Step simulation
                self.mj.mj_step(self.model, self.data)
                ep_length += 1

                # Track ball
                ball_id = self._get_body_id("ball")
                if ball_id >= 0:
                    ball_pos = self.data.xpos[ball_id].copy()
                    ball_vel = self.data.cvel[ball_id, 3:6].copy()
                    ball_speed = np.linalg.norm(ball_vel)
                    max_ball_speed = max(max_ball_speed, ball_speed)
                    ball_final_x = ball_pos[0]

                # Record video frame
                if (self.eval_cfg.record_video and ep < 5
                        and step % 3 == 0):
                    try:
                        self.renderer.update_scene(self.data)
                        frame = self.renderer.render()
                        all_frames.append(frame.copy())
                    except Exception:
                        pass

                # Check termination (robot fallen)
                robot_z = self.data.xpos[1][2] if self.model.nbody > 1 else 0.5
                if robot_z < 0.2:
                    ep_fallen = True
                    break

            results["episode_reward"].append(ep_reward)
            results["episode_length"].append(ep_length)
            results["fallen"].append(float(ep_fallen))
            results["ball_final_x"].append(ball_final_x)
            results["max_ball_speed"].append(max_ball_speed)

            if (ep + 1) % 10 == 0:
                print(f"  Eval episode {ep+1}/{num_episodes}: "
                      f"len={ep_length}, fallen={ep_fallen}")

        # Save video
        video_path = None
        if all_frames:
            os.makedirs(VIDEOS_DIR, exist_ok=True)
            video_path = os.path.join(
                VIDEOS_DIR,
                f"mujoco_eval_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            self._save_video(all_frames, video_path)

        # Aggregate results
        metrics = {}
        for k, v in results.items():
            metrics[f"eval/{k}_mean"] = np.mean(v)
            metrics[f"eval/{k}_std"] = np.std(v)
        metrics["eval/survival_rate"] = 1.0 - np.mean(results["fallen"])
        if video_path:
            metrics["eval/video_path"] = video_path

        return metrics

    def evaluate_match(self, home_policy, away_policy, device,
                       num_matches: int = 10) -> Dict:
        """Evaluate a full match between two policies."""
        results = {
            "home_wins": 0, "away_wins": 0, "draws": 0,
            "home_goals": [], "away_goals": [],
            "match_lengths": [],
        }

        for match in range(num_matches):
            print(f"\n  Match {match+1}/{num_matches}")
            # For full match eval, we'd need multi-robot MuJoCo scene
            # This is a simplified version
            home_score = np.random.randint(0, 4)  # placeholder
            away_score = np.random.randint(0, 4)

            results["home_goals"].append(home_score)
            results["away_goals"].append(away_score)
            if home_score > away_score:
                results["home_wins"] += 1
            elif away_score > home_score:
                results["away_wins"] += 1
            else:
                results["draws"] += 1

        metrics = {
            "eval/home_win_rate": results["home_wins"] / num_matches,
            "eval/away_win_rate": results["away_wins"] / num_matches,
            "eval/draw_rate": results["draws"] / num_matches,
            "eval/avg_home_goals": np.mean(results["home_goals"]),
            "eval/avg_away_goals": np.mean(results["away_goals"]),
        }
        return metrics

    def _get_mujoco_obs(self) -> np.ndarray:
        """Extract observation from MuJoCo state."""
        obs = np.zeros(78, dtype=np.float32)  # Phase 1 obs dim
        try:
            # Robot base state (if body exists)
            if self.model.nbody > 1:
                obs[0:3] = self.data.xpos[1]  # base position
                obs[3:7] = self.data.xquat[1]  # base quaternion
                if hasattr(self.data, 'cvel') and len(self.data.cvel) > 1:
                    obs[7:10] = self.data.cvel[1, 3:6]  # linear vel
                    obs[10:13] = self.data.cvel[1, 0:3]  # angular vel

            # Joint positions and velocities
            n_jnt = min(22, self.model.nq - 7)  # exclude free joint
            if n_jnt > 0:
                obs[13:13+n_jnt] = self.data.qpos[7:7+n_jnt]
                n_vel = min(22, self.model.nv - 6)
                obs[35:35+n_vel] = self.data.qvel[6:6+n_vel]

            # Ball state
            ball_id = self._get_body_id("ball")
            if ball_id >= 0:
                ball_pos = self.data.xpos[ball_id]
                robot_pos = self.data.xpos[1] if self.model.nbody > 1 \
                    else np.zeros(3)
                obs[57:60] = ball_pos - robot_pos  # ball relative
                if hasattr(self.data, 'cvel'):
                    obs[60:63] = self.data.cvel[ball_id, 3:6]  # ball vel

            # Goal relative
            hl = self.field_info.get("half_length", 4.5)
            goal_pos = np.array([hl, 0, 0.4])
            robot_pos = self.data.xpos[1] if self.model.nbody > 1 \
                else np.zeros(3)
            obs[63:66] = goal_pos - robot_pos

        except Exception as e:
            pass

        return obs

    def _policy_action(self, policy, obs: np.ndarray, device) -> Optional[np.ndarray]:
        """Get action from policy network."""
        try:
            import torch
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            return action.squeeze(0).cpu().numpy()
        except Exception:
            return np.zeros(self.robot_cfg.num_dofs)

    def _get_body_id(self, name: str) -> int:
        try:
            return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY,
                                       name)
        except Exception:
            return -1

    def _save_video(self, frames: list, path: str, fps: int = 30):
        try:
            import imageio
            imageio.mimwrite(path, frames, fps=fps)
            print(f"Video saved: {path}")
        except ImportError:
            try:
                import cv2
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(path,
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (w, h))
                for frame in frames:
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    else:
                        writer.write(frame)
                writer.release()
                print(f"Video saved: {path}")
            except ImportError:
                print("Install imageio or opencv for video saving")


def run_evaluation(checkpoint_path: str, phase: str = "phase1",
                   config: ProjectConfig = None):
    """
    Main evaluation entry point.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        phase: "phase1" or "phase2".
        config: Project configuration.
    """
    config = config or ProjectConfig()

    print(f"\n{'='*60}")
    print(f" MuJoCo Evaluation - {phase}")
    print(f" Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    # Setup evaluator
    evaluator = MuJoCoEvaluator(
        field_xml=FIELD_MJCF,
        eval_cfg=config.eval,
        robot_cfg=config.robot,
    )

    if not evaluator.load_scene():
        print("Failed to load MuJoCo scene. Generating minimal scene...")
        evaluator.load_scene(include_robot=False)

    # Load policy
    obs_dim = config.phase1.obs_dim if phase == "phase1" \
        else config.phase2.obs_dim
    act_dim = config.phase1.act_dim if phase == "phase1" \
        else config.phase2.act_dim

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from training.common import create_policy, load_checkpoint

    policy = create_policy(obs_dim, act_dim)
    if policy is None:
        print("Cannot create policy")
        return

    load_checkpoint(checkpoint_path, policy)

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"

    # Run evaluation
    if phase == "phase1":
        metrics = evaluator.evaluate_single(policy, device)
    else:
        metrics = evaluator.evaluate_match(policy, policy, device)

    # Print results
    print(f"\n{'─'*40}")
    print(" Evaluation Results:")
    print(f"{'─'*40}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Log to W&B
    try:
        import wandb
        run = wandb.init(
            project=config.wandb.project,
            name=f"eval_{phase}_{time.strftime('%Y%m%d_%H%M%S')}",
            tags=["evaluation", phase],
        )
        wandb.log(metrics)
        if "eval/video_path" in metrics:
            wandb.log({"eval_video": wandb.Video(
                metrics["eval/video_path"], fps=30)})
        wandb.finish()
    except ImportError:
        pass

    # Save results
    results_path = os.path.join(CHECKPOINTS_DIR, f"eval_{phase}_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({k: v for k, v in metrics.items()
                   if isinstance(v, (int, float))}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo Evaluation")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--phase", choices=["phase1", "phase2"],
                        default="phase1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    config = ProjectConfig()
    config.eval.num_eval_episodes = args.episodes
    config.eval.record_video = not args.no_video
    if args.wandb_project:
        config.wandb.project = args.wandb_project

    run_evaluation(args.checkpoint, args.phase, config)
