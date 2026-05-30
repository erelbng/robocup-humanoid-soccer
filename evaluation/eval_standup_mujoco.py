"""Sim2sim MuJoCo evaluation for the standup skill.

The skill-library refactor left `evaluation/evaluate.py` wired to the old
monolithic phase1/phase2 pipeline (soccer scene, wrong obs layout) — it
cannot evaluate a single skill. This script is the standup-specific
counterpart: it loads a trained policy (normally the proprio-only STUDENT,
obs_dim 78), drops the K1 into a fallen pose in MuJoCo, runs the policy
through the SAME control stack as training (residual-delta action → per-joint
PD torque at 50 Hz / 500 Hz physics), and measures whether it stands up.

Why this is a faithful sim2sim check:
  * MuJoCo physics (not Genesis) — validates transfer across simulators.
  * The K1 MJCF uses torque `<motor>` actuators, so we replicate the
    T1-style per-joint PD controller (kp_hip/knee=200, ankle=50, arm=50,
    head=20) exactly as `SkillEnv` does, clipped to the motor force ranges.
  * Obs is the identical 78-dim `compute_common_obs`, normalized with the
    checkpoint's saved `obs_norm` (the policy was trained on normalized obs).

Usage:
    python -m evaluation.eval_standup_mujoco \
        checkpoints/skill_standup/student_standup_step50000000.pt \
        --episodes 20 --record-video

    # works on a teacher/single checkpoint too IF it is proprio-only
    # (obs_dim 78). A privileged teacher (obs_dim 94) can't be run in pure
    # proprio MuJoCo and is rejected with a clear message.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

try:
    import mujoco
except ImportError:
    raise SystemExit("MuJoCo required: pip install mujoco")

import torch

from configs.config import K1RobotConfig
from skills.common_obs import (SKILL_BASE_OBS_DIM, compute_common_obs,
                               projected_gravity)
from training.algorithms.networks import PPOActorCritic
from training.normalizers import RunningMeanStd

_XML = os.path.join(os.path.dirname(__file__), "..", "models", "robot", "K1",
                    "K1_22dof.xml")

# Control / success constants — match StandupConfig defaults.
DT = 0.02              # policy control dt (50 Hz)
SIM_DT = 0.002         # physics dt (500 Hz)
ACTION_REPEAT = int(round(DT / SIM_DT))   # 10
ACTION_DELTA_MAX = 0.5
TARGET_HEIGHT = 0.55
UPRIGHT_THRESHOLD = 0.92
SUCCESS_HOLD_STEPS = 50         # 1.0 s sustained
MAX_EPISODE_STEPS = 250         # 5 s
SETTLE_STEPS = 60               # physics steps to settle the fallen pose


def _per_joint_gains(cfg: K1RobotConfig):
    """Replicate SkillEnv._build_per_joint_gains (T1-style buckets)."""
    n = cfg.num_dofs
    kp = np.full(n, cfg.kp, dtype=np.float64)
    kd = np.full(n, cfg.kd, dtype=np.float64)
    for i, name in enumerate(cfg.joint_names):
        low = name.lower()
        if "head" in low:
            kp[i], kd[i] = cfg.kp_head, cfg.kd_head
        elif "shoulder" in low or "elbow" in low or "arm" in low:
            kp[i], kd[i] = cfg.kp_arm, cfg.kd_arm
        elif "knee" in low:
            kp[i], kd[i] = cfg.kp_knee, cfg.kd_knee
        elif "ankle" in low:
            kp[i], kd[i] = cfg.kp_ankle, cfg.kd_ankle
        elif "hip" in low:
            kp[i], kd[i] = cfg.kp_hip, cfg.kd_hip
    return kp, kd


def _upright(quat: np.ndarray) -> float:
    """cos(trunk-z, world-z): 1 upright, 0 sideways, -1 inverted."""
    g = projected_gravity(quat[None, :])   # (1,3) body-frame gravity dir
    return float(-g[0, 2])


# Canonical fallen orientations (w, x, y, z): supine (on back), prone (on
# face), and two side-lies — a representative spread of fallen starts.
_FALLEN_QUATS = {
    "supine": np.array([0.7071, 0.7071, 0.0, 0.0], np.float64),   # +90° about x
    "prone":  np.array([0.7071, -0.7071, 0.0, 0.0], np.float64),
    "left":   np.array([0.7071, 0.0, 0.7071, 0.0], np.float64),   # +90° about y
    "right":  np.array([0.7071, 0.0, -0.7071, 0.0], np.float64),
}


class StandupMujocoEval:
    def __init__(self, checkpoint: str, device: str = "cpu",
                 record_video: bool = False):
        self.cfg = K1RobotConfig()
        self.device = torch.device(device)
        self.record_video = record_video

        # ── model ──
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(_XML))
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        # joint / actuator / dof address maps in cfg.joint_names order
        self.qpos_adr, self.dof_adr, self.act_id = [], [], []
        for name in self.cfg.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jid < 0 or aid < 0:
                raise RuntimeError(f"joint/actuator '{name}' not in MJCF")
            self.qpos_adr.append(self.model.jnt_qposadr[jid])
            self.dof_adr.append(self.model.jnt_dofadr[jid])
            self.act_id.append(aid)
        self.qpos_adr = np.array(self.qpos_adr)
        self.dof_adr = np.array(self.dof_adr)
        self.act_id = np.array(self.act_id)
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          "Trunk")
        # free-joint base addresses
        self.base_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                              "world_joint")]

        self.kp, self.kd = _per_joint_gains(self.cfg)
        self.force_lo = self.model.actuator_forcerange[self.act_id, 0].copy()
        self.force_hi = self.model.actuator_forcerange[self.act_id, 1].copy()
        self.default = np.asarray(self.cfg.default_joint_pos, np.float64)

        # ── policy ──
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("policy_state_dict") or ckpt.get("actor_state_dict")
        if sd is None:
            raise RuntimeError("checkpoint has no policy/actor_state_dict")
        obs_dim = sd["actor_trunk.0.weight"].shape[1]
        act_dim = sd["actor_head.weight"].shape[0]
        if obs_dim != SKILL_BASE_OBS_DIM:
            raise SystemExit(
                f"checkpoint obs_dim={obs_dim} but this proprio MuJoCo eval "
                f"only supports {SKILL_BASE_OBS_DIM} (the deployable student / "
                "proprio_only policy). A privileged teacher (obs_dim 94) "
                "needs contact+DR obs unavailable in pure proprio.")
        # n_critics is irrelevant for eval (actor only); load non-strict.
        self.policy = PPOActorCritic(obs_dim, act_dim).to(self.device)
        self.policy.load_state_dict(sd, strict=False)
        self.policy.eval()

        self.obs_norm = RunningMeanStd(shape=(obs_dim,))
        if "obs_norm" in ckpt:
            self.obs_norm.load_state_dict(ckpt["obs_norm"])
        else:
            print("[eval] WARNING: no obs_norm in checkpoint — using identity "
                  "normalization (results may be wrong if the policy trained "
                  "with obs normalization).")

        self.renderer = (mujoco.Renderer(self.model, 480, 640)
                         if record_video else None)

    # ── state helpers ──
    def _q(self):
        return self.data.qpos[self.qpos_adr].copy()

    def _qd(self):
        return self.data.qvel[self.dof_adr].copy()

    def _root_quat(self):
        return self.data.qpos[self.base_qadr + 3: self.base_qadr + 7].copy()

    def _root_z(self):
        return float(self.data.qpos[self.base_qadr + 2])

    def _root_lin_vel_world(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.trunk_id, v, 0)   # 0 = world frame
        return v[3:6].copy()    # [rot(3), lin(3)] → translational part

    def _root_ang_vel_body(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.trunk_id, v, 1)   # 1 = local/body frame
        return v[0:3].copy()    # rotational part, body frame

    def _build_obs(self, last_action: np.ndarray, step_count: int):
        obs = compute_common_obs(
            root_pos=np.array([[0.0, 0.0, self._root_z()]], np.float32),
            root_quat=self._root_quat()[None, :].astype(np.float32),
            root_lin_vel=self._root_lin_vel_world()[None, :].astype(np.float32),
            root_ang_vel=self._root_ang_vel_body()[None, :].astype(np.float32),
            joint_pos=self._q()[None, :].astype(np.float32),
            joint_vel=self._qd()[None, :].astype(np.float32),
            last_action=last_action[None, :].astype(np.float32),
            step_count=np.array([step_count], np.int64),
            default_joint_pos=self.default.astype(np.float32),
            control_dt=DT, gait_freq_hz=self.cfg_gait_freq,
        )
        return self.obs_norm.normalize(obs).astype(np.float32)

    cfg_gait_freq = 1.5

    def _apply_pd(self, targets: np.ndarray):
        q, qd = self._q(), self._qd()
        tau = self.kp * (targets - q) - self.kd * qd
        tau = np.clip(tau, self.force_lo, self.force_hi)
        self.data.ctrl[self.act_id] = tau

    def _reset_fallen(self, pose: str, rng: np.random.Generator):
        mujoco.mj_resetData(self.model, self.data)
        # base: low height, fallen orientation
        self.data.qpos[self.base_qadr + 0] = 0.0
        self.data.qpos[self.base_qadr + 1] = 0.0
        self.data.qpos[self.base_qadr + 2] = 0.30
        self.data.qpos[self.base_qadr + 3: self.base_qadr + 7] = _FALLEN_QUATS[pose]
        # joints: default + small jitter
        jitter = rng.normal(0, 0.05, size=self.cfg.num_dofs)
        self.data.qpos[self.qpos_adr] = self.default + jitter
        mujoco.mj_forward(self.model, self.data)
        # settle: hold default pose via PD, let it fall/relax to the ground
        for _ in range(SETTLE_STEPS):
            self._apply_pd(self.default)
            mujoco.mj_step(self.model, self.data)

    # ── episode ──
    def run_episode(self, pose: str, rng, frames=None):
        self._reset_fallen(pose, rng)
        last_action = np.zeros(self.cfg.num_dofs, np.float32)
        streak, t_first = 0, None
        for step in range(MAX_EPISODE_STEPS):
            obs = self._build_obs(last_action, step)
            with torch.no_grad():
                a, _, _ = self.policy.act(
                    torch.as_tensor(obs, device=self.device),
                    deterministic=True)
            delta = np.clip(a.cpu().numpy()[0], -ACTION_DELTA_MAX, ACTION_DELTA_MAX)
            targets = np.clip(self.default + delta, -np.pi, np.pi)
            last_action = delta.astype(np.float32)
            for _ in range(ACTION_REPEAT):
                self._apply_pd(targets)
                mujoco.mj_step(self.model, self.data)

            up = _upright(self._root_quat())
            z = self._root_z()
            success_frame = (up > UPRIGHT_THRESHOLD) and (z > TARGET_HEIGHT - 0.10)
            streak = streak + 1 if success_frame else 0
            if streak >= SUCCESS_HOLD_STEPS and t_first is None:
                t_first = step - (SUCCESS_HOLD_STEPS - 1)

            if frames is not None and step % 2 == 0:    # ~25 fps
                self.renderer.update_scene(self.data, camera=-1)
                frames.append(self.renderer.render().copy())

        return {
            "pose": pose,
            "success": t_first is not None,
            "time_to_stand_s": (t_first * DT) if t_first is not None else None,
            "final_up": _upright(self._root_quat()),
            "final_z": self._root_z(),
        }

    def evaluate(self, episodes: int, video_path: str = None):
        rng = np.random.default_rng(0)
        poses = list(_FALLEN_QUATS)
        frames = [] if self.record_video else None
        results = []
        for ep in range(episodes):
            pose = poses[ep % len(poses)]
            # only film the first episode of each pose, up to 4 clips
            ep_frames = frames if (self.record_video and ep < len(poses)) else None
            r = self.run_episode(pose, rng, frames=ep_frames)
            results.append(r)
            tts = f"{r['time_to_stand_s']:.2f}s" if r["success"] else "—"
            print(f"  ep {ep+1:3d} [{pose:6s}] success={r['success']!s:5s} "
                  f"t_stand={tts:6s} final_up={r['final_up']:.2f} "
                  f"final_z={r['final_z']:.3f}")

        succ = [r for r in results if r["success"]]
        rate = len(succ) / len(results)
        print(f"\n{'='*54}\n STANDUP SIM2SIM (MuJoCo) — {len(results)} episodes")
        print(f"   success rate     : {rate*100:.1f}%  ({len(succ)}/{len(results)})")
        if succ:
            tts = np.array([r["time_to_stand_s"] for r in succ])
            print(f"   time-to-stand    : {tts.mean():.2f}s "
                  f"(min {tts.min():.2f}, max {tts.max():.2f})")
        print(f"   mean final upright: {np.mean([r['final_up'] for r in results]):.3f}")
        print(f"   mean final height : {np.mean([r['final_z'] for r in results]):.3f}")
        # per-pose breakdown
        for p in poses:
            pr = [r for r in results if r["pose"] == p]
            if pr:
                print(f"   {p:6s}: {sum(r['success'] for r in pr)}/{len(pr)}")
        print('='*54)

        if frames:
            import imageio
            os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
            imageio.mimsave(video_path, frames, fps=25)
            print(f"   video → {video_path}")
        return results


def main():
    ap = argparse.ArgumentParser(description="Standup sim2sim MuJoCo eval")
    ap.add_argument("checkpoint", help="student/proprio policy .pt (obs_dim 78)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--video-path", type=str, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ev = StandupMujocoEval(args.checkpoint, device=args.device,
                           record_video=args.record_video)
    vp = args.video_path or os.path.join(
        "videos", f"standup_eval_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    ev.evaluate(args.episodes, video_path=vp)


if __name__ == "__main__":
    main()
