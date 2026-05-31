"""Sim2sim MuJoCo evaluation for the standup skill (clean rewrite).

Loads the manufacturer K1 MJCF and runs a trained policy through the SAME
control contract as Genesis training, then reports standup success.

Key design choice vs the earlier version: control is done with MuJoCo's
NATIVE position-servo actuators (implicit PD), set up programmatically from
the per-joint gains in K1RobotConfig — this is the standard, stable MuJoCo
deploy pattern (`ctrl = target_angle`, MuJoCo computes kp·(target−q) − kd·q̇
each substep), rather than an explicit hand-rolled torque loop which can be
marginally unstable on a violent, contact-rich maneuver.

Control contract (matches `SkillEnv`):
  * 50 Hz policy (dt=0.02), 500 Hz physics (sim_dt=0.002), action_repeat=10.
  * Residual action: target = default_pose + clip(action, ±0.5), clip ±π.
  * Per-joint T1 gains from K1RobotConfig (kp/kd groups).
  * Obs = the identical 78-dim compute_common_obs (+ 8 contact + 8 DR-nominal
    for a privileged 94-dim teacher), normalized with the checkpoint obs_norm.

Usage:
    python -m evaluation.eval_standup_mujoco CKPT.pt --episodes 20 [--record-video]
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

DT = 0.02
SIM_DT = 0.002
ACTION_REPEAT = int(round(DT / SIM_DT))     # 10
ACTION_DELTA_MAX = 0.5
TARGET_HEIGHT = 0.55
UPRIGHT_THRESHOLD = 0.92
HEIGHT_BAR = TARGET_HEIGHT - 0.10           # 0.45
SUCCESS_HOLD_STEPS = 50                      # 1.0 s
MAX_EPISODE_STEPS = 250                      # 5 s
SETTLE_STEPS = 300                           # 0.6 s — matches Genesis settle pool

_FOOT_BODIES = ("left_foot_link", "right_foot_link")
_HAND_BODIES = ("left_hand_link", "right_hand_link")
_CONTACT_Z = 0.05


def _per_joint_gains(cfg):
    """Replicate SkillEnv._build_per_joint_gains (T1 buckets)."""
    n = cfg.num_dofs
    kp = np.full(n, cfg.kp, np.float64)
    kd = np.full(n, cfg.kd, np.float64)
    for i, name in enumerate(cfg.joint_names):
        lo = name.lower()
        if "head" in lo:
            kp[i], kd[i] = cfg.kp_head, cfg.kd_head
        elif "shoulder" in lo or "elbow" in lo or "arm" in lo:
            kp[i], kd[i] = cfg.kp_arm, cfg.kd_arm
        elif "knee" in lo:
            kp[i], kd[i] = cfg.kp_knee, cfg.kd_knee
        elif "ankle" in lo:
            kp[i], kd[i] = cfg.kp_ankle, cfg.kd_ankle
        elif "hip" in lo:
            kp[i], kd[i] = cfg.kp_hip, cfg.kd_hip
    return kp, kd


def _upright(quat):
    return float(-projected_gravity(quat[None, :])[0, 2])


def _rand_quat(rng):
    u = rng.uniform(0, 1, 3)
    s1, s0 = np.sqrt(1 - u[0]), np.sqrt(u[0])
    return np.array([s1*np.sin(2*np.pi*u[1]), s1*np.cos(2*np.pi*u[1]),
                     s0*np.sin(2*np.pi*u[2]), s0*np.cos(2*np.pi*u[2])], np.float64)


class StandupMujocoEval:
    def __init__(self, checkpoint, device="cpu", record_video=False,
                 clamp_torque=False, match_genesis_dynamics=False):
        self.cfg = K1RobotConfig()
        self.device = torch.device(device)
        self.record_video = record_video

        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(_XML))
        self.model.opt.timestep = SIM_DT

        # ── match Genesis joint dynamics (sim2sim CRITICAL) ──
        # The K1 URDF has NO <dynamics>, so Genesis trains with zero joint
        # armature/damping/frictionloss (pure PD). The MJCF bakes in armature
        # (legs ~0.05-0.10 = significant rotor inertia), which makes MuJoCo
        # joints heavier/sluggish than training and is the prime transfer
        # killer (robot stays flat). Zero them so eval dynamics == train.
        if match_genesis_dynamics:
            self.model.dof_armature[:] = 0.0
            self.model.dof_damping[:] = 0.0
            self.model.dof_frictionloss[:] = 0.0
            print("[eval] joint armature/damping/frictionloss ZEROED "
                  "to match Genesis (URDF has no <dynamics>)")
        self.data = mujoco.MjData(self.model)

        # name → indices (config joint order)
        self.qadr, self.dadr, self.aid = [], [], []
        for nm in self.cfg.joint_names:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            if j < 0 or a < 0:
                raise RuntimeError(f"joint/actuator '{nm}' missing in MJCF")
            self.qadr.append(self.model.jnt_qposadr[j])
            self.dadr.append(self.model.jnt_dofadr[j])
            self.aid.append(a)
        self.qadr = np.array(self.qadr); self.dadr = np.array(self.dadr)
        self.aid = np.array(self.aid)
        self.base_q = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")]
        self.base_d = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")]
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
        self.default = np.asarray(self.cfg.default_joint_pos, np.float64)

        # ── configure MuJoCo native position-servo PD on each actuator ──
        kp, kd = _per_joint_gains(self.cfg)
        for k, a in enumerate(self.aid):
            self.model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gainprm[a] = 0.0
            self.model.actuator_gainprm[a][0] = kp[k]
            self.model.actuator_biasprm[a] = 0.0
            self.model.actuator_biasprm[a][1] = -kp[k]
            self.model.actuator_biasprm[a][2] = -kd[k]
            self.model.actuator_ctrllimited[a] = 0          # target is an angle
            self.model.actuator_forcelimited[a] = 1 if clamp_torque else 0
        print(f"[eval] position-servo PD: kp {kp.min():.0f}-{kp.max():.0f}, "
              f"force clamp {'ON' if clamp_torque else 'OFF (matches Genesis)'}")

        # ── policy ──
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("policy_state_dict") or ckpt.get("actor_state_dict")
        obs_dim = sd["actor_trunk.0.weight"].shape[1]
        act_dim = sd["actor_head.weight"].shape[0]
        if obs_dim not in (78, 86, 94):
            raise SystemExit(f"unsupported obs_dim {obs_dim}")
        self.use_contact = obs_dim >= 86
        self.use_priv = obs_dim >= 94
        if self.use_contact:
            self.foot_b = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                           for n in _FOOT_BODIES]
            self.hand_b = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                           for n in _HAND_BODIES]
        self.policy = PPOActorCritic(obs_dim, act_dim).to(self.device)
        self.policy.load_state_dict(sd, strict=False)
        self.policy.eval()
        self.obs_norm = RunningMeanStd(shape=(obs_dim,))
        if "obs_norm" in ckpt:
            self.obs_norm.load_state_dict(ckpt["obs_norm"])
        else:
            print("[eval] WARNING: no obs_norm in checkpoint")
        print(f"[eval] obs_dim={obs_dim} (contact={self.use_contact}, "
              f"privileged={self.use_priv})")

        self.renderer = mujoco.Renderer(self.model, 480, 640) if record_video else None

    # ── state ──
    def _q(self):  return self.data.qpos[self.qadr].copy()
    def _qd(self): return self.data.qvel[self.dadr].copy()
    def _quat(self): return self.data.qpos[self.base_q+3:self.base_q+7].copy()
    def _z(self):  return float(self.data.qpos[self.base_q+2])

    def _vel6(self, local):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.trunk, v, local)
        return v   # [ang(3), lin(3)]

    def _contact_obs(self):
        fz = [float(self.data.xpos[b][2]) for b in self.foot_b]
        hz = [float(self.data.xpos[b][2]) for b in self.hand_b]
        zs = fz + hz
        return np.array([zs + [1.0 if z < _CONTACT_Z else 0.0 for z in zs]],
                        np.float32)

    def _obs(self, last_action, step):
        obs = compute_common_obs(
            root_pos=np.array([[0., 0., self._z()]], np.float32),
            root_quat=self._quat()[None, :].astype(np.float32),
            root_lin_vel=self._vel6(0)[3:6][None, :].astype(np.float32),   # world lin
            root_ang_vel=self._vel6(1)[0:3][None, :].astype(np.float32),   # body ang
            joint_pos=self._q()[None, :].astype(np.float32),
            joint_vel=self._qd()[None, :].astype(np.float32),
            last_action=last_action[None, :].astype(np.float32),
            step_count=np.array([step], np.int64),
            default_joint_pos=self.default.astype(np.float32),
            control_dt=DT, gait_freq_hz=1.5,
        )
        if self.use_contact:
            obs = np.concatenate([obs, self._contact_obs()], 1)
        if self.use_priv:
            obs = np.concatenate(
                [obs, self.obs_norm.mean[None, 86:94].astype(np.float32)], 1)
        return self.obs_norm.normalize(obs).astype(np.float32)

    def _reset_fallen(self, rng):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.base_q+0:self.base_q+2] = 0.0
        self.data.qpos[self.base_q+2] = rng.uniform(0.8, 1.2)
        self.data.qpos[self.base_q+3:self.base_q+7] = _rand_quat(rng)
        self.data.qpos[self.qadr] = self.default + rng.normal(0, 0.2, self.cfg.num_dofs)
        # settle holding the default pose (servo target = default)
        self.data.ctrl[self.aid] = self.default
        mujoco.mj_forward(self.model, self.data)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
        # start the policy from rest (Genesis resets pool states zero-velocity)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def run_episode(self, rng, frames=None, trace=False):
        self._reset_fallen(rng)
        last = np.zeros(self.cfg.num_dofs, np.float32)
        streak, t_first = 0, None
        for step in range(MAX_EPISODE_STEPS):
            obs = self._obs(last, step)
            with torch.no_grad():
                a, _, _ = self.policy.act(torch.as_tensor(obs, device=self.device),
                                          deterministic=True)
            delta = np.clip(a.cpu().numpy()[0], -ACTION_DELTA_MAX, ACTION_DELTA_MAX)
            target = np.clip(self.default + delta, -np.pi, np.pi)
            last = delta.astype(np.float32)
            self.data.ctrl[self.aid] = target          # position servo target
            for _ in range(ACTION_REPEAT):
                mujoco.mj_step(self.model, self.data)
            up, z = _upright(self._quat()), self._z()
            streak = streak + 1 if (up > UPRIGHT_THRESHOLD and z > HEIGHT_BAR) else 0
            if streak >= SUCCESS_HOLD_STEPS and t_first is None:
                t_first = step - (SUCCESS_HOLD_STEPS - 1)
            if trace and step % 25 == 0:
                print(f"      step {step:3d}: up={up:+.2f} z={z:.3f}")
            if frames is not None and step % 2 == 0:
                self.renderer.update_scene(self.data, camera=-1)
                frames.append(self.renderer.render().copy())
        return {"success": t_first is not None,
                "t": (t_first*DT) if t_first is not None else None,
                "up": _upright(self._quat()), "z": self._z()}

    def evaluate(self, episodes, video_path=None):
        rng = np.random.default_rng(0)
        frames = [] if self.record_video else None
        res = []
        for ep in range(episodes):
            fr = frames if (self.record_video and ep < 4) else None
            r = self.run_episode(rng, frames=fr, trace=(ep == 0))
            res.append(r)
            t = f"{r['t']:.2f}s" if r["success"] else "—"
            print(f"  ep {ep+1:3d} success={str(r['success']):5s} t={t:6s} "
                  f"up={r['up']:+.2f} z={r['z']:.3f}")
        succ = [r for r in res if r["success"]]
        print(f"\n{'='*52}\n STANDUP MUJOCO EVAL — {len(res)} episodes")
        print(f"   success rate     : {100*len(succ)/len(res):.1f}%  "
              f"({len(succ)}/{len(res)})")
        if succ:
            t = np.array([r["t"] for r in succ])
            print(f"   time-to-stand    : {t.mean():.2f}s [{t.min():.2f}-{t.max():.2f}]")
        print(f"   mean final up    : {np.mean([r['up'] for r in res]):.3f}")
        print(f"   mean final z     : {np.mean([r['z'] for r in res]):.3f}")
        print('='*52)
        if frames:
            import imageio
            os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
            imageio.mimsave(video_path, frames, fps=25)
            print(f"   video → {video_path}")
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--video-path", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clamp-torque", action="store_true",
                    help="clamp to MJCF motor force limits (default off: match "
                         "Genesis's unclamped PD)")
    ap.add_argument("--zero-armature", action="store_true",
                    help="diagnostic: zero the MJCF joint armature (unstable; "
                         "default keeps the real armature, which training now "
                         "matches via set_dofs_armature)")
    a = ap.parse_args()
    ev = StandupMujocoEval(a.checkpoint, device=a.device,
                           record_video=a.record_video, clamp_torque=a.clamp_torque,
                           match_genesis_dynamics=a.zero_armature)
    vp = a.video_path or os.path.join(
        "videos", f"standup_eval_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    ev.evaluate(a.episodes, video_path=vp)


if __name__ == "__main__":
    main()
