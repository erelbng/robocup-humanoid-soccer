"""MuJoCo eval + video for the WALK skill.

Adapted from eval_standup_mujoco.py (same MuJoCo model, per-joint position-servo
PD, obs_norm-from-checkpoint, sim2sim dynamics handling). Differences:
  * spawns UPRIGHT at the default pose (not fallen),
  * builds the 85-dim walk obs = compute_common_obs(78) + walk_cmd(5) + head(2),
  * drives a constant forward command (vx) so we can SEE whether it walks,
  * logs quantitative traces — forward displacement, height, uprightness, and
    per-foot ground contact — which reveal HOW it moves (real step vs slide vs
    lurch) even from still frames.

    python -m evaluation.eval_walk_mujoco CKPT.pt --vx 0.4 --record-video
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

try:
    import mujoco
except Exception:
    raise SystemExit("MuJoCo required: pip install mujoco")

from configs.config import K1RobotConfig
from skills.common_obs import compute_common_obs, projected_gravity
from training.algorithms.networks import PPOActorCritic
from training.normalizers import RunningMeanStd

_XML = os.path.join(os.path.dirname(__file__), "..", "models", "robot", "K1",
                    "K1_22dof.xml")
DT, SIM_DT = 0.02, 0.002
ACTION_REPEAT = int(round(DT / SIM_DT))
ACTION_DELTA_MAX = 0.5
_FOOT_BODIES = ("left_foot_link", "right_foot_link")


def _per_joint_gains(cfg):
    n = cfg.num_dofs
    kp = np.full(n, cfg.kp, np.float64); kd = np.full(n, cfg.kd, np.float64)
    for i, name in enumerate(cfg.joint_names):
        lo = name.lower()
        if "head" in lo: kp[i], kd[i] = cfg.kp_head, cfg.kd_head
        elif "shoulder" in lo or "elbow" in lo or "arm" in lo: kp[i], kd[i] = cfg.kp_arm, cfg.kd_arm
        elif "knee" in lo: kp[i], kd[i] = cfg.kp_knee, cfg.kd_knee
        elif "ankle" in lo: kp[i], kd[i] = cfg.kp_ankle, cfg.kd_ankle
        elif "hip" in lo: kp[i], kd[i] = cfg.kp_hip, cfg.kd_hip
    return kp, kd


def _upright(quat):
    return float(-projected_gravity(quat[None, :])[0, 2])


class WalkMujocoEval:
    def __init__(self, checkpoint, vx=0.4, vy=0.0, vyaw=0.0, foot_clear=0.08,
                 step_freq=1.5, device="cpu", record_video=False):
        self.cfg = K1RobotConfig(); self.device = torch.device(device)
        self.record_video = record_video
        self.command = np.array([[vx, vy, vyaw, foot_clear, step_freq]], np.float32)
        self.head_command = np.array([[0.0, 0.0]], np.float32)

        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(_XML))
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        self.qadr, self.dadr, self.aid = [], [], []
        for nm in self.cfg.joint_names:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            self.qadr.append(self.model.jnt_qposadr[j]); self.dadr.append(self.model.jnt_dofadr[j]); self.aid.append(a)
        self.qadr = np.array(self.qadr); self.dadr = np.array(self.dadr); self.aid = np.array(self.aid)
        self.base_q = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")]
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
        self.default = np.asarray(self.cfg.default_joint_pos, np.float64)
        self.foot_b = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in _FOOT_BODIES]

        kp, kd = _per_joint_gains(self.cfg)
        for k, a in enumerate(self.aid):
            self.model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gainprm[a] = 0.0; self.model.actuator_gainprm[a][0] = kp[k]
            self.model.actuator_biasprm[a] = 0.0
            self.model.actuator_biasprm[a][1] = -kp[k]; self.model.actuator_biasprm[a][2] = -kd[k]
            self.model.actuator_ctrllimited[a] = 0; self.model.actuator_forcelimited[a] = 0

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("policy_state_dict") or ckpt.get("actor_state_dict")
        obs_dim = sd["actor_trunk.0.weight"].shape[1]
        act_dim = sd["actor_head.weight"].shape[0]
        self.obs_dim = obs_dim
        self.policy = PPOActorCritic(obs_dim, act_dim).to(self.device)
        self.policy.load_state_dict(sd, strict=False); self.policy.eval()
        self.obs_norm = RunningMeanStd(shape=(obs_dim,))
        if "obs_norm" in ckpt:
            self.obs_norm.load_state_dict(ckpt["obs_norm"])
        else:
            print("[eval] WARNING: no obs_norm")
        print(f"[eval] obs_dim={obs_dim} (walk expects 85)  cmd={self.command.tolist()}")
        self.renderer = mujoco.Renderer(self.model, 480, 640) if record_video else None

    def _q(self): return self.data.qpos[self.qadr].copy()
    def _qd(self): return self.data.qvel[self.dadr].copy()
    def _quat(self): return self.data.qpos[self.base_q+3:self.base_q+7].copy()
    def _z(self): return float(self.data.qpos[self.base_q+2])
    def _x(self): return float(self.data.qpos[self.base_q+0])

    def _vel6(self, local):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.trunk, v, local)
        return v

    def _obs(self, last_action, step):
        obs = compute_common_obs(
            root_pos=np.array([[0., 0., self._z()]], np.float32),
            root_quat=self._quat()[None, :].astype(np.float32),
            root_lin_vel=self._vel6(0)[3:6][None, :].astype(np.float32),
            root_ang_vel=self._vel6(1)[0:3][None, :].astype(np.float32),
            joint_pos=self._q()[None, :].astype(np.float32),
            joint_vel=self._qd()[None, :].astype(np.float32),
            last_action=last_action[None, :].astype(np.float32),
            step_count=np.array([step], np.int64),
            default_joint_pos=self.default.astype(np.float32),
            control_dt=DT, gait_freq_hz=1.5,
        )
        parts = [obs, self.command]
        if self.obs_dim >= 85:
            parts.append(self.head_command)
        obs = np.concatenate(parts, 1)[:, :self.obs_dim]
        return self.obs_norm.normalize(obs).astype(np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.base_q+0:self.base_q+2] = 0.0
        # Spawn slightly ABOVE the natural standing height with the default
        # (bent-knee) pose, then settle so the feet land on the ground — MuJoCo
        # K1 stands at base z≈0.6; spawning at 0.5 buried the feet → explosion.
        self.data.qpos[self.base_q+2] = 0.75
        self.data.qpos[self.base_q+3:self.base_q+7] = np.array([1., 0., 0., 0.])
        self.data.qpos[self.qadr] = self.default
        self.data.ctrl[self.aid] = self.default
        mujoco.mj_forward(self.model, self.data)
        for _ in range(400):                            # settle onto feet
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.qpos[self.base_q+0:self.base_q+2] = 0.0   # re-zero xy after settle
        mujoco.mj_forward(self.model, self.data)

    def run(self, steps=500, video_path=None, strip_path=None):
        self.reset()
        last = np.zeros(self.cfg.num_dofs, np.float32)
        x0 = self._x()
        frames, strip = [], []
        trace = []
        for step in range(steps):
            obs = self._obs(last, step)
            with torch.no_grad():
                a, _, _ = self.policy.act(torch.as_tensor(obs, device=self.device), deterministic=True)
            delta = np.clip(a.cpu().numpy()[0], -ACTION_DELTA_MAX, ACTION_DELTA_MAX)
            self.data.ctrl[self.aid] = np.clip(self.default + delta, -np.pi, np.pi)
            last = delta.astype(np.float32)
            for _ in range(ACTION_REPEAT):
                mujoco.mj_step(self.model, self.data)
            fz = [float(self.data.xpos[b][2]) for b in self.foot_b]
            vx_act = self._vel6(1)[3]            # body-frame forward velocity
            trace.append((step*DT, self._x()-x0, self._z(), _upright(self._quat()),
                          vx_act, fz[0], fz[1]))
            if self.renderer is not None and step % 2 == 0:
                self.renderer.update_scene(self.data, camera=-1)
                fr = self.renderer.render().copy(); frames.append(fr)
                if step % (max(1, steps//8)) == 0 and len(strip) < 8:
                    strip.append(fr)
            if self._z() < 0.25:
                print(f"[eval] FELL at t={step*DT:.2f}s"); break

        tr = np.array(trace)
        # ── quantitative diagnosis ──
        dur = tr[-1,0]; dx = tr[-1,1]
        print(f"\n{'='*56}\n WALK MUJOCO EVAL  (cmd vx={self.command[0,0]})")
        print(f"  duration            : {dur:.2f}s ({len(tr)} steps)")
        print(f"  forward displacement: {dx:+.3f} m  -> avg speed {dx/max(dur,1e-6):+.3f} m/s")
        print(f"  mean body vx        : {tr[:,4].mean():+.3f} m/s (cmd {self.command[0,0]})")
        print(f"  height z            : mean {tr[:,2].mean():.3f}  min {tr[:,2].min():.3f}  (stand ~0.5)")
        print(f"  uprightness         : mean {tr[:,3].mean():.3f}  min {tr[:,3].min():.3f}  (1=vertical)")
        # foot lift: fraction of time each foot is clearly off the ground (>0.10 link z)
        lift_l = float((tr[:,5] > 0.10).mean()); lift_r = float((tr[:,6] > 0.10).mean())
        both_down = float(((tr[:,5] < 0.09) & (tr[:,6] < 0.09)).mean())
        print(f"  foot airborne frac  : L {lift_l:.2f}  R {lift_r:.2f}  | both-down frac {both_down:.2f}")
        print(f"  >> diagnosis: ", end="")
        if dx < 0.15: print("barely translates (≈in place)")
        elif tr[:,3].min() < 0.7: print("translates but TIPS/lurches (low uprightness)")
        elif both_down > 0.8: print("translates with BOTH FEET DOWN → SLIDING/skating, not stepping")
        elif (lift_l+lift_r) < 0.3: print("translates with little foot lift → shuffle/glide")
        else: print("steps + translates (looks like real walking)")
        print('='*56)

        if frames and video_path:
            import imageio
            os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
            imageio.mimsave(video_path, frames, fps=25)
            print(f"  video → {video_path}")
        if strip and strip_path:
            import imageio
            montage = np.concatenate(strip, axis=1)   # side-by-side time strip
            imageio.imwrite(strip_path, montage)
            print(f"  frame strip → {strip_path}")
        return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--video-path", default=None)
    ap.add_argument("--strip-path", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ev = WalkMujocoEval(a.checkpoint, vx=a.vx, device=a.device, record_video=a.record_video)
    ts = time.strftime("%Y%m%d_%H%M%S")
    vp = a.video_path or (f"videos/walk_eval_{ts}.mp4" if a.record_video else None)
    sp = a.strip_path or (f"videos/walk_strip_{ts}.png" if a.record_video else None)
    ev.run(steps=a.steps, video_path=vp, strip_path=sp)


if __name__ == "__main__":
    main()
