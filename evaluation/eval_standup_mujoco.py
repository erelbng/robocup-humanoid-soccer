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
SETTLE_STEPS = 300              # physics steps to settle the fallen pose
                                # (0.6s — matches the Genesis settle pool)


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


def _random_unit_quat(rng):
    """Uniform sample on SO(3) (Shoemake) → (w, x, y, z). Matches the
    Genesis env's settle-pool orientation sampling so the eval's fallen
    starts come from the same distribution the teacher trained on."""
    u = rng.uniform(0.0, 1.0, size=3)
    s1, s0 = np.sqrt(1.0 - u[0]), np.sqrt(u[0])
    return np.array([s1 * np.sin(2*np.pi*u[1]), s1 * np.cos(2*np.pi*u[1]),
                     s0 * np.sin(2*np.pi*u[2]), s0 * np.cos(2*np.pi*u[2])],
                    dtype=np.float64)


class StandupMujocoEval:
    def __init__(self, checkpoint: str, device: str = "cpu",
                 record_video: bool = False,
                 upright_thr: float = UPRIGHT_THRESHOLD,
                 height_bar: float = TARGET_HEIGHT - 0.10,
                 floor_friction: float = None,
                 torque_scale: float = 1.0):
        self.cfg = K1RobotConfig()
        self.device = torch.device(device)
        self.record_video = record_video
        self.upright_thr = float(upright_thr)
        self.height_bar = float(height_bar)
        torque_scale = float(torque_scale)

        # ── model ──
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(_XML))
        self.model.opt.timestep = SIM_DT
        # Optional: override ground friction (the MJCF ships friction 0.4,
        # below Genesis's training range 0.5–1.5). Set condim=3 so tangential
        # friction is definitely active, and bump sliding friction to match
        # Genesis nominal (~1.0). Lets us test whether low floor traction is
        # what stops the policy completing the stand.
        if floor_friction is not None:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    "ground")
            if gid >= 0:
                self.model.geom_condim[gid] = 3
                self.model.geom_friction[gid] = [float(floor_friction),
                                                 0.005, 0.0001]
                print(f"[eval] ground friction → {floor_friction} (condim=3)")
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
        # torque_scale > 1 relaxes the motor force limits — used to test
        # whether Genesis was applying PD torque beyond the MJCF/URDF limits
        # (it never sets a force range, so it may run unclipped PD).
        self.force_lo = self.model.actuator_forcerange[self.act_id, 0] * torque_scale
        self.force_hi = self.model.actuator_forcerange[self.act_id, 1] * torque_scale
        self.default = np.asarray(self.cfg.default_joint_pos, np.float64)
        # torque-saturation diagnostics
        self._tau_abs_max = 0.0
        self._tau_sat = 0
        self._tau_n = 0

        # ── policy ──
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("policy_state_dict") or ckpt.get("actor_state_dict")
        if sd is None:
            raise RuntimeError("checkpoint has no policy/actor_state_dict")
        obs_dim = sd["actor_trunk.0.weight"].shape[1]
        act_dim = sd["actor_head.weight"].shape[0]
        # Supported widths:
        #   78  proprio student / proprio_only           (deployable)
        #   86  proprio + 8 contact addons               (sim training, no DR)
        #   94  proprio + 8 contact + 8 privileged DR    (teacher)
        # The contact dims are computed faithfully from MuJoCo foot/hand z;
        # the privileged DR dims (teacher only) have no MuJoCo counterpart, so
        # we feed the obs_norm mean → they normalize to ~0 ("average dynamics").
        # The teacher is trained to be robust to the DR axis, so this is a fair
        # nominal — and lets us sanity-check the eval against a known-~45%
        # policy. Don't read a teacher score here as deployable performance.
        if obs_dim not in (SKILL_BASE_OBS_DIM, SKILL_BASE_OBS_DIM + 8,
                           SKILL_BASE_OBS_DIM + 16):
            raise SystemExit(f"unsupported checkpoint obs_dim={obs_dim} "
                             "(expected 78, 86, or 94)")
        self.use_contact = obs_dim >= SKILL_BASE_OBS_DIM + 8
        self.use_privileged = obs_dim >= SKILL_BASE_OBS_DIM + 16
        if self.use_privileged:
            print("[eval] NOTE: privileged teacher (obs_dim 94) — privileged "
                  "DR dims fed as nominal (obs_norm mean). This is an "
                  "eval-fidelity sanity check, NOT deployable performance.")
        if self.use_contact:
            self.foot_body = [mujoco.mj_name2id(self.model,
                              mujoco.mjtObj.mjOBJ_BODY, n)
                              for n in ("left_foot_link", "right_foot_link")]
            self.hand_body = [mujoco.mj_name2id(self.model,
                              mujoco.mjtObj.mjOBJ_BODY, n)
                              for n in ("left_hand_link", "right_hand_link")]
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
        if self.use_contact:
            obs = np.concatenate([obs, self._contact_obs()], axis=1)
        if self.use_privileged:
            # Nominal privileged: obs_norm mean for those 8 dims → normalizes
            # to ~0 ("average dynamics"). No MuJoCo counterpart for DR.
            priv = self.obs_norm.mean[None, SKILL_BASE_OBS_DIM + 8:
                                      SKILL_BASE_OBS_DIM + 16].astype(np.float32)
            obs = np.concatenate([obs, priv], axis=1)
        return self.obs_norm.normalize(obs).astype(np.float32)

    def _contact_obs(self):
        """(1, 8) = [lf_z, rf_z, lh_z, rh_z, lf_c, rf_c, lh_c, rh_c] — matches
        env._read_contact_state: world-frame z + contact bool (z < 0.05)."""
        fz = [float(self.data.xpos[b][2]) for b in self.foot_body]
        hz = [float(self.data.xpos[b][2]) for b in self.hand_body]
        zs = fz + hz
        c = [1.0 if z < 0.05 else 0.0 for z in zs]
        return np.array([zs + c], dtype=np.float32)

    cfg_gait_freq = 1.5

    def _apply_pd(self, targets: np.ndarray):
        q, qd = self._q(), self._qd()
        tau_raw = self.kp * (targets - q) - self.kd * qd
        tau = np.clip(tau_raw, self.force_lo, self.force_hi)
        # diagnostics: how often does the raw PD command exceed the limits?
        self._tau_abs_max = max(self._tau_abs_max, float(np.max(np.abs(tau_raw))))
        self._tau_sat += int(np.sum(tau_raw != tau))
        self._tau_n += tau_raw.size
        self.data.ctrl[self.act_id] = tau

    def _reset_fallen(self, rng: np.random.Generator):
        """Match the Genesis settle pool: spawn high with a RANDOM orientation
        + joint noise, hold the default pose via PD, and let physics settle it
        into a natural fallen pose. (The previous version dropped 4 stiff
        canonical poses from 0.30 m with only a 0.12 s settle — out of the
        teacher's training distribution, which is why it couldn't complete.)"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.base_qadr + 0] = 0.0
        self.data.qpos[self.base_qadr + 1] = 0.0
        self.data.qpos[self.base_qadr + 2] = rng.uniform(0.8, 1.2)
        self.data.qpos[self.base_qadr + 3: self.base_qadr + 7] = \
            _random_unit_quat(rng)
        self.data.qpos[self.qpos_adr] = self.default + rng.normal(
            0, 0.2, size=self.cfg.num_dofs)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(SETTLE_STEPS):
            self._apply_pd(self.default)
            mujoco.mj_step(self.model, self.data)

    # ── episode ──
    def run_episode(self, rng, frames=None):
        self._reset_fallen(rng)
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
            success_frame = (up > self.upright_thr) and (z > self.height_bar)
            streak = streak + 1 if success_frame else 0
            if streak >= SUCCESS_HOLD_STEPS and t_first is None:
                t_first = step - (SUCCESS_HOLD_STEPS - 1)

            if frames is not None and step % 2 == 0:    # ~25 fps
                self.renderer.update_scene(self.data, camera=-1)
                frames.append(self.renderer.render().copy())

        return {
            "success": t_first is not None,
            "time_to_stand_s": (t_first * DT) if t_first is not None else None,
            "final_up": _upright(self._root_quat()),
            "final_z": self._root_z(),
        }

    def evaluate(self, episodes: int, video_path: str = None):
        rng = np.random.default_rng(0)
        frames = [] if self.record_video else None
        results = []
        for ep in range(episodes):
            ep_frames = frames if (self.record_video and ep < 4) else None
            r = self.run_episode(rng, frames=ep_frames)
            results.append(r)
            tts = f"{r['time_to_stand_s']:.2f}s" if r["success"] else "—"
            print(f"  ep {ep+1:3d} success={r['success']!s:5s} "
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
        sat = 100.0 * self._tau_sat / max(self._tau_n, 1)
        print(f"   PD torque: max|tau|={self._tau_abs_max:.0f} Nm, "
              f"{sat:.1f}% of joint-steps hit the force limit")
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
    ap.add_argument("--upright-threshold", type=float, default=UPRIGHT_THRESHOLD,
                    help="success: trunk upright cos > this (default 0.92)")
    ap.add_argument("--height-bar", type=float, default=TARGET_HEIGHT - 0.10,
                    help="success: trunk z > this (default 0.45)")
    ap.add_argument("--floor-friction", type=float, default=None,
                    help="override ground sliding friction + condim=3 "
                         "(MJCF ships 0.4; Genesis nominal ~1.0)")
    ap.add_argument("--torque-scale", type=float, default=1.0,
                    help="multiply motor force limits (set high, e.g. 100, to "
                         "test whether Genesis ran unclipped PD)")
    args = ap.parse_args()

    ev = StandupMujocoEval(args.checkpoint, device=args.device,
                           record_video=args.record_video,
                           upright_thr=args.upright_threshold,
                           height_bar=args.height_bar,
                           floor_friction=args.floor_friction,
                           torque_scale=args.torque_scale)
    print(f"[eval] success bar: upright>{args.upright_threshold} & "
          f"z>{args.height_bar} held {SUCCESS_HOLD_STEPS} steps")
    vp = args.video_path or os.path.join(
        "videos", f"standup_eval_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    ev.evaluate(args.episodes, video_path=vp)


if __name__ == "__main__":
    main()
