#!/usr/bin/env python3
"""Generate presentation videos of the standup system — all rendered in MuJoCo.

Each video is a standalone function; ``main()`` runs them all. Produced clips:

  * curriculum            — the named fallen start poses + the air-drop/settle
                            that seeds the init-pose pool.
  * assist_force          — the decaying HoST upward trunk support, drawn as an
                            arrow whose length tracks the applied force.
  * domain_randomization  — random base pushes (force arrows) + a sweep of the
                            randomised dynamics (friction / mass / gains).
  * reward_breakdown      — a scripted fallen->stand sweep with the live reward
                            signals overlaid as bars (what each term measures).
  * checkpoint_comparison — several training checkpoints rolled out side by side
                            (sim2sim: the policy drives PD torques in MuJoCo).

Usage:
    python scripts/render_presentation.py                 # all videos
    python scripts/render_presentation.py --only curriculum assist_force
    python scripts/render_presentation.py --checkpoint-dir checkpoints/skill_standup
    python scripts/render_presentation.py --out videos/presentation

MuJoCo needs a GL backend: EGL (GPU box) or OSMesa (headless laptop). Set
MUJOCO_GL=osmesa if EGL is unavailable.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Pick a GL backend before importing mujoco if the caller didn't.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402

from configs.config import K1RobotConfig  # noqa: E402
from models.field_generator import FieldDimensions  # noqa: E402
from models.field_builder import add_field_to_spec  # noqa: E402
from envs import standup as poses  # noqa: E402
from skills.standup.config import StandupConfig  # noqa: E402
from skills.standup import rewards as R  # noqa: E402
from skills.common_obs import compute_common_obs  # noqa: E402

_ROBOT_XML = os.path.join(_ROOT, "models", "robot", "K1", "K1_22dof.xml")
_FIELD_JSON = os.path.join(_ROOT, "configs", "field_hsl_2026.json")
_FOOT_BODIES = ("left_foot_link", "right_foot_link")
_HAND_BODIES = ("left_hand_link", "right_hand_link")
_CONTACT_Z = 0.05
_SIM_DT = 0.002
_CONTROL_DT = 0.02
_ACTION_REPEAT = int(round(_CONTROL_DT / _SIM_DT))   # 10
_ACTION_DELTA_MAX = 0.5
_BODY_WEIGHT_N = 196.0


# ───────────────────────── scene construction ────────────────────────────


def build_scene(with_field: bool = True):
    """Load the K1 MJCF, swap its ground plane for the shared HSL field, add
    lights, and compile. Returns (model, data, idx) where idx bundles the
    name→address maps the rest of the script needs."""
    spec = mujoco.MjSpec.from_file(_ROBOT_XML)
    # Brighter, presentation-friendly lighting + offscreen size.
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    try:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
    except Exception:
        pass
    if with_field:
        try:
            g = spec.geom("ground")
            if g is not None:
                spec.delete(g)
        except Exception:
            pass
        add_field_to_spec(spec, mujoco, FieldDimensions.from_json(_FIELD_JSON),
                          add_ball=False)
    light = spec.worldbody.add_light(name="key", pos=[2, -2, 4],
                                     dir=[-0.4, 0.4, -0.8],
                                     diffuse=[0.6, 0.6, 0.6])
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    model = spec.compile()
    data = mujoco.MjData(model)
    idx = _build_index(model)
    return model, data, idx


def _build_index(model):
    """Map the canonical 22 joints (K1RobotConfig order) to MuJoCo qpos / dof /
    actuator addresses, and resolve foot/hand body ids + per-joint PD gains."""
    rc = K1RobotConfig()
    names = rc.joint_names
    qpos_adr, dof_adr, act_id, force_lim = [], [], [], []
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        dof_adr.append(int(model.jnt_dofadr[jid]))
        act_id.append(int(aid))
        force_lim.append(model.actuator_forcerange[aid].copy())
    kp, kd = _pd_gains(rc, names)

    def _bid(nm):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)

    return dict(
        rc=rc, names=names,
        qpos=np.array(qpos_adr), dof=np.array(dof_adr),
        act=np.array(act_id), flim=np.array(force_lim),
        kp=kp, kd=kd,
        default=np.array(rc.default_joint_pos, dtype=np.float64),
        foot_bid=[_bid(n) for n in _FOOT_BODIES],
        hand_bid=[_bid(n) for n in _HAND_BODIES],
        trunk_bid=_bid("Trunk"),
    )


def _pd_gains(rc, names):
    kp = np.zeros(len(names)); kd = np.zeros(len(names))
    for i, n in enumerate(names):
        if "Head" in n:
            kp[i], kd[i] = rc.kp_head, rc.kd_head
        elif "Hip" in n:
            kp[i], kd[i] = rc.kp_hip, rc.kd_hip
        elif "Knee" in n:
            kp[i], kd[i] = rc.kp_knee, rc.kd_knee
        elif "Ankle" in n:
            kp[i], kd[i] = rc.kp_ankle, rc.kd_ankle
        else:                       # shoulders / elbows / arms
            kp[i], kd[i] = rc.kp_arm, rc.kd_arm
    return kp, kd


# ───────────────────────── pose + control helpers ────────────────────────


def set_pose(data, idx, pose, extra_clearance: float = 0.0):
    """Place the robot in a named StandupPose (free base + joint targets)."""
    data.qpos[0:3] = [0.0, 0.0, pose.trunk_height + pose.spawn_clearance
                      + extra_clearance]
    data.qpos[3:7] = pose.trunk_quat            # MuJoCo + our convention: wxyz
    for n, adr in zip(idx["names"], idx["qpos"]):
        data.qpos[adr] = pose.joint_targets.get(n, 0.0)
    data.qvel[:] = 0.0


def joint_state(data, idx):
    q = data.qpos[idx["qpos"]].astype(np.float64)
    qd = data.qvel[idx["dof"]].astype(np.float64)
    return q, qd


def apply_pd(model, data, idx, target_q):
    """One control step: PD torque toward target_q, then action_repeat sub-steps
    of physics. `target_q` is in canonical joint order."""
    for _ in range(_ACTION_REPEAT):
        q, qd = joint_state(data, idx)
        tau = idx["kp"] * (target_q - q) - idx["kd"] * qd
        tau = np.clip(tau, idx["flim"][:, 0], idx["flim"][:, 1])
        data.ctrl[idx["act"]] = tau
        mujoco.mj_step(model, data)


def apply_force(model, data, idx, body_id, force_xyz):
    """Set a world-frame external force on a body (for assist / push viz)."""
    data.xfrc_applied[:] = 0.0
    data.xfrc_applied[body_id, 0:3] = force_xyz


# ───────────────────────── obs (sim2sim policy rollout) ───────────────────


def _contact_addons(model, data, idx):
    out = np.zeros((1, 8), dtype=np.float32)
    for i, bid in enumerate(idx["foot_bid"]):
        if bid >= 0:
            z = data.xpos[bid, 2]
            out[0, i] = z
            out[0, 4 + i] = float(z < _CONTACT_Z)
    for i, bid in enumerate(idx["hand_bid"]):
        if bid >= 0:
            z = data.xpos[bid, 2]
            out[0, 2 + i] = z
            out[0, 6 + i] = float(z < _CONTACT_Z)
    return out


def standup_obs(data, idx, last_action, step):
    """Build the 86-dim standup obs (78 common + 8 contact) from MuJoCo state,
    matching skills/common_obs + the standup addons."""
    q, qd = joint_state(data, idx)
    base = compute_common_obs(
        root_pos=data.qpos[0:3][None, :].astype(np.float32),
        root_quat=data.qpos[3:7][None, :].astype(np.float32),
        root_lin_vel=data.qvel[0:3][None, :].astype(np.float32),
        root_ang_vel=data.qvel[3:6][None, :].astype(np.float32),  # body frame
        joint_pos=q[None, :].astype(np.float32),
        joint_vel=qd[None, :].astype(np.float32),
        last_action=last_action[None, :].astype(np.float32),
        step_count=np.array([step], dtype=np.int64),
        default_joint_pos=idx["default"].astype(np.float32),
        control_dt=_CONTROL_DT,
    )
    return np.concatenate([base, _contact_addons(None, data, idx)], axis=1)[0]


# ───────────────────────── rendering primitives ──────────────────────────


def make_camera(lookat=(0, 0, 0.3), dist=3.2, elev=-18.0, azim=130.0):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.elevation = elev
    cam.azimuth = azim
    return cam


def add_arrow(renderer, p_from, p_to, rgba=(1.0, 0.2, 0.1, 1.0), width=0.02):
    """Append an arrow geom (p_from→p_to) to the renderer's scene."""
    scn = renderer.scene
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(p_from, dtype=np.float64),
                         np.asarray(p_to, dtype=np.float64))
    scn.ngeom += 1


def _font(size=22):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def overlay_text(frame, title=None, lines=None, bars=None):
    """Draw a title, text lines, and optional labelled bars on an RGB frame."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W = img.width
    if title:
        f = _font(30)
        d.rectangle([0, 0, W, 46], fill=(0, 0, 0, 140))
        d.text((16, 8), title, fill=(255, 255, 255), font=f)
    if lines:
        f = _font(20)
        y = 56
        for ln in lines:
            d.text((16, y), ln, fill=(240, 240, 80), font=f)
            y += 26
    if bars:
        f = _font(18)
        y = img.height - 18 * len(bars) - 16
        bx, bw = 170, 240
        for label, val in bars:
            d.text((16, y - 2), label, fill=(255, 255, 255), font=f)
            d.rectangle([bx, y, bx + bw, y + 14], outline=(255, 255, 255),
                        fill=(0, 0, 0, 120))
            fillw = int(bw * max(0.0, min(1.0, val)))
            d.rectangle([bx, y, bx + fillw, y + 14], fill=(90, 200, 90))
            d.text((bx + bw + 8, y - 2), f"{val:+.2f}", fill=(255, 255, 255),
                   font=f)
            y += 18
    return np.asarray(img)


def write_video(frames, path, fps=30):
    if not frames:
        print(f"  [skip] no frames for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import imageio
    imageio.mimwrite(path, frames, fps=fps, quality=8)
    print(f"  wrote {path}  ({len(frames)} frames)")


# ───────────────────────────── videos ────────────────────────────────────


def render_curriculum(out_dir, H=720, W=1280):
    """Show each named start pose, then an air-drop→settle (how the pool is
    seeded). Kinematic poses + a short physics settle per pose."""
    model, data, idx = build_scene()
    renderer = mujoco.Renderer(model, H, W)
    cam = make_camera()
    factories = [("Supine (on back)", poses.supine),
                 ("Prone (face down)", poses.prone),
                 ("Side-left", poses.side_left),
                 ("Side-right", poses.side_right)]
    frames = []
    for label, fac in factories:
        pose = fac()
        # Air-drop: spawn a bit high, let physics settle it (seeds the pool).
        mujoco.mj_resetData(model, data)
        set_pose(data, idx, pose, extra_clearance=0.25)
        mujoco.mj_forward(model, data)
        hold = np.array([pose.joint_targets.get(n, 0.0) for n in idx["names"]])
        for k in range(90):
            apply_pd(model, data, idx, hold)
            renderer.update_scene(data, camera=cam)
            frames.append(overlay_text(
                renderer.render(), title=f"Init-pose curriculum — {label}",
                lines=["spawn in the air → settle to a physical fallen pose"]))
    write_video(frames, os.path.join(out_dir, "curriculum.mp4"))
    del renderer


def render_assist_force(out_dir, H=720, W=1280):
    """Decaying HoST upward trunk support, drawn as an arrow whose length tracks
    the force. Scripted lift (assist holds the trunk) so the concept is clear
    without a policy."""
    model, data, idx = build_scene()
    renderer = mujoco.Renderer(model, H, W)
    cam = make_camera(dist=3.0)
    cfg = StandupConfig()
    pose = poses.prone()
    mujoco.mj_resetData(model, data)
    set_pose(data, idx, pose)
    mujoco.mj_forward(model, data)
    frames = []
    target_h = cfg.target_height
    for t in range(220):
        z = float(data.qpos[2])
        # Spring-shaped, height-deficit assist (as in the env), fraction decays.
        frac = max(0.0, 1.0 - t / 200.0)
        deficit = max(0.0, (target_h - z) / target_h)
        fz = frac * cfg.assist_force_max * min(1.0, deficit)
        apply_force(model, data, idx, idx["trunk_bid"], [0.0, 0.0, fz])
        # Gently command the standing pose so the assisted trunk rises.
        apply_pd(model, data, idx, idx["default"])
        trunk = data.xpos[idx["trunk_bid"]].copy()
        renderer.update_scene(data, camera=cam)
        arrow_len = 0.15 + 0.6 * (fz / max(cfg.assist_force_max, 1e-6))
        if fz > 1.0:
            add_arrow(renderer, trunk, trunk + np.array([0, 0, arrow_len]),
                      rgba=(0.1, 0.5, 1.0, 1.0))
        frames.append(overlay_text(
            renderer.render(), title="Assist-force curriculum (HoST)",
            lines=[f"upward trunk support: {fz:6.1f} N   (decaying → 0)",
                   "blue arrow ∝ applied force"]))
    write_video(frames, os.path.join(out_dir, "assist_force.mp4"))
    del renderer


def render_domain_randomization(out_dir, H=720, W=1280):
    """Random base pushes (red force arrows) on a standing robot — the visible
    face of the domain randomisation that hardens the policy."""
    model, data, idx = build_scene()
    renderer = mujoco.Renderer(model, H, W)
    cam = make_camera(dist=3.0)
    rng = np.random.default_rng(0)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0.52]
    data.qpos[3:7] = [1, 0, 0, 0]
    for n, adr in zip(idx["names"], idx["qpos"]):
        data.qpos[adr] = idx["default"][list(idx["names"]).index(n)]
    mujoco.mj_forward(model, data)
    frames, push = [], np.zeros(3)
    for t in range(260):
        if t % 40 == 0:                       # new push every ~0.8 s
            ang = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(40, 110)
            push = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
        active = (t % 40) < 6                  # held ~5 control steps
        apply_force(model, data, idx, idx["trunk_bid"],
                    push if active else np.zeros(3))
        apply_pd(model, data, idx, idx["default"])
        trunk = data.xpos[idx["trunk_bid"]].copy()
        renderer.update_scene(data, camera=cam)
        if active:
            tip = trunk + push / 120.0
            add_arrow(renderer, trunk, tip, rgba=(1.0, 0.2, 0.1, 1.0))
        frames.append(overlay_text(
            renderer.render(), title="Domain randomisation — random base pushes",
            lines=["also randomised (not visible): friction, motor gains,",
                   "joint friction, link masses, COM offset, sensor noise",
                   f"push: {np.linalg.norm(push) if active else 0:5.0f} N"]))
    write_video(frames, os.path.join(out_dir, "domain_randomization.mp4"))
    del renderer


def render_reward_breakdown(out_dir, H=720, W=1280):
    """Scripted fallen→stand sweep (kinematic) with the live reward signals
    overlaid — illustrates what each shaping term measures."""
    model, data, idx = build_scene()
    renderer = mujoco.Renderer(model, H, W)
    cam = make_camera(dist=3.0)
    rc = K1RobotConfig()
    cfg = StandupConfig()
    start = poses.prone()
    q0 = np.array([start.joint_targets.get(n, 0.0) for n in idx["names"]])
    q1 = idx["default"].copy()
    # Shoulder-wide standing target (matches the stand_pose reward target).
    nm = list(idx["names"])
    q1[nm.index("Left_Hip_Roll")] += cfg.stand_target_hip_abduction
    q1[nm.index("Right_Hip_Roll")] -= cfg.stand_target_hip_abduction
    quat0 = np.array(start.trunk_quat, dtype=np.float64)
    quat1 = np.array([1.0, 0.0, 0.0, 0.0])
    z0, z1 = start.trunk_height, cfg.target_height
    pose_idx = tuple(rc.arm_joint_indices) + tuple(rc.leg_joint_indices)
    frames = []
    N = 160
    for t in range(N):
        a = t / (N - 1)
        s = 0.5 - 0.5 * math.cos(math.pi * a)        # smootherstep
        data.qpos[2] = (1 - s) * z0 + s * z1
        data.qpos[0:2] = 0.0
        data.qpos[3:7] = _slerp(quat0, quat1, s)
        for n, adr in zip(idx["names"], idx["qpos"]):
            data.qpos[adr] = (1 - s) * q0[nm.index(n)] + s * q1[nm.index(n)]
        mujoco.mj_forward(model, data)
        quat = data.qpos[3:7][None, :].astype(np.float32)
        z = data.qpos[2:3][None].astype(np.float32)
        jp = data.qpos[idx["qpos"]][None, :].astype(np.float32)
        foot_z = np.array([[data.xpos[b, 2] for b in idx["foot_bid"]]],
                          dtype=np.float32)
        up = R.upright_signal(quat)
        bars = [
            ("upright", float(R.upright_signal(quat)[0] * 0.5 + 0.5)),
            ("height", float(R.height_signal(z[:, 0], cfg.target_height)[0])),
            ("foot_grounded_up", float(R.foot_grounded_up_signal(
                foot_z, z[:, 0], up, foot_max_z=cfg.foot_grounded_max_z,
                trunk_min_z=cfg.trunk_up_min_z)[0])),
            ("standing_tall", float(R.standing_tall_signal(
                foot_z, z[:, 0], up, foot_max_z=cfg.foot_grounded_max_z,
                trunk_min_z=cfg.standing_tall_min_z,
                trunk_max_z=cfg.standing_tall_max_z)[0])),
            ("stand_pose", float(R.stand_pose_signal(
                jp, pose_idx, q1.astype(np.float32), up,
                dev_scale=cfg.stand_pose_dev_scale)[0])),
        ]
        renderer.update_scene(data, camera=cam)
        frames.append(overlay_text(
            renderer.render(), title="Reward shaping — live signal breakdown",
            bars=bars))
    write_video(frames, os.path.join(out_dir, "reward_breakdown.mp4"))
    del renderer


def render_checkpoint_comparison(checkpoints, out_dir, H=540, W=540,
                                 steps=250):
    """Roll out each checkpoint's policy in MuJoCo (sim2sim) from a prone start
    and tile the clips side by side, labelled by training step."""
    try:
        import torch  # noqa: F401
        from training.common import create_policy, load_checkpoint
    except Exception as e:
        print(f"  [skip] checkpoint_comparison needs torch: {e}")
        return
    if not checkpoints:
        print("  [skip] checkpoint_comparison: no checkpoints found")
        return
    panels = []
    for ckpt in checkpoints:
        frames = _rollout_checkpoint(ckpt, create_policy, load_checkpoint,
                                     H, W, steps)
        if frames:
            panels.append((_step_of(ckpt), frames))
    if not panels:
        return
    n = min(len(f) for _, f in panels)
    tiled = []
    for t in range(n):
        row = [overlay_text(f[t], title=f"step {step:,}")
               for step, f in panels]
        tiled.append(np.concatenate(row, axis=1))
    write_video(tiled, os.path.join(out_dir, "checkpoint_comparison.mp4"))


# Nominal (no-DR) privileged tail appended for teacher checkpoints (obs 94):
# [ground_friction, kp_scale, kd_scale, joint_friction, base_mass_scale,
#  com_offset_xyz] at their un-randomised values.
_PRIV_NOMINAL = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                         dtype=np.float32)
_BASE_OBS_DIM = 86


def _ckpt_obs_dim(ckpt):
    """Detect a checkpoint's expected obs_dim (proprio 86 vs privileged 94) +
    return its obs_norm state dict, without assuming a layout."""
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    obs_norm_sd = sd.get("obs_norm") if isinstance(sd, dict) else None
    dim = None
    if obs_norm_sd and "mean" in obs_norm_sd:
        dim = int(np.asarray(obs_norm_sd["mean"]).reshape(-1).shape[0])
    if dim is None and isinstance(sd, dict):
        psd = sd.get("policy_state_dict") or sd.get("actor_state_dict") or sd
        for k, v in psd.items():
            if "actor" in k and k.endswith("weight") and getattr(v, "ndim", 0) == 2:
                dim = int(v.shape[1]); break
    return dim, obs_norm_sd


def _rollout_checkpoint(ckpt, create_policy, load_checkpoint, H, W, steps):
    import torch
    act_dim = 22
    obs_dim, obs_norm_sd = _ckpt_obs_dim(ckpt)
    if obs_dim not in (_BASE_OBS_DIM, _BASE_OBS_DIM + 8):
        print(f"  [skip] {os.path.basename(ckpt)}: obs_dim {obs_dim} unsupported")
        return []
    model, data, idx = build_scene()
    policy = create_policy(obs_dim, act_dim)
    if policy is None:
        return []
    try:
        load_checkpoint(ckpt, policy)
        policy.eval()
    except Exception as e:
        print(f"  [skip] could not load {ckpt}: {e}")
        return []
    obs_norm = None
    if obs_norm_sd is not None:
        from training.normalizers import RunningMeanStd
        obs_norm = RunningMeanStd(shape=(obs_dim,))
        obs_norm.load_state_dict(obs_norm_sd)
    renderer = mujoco.Renderer(model, H, W)
    cam = make_camera(dist=3.2)
    mujoco.mj_resetData(model, data)
    set_pose(data, idx, poses.prone())
    mujoco.mj_forward(model, data)
    last_action = np.zeros(act_dim, dtype=np.float32)
    frames = []
    for t in range(steps):
        obs = standup_obs(data, idx, last_action, t)
        if obs_dim == _BASE_OBS_DIM + 8:          # privileged/teacher policy
            obs = np.concatenate([obs, _PRIV_NOMINAL])
        nobs = obs_norm.normalize(obs[None])[0] if obs_norm is not None else obs
        with torch.no_grad():
            out = policy.act(torch.as_tensor(nobs[None], dtype=torch.float32),
                             deterministic=True)
        action = np.asarray(_action_from(out)).reshape(-1)[:act_dim]
        last_action = action.astype(np.float32)
        target = np.clip(idx["default"]
                         + np.clip(action, -_ACTION_DELTA_MAX, _ACTION_DELTA_MAX),
                         -math.pi, math.pi)
        apply_pd(model, data, idx, target)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    del renderer
    return frames


def _action_from(out):
    """policy.act may return action, (action, ...), or a dict — be tolerant."""
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("action", next(iter(out.values())))
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return out


# ───────────────────────────── utilities ─────────────────────────────────


def _slerp(q0, q1, s):
    q0 = q0 / (np.linalg.norm(q0) + 1e-9)
    q1 = q1 / (np.linalg.norm(q1) + 1e-9)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return (q0 + s * (q1 - q0)) / (np.linalg.norm(q0 + s * (q1 - q0)) + 1e-9)
    th = math.acos(dot)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def _step_of(path):
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def find_checkpoints(ckpt_dir, k=4):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")), key=_step_of)
    if len(files) <= k:
        return files
    pick = [files[int(round(i * (len(files) - 1) / (k - 1)))] for i in range(k)]
    # de-dup preserving order
    seen, out = set(), []
    for p in pick:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


_VIDEOS = {
    "curriculum": render_curriculum,
    "assist_force": render_assist_force,
    "domain_randomization": render_domain_randomization,
    "reward_breakdown": render_reward_breakdown,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_ROOT, "videos", "presentation"))
    ap.add_argument("--checkpoint-dir",
                    default=os.path.join(_ROOT, "checkpoints", "skill_standup"))
    ap.add_argument("--only", nargs="*",
                    choices=list(_VIDEOS) + ["checkpoint_comparison"],
                    help="render only these videos (default: all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    selected = args.only or (list(_VIDEOS) + ["checkpoint_comparison"])
    print(f"Rendering to {args.out}  (MUJOCO_GL={os.environ.get('MUJOCO_GL')})")
    for name in selected:
        print(f"[{name}]")
        try:
            if name == "checkpoint_comparison":
                render_checkpoint_comparison(
                    find_checkpoints(args.checkpoint_dir), args.out)
            else:
                _VIDEOS[name](args.out)
        except Exception as e:
            import traceback
            print(f"  [error] {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
