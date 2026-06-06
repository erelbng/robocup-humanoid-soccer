"""create_standup_amp.py — physics-grounded AMP reference generator for K1 stand-up.

WHAT THIS PRODUCES
------------------
A reference-motion AMP dataset (``data/motions/k1_standup_amp.npz``) of
``(M, 2*AMP_OBS_DIM)`` transitions, directly loadable by
``training.algorithms.amp.load_motion_dataset`` and used as the style reference
for the standup skill (``StandupConfig.amp_motion_file``).

WHY IT LOOKS RIGHT (the bugs in the old versions)
-------------------------------------------------
AMP references are KINEMATIC clips — like mocap, they are never forward-simulated
(the *policy* + physics is what makes the motion dynamically feasible; the
reference only has to look like a plausible, grounded motion). The two previous
generators failed at that "grounded" part:

  * ``create_standup_amp`` (old) LERP'd the trunk quaternion component-wise →
    invalid/denormalised rotations (the "exploded / floating in the air" look)
    and ground-locked on the FEET only → the trunk floated while the robot was
    lying down (feet aren't the contact then).
  * ``create_mink_standup`` settled with physics but drove the body with IK,
    which cannot lift the floating base against gravity → the robot never stood.

This generator fixes both and stays faithful to real physics where it matters:

  1. START FROM A REAL SETTLED FALLEN POSE.  The same trick the standup training
     uses to build its init-pose curriculum (``skills/standup/env._build_pose_pool``):
     spawn one of the ``envs/standup`` fallen poses (supine / prone / side) and let
     MuJoCo settle it under gravity + contact. The get-up therefore BEGINS at a
     genuine, physically-consistent fallen state (matching how the policy is reset).
  2. SLERP the trunk orientation between keyframes (no more denormalised quats).
  3. GROUND-PROJECT EVERY FRAME using MuJoCo forward kinematics: shift the root z
     so the LOWEST collision geom over the WHOLE body rests on the floor. This
     generalises the old feet-only lock to handle every phase of the get-up (back
     → elbows → knees → feet) so the robot is never floating and never buried.

VISUALISATION (adopted from create_mink_standup)
------------------------------------------------
  * offscreen keyframe grid with per-frame time labels (one row per motion type),
  * optional MP4 of a sample motion,
  * optional slow-motion passive-viewer replay.

USAGE
-----
    .venv/bin/python scripts/create_standup_amp.py                 # full run
    .venv/bin/python scripts/create_standup_amp.py --no-viewer     # headless
    .venv/bin/python scripts/create_standup_amp.py --variants 40 --video
    .venv/bin/python scripts/create_standup_amp.py --types back belly side_left side_right
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mujoco  # noqa: E402

from configs.config import K1RobotConfig  # noqa: E402
from envs.standup import prone, side_left, side_right, supine  # noqa: E402
from skills.common_obs import body_frame_velocity, projected_gravity  # noqa: E402
from training.algorithms.amp import AMP_OBS_DIM, build_amp_obs  # noqa: E402

# ─── paths / constants ───────────────────────────────────────────────────────

XML_PATH = os.path.join(PROJECT_ROOT, "models/robot/K1/K1_22dof.xml")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/motions/k1_standup_amp.npz")
IMAGE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/motions/k1_standup_amp_keyframes.png")
VIDEO_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/motions/k1_standup_amp_sample.mp4")

DT = 0.02          # control / output timestep (50 Hz) — matches AMP obs cadence
FPS = 50

_CFG = K1RobotConfig()
JOINT_NAMES = list(_CFG.joint_names)
N_DOF = len(JOINT_NAMES)
DEFAULT_JPOS = np.asarray(_CFG.default_joint_pos, dtype=np.float64)
_JIDX = {n: i for i, n in enumerate(JOINT_NAMES)}

# The geom-frame foot link sits a few cm above the sole; a planted foot reads a
# small constant z, so "clearance" is measured relative to each foot's own
# minimum over the clip (matches scripts/retarget/build_amp_reference.py).
FOOT_BODIES = ("left_foot_link", "right_foot_link")


# ─── PD gains (authoritative K1 values from K1RobotConfig) ────────────────────


def _gain_arrays():
    """Per-joint (kp, kd) arrays in JOINT_NAMES order, from K1RobotConfig."""
    kp = np.zeros(N_DOF)
    kd = np.zeros(N_DOF)
    for i, n in enumerate(JOINT_NAMES):
        if "Head" in n:
            kp[i], kd[i] = _CFG.kp_head, _CFG.kd_head
        elif "Shoulder" in n or "Elbow" in n:
            kp[i], kd[i] = _CFG.kp_arm, _CFG.kd_arm
        elif "Hip" in n:
            kp[i], kd[i] = _CFG.kp_hip, _CFG.kd_hip
        elif "Knee" in n:
            kp[i], kd[i] = _CFG.kp_knee, _CFG.kd_knee
        elif "Ankle" in n:
            kp[i], kd[i] = _CFG.kp_ankle, _CFG.kd_ankle
        else:
            kp[i], kd[i] = _CFG.kp, _CFG.kd
    return kp, kd


# ─── small math helpers ──────────────────────────────────────────────────────


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def slerp(q0, q1, t):
    """Spherical interpolation of (w,x,y,z) quaternions — the fix for the
    component-wise LERP that produced denormalised, exploded rotations."""
    q0 = np.asarray(q0, np.float64)
    q1 = np.asarray(q1, np.float64)
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:                      # nearly parallel → normalised lerp
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    th = np.arccos(dot)
    s = np.sin(th)
    return (np.sin((1.0 - t) * th) / s) * q0 + (np.sin(t * th) / s) * q1


def quat_ang_vel(qc, qn, dt):
    """Body-ish angular velocity from consecutive (w,x,y,z) quats (finite diff)."""
    w0, x0, y0, z0 = qc.T
    w1, x1, y1, z1 = qn.T
    dw = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
    dx = x1 * w0 - w1 * x0 - y1 * z0 + z1 * y0
    dy = y1 * w0 + w1 * y0 + x1 * z0 - z1 * x0
    dz = z1 * w0 - w1 * z0 - x1 * y0 + y1 * x0
    s = np.sign(dw)
    s[s == 0] = 1.0
    return 2.0 * np.stack([dx, dy, dz], axis=1) * s[:, None] / dt


def jpos_from_pose(pose) -> np.ndarray:
    """22-dim joint target from a StandupPose's joint_targets dict (defaults
    fill the unspecified joints)."""
    j = DEFAULT_JPOS.copy()
    for name, angle in pose.joint_targets.items():
        if name in _JIDX:
            j[_JIDX[name]] = float(angle)
    return j


# ─── get-up keyframe joint poses (the "functionality": authored joint targets) ─
#
# A real get-up pushes off the GROUND WITH THE HANDS, so the choreography routes
# both the back- and belly-start motions through a "bear / quadruped" stage where
# the hands are planted forward on the floor and push the front of the body up
# (verified by FK: in these stages the hand links are the lowest geoms, so the
# whole-body ground projection makes them load-bearing). The hands then lift off
# as the feet take over for squat → stand.
#
# K1 arm convention (shoulder-roll zero is the T-pose): shoulder_pitch≈1.2 swings
# the upper arm forward/down, roll ±1.5 brings it in toward the floor, elbow_pitch
# bends the forearm down — together they plant the hand ~0.07 m up, ~0.25 m ahead
# of the trunk. Leg angles set hip-pitch (flex) / knee / ankle-pitch.


def _arms(j, shoulder_pitch, shoulder_roll, elbow_pitch):
    """Set both arms symmetrically. `shoulder_roll` is the LEFT value; the right
    mirrors it (right = -left), matching the K1 default (Left roll -1.4, Right
    +1.4 = arms hanging at the sides). So roll ≈ -1.4 hangs the arms down for an
    UPRIGHT trunk; POSITIVE roll swings them the other way (used by the prone
    cobra/bear, where the flipped body makes positive roll point the hands at the
    floor)."""
    j = j.copy()
    j[_JIDX["ALeft_Shoulder_Pitch"]] = shoulder_pitch
    j[_JIDX["ARight_Shoulder_Pitch"]] = shoulder_pitch
    j[_JIDX["Left_Shoulder_Roll"]] = shoulder_roll
    j[_JIDX["Right_Shoulder_Roll"]] = -shoulder_roll
    j[_JIDX["Left_Elbow_Pitch"]] = elbow_pitch
    j[_JIDX["Right_Elbow_Pitch"]] = elbow_pitch
    return j


def _legs(j, hip_pitch, knee, ankle_pitch, hip_roll=None):
    """Set both legs symmetrically. `hip_roll` (optional) abducts the knees apart
    (mirrored L/R) for the wide-knee squat seen in real K1 get-ups."""
    j = j.copy()
    for n in JOINT_NAMES:
        if "Hip_Pitch" in n:
            j[_JIDX[n]] = hip_pitch
        elif "Knee" in n:
            j[_JIDX[n]] = knee
        elif "Ankle_Pitch" in n:
            j[_JIDX[n]] = ankle_pitch
    if hip_roll is not None:
        j[_JIDX["Left_Hip_Roll"]] = hip_roll
        j[_JIDX["Right_Hip_Roll"]] = -hip_roll
    return j


def quat_pitch(angle: float) -> np.ndarray:
    """Trunk forward-pitch quaternion (rotation about world Y): 0 = upright,
    ~π/2 = lying. Used for the intermediate (cobra/bear/squat) trunk targets."""
    return np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])


def getup_choreography(motion_type: str, j_start, q_start, rng):
    """Return (kf_joints, kf_quats, seg_frames) for a hands-pushing get-up that
    starts at the settled fallen state (j_start, q_start). `rng` adds mild
    per-clip variation. Both routes pass through the hands-down 'bear' stage."""
    r = lambda s: rng.uniform(-s, s)  # noqa: E731

    # hands planted forward on the floor, chest just lifting (early push)
    cobra = _arms(_legs(DEFAULT_JPOS, 0.2 + r(0.1), 0.6 + r(0.1), 0.1),
                  1.15 + r(0.05), 1.45 + r(0.05), -1.0 + r(0.1))
    # quadruped: hands still planted forward, hips up over deeply-flexed legs
    bear = _arms(_legs(DEFAULT_JPOS, -1.3 + r(0.1), 1.8 + r(0.1), -0.6 + r(0.1)),
                 1.15 + r(0.05), 1.45 + r(0.05), -0.7 + r(0.1))
    # squat: feet take over, hands lifting back toward the sides
    squat = _arms(_legs(DEFAULT_JPOS, -1.0 + r(0.1), 1.7 + r(0.1), -0.8 + r(0.1)),
                  0.0, -1.4, 0.0)
    stand = DEFAULT_JPOS.copy()

    if motion_type == "belly":
        # prone → cobra (hands plant, push chest up) → bear → squat → stand
        kf_joints = [j_start, cobra, bear, squat, stand]
        kf_quats = [q_start, quat_pitch(1.4), quat_pitch(0.9),
                    quat_pitch(0.0), quat_pitch(0.0)]
        base_frames = (38, 36, 34, 30)
    else:
        # back: TUCK the knees in while still on the back → ROCK forward onto the
        # planted feet into a wide deep squat (trunk leaning slightly forward over
        # the knees, arms out for balance) → rise to stand. The robot stays
        # FACE-UP throughout (trunk pitch -π/2 → 0, never toward +π/2 / prone), so
        # it does NOT roll onto its belly — this mirrors the real K1 fast get-up
        # (sit-up-and-rock-to-feet), not a push-up. Arms swing OUT for
        # counterbalance (shoulder roll lifts them toward horizontal), not planted.
        # Arms hang at the SIDES (roll ≈ -1.4 = default) during the tuck/rock so
        # they don't fly up; they raise slightly out + forward (roll -1.05) for
        # counterbalance in the deep squat. NEGATIVE roll = hanging (see _arms).
        tuck = _arms(_legs(DEFAULT_JPOS, -1.9 + r(0.1), 2.05 + r(0.1), 0.3 + r(0.1)),
                     0.15 + r(0.05), -1.35 + r(0.05), -0.2 + r(0.05))
        deep_squat = _arms(
            _legs(DEFAULT_JPOS, -1.3 + r(0.1), 1.9 + r(0.1), -0.5 + r(0.1),
                  hip_roll=0.15),
            0.45 + r(0.05), -1.05 + r(0.05), -0.3 + r(0.05))
        kf_joints = [j_start, tuck, deep_squat, squat, stand]
        kf_quats = [q_start, quat_pitch(-1.0), quat_pitch(0.3),
                    quat_pitch(0.1), quat_pitch(0.0)]
        base_frames = (26, 42, 28, 28)

    seg_frames = [int(round(f + rng.integers(-4, 5))) for f in base_frames]
    return kf_joints, kf_quats, seg_frames


# ─── physics + kinematics primitives ─────────────────────────────────────────


class K1Sim:
    """Thin MuJoCo wrapper: PD settle, ground projection, FK queries."""

    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.kp, self.kd = _gain_arrays()
        self.frc = self.model.actuator_forcerange[:, 1].copy()
        self.sim_dt = float(self.model.opt.timestep)
        self.foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in FOOT_BODIES
        ]

    # -- physics --
    def settle_pose(self, pose, rng, settle_seconds=1.2, jitter=0.06) -> np.ndarray:
        """Spawn a StandupPose above the floor and let it settle under gravity +
        contact while a PD holds the joint targets. Returns the settled qpos —
        a real, grounded fallen start state (init-pose-curriculum technique)."""
        m, d = self.model, self.data
        jtgt = jpos_from_pose(pose)
        jtgt = jtgt + rng.standard_normal(N_DOF) * jitter
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[:3] = [0.0, 0.0, pose.trunk_height + pose.spawn_clearance]
        d.qpos[3:7] = pose.trunk_quat
        d.qpos[7:] = jtgt
        mujoco.mj_forward(m, d)
        nsteps = int(settle_seconds / self.sim_dt)
        for _ in range(nsteps):
            q, v = d.qpos[7:], d.qvel[6:]
            d.ctrl[:] = np.clip(self.kp * (jtgt - q) - self.kd * v, -self.frc, self.frc)
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        qpos = d.qpos.copy()
        qpos[0:2] = 0.0                     # recentre xy
        return self.ground_project(qpos)

    # -- kinematics --
    def ground_project(self, qpos: np.ndarray) -> np.ndarray:
        """Shift root z so the lowest collision geom over the WHOLE body touches
        the floor (z=0). Generalises the old feet-only lock → never floats, never
        buried, in every phase of the get-up."""
        d = self.data
        qpos = qpos.copy()
        d.qpos[:] = qpos
        mujoco.mj_forward(self.model, d)
        qpos[2] -= float(d.geom_xpos[:, 2].min())
        return qpos

    def foot_heights(self, qpos: np.ndarray):
        d = self.data
        d.qpos[:] = qpos
        mujoco.mj_forward(self.model, d)
        return np.array([d.xpos[i][2] for i in self.foot_ids])


# ─── trajectory construction ─────────────────────────────────────────────────


def build_getup_trajectory(sim: K1Sim, q_start: np.ndarray,
                           kf_joints, kf_quats, seg_frames) -> np.ndarray:
    """Interpolate (smoothstep joints, slerp trunk quat) through the keyframes,
    ground-projecting every frame. Returns (M, 29) qpos. The first keyframe is
    the settled fallen state."""
    traj = []
    for s in range(len(seg_frames)):
        j0, j1 = kf_joints[s], kf_joints[s + 1]
        q0, q1 = kf_quats[s], kf_quats[s + 1]
        n = seg_frames[s]
        for i in range(n):
            a = smoothstep(i / max(1, n - 1))
            jj = (1.0 - a) * j0 + a * j1
            qq = slerp(q0, q1, a)
            qq = qq / np.linalg.norm(qq)
            full = np.zeros(29)
            full[0:3] = q_start[0:3]
            full[3:7] = qq
            full[7:] = jj
            traj.append(sim.ground_project(full))
    return np.asarray(traj)


def make_motion(sim: K1Sim, motion_type: str, rng: np.random.Generator) -> np.ndarray:
    """Build one grounded, hands-pushing get-up qpos trajectory for the fallen
    start type. The first keyframe is the physics-settled fallen state."""
    pose_fn = {
        "back": supine, "belly": prone,
        "side_left": side_left, "side_right": side_right,
    }[motion_type]
    q_settled = sim.settle_pose(pose_fn(), rng)
    j_start = q_settled[7:].copy()
    q_start = q_settled[3:7].copy()

    kf_joints, kf_quats, seg_frames = getup_choreography(
        "belly" if motion_type == "belly" else "back", j_start, q_start, rng)
    return build_getup_trajectory(sim, q_settled, kf_joints, kf_quats, seg_frames)


# ─── AMP conversion (canonical build_amp_obs → loadable dataset) ──────────────


def trajectory_to_amp(sim: K1Sim, traj: np.ndarray) -> np.ndarray:
    """(M,29) qpos → (M-1, 2*AMP_OBS_DIM) transitions via the canonical
    training.algorithms.amp.build_amp_obs (53-dim) layout."""
    M = len(traj)
    root_pos = traj[:, :3]
    root_quat = traj[:, 3:7]
    dof_pos = traj[:, 7:]

    dof_vel = np.zeros_like(dof_pos)
    dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / DT
    dof_vel[-1] = dof_vel[-2]

    ang_vel = np.zeros((M, 3))
    ang_vel[:-1] = quat_ang_vel(root_quat[:-1], root_quat[1:], DT)
    ang_vel[-1] = ang_vel[-2]

    foot_z = np.array([sim.foot_heights(traj[i]) for i in range(M)])
    ground = foot_z.min(0)                       # per-foot planted height
    foot_clear = np.clip(foot_z - ground, 0.0, 0.5)

    amp = np.zeros((M, AMP_OBS_DIM), dtype=np.float32)
    for i in range(M):
        q = root_quat[i:i + 1].astype(np.float32)
        ang_b = body_frame_velocity(q, ang_vel[i:i + 1].astype(np.float32))
        amp[i] = build_amp_obs(
            root_pos[i, 2],
            projected_gravity(q),
            ang_b,
            dof_pos[i:i + 1].astype(np.float32),
            dof_vel[i:i + 1].astype(np.float32),
            foot_clear[i:i + 1].astype(np.float32),
        )
    return np.concatenate([amp[:-1], amp[1:]], axis=1).astype(np.float32)


# ─── visualisation (adopted from create_mink_standup) ────────────────────────


def _free_camera(model):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 2.5
    cam.azimuth = -45.0
    cam.elevation = -15.0
    cam.lookat[:] = [0.0, 0.0, 0.35]
    return cam


def save_keyframe_grid(sim: K1Sim, samples, output_path, num_cols=6):
    """samples: list of (label, traj). One row per sample, num_cols snapshots
    with time labels."""
    import matplotlib.pyplot as plt

    model, data = sim.model, sim.data
    print(f"\n[viz] rendering keyframe grid → {output_path}")
    renderer = mujoco.Renderer(model, 400, 400)
    cam = _free_camera(model)

    rows = len(samples)
    fig, axes = plt.subplots(rows, num_cols, figsize=(3 * num_cols, 3 * rows))
    fig.patch.set_facecolor("white")
    if rows == 1:
        axes = axes[None, :]

    for r, (label, traj) in enumerate(samples):
        idxs = np.linspace(0, len(traj) - 1, num_cols, dtype=int)
        for col, fi in enumerate(idxs):
            data.qpos[:] = traj[fi]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            ax = axes[r, col]
            ax.imshow(renderer.render())
            ax.axis("off")
            tlabel = f"t = {fi * DT:.2f}s"
            ax.set_title(f"{label}\n{tlabel}" if col == 0 else tlabel,
                         fontsize=10, fontweight="bold" if col == 0 else "normal",
                         loc="left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    renderer.close()


def save_motion_video(sim: K1Sim, traj, output_path, fps=FPS):
    import imageio.v2 as imageio

    model, data = sim.model, sim.data
    print(f"[viz] saving video → {output_path}")
    renderer = mujoco.Renderer(model, 640, 480)
    cam = _free_camera(model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    for q in traj:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
    writer.close()
    renderer.close()


def replay(sim: K1Sim, samples, speed=0.25):
    import time

    import mujoco.viewer

    model, data = sim.model, sim.data
    delay = (1.0 / FPS) / speed
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        while viewer.is_running():
            for label, traj in samples:
                print(f"[viz] replaying {label} @ {speed}x")
                for q in traj:
                    if not viewer.is_running():
                        return
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(delay)
                time.sleep(0.7)


# ─── main ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=int, default=25,
                    help="motions generated per type (default 25)")
    ap.add_argument("--types", nargs="+", default=["back", "belly"],
                    choices=["back", "belly", "side_left", "side_right"],
                    help="fallen-start types to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-viewer", action="store_true",
                    help="skip the slow-motion passive-viewer replay")
    ap.add_argument("--no-grid", action="store_true",
                    help="skip the offscreen keyframe-grid render")
    ap.add_argument("--video", action="store_true",
                    help="also save an MP4 of one sample motion")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = K1Sim(XML_PATH)
    print(f"[create_standup_amp] sim_dt={sim.sim_dt}  AMP_OBS_DIM={AMP_OBS_DIM}")

    all_transitions = []
    grid_samples = []
    for mt in args.types:
        kept = 0
        first_traj = None
        for _ in range(args.variants):
            traj = make_motion(sim, mt, rng)
            # sanity filter: must finish reasonably upright + tall (a clean clip)
            qf = traj[-1]
            upright = 1.0 - 2.0 * (qf[4] ** 2 + qf[5] ** 2)
            if not (qf[2] > 0.45 and upright > 0.9):
                continue
            all_transitions.append(trajectory_to_amp(sim, traj))
            kept += 1
            if first_traj is None:
                first_traj = traj
        print(f"[create_standup_amp] type '{mt}': kept {kept}/{args.variants} clips")
        if first_traj is not None:
            grid_samples.append((mt, first_traj))

    if not all_transitions:
        raise SystemExit("[create_standup_amp] no clips passed the sanity filter!")

    transitions = np.concatenate(all_transitions, axis=0)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, transitions=transitions)
    print(f"[create_standup_amp] saved {transitions.shape} → {OUTPUT_PATH}")

    if not args.no_grid and grid_samples:
        save_keyframe_grid(sim, grid_samples, IMAGE_OUTPUT_PATH)
    if args.video and grid_samples:
        save_motion_video(sim, grid_samples[0][1], VIDEO_OUTPUT_PATH)
    if not args.no_viewer and grid_samples:
        try:
            replay(sim, grid_samples)
        except Exception as e:  # headless / no display
            print(f"[viz] viewer unavailable ({type(e).__name__}: {e}); skipping replay")


if __name__ == "__main__":
    main()
