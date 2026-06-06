"""
Generate a simple stand-up AMP motion using Mink IK.
Fixes kinematic over-stretching and orientation lock.
Includes physical settling, slerp rotations, slow-motion replay, and an offscreen visual frame generator.
"""

import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import mink
except ImportError:
    print("[error] pip install mink")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[error] pip install matplotlib (required for saving the image grid)")
    sys.exit(1)

from skills.common_obs import body_frame_velocity, projected_gravity
from training.algorithms.amp import AMP_OBS_DIM, build_amp_obs

# ---------------------------------------------------------------------

XML_PATH = os.path.join(
    PROJECT_ROOT,
    "models/robot/K1/K1_22dof.xml",
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data/motions/k1_standup_mink.npz",
)

IMAGE_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data/motions/k1_standup_keyframes.png",
)

DT = 0.02
FPS = 50
USE_VIEWER = True

# ---------------------------------------------------------------------


def lerp(a, b, alpha):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (1.0 - alpha) * a + alpha * b


def slerp(q0, q1, alpha):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.sum(q0 * q1)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        res = (1.0 - alpha) * q0 + alpha * q1
        return res / np.linalg.norm(res)

    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0) + (s1 * q1)


def make_se3(position, quat_wxyz=None):
    position = np.asarray(position, dtype=np.float64)
    if quat_wxyz is None:
        rot = mink.SO3.identity()
    else:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
        quat_wxyz /= np.linalg.norm(quat_wxyz)
        rot = mink.SO3(quat_wxyz)
    return mink.SE3.from_rotation_and_translation(rot, position)


def quat_ang_vel(qc, qn, dt):
    w0, x0, y0, z0 = qc.T
    w1, x1, y1, z1 = qn.T

    dw = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
    dx = x1 * w0 - w1 * x0 - y1 * z0 + z1 * y0
    dy = y1 * w0 + w1 * y0 + x1 * z0 - z1 * x0
    dz = z1 * w0 - w1 * z0 - x1 * y0 + y1 * x0

    s = np.sign(dw)
    s[s == 0] = 1.0
    return 2.0 * np.stack([dx * s, dy * s, dz * s], axis=1) / dt


# ---------------------------------------------------------------------


def generate_trajectory(model, configuration, tasks, keyframes, viewer=None):
    num_frames = int(keyframes[-1]["t"] * FPS)
    qpos_traj = []

    trunk_task, l_foot_task, r_foot_task, l_hand_task, r_hand_task = tasks
    data = configuration.data

    # --- PHASE 1: DROP AND SETTLE ---
    # Manually place robot slightly above floor and let it fall
    data.qpos[:3] = [0, 0, 0.5]  # Spawn high
    data.qpos[3:7] = [1, 0, 0, 0]  # Upright initially
    mujoco.mj_forward(model, data)

    print("Settling robot to ground...")
    for _ in range(500):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()

    # --- PHASE 2: CAPTURE SETTLED POSE ---
    # This is the REAL t=0.0 configuration
    settled_qpos = data.qpos.copy()
    configuration.update(settled_qpos)

    # --- PHASE 3: INTERPOLATE FROM SETTLED POSE ---
    # Update first keyframe to match reality
    keyframes[0]["trunk_pos"] = data.qpos[:3].copy()
    keyframes[0]["trunk_quat"] = data.qpos[3:7].copy()

    print(f"Generating trajectory from settled pose at {keyframes[0]['trunk_pos']}...")

    for i in range(num_frames):
        t = i * DT
        for k in range(len(keyframes) - 1):
            curr_k = keyframes[k]
            next_k = keyframes[k + 1]

            if curr_k["t"] <= t <= next_k["t"]:
                alpha = (t - curr_k["t"]) / (next_k["t"] - curr_k["t"])

                # Use Slerp for rotation, Lerp for position
                tp = lerp(curr_k["trunk_pos"], next_k["trunk_pos"], alpha)
                tq = slerp(curr_k["trunk_quat"], next_k["trunk_quat"], alpha)

                # Apply targets
                trunk_task.set_target(make_se3(tp, tq))
                l_foot_task.set_target(
                    make_se3(lerp(curr_k["l_foot_pos"], next_k["l_foot_pos"], alpha))
                )
                r_foot_task.set_target(
                    make_se3(lerp(curr_k["r_foot_pos"], next_k["r_foot_pos"], alpha))
                )
                l_hand_task.set_target(
                    make_se3(lerp(curr_k["l_hand_pos"], next_k["l_hand_pos"], alpha))
                )
                r_hand_task.set_target(
                    make_se3(lerp(curr_k["r_hand_pos"], next_k["r_hand_pos"], alpha))
                )
                break

        vel = mink.solve_ik(configuration, tasks, DT, "daqp", 1e-3)
        configuration.integrate_inplace(vel, DT)
        if viewer is not None:
            viewer.sync()
        qpos_traj.append(configuration.q.copy())

    return np.asarray(qpos_traj)


def process_transitions(model, data, qpos_traj):
    M = len(qpos_traj)
    root_pos = qpos_traj[:, :3]
    root_quat = qpos_traj[:, 3:7]
    dof_pos = qpos_traj[:, 7:]
    dof_vel = np.zeros_like(dof_pos)
    dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / DT
    dof_vel[-1] = dof_vel[-2]

    ang_vel_world = np.zeros((M, 3))
    ang_vel_world[:-1] = quat_ang_vel(root_quat[:-1], root_quat[1:], DT)
    ang_vel_world[-1] = ang_vel_world[-2]

    l_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    r_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")

    foot_clear = np.zeros((M, 2))
    for i in range(M):
        data.qpos[:] = qpos_traj[i]
        mujoco.mj_forward(model, data)
        foot_clear[i] = [
            data.xpos[l_foot_id][2] - 0.02,
            data.xpos[r_foot_id][2] - 0.02,
        ]

    foot_clear = np.clip(foot_clear, 0.0, 0.5)
    amp_obs = np.zeros((M, AMP_OBS_DIM), dtype=np.float32)

    for i in range(M):
        q = root_quat[i : i + 1].astype(np.float32)
        ang_b = body_frame_velocity(q, ang_vel_world[i : i + 1].astype(np.float32))

        amp_obs[i] = build_amp_obs(
            root_pos[i, 2],
            projected_gravity(q),
            ang_b,
            dof_pos[i : i + 1].astype(np.float32),
            dof_vel[i : i + 1].astype(np.float32),
            foot_clear[i : i + 1].astype(np.float32),
        )

    return np.concatenate([amp_obs[:-1], amp_obs[1:]], axis=1).astype(np.float32)


def save_keyframe_grid(model, data, traj_belly, traj_back, output_path, num_cols=6):
    print(f"\n[mink] Rendering offscreen keyframe grid to {output_path}...")
    renderer = mujoco.Renderer(model, 400, 400)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 2.5
    cam.azimuth = -45.0
    cam.elevation = -15.0
    cam.lookat[:] = [0.0, 0.0, 0.3]

    fig, axes = plt.subplots(2, num_cols, figsize=(18, 6))
    fig.patch.set_facecolor("white")

    def render_row(traj, row_idx, title):
        indices = np.linspace(0, len(traj) - 1, num_cols, dtype=int)
        for col_idx, traj_idx in enumerate(indices):
            data.qpos[:] = traj[traj_idx]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            img = renderer.render()

            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")

            time_sec = traj_idx * DT
            if col_idx == 0:
                ax.set_title(
                    f"{title}\nt = {time_sec:.2f}s",
                    fontsize=12,
                    fontweight="bold",
                    loc="left",
                )
            else:
                ax.set_title(f"t = {time_sec:.2f}s", fontsize=10)

    render_row(traj_belly, 0, "Belly-to-Stand")
    render_row(traj_back, 1, "Back-to-Stand")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    renderer.close()


def replay_trajectory(model, data, viewer, qpos_traj, fps, speed_factor=0.25):
    delay = (1.0 / fps) / speed_factor
    for q in qpos_traj:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(delay)


# ---------------------------------------------------------------------


def create_motion():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    configuration = mink.Configuration(model)
    data = configuration.data

    print("nq =", model.nq)
    print("nv =", model.nv)

    # STRICT POSITIONING, ZERO ORIENTATION COST ON LIMBS
    trunk_task = mink.FrameTask(
        frame_name="Trunk", frame_type="body", position_cost=1.0, orientation_cost=2.0
    )
    l_foot_task = mink.FrameTask(
        frame_name="left_foot_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
    )
    r_foot_task = mink.FrameTask(
        frame_name="right_foot_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
    )
    l_hand_task = mink.FrameTask(
        frame_name="left_hand_link",
        frame_type="body",
        position_cost=3.0,
        orientation_cost=0.0,
    )
    r_hand_task = mink.FrameTask(
        frame_name="right_hand_link",
        frame_type="body",
        position_cost=3.0,
        orientation_cost=0.0,
    )

    tasks = [trunk_task, l_foot_task, r_foot_task, l_hand_task, r_hand_task]

    # Max standing Trunk Z is ~0.52m based on K1 kinematics.
    # Max leg reach is ~0.43m. Max hand reach is ~0.30m.

    keyframes_belly = [
        # Prone. Limbs well within physical reach.
        {
            "t": 0.0,
            "trunk_pos": [0.0, 0.0, 0.12],
            "trunk_quat": [0.707, 0.0, 0.707, 0.0],
            "l_foot_pos": [-0.40, 0.10, 0.03],
            "r_foot_pos": [-0.40, -0.10, 0.03],
            "l_hand_pos": [0.15, 0.25, 0.03],
            "r_hand_pos": [0.15, -0.25, 0.03],
        },
        # Push-up Torso
        {
            "t": 1.5,
            "trunk_pos": [-0.10, 0.0, 0.30],
            "trunk_quat": [0.866, 0.0, 0.500, 0.0],
            "l_foot_pos": [-0.40, 0.10, 0.03],
            "r_foot_pos": [-0.40, -0.10, 0.03],
            "l_hand_pos": [0.15, 0.25, 0.03],
            "r_hand_pos": [0.15, -0.25, 0.03],
        },
        # Knees forward (Squat entry)
        {
            "t": 3.0,
            "trunk_pos": [-0.15, 0.0, 0.35],
            "trunk_quat": [0.965, 0.0, 0.258, 0.0],
            "l_foot_pos": [-0.15, 0.10, 0.03],
            "r_foot_pos": [-0.15, -0.10, 0.03],
            "l_hand_pos": [0.15, 0.25, 0.03],
            "r_hand_pos": [0.15, -0.25, 0.03],
        },
        # Deep Squat (Hands lift off)
        {
            "t": 4.5,
            "trunk_pos": [-0.05, 0.0, 0.40],
            "trunk_quat": [1.0, 0.0, 0.0, 0.0],
            "l_foot_pos": [-0.05, 0.10, 0.03],
            "r_foot_pos": [-0.05, -0.10, 0.03],
            "l_hand_pos": [0.15, 0.25, 0.25],
            "r_hand_pos": [0.15, -0.25, 0.25],
        },
        # Stand Upright (Z=0.52m)
        {
            "t": 6.0,
            "trunk_pos": [0.0, 0.0, 0.52],
            "trunk_quat": [1.0, 0.0, 0.0, 0.0],
            "l_foot_pos": [0.0, 0.10, 0.03],
            "r_foot_pos": [0.0, -0.10, 0.03],
            "l_hand_pos": [0.0, 0.25, 0.35],
            "r_hand_pos": [0.0, -0.25, 0.35],
        },
    ]

    keyframes_back = [
        # Supine. Limbs well within physical reach.
        {
            "t": 0.0,
            "trunk_pos": [0.0, 0.0, 0.12],
            "trunk_quat": [0.707, 0.0, -0.707, 0.0],
            "l_foot_pos": [0.40, 0.10, 0.03],
            "r_foot_pos": [0.40, -0.10, 0.03],
            "l_hand_pos": [-0.15, 0.25, 0.03],
            "r_hand_pos": [-0.15, -0.25, 0.03],
        },
        # Sit-up
        {
            "t": 1.5,
            "trunk_pos": [-0.05, 0.0, 0.30],
            "trunk_quat": [1.0, 0.0, 0.0, 0.0],
            "l_foot_pos": [0.25, 0.10, 0.03],
            "r_foot_pos": [0.25, -0.10, 0.03],
            "l_hand_pos": [-0.20, 0.25, 0.03],
            "r_hand_pos": [-0.20, -0.25, 0.03],
        },
        # Lean Forward / Bridge
        {
            "t": 3.0,
            "trunk_pos": [0.10, 0.0, 0.35],
            "trunk_quat": [0.965, 0.0, 0.258, 0.0],
            "l_foot_pos": [0.10, 0.10, 0.03],
            "r_foot_pos": [0.10, -0.10, 0.03],
            "l_hand_pos": [-0.20, 0.25, 0.03],
            "r_hand_pos": [-0.20, -0.25, 0.03],
        },
        # Deep Squat (Hands lift off)
        {
            "t": 4.5,
            "trunk_pos": [0.05, 0.0, 0.40],
            "trunk_quat": [1.0, 0.0, 0.0, 0.0],
            "l_foot_pos": [0.05, 0.10, 0.03],
            "r_foot_pos": [0.05, -0.10, 0.03],
            "l_hand_pos": [0.10, 0.25, 0.25],
            "r_hand_pos": [0.10, -0.25, 0.25],
        },
        # Stand Upright (Z=0.52m)
        {
            "t": 6.0,
            "trunk_pos": [0.0, 0.0, 0.52],
            "trunk_quat": [1.0, 0.0, 0.0, 0.0],
            "l_foot_pos": [0.0, 0.10, 0.03],
            "r_foot_pos": [0.0, -0.10, 0.03],
            "l_hand_pos": [0.0, 0.25, 0.35],
            "r_hand_pos": [0.0, -0.25, 0.35],
        },
    ]

    viewer = None
    if USE_VIEWER:
        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )

    print("\n--- Generating Belly-to-Stand ---")
    traj_belly = generate_trajectory(
        model, configuration, tasks, keyframes_belly, viewer
    )
    trans_belly = process_transitions(model, data, traj_belly)

    print("\n--- Generating Back-to-Stand ---")
    traj_back = generate_trajectory(model, configuration, tasks, keyframes_back, viewer)
    trans_back = process_transitions(model, data, traj_back)

    save_keyframe_grid(model, data, traj_belly, traj_back, IMAGE_OUTPUT_PATH)

    all_transitions = np.concatenate([trans_belly, trans_back], axis=0)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, transitions=all_transitions)
    print(f"\n[mink] Saved {all_transitions.shape[0]} transitions to {OUTPUT_PATH}")

    if viewer is not None:
        print("\n[mink] Replaying Belly-to-Stand at 0.25x speed...")
        replay_trajectory(model, data, viewer, traj_belly, FPS, speed_factor=0.25)
        time.sleep(1.0)
        print("\n[mink] Replaying Back-to-Stand at 0.25x speed...")
        replay_trajectory(model, data, viewer, traj_back, FPS, speed_factor=0.25)
        time.sleep(2.0)
        viewer.close()


if __name__ == "__main__":
    create_motion()
