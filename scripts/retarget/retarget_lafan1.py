"""Headless retarget (no viewer) of a LAFAN1 walk clip -> K1, saving the raw
qpos trajectory + the robot joint order. Run in the GMR venv."""
import sys, numpy as np
sys.path.insert(0, "/tmp/GMR")
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
import mujoco

bvh = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lafan1/walk1_subject1.bvh"
frames, height = load_bvh_file(bvh, format="lafan1")
r = GMR(src_human="bvh_lafan1", tgt_robot="booster_k1", actual_human_height=height)

qpos = []
for f in frames:
    q = r.retarget(f)
    qpos.append(np.asarray(q, np.float32))
qpos = np.stack(qpos)                       # (N, 7+ndof)
print("qpos:", qpos.shape)

# robot joint order from the GMR K1 mujoco model (hinge joints, in qpos order)
m = r.robot_model if hasattr(r, "robot_model") else None
names = []
try:
    model = m if m is not None else mujoco.MjModel.from_xml_path(
        "/tmp/GMR/assets/booster_k1/K1_serial.xml")
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
except Exception as e:
    print("joint-name read failed:", e)
print("n hinge joints:", len(names))
print("joint order:", names)
np.savez("/tmp/k1_walk_qpos.npz", qpos=qpos, joint_names=np.array(names, dtype=object))
print("saved /tmp/k1_walk_qpos.npz")
