"""Convert retargeted K1 qpos (/tmp/k1_walk_qpos.npz, 30 fps) into a 53-dim AMP
reference (build_amp_obs incl. per-foot CLEARANCE) at 50 Hz. Foot heights come
from MuJoCo forward-kinematics on the K1 model. Run in the PROJECT venv."""
import numpy as np, mujoco, os
from configs.config import K1RobotConfig
from skills.common_obs import projected_gravity, body_frame_velocity
from training.algorithms.amp import build_amp_obs, AMP_OBS_DIM

SRC_FPS, DT = 30.0, 1.0/50.0
XML = os.path.join(os.path.dirname(__file__), "..", "models", "robot", "K1", "K1_22dof.xml")
XML = "/home/ric/htwk-robots/robocup_humanoid_soccer/models/robot/K1/K1_22dof.xml"

d = np.load("/tmp/k1_walk_qpos.npz", allow_pickle=True)
qpos = d["qpos"].astype(np.float64)              # (N,29): pos3, quat4(wxyz), dof22
N = qpos.shape[0]
root_pos, quat, dof = qpos[:, :3], qpos[:, 3:7], qpos[:, 7:]
quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
for i in range(1, N):
    if np.dot(quat[i], quat[i-1]) < 0: quat[i] = -quat[i]

# resample 30->50 fps
t_src = np.arange(N)/SRC_FPS; t_new = np.arange(0.0, t_src[-1], DT)
def interp(c): return np.stack([np.interp(t_new, t_src, c[:,k]) for k in range(c.shape[1])],1)
root_pos, dof = interp(root_pos), interp(dof)
quat = interp(quat); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
M = len(t_new); print("resampled frames:", M)

# ── MuJoCo FK for per-foot height ──
m = mujoco.MjModel.from_xml_path(os.path.abspath(XML)); data = mujoco.MjData(m)
cfg = K1RobotConfig()
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in cfg.joint_names]
base_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")]
foot_b = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_foot_link","right_foot_link")]
foot_z = np.zeros((M, 2))
for i in range(M):
    data.qpos[base_q:base_q+3] = root_pos[i]
    data.qpos[base_q+3:base_q+7] = quat[i]
    for k, a in enumerate(qadr): data.qpos[a] = dof[i, k]
    mujoco.mj_forward(m, data)
    foot_z[i] = [data.xpos[foot_b[0]][2], data.xpos[foot_b[1]][2]]
ground = foot_z.min(0)                            # per-foot lowest = ground contact
foot_clear = np.clip(foot_z - ground, 0.0, 0.5)
print("foot ground z:", np.round(ground,3), " clearance max:", np.round(foot_clear.max(0),3),
      " swing frac (>0.03):", np.round((foot_clear>0.03).mean(0),2))

# ── angular velocity + dof velocity ──
def quat_ang_vel(qc, qn, dt):
    w0,x0,y0,z0=qc.T; w1,x1,y1,z1=qn.T; cw,cx,cy,cz=w0,-x0,-y0,-z0
    dw=w1*cw-x1*cx-y1*cy-z1*cz; dx=w1*cx+x1*cw+y1*cz-z1*cy
    dy=w1*cy-x1*cz+y1*cw+z1*cx; dz=w1*cz+x1*cy-y1*cx+z1*cw
    s=np.sign(dw); s[s==0]=1
    return 2.0*np.stack([dx*s,dy*s,dz*s],1)/dt
ang_world=np.zeros((M,3)); ang_world[:-1]=quat_ang_vel(quat[:-1],quat[1:],DT); ang_world[-1]=ang_world[-2]
dof_vel=np.zeros_like(dof); dof_vel[:-1]=(dof[1:]-dof[:-1])/DT; dof_vel[-1]=dof_vel[-2]

amp=np.zeros((M,AMP_OBS_DIM),np.float32)
for i in range(M):
    qg=quat[i:i+1].astype(np.float32)
    ang_b=body_frame_velocity(qg, ang_world[i:i+1].astype(np.float32))
    amp[i]=build_amp_obs(root_pos[i,2], projected_gravity(qg), ang_b,
                         dof[i:i+1].astype(np.float32), dof_vel[i:i+1].astype(np.float32),
                         foot_clear[i:i+1].astype(np.float32))
trans=np.concatenate([amp[:-1],amp[1:]],axis=1).astype(np.float32)
print("transitions:", trans.shape, "(AMP_OBS_DIM=%d)"%AMP_OBS_DIM)
np.savez("/tmp/k1_walk_amp.npz", transitions=trans)
print("saved /tmp/k1_walk_amp.npz")
