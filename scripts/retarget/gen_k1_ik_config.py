"""Rebuild bvh_lafan1_to_k1.json using the LAFAN1-calibrated quaternions from
bvh_lafan1_to_t1_29dof.json (correct human-source frames) mapped onto K1 body
names. The first attempt reused smplx_to_k1 quats (SMPL-X source) → arms stuck
overhead + crouch. T1 is the same manufacturer (Booster), so its LAFAN1 frame
calibration should transfer to K1 far better."""
import json, os
CFG = "/tmp/GMR/general_motion_retargeting/ik_configs"

Q_ROOT = [0.5, 0.5, 0.5, 0.5]
Q_LEG = [-0.5, 0.5, 0.5, -0.5]
Q_FOOT = [-0.70710678, 0.70710678, 0.0, 0.0]
Q_LARM = [0.70710678, 0.0, 0.0, -0.70710678]
Q_RARM = [0.0, -0.70710678, 0.70710678, 0.0]
O = [0.0, 0.0, 0.0]

# robot_body: (human_joint, quat, [pw1,rw1], [pw2,rw2])  — weights per table
M = {
    "Trunk":          ("Hips",        Q_ROOT, [0, 10], [10, 5]),
    "Left_Hip_Yaw":   ("LeftUpLeg",   Q_LEG,  [0, 10], [10, 5]),
    "Right_Hip_Yaw":  ("RightUpLeg",  Q_LEG,  [0, 10], [10, 5]),
    "Left_Shank":     ("LeftLeg",     Q_LEG,  [0, 10], [10, 5]),
    "Right_Shank":    ("RightLeg",    Q_LEG,  [0, 10], [10, 5]),
    "left_foot_link": ("LeftFootMod", Q_FOOT, [100, 50], [100, 50]),
    "right_foot_link":("RightFootMod",Q_FOOT, [100, 50], [100, 50]),
    "Left_Arm_3":     ("LeftArm",     Q_LARM, [0, 10], [5, 10]),
    "Right_Arm_3":    ("RightArm",    Q_RARM, [0, 10], [5, 10]),
    "left_hand_link": ("LeftHand",    Q_LARM, [0, 10], [10, 5]),
    "right_hand_link":("RightHand",   Q_RARM, [0, 10], [10, 5]),
    "Head_2":         ("Head",        Q_ROOT, [0, 10], [0, 10]),
}
scale = {"Hips": 0.6, "LeftUpLeg": 0.6, "RightUpLeg": 0.6, "LeftLeg": 0.6,
         "RightLeg": 0.6, "LeftFootMod": 0.6, "RightFootMod": 0.6,
         "LeftArm": 0.7, "RightArm": 0.7, "LeftHand": 0.7, "RightHand": 0.7,
         "Head": 0.6}

def table(idx):
    t = {}
    for rb, (hj, q, w1, w2) in M.items():
        w = w1 if idx == 1 else w2
        t[rb] = [hj, w[0], w[1], O, q]
    return t

out = {
    "robot_root_name": "Trunk", "human_root_name": "Hips",
    "ground_height": 0.0, "human_height_assumption": 1.8,
    "use_ik_match_table1": True, "use_ik_match_table2": True,
    "human_scale_table": scale,
    "ik_match_table1": table(1), "ik_match_table2": table(2),
}
json.dump(out, open(os.path.join(CFG, "bvh_lafan1_to_k1.json"), "w"), indent=2)
print("rewrote bvh_lafan1_to_k1.json with T1-LAFAN1 quats")
