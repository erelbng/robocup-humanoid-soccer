# LAFAN1 → K1 retargeting → AMP reference

Reproduces `data/motions/k1_lafan1_walk_amp.npz`, the AMP reference-motion
dataset used by the walk skill (`--amp-motion-file`). Pipeline: human mocap →
(GMR retarget) → K1 joint trajectory → (FK + feature extraction) → 53-dim
`build_amp_obs` transitions.

## Prerequisites

- **GMR** (General Motion Retargeting, https://github.com/YanjieZe/GMR) — it
  supports Booster K1 natively, CPU/headless. Clone it, make a venv, and
  `pip install -e .` plus `pip install mink torch` (CPU torch is fine).
- **LAFAN1** BVH clips (free): `lafan1.zip` from
  https://github.com/ubisoft/ubisoft-laforge-animation-dataset — unzip to get
  `walk1_subject1.bvh`, etc.

## Steps

1. **Register K1 for LAFAN1 BVH in GMR.** GMR ships `smplx_to_k1.json` but no
   LAFAN1→K1 config. `gen_k1_ik_config.py` writes `bvh_lafan1_to_k1.json`
   (also committed here) — it uses the LAFAN1-calibrated quaternions from
   `bvh_lafan1_to_t1_29dof.json` mapped onto K1 body names (T1 and K1 are both
   Booster robots, so the frame conventions transfer; the SMPL-X quats did
   NOT and produced arms-overhead garbage). Copy `bvh_lafan1_to_k1.json` into
   `GMR/general_motion_retargeting/ik_configs/`, and register `booster_k1` in
   GMR's `params.py` (`bvh_lafan1` table) + `scripts/bvh_to_robot.py` choices.

   **Verify the retarget visually** before trusting it — render with GMR's
   `--record_video` and check it looks like upright walking with arms down.

2. **Retarget → raw K1 qpos.** `retarget_lafan1.py <bvh>` runs GMR headless
   (no viewer) and saves `/tmp/k1_walk_qpos.npz` = per-frame `[root_pos(3),
   root_quat wxyz(4), dof(22)]`. It also dumps GMR's joint order — verified
   identical to `K1RobotConfig.joint_names` (no remap needed).

3. **Build the AMP reference.** `build_amp_reference.py` (run in the PROJECT
   venv) resamples 30→50 Hz (the control rate — else `dof_vel` scale
   mismatches), computes per-foot clearance via MuJoCo forward-kinematics on
   the K1 model, builds 53-dim `build_amp_obs` features, and saves
   `/tmp/k1_walk_amp.npz` → copy to `data/motions/`.

## Notes / gotchas

- **Foot clearance is RELATIVE** (height above each sim's standing foot height)
  so the MuJoCo-FK reference aligns with the Genesis policy obs despite the
  sims' differing foot-link-frame offsets.
- Joint order, quat convention (wxyz), and absolute (not default-relative)
  joint angles must match `skills/walk/env.py:amp_observation` exactly, or the
  discriminator separates on the artifact instead of the gait.
- To enrich the reference, retarget more clips (walks/runs) and concatenate the
  transition arrays.
