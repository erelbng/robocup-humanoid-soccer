"""Standup sim2sim eval — does a Genesis-trained policy actually STAND in MuJoCo?

Genesis training reports trunk height, but a policy can reach `z≈0.54` there by
leaning on the HoST assist force (a training-only upward crane) and/or on
Genesis-specific contact dynamics. The real test of a deployable get-up is
whether it stands in a *different* simulator **with no assist force**. This
script rolls each checkpoint out in MuJoCo from each named fallen pose and
reports, per pose:

  * stood   — did it reach a SUSTAINED stand (upright + tall + feet-under-base
              held for `success_hold_steps`)?  ← the headline transfer metric
  * max_z   — highest trunk height reached (m)
  * end_z   — mean trunk height over the last second (m); ~0.30 = crouch-stall
  * end_up  — mean upright cos-tilt over the last second (1 = vertical)
  * t_stand — control step of first sustained stand (— if never)

It reuses the EXACT rollout pipeline validated against training: the helpers in
`render_presentation` apply the same per-joint K1 PD gains, the same residual-
action convention (`clip(Δ, ±0.5)` + default pose), and the same dt /
action_repeat. No rendering happens (physics only), so no GL backend is needed.

Usage (run where torch + mujoco live, e.g. the training box):

    python scripts/eval_standup_sim2sim.py checkpoints/skill_standup            # latest in dir
    python scripts/eval_standup_sim2sim.py path/to/skill_standup_step*.pt       # explicit ckpts
    python scripts/eval_standup_sim2sim.py <dir> --poses prone supine --trials 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _p in (_ROOT, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The faithful MuJoCo rollout pipeline (scene build, pose set, obs, PD step).
import render_presentation as rp  # noqa: E402

from skills.standup import rewards as R  # noqa: E402
from skills.standup.config import StandupConfig  # noqa: E402
from training.common import create_policy, load_checkpoint  # noqa: E402


def _foot_zxy(data, idx):
    """(foot_z (1,2), foot_xy (1,2,2)) world-frame from MuJoCo body xpos."""
    fz = np.zeros((1, 2), dtype=np.float32)
    fxy = np.zeros((1, 2, 2), dtype=np.float32)
    for i, bid in enumerate(idx["foot_bid"]):
        if bid >= 0:
            fz[0, i] = data.xpos[bid, 2]
            fxy[0, i, :] = data.xpos[bid, :2]
    return fz, fxy


def _pose_target_vec(pose, idx):
    """Canonical-order joint target vector for a StandupPose (for settle PD)."""
    return np.array([pose.joint_targets.get(n, 0.0) for n in idx["names"]],
                    dtype=np.float64)


def rollout(policy, obs_norm, model, data, idx, pose, cfg, *,
            steps=250, settle=25, jitter=0.05, rng=None):
    """One MuJoCo episode from `pose`. Returns a per-step metrics dict.

    A short `settle` phase (PD-holding the pose's joint targets) lets the robot
    rest on the floor first — matching training, which resets from physics-
    settled pool states rather than the raw reference pose."""
    import torch

    rng = rng or np.random.default_rng()
    mujoco = rp.mujoco
    mujoco.mj_resetData(model, data)
    rp.set_pose(data, idx, pose)
    if jitter > 0.0:
        data.qpos[idx["qpos"]] += rng.normal(0.0, jitter, size=len(idx["qpos"]))
    mujoco.mj_forward(model, data)

    pose_vec = _pose_target_vec(pose, idx)
    for _ in range(settle):
        rp.apply_pd(model, data, idx, pose_vec)

    obs_dim = rp._BASE_OBS_DIM
    last_action = np.zeros(idx["default"].shape[0], dtype=np.float32)
    zs, ups, feet_ok = [], [], []
    for t in range(steps):
        obs = rp.standup_obs(data, idx, last_action, t)
        nobs = obs_norm.normalize(obs[None])[0] if obs_norm is not None else obs
        with torch.no_grad():
            out = policy.act(torch.as_tensor(nobs[None], dtype=torch.float32),
                             deterministic=True)
        action = np.asarray(rp._action_from(out)).reshape(-1)[:last_action.shape[0]]
        last_action = action.astype(np.float32)
        target = np.clip(
            idx["default"] + np.clip(action, -rp._ACTION_DELTA_MAX,
                                     rp._ACTION_DELTA_MAX),
            -np.pi, np.pi)
        rp.apply_pd(model, data, idx, target)

        quat = data.qpos[3:7][None, :].astype(np.float32)
        z = float(data.qpos[2])
        fz, fxy = _foot_zxy(data, idx)
        ok = R.standing_on_feet_mask(
            fz, fxy, data.qpos[0:2][None, :].astype(np.float32),
            foot_max_z=cfg.success_foot_max_z,
            under_base_max_d=cfg.success_under_base_max_d)[0]
        zs.append(z)
        ups.append(float(R.upright_signal(quat)[0]))
        feet_ok.append(bool(ok))

    zs = np.asarray(zs)
    ups = np.asarray(ups)
    feet_ok = np.asarray(feet_ok)

    # Per-frame "looks standing" at the FINAL curriculum thresholds, gated on
    # feet-under-base — identical to the training success detector.
    frame_ok = ((ups > cfg.upright_threshold)
                & (zs > cfg.target_height - 0.10)
                & feet_ok)
    # First step at which `success_hold_steps` consecutive frames are ok.
    hold = cfg.success_hold_steps
    t_stand = -1
    run = 0
    for t, f in enumerate(frame_ok):
        run = run + 1 if f else 0
        if run >= hold:
            t_stand = t - hold + 1
            break
    tail = max(1, int(round(1.0 / cfg.dt)))  # last ~1 s
    return dict(
        stood=t_stand >= 0,
        t_stand=t_stand,
        max_z=float(zs.max()),
        end_z=float(zs[-tail:].mean()),
        end_up=float(ups[-tail:].mean()),
        frac_ok=float(frame_ok.mean()),
    )


def _ckpt_step(path):
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_checkpoints(arg, latest_only):
    paths = []
    for a in arg:
        if os.path.isdir(a):
            paths += glob.glob(os.path.join(a, "*.pt"))
        else:
            paths += glob.glob(a)
    paths = sorted(set(paths), key=_ckpt_step)
    if latest_only and paths:
        paths = paths[-1:]
    return paths


def eval_checkpoint(ckpt, poses_list, cfg, *, steps, trials, settle, jitter,
                    seed):
    obs_dim, obs_norm_sd = rp._ckpt_obs_dim(ckpt)
    if obs_dim != rp._BASE_OBS_DIM:
        print(f"  [skip] {os.path.basename(ckpt)}: obs_dim {obs_dim} "
              f"!= {rp._BASE_OBS_DIM} (not a contact-obs standup ckpt)")
        return None
    model, data, idx = rp.build_scene()
    policy = create_policy(obs_dim, idx["default"].shape[0])
    if policy is None:
        print("  [skip] create_policy returned None")
        return None
    load_checkpoint(ckpt, policy)
    policy.eval()
    obs_norm = None
    if obs_norm_sd is not None:
        from training.normalizers import RunningMeanStd
        obs_norm = RunningMeanStd(shape=(obs_dim,))
        obs_norm.load_state_dict(obs_norm_sd)

    rng = np.random.default_rng(seed)
    results = {}
    for pose_fn in poses_list:
        pose = pose_fn()
        runs = [rollout(policy, obs_norm, model, data, idx, pose, cfg,
                        steps=steps, settle=settle, jitter=jitter, rng=rng)
                for _ in range(trials)]
        results[pose.name] = dict(
            stood_rate=float(np.mean([r["stood"] for r in runs])),
            max_z=float(np.mean([r["max_z"] for r in runs])),
            end_z=float(np.mean([r["end_z"] for r in runs])),
            end_up=float(np.mean([r["end_up"] for r in runs])),
            frac_ok=float(np.mean([r["frac_ok"] for r in runs])),
        )
    return results


_POSE_FNS = {
    "prone": lambda: __import__("envs.standup", fromlist=["prone"]).prone(),
    "supine": lambda: __import__("envs.standup", fromlist=["supine"]).supine(),
    "side_left": lambda: __import__("envs.standup",
                                    fromlist=["side_left"]).side_left(),
    "side_right": lambda: __import__("envs.standup",
                                     fromlist=["side_right"]).side_right(),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="+",
                    help="checkpoint .pt file(s), a glob, or a directory")
    ap.add_argument("--poses", nargs="*", default=list(_POSE_FNS),
                    choices=list(_POSE_FNS))
    ap.add_argument("--steps", type=int, default=250,
                    help="control steps per episode (250 = 5 s at 50 Hz)")
    ap.add_argument("--trials", type=int, default=4,
                    help="episodes per pose (small reset jitter each)")
    ap.add_argument("--settle", type=int, default=25,
                    help="control steps PD-holding the pose before the policy")
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="per-trial joint reset noise (rad), 0 = deterministic")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="evaluate every ckpt found (default: latest only)")
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = StandupConfig()
    poses_list = [_POSE_FNS[p] for p in args.poses]
    ckpts = resolve_checkpoints(args.checkpoint,
                                latest_only=not args.all_checkpoints)
    if not ckpts:
        print("No checkpoints found.")
        return
    print(f"sim2sim standup eval — MuJoCo, NO assist force\n"
          f"  thresholds: upright>{cfg.upright_threshold} z>"
          f"{cfg.target_height - 0.10:.2f} held {cfg.success_hold_steps} steps "
          f"| trials/pose={args.trials} steps={args.steps}\n")

    out = {}
    for ckpt in ckpts:
        print(f"=== {os.path.basename(ckpt)} (step {_ckpt_step(ckpt):,}) ===")
        res = eval_checkpoint(ckpt, poses_list, cfg, steps=args.steps,
                              trials=args.trials, settle=args.settle,
                              jitter=args.jitter, seed=args.seed)
        if res is None:
            continue
        out[ckpt] = res
        print(f"  {'pose':<11} {'stood':>7} {'max_z':>7} {'end_z':>7} "
              f"{'end_up':>7} {'%ok':>6}")
        stood_rates = []
        for name, m in res.items():
            stood_rates.append(m["stood_rate"])
            print(f"  {name:<11} {m['stood_rate']*100:6.0f}% {m['max_z']:7.3f} "
                  f"{m['end_z']:7.3f} {m['end_up']:7.3f} "
                  f"{m['frac_ok']*100:5.0f}%")
        overall = float(np.mean(stood_rates)) if stood_rates else 0.0
        verdict = ("TRANSFERS" if overall > 0.5
                   else "CROUCH-STALL / NO TRANSFER" if overall < 0.1
                   else "PARTIAL")
        print(f"  → overall stood: {overall*100:.0f}%   [{verdict}]\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({os.path.basename(k): v for k, v in out.items()}, f,
                      indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
