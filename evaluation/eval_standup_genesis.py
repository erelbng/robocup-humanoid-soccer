"""Genesis-side standup eval — the cross-check for the MuJoCo sim2sim result.

Runs a frozen policy deterministically in the SAME Genesis env used for
training (same fallen settle-pool init, same control stack, same success
detector), with the assist force OFF and the success criteria forced to
their final/deployment values. This isolates the question raised by the
MuJoCo eval (teacher 0%, only ~halfway up):

  * If the teacher stands ~40-50% HERE → the policy is fine in its native
    sim and the MuJoCo 0% is a genuine Genesis→MuJoCo transfer gap (the
    MuJoCo eval logic is trustworthy).
  * If it ALSO fails here → the failure is in eval logic shared in spirit
    (init / obs / success), not physics — investigate the eval, not sim2sim.

Usage:
    python -m evaluation.eval_standup_genesis \
        checkpoints/skill_standup/skill_standup_step432668672.pt --num-envs 256
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from skills.standup.config import StandupConfig
from skills.standup.env import K1StandupEnv
from skills.common_obs import _to_np, projected_gravity
from training.algorithms.networks import PPOActorCritic
from training.normalizers import RunningMeanStd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--no-dr", action="store_true",
                    help="disable domain randomization + obs noise (cleanest "
                         "policy read; default keeps training-like DR on)")
    args = ap.parse_args()

    cfg = StandupConfig()
    cfg.num_envs = args.num_envs
    cfg.assist_force_enabled = False    # evaluate the UNAIDED policy
    cfg.easy_pool_enabled = False       # all fallen starts (no easy curriculum)

    # Teacher checkpoints are privileged (obs_dim 94); a student/single is 78.
    # Infer from the checkpoint's actor input width.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("policy_state_dict") or ckpt.get("actor_state_dict")
    ckpt_obs_dim = sd["actor_trunk.0.weight"].shape[1]
    include_priv = (ckpt_obs_dim >= 94)

    dr_cfg = None
    if args.no_dr:
        from envs.domain_randomization import DomainRandConfig
        dr_cfg = DomainRandConfig(enabled=False)

    env = K1StandupEnv(cfg=cfg, num_envs=args.num_envs,
                       include_privileged=include_priv, backend="gpu",
                       dr_cfg=dr_cfg)
    obs = env.reset()
    obs_dim, act_dim = env.obs_dim, env.act_dim
    assert obs_dim == ckpt_obs_dim, \
        f"env obs_dim {obs_dim} != checkpoint {ckpt_obs_dim}"

    # Force the success-criteria curriculum to its FINAL (deployment) values,
    # so this matches the MuJoCo eval's strict bar (up>0.92 & z>0.45, hold 1s)
    # rather than the loose early-curriculum values.
    env._total_env_steps_seen = 10 ** 12
    print(f"[gen-eval] hold={env._current_hold_steps()} "
          f"upright_thr={env._current_upright_threshold():.2f} "
          f"target_h={env._current_target_height():.2f} "
          f"(assist OFF, DR {'OFF' if args.no_dr else 'ON'}, obs_dim={obs_dim})")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PPOActorCritic(obs_dim, act_dim).to(dev)
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    on = RunningMeanStd(shape=(obs_dim,))
    if "obs_norm" in ckpt:
        on.load_state_dict(ckpt["obs_norm"])

    # Run one episode (all envs are in lockstep; standup terminates only on
    # timeout). Stop a few steps short of MAX so the auto-reset hasn't cleared
    # the per-env achieved_sustained latch yet.
    try:
        kp_rb = _to_np(env.robot.get_dofs_kp(env.dof_indices))
        print(f"[gen-eval] internal PD kp readback: max={kp_rb.max():.1f} "
              f"(0 ⇒ internal PD disabled, torque control active)")
    except Exception as e:
        print(f"[gen-eval] kp readback failed: {e}")

    steps = env.MAX_EPISODE_STEPS - 3
    ctrl_max, tot_max = 0.0, 0.0
    for _ in range(steps):
        o = on.normalize(obs).astype(np.float32)
        with torch.no_grad():
            a, _, _ = policy.act(torch.as_tensor(o, device=dev),
                                 deterministic=True)
        obs, _, _, _ = env.step(a.cpu().numpy())
        try:
            cf = _to_np(env.robot.get_dofs_control_force(env.dof_indices))
            ctrl_max = max(ctrl_max, float(np.max(np.abs(cf))))
        except Exception:
            pass
        try:
            f = _to_np(env.robot.get_dofs_force(env.dof_indices))
            tot_max = max(tot_max, float(np.max(np.abs(f))))
        except Exception:
            pass
    print(f"[gen-eval] max |actuator/control torque| = {ctrl_max:.0f} Nm "
          f"| max |total dof force| = {tot_max:.0f} Nm  (limits 6-40)")

    achieved = env._achieved_sustained.copy()
    frame_succ = env._frame_success.copy()
    quat = _to_np(env.robot.get_quat())
    pos = _to_np(env.robot.get_pos())
    up = -projected_gravity(quat)[:, 2]

    print(f"\n{'='*54}\n STANDUP GENESIS EVAL — {args.num_envs} envs, deterministic")
    print(f"   achieved_sustained rate : {achieved.mean()*100:.1f}%  "
          f"({int(achieved.sum())}/{args.num_envs})")
    print(f"   frame_success rate (now): {frame_succ.mean()*100:.1f}%")
    print(f"   mean final upright      : {up.mean():.3f}")
    print(f"   mean final height       : {pos[:, 2].mean():.3f}")
    print('='*54)
    env.close()


if __name__ == "__main__":
    main()
