#!/usr/bin/env python3
"""Render screenshots of the standup pose pools for visual inspection.

Builds the REAL ``K1StandupEnv`` pose pools (identical settle physics +
filters as training), then for each pool samples random states, sets a
single env (env 0, alone at the world origin) to each, and saves an
off-screen render to ``<out_dir>/<pool>/<pool>_NNN.png``.

Each filename embeds two diagnostics so problems are visible at a glance:

  * ``z`` — trunk (base-link) height.
  * ``minc`` — the LOWEST foot/hand link z. For a clean pose this is ≳ 0;
    a negative value means a limb is stuck in the floor (the penetration
    bug). Such frames are additionally prefixed with ``PEN_`` and counted
    in the per-run ``manifest.json``.

This is the visual counterpart to the pose-pool penetration filter
(`pose_pool_penetration_eps`) and the raised `spawn_clearance` — run it to
confirm the back/belly/side starts settle above the carpet.

Usage:
    python -m scripts.eval_pose_pools
    python -m scripts.eval_pose_pools --per-pool 100 --out-dir pose_pool_shots
    python -m scripts.eval_pose_pools --pools supine prone --cpu
    python -m scripts.eval_pose_pools --no-jitter      # show raw pool states
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _to_np(x):
    """Genesis returns torch tensors; standardise to numpy."""
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _render_rgb(camera, scene):
    """Render the off-screen camera → uint8 RGB (H, W, 3).

    Genesis caches the rendered scene in ``visualizer._t`` and only updates
    it when ``scene._t`` changes (i.e. after ``scene.step()``). Since we
    set poses directly via ``robot.set_pos / set_quat / set_dofs_position``
    without stepping physics, ``scene._t`` never advances → every call to
    ``camera.render()`` returns the identical (stale) frame.

    Two ways to flush the render buffer:
      a) ``scene.step()`` — advances _t but moves the pose one physics tick.
      b) ``camera.render(force_render=True)`` — bypasses the _t guard and
         calls ``update_visual_states(force_render=True)`` directly.

    We use (b) so poses are exactly what we set, with no physics drift.
    As a safety fallback we also nudge ``scene._t`` manually in case an
    older Genesis build doesn't honour force_render at the rasterizer level.
    """
    # Nudge the scene timestamp so the visualizer's early-return guard
    # is bypassed even if force_render is not propagated all the way down.
    try:
        scene._t += 1
    except Exception:
        pass

    rendered = camera.render(force_render=True)
    rgb = rendered[0] if isinstance(rendered, tuple) else rendered
    rgb = _to_np(rgb)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb


def run(args):
    from skills.standup.config import StandupConfig
    from skills.standup.env import K1StandupEnv

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = StandupConfig()
    # Render every pool regardless of where the curriculum would start.
    backend = "cpu" if args.cpu else "gpu"

    print("=" * 64)
    print(" Standup pose-pool screenshot eval")
    print("=" * 64)
    print(f" out_dir:  {out_dir}")
    print(f" pools:    {args.pools}")
    print(f" per_pool: {args.per_pool}")
    print(f" num_envs: {args.num_envs}  (pool size = num_envs × rounds)")
    print(f" backend:  {backend}")
    print(f" jitter:   {args.with_jitter} "
          f"(reset joint jitter ±{cfg.joint_jitter_rad} rad; default OFF)")
    print(f" pen eps:  {cfg.pose_pool_penetration_eps} m")
    print()

    env = K1StandupEnv(cfg, num_envs=args.num_envs, render=False,
                       backend=backend, seed=args.seed)

    # First reset builds all pose pools (named + settle) via real settle
    # physics + the in-class / penetration filters.
    print("[eval] building pose pools (first reset)...")
    env.reset()

    camera = getattr(env, "camera", None)
    if camera is None:
        print("[eval] ERROR: env has no camera (offscreen render "
              "unavailable). Aborting.")
        return 2

    # Close 3/4 view of the lying robot at the origin. fov is fixed at the
    # env's camera creation (50°); we only move the camera pose.
    cam_pos = tuple(args.cam_pos)
    cam_lookat = tuple(args.cam_lookat)
    try:
        camera.set_pose(pos=cam_pos, lookat=cam_lookat)
    except Exception as e:
        print(f"[eval] camera.set_pose failed ({e}); using default pose")

    # Jitter note: the eval script can optionally add reset-time joint jitter
    # (cfg.joint_jitter_rad) to pool states before rendering, mirroring what
    # _reset_robot_pose does during training. However, the eval renders WITHOUT
    # a subsequent physics step, so jitter-induced penetrations are visible as
    # PEN_ frames even though training resolves them immediately via the first
    # scene.step(). The default is therefore NO JITTER (--with-jitter to opt in)
    # so screenshots show the actual pool states the filter approved.
    if args.with_jitter:
        print("[eval] jitter ON: adding ±{:.2f} rad joint noise (training-replica "
              "view — PEN_ may be physics-resolvable artifacts)".format(
              cfg.joint_jitter_rad))
    else:
        print("[eval] jitter OFF (showing clean pool states; use --with-jitter "
              "to mirror training reset behaviour)")

    rng = np.random.default_rng(args.seed)
    idx0 = np.array([0], dtype=np.int64)
    eps = float(cfg.pose_pool_penetration_eps)

    manifest = {"pools": {}, "config": {
        "per_pool": args.per_pool, "num_envs": args.num_envs,
        "jitter": args.with_jitter,
        "joint_jitter_rad": float(cfg.joint_jitter_rad),
        "penetration_eps": eps,
        "cam_pos": list(cam_pos), "cam_lookat": list(cam_lookat),
    }}

    from PIL import Image

    total_saved = 0
    total_pen = 0

    for pool_name in args.pools:
        pool_dir = out_dir / pool_name
        pool_dir.mkdir(parents=True, exist_ok=True)

        # Report the underlying pool size (0 → _sample_from_pool falls back
        # to the settle pool with a one-time warning).
        if pool_name == "random":
            size = int(env._pool_size)
            settle_info = f"settle={cfg.settle_steps} steps"
        else:
            p = env._named_pools.get(pool_name)
            size = int(p["size"]) if p else 0
            is_side = pool_name.startswith("side_")
            steps = (cfg.pose_pool_side_settle_steps if is_side
                     else cfg.pose_pool_settle_steps)
            rounds = (cfg.pose_pool_side_rounds if is_side
                      else cfg.pose_pool_rounds)
            settle_info = f"settle={steps} steps × {rounds} rounds"
        print(f"[eval] pool '{pool_name}': {size} states "
              f"({settle_info}) → rendering {args.per_pool} samples")

        # Sample all requested states up-front, then render one by one.
        pos, quat, jpos = env._sample_from_pool(pool_name, args.per_pool)

        pen_count = 0
        for i in range(args.per_pool):
            p1 = pos[i:i + 1].copy()
            q1 = quat[i:i + 1].copy()
            j1 = jpos[i:i + 1].copy()
            if args.with_jitter and cfg.joint_jitter_rad > 0:
                j1 = j1 + (rng.standard_normal(j1.shape).astype(np.float32)
                           * cfg.joint_jitter_rad)

            try:
                env.robot.set_pos(p1, envs_idx=idx0)
                env.robot.set_quat(q1, envs_idx=idx0)
                env.robot.set_dofs_position(j1, env.dof_indices,
                                            envs_idx=idx0, zero_velocity=True)
                env.robot.zero_all_dofs_velocity(envs_idx=idx0)
            except Exception as e:
                print(f"[eval] set pose failed ({pool_name} #{i}): {e}")
                continue

            base_z = float(_to_np(env.robot.get_pos())[0, 2])
            minc = float(env._min_contact_link_z()[0])
            penetrating = minc < -eps
            if penetrating:
                pen_count += 1

            prefix = "PEN_" if penetrating else ""
            fname = (f"{prefix}{pool_name}_{i:03d}"
                     f"_z{base_z:.3f}_minc{minc:+.3f}.png")
            try:
                rgb = _render_rgb(camera, env.scene)
                Image.fromarray(rgb).save(pool_dir / fname)
                total_saved += 1
            except Exception as e:
                print(f"[eval] render/save failed ({pool_name} #{i}): {e}")

        total_pen += pen_count
        manifest["pools"][pool_name] = {
            "pool_size": size,
            "rendered": args.per_pool,
            "penetrating": pen_count,
        }
        print(f"        penetrating (minc < -{eps}): {pen_count}"
              f"/{args.per_pool}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"[eval] done — {total_saved} screenshots in {out_dir}")
    print(f"[eval] penetrating frames: {total_pen} (see PEN_* files / "
          "manifest.json)")
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default="pose_pool_shots",
                   help="Directory to save screenshots into")
    p.add_argument("--per-pool", type=int, default=100,
                   help="Random poses (screenshots) per pool (default 100)")
    p.add_argument("--pools", nargs="+",
                   default=["prone", "supine", "side_left", "side_right",
                            "random"],
                   help="Which pools to render")
    p.add_argument("--num-envs", type=int, default=128,
                   help="Envs used to build the pools (pool size scales with "
                        "this; default 128)")
    p.add_argument("--cpu", action="store_true",
                   help="Use CPU backend (default GPU)")
    p.add_argument("--with-jitter", action="store_true",
                   help="Add reset-time joint jitter (±joint_jitter_rad) before "
                        "rendering — mirrors training reset behaviour but may show "
                        "PEN_ frames that physics resolves on the first scene.step()")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for pool build + sampling")
    p.add_argument("--cam-pos", type=float, nargs=3, default=[1.4, -1.4, 0.8],
                   help="Camera world position (x y z)")
    p.add_argument("--cam-lookat", type=float, nargs=3,
                   default=[0.0, 0.0, 0.12],
                   help="Camera look-at point (x y z)")
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
