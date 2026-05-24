#!/usr/bin/env python3
"""
Genesis spawn debugger for the RoboCup K1 setup.

Runs a Genesis scene with the soccer field + 1 K1 robot + ball, then:

  * prints scene/entity counts, dof layout, joint→dof_idx mapping,
  * dumps initial positions of robot base, ball, every link, every joint,
  * steps simulation N times with ZERO actuation and prints positions
    again (so you can see the robot collapse under gravity and the ball
    settle on the carpet, instead of either flying off or falling through),
  * optionally captures viewer screenshots before/after stepping.

Usage:
    python -m scripts.debug_genesis_spawn                # headless, prints state
    python -m scripts.debug_genesis_spawn --render       # opens Genesis viewer
    python -m scripts.debug_genesis_spawn --steps 500    # more sim steps
    python -m scripts.debug_genesis_spawn --no-field     # plane only (isolate robot)
    python -m scripts.debug_genesis_spawn --screenshot out.png

Designed to stay useful as the project evolves: keep all "what is the
scene actually doing on spawn" checks in here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_paths():
    """Resolve all the asset paths the debugger needs."""
    return {
        "urdf": PROJECT_ROOT / "models" / "robot" / "K1" / "K1_22dof.urdf",
        "field_info": PROJECT_ROOT / "models" / "field" / "field_info.json",
    }


def _to_np(x):
    """Genesis returns torch tensors; standardise to numpy for printing."""
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _load_field_info(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "length": 9.0, "width": 6.0,
        "half_length": 4.5, "half_width": 3.0,
        "total_length": 11.0, "total_width": 8.0,
        "goal_width": 2.6, "goal_height": 0.8, "goal_depth": 0.6,
        "penalty_area_length": 1.0, "penalty_area_width": 3.0,
        "penalty_mark_distance": 1.5, "center_circle_radius": 0.75,
        "border_strip_width": 1.0,
    }


def _build_field(scene, field_info: dict, gs):
    """Field carpet + lines + goals using Genesis-native surface colors.

    Kept inline so the debug script has no hidden dependency on the
    auto-generated builder (which has bugs we're separately fixing).
    """
    import math

    tl = field_info["total_length"]
    tw = field_info["total_width"]
    hl = field_info["half_length"]
    hw = field_info["half_width"]
    gw = field_info["goal_width"] / 2
    gh = field_info["goal_height"]
    gd = field_info["goal_depth"]
    pal = field_info["penalty_area_length"]
    paw = field_info["penalty_area_width"] / 2
    ccr = field_info["center_circle_radius"]

    green = gs.surfaces.Default(color=(0.1, 0.55, 0.1, 1.0), roughness=0.9)
    white = gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0), roughness=0.6)
    post = gs.surfaces.Default(color=(0.95, 0.95, 0.95, 1.0), roughness=0.4)

    scene.add_entity(
        gs.morphs.Box(size=(tl, tw, 0.02), pos=(0, 0, -0.01), fixed=True,
                      collision=True, visualization=True),
        surface=green,
    )

    lh = 0.003  # raise lines slightly so they render above carpet
    lz = lh / 2 + 0.002  # above carpet top (carpet top is at z=0)
    lw = 0.05

    # Touchlines + goal lines + center line
    for sign in (1, -1):
        scene.add_entity(
            gs.morphs.Box(size=(2 * hl, lw, lh), pos=(0, sign * hw, lz),
                          fixed=True, collision=False),
            surface=white,
        )
        scene.add_entity(
            gs.morphs.Box(size=(lw, 2 * hw, lh), pos=(sign * hl, 0, lz),
                          fixed=True, collision=False),
            surface=white,
        )
    scene.add_entity(
        gs.morphs.Box(size=(lw, 2 * hw, lh), pos=(0, 0, lz),
                      fixed=True, collision=False),
        surface=white,
    )

    # Penalty areas
    for sign in (1, -1):
        scene.add_entity(
            gs.morphs.Box(size=(lw, 2 * paw, lh),
                          pos=(sign * (hl - pal), 0, lz),
                          fixed=True, collision=False),
            surface=white,
        )
        for ysign in (1, -1):
            scene.add_entity(
                gs.morphs.Box(size=(pal, lw, lh),
                              pos=(sign * (hl - pal / 2), ysign * paw, lz),
                              fixed=True, collision=False),
                surface=white,
            )

    # Center circle approximated by polygon edges
    n_seg = 48
    for i in range(n_seg):
        a0 = 2 * math.pi * i / n_seg
        a1 = 2 * math.pi * (i + 1) / n_seg
        cx = (ccr * math.cos(a0) + ccr * math.cos(a1)) / 2
        cy = (ccr * math.sin(a0) + ccr * math.sin(a1)) / 2
        seg_len = 2 * ccr * math.sin(math.pi / n_seg)
        angle = (a0 + a1) / 2
        scene.add_entity(
            gs.morphs.Box(size=(seg_len, lw, lh), pos=(cx, cy, lz),
                          euler=(0, 0, angle), fixed=True, collision=False),
            surface=white,
        )

    # Goal frames (collidable so the ball bounces off)
    for sign in (1, -1):
        gx = sign * hl
        for ysign in (1, -1):
            scene.add_entity(
                gs.morphs.Cylinder(radius=0.05, height=gh,
                                   pos=(gx + sign * gd / 2, ysign * gw,
                                        gh / 2),
                                   fixed=True, collision=True),
                surface=post,
            )
        scene.add_entity(
            gs.morphs.Box(size=(0.1, 2 * gw + 0.1, 0.1),
                          pos=(gx + sign * gd / 2, 0, gh),
                          fixed=True, collision=True),
            surface=post,
        )


def _print_robot_summary(robot):
    """Print joint → dof_idx mapping, link → pos, base pose."""
    print(f"  n_dofs={robot.n_dofs}  n_links={robot.n_links}  n_joints={robot.n_joints}")
    print()
    print("  ── Joints (Genesis order):")
    actuated = []
    for j in robot.joints:
        try:
            idxs = j.dofs_idx_local
        except Exception:
            idxs = None
        try:
            jt = str(j.type)
        except Exception:
            jt = "?"
        print(f"    {str(j.name):<28}  dofs={idxs}  type={jt}")
        if idxs and len(idxs) == 1:
            actuated.append((idxs[0], j.name))
    actuated.sort()
    print()
    print("  ── Actuated joints (single-DoF, sorted):")
    for di, n in actuated:
        print(f"    dof_idx={di:>3}  {n}")
    return [di for di, _ in actuated]


def _print_state(label, robot, ball):
    pos = _to_np(robot.get_pos())
    quat = _to_np(robot.get_quat())
    vel = _to_np(robot.get_vel())
    dofq = _to_np(robot.get_dofs_position())
    dofv = _to_np(robot.get_dofs_velocity())
    bpos = _to_np(ball.get_pos())
    bvel = _to_np(ball.get_vel())

    base_z = float(np.atleast_1d(pos).flatten()[2])
    ball_z = float(np.atleast_1d(bpos).flatten()[2])

    print(f"  ─ {label} ─")
    print(f"    base pos:  {np.array2string(np.atleast_1d(pos).flatten(), precision=3)}")
    print(f"    base quat: {np.array2string(np.atleast_1d(quat).flatten(), precision=3)}")
    print(f"    base vel:  {np.array2string(np.atleast_1d(vel).flatten(), precision=3)}")
    print(f"    ball pos:  {np.array2string(np.atleast_1d(bpos).flatten(), precision=3)}")
    print(f"    ball vel:  {np.array2string(np.atleast_1d(bvel).flatten(), precision=3)}")
    print(f"    dof q:     {np.array2string(np.atleast_1d(dofq).flatten(), precision=2, max_line_width=110)}")
    print(f"    dof qdot:  {np.array2string(np.atleast_1d(dofv).flatten(), precision=2, max_line_width=110)}")
    print()
    return {"base_z": base_z, "ball_z": ball_z}


def _capture_screenshot(scene, camera, out_path: str):
    """Save a render to disk. Uses an off-screen camera regardless of viewer."""
    if camera is None:
        return False
    try:
        rendered = camera.render()
        rgb = rendered[0] if isinstance(rendered, tuple) else rendered
        rgb = _to_np(rgb)
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(rgb).save(out_path)
        print(f"  screenshot saved: {out_path}")
        return True
    except Exception as e:
        print(f"  screenshot failed: {e}")
        return False


def run(args):
    paths = _resolve_paths()
    if not paths["urdf"].exists():
        print(f"ERROR: URDF not found at {paths['urdf']}")
        sys.exit(2)
    field_info = _load_field_info(paths["field_info"])

    print("=" * 64)
    print(" Genesis Spawn Debugger")
    print("=" * 64)
    print(f" urdf:       {paths['urdf']}")
    print(f" field_info: {paths['field_info']} (exists={paths['field_info'].exists()})")
    print(f" render:     {args.render}")
    print(f" with_field: {not args.no_field}")
    print(f" robot z:    {args.robot_z}")
    print(f" ball z:     {args.ball_z}")
    print(f" steps:      {args.steps}")
    print()

    import genesis as gs

    backend = gs.gpu if args.gpu else gs.cpu
    try:
        gs.init(backend=backend, precision="32", logging_level="warning", seed=0)
    except Exception as e:
        print(f"  (genesis init skipped: {e})")

    scene = gs.Scene(
        show_viewer=args.render,
        sim_options=gs.options.SimOptions(dt=0.002, substeps=2, gravity=(0, 0, -9.81)),
        viewer_options=(
            gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(0, -6, 4),
                camera_lookat=(0, 0, 0.5),
                camera_fov=50,
                max_FPS=60,
            )
            if args.render
            else None
        ),
        vis_options=gs.options.VisOptions(show_world_frame=True,
                                          ambient_light=(0.4, 0.4, 0.4)),
    )

    if not args.no_field:
        print("[scene] building field...")
        _build_field(scene, field_info, gs)
    else:
        print("[scene] plane only (no_field)")
        scene.add_entity(gs.morphs.Plane())

    print("[scene] adding K1 robot...")
    robot = scene.add_entity(
        gs.morphs.URDF(file=str(paths["urdf"]), pos=(0, 0, args.robot_z),
                       merge_fixed_links=True),
    )

    print("[scene] adding ball...")
    ball_surface = gs.surfaces.Default(color=(0.95, 0.95, 0.95, 1.0),
                                       roughness=0.7)
    ball = scene.add_entity(
        gs.morphs.Sphere(radius=0.07, pos=(args.ball_x, 0, args.ball_z),
                         collision=True),
        surface=ball_surface,
    )

    # Off-screen camera for screenshots even in headless mode
    camera = scene.add_camera(res=(1280, 720), pos=(0, -6, 4),
                              lookat=(0, 0, 0.5), fov=50)

    scene.build()

    print()
    print("[robot] summary:")
    actuated_idx = _print_robot_summary(robot)

    print("[state] initial (after build, before stepping):")
    s0 = _print_state("t=0", robot, ball)

    if args.screenshot:
        out = args.screenshot
        if not out.endswith(".png"):
            out = out + "_t0.png"
        else:
            out = out.replace(".png", "_t0.png")
        _capture_screenshot(scene, camera, out)

    # Step with ZERO control force — robot should collapse, ball should settle
    print(f"[sim] stepping {args.steps} times with NO actuation (gravity only)...")
    t_start = time.time()
    for _ in range(args.steps):
        scene.step()
    dt = time.time() - t_start
    print(f"  done in {dt:.2f}s "
          f"({args.steps / max(dt, 1e-6):.0f} steps/s)")
    print()

    print(f"[state] after {args.steps} unactuated steps:")
    s1 = _print_state(f"t={args.steps * 0.002:.2f}s", robot, ball)

    if args.screenshot:
        out = args.screenshot
        if out.endswith(".png"):
            out = out.replace(".png", "_tN.png")
        else:
            out = out + "_tN.png"
        _capture_screenshot(scene, camera, out)

    # Diagnostics
    print("[diagnostics]")
    fell_through = s1["ball_z"] < -0.05
    collapsed = s1["base_z"] < 0.4
    floating = s1["base_z"] > args.robot_z - 0.02

    if fell_through:
        print("  ✗ ball Z is below ground — ball fell through the carpet/plane")
    else:
        print(f"  ✓ ball settled at z={s1['ball_z']:.3f} (radius 0.07)")

    if collapsed:
        print(f"  ✓ robot base dropped from {args.robot_z:.2f}m → "
              f"{s1['base_z']:.3f}m — collapsing as expected")
    elif floating:
        print(f"  ✗ robot base barely moved ({args.robot_z:.2f} → "
              f"{s1['base_z']:.3f}) — something is holding it up")
    else:
        print(f"  ~ robot base at z={s1['base_z']:.3f} after {args.steps} steps")

    if args.render:
        print()
        print("Viewer left open. Ctrl+C to quit.")
        try:
            while True:
                scene.step()
        except KeyboardInterrupt:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=500,
                   help="Sim steps to take after spawn (default 500 = 1s at 500Hz)")
    p.add_argument("--robot-z", type=float, default=1.0,
                   help="Initial Z of K1 base link (default 1.0m)")
    p.add_argument("--ball-x", type=float, default=1.0,
                   help="Initial X of the ball (default 1.0m in front)")
    p.add_argument("--ball-z", type=float, default=0.30,
                   help="Initial Z of the ball — drop from height to verify "
                        "no fall-through (default 0.30m)")
    p.add_argument("--no-field", action="store_true",
                   help="Skip field, use a Plane only")
    p.add_argument("--render", action="store_true",
                   help="Open the Genesis viewer (requires a display)")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU backend (default CPU for portability)")
    p.add_argument("--screenshot", type=str, default=None,
                   help="Save before/after PNG to this path (suffixes _t0/_tN)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
