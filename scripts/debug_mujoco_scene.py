#!/usr/bin/env python3
"""
MuJoCo scene debugger for the RoboCup K1 setup.

Loads the K1 MJCF as the model root (so meshes / asset paths resolve cleanly),
then uses MuJoCo's MjSpec API to splice in a soccer field (carpet, lines,
goals) and a ball. Reports body/joint info and (optionally) renders before /
after PNGs so you can visually confirm the robot is actually in the scene.

This is the MuJoCo counterpart to scripts/debug_genesis_spawn.py and the
primary way to sanity-check the sim2sim evaluation path.

Background — why MjSpec instead of <include>:
The K1_22dof.xml file is a full <mujoco> document (with its own <compiler>,
<option>, <asset>, <worldbody>, ground plane). MJCF's <include> splices an
element AS-IS, so including K1 multiple times duplicates assets and names,
and including it under <worldbody> dumps <compiler> as a worldbody child →
schema error. MjSpec (mujoco 3.x) lets us load K1 once and add field
geometry into its existing worldbody.

Usage:
    python -m scripts.debug_mujoco_scene
    python -m scripts.debug_mujoco_scene --steps 500 --screenshot debug.png
    python -m scripts.debug_mujoco_scene --no-field  # robot only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _field_info() -> dict:
    p = PROJECT_ROOT / "models" / "field" / "field_info.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"half_length": 4.5, "half_width": 3.0,
            "total_length": 11.0, "total_width": 8.0,
            "goal_width": 2.6, "goal_height": 0.8, "goal_depth": 0.6,
            "penalty_area_length": 1.0, "penalty_area_width": 3.0,
            "center_circle_radius": 0.75, "penalty_mark_distance": 1.5}


def _add_ball_texture(spec, mj):
    """Register the telstar texture + material in the spec. Returns the
    material name to use on the ball geom, or None if the texture file
    isn't present yet."""
    ball_tex_path = PROJECT_ROOT / "models" / "textures" / "ball.png"
    if not ball_tex_path.exists():
        return None
    tex = spec.add_texture()
    tex.name = "ball_tex"
    tex.type = mj.mjtTexture.mjTEXTURE_2D
    tex.file = str(ball_tex_path)
    mat = spec.add_material()
    mat.name = "ball_mat"
    mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "ball_tex"
    return "ball_mat"


def _add_field(spec, field: dict, mj, ball_material: str | None = None):
    """Mutate `spec.worldbody` to add carpet, lines, goals, and a free ball.

    Field elements are non-colliding (contype=conaffinity=0) so they don't
    pollute robot contact resolution. The carpet sits at z=-0.011 to keep
    the existing K1 ground plane at z=0 doing the actual physics.
    """
    wb = spec.worldbody
    hl = field["half_length"]
    hw = field["half_width"]
    tl = field["total_length"]
    tw = field["total_width"]
    gw = field["goal_width"] / 2
    gh = field["goal_height"]
    gd = field["goal_depth"]
    pal = field["penalty_area_length"]
    paw = field["penalty_area_width"] / 2
    ccr = field["center_circle_radius"]

    def box(name, size, pos, rgba=(1, 1, 1, 1), collide=False, yaw=0.0):
        g = wb.add_geom()
        g.name = name
        g.type = mj.mjtGeom.mjGEOM_BOX
        g.size = list(size)
        g.pos = list(pos)
        g.rgba = list(rgba)
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        if yaw:
            half = yaw / 2.0
            g.quat = [math.cos(half), 0.0, 0.0, math.sin(half)]
        return g

    def cyl(name, radius, half_height, pos, rgba=(0.95, 0.95, 0.95, 1),
            collide=True):
        g = wb.add_geom()
        g.name = name
        g.type = mj.mjtGeom.mjGEOM_CYLINDER
        g.size = [radius, half_height, 0.0]
        g.pos = list(pos)
        g.rgba = list(rgba)
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        return g

    # Carpet — collidable replacement for the K1's own ground plane (which
    # the caller removes before invoking us).
    g = wb.add_geom()
    g.name = "field_carpet"
    g.type = mj.mjtGeom.mjGEOM_BOX
    g.size = [tl / 2, tw / 2, 0.005]
    g.pos = [0, 0, -0.005]
    g.rgba = [0.10, 0.55, 0.10, 1.0]
    g.friction = [1.0, 0.005, 0.0001]

    # Field outline
    lz = 0.002
    for sign in (1, -1):
        box(f"touchline_{'p' if sign>0 else 'n'}", (hl, 0.025, 0.001),
            (0, sign * hw, lz))
        box(f"goalline_{'p' if sign>0 else 'n'}", (0.025, hw, 0.001),
            (sign * hl, 0, lz))
    box("centerline", (0.025, hw, 0.001), (0, 0, lz))

    # Penalty areas
    for sign in (1, -1):
        side = "p" if sign > 0 else "n"
        box(f"pa_front_{side}", (0.025, paw, 0.001),
            (sign * (hl - pal), 0, lz))
        for ysign in (1, -1):
            ys = "t" if ysign > 0 else "b"
            box(f"pa_side_{side}_{ys}", (pal / 2, 0.025, 0.001),
                (sign * (hl - pal / 2), ysign * paw, lz))

    # Center circle segments
    n_seg = 36
    for i in range(n_seg):
        a0 = 2 * math.pi * i / n_seg
        a1 = 2 * math.pi * (i + 1) / n_seg
        cx = (ccr * math.cos(a0) + ccr * math.cos(a1)) / 2
        cy = (ccr * math.sin(a0) + ccr * math.sin(a1)) / 2
        seg_len = 2 * ccr * math.sin(math.pi / n_seg)
        ang = (a0 + a1) / 2
        box(f"cc_{i}", (seg_len / 2, 0.025, 0.001),
            (cx, cy, lz), yaw=ang)

    # Goal posts and crossbars (collidable)
    for sign in (1, -1):
        side = "p" if sign > 0 else "n"
        gx = sign * hl
        for ysign in (1, -1):
            ys = "L" if ysign > 0 else "R"
            cyl(f"post_{side}_{ys}", 0.05, gh / 2,
                (gx + sign * gd / 2, ysign * gw, gh / 2))
        box(f"crossbar_{side}", (0.05, gw + 0.05, 0.05),
            (gx + sign * gd / 2, 0, gh),
            rgba=(0.95, 0.95, 0.95, 1.0), collide=True)

    ball = wb.add_body()
    ball.name = "ball"
    ball.pos = [1.0, 0.0, 0.07]
    ball.add_freejoint()
    bg = ball.add_geom()
    bg.name = "ball_geom"
    bg.type = mj.mjtGeom.mjGEOM_SPHERE
    bg.size = [0.07, 0, 0]
    bg.mass = 0.2
    if ball_material:
        bg.material = ball_material
    else:
        bg.rgba = [0.95, 0.95, 0.95, 1.0]
    bg.friction = [0.8, 0.005, 0.0001]
    bg.condim = 4
    bg.solref = [0.01, 1.0]
    bg.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]


def _summarise(model, data, mj):
    print(f"  nbody={model.nbody}  njnt={model.njnt}  nq={model.nq}  nv={model.nv}  nu={model.nu}")
    print()
    print("  bodies:")
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"<body{i}>"
        pos = data.xpos[i]
        print(f"    [{i:>2}] {name:<28} pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
    print()
    print("  joints:")
    for i in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) or f"<joint{i}>"
        jt = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        qadr = int(model.jnt_qposadr[i])
        print(f"    [{i:>2}] {name:<28} type={jt:<6} qpos_adr={qadr}")


def _shoot(renderer, data, args, label: str, camera=None):
    if renderer is None or not args.screenshot:
        return
    if camera is not None:
        renderer.update_scene(data, camera=camera)
    else:
        renderer.update_scene(data)
    rgb = renderer.render()
    from PIL import Image
    path = args.screenshot
    suffix = f"_{label}.png"
    if path.endswith(".png"):
        path = path[:-4] + suffix
    else:
        path = path + suffix
    Image.fromarray(rgb).save(path)
    print(f"  screenshot {label}: {path}")


def run(args):
    try:
        import mujoco
    except ImportError:
        print("ERROR: mujoco package not installed")
        sys.exit(2)

    robot_xml = PROJECT_ROOT / "models" / "robot" / "K1" / "K1_22dof.xml"
    field = _field_info()

    print("=" * 64)
    print(" MuJoCo Scene Debugger (MjSpec-based)")
    print("=" * 64)
    print(f" robot_xml: {robot_xml} (exists={robot_xml.exists()})")
    print(f" with_field: {not args.no_field}")
    print(f" steps: {args.steps}")
    print()

    if not robot_xml.exists():
        print("  ✗ K1 MJCF missing")
        sys.exit(2)

    try:
        spec = mujoco.MjSpec.from_file(str(robot_xml))
    except Exception as e:
        print(f"  ✗ failed to parse K1 MJCF: {e}")
        sys.exit(3)

    if not args.no_field:
        # K1_22dof.xml ships with a checker-pattern ground plane that
        # otherwise hides the green carpet we're about to add. Remove it
        # via the spec-level delete (MjsGeom has no .delete()).
        try:
            ground = spec.geom("ground")
            if ground is not None:
                spec.delete(ground)
        except Exception as e:
            print(f"  (could not remove K1 ground plane: {e})")
        ball_mat = _add_ball_texture(spec, mujoco)
        _add_field(spec, field, mujoco, ball_material=ball_mat)

    try:
        model = spec.compile()
    except Exception as e:
        print(f"  ✗ MjSpec compile failed: {e}")
        sys.exit(3)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("  ✓ compiled & initialised")
    print()

    if args.dump_xml:
        try:
            xml_str = spec.to_xml()
            Path(args.dump_xml).write_text(xml_str)
            print(f" XML dumped to {args.dump_xml}")
        except Exception as e:
            print(f" XML dump failed: {e}")

    _summarise(model, data, mujoco)

    renderer = None
    try:
        # MuJoCo's default offscreen framebuffer is 640x480; clamp render
        # resolution to that to avoid "framebuffer too small" errors.
        renderer = mujoco.Renderer(model, height=480, width=640)
        # Override the free camera so the field+robot fit in frame
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance = 4.0
        cam.lookat[:] = [0.3, 0.0, 0.5]
        cam.azimuth = 110.0
        cam.elevation = -18.0
        renderer._camera = cam  # internal, but renderer.render(camera=...) also works
    except Exception as e:
        print(f"  (renderer unavailable: {e})")
        cam = None

    _shoot(renderer, data, args, "t0", camera=cam)

    print()
    print(f"[sim] stepping {args.steps} times (no ctrl applied)...")
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    print(f"  done. simtime={data.time:.3f}s")
    print()

    print("[state] post-step body positions:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body{i}>"
        pos = data.xpos[i]
        print(f"    [{i:>2}] {name:<28} pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")

    _shoot(renderer, data, args, "tN", camera=cam)

    # Trunk is body[1] for K1 (body[0] is world)
    if model.nbody > 1:
        z = float(data.xpos[1, 2])
        print()
        if z < 0.05:
            print(f"  ✗ trunk fell to z={z:.3f} — possibly through floor")
        elif z < 0.5:
            print(f"  ✓ trunk dropped to z={z:.3f} — unactuated collapse OK")
        else:
            print(f"  ~ trunk at z={z:.3f} — barely moved")

    # Find ball
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball_id >= 0:
        bz = float(data.xpos[ball_id, 2])
        if bz < -0.05:
            print(f"  ✗ ball at z={bz:.3f} — fell through carpet")
        else:
            print(f"  ✓ ball at z={bz:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--no-field", action="store_true",
                   help="Robot only — skip field/ball additions")
    p.add_argument("--screenshot", type=str, default=None,
                   help="Save before/after PNGs (suffixes _t0/_tN)")
    p.add_argument("--dump-xml", type=str, default=None,
                   help="Write spec.to_xml() to this path")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
