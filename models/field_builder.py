#!/usr/bin/env python3
"""Single source of truth for RoboCup HSL field *geometry* + appearance.

Historically the field was rebuilt in four places that drifted apart
(``field_generator.py`` MuJoCo + Genesis, ``evaluation/evaluate.py``,
``scripts/debug_mujoco_scene.py``, and the static ``match_scene.xml``), so
some paths drew a full field and others drew only an outline on flat neon
green. This module centralises the markings so every renderer agrees and
looks like an actual pitch:

  * ``field_geoms(field)``        — the marking/goal geometry as plain dicts,
                                    consumed by both the XML and MjSpec paths.
  * ``build_match_scene_xml(...)``— a complete, good-looking MuJoCo scene
                                    (grass texture, crisp raised lines, center
                                    circle, penalty + corner arcs, goals+nets,
                                    skybox, cameras, ball).
  * ``add_field_to_spec(...)``    — the same geometry spliced into an existing
                                    MjSpec (the eval / multi-robot path).

All distances follow the HSL convention: dimensions are between the MIDDLE of
lines; arc radii are centre→middle-of-line.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import List
from xml.dom import minidom

from models.field_generator import FieldDimensions


# Appearance constants — shared so the XML and MjSpec paths match exactly.
LINE_W = 0.05            # white line width (m) — HSL lineWidth
LINE_Z = 0.012          # line centre height: sits clearly above carpet top
LINE_HH = 0.0025        # line half-height (m): thin but no z-fighting
ARC_SEGMENTS = 40       # full-circle segment count (arcs scale down from this)

# Two greens for a mowed-grass checker — deliberately darker / less saturated
# than the old flat (0.1, 0.6, 0.1), which blew out to neon under the lights.
GRASS_RGB1 = (0.21, 0.50, 0.21)
GRASS_RGB2 = (0.16, 0.42, 0.16)


def _arc(name: str, cx: float, cy: float, r: float,
         a0: float, a1: float, *, n: int = ARC_SEGMENTS,
         keep=None) -> List[dict]:
    """Approximate a circular arc [a0, a1] (radians) of radius `r` centred at
    (cx, cy) with thin rotated boxes. `keep(x, y)` optionally filters segment
    midpoints (used to draw only the part of the penalty arc outside the box)."""
    geoms: List[dict] = []
    steps = max(2, int(round(n * abs(a1 - a0) / (2 * math.pi))))
    for i in range(steps):
        t0 = a0 + (a1 - a0) * i / steps
        t1 = a0 + (a1 - a0) * (i + 1) / steps
        x0, y0 = cx + r * math.cos(t0), cy + r * math.sin(t0)
        x1, y1 = cx + r * math.cos(t1), cy + r * math.sin(t1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if keep is not None and not keep(mx, my):
            continue
        seg_len = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        geoms.append(dict(name=f"{name}_{i}", kind="line", type="box",
                          size=(seg_len / 2 + LINE_W / 2, LINE_W / 2, LINE_HH),
                          pos=(mx, my, LINE_Z), yaw=yaw))
    return geoms


def field_geoms(f: FieldDimensions) -> List[dict]:
    """Return every field marking + goal as a list of geom dicts.

    Each dict has: name, kind ∈ {carpet, line, mark, post, bar, net},
    type ∈ {box, cylinder}, size, pos, optional yaw. Lines/marks are visual
    (non-colliding); posts/bars/carpet collide so the ball bounces."""
    hl, hw = f.half_length, f.half_width
    lw, lz, hh = LINE_W, LINE_Z, LINE_HH
    g: List[dict] = []

    # ── Carpet (collidable ground; top face at z=0) ──────────────────
    g.append(dict(name="carpet", kind="carpet", type="box",
                  size=(f.total_length / 2, f.total_width / 2, 0.01),
                  pos=(0, 0, -0.01)))

    # ── Outline + halfway line ───────────────────────────────────────
    for s, tag in ((1, "p"), (-1, "n")):
        g.append(dict(name=f"touchline_{tag}", kind="line", type="box",
                      size=(hl + lw / 2, lw / 2, hh), pos=(0, s * hw, lz)))
        g.append(dict(name=f"goalline_{tag}", kind="line", type="box",
                      size=(lw / 2, hw + lw / 2, hh), pos=(s * hl, 0, lz)))
    g.append(dict(name="halfway", kind="line", type="box",
                  size=(lw / 2, hw, hh), pos=(0, 0, lz)))

    # ── Center circle + center mark ──────────────────────────────────
    g += _arc("center_circle", 0, 0, f.center_circle_radius, 0, 2 * math.pi)
    g.append(dict(name="center_mark", kind="mark", type="cylinder",
                  size=(f.penalty_mark_diameter / 2, hh), pos=(0, 0, lz)))

    # ── Penalty areas + penalty marks ────────────────────────────────
    # (No penalty "D" arc: the HSL kid-size field places the penalty mark
    # OUTSIDE the penalty area, so there is no standard arc marking.)
    pal, paw = f.penalty_area_length, f.penalty_area_width / 2
    pmd = f.penalty_mark_distance
    for s, tag in ((1, "p"), (-1, "n")):
        x_front = s * (hl - pal)
        g.append(dict(name=f"pa_front_{tag}", kind="line", type="box",
                      size=(lw / 2, paw + lw / 2, hh), pos=(x_front, 0, lz)))
        for ys, yt in ((1, "t"), (-1, "b")):
            g.append(dict(name=f"pa_side_{tag}_{yt}", kind="line", type="box",
                          size=(pal / 2, lw / 2, hh),
                          pos=(s * (hl - pal / 2), ys * paw, lz)))
        # Penalty mark
        xm = s * (hl - pmd)
        g.append(dict(name=f"pmark_{tag}", kind="mark", type="cylinder",
                      size=(f.penalty_mark_diameter / 2, hh), pos=(xm, 0, lz)))

    # ── Corner arcs (quarter circles at the 4 corners) ───────────────
    car = f.corner_arc_radius
    for sx in (1, -1):
        for sy in (1, -1):
            cx, cy = sx * hl, sy * hw
            a0 = math.atan2(-sy, 0)          # start pointing along touchline
            g += _arc(f"corner_{sx}_{sy}", cx, cy, car,
                      a0, a0 - sx * sy * math.pi / 2, n=ARC_SEGMENTS)

    # ── Goals (posts + crossbar collide; net is visual) ──────────────
    gw = f.goal_inner_width / 2
    gd, gh, pr = f.goal_depth, f.goal_height, f.goal_post_diameter / 2
    for s, tag in ((1, "p"), (-1, "n")):
        gx = s * hl
        for ys, yt in ((1, "L"), (-1, "R")):
            g.append(dict(name=f"post_{tag}_{yt}", kind="post", type="cylinder",
                          size=(pr, gh / 2), pos=(gx + s * gd / 2, ys * gw, gh / 2)))
        g.append(dict(name=f"crossbar_{tag}", kind="bar", type="box",
                      size=(pr, gw + pr, pr), pos=(gx + s * gd / 2, 0, gh)))
        # Net: back wall + two sides (thin, translucent)
        g.append(dict(name=f"net_back_{tag}", kind="net", type="box",
                      size=(0.004, gw, gh / 2), pos=(gx + s * gd, 0, gh / 2)))
        for ys, yt in ((1, "L"), (-1, "R")):
            g.append(dict(name=f"net_side_{tag}_{yt}", kind="net", type="box",
                          size=(gd / 2, 0.004, gh / 2),
                          pos=(gx + s * gd / 2, ys * gw, gh / 2)))
    return g


# ─────────────────────────── XML (static scene) ──────────────────────────


def build_match_scene_xml(f: FieldDimensions, *, with_ball: bool = True,
                          with_cameras: bool = True) -> str:
    """Return a complete, good-looking MuJoCo XML scene for the field."""
    root = ET.Element("mujoco", model="robocup_hsl_field")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    opt = ET.SubElement(root, "option", timestep="0.002",
                        gravity="0 0 -9.81", integrator="implicitfast")
    ET.SubElement(opt, "flag", contact="enable")

    visual = ET.SubElement(root, "visual")
    # Softer headlight than before — the grass texture provides the colour,
    # so we don't need to blast it white (which produced the neon look).
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35",
                  diffuse="0.55 0.55 0.55", specular="0.1 0.1 0.1")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")

    asset = ET.SubElement(root, "asset")
    # Mowed-grass checker (two greens) + a faint per-texel noise via checker.
    ET.SubElement(asset, "texture", name="grass", type="2d", builtin="checker",
                  rgb1="{} {} {}".format(*GRASS_RGB1),
                  rgb2="{} {} {}".format(*GRASS_RGB2),
                  width="300", height="300")
    # texrepeat tuned to ~0.9 m mow cells across the total carpet.
    nx = max(1, round(f.total_length / 0.9))
    ny = max(1, round(f.total_width / 0.9))
    ET.SubElement(asset, "material", name="grass_mat", texture="grass",
                  texrepeat=f"{nx} {ny}", texuniform="true",
                  specular="0.05", shininess="0.05", reflectance="0.0")
    # Crisp lines: emissive white so they read bright regardless of lighting.
    ET.SubElement(asset, "material", name="line_mat", rgba="0.95 0.95 0.95 1",
                  emission="0.45", specular="0.0", shininess="0.0")
    ET.SubElement(asset, "material", name="goal_mat", rgba="0.92 0.92 0.92 1",
                  emission="0.15", specular="0.3", shininess="0.4")
    ET.SubElement(asset, "material", name="net_mat", rgba="0.85 0.88 0.9 0.28",
                  specular="0.0")
    ET.SubElement(asset, "texture", name="ball_tex", type="2d",
                  builtin="checker", rgb1="0.95 0.95 0.95", rgb2="0.08 0.08 0.08",
                  width="64", height="64")
    ET.SubElement(asset, "material", name="ball_mat", texture="ball_tex",
                  texrepeat="3 3")
    ET.SubElement(asset, "texture", name="skybox", type="skybox",
                  builtin="gradient", rgb1="0.55 0.70 0.95",
                  rgb2="0.10 0.14 0.28", width="512", height="512")

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", condim="3", friction="1.0 0.005 0.0001")

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", name="sun", pos="0 0 8", dir="0 0 -1",
                  diffuse="0.7 0.7 0.7", specular="0.2 0.2 0.2",
                  directional="true", castshadow="true")
    ET.SubElement(wb, "light", name="fill", pos="5 -6 5", dir="-0.6 0.7 -0.6",
                  diffuse="0.3 0.3 0.3", directional="true", castshadow="false")

    mat_for = {"carpet": "grass_mat", "line": "line_mat", "mark": "line_mat",
               "post": "goal_mat", "bar": "goal_mat", "net": "net_mat"}
    collide = {"carpet", "post", "bar"}
    for d in field_geoms(f):
        attrs = dict(name=d["name"], type=d["type"],
                     size=" ".join(f"{v:.5f}" for v in d["size"]),
                     pos=" ".join(f"{v:.5f}" for v in d["pos"]),
                     material=mat_for[d["kind"]])
        if "yaw" in d:
            attrs["euler"] = f"0 0 {d['yaw']:.5f}"
        if d["kind"] not in collide:
            attrs["contype"] = "0"
            attrs["conaffinity"] = "0"
        ET.SubElement(wb, "geom", **attrs)

    if with_ball:
        ball = ET.SubElement(wb, "body", name="ball", pos="0 0 0.07")
        ET.SubElement(ball, "freejoint", name="ball_joint")
        ET.SubElement(ball, "geom", name="ball_geom", type="sphere",
                      size="0.07", mass="0.35", material="ball_mat",
                      condim="4", friction="0.8 0.005 0.0001",
                      solref="0.01 1.0", solimp="0.9 0.95 0.001")

    if with_cameras:
        ET.SubElement(wb, "camera", name="top_down", pos="0 0 12",
                      euler="0 0 0", fovy="50")
        ET.SubElement(wb, "camera", name="side_view", pos="0 -8 3",
                      euler="1.2 0 0", fovy="60")
        ET.SubElement(wb, "camera", name="broadcast", pos="0 -7.5 4.2",
                      euler="1.02 0 0", fovy="55")

    xml = minidom.parseString(ET.tostring(root, encoding="unicode")
                              ).toprettyxml(indent="  ")
    body = "\n".join(ln for ln in xml.split("\n") if not ln.startswith("<?xml"))
    return '<?xml version="1.0" encoding="utf-8"?>\n' + body


# ─────────────────────────── MjSpec (eval path) ──────────────────────────


def add_field_to_spec(spec, mj, f: FieldDimensions, *,
                      add_ball: bool = True):
    """Splice carpet + markings + goals (+ optional ball) into an existing
    MjSpec worldbody. Field markings are non-colliding; carpet/posts/bars
    collide so the ball physics works. Returns the ball material name (or
    None). Shares `field_geoms` with the XML path so the two never drift."""
    wb = spec.worldbody

    def _mat(name, **kw):
        m = spec.add_material()
        m.name = name
        for k, v in kw.items():
            setattr(m, k, v)
        return m

    # Grass texture + material
    gtex = spec.add_texture()
    gtex.name = "grass"
    gtex.type = mj.mjtTexture.mjTEXTURE_2D
    gtex.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER
    gtex.width = gtex.height = 300
    gtex.rgb1 = list(GRASS_RGB1)
    gtex.rgb2 = list(GRASS_RGB2)
    nx = max(1, round(f.total_length / 0.9))
    ny = max(1, round(f.total_width / 0.9))
    grass = spec.add_material()
    grass.name = "grass_mat"
    grass.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "grass"
    grass.texrepeat = [nx, ny]
    grass.texuniform = True
    line = _mat("line_mat", rgba=[0.95, 0.95, 0.95, 1.0], emission=0.45)
    goalm = _mat("goal_mat", rgba=[0.92, 0.92, 0.92, 1.0], emission=0.15,
                 specular=0.3, shininess=0.4)
    netm = _mat("net_mat", rgba=[0.85, 0.88, 0.9, 0.28])

    mat_for = {"carpet": "grass_mat", "line": "line_mat", "mark": "line_mat",
               "post": "goal_mat", "bar": "goal_mat", "net": "net_mat"}
    collide = {"carpet", "post", "bar"}
    for d in field_geoms(f):
        g = wb.add_geom()
        g.name = d["name"]
        g.type = (mj.mjtGeom.mjGEOM_BOX if d["type"] == "box"
                  else mj.mjtGeom.mjGEOM_CYLINDER)
        g.size = list(d["size"]) + ([0.0] if d["type"] == "cylinder" else [])
        g.pos = list(d["pos"])
        g.material = mat_for[d["kind"]]
        if "yaw" in d:
            h = d["yaw"] / 2.0
            g.quat = [math.cos(h), 0.0, 0.0, math.sin(h)]
        if d["kind"] == "carpet":
            g.friction = [1.0, 0.005, 0.0001]
        if d["kind"] not in collide:
            g.contype = 0
            g.conaffinity = 0

    if add_ball:
        ball = wb.add_body()
        ball.name = "ball"
        ball.pos = [1.0, 0.0, 0.07]
        ball.add_freejoint()
        bg = ball.add_geom()
        bg.name = "ball_geom"
        bg.type = mj.mjtGeom.mjGEOM_SPHERE
        bg.size = [0.07, 0, 0]
        bg.mass = 0.2
        bg.rgba = [0.95, 0.95, 0.95, 1.0]
        bg.friction = [0.8, 0.005, 0.0001]
        bg.condim = 4
        bg.solref = [0.01, 1.0]
        bg.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]


def _write_match_scene(json_path: str, out_path: str) -> None:
    f = FieldDimensions.from_json(json_path)
    with open(out_path, "w") as fh:
        fh.write(build_match_scene_xml(f))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Regenerate the static field scene")
    p.add_argument("--json", default="configs/field_hsl_2026.json")
    p.add_argument("-o", "--out", default="models/field/match_scene.xml")
    a = p.parse_args()
    _write_match_scene(a.json, a.out)
