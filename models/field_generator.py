#!/usr/bin/env python3
"""
Field Model Generator for RoboCup Humanoid Soccer League.

Based on the official HSL-Rules FieldGenerator (generateField.py), this script
converts field dimensions from JSON into:
  1. MuJoCo XML (.xml) scene for evaluation
  2. Genesis-compatible scene builder for training

All distances in the JSON are between the middle of lines.
Radiuses are from center point to middle of a line.
"""

import json
import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass, field as dc_field
from typing import Optional


# ── PLY mesh helpers ──────────────────────────────────────────────────────────

def _write_ply(path: str, vertices, faces) -> None:
    """Write an ASCII PLY triangle mesh."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\nend_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def _ring_mesh(inner_r: float, outer_r: float, n: int = 64):
    """Flat annular ring (top face only, CCW winding → +Z normals)."""
    verts, tris = [], []
    for j in range(n):
        a = 2 * math.pi * j / n
        verts.append((inner_r * math.cos(a), inner_r * math.sin(a), 0.0))
    for j in range(n):
        a = 2 * math.pi * j / n
        verts.append((outer_r * math.cos(a), outer_r * math.sin(a), 0.0))
    for j in range(n):
        i0, i1 = j, (j + 1) % n
        o0, o1 = n + j, n + (j + 1) % n
        tris.append((i0, o0, i1))
        tris.append((i1, o0, o1))
    return verts, tris


def _arc_mesh(inner_r: float, outer_r: float, a0: float, a1: float, n: int = 16):
    """Flat arc-band (top face only, CCW winding → +Z normals). Requires a0 < a1."""
    if a0 > a1:
        a0, a1 = a1, a0
    verts, tris = [], []
    for j in range(n + 1):
        a = a0 + (a1 - a0) * j / n
        verts.append((inner_r * math.cos(a), inner_r * math.sin(a), 0.0))
    for j in range(n + 1):
        a = a0 + (a1 - a0) * j / n
        verts.append((outer_r * math.cos(a), outer_r * math.sin(a), 0.0))
    for j in range(n):
        i0, i1 = j, j + 1
        o0, o1 = (n + 1) + j, (n + 1) + j + 1
        tris.append((i0, o0, i1))
        tris.append((i1, o0, o1))
    return verts, tris


@dataclass
class FieldDimensions:
    """Parsed field dimensions from the RoboCup HSL JSON specification."""
    # Field
    length: float = 9.0
    width: float = 6.0
    line_width: float = 0.05
    penalty_mark_diameter: float = 0.1
    penalty_area_length: float = 1.0
    penalty_area_width: float = 3.0
    penalty_mark_distance: float = 1.5
    center_circle_diameter: float = 1.5
    border_strip_width: float = 1.0
    corner_arc_radius: float = 0.5
    # Goal
    goal_post_diameter: float = 0.1
    goal_height: float = 0.8
    goal_inner_width: float = 2.6
    goal_depth: float = 0.6
    # Derived
    has_goal_area: bool = False
    goal_area_length: float = 0.0
    goal_area_width: float = 0.0

    @classmethod
    def from_json(cls, path: str) -> "FieldDimensions":
        with open(path) as f:
            o = json.load(f)
        fd = cls()
        fld = o["field"]
        fd.length = fld["length"]
        fd.width = fld["width"]
        fd.line_width = fld["lineWidth"]
        if "penaltyMarkDiameter" in fld:
            fd.penalty_mark_diameter = fld["penaltyMarkDiameter"]
        elif "penaltyMarkSize" in fld:
            fd.penalty_mark_diameter = fld["penaltyMarkSize"]
        fd.penalty_area_length = fld["penaltyAreaLength"]
        fd.penalty_area_width = fld["penaltyAreaWidth"]
        fd.penalty_mark_distance = fld["penaltyMarkDistance"]
        fd.center_circle_diameter = fld["centerCircleDiameter"]
        fd.border_strip_width = fld["borderStripWidth"]
        if "cornerArcRadius" in fld:
            fd.corner_arc_radius = fld["cornerArcRadius"]
        if "goalAreaLength" in fld:
            fd.has_goal_area = True
            fd.goal_area_length = fld["goalAreaLength"]
            fd.goal_area_width = fld["goalAreaWidth"]
        g = o["goal"]
        fd.goal_post_diameter = g["postDiameter"]
        fd.goal_height = g["height"]
        fd.goal_inner_width = g["innerWidth"]
        fd.goal_depth = g["depth"]
        return fd

    @property
    def total_length(self):
        return self.length + 2 * self.border_strip_width

    @property
    def total_width(self):
        return self.width + 2 * self.border_strip_width

    @property
    def half_length(self):
        return self.length / 2

    @property
    def half_width(self):
        return self.width / 2

    @property
    def center_circle_radius(self):
        return self.center_circle_diameter / 2


class MuJoCoFieldGenerator:
    """Generates a MuJoCo XML model of the RoboCup soccer field."""

    def __init__(self, field: FieldDimensions):
        self.f = field
        self.line_height = 0.001  # Lines are very thin raised surfaces
        self.carpet_height = 0.01

    def generate(self, output_path: str, num_robots_per_team: int = 4,
                 robot_mjcf_path: str = None):
        root = ET.Element("mujoco", model="robocup_hsl_field")

        # Compiler settings
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")

        # Options
        ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81",
                      integrator="implicitfast")

        # Visual settings
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4",
                      diffuse="0.8 0.8 0.8")
        ET.SubElement(visual, "quality", shadowsize="4096")

        # Assets
        asset = ET.SubElement(root, "asset")
        # Green carpet texture
        ET.SubElement(asset, "texture", name="field_green", type="2d",
                      builtin="flat", rgb1="0.1 0.6 0.1", rgb2="0.1 0.55 0.1",
                      width="512", height="512")
        ET.SubElement(asset, "material", name="field_mat", texture="field_green",
                      texrepeat="4 4", specular="0.1", shininess="0.1")
        # White line material
        ET.SubElement(asset, "material", name="line_mat", rgba="1 1 1 1",
                      specular="0.0", shininess="0.0")
        # Goal post material
        ET.SubElement(asset, "material", name="goal_mat", rgba="1 1 1 1",
                      specular="0.3", shininess="0.5")
        # Ball material
        ET.SubElement(asset, "texture", name="ball_tex", type="2d",
                      builtin="checker", rgb1="1 1 1", rgb2="0.1 0.1 0.1",
                      width="64", height="64")
        ET.SubElement(asset, "material", name="ball_mat", texture="ball_tex",
                      texrepeat="4 4")
        # Skybox
        ET.SubElement(asset, "texture", name="skybox", type="skybox",
                      builtin="gradient", rgb1="0.4 0.6 0.9", rgb2="0.1 0.1 0.3",
                      width="512", height="512")

        # Default settings
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "geom", condim="3", friction="1.0 0.005 0.0001")

        # Worldbody
        worldbody = ET.SubElement(root, "worldbody")

        # Lighting
        ET.SubElement(worldbody, "light", name="overhead", pos="0 0 8",
                      dir="0 0 -1", diffuse="1 1 1", specular="0.3 0.3 0.3",
                      cutoff="60", directional="true")

        # Ground plane (carpet)
        self._add_carpet(worldbody)
        # Field lines
        self._add_field_lines(worldbody)
        # Goals
        self._add_goals(worldbody)
        # Center circle (approximated with segments)
        self._add_center_circle(worldbody)
        # Penalty marks
        self._add_penalty_marks(worldbody)
        # Ball
        self._add_ball(worldbody)

        # Robot placement markers (comments for where to include robots)
        self._add_robot_placement_comments(root, num_robots_per_team,
                                           robot_mjcf_path)

        # Contact exclusions
        contact = ET.SubElement(root, "contact")
        # Exclude ball-line contacts to avoid jitter
        ET.SubElement(contact, "exclude", body1="ball_body", body2="field_carpet")

        # Write XML
        xml_str = minidom.parseString(
            ET.tostring(root, encoding="unicode")
        ).toprettyxml(indent="  ")
        # Remove extra XML declaration
        lines = xml_str.split("\n")
        if lines[0].startswith("<?xml"):
            lines = lines[1:]
        xml_str = "\n".join(lines)

        with open(output_path, "w") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(xml_str)

        print(f"MuJoCo field model written to {output_path}")

    def _add_carpet(self, worldbody):
        """Add the green carpet ground."""
        body = ET.SubElement(worldbody, "body", name="field_carpet",
                             pos="0 0 0")
        ET.SubElement(body, "geom", name="carpet", type="box",
                      size=f"{self.f.total_length/2} {self.f.total_width/2} {self.carpet_height}",
                      pos=f"0 0 -{self.carpet_height}",
                      material="field_mat", conaffinity="1", contype="1")

    def _add_field_lines(self, worldbody):
        """Add all field boundary lines as thin white boxes."""
        lw = self.f.line_width
        lh = self.line_height
        hl = self.f.half_length
        hw = self.f.half_width
        h = lh / 2

        lines_body = ET.SubElement(worldbody, "body", name="field_lines",
                                   pos="0 0 0")

        # Touchlines (long sides)
        for sign, name in [(1, "touchline_pos"), (-1, "touchline_neg")]:
            ET.SubElement(lines_body, "geom", name=name, type="box",
                          size=f"{hl} {lw/2} {h}",
                          pos=f"0 {sign * hw} {h}",
                          material="line_mat", conaffinity="0", contype="0")

        # Goal lines (short sides)
        for sign, name in [(1, "goalline_pos"), (-1, "goalline_neg")]:
            ET.SubElement(lines_body, "geom", name=name, type="box",
                          size=f"{lw/2} {hw} {h}",
                          pos=f"{sign * hl} 0 {h}",
                          material="line_mat", conaffinity="0", contype="0")

        # Center line
        ET.SubElement(lines_body, "geom", name="center_line", type="box",
                      size=f"{lw/2} {hw} {h}",
                      pos=f"0 0 {h}",
                      material="line_mat", conaffinity="0", contype="0")

        # Penalty areas
        pa_l = self.f.penalty_area_length
        pa_w = self.f.penalty_area_width / 2
        for sign, side in [(1, "pos"), (-1, "neg")]:
            x_base = sign * hl
            x_front = sign * (hl - pa_l)
            # Front line of penalty area
            ET.SubElement(lines_body, "geom",
                          name=f"penalty_front_{side}", type="box",
                          size=f"{lw/2} {pa_w} {h}",
                          pos=f"{x_front} 0 {h}",
                          material="line_mat", conaffinity="0", contype="0")
            # Side lines of penalty area
            for y_sign, y_name in [(1, "top"), (-1, "bot")]:
                ET.SubElement(lines_body, "geom",
                              name=f"penalty_side_{side}_{y_name}", type="box",
                              size=f"{pa_l/2} {lw/2} {h}",
                              pos=f"{x_base - sign*pa_l/2} {y_sign * pa_w} {h}",
                              material="line_mat", conaffinity="0", contype="0")

        # Goal areas (if present)
        if self.f.has_goal_area:
            ga_l = self.f.goal_area_length
            ga_w = self.f.goal_area_width / 2
            for sign, side in [(1, "pos"), (-1, "neg")]:
                x_front = sign * (hl - ga_l)
                ET.SubElement(lines_body, "geom",
                              name=f"goalarea_front_{side}", type="box",
                              size=f"{lw/2} {ga_w} {h}",
                              pos=f"{x_front} 0 {h}",
                              material="line_mat", conaffinity="0", contype="0")
                for y_sign, y_name in [(1, "top"), (-1, "bot")]:
                    x_base = sign * hl
                    ET.SubElement(lines_body, "geom",
                                  name=f"goalarea_side_{side}_{y_name}",
                                  type="box",
                                  size=f"{ga_l/2} {lw/2} {h}",
                                  pos=f"{x_base - sign*ga_l/2} {y_sign * ga_w} {h}",
                                  material="line_mat", conaffinity="0",
                                  contype="0")

    def _add_center_circle(self, worldbody, num_segments=48):
        """Approximate center circle with line segments."""
        r = self.f.center_circle_radius
        lw = self.f.line_width
        lh = self.line_height
        h = lh / 2

        cc_body = ET.SubElement(worldbody, "body", name="center_circle",
                                pos="0 0 0")
        for i in range(num_segments):
            angle0 = 2 * math.pi * i / num_segments
            angle1 = 2 * math.pi * (i + 1) / num_segments
            x0, y0 = r * math.cos(angle0), r * math.sin(angle0)
            x1, y1 = r * math.cos(angle1), r * math.sin(angle1)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            seg_len = math.sqrt((x1 - x0)**2 + (y1 - y0)**2) / 2
            angle = math.atan2(y1 - y0, x1 - x0)

            ET.SubElement(cc_body, "geom", name=f"cc_seg_{i}", type="box",
                          size=f"{seg_len} {lw/2} {h}",
                          pos=f"{cx} {cy} {h}",
                          euler=f"0 0 {angle}",
                          material="line_mat", conaffinity="0", contype="0")

    def _add_penalty_marks(self, worldbody):
        """Add penalty marks and center mark as cylinders (disk marks)."""
        r = self.f.penalty_mark_diameter / 2
        h = self.line_height

        # Center mark
        ET.SubElement(worldbody, "geom", name="center_mark", type="cylinder",
                      size=f"{r} {h/2}",
                      pos=f"0 0 {h/2}",
                      material="line_mat", conaffinity="0", contype="0")

        # Penalty marks
        for sign, side in [(1, "pos"), (-1, "neg")]:
            x = sign * (self.f.half_length - self.f.penalty_mark_distance)
            ET.SubElement(worldbody, "geom",
                          name=f"penalty_mark_{side}", type="cylinder",
                          size=f"{r} {h/2}",
                          pos=f"{x} 0 {h/2}",
                          material="line_mat", conaffinity="0", contype="0")

    def _add_goals(self, worldbody):
        """Add goal structures (posts, crossbar, net approximation)."""
        gw = self.f.goal_inner_width / 2
        gd = self.f.goal_depth
        gh = self.f.goal_height
        pr = self.f.goal_post_diameter / 2

        for sign, side in [(1, "pos"), (-1, "neg")]:
            goal_body = ET.SubElement(worldbody, "body",
                                      name=f"goal_{side}",
                                      pos=f"{sign * self.f.half_length} 0 0")

            # Left post
            ET.SubElement(goal_body, "geom", name=f"post_left_{side}",
                          type="cylinder",
                          size=f"{pr} {gh/2}",
                          pos=f"{sign * gd/2} {gw} {gh/2}",
                          material="goal_mat",
                          conaffinity="1", contype="1")
            # Right post
            ET.SubElement(goal_body, "geom", name=f"post_right_{side}",
                          type="cylinder",
                          size=f"{pr} {gh/2}",
                          pos=f"{sign * gd/2} {-gw} {gh/2}",
                          material="goal_mat",
                          conaffinity="1", contype="1")
            # Crossbar
            ET.SubElement(goal_body, "geom", name=f"crossbar_{side}",
                          type="capsule",
                          size=f"{pr}",
                          fromto=f"{sign * gd/2} {-gw} {gh} "
                                 f"{sign * gd/2} {gw} {gh}",
                          material="goal_mat",
                          conaffinity="1", contype="1")
            # Back bar (bottom)
            ET.SubElement(goal_body, "geom", name=f"backbar_{side}",
                          type="capsule",
                          size=f"{pr * 0.5}",
                          fromto=f"{sign * gd} {-gw} 0 "
                                 f"{sign * gd} {gw} 0",
                          material="goal_mat",
                          conaffinity="1", contype="1")
            # Net approximation - back wall
            ET.SubElement(goal_body, "geom", name=f"net_back_{side}",
                          type="box",
                          size=f"{0.005} {gw} {gh/2}",
                          pos=f"{sign * gd} 0 {gh/2}",
                          rgba="0.8 0.8 0.8 0.3",
                          conaffinity="1", contype="1")
            # Net sides
            for y_sign, y_name in [(1, "left"), (-1, "right")]:
                ET.SubElement(goal_body, "geom",
                              name=f"net_side_{y_name}_{side}",
                              type="box",
                              size=f"{gd/2} {0.005} {gh/2}",
                              pos=f"{sign * gd/2} {y_sign * gw} {gh/2}",
                              rgba="0.8 0.8 0.8 0.3",
                              conaffinity="1", contype="1")

    def _add_ball(self, worldbody):
        """Add a size-1 soccer ball (diameter ~14cm)."""
        ball_body = ET.SubElement(worldbody, "body", name="ball_body",
                                  pos="0 0 0.07")
        ET.SubElement(ball_body, "joint", name="ball_free", type="free")
        ET.SubElement(ball_body, "geom", name="ball", type="sphere",
                      size="0.07", mass="0.2",
                      material="ball_mat",
                      friction="0.8 0.005 0.0001",
                      condim="4",
                      conaffinity="1", contype="1",
                      solref="0.01 1.0",
                      solimp="0.9 0.95 0.001")
        ET.SubElement(ball_body, "site", name="ball_site", size="0.001")

    def _add_robot_placement_comments(self, root, num_robots, robot_mjcf_path):
        """Add include directives or placement info for robots.

        Default behavior is to ONLY append a comment with kickoff positions;
        actual robot inclusion must be done at load time via MjSpec because
        K1_22dof.xml is a self-contained <mujoco> document and cannot be
        MJCF-included as a worldbody child.
        """
        if num_robots <= 0:
            root.append(ET.Comment(
                " Robots intentionally NOT included here — use MjSpec to "
                "attach K1 instances at load time. See "
                "scripts/debug_mujoco_scene.py / evaluation/evaluate.py "))
            return

        # Add comment with placement positions
        root.append(ET.Comment(
            " Robot placement positions (x, y, heading_rad) for each team "))

        # Calculate starting positions
        positions = self._get_kickoff_positions(num_robots)

        # If robot MJCF path provided, add includes
        if robot_mjcf_path:
            # Create a worldbody if it doesn't exist (though it should)
            worldbody = root.find("worldbody")
            if worldbody is None:
                worldbody = ET.SubElement(root, "worldbody")

            for team_idx, team in enumerate(["home", "away"]):
                for i, (x, y, heading) in enumerate(positions[team]):
                    # Create a body for the robot
                    robot_body = ET.SubElement(worldbody, "body", 
                                               name=f"{team}_player_{i}",
                                               pos=f"{x} {y} 0")
                    # Include the robot MJCF
                    include = ET.SubElement(robot_body, "include",
                                           file=robot_mjcf_path)
                    # Add heading as a comment or metadata
                    robot_body.append(ET.Comment(f" Heading: {heading:.2f} "))

    def _get_kickoff_positions(self, num_per_team: int = 4) -> dict:
        """Calculate standard kickoff positions for both teams."""
        hl = self.f.half_length
        hw = self.f.half_width

        # Home team defends negative-x goal, attacks positive-x
        # Away team defends positive-x goal, attacks negative-x
        home_positions = []
        away_positions = []

        if num_per_team >= 1:
            # Goalkeepers
            home_positions.append((-hl + 0.3, 0, 0))
            away_positions.append((hl - 0.3, 0, math.pi))

        if num_per_team >= 2:
            # Defenders
            home_positions.append((-hl/2, hw/4, 0))
            away_positions.append((hl/2, -hw/4, math.pi))

        if num_per_team >= 3:
            # Midfielders
            home_positions.append((-1.0, -hw/4, 0))
            away_positions.append((1.0, hw/4, math.pi))

        if num_per_team >= 4:
            # Strikers (near center, offset from center line)
            home_positions.append((-0.5, 0, 0))
            away_positions.append((0.5, 0, math.pi))

        for i in range(4, num_per_team):
            y_off = (i - 3) * 0.5 * (1 if i % 2 else -1)
            home_positions.append((-hl/3, y_off, 0))
            away_positions.append((hl/3, -y_off, math.pi))

        return {"home": home_positions, "away": away_positions}

    def get_field_info(self) -> dict:
        """Return field info dict for use in training environments."""
        return {
            "length": self.f.length,
            "width": self.f.width,
            "half_length": self.f.half_length,
            "half_width": self.f.half_width,
            "total_length": self.f.total_length,
            "total_width": self.f.total_width,
            "penalty_area_length": self.f.penalty_area_length,
            "penalty_area_width": self.f.penalty_area_width,
            "penalty_mark_distance": self.f.penalty_mark_distance,
            "center_circle_radius": self.f.center_circle_radius,
            "goal_width": self.f.goal_inner_width,
            "goal_height": self.f.goal_height,
            "goal_depth": self.f.goal_depth,
            "border_strip_width": self.f.border_strip_width,
        }


class GenesisFieldBuilder:
    """Builds a soccer field scene for the Genesis simulator."""

    def __init__(self, field: FieldDimensions):
        self.f = field

    def build_scene_code(self) -> str:
        """Generate Python code to build the field in Genesis.

        Colors go through `surface=gs.surfaces.Default(color=...)`. Genesis's
        `materials.Rigid` is a *physics* material (friction, restitution) and
        does NOT accept color — passing one raises a pydantic ValidationError,
        which is what the original auto-generated builder did.
        """
        hl = self.f.half_length
        hw = self.f.half_width
        gw = self.f.goal_inner_width / 2
        gd = self.f.goal_depth
        gh = self.f.goal_height
        pr = self.f.goal_post_diameter / 2
        lw = self.f.line_width
        tl = self.f.total_length
        tw = self.f.total_width
        ccr = self.f.center_circle_radius
        pal = self.f.penalty_area_length
        paw = self.f.penalty_area_width / 2

        return f'''
def build_soccer_field(scene, physics_only: bool = False):
    """Add RoboCup HSL soccer field entities to a Genesis scene.

    Field dimensions: {self.f.length}m x {self.f.width}m
    Total with border: {tl}m x {tw}m

    Args:
        physics_only: When True, skip all visual-only entities (field lines,
            center circle, goal nets). Reduces entity count from ~90 to ~9,
            critical for vectorised training where each entity is replicated
            per env.

    Returns the carpet entity so callers can identify it (e.g. for contact
    filtering). Field lines and the center circle are visual-only
    (collision=False) — they should not perturb robot contact dynamics.
    """
    import genesis as gs
    import math
    import os

    green = gs.surfaces.Default(color=(0.18, 0.45, 0.18, 1.0), roughness=0.9)
    white = gs.surfaces.Default(color=(0.95, 0.95, 0.95, 1.0), roughness=0.6)
    post  = gs.surfaces.Default(color=(0.95, 0.95, 0.95, 1.0), roughness=0.4)
    net   = gs.surfaces.Default(color=(0.85, 0.85, 0.85, 0.35), roughness=0.9)

    # ── Green carpet ──────────────────────────────────────────────
    carpet = scene.add_entity(
        gs.morphs.Box(size=({tl}, {tw}, 0.02), pos=(0, 0, -0.01),
                      fixed=True, collision=True),
        surface=green,
    )

    if not physics_only:
        # ── Field lines (thin white boxes, visual only) ───────────────
        line_h = 0.003
        line_z = line_h / 2 + 0.002  # sit just above carpet top (z=0)

        # Touchlines
        for sign in [1, -1]:
            scene.add_entity(
                gs.morphs.Box(size=({hl * 2}, {lw}, line_h),
                              pos=(0, sign * {hw}, line_z),
                              fixed=True, collision=False),
                surface=white,
            )

        # Goal lines
        for sign in [1, -1]:
            scene.add_entity(
                gs.morphs.Box(size=({lw}, {hw * 2}, line_h),
                              pos=(sign * {hl}, 0, line_z),
                              fixed=True, collision=False),
                surface=white,
            )

        # Center line
        scene.add_entity(
            gs.morphs.Box(size=({lw}, {hw * 2}, line_h),
                          pos=(0, 0, line_z),
                          fixed=True, collision=False),
            surface=white,
        )

        # Penalty areas
        for sign in [1, -1]:
            scene.add_entity(
                gs.morphs.Box(size=({lw}, {paw * 2}, line_h),
                              pos=(sign * ({hl} - {pal}), 0, line_z),
                              fixed=True, collision=False),
                surface=white,
            )
            for y_sign in [1, -1]:
                scene.add_entity(
                    gs.morphs.Box(size=({pal}, {lw}, line_h),
                                  pos=(sign * ({hl} - {pal}/2), y_sign * {paw}, line_z),
                                  fixed=True, collision=False),
                    surface=white,
                )

        # Center circle (smooth ring mesh)
        _d = os.path.dirname(os.path.abspath(__file__))
        scene.add_entity(
            gs.morphs.Mesh(file=os.path.join(_d, "meshes", "center_circle.ply"),
                           pos=(0, 0, line_z), fixed=True, collision=False),
            surface=white,
        )

        # Penalty marks + center mark (thin discs)
        pmr = {self.f.penalty_mark_diameter / 2}
        scene.add_entity(
            gs.morphs.Cylinder(radius=pmr, height=line_h,
                               pos=(0, 0, line_z), fixed=True, collision=False),
            surface=white,
        )
        for sign in [1, -1]:
            scene.add_entity(
                gs.morphs.Cylinder(radius=pmr, height=line_h,
                                   pos=(sign * ({hl} - {self.f.penalty_mark_distance}), 0, line_z),
                                   fixed=True, collision=False),
                surface=white,
            )

        # Corner arcs (smooth quarter-circle arc meshes)
        for sx, sy, tag in [(1, 1, "pp"), (1, -1, "pn"), (-1, 1, "np"), (-1, -1, "nn")]:
            scene.add_entity(
                gs.morphs.Mesh(file=os.path.join(_d, "meshes", f"corner_arc_{{tag}}.ply"),
                               pos=(sx * {hl}, sy * {hw}, line_z),
                               fixed=True, collision=False),
                surface=white,
            )

    # ── Goals (collidable so the ball bounces off) ────────────────
    for sign in [1, -1]:
        gx = sign * {hl}
        for y_sign in [1, -1]:
            scene.add_entity(
                gs.morphs.Cylinder(radius={pr}, height={gh},
                                   pos=(gx + sign * {gd/2}, y_sign * {gw}, {gh/2}),
                                   fixed=True, collision=True),
                surface=post,
            )
        scene.add_entity(
            gs.morphs.Box(size=({pr * 2}, {gw * 2 + pr * 2}, {pr * 2}),
                          pos=(gx + sign * {gd/2}, 0, {gh}),
                          fixed=True, collision=True),
            surface=post,
        )
        if not physics_only:
            scene.add_entity(
                gs.morphs.Box(size=(0.01, {gw * 2}, {gh}),
                              pos=(gx + sign * {gd}, 0, {gh / 2}),
                              fixed=True, collision=True),
                surface=net,
            )

    return carpet
'''


def generate_field_assets(json_path: str, output_dir: str):
    """Main entry point: generate both MuJoCo and Genesis field assets.

    Note: the MuJoCo field XML is intentionally produced with NO robot
    includes. Including K1_22dof.xml directly fails — it's a full
    <mujoco> document with its own <compiler>/<worldbody>/ground plane.
    Multi-robot scenes must be assembled with MjSpec (see
    scripts/debug_mujoco_scene.py) or by attaching robots at load time.
    """
    os.makedirs(output_dir, exist_ok=True)

    field = FieldDimensions.from_json(json_path)
    print(f"Field: {field.length}m x {field.width}m "
          f"(total: {field.total_length}m x {field.total_width}m)")

    # MuJoCo XML — field + ball only, no robot includes
    mujoco_gen = MuJoCoFieldGenerator(field)
    mujoco_path = os.path.join(output_dir, "field_robocup.xml")
    mujoco_gen.generate(mujoco_path, num_robots_per_team=0)

    # PLY mesh files for smooth circle / arc markings
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)
    lw2 = field.line_width / 2
    verts, tris = _ring_mesh(field.center_circle_radius - lw2,
                             field.center_circle_radius + lw2)
    _write_ply(os.path.join(meshes_dir, "center_circle.ply"), verts, tris)
    # Corner arc angles (CCW, a0 < a1): derived from original CW sweeps per corner
    _corner_arc_angles = {
        "pp": (-math.pi,      -math.pi / 2),   # corner (+hl, +hw)
        "pn": ( math.pi / 2,  math.pi),         # corner (+hl, -hw)
        "np": (-math.pi / 2,  0.0),             # corner (-hl, +hw)
        "nn": ( 0.0,          math.pi / 2),     # corner (-hl, -hw)
    }
    for tag, (a0, a1) in _corner_arc_angles.items():
        verts, tris = _arc_mesh(field.corner_arc_radius - lw2,
                                field.corner_arc_radius + lw2, a0, a1)
        _write_ply(os.path.join(meshes_dir, f"corner_arc_{tag}.ply"), verts, tris)
    print(f"Field meshes written to {meshes_dir}")

    # Genesis builder code
    genesis_gen = GenesisFieldBuilder(field)
    genesis_path = os.path.join(output_dir, "field_genesis_builder.py")
    with open(genesis_path, "w") as f:
        f.write('"""Auto-generated Genesis field builder."""\n')
        f.write(genesis_gen.build_scene_code())
    print(f"Genesis field builder written to {genesis_path}")

    # Field info JSON for training
    info = mujoco_gen.get_field_info()
    info_path = os.path.join(output_dir, "field_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Field info written to {info_path}")

    return field


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate RoboCup field models")
    parser.add_argument("field_json", help="Path to field dimensions JSON")
    parser.add_argument("-o", "--output-dir", default="models/field",
                        help="Output directory")
    args = parser.parse_args()
    generate_field_assets(args.field_json, args.output_dir)
