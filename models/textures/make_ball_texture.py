#!/usr/bin/env python3
"""
Generate a Telstar-style (pentagon/hexagon) soccer ball texture.

The output is an equirectangular RGB PNG that can be wrapped onto a sphere
in MuJoCo (<texture type="2d">) and used as a `diffuse_texture` for the
ball in Genesis. The pattern is derived from a truncated icosahedron
unfolded into spherical coordinates and rasterised.

This is deliberately self-contained: no external textures to download,
no licensing concerns, regenerates deterministically.

Usage:
    python -m models.textures.make_ball_texture --out models/textures/ball.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _icosahedron_vertices():
    """12 vertices of a regular icosahedron, on the unit sphere."""
    phi = (1.0 + math.sqrt(5)) / 2.0
    raw = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            raw.append((0, s1, s2 * phi))
            raw.append((s1, s2 * phi, 0))
            raw.append((s1 * phi, 0, s2))
    v = np.array(raw, dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _icosahedron_face_centroids():
    """20 face centroids of a regular icosahedron, on the unit sphere.

    Each face is a triangle of three icosahedron vertices; the centroid is
    the (re-normalised) average. We detect faces by finding all triples of
    vertices that are mutually closer than a small angle threshold.
    """
    v = _icosahedron_vertices()
    # Pairwise cosine distances; an icosahedron edge has cos(theta) = 1/sqrt(5).
    cos_edge = 1.0 / math.sqrt(5)
    cos_mat = v @ v.T
    # Slightly relaxed tolerance to allow for floating noise
    edge_mask = cos_mat > (cos_edge - 1e-6)
    n = v.shape[0]
    faces = set()
    for i in range(n):
        nbrs = [j for j in range(n) if j != i and edge_mask[i, j]]
        for jj in range(len(nbrs)):
            for kk in range(jj + 1, len(nbrs)):
                j, k = nbrs[jj], nbrs[kk]
                if edge_mask[j, k]:
                    faces.add(tuple(sorted((i, j, k))))
    centroids = []
    for f in faces:
        c = v[list(f)].sum(axis=0)
        c /= np.linalg.norm(c)
        centroids.append(c)
    return np.array(centroids, dtype=np.float64)


def _truncated_icosahedron_face_centers():
    """Return (12 pentagon centers, 20 hexagon centers) on the unit sphere.

    For a truncated icosahedron, pentagons sit at the icosahedron's vertices
    and hexagons sit at the icosahedron's triangular-face centroids. Both
    sets together are the 32 face directions of the soccer-ball polyhedron.
    """
    pent = _icosahedron_vertices()
    hex_dirs = _icosahedron_face_centroids()
    return pent, hex_dirs


def build_texture(width: int = 1024, height: int = 512,
                  pent_color=(20, 20, 20), hex_color=(245, 245, 245),
                  seam_color=(40, 40, 40),
                  pent_radius: float = 0.31,
                  hex_radius: float = 0.355,
                  seam_thickness: float = 0.008) -> np.ndarray:
    """Render an equirectangular RGB image of the ball.

    For every texel (u,v) we project to a unit sphere direction d and
    measure the angular distance to the nearest pentagon / hexagon center.
    If we're INSIDE a face we paint pent/hex color; near the boundary (the
    "seam") we paint dark.
    """
    pent_dirs, hex_dirs = _truncated_icosahedron_face_centers()

    # Build (H, W) grid of sphere directions
    ys = (0.5 - (np.arange(height) + 0.5) / height) * math.pi      # lat
    xs = ((np.arange(width) + 0.5) / width - 0.5) * 2 * math.pi    # lon
    lat, lon = np.meshgrid(ys, xs, indexing="ij")
    cz = np.cos(lat)
    dx = cz * np.cos(lon)
    dy = cz * np.sin(lon)
    dz = np.sin(lat)
    dirs = np.stack([dx, dy, dz], axis=-1)  # (H, W, 3)
    dirs_flat = dirs.reshape(-1, 3)

    # Cosine of angular distance to each face center
    cos_pent = dirs_flat @ pent_dirs.T  # (HW, n_pent)
    cos_hex = dirs_flat @ hex_dirs.T    # (HW, n_hex)
    nearest_pent_cos = cos_pent.max(axis=1)
    nearest_hex_cos = cos_hex.max(axis=1)

    inside_pent = nearest_pent_cos > math.cos(pent_radius)
    inside_hex = nearest_hex_cos > math.cos(hex_radius)

    # Seam: close to the boundary of EITHER region
    near_pent_edge = (nearest_pent_cos > math.cos(pent_radius + seam_thickness)) & \
                     (nearest_pent_cos < math.cos(pent_radius - seam_thickness * 0.6))
    near_hex_edge = (nearest_hex_cos > math.cos(hex_radius + seam_thickness)) & \
                    (nearest_hex_cos < math.cos(hex_radius - seam_thickness * 0.6))
    seam = near_pent_edge | near_hex_edge

    # Compose colors
    rgb = np.zeros((dirs_flat.shape[0], 3), dtype=np.uint8)
    rgb[:] = hex_color  # default: hex / white background
    rgb[inside_hex] = hex_color
    rgb[inside_pent] = pent_color
    rgb[seam] = seam_color

    return rgb.reshape(height, width, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="models/textures/ball.png")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=512)
    args = p.parse_args()

    img = build_texture(args.width, args.height)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(out)
    except ImportError:
        # Fallback to imageio if PIL is missing
        import imageio.v2 as imageio
        imageio.imwrite(str(out), img)
    print(f"Wrote {out} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
