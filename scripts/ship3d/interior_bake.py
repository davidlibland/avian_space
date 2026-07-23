#!/usr/bin/env python3
"""Blender bake of an interior room — the 3D answer to the PIL prototype.

Models a station shop as real geometry (floor + three walls with height and
thickness + a counter) and renders it under a gentle PERSPECTIVE camera, so
the far wall shows its front face and the side walls show their inner faces
converging inward — the limezu/moderninteriors depth, physically correct
instead of faked. Same toon materials + Freestyle ink rig as the ships and
buildings, so it matches the rest of the game.

This is a LOOK TEST (one composited room to `out/`), not yet a tileset —
we decide how to slice it into edge pieces after seeing it.

Run:  scripts/.blender_venv/bin/python scripts/ship3d/interior_bake.py
"""
import math
import os
import bpy
import blender_gen as B

OUT = os.path.join(os.path.dirname(__file__), "out")
RES = 900

# palette (toon)
FLOOR = (0.30, 0.34, 0.40)
WALL = (0.52, 0.55, 0.60)
WALL_TOP = (0.62, 0.65, 0.71)
TRIM = (0.20, 0.62, 0.78)
WOOD = (0.55, 0.40, 0.26)
WOOD_TOP = (0.68, 0.52, 0.34)
METAL = (0.70, 0.73, 0.78)


def build_room():
    B.reset()
    m_floor = B.toon_material("floor", FLOOR)
    m_wall = B.toon_material("wall", WALL)
    m_walltop = B.toon_material("walltop", WALL_TOP)
    m_trim = B.glow_material("trim", TRIM, strength=3.0)
    m_wood = B.toon_material("wood", WOOD)
    m_woodtop = B.toon_material("woodtop", WOOD_TOP)
    m_metal = B.toon_material("metal", METAL)

    # Room footprint in tile units (1 unit = 1 tile).
    W, D = 9.0, 6.0
    wall_h = 1.5
    t = 0.35  # wall thickness

    # Floor
    B.add_box("floor", (0, 0, -0.05), (W, D, 0.1), m_floor, bevel=0.0)
    # floor panel seams (thin insets) — a few glowing guide strips
    for gx in (-W / 4, W / 4):
        B.add_box("seam", (gx, 0, 0.005), (0.03, D * 0.9, 0.01), m_trim, bevel=0.0)

    # Three walls (back = +Y/far, left = -X, right = +X); front open toward cam.
    def wall(name, loc, size):
        B.add_box(name, (loc[0], loc[1], wall_h / 2), (size[0], size[1], wall_h), m_wall,
                  bevel=0.03)
        # brighter cap on top
        B.add_box(name + "_cap", (loc[0], loc[1], wall_h + 0.02), (size[0], size[1], 0.08),
                  m_walltop, bevel=0.0)
        # a glowing trim strip along the base, room-facing
        return

    back_y = D / 2 - t / 2
    wall("back", (0, back_y), (W, t))
    wall("left", (-W / 2 + t / 2, 0), (t, D))
    wall("right", (W / 2 - t / 2, 0), (t, D))
    # base trim glow along the far wall
    B.add_box("trim_back", (0, back_y - t / 2, 0.12), (W * 0.94, 0.04, 0.06), m_trim, bevel=0.0)

    # Back counter (wood) with a metal top lip
    cx, cy = -W / 4, back_y - t - 0.7
    B.add_box("counter", (cx, cy, 0.45), (3.0, 0.8, 0.9), m_wood, bevel=0.05)
    B.add_box("counter_top", (cx, cy, 0.92), (3.0, 0.8, 0.06), m_woodtop, bevel=0.0)

    # Shelf rack (metal) against the far wall, other side
    sx = W / 4
    B.add_box("shelf", (sx, back_y - t - 0.4, 0.8), (2.2, 0.5, 1.6), m_metal, bevel=0.04)
    for sh in (0.5, 1.0, 1.4):
        B.add_box(f"shelf_l{sh}", (sx, back_y - t - 0.15, sh), (2.0, 0.05, 0.05), m_walltop,
                  bevel=0.0)

    # A crate near the front-left for scale
    B.add_box("crate", (-W / 2 + 1.2, -D / 2 + 1.3, 0.35), (0.7, 0.7, 0.7), m_wood, bevel=0.06)


def setup_perspective(elev_deg=52.0, dist=17.0, lens=58.0):
    """Gentle perspective from above-front so side walls converge inward."""
    scene = bpy.context.scene
    cam = scene.camera
    cam.data.type = "PERSP"
    cam.data.lens = lens  # longer lens = subtler convergence (limezu is gentle)
    E = math.radians(elev_deg)
    cam.constraints.clear()
    cam.location = (0.0, -dist * math.cos(E), dist * math.sin(E))
    tgt = bpy.data.objects.new("tgt", None)
    tgt.location = (0, 0.4, 0.6)
    scene.collection.objects.link(tgt)
    c = cam.constraints.new("TRACK_TO")
    c.target = tgt


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_room()
    B.setup_scene(ortho=14.0, res=RES, freestyle_thick=1.4)
    setup_perspective()
    path = os.path.join(OUT, "_interior_room_blender.png")
    B.render_to(path)
    print("wrote", os.path.abspath(path))
