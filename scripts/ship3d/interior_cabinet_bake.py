#!/usr/bin/env python3
"""Cabinet/oblique interior — approach #1 from the research.

ONE fixed orthographic oblique projection for the WHOLE scene (no per-side
mirrored cameras — that was the un-tileable mistake). Every wall leans the
same way, so it reads as a room viewed from one front-right corner, and it
tiles because there is a single consistent projection. Walls are thick
(~0.5u ≈ 16px, per the OpenGameArt "Perspective Walls Template").

This first pass renders a COHERENT room (walls line up by construction) so
we can judge the look; the tile-slicing for the engine follows once the
look is agreed.

Run:  scripts/.blender_venv/bin/python scripts/ship3d/interior_cabinet_bake.py
"""
import math
import os
import bpy
import blender_gen as B

OUT = os.path.join(os.path.dirname(__file__), "out")
RES = 900

FLOOR = (0.30, 0.34, 0.40)
FLOOR_HI = (0.37, 0.41, 0.48)
WALL = (0.50, 0.53, 0.59)
WALL_TOP = (0.63, 0.66, 0.72)
TRIM = (0.25, 0.72, 0.86)
WOOD = (0.55, 0.40, 0.26)
METAL = (0.70, 0.73, 0.78)

# Cabinet projection = a SHEAR of height into the screen, NOT a camera
# rotation. Camera looks straight down (floor stays an axis-aligned square
# grid); height z is skewed up-and-slightly-right so walls lean consistently.
# SHX/SHY are the screen offset per unit of height (cabinet ≈ 0.5 total).
SHX, SHY = 0.26, 0.42
WALL_H = 1.0     # wall height (units)
WALL_T = 0.5     # wall thickness ≈ 16px on a 32px tile


def build_room(W=9, D=6):
    B.reset()
    mf = B.toon_material("floor", FLOOR)
    mfh = B.toon_material("floorhi", FLOOR_HI)
    mw = B.toon_material("wall", WALL)
    mwt = B.toon_material("walltop", WALL_TOP)
    mt = B.glow_material("trim", TRIM, strength=2.6)
    mwood = B.toon_material("wood", WOOD)
    mmetal = B.toon_material("metal", METAL)

    # floor grid
    B.add_box("floor", (0, 0, -0.05), (W, D, 0.1), mf)
    for gx in range(-(W // 2), W // 2 + 1):
        B.add_box(f"sx{gx}", (gx, 0, 0.005), (0.03, D, 0.01), mt)
    for gy in range(-(D // 2), D // 2 + 1):
        B.add_box(f"sy{gy}", (0, gy, 0.005), (W, 0.03, 0.01), mt)

    t = WALL_T
    hw, hd = W / 2, D / 2

    def wall_run(name, cx, cy, sx, sy):
        B.add_box(name, (cx, cy, WALL_H / 2), (sx, sy, WALL_H), mw, bevel=0.03)
        B.add_box(name + "_top", (cx, cy, WALL_H + 0.03), (sx, sy, 0.08), mwt)

    # perimeter walls (north=+y far, south=near open? keep south open for the view)
    wall_run("back", 0, hd - t / 2, W, t)
    wall_run("left", -hw + t / 2, 0, t, D)
    wall_run("right", hw - t / 2, 0, t, D)
    # base trim glow on the room side of the far wall
    B.add_box("trimN", (0, hd - t, 0.1), (W * 0.94, 0.05, 0.06), mt)

    # a counter + shelf so scale + occlusion read
    B.add_box("counter", (-W / 4, hd - t - 0.7, 0.45), (3.0, 0.8, 0.9), mwood, bevel=0.05)
    B.add_box("counter_top", (-W / 4, hd - t - 0.7, 0.92), (3.0, 0.8, 0.06),
              B.toon_material("woodtop", (0.68, 0.52, 0.34)))
    B.add_box("shelf", (W / 4, hd - t - 0.4, 0.8), (2.2, 0.5, 1.6), mmetal, bevel=0.04)
    B.add_box("crate", (-hw + 1.3, -hd + 1.4, 0.35), (0.7, 0.7, 0.7), mwood, bevel=0.06)


def apply_cabinet_shear():
    """Skew every vertex: x += z*SHX, y += z*SHY (world space). Floor (z≈0)
    stays put — square grid — while wall height leans up-and-right."""
    from mathutils import Matrix
    S = Matrix(((1, 0, SHX, 0), (0, 1, SHY, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.matrix_world = S @ obj.matrix_world


def set_topdown():
    scene = bpy.context.scene
    cam = scene.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 10.5
    cam.constraints.clear()
    cam.location = (0, 0, 20)
    cam.rotation_euler = (0, 0, 0)   # straight down → floor is axis-aligned


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_room()
    apply_cabinet_shear()
    B.setup_scene(ortho=10.5, res=RES, freestyle_thick=1.4)
    set_topdown()
    path = os.path.join(OUT, "_interior_cabinet_room.png")
    B.render_to(path)
    print("wrote", os.path.abspath(path))
