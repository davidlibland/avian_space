#!/usr/bin/env python3
"""Bake a tileable interior wall/floor set in Blender (toon + ink).

The trick that makes converging walls tileable: decouple the square-tiling
CAP (rendered straight-down, so it composites on the grid like any tile)
from the FACES (rendered per-side from a front-oblique camera, so the far
wall shows its front face and the left/right walls show inner faces that
lean toward the room). Each face is a fixed per-side sprite → the whole
left edge shares one look, the whole right edge another → it tiles.

Pieces → out/tiles/*.png:
  floor           32x32   straight-down floor
  cap             32x32   straight-down wall top (bright bevelled cap)
  face_n          32xFH   north wall front face (hangs south into room)
  face_w          FWxFH   west (left) wall inner face, leaning right
  face_e          FWxFH   east (right) wall inner face, leaning left

Run:  scripts/.blender_venv/bin/python scripts/ship3d/interior_tileset_bake.py
"""
import math
import os
import bpy
import blender_gen as B

OUT = os.path.join(os.path.dirname(__file__), "out", "tiles")
SS = 6            # supersample: render at 32*SS then downscale in the compositor
TILE = 32
FH = 46          # face height (px, ~1.4 tiles)
FW = 18          # side-face width (px) — the sliver of inner wall seen at an angle

FLOOR = (0.30, 0.34, 0.40)
FLOOR_HI = (0.38, 0.42, 0.49)
WALL = (0.52, 0.55, 0.61)
WALL_HI = (0.64, 0.67, 0.73)
WALL_LO = (0.34, 0.37, 0.43)
TRIM = (0.25, 0.72, 0.86)


def cam_ortho_down(scale):
    """Straight-down ortho — square, tileable footprint."""
    sc = bpy.context.scene
    cam = sc.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    cam.constraints.clear()
    cam.location = (0, 0, 10)
    cam.rotation_euler = (0, 0, 0)


def cam_front_oblique(scale, elev=48.0, azim=0.0, target=(0, 0, 0.5)):
    """Front-oblique ortho — shows a wall's face; azim leans the view."""
    sc = bpy.context.scene
    cam = sc.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    cam.constraints.clear()
    E, A = math.radians(elev), math.radians(azim)
    d = 30
    cam.location = (d * math.cos(E) * math.sin(A), -d * math.cos(E) * math.cos(A), d * math.sin(E))
    tgt = bpy.data.objects.new("tgt", None)
    tgt.location = target
    sc.collection.objects.link(tgt)
    c = cam.constraints.new("TRACK_TO")
    c.target = tgt


def render(path, res_x, res_y):
    sc = bpy.context.scene
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.render.filepath = os.path.abspath(path)
    bpy.ops.render.render(write_still=True)


def bake_floor():
    B.reset()
    mf = B.toon_material("floor", FLOOR)
    mh = B.toon_material("floorhi", FLOOR_HI)
    mt = B.glow_material("trim", TRIM, strength=2.5)
    B.add_box("floor", (0, 0, 0), (1.0, 1.0, 0.1), mf, bevel=0.0)
    # 2x2 panel with a faint raised centre + seam glow
    B.add_box("panel", (0, 0, 0.055), (0.9, 0.9, 0.02), mh, bevel=0.06)
    B.add_box("seamx", (0, 0, 0.06), (0.02, 0.94, 0.01), mt, bevel=0.0)
    B.add_box("seamy", (0, 0, 0.06), (0.94, 0.02, 0.01), mt, bevel=0.0)
    B.setup_scene(ortho=1.0, res=TILE * SS, freestyle_thick=1.2)
    cam_ortho_down(1.0)
    render(os.path.join(OUT, "floor.png"), TILE * SS, TILE * SS)


def bake_cap():
    B.reset()
    mw = B.toon_material("wall", WALL)
    mh = B.toon_material("wallhi", WALL_HI)
    B.add_box("cap", (0, 0, 0), (1.0, 1.0, 0.2), mw, bevel=0.05)
    B.add_box("caphi", (0, 0, 0.11), (0.86, 0.86, 0.04), mh, bevel=0.08)
    B.setup_scene(ortho=1.0, res=TILE * SS, freestyle_thick=1.2)
    cam_ortho_down(1.0)
    render(os.path.join(OUT, "cap.png"), TILE * SS, TILE * SS)


# The isometric bake, per the design correction: ONE fixed elevation and a
# constant camera z, with the camera placed a FIXED distance along each
# wall's room-facing normal. So every wall is viewed straight down its own
# normal at the same angle → each direction's faces are identical and tile
# within a run, and the left wall is seen from its right / the right from
# its left (the convergence) purely from where its normal points.
ISO_ELEV = 34.0  # fixed for every wall


def bake_face(name, run_axis, normal_azim):
    """`run_axis`: 'x' wall runs east-west (N/S edges), 'y' runs north-south
    (E/W edges). `normal_azim`: azimuth of the room-facing normal (deg) — the
    camera sits along it at the fixed iso elevation."""
    B.reset()
    mw = B.toon_material("wall", WALL)
    mh = B.toon_material("wallhi", WALL_HI)
    mt = B.glow_material("trim", TRIM, strength=3.0)
    mf = B.toon_material("floor", FLOOR)
    if run_axis == "x":
        wall_size = (1.0, 0.35, 1.4)
        base = (0, -0.18, 0.09)
        base_size = (0.96, 0.02, 0.18)
        apron = (0, -0.9, -0.05)
    else:  # 'y'
        wall_size = (0.35, 1.0, 1.4)
        base = (-0.18, 0, 0.09) if normal_azim < 0 else (0.18, 0, 0.09)
        base_size = (0.02, 0.96, 0.18)
        apron = (0.9 if normal_azim > 0 else -0.9, 0, -0.05)
    B.add_box("wall", (0, 0, 0.7), wall_size, mw, bevel=0.03)
    B.add_box("wallcap", (0, 0, 1.42), (wall_size[0], wall_size[1], 0.06), mh, bevel=0.0)
    B.add_box("wallbase", base, base_size, mt, bevel=0.0)  # room-facing base glow
    B.add_box("apron", apron, (1.4, 1.4, 0.1), mf, bevel=0.0)
    B.setup_scene(ortho=2.4, res=TILE * SS, freestyle_thick=1.2)
    E, A = math.radians(ISO_ELEV), math.radians(normal_azim)
    d = 30
    sc = bpy.context.scene
    cam = sc.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.4
    cam.constraints.clear()
    # Camera along the room-facing normal, constant z.
    cam.location = (d * math.cos(E) * math.sin(A), -d * math.cos(E) * math.cos(A), d * math.sin(E))
    tgt = bpy.data.objects.new("tgt", None)
    tgt.location = (0, 0, 0.6)
    sc.collection.objects.link(tgt)
    c = cam.constraints.new("TRACK_TO")
    c.target = tgt
    r = TILE * SS
    render(os.path.join(OUT, f"{name}.png"), r, r)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    bake_floor()
    bake_cap()
    # normal_azim = direction the room-facing wall surface points, so the
    # camera views it head-on at the fixed iso elevation.
    bake_face("face_n", "x", 0.0)     # far wall: room to south → south face
    bake_face("face_w", "y", 90.0)    # left wall: room to east → east face (cam to its right)
    bake_face("face_e", "y", -90.0)   # right wall: room to west → west face (cam to its left)
    print("baked tiles →", os.path.abspath(OUT))
