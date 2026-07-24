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


# PERSPECTIVE camera, orientation FIXED (azimuth 0, no yaw), translated a
# little to the RIGHT and ABOVE the tile it renders. The look direction never
# rotates about Z; the depth comes purely from the camera sitting right-and-
# above each tile under perspective, so we catch the tile's right + top faces.
# Because the camera moves WITH each tile (same relative offset every time),
# every tile renders in an identical local frame — no global shear to stagger
# straight runs — and pieces tile.
CAM_ELEV = 50.0     # "above" — height of the look direction
CAM_RIGHT = 1.7     # "a little to the right" — lateral camera offset (units)
CAM_LENS = 40.0     # perspective focal length (mm)
CAM_DIST = 12.0     # how close the camera sits (stronger perspective when small)
CAM_TARGET_Z = 0.55


def set_camera():
    """Perspective, look direction fixed at azimuth 0 / elevation CAM_ELEV,
    camera translated right by CAM_RIGHT. Camera AND its aim-point shift right
    together, so the view direction stays azimuth 0 (no Z rotation) — the tile
    just sits left of the view axis and is seen from its right."""
    E = math.radians(CAM_ELEV)
    sc = bpy.context.scene
    cam = sc.camera
    cam.data.type = "PERSP"
    cam.data.lens = CAM_LENS
    cam.constraints.clear()
    cam.location = (CAM_RIGHT, -CAM_DIST * math.cos(E), CAM_DIST * math.sin(E))
    tgt = bpy.data.objects.new("tgt", None)
    tgt.location = (CAM_RIGHT, 0, CAM_TARGET_Z)  # aim shifted right too → yaw 0
    sc.collection.objects.link(tgt)
    c = cam.constraints.new("TRACK_TO")
    c.target = tgt


def bake_face(name, run_axis, room_dir):
    """`run_axis`: 'x' wall runs east-west (a N/S room edge), 'y' runs
    north-south (an E/W edge). `room_dir`: unit (dx,dy) toward the room, so
    the apron + base glow sit on the room-facing side. All from the one
    fixed front-right camera."""
    B.reset()
    mw = B.toon_material("wall", WALL)
    mh = B.toon_material("wallhi", WALL_HI)
    mt = B.glow_material("trim", TRIM, strength=3.0)
    mf = B.toon_material("floor", FLOOR)
    if run_axis == "x":
        wall_size = (1.0, 0.35, 1.4)
    else:
        wall_size = (0.35, 1.0, 1.4)
    dx, dy = room_dir
    _ = mf  # apron dropped — walls bake on transparency, floor drawn separately
    B.add_box("wall", (0, 0, 0.7), wall_size, mw, bevel=0.03)
    B.add_box("wallcap", (0, 0, 1.42), (wall_size[0], wall_size[1], 0.06), mh, bevel=0.0)
    # base glow on the room-facing edge
    B.add_box("wallbase", (dx * 0.19, dy * 0.19, 0.06),
              (0.96 if dx == 0 else 0.03, 0.96 if dy == 0 else 0.03, 0.12), mt, bevel=0.0)
    B.setup_scene(ortho=2.6, res=TILE * SS, freestyle_thick=1.2)  # overridden to persp
    set_camera()
    r = TILE * SS
    render(os.path.join(OUT, f"{name}.png"), r, r)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    bake_floor()
    bake_cap()
    bake_face("face_n", "x", (0, -1))   # far wall: room to south
    bake_face("face_w", "y", (1, 0))    # left wall: room to east (its inner face)
    bake_face("face_e", "y", (-1, 0))   # right wall: room to west (its inner face)
    print("baked tiles →", os.path.abspath(OUT))
