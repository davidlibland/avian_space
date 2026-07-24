#!/usr/bin/env python3
"""Bake the cabinet interior tileset — floor + 4 edges + 4 corners.

ONE unified model so edges and corners match exactly: a tile is defined by
the SET of cell-edges that carry a wall slab. Every slab is a single-cell
piece (1.0 along the run, WALL_T across), so a corner is just the UNION of
two edge slabs inside the SAME cell — its ground footprint stays within one
grid square (only the cabinet shear spills the leaning height over the
neighbours, exactly like the straight walls).

  edges:   N  S  E  W                     one slab
  corners: CNW={N,W} CNE={N,E}            two slabs, L inside one cell
           CSW={S,W} CSE={S,E}

Everything is rendered under ONE fixed cabinet shear (height skewed
up-and-right) with a straight-down ortho camera, on transparent film, so the
pieces tile and share the game's toon+ink look.

Run:  scripts/.blender_venv/bin/python scripts/ship3d/interior_cabinet_tiles.py
Out:  scripts/ship3d/out/tiles/_it_*.png   (256 px/unit; anchor = cell centre)
"""
import os

import bpy
from mathutils import Matrix

import blender_gen as B

OUT = os.path.join(os.path.dirname(__file__), "out", "tiles")

PXU = 256           # pixels per world unit
FRAME = 4.0         # ortho span (units) → 1024 px frame, room for the lean
RES = int(FRAME * PXU)

# cabinet shear: screen offset per unit of height (matches the compose/engine)
SHX, SHY = 0.26, 0.42
WALL_H = 1.1
WALL_T = 0.48       # wall thickness (fraction of a cell) — chunky, reads as solid

FLOOR = (0.32, 0.36, 0.42)
FLOOR_HI = (0.39, 0.43, 0.50)
FLOOR_SEAM = (0.24, 0.27, 0.32)   # recessed panel seam (darker, not a glow)
WALL = (0.52, 0.55, 0.61)
WALL_TOP = (0.66, 0.69, 0.75)
TRIM = (0.25, 0.72, 0.86)


def _shear():
    S = Matrix(((1, 0, SHX, 0), (0, 1, SHY, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            o.matrix_world = S @ o.matrix_world


def _topdown():
    cam = bpy.context.scene.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = FRAME
    cam.location = (0, 0, 20)
    cam.rotation_euler = (0, 0, 0)


# slab geometry per edge: (centre, size). 1.0 along the run, WALL_T across,
# tucked against that edge of the cell.
def _edge_slab(edge):
    h = WALL_H
    if edge == "N":
        return (0, 0.5 - WALL_T / 2, h / 2), (1.0, WALL_T, h)
    if edge == "S":
        return (0, -0.5 + WALL_T / 2, h / 2), (1.0, WALL_T, h)
    if edge == "E":
        return (0.5 - WALL_T / 2, 0, h / 2), (WALL_T, 1.0, h)
    if edge == "W":
        return (-0.5 + WALL_T / 2, 0, h / 2), (WALL_T, 1.0, h)
    raise ValueError(edge)


# room-facing base-trim strip for an edge (the inner face that fronts the floor)
def _edge_trim(edge):
    z, t = 0.06, 0.05
    if edge == "N":
        return (0, 0.5 - WALL_T, z), (0.96, t, 0.10)
    if edge == "S":
        return (0, -0.5 + WALL_T, z), (0.96, t, 0.10)
    if edge == "E":
        return (0.5 - WALL_T, 0, z), (t, 0.96, 0.10)
    if edge == "W":
        return (-0.5 + WALL_T, 0, z), (t, 0.96, 0.10)
    raise ValueError(edge)


def bake_tile(name, edges):
    B.reset()
    mw = B.toon_material("wall", WALL)
    mwt = B.toon_material("walltop", WALL_TOP)
    mt = B.glow_material("trim", TRIM, strength=2.6)
    if edges == ("V",):
        # solid interior wall cell: a full-cell block seen only as its cap
        # (its faces are covered by neighbouring walls). Fills deep/thick walls.
        B.add_box("w_V", (0, 0, WALL_H / 2), (1.0, 1.0, WALL_H), mw, bevel=0.03)
        B.add_box("cap_V", (0, 0, WALL_H + 0.03), (1.0, 1.0, 0.08), mwt)
    else:
        for e in edges:
            (cx, cy, cz), (sx, sy, sz) = _edge_slab(e)
            B.add_box(f"w_{e}", (cx, cy, cz), (sx, sy, sz), mw, bevel=0.03)
            B.add_box(f"cap_{e}", (cx, cy, WALL_H + 0.03), (sx, sy, 0.08), mwt)
            (tx, ty, tz), (tsx, tsy, tsz) = _edge_trim(e)
            B.add_box(f"tr_{e}", (tx, ty, tz), (tsx, tsy, tsz), mt)
    _shear()
    B.setup_scene(ortho=FRAME, res=RES, freestyle_thick=1.4)
    _topdown()
    B.render_to(os.path.join(OUT, f"_it_{name}.png"))


def bake_floor():
    B.reset()
    mf = B.toon_material("floor", FLOOR)
    mfh = B.toon_material("floorhi", FLOOR_HI)
    ms = B.toon_material("seam", FLOOR_SEAM)
    B.add_box("floor", (0, 0, -0.05), (1.0, 1.0, 0.1), mf)
    B.add_box("panel", (0, 0, 0.005), (0.9, 0.9, 0.02), mfh, bevel=0.06)
    # subtle recessed panel seams (a calm dark inset, not a glowing grid)
    B.add_box("sx", (0, 0, 0.008), (0.03, 0.98, 0.006), ms)
    B.add_box("sy", (0, 0, 0.008), (0.98, 0.03, 0.006), ms)
    B.setup_scene(ortho=1.0, res=PXU, freestyle_thick=1.2)
    _topdown()
    bpy.context.scene.camera.data.ortho_scale = 1.0
    B.render_to(os.path.join(OUT, "_it_floor.png"))


TILES = {
    "N": ("N",), "S": ("S",), "E": ("E",), "W": ("W",),
    "CNW": ("N", "W"), "CNE": ("N", "E"), "CSW": ("S", "W"), "CSE": ("S", "E"),
    "V": ("V",),
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    bake_floor()
    for name, edges in TILES.items():
        bake_tile(name, edges)
        print("baked", name)
    print("tiles →", os.path.abspath(OUT))
