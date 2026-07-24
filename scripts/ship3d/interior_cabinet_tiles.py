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

# Per-venue palettes + a light material detail on the room-facing wall face,
# so each maze reads as its own place: station panels, mine rock, warehouse
# corrugation, substation conduit. Geometry is identical across venues (so
# the autotiling still lines up); only colours + the face greeble change.
VENUES = {
    # station shops — clean cool panels, cyan baseboard glow
    "interior": dict(floor=(0.32, 0.36, 0.42), floor_hi=(0.39, 0.43, 0.50),
                     seam=(0.24, 0.27, 0.32), wall=(0.52, 0.55, 0.61),
                     wall_top=(0.66, 0.69, 0.75), trim=(0.25, 0.72, 0.86), detail=None),
    # mine — dirt floor, rough rock walls, warm lantern-amber trim
    "mine": dict(floor=(0.30, 0.25, 0.20), floor_hi=(0.35, 0.29, 0.23),
                 seam=(0.19, 0.15, 0.11), wall=(0.45, 0.40, 0.35),
                 wall_top=(0.55, 0.49, 0.43), trim=(0.96, 0.62, 0.24), detail="rock"),
    # warehouse — concrete deck, corrugated steel walls, safety-yellow trim
    "warehouse": dict(floor=(0.44, 0.45, 0.47), floor_hi=(0.50, 0.51, 0.53),
                      seam=(0.30, 0.31, 0.33), wall=(0.50, 0.54, 0.60),
                      wall_top=(0.62, 0.66, 0.72), trim=(0.95, 0.80, 0.24), detail="ribs"),
    # substation — grating floor, dark conduit walls, electric-teal trim
    "substation": dict(floor=(0.24, 0.28, 0.30), floor_hi=(0.29, 0.34, 0.36),
                       seam=(0.15, 0.19, 0.21), wall=(0.38, 0.43, 0.48),
                       wall_top=(0.48, 0.54, 0.60), trim=(0.30, 0.92, 0.72), detail="conduit"),
}


# Room-facing face of an edge: (run_axis, (fixed_axis, coord), normal_sign).
def _face(edge):
    inner = 0.5 - WALL_T
    if edge == "N":
        return "x", ("y", inner), -1.0
    if edge == "S":
        return "x", ("y", -inner), 1.0
    if edge == "E":
        return "y", ("x", inner), -1.0
    if edge == "W":
        return "y", ("x", -inner), 1.0
    raise ValueError(edge)


def _face_detail(edge, kind, mat, glow):
    """Add venue greebles on the room-facing wall face of one edge slab."""
    if kind is None:
        return
    run, (fax, fc), ns = _face(edge)
    h = WALL_H

    def place(name, along, out, z, size_run, size_out, size_z, m):
        # `along` runs down the face; `out` proud of the face by that much.
        cf = fc + ns * out
        if run == "x":
            loc, size = (along, cf, z), (size_run, size_out, size_z)
        else:
            loc, size = (cf, along, z), (size_out, size_run, size_z)
        B.add_box(name, loc, size, m, bevel=0.01)

    if kind == "ribs":  # warehouse corrugation — vertical fins
        for i, p in enumerate((-0.4, -0.2, 0.0, 0.2, 0.4)):
            place(f"rib{edge}{i}", p, 0.03, h * 0.5, 0.05, 0.06, h * 0.9, mat)
    elif kind == "conduit":  # substation — a pipe run + a glowing node
        place(f"pipe{edge}", 0.0, 0.05, h * 0.58, 0.92, 0.08, 0.1, mat)
        place(f"node{edge}", 0.26, 0.07, h * 0.58, 0.1, 0.06, 0.14, glow)
    elif kind == "rock":  # mine — a couple of irregular boulders
        for i, (p, z, s) in enumerate(((-0.28, h * 0.32, 0.26),
                                       (0.14, h * 0.55, 0.22),
                                       (0.36, h * 0.28, 0.18))):
            place(f"rock{edge}{i}", p, 0.02, z, s, 0.1, s, mat)


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


def bake_tile(venue, pal, name, edges):
    B.reset()
    mw = B.toon_material("wall", pal["wall"])
    mwt = B.toon_material("walltop", pal["wall_top"])
    mt = B.glow_material("trim", pal["trim"], strength=2.6)
    md = B.toon_material("detail", pal["wall"])          # greebles match the wall
    mg = B.glow_material("detailglow", pal["trim"], strength=3.2)
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
            _face_detail(e, pal["detail"], md, mg)
    _shear()
    B.setup_scene(ortho=FRAME, res=RES, freestyle_thick=1.4)
    _topdown()
    B.render_to(os.path.join(OUT, f"_it_{venue}_{name}.png"))


def bake_floor(venue, pal):
    B.reset()
    mf = B.toon_material("floor", pal["floor"])
    mfh = B.toon_material("floorhi", pal["floor_hi"])
    ms = B.toon_material("seam", pal["seam"])
    B.add_box("floor", (0, 0, -0.05), (1.0, 1.0, 0.1), mf)
    B.add_box("panel", (0, 0, 0.005), (0.9, 0.9, 0.02), mfh, bevel=0.06)
    # subtle recessed panel seams (a calm dark inset, not a glowing grid)
    B.add_box("sx", (0, 0, 0.008), (0.03, 0.98, 0.006), ms)
    B.add_box("sy", (0, 0, 0.008), (0.98, 0.03, 0.006), ms)
    B.setup_scene(ortho=1.0, res=PXU, freestyle_thick=1.2)
    _topdown()
    bpy.context.scene.camera.data.ortho_scale = 1.0
    B.render_to(os.path.join(OUT, f"_it_{venue}_floor.png"))


TILES = {
    "N": ("N",), "S": ("S",), "E": ("E",), "W": ("W",),
    "CNW": ("N", "W"), "CNE": ("N", "E"), "CSW": ("S", "W"), "CSE": ("S", "E"),
    "V": ("V",),
}

if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or list(VENUES)
    for venue in only:
        pal = VENUES[venue]
        bake_floor(venue, pal)
        for name, edges in TILES.items():
            bake_tile(venue, pal, name, edges)
        print("baked venue", venue)
    print("tiles →", os.path.abspath(OUT))
