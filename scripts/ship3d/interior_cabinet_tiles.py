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
# `floor_style`: "panel" = a tiled deck panel (one variant is plenty);
# "dirt" = organic soil with scattered rock, baked in several interchangeable
# variants so a dug-out floor never reads as a grid. `floors` = variant count.
VENUES = {
    # station shops — clean cool panels, cyan baseboard glow
    "interior": dict(floor=(0.32, 0.36, 0.42), floor_hi=(0.39, 0.43, 0.50),
                     seam=(0.24, 0.27, 0.32), wall=(0.52, 0.55, 0.61),
                     wall_top=(0.66, 0.69, 0.75), trim=(0.25, 0.72, 0.86),
                     detail=None, floor_style="panel", floors=1, walls=1),
    # mine — dirt floor (4 rock-scatter variants), rough rock walls, amber trim
    "mine": dict(floor=(0.34, 0.27, 0.20), floor_hi=(0.39, 0.31, 0.23),
                 seam=(0.19, 0.15, 0.11), wall=(0.45, 0.40, 0.35),
                 wall_top=(0.55, 0.49, 0.43), trim=(0.96, 0.62, 0.24),
                 detail="rock", floor_style="dirt", floors=4, walls=3),
    # warehouse — concrete deck, corrugated steel walls, safety-yellow trim
    "warehouse": dict(floor=(0.44, 0.45, 0.47), floor_hi=(0.50, 0.51, 0.53),
                      seam=(0.30, 0.31, 0.33), wall=(0.50, 0.54, 0.60),
                      wall_top=(0.62, 0.66, 0.72), trim=(0.95, 0.80, 0.24),
                      detail="ribs", floor_style="panel", floors=1, walls=1),
    # substation — grating floor, dark conduit walls, electric-teal trim
    "substation": dict(floor=(0.24, 0.28, 0.30), floor_hi=(0.29, 0.34, 0.36),
                       seam=(0.15, 0.19, 0.21), wall=(0.38, 0.43, 0.48),
                       wall_top=(0.48, 0.54, 0.60), trim=(0.30, 0.92, 0.72),
                       detail="conduit", floor_style="panel", floors=1, walls=1),
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


def _face_detail(edge, kind, variant, mat, glow):
    """Add venue greebles on the room-facing wall face of one edge slab.
    `variant` seeds the organic (rock) placement so several wall tiles shuffle
    across a run without visibly repeating."""
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
    elif kind == "rock":  # mine — rough dug rock, clustered lumps + cap bumps
        import random
        rng = random.Random(700 + variant * 37 + (ord(edge[-1]) * 13))
        for i in range(7):  # lumps over the room-facing face
            along = rng.uniform(-0.43, 0.43)
            z = rng.uniform(0.12, 0.96) * h
            s = rng.uniform(0.1, 0.26)
            place(f"rk{edge}{i}", along, 0.02 + rng.uniform(0, 0.04), z,
                  s, 0.12 + rng.uniform(0, 0.06), s * rng.uniform(0.7, 1.2), mat)
        # a few bumps riding the top cap so the ridge isn't a clean edge
        (cx, cy, _), _ = _edge_slab(edge)
        for i in range(4):
            along = rng.uniform(-0.4, 0.4)
            s = rng.uniform(0.12, 0.24)
            loc = (along, cy, h * 0.96) if run == "x" else (cx, along, h * 0.96)
            B.add_box(f"rkc{edge}{i}", loc, (s, s, 0.14), mat, bevel=0.4)


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


def _cap_rock(variant, mat):
    """Bumpy top for a deep (V) cap — variant-seeded so caps vary too."""
    import random
    rng = random.Random(311 + variant * 53)
    for i in range(9):
        x, y = rng.uniform(-0.42, 0.42), rng.uniform(-0.42, 0.42)
        s = rng.uniform(0.12, 0.26)
        B.add_box(f"vrk{i}", (x, y, WALL_H * 0.95 + rng.uniform(0, 0.08)),
                  (s, s * rng.uniform(0.7, 1.3), 0.16), mat, bevel=0.4)


def bake_tile(venue, pal, name, edges, variant=0):
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
        if pal["detail"] == "rock":
            _cap_rock(variant, mw)
    else:
        for e in edges:
            (cx, cy, cz), (sx, sy, sz) = _edge_slab(e)
            B.add_box(f"w_{e}", (cx, cy, cz), (sx, sy, sz), mw, bevel=0.03)
            B.add_box(f"cap_{e}", (cx, cy, WALL_H + 0.03), (sx, sy, 0.08), mwt)
            (tx, ty, tz), (tsx, tsy, tsz) = _edge_trim(e)
            B.add_box(f"tr_{e}", (tx, ty, tz), (tsx, tsy, tsz), mt)
            _face_detail(e, pal["detail"], variant, md, mg)
    _shear()
    B.setup_scene(ortho=FRAME, res=RES, freestyle_thick=1.4)
    _topdown()
    B.render_to(os.path.join(OUT, f"_it_{venue}_{name}_v{variant}.png"))


def bake_floor(venue, pal, variant=0):
    import random
    B.reset()
    mf = B.toon_material("floor", pal["floor"])
    if pal.get("floor_style") == "dirt":
        # Organic soil: a flat base with scattered rounded rock/clods whose
        # placement varies per `variant`, so several tiles shuffle across the
        # floor without a visible grid. No panel seams. Pebbles keep the ink
        # outline (they read as stones); flat clods are rounded hard so their
        # edges stay soft, not boxy.
        rng = random.Random(9001 + variant * 17)
        # Oversized base so its Freestyle outline falls OFF-frame — no grid
        # line around each dirt tile, so neighbours blend seamlessly.
        B.add_box("floor", (0, 0, -0.05), (1.5, 1.5, 0.1), mf)
        light = tuple(min(1.0, c * 1.16) for c in pal["floor"])
        dark = tuple(c * 0.82 for c in pal["floor"])
        ml = B.toon_material(f"peblt{variant}", light)
        md = B.toon_material(f"pebdk{variant}", dark)
        # a few broad low clods (packed-earth patches), rounded so they blend
        for i in range(5):
            x, y = rng.uniform(-0.34, 0.34), rng.uniform(-0.34, 0.34)
            s = rng.uniform(0.18, 0.30)
            B.add_box(f"clod{i}", (x, y, 0.004), (s, s * rng.uniform(0.7, 1.3), 0.02),
                      md if rng.random() < 0.5 else ml, bevel=0.5)
        # scattered stones with a little height (kept off the tile edge so they
        # don't get cut mid-stone where tiles meet)
        for i in range(10):
            x, y = rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)
            r = rng.uniform(0.025, 0.075)
            B.add_box(f"peb{i}", (x, y, r * 0.4), (r * 2, r * 2 * rng.uniform(0.7, 1.2), r * 0.9),
                      ml if rng.random() < 0.5 else md, bevel=0.45)
    else:
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
    B.render_to(os.path.join(OUT, f"_it_{venue}_floor{variant}.png"))


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
        for v in range(pal.get("floors", 1)):
            bake_floor(venue, pal, v)
        for wv in range(pal.get("walls", 1)):
            for name, edges in TILES.items():
                bake_tile(venue, pal, name, edges, wv)
        print("baked venue", venue)
    print("tiles →", os.path.abspath(OUT))
