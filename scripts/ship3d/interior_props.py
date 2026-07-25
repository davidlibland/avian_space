"""
interior_props.py — furniture & fixtures for building interiors.

Every walk-in shop and maze venue dresses its floor with these props
(src/surface/interiors.rs `prop_meta` is the footprint contract). Baked in
the buildings3d house style: forward-facing oblique (ELEV 50°), toon
shading, Freestyle ink, 1 blender unit = 1 tile, rendered at ~34 px/tile
then tightly cropped so the game's BottomCenter anchor lands on the prop's
front-bottom edge.

Run:  scripts/.blender_venv/bin/python interior_props.py [preview]
      preview -> montage in out/ only; otherwise writes
      assets/sprites/worlds/interior_props/<name>.png
"""

import math
import os
import sys

import bpy
from PIL import Image, ImageDraw, ImageFilter

from mathutils import Matrix

from blender_gen import (add_box, add_cylinder, add_sphere, glow_material,
                         render_to, reset, setup_scene, toon_material)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
DEST = os.path.join(HERE, "..", "..", "assets", "sprites", "worlds",
                    "interior_props")
ELEV = 50.0
RES = 512
PX_PER_TILE = 34  # matches buildings3d — game downsamples to 32

# Props stand in cabinet-projected rooms, so they must lean the SAME way as the
# walls (scripts/ship3d/interior_cabinet_tiles.py) or they read as decals lying
# on the floor instead of objects standing on it.
SHX, SHY = 0.26, 0.42

# Cel bands. The stock ramp (0.5 / 0.82 / 1.0) crushes everything a top-down
# camera sees into the top band, so every surface came out the same value —
# the "flat" look. Four wider-spread steps keep the hard-edged toon feel but
# give tops, sides and shadowed faces distinct values.
BANDS = ((0.0, 0.26), (0.30, 0.46), (0.56, 0.72), (0.82, 1.0))


def C(*rgb):
    return tuple(v / 255.0 for v in rgb)


def mats():
    return {
        "wood": toon_material("wood", C(122, 84, 52), bands=BANDS),
        "wood_dark": toon_material("wood_dark", C(84, 56, 36), bands=BANDS),
        "steel": toon_material("steel", C(150, 156, 166), bands=BANDS),
        "iron": toon_material("iron", C(72, 74, 80), bands=BANDS),
        "brass": toon_material("brass", C(180, 134, 62), spec=0.5, bands=BANDS),
        "cloth_red": toon_material("cloth_red", C(168, 62, 48), bands=BANDS),
        "cloth_tan": toon_material("cloth_tan", C(196, 168, 120), bands=BANDS),
        "crate": toon_material("crate", C(146, 112, 66), bands=BANDS),
        "cont_red": toon_material("cont_red", C(150, 62, 44), bands=BANDS),
        "cont_blue": toon_material("cont_blue", C(52, 84, 130), bands=BANDS),
        "glow_warm": glow_material("glow_warm", C(255, 196, 96), 2.4),
        "glow_cyan": glow_material("glow_cyan", C(80, 220, 255), 2.4),
        "pipe": toon_material("pipe", C(96, 110, 116), bands=BANDS),
        "bottle_g": toon_material("bottle_g", C(70, 140, 90), spec=0.6, bands=BANDS),
        "dark": toon_material("dark", C(30, 30, 34), bands=BANDS),
    }


def cam_oblique():
    """Straight-down ortho. Depth comes from the cabinet SHEAR applied to the
    geometry, exactly as the wall tiles do it — never from tilting the camera,
    which would disagree with the walls."""
    cam = bpy.context.scene.camera
    cam.data.type = "ORTHO"
    cam.constraints.clear()
    cam.location = (0.0, 0.0, 20.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)


def apply_shear():
    S = Matrix(((1, 0, SHX, 0), (0, 1, SHY, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            o.matrix_world = S @ o.matrix_world


def light_for_contrast():
    """Drop the key toward the horizon and pull the fill down. A high key plus
    a strong fill lights tops and sides almost equally, which is what collapsed
    them into one band; a lower key separates them."""
    sc = bpy.context.scene
    for o in sc.objects:
        if o.type != "LIGHT":
            continue
        if o.name.startswith("key"):
            o.rotation_euler = (math.radians(32), math.radians(-26), math.radians(18))
            o.data.energy = 5.2
        elif o.name.startswith("fill"):
            o.data.energy = 0.7
        elif o.name.startswith("rim"):
            o.data.energy = 1.4


# ── builders: geometry in tile units, front edge at y = -depth/2 ───────────
def bar_counter(M):
    add_box("top", (0, 0, 0.75), (3.0, 0.9, 0.12), M["wood_dark"], bevel=0.03)
    add_box("body", (0, 0, 0.38), (2.9, 0.8, 0.72), M["wood"], bevel=0.03)
    for i, x in enumerate((-1.0, -0.3, 0.5, 1.1)):
        add_cylinder(f"btl{i}", (x, 0.1, 0.95), 0.055, 0.26,
                     M["bottle_g" if i % 2 else "brass"], axis="z")


def table_round(M):
    add_cylinder("top", (0, 0, 0.62), 0.46, 0.07, M["wood"], axis="z")
    add_cylinder("leg", (0, 0, 0.3), 0.07, 0.6, M["wood_dark"], axis="z")
    add_cylinder("base", (0, 0, 0.03), 0.2, 0.06, M["wood_dark"], axis="z")


def stool(M):
    add_cylinder("seat", (0, 0, 0.42), 0.2, 0.07, M["cloth_red"], axis="z")
    add_cylinder("leg", (0, 0, 0.2), 0.05, 0.4, M["iron"], axis="z")


def shelf_rack(M):
    """Open shelving: uprights + several visible shelf levels with stock on
    them, so it reads as storage rather than a grey slab."""
    for i, x in enumerate((-0.92, 0.92)):
        add_box(f"up{i}", (x, 0, 0.85), (0.1, 0.5, 1.7), M["iron"], bevel=0.02)
    for i, z in enumerate((0.36, 0.86, 1.36)):
        add_box(f"shelf{i}", (0, 0, z), (1.92, 0.52, 0.07), M["steel"], bevel=0.02)
        for j, x in enumerate((-0.6, -0.1, 0.5)):
            box = M["crate"] if (i + j) % 2 else M["cont_blue"]
            add_box(f"bx{i}{j}", (x + 0.1 * j, 0.0, z + 0.2),
                    (0.34, 0.34, 0.3), box, bevel=0.03)


def market_stall(M):
    add_box("counter", (0, -0.2, 0.4), (1.9, 0.6, 0.8), M["wood"], bevel=0.03)
    for sx in (-0.8, 0.8):
        add_cylinder(f"post{sx}", (sx, 0.25, 0.8), 0.05, 1.6, M["wood_dark"],
                     axis="z")
    awn = add_box("awning", (0, 0.1, 1.62), (2.1, 1.0, 0.06), M["cloth_red"],
                  bevel=0.02)
    awn.rotation_euler = (math.radians(12), 0, 0)
    for i in range(3):
        add_sphere(f"produce{i}", (-0.55 + i * 0.55, -0.25, 0.87),
                   (0.16, 0.16, 0.13), M["cloth_tan" if i % 2 else "bottle_g"])


def engine_bench(M):
    """A workbench alone is an anonymous slab. What says REPAIR SHOP is an
    engine cradled under a hoist, with tools on the bench and a lit readout."""
    add_box("bench", (0, 0, 0.34), (1.9, 0.8, 0.68), M["iron"], bevel=0.03)
    add_box("top", (0, 0, 0.71), (1.96, 0.86, 0.08), M["steel"], bevel=0.02)
    # engine block being worked on, with nozzle + cyan diagnostic glow
    add_cylinder("block", (-0.34, 0, 1.02), 0.3, 0.86, M["steel"], axis="x")
    for i, x in enumerate((-0.62, -0.34, -0.06)):
        add_cylinder(f"ring{i}", (x, 0, 1.02), 0.33, 0.06, M["iron"], axis="x")
    add_cylinder("nozzle", (0.28, 0, 1.02), 0.2, 0.3, M["brass"], axis="x", r2=0.3)
    add_sphere("glow", (-0.86, 0, 1.02), (0.1, 0.1, 0.1), M["glow_cyan"])
    # hoist: upright + arm over the block, with a hanging chain
    add_box("post", (0.86, 0.26, 0.9), (0.14, 0.14, 1.8), M["iron"], bevel=0.02)
    add_box("arm", (0.1, 0.26, 1.74), (1.6, 0.12, 0.12), M["iron"], bevel=0.02)
    add_cylinder("chain", (-0.34, 0.26, 1.44), 0.03, 0.5, M["dark"], axis="z")
    add_box("hook", (-0.34, 0.26, 1.17), (0.08, 0.08, 0.12), M["steel"], bevel=0.02)
    # a couple of tools on the bench so the top isn't bare
    add_box("wrench", (0.62, -0.16, 0.78), (0.44, 0.09, 0.05), M["steel"], bevel=0.02)
    add_box("tray", (0.5, 0.2, 0.78), (0.34, 0.26, 0.06), M["iron"], bevel=0.02)


def tool_rack(M):
    """A pegboard reads as tools only if the tools have TOOL shapes: a wrench
    with a head, a hammer with a claw, a driver, hung in a row over a bin."""
    add_box("board", (0, 0.2, 0.86), (1.0, 0.08, 1.3), M["wood_dark"])
    add_box("edge", (0, 0.2, 1.53), (1.04, 0.1, 0.06), M["iron"], bevel=0.02)
    # wrench: shaft + open head
    add_box("wr_shaft", (-0.32, 0.12, 1.02), (0.06, 0.05, 0.52), M["steel"], bevel=0.02)
    add_box("wr_head", (-0.32, 0.12, 1.32), (0.17, 0.06, 0.16), M["steel"], bevel=0.04)
    # hammer: handle + head across the top
    add_box("hm_handle", (0.0, 0.12, 0.98), (0.05, 0.05, 0.46), M["wood"], bevel=0.02)
    add_box("hm_head", (0.02, 0.12, 1.28), (0.24, 0.07, 0.11), M["iron"], bevel=0.03)
    # driver: thin shaft + fat grip
    add_box("dr_shaft", (0.32, 0.12, 1.1), (0.04, 0.04, 0.34), M["steel"], bevel=0.01)
    add_box("dr_grip", (0.32, 0.12, 0.84), (0.09, 0.08, 0.24), M["cont_red"], bevel=0.04)
    # parts bin along the bottom
    add_box("bin", (0, 0.04, 0.16), (0.92, 0.3, 0.32), M["iron"], bevel=0.03)
    for i, x in enumerate((-0.28, 0.0, 0.28)):
        add_box(f"nut{i}", (x, -0.04, 0.34), (0.14, 0.14, 0.06), M["brass"], bevel=0.02)


def fuel_pump(M):
    """Read-at-32px checklist: tall body, lit display, and above all a HOSE
    looping out to a nozzle — the one silhouette nothing else in the game has."""
    add_box("base", (0, 0, 0.09), (0.72, 0.5, 0.18), M["iron"], bevel=0.04)
    add_box("body", (0, 0, 0.78), (0.62, 0.42, 1.2), M["cont_red"], bevel=0.07)
    add_box("face", (0, -0.23, 1.02), (0.48, 0.06, 0.46), M["glow_warm"])
    # Under cabinet projection a tall prop shows a lot of TOP face, so keep the
    # crown small — a wide cap plate just becomes a grey lid hiding the body.
    add_box("cap", (0, -0.06, 1.42), (0.4, 0.26, 0.08), M["iron"], bevel=0.03)
    # hose: an arc of short segments from the flank out and down to a nozzle
    for i, (dx, dz) in enumerate(((0.36, 1.18), (0.52, 1.06), (0.60, 0.86),
                                  (0.62, 0.62), (0.60, 0.40))):
        add_cylinder(f"hose{i}", (dx, 0.0, dz), 0.05, 0.16, M["dark"], axis="z")
    add_box("nozzle", (0.60, -0.02, 0.26), (0.13, 0.13, 0.3), M["brass"], bevel=0.03)
    add_box("holster", (-0.36, -0.06, 0.62), (0.1, 0.16, 0.26), M["iron"], bevel=0.02)


def war_desk(M):
    add_box("desk", (0, 0, 0.4), (1.9, 0.9, 0.8), M["wood_dark"], bevel=0.03)
    add_box("map", (0, 0, 0.83), (1.6, 0.7, 0.05), M["glow_cyan"])
    add_box("papers", (0.6, -0.2, 0.85), (0.3, 0.24, 0.04), M["cloth_tan"])


def flag_stand(M):
    add_cylinder("pole", (0, 0, 0.9), 0.04, 1.8, M["brass"], axis="z")
    add_cylinder("base", (0, 0, 0.04), 0.18, 0.08, M["iron"], axis="z")
    flag = add_box("flag", (0.3, 0, 1.5), (0.6, 0.03, 0.4), M["cloth_red"])
    flag.rotation_euler = (0, math.radians(-6), 0)


def crate_stack(M):
    add_box("a", (-0.1, 0.05, 0.25), (0.55, 0.5, 0.5), M["crate"], bevel=0.04)
    add_box("b", (0.25, -0.15, 0.22), (0.44, 0.42, 0.44), M["crate"],
            bevel=0.04)
    add_box("c", (0.02, -0.03, 0.68), (0.42, 0.4, 0.4), M["wood"], bevel=0.04)


def container(M, color):
    add_box("body", (0, 0, 0.6), (3.9, 1.9, 1.2), M[color], bevel=0.04)
    for x in (-1.5, -0.5, 0.5, 1.5):
        add_box(f"rib{x}", (x, 0, 0.6), (0.1, 1.96, 1.24), M["iron"])
    add_box("door", (1.93, 0, 0.6), (0.06, 1.7, 1.05), M["iron"])


def ore_cart(M):
    add_box("tub", (0, 0, 0.45), (0.8, 0.55, 0.5), M["iron"], bevel=0.05)
    for sx in (-0.25, 0.25):
        add_cylinder(f"wheel{sx}", (sx, -0.3, 0.14), 0.13, 0.08, M["dark"],
                     axis="y")
    for i in range(4):
        add_sphere(f"ore{i}", (-0.2 + (i % 2) * 0.35, -0.1 + (i // 2) * 0.2,
                               0.72), (0.14, 0.13, 0.11), M["dark"])


def timber_brace(M):
    for sx in (-0.42, 0.42):
        add_box(f"post{sx}", (sx, 0, 0.7), (0.16, 0.16, 1.4), M["wood_dark"])
    add_box("beam", (0, 0, 1.42), (1.1, 0.18, 0.16), M["wood"])


def lantern(M):
    add_cylinder("post", (0, 0, 0.55), 0.045, 1.1, M["iron"], axis="z")
    add_sphere("light", (0, 0, 1.18), (0.14, 0.14, 0.16), M["glow_warm"])
    add_box("cap", (0, 0, 1.38), (0.22, 0.22, 0.06), M["iron"])


def pump_unit(M):
    add_box("base", (0, 0, 0.3), (1.9, 1.9, 0.6), M["iron"], bevel=0.04)
    add_cylinder("tank", (-0.4, 0, 1.0), 0.45, 1.0, M["pipe"], axis="z")
    add_cylinder("pipe1", (0.55, 0, 0.9), 0.12, 1.0, M["pipe"], axis="z")
    add_cylinder("pipe2", (0.2, 0, 1.35), 0.1, 1.2, M["pipe"], axis="x")
    add_sphere("gauge", (0.55, -0.15, 1.3), (0.09, 0.09, 0.09),
               M["glow_warm"])


def pipe_valve(M):
    add_cylinder("riser", (0, 0, 0.45), 0.12, 0.9, M["pipe"], axis="z")
    add_cylinder("wheel", (0, 0, 0.95), 0.24, 0.07, M["cont_red"], axis="z")
    add_cylinder("stem", (0, 0, 0.86), 0.04, 0.2, M["steel"], axis="z")


def stairs(M, down):
    # A hatch + steps: DOWN shows a dark open shaft with the top steps
    # catching the lamp; UP is a stepped rise with a handrail.
    add_box("rim", (0, 0, 0.09), (0.98, 0.98, 0.18), M["steel"], bevel=0.02)
    if down:
        add_box("pit", (0, 0, 0.19), (0.78, 0.78, 0.02), M["dark"])
        for i in range(3):
            add_box(f"step{i}", (0, -0.25 + i * 0.22, 0.2 - i * 0.001),
                    (0.7, 0.18, 0.02), M["iron"])
    else:
        for i in range(4):
            add_box(f"step{i}", (0, -0.3 + i * 0.2, 0.2 + i * 0.14),
                    (0.7, 0.2, 0.1), M["steel"])
        add_cylinder("rail", (0.4, 0, 0.75), 0.035, 0.9, M["brass"], axis="y")
    add_sphere("lamp", (0.42, 0.42, 0.42), (0.08, 0.08, 0.09),
               M["glow_warm" if down else "glow_cyan"])


def exit_door_frame(M):
    # The static part of the way out: jambs, lintel, EXIT bar. The door
    # PANELS are baked separately as animation frames (exit_door_panel*).
    for sx in (-0.7, 0.7):
        add_box(f"jamb{sx}", (sx, 0, 1.05), (0.22, 0.28, 2.1), M["steel"])
    add_box("lintel", (0, 0, 2.2), (1.7, 0.3, 0.22), M["steel"])
    exit_green = glow_material("exit_green", C(30, 160, 60), 1.6)
    add_box("sign", (0, -0.08, 2.5), (1.2, 0.14, 0.34), exit_green)
    add_box("sign_frame", (0, 0.0, 2.5), (1.3, 0.1, 0.42), M["dark"])


def exit_door_panels(M, openness):
    # Roll-up panels, top pinned under the lintel; light spills through
    # the centre gap. openness 1.0 = fully rolled (nothing to draw).
    h = 1.9 * (1.0 - openness)
    if h < 0.03:
        return
    zc = 2.05 - h / 2
    add_box("panel_l", (-0.4, 0.1, zc), (0.48, 0.08, h), M["iron"])
    add_box("panel_r", (0.4, 0.1, zc), (0.48, 0.08, h), M["iron"])
    add_box("gap", (0, 0.14, zc), (0.32, 0.06, h), M["glow_warm"])


def exit_door(M):
    exit_door_frame(M)


def ladder_up(M):
    # A ladder climbing OUT — rails, rungs, a hatch rim of daylight above.
    for sx in (-0.28, 0.28):
        add_cylinder(f"rail{sx}", (sx, 0.1, 1.05), 0.05, 2.1, M["brass"],
                     axis="z")
    for i in range(5):
        add_cylinder(f"rung{i}", (0, 0.1, 0.35 + i * 0.42), 0.04, 0.62,
                     M["steel"], axis="x")
    day = glow_material("daylight", C(255, 244, 200), 2.8)
    add_box("hatch_glow", (0, 0.1, 2.2), (0.9, 0.5, 0.1), day)
    add_box("hatch_rim", (0, 0.1, 2.3), (1.0, 0.6, 0.08), M["iron"])


def pebbles(M, seed):
    import random
    r = random.Random(seed)
    for i in range(r.randint(4, 7)):
        x, y = r.uniform(-0.4, 0.4), r.uniform(-0.35, 0.35)
        sz = r.uniform(0.04, 0.11)
        add_sphere(f"peb{i}", (x, y, sz * 0.6), (sz, sz * 0.9, sz * 0.7),
                   M["iron" if r.random() < 0.4 else "pipe"])


def ore_chunk(M):
    add_sphere("rock", (0, 0, 0.14), (0.2, 0.17, 0.14), M["iron"])
    glint = glow_material("ore_glint", C(255, 190, 80), 1.8)
    for x, y in ((-0.06, -0.05), (0.08, 0.02), (0.0, -0.12)):
        add_sphere(f"vein{x}", (x, y, 0.22), (0.035, 0.035, 0.03), glint)


def ore_pile(M):
    for i, (x, y, sz) in enumerate(
        ((0, 0, 0.28), (-0.25, 0.1, 0.18), (0.24, -0.05, 0.16), (0.05, 0.22, 0.15))
    ):
        add_sphere(f"lump{i}", (x, y, sz * 0.7), (sz, sz * 0.9, sz * 0.75), M["pipe"])
    glint = glow_material("pile_glint", C(255, 190, 80), 1.6)
    for x, y in ((-0.1, -0.1), (0.15, 0.12)):
        add_sphere(f"g{x}", (x, y, 0.32), (0.04, 0.04, 0.035), glint)


def pickaxe(M):
    haft = add_cylinder("haft", (0, 0, 0.08), 0.035, 0.9, M["wood"], axis="y")
    haft.rotation_euler = (0, 0, 0.5)
    head = add_box("head", (0.18, -0.32, 0.1), (0.5, 0.07, 0.07), M["iron"])
    head.rotation_euler = (0, 0, 0.5 + 1.57)


def crystal(M):
    shard = glow_material("shard", C(120, 220, 255), 2.0)
    for i, (x, y, h, r) in enumerate(
        ((0, 0, 0.5, 0.09), (-0.14, 0.08, 0.3, 0.06), (0.13, -0.06, 0.36, 0.07))
    ):
        c = add_cylinder(f"cr{i}", (x, y, h / 2), r, h, shard, axis="z", r2=0.01)
        c.rotation_euler = (0.1 * i, 0.12 * (i - 1), 0)
    add_sphere("base", (0, 0, 0.05), (0.22, 0.2, 0.08), M["iron"])


def pallet(M):
    for i in range(4):
        add_box(f"slat{i}", (0, -0.33 + i * 0.22, 0.09), (0.9, 0.16, 0.04), M["wood"])
    for sx in (-0.35, 0, 0.35):
        add_box(f"bearer{sx}", (sx, 0, 0.04), (0.12, 0.85, 0.07), M["wood_dark"])


def barrel(M):
    add_cylinder("drum", (0, 0, 0.34), 0.24, 0.68, M["cont_blue"], axis="z", seg=14)
    for z in (0.14, 0.34, 0.54):
        add_cylinder(f"band{z}", (0, 0, z), 0.25, 0.04, M["iron"], axis="z", seg=14)


def box_spill(M):
    b = add_box("box", (-0.1, 0.05, 0.16), (0.4, 0.4, 0.32), M["crate"], bevel=0.03)
    b.rotation_euler = (0, 0.5, 0.3)
    for i, (x, y) in enumerate(((0.2, -0.15), (0.32, 0.05), (0.15, 0.18), (0.4, -0.05))):
        add_sphere(f"good{i}", (x, y, 0.05), (0.05, 0.05, 0.045),
                   M["cloth_tan" if i % 2 else "bottle_g"])


def cable_coil(M):
    for i in range(3):
        add_cylinder(f"loop{i}", (0, 0, 0.05 + i * 0.05), 0.2 - i * 0.02, 0.05,
                     M["dark"], axis="z", seg=16)
    add_cylinder("tail", (0.3, 0.1, 0.03), 0.025, 0.5, M["dark"], axis="y")


def pipe_segment(M):
    p = add_cylinder("pipe", (0, 0, 0.12), 0.11, 0.9, M["pipe"], axis="y", seg=12)
    p.rotation_euler = (0, 0, 0.4)
    add_cylinder("flange", (0.12, -0.4, 0.12), 0.15, 0.06, M["iron"], axis="y", seg=12)


def warning_cone(M):
    cone = add_cylinder("cone", (0, 0, 0.24), 0.16, 0.48, M["cont_red"], axis="z",
                        r2=0.03, seg=12)
    add_box("base", (0, 0, 0.02), (0.36, 0.36, 0.04), M["cont_red"])
    add_cylinder("stripe", (0, 0, 0.28), 0.115, 0.09, M["steel"], axis="z", seg=12)


def coolant_puddle(M):
    glow = glow_material("puddle", C(60, 210, 220), 1.4)
    add_sphere("pool", (0, 0, 0.012), (0.4, 0.3, 0.012), glow)
    add_sphere("pool2", (0.3, 0.18, 0.01), (0.14, 0.1, 0.01), glow)


def jail_bars(M):
    # A barred cell front for the garrison's holding nook.
    add_box("sill", (0, 0, 0.06), (0.98, 0.24, 0.12), M["iron"])
    add_box("lintel", (0, 0, 2.12), (0.98, 0.24, 0.12), M["iron"])
    for i in range(5):
        add_cylinder(f"bar{i}", (-0.4 + i * 0.2, 0, 1.09), 0.035, 2.0,
                     M["steel"], axis="z")


def jail_gate(M, openness):
    # The cell GATE: bars retract up into the lintel as it opens. Posts
    # and lintel are in every frame; only the bars travel.
    for sx in (-0.46, 0.46):
        add_box(f"post{sx}", (sx, 0, 1.06), (0.12, 0.24, 2.12), M["iron"])
    add_box("lintel", (0, 0, 2.12), (0.98, 0.24, 0.16), M["iron"])
    h = 1.9 * (1.0 - openness)
    if h > 0.05:
        zc = 2.0 - h / 2
        for i in range(4):
            add_cylinder(f"bar{i}", (-0.3 + i * 0.2, 0, zc), 0.035, h,
                         M["steel"], axis="z")


def gauge_panel(M):
    add_box("panel", (0, 0.12, 0.5), (0.7, 0.1, 0.9), M["steel"])
    for i in range(2):
        add_cylinder(f"dial{i}", (-0.15 + i * 0.3, 0.05, 0.68), 0.09, 0.04,
                     M["glow_warm" if i else "glow_cyan"], axis="y", seg=12)
    add_box("conduit", (0, 0.14, 0.05), (0.1, 0.08, 0.2), M["pipe"])


PROPS = [
    ("exit_door", exit_door, 2.0),
    ("pebbles_a", lambda M: pebbles(M, 1), 1.0),
    ("pebbles_b", lambda M: pebbles(M, 7), 1.0),
    ("ore_chunk", ore_chunk, 0.9),
    ("ore_pile", ore_pile, 1.2),
    ("pickaxe", pickaxe, 1.2),
    ("crystal", crystal, 1.2),
    ("pallet", pallet, 1.3),
    ("barrel", barrel, 1.1),
    ("box_spill", box_spill, 1.2),
    ("cable_coil", cable_coil, 1.1),
    ("pipe_segment", pipe_segment, 1.3),
    ("warning_cone", warning_cone, 1.0),
    ("coolant_puddle", coolant_puddle, 1.1),
    ("gauge_panel", gauge_panel, 1.5),
    ("jail_bars", jail_bars, 1.4),
    ("ladder_up", ladder_up, 1.9),
    ("bar_counter", bar_counter, 3.4),
    ("table_round", table_round, 1.3),
    ("stool", stool, 1.0),
    ("shelf_rack", shelf_rack, 2.3),
    ("market_stall", market_stall, 2.6),
    ("engine_bench", engine_bench, 2.4),
    ("tool_rack", tool_rack, 1.6),
    ("fuel_pump", fuel_pump, 1.5),
    ("war_desk", war_desk, 2.3),
    ("flag_stand", flag_stand, 1.9),
    ("crate_stack", crate_stack, 1.4),
    ("container_a", lambda M: container(M, "cont_red"), 4.5),
    ("container_b", lambda M: container(M, "cont_blue"), 4.5),
    ("ore_cart", ore_cart, 1.2),
    ("timber_brace", timber_brace, 1.7),
    ("lantern", lantern, 1.6),
    ("pump_unit", pump_unit, 2.6),
    ("pipe_valve", pipe_valve, 1.3),
    ("stairs_down", lambda M: stairs(M, True), 1.3),
    ("stairs_up", lambda M: stairs(M, False), 1.3),
]


def render_prop(name, builder, extent):
    reset()
    builder(mats())
    apply_shear()
    setup_scene(extent * 2.2, RES, freestyle_thick=1.4)
    cam_oblique()
    light_for_contrast()
    tmp = os.path.join(OUT, f"_prop_{name}.png")
    render_to(tmp)
    img = Image.open(tmp).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    # Scale so the render's tile pitch lands on PX_PER_TILE → 32 in game.
    px_per_unit = RES / (extent * 2.2)
    scale = PX_PER_TILE / px_per_unit
    img = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.LANCZOS,
    )
    return add_contact_shadow(img)


def add_contact_shadow(img):
    """Soft ellipse where the prop meets the floor. Without one nothing looks
    like it is TOUCHING the ground, no matter how well it is shaded."""
    pad = max(3, img.width // 10)
    w, h = img.width + 2 * pad, img.height + pad
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(sh)
    ew, eh = int(img.width * 0.92), max(4, int(img.width * 0.22))
    cx, cy = w // 2, img.height + pad // 2
    d.ellipse((cx - ew // 2, cy - eh // 2, cx + ew // 2, cy + eh // 2),
              fill=(12, 14, 20, 115))
    sh = sh.filter(ImageFilter.GaussianBlur(max(1.2, img.width / 26)))
    sh.alpha_composite(img, (pad, 0))
    return sh


def main():
    preview = "preview" in sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DEST, exist_ok=True)
    rendered = []
    for name, builder, extent in PROPS:
        print(f"— {name}")
        img = render_prop(name, builder, extent)
        if not preview:
            img.save(os.path.join(DEST, f"{name}.png"))
        rendered.append((name, img))
    cell, pad, cols = 150, 8, 5
    rows = (len(rendered) + cols - 1) // cols
    cv = Image.new("RGBA", (pad + cols * (cell + pad),
                            pad + rows * (cell + 22 + pad)), (30, 32, 40, 255))
    d = ImageDraw.Draw(cv)
    for i, (name, img) in enumerate(rendered):
        x = pad + (i % cols) * (cell + pad)
        y = pad + (i // cols) * (cell + 22 + pad)
        im = img
        if im.width > cell or im.height > cell:
            f = min(cell / im.width, cell / im.height)
            im = im.resize((int(im.width * f), int(im.height * f)),
                           Image.LANCZOS)
        cv.alpha_composite(im, (x + (cell - im.width) // 2,
                                y + cell - im.height))
        d.text((x + 4, y + cell + 4), name, fill=(220, 224, 232, 255))
    cv.save(os.path.join(OUT, "interior_props_montage.png"))
    print("montage -> out/interior_props_montage.png")

    # ── exit-door panel frames, bbox-aligned with the static frame so the
    # in-game overlay (BuildingDoor) sits pixel-perfect over the prop ──
    extent = 2.0
    def render_raw(build):
        reset()
        setup_scene(extent * 2.2, RES, freestyle_thick=1.4)
        cam_oblique()
        build(mats())
        tmp = os.path.join(OUT, "_prop_exitdoor_pass.png")
        render_to(tmp)
        return Image.open(tmp).convert("RGBA")

    union = render_raw(lambda M: (exit_door_frame(M), exit_door_panels(M, 0.0)))
    bbox = union.getbbox()
    px_per_unit = RES / (extent * 2.2)
    scale = PX_PER_TILE / px_per_unit
    def finish(img):
        img = img.crop(bbox)
        return img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
    if not preview:
        for k in range(4):
            o = k / 3.0
            if k == 3:
                w, h = finish(union).size
                Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(
                    os.path.join(DEST, f"exit_door_panel{k}.png"))
            else:
                img = render_raw(lambda M, o=o: exit_door_panels(M, o))
                finish(img).save(os.path.join(DEST, f"exit_door_panel{k}.png"))
        # The static prop itself, same bbox.
        finish(render_raw(exit_door_frame)).save(os.path.join(DEST, "exit_door.png"))

        # ── jail gate frames (frame 0 = closed defines the shared bbox) ──
        g0 = render_raw(lambda M: jail_gate(M, 0.0))
        gb = g0.getbbox()
        def gfinish(img):
            img = img.crop(gb)
            return img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                Image.LANCZOS,
            )
        for k in range(4):
            img = render_raw(lambda M, o=k / 3.0: jail_gate(M, o))
            gfinish(img).save(os.path.join(DEST, f"jail_gate{k}.png"))
    print("exit door frames baked")


if __name__ == "__main__":
    main()
