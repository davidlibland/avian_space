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
from PIL import Image, ImageDraw

from blender_gen import (add_box, add_cylinder, add_sphere, glow_material,
                         render_to, reset, setup_scene, toon_material)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
DEST = os.path.join(HERE, "..", "..", "assets", "sprites", "worlds",
                    "interior_props")
ELEV = 50.0
RES = 512
PX_PER_TILE = 34  # matches buildings3d — game downsamples to 32


def C(*rgb):
    return tuple(v / 255.0 for v in rgb)


def mats():
    return {
        "wood": toon_material("wood", C(122, 84, 52)),
        "wood_dark": toon_material("wood_dark", C(84, 56, 36)),
        "steel": toon_material("steel", C(150, 156, 166)),
        "iron": toon_material("iron", C(72, 74, 80)),
        "brass": toon_material("brass", C(180, 134, 62), spec=0.5),
        "cloth_red": toon_material("cloth_red", C(168, 62, 48)),
        "cloth_tan": toon_material("cloth_tan", C(196, 168, 120)),
        "crate": toon_material("crate", C(146, 112, 66)),
        "cont_red": toon_material("cont_red", C(150, 62, 44)),
        "cont_blue": toon_material("cont_blue", C(52, 84, 130)),
        "glow_warm": glow_material("glow_warm", C(255, 196, 96), 2.4),
        "glow_cyan": glow_material("glow_cyan", C(80, 220, 255), 2.4),
        "pipe": toon_material("pipe", C(96, 110, 116)),
        "bottle_g": toon_material("bottle_g", C(70, 140, 90), spec=0.6),
        "dark": toon_material("dark", C(30, 30, 34)),
    }


def cam_oblique():
    scene = bpy.context.scene
    cam = scene.camera
    E = math.radians(ELEV)
    dist = 14.0
    cam.location = (0.0, -dist * math.cos(E), dist * math.sin(E))
    tgt = bpy.data.objects.new("tgt", None)
    scene.collection.objects.link(tgt)
    tgt.location = (0, 0, 0.4)
    tr = cam.constraints.new("TRACK_TO")
    tr.target = tgt
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"


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
    add_box("frame", (0, 0, 0.7), (1.9, 0.5, 1.4), M["steel"], bevel=0.02)
    for z in (0.35, 0.8, 1.25):
        add_box(f"shelf{z}", (0, 0, z), (1.8, 0.46, 0.05), M["iron"])
        for i in range(3):
            add_box(f"box{z}{i}", (-0.6 + i * 0.6, 0, z + 0.14),
                    (0.34, 0.3, 0.22), M["crate"], bevel=0.03)


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
    add_box("bench", (0, 0, 0.35), (1.9, 0.8, 0.7), M["iron"], bevel=0.03)
    add_cylinder("block", (-0.3, 0, 0.95), 0.32, 0.9, M["steel"], axis="x")
    add_cylinder("exhaust", (0.55, 0, 0.95), 0.12, 0.5, M["brass"], axis="x")
    add_sphere("glow", (-0.75, 0, 0.95), (0.1, 0.1, 0.1), M["glow_cyan"])


def tool_rack(M):
    add_box("board", (0, 0.15, 0.8), (0.9, 0.1, 1.2), M["wood_dark"])
    for i, z in enumerate((0.5, 0.85, 1.2)):
        add_box(f"tool{i}", (-0.2 + (i % 2) * 0.4, 0.05, z),
                (0.08, 0.06, 0.3), M["steel"])


def fuel_pump(M):
    add_box("body", (0, 0, 0.6), (0.6, 0.4, 1.2), M["cont_red"], bevel=0.06)
    add_box("face", (0, -0.18, 0.85), (0.4, 0.06, 0.3), M["glow_warm"])
    add_cylinder("hose", (0.32, 0, 0.5), 0.045, 0.8, M["dark"], axis="z")


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


PROPS = [
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
    setup_scene(extent * 2.2, RES, freestyle_thick=1.4)
    cam_oblique()
    builder(mats())
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
    return img


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


if __name__ == "__main__":
    main()
