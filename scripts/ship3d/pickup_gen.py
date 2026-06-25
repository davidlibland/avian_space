"""
pickup_gen.py — tumbling "pure deposit" crystal atlas for pickups.

Bakes a single solid faceted crystal as a 4×4 / 16-frame tumble atlas (white,
tintable per commodity).  The game tints it by commodity colour, spins it
in-plane via physics, and advances the tumble frame over time.

Run:  scripts/.blender_venv/bin/python pickup_gen.py
Out:  assets/sprites/pickups/crystal.png (4×4 atlas) + out/_pickups.png preview
"""

import math
import os
import random

import bpy  # noqa
import bmesh
import numpy as np
from PIL import Image, ImageDraw

from blender_gen import render_to, reset, setup_scene, toon_material

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                         "assets", "sprites", "pickups")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(ASSET_DIR, exist_ok=True)

T = 16
TILE = 64
COLS = 4


def C(*rgb):
    return tuple(v / 255.0 for v in rgb)


def build_crystal():
    """A SINGLE solid faceted crystal, normalized so it never crops tumbling."""
    mat = toon_material("cry", C(232, 233, 238), spec=1.1, spec_sharp=0.82)
    rng = random.Random(8)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.5)
    for v in bm.verts:
        v.co = v.co * (1.0 + rng.uniform(-0.32, 0.32))   # angular faceting
        v.co.z *= 1.45                                    # gem elongation
        v.co.x *= 0.82
    # normalize max extent so any tumble angle fits the frame
    mx = max(v.co.length for v in bm.verts)
    for v in bm.verts:
        v.co *= 0.85 / mx
    me = bpy.data.meshes.new("crystal")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = False
    ob = bpy.data.objects.new("crystal", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def render_tumble(empty):
    axis = np.array([0.3, 1.0, 0.4]); axis /= np.linalg.norm(axis)
    empty.rotation_mode = "AXIS_ANGLE"
    frames = []
    for t in range(T):
        ang = 2 * math.pi * t / T
        empty.rotation_axis_angle = (ang, axis[0], axis[1], axis[2])
        tmp = os.path.join(OUT, "_pickup_tmp.png")
        render_to(tmp)
        f = Image.open(tmp).convert("RGBA")
        if f.size != (TILE, TILE):
            f = f.resize((TILE, TILE), Image.LANCZOS)
        frames.append(f)
    return frames


def main():
    reset()
    crystal = build_crystal()
    empty = bpy.data.objects.new("tumble", None)
    bpy.context.scene.collection.objects.link(empty)
    crystal.parent = empty
    setup_scene(2.0, TILE)
    bpy.context.scene.view_layers[0].freestyle_settings.linesets[0].select_crease = False

    frames = render_tumble(empty)
    atlas = Image.new("RGBA", (COLS * TILE, COLS * TILE), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        r, c = divmod(i, COLS)
        atlas.paste(f, (c * TILE, r * TILE), f)
    atlas.save(os.path.join(ASSET_DIR, "crystal.png"))
    print(f"wrote crystal.png ({COLS}x{COLS} tumble atlas, {TILE}px tiles)")

    # preview: tumble strip + tinted (one frame)
    def tint(img, color):
        a = np.array(img.convert("RGBA"), float)
        for k in range(3):
            a[..., k] *= color[k] / 255.0
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"), "RGBA")

    cell = 110
    cv = Image.new("RGBA", (8 * (cell + 6) + 6, 2 * (cell + 6) + 30), (22, 24, 30, 255))
    d = ImageDraw.Draw(cv)
    d.text((8, 2), "TUMBLE (every other frame)", fill=(240, 240, 250, 255))
    for k in range(8):
        im = frames[k * 2].resize((cell, cell), Image.LANCZOS)
        b = Image.new("RGBA", (cell, cell), (44, 46, 54, 255)); b.alpha_composite(im)
        cv.paste(b, (6 + k * (cell + 6), 22))
    y2 = 22 + cell + 16
    d.text((8, y2 - 14), "TINTED (frame 0)", fill=(240, 240, 250, 255))
    ores = [("white", (255, 255, 255)), ("gold", (236, 190, 72)), ("iron", (150, 92, 66)),
            ("silicates", (122, 138, 162)), ("platinum", (206, 210, 214)),
            ("chemicals", (110, 180, 130))]
    for k, (name, col) in enumerate(ores):
        im = tint(frames[0], col).resize((cell, cell), Image.LANCZOS)
        b = Image.new("RGBA", (cell, cell), (44, 46, 54, 255)); b.alpha_composite(im)
        cv.paste(b, (6 + k * (cell + 6), y2))
        d.text((6 + k * (cell + 6) + 4, y2 + cell - 14), name, fill=(235, 235, 245, 255))
    cv.save(os.path.join(OUT, "_pickups.png"))
    print("saved _pickups.png")


if __name__ == "__main__":
    main()
