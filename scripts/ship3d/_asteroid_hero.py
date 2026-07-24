"""Larger crisp showcase: tumble + mineral embedding + per-asteroid variety.
Reuses asteroid_proto's geometry/materials at higher resolution."""
import math
import os
import random

import bpy  # noqa
import numpy as np
from PIL import Image, ImageDraw

import asteroid_proto as A
from blender_gen import render_to, reset, setup_scene, toon_material

OUT = A.OUT
RES = 256
NANG = 6


def build(seed):
    reset()
    rng = random.Random(seed)
    rock_mat = A.rock_material("rock", crack_scale=A.crack_scale_for(seed))
    dep_mat = toon_material("dep", A.C(242, 240, 236), spec=0.4)
    bumps = A.gen_bumps(rng)
    waves = A.gen_waves(rng)
    rock = A.make_asteroid("rock", bumps, waves, 1.0, rock_mat, rng, subdiv=3)
    deposits = A.make_deposits(bumps, 1.0, dep_mat, rng, clusters=6)
    empty = bpy.data.objects.new("tumble", None)
    bpy.context.scene.collection.objects.link(empty)
    for o in [rock] + deposits:
        o.parent = empty
    setup_scene(2.9, RES)
    bpy.context.scene.view_layers[0].freestyle_settings.linesets[0].select_crease = False
    return rock, deposits, empty


def render_angle(empty, ang):
    axis = np.array([1.0, 0.4, 0.25]); axis /= np.linalg.norm(axis)
    empty.rotation_mode = "AXIS_ANGLE"
    empty.rotation_axis_angle = (ang, axis[0], axis[1], axis[2])
    tmp = os.path.join(OUT, "_hero_tmp.png")
    render_to(tmp)
    return Image.open(tmp).convert("RGBA")


def main():
    # ── main asteroid (seed 7): tumble frames + deposit mask
    rock, deposits, empty = build(7)
    scene = bpy.context.scene
    angles = [2 * math.pi * i / NANG for i in range(NANG)]
    for o in deposits:
        o.hide_render = True
    rock_frames = [render_angle(empty, a) for a in angles]
    for o in deposits:
        o.hide_render = False
    rock.data.materials.clear()
    rock.data.materials.append(A.holdout_material())
    scene.render.use_freestyle = False
    mask_frames = [render_angle(empty, a) for a in angles]
    best = max(range(NANG), key=lambda i: np.array(mask_frames[i])[..., 3].sum())

    # ── variety: a few other seeds, one frame each (rock only)
    variety = [rock_frames[1]]
    for s in (3, 11, 23):
        rk, deps, emp = build(s)
        for o in deps:
            o.hide_render = True
        variety.append(render_angle(emp, 0.7))

    def tint(mask, color):
        a = np.array(mask, float)
        for k in range(3):
            a[..., k] *= color[k] / 255.0
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"), "RGBA")

    def embed(i, color):
        r = rock_frames[i].copy()
        if color is not None:
            r.alpha_composite(tint(mask_frames[i], color))
        return r

    cell, pad = 190, 12
    bgc, cellbg = (18, 20, 26, 255), (44, 46, 54, 255)

    def place(cv, img, x, y, label=""):
        im = img.resize((cell, cell), Image.LANCZOS)
        b = Image.new("RGBA", (cell, cell), cellbg)
        b.alpha_composite(im)
        cv.paste(b, (x, y))
        if label:
            ImageDraw.Draw(cv).text((x + 6, y + cell - 16), label, fill=(235, 235, 245, 255))

    ores = [("clean rock", None), ("gold", (236, 190, 72)), ("iron", (150, 92, 66)),
            ("silicates", (122, 138, 162)), ("platinum", (206, 210, 214))]
    cols = 5
    W = cols * cell + (cols + 1) * pad
    H = 3 * cell + 4 * pad + 66
    cv = Image.new("RGBA", (W, H), bgc)
    d = ImageDraw.Draw(cv)

    d.text((pad, 6), "TUMBLING (same asteroid, different faces)", fill=(240, 240, 250, 255))
    for k, i in enumerate([0, 1, 3, 4]):
        place(cv, rock_frames[i], pad + k * (cell + pad), 26)

    y2 = 26 + cell + 26
    d.text((pad, y2 - 18), "MINERAL EMBEDDING (same rock, ore tinted on the deposits)",
           fill=(240, 240, 250, 255))
    for k, (name, col) in enumerate(ores):
        place(cv, embed(best, col), pad + k * (cell + pad), y2, name)

    y3 = y2 + cell + 26
    d.text((pad, y3 - 18), "PER-ASTEROID VARIETY (different seeds → different shape + cracks)",
           fill=(240, 240, 250, 255))
    for k, im in enumerate(variety):
        place(cv, im, pad + k * (cell + pad), y3)

    cv.save(os.path.join(OUT, "_asteroid_hero.png"))
    print("saved _asteroid_hero.png")


if __name__ == "__main__":
    main()
