"""
asteroid_gen.py — bake the asteroid shape library for the game.

For each of N seeds, bakes two synced tumble atlases (4x4 = 16 frames):
  rock_<i>.png  — the lit, cracked, textured rock tumbling
  dep_<i>.png   — its ore deposits only (white, tumbling in sync) for runtime tint

Each shape is normalized so it never crops at any tumble angle.

Run:  scripts/.blender_venv/bin/python asteroid_gen.py
Out:  assets/sprites/asteroids/rock_<i>.png, dep_<i>.png
"""

import math
import os
import random

import bpy  # noqa
import numpy as np
from PIL import Image

import asteroid_proto as A
from blender_gen import render_to, reset, setup_scene, toon_material

N_SHAPES = 12
T = 64               # tumble frames (more = smoother tumble)
TILE = 128
COLS = 8             # 8×8 grid = 64 tiles
ROWS = 8
ORTHO = 2.2          # frame; with the 0.9 fit-normalize this never crops
FIT = 0.9
ASTEROID_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                            "assets", "sprites", "asteroids")
TMP = os.path.join(A.OUT, "_ast_tmp.png")
os.makedirs(ASTEROID_DIR, exist_ok=True)


def C(*rgb):
    return tuple(v / 255.0 for v in rgb)


def render_tumble(empty, axis):
    axis = axis / np.linalg.norm(axis)
    empty.rotation_mode = "AXIS_ANGLE"
    frames = []
    for t in range(T):
        ang = 2 * math.pi * t / T
        empty.rotation_axis_angle = (ang, axis[0], axis[1], axis[2])
        render_to(TMP)
        f = Image.open(TMP).convert("RGBA")
        if f.size != (TILE, TILE):
            f = f.resize((TILE, TILE), Image.LANCZOS)
        frames.append(f)
    return frames


def pack(frames):
    atlas = Image.new("RGBA", (COLS * TILE, ROWS * TILE), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        r, c = divmod(i, COLS)
        atlas.paste(f, (c * TILE, r * TILE), f)
    return atlas


def bake_shape(seed, idx):
    reset()
    rng = random.Random(seed)
    rock_mat = A.rock_material(f"rock{idx}", crack_scale=A.crack_scale_for(seed))
    dep_mat = toon_material(f"dep{idx}", C(242, 240, 236), spec=0.4)
    bumps = A.gen_bumps(rng)
    waves = A.gen_waves(rng)
    fit = FIT / A.max_disp(bumps)                      # normalize → no crop
    rock = A.make_asteroid(f"rock{idx}", bumps, waves, fit, rock_mat, rng, subdiv=3)
    deposits = A.make_deposits(bumps, fit, dep_mat, rng, clusters=rng.randint(4, 7))

    empty = bpy.data.objects.new("tumble", None)
    bpy.context.scene.collection.objects.link(empty)
    for o in [rock] + deposits:
        o.parent = empty

    setup_scene(ORTHO, TILE)
    bpy.context.scene.view_layers[0].freestyle_settings.linesets[0].select_crease = False
    scene = bpy.context.scene

    # per-shape tumble axis (varies the motion across the field)
    arng = random.Random(seed * 31 + 5)
    axis = np.array([arng.uniform(0.3, 1.0), arng.uniform(-0.6, 0.6),
                     arng.uniform(0.1, 0.6)])

    for o in deposits:
        o.hide_render = True
    rock_frames = render_tumble(empty, axis)

    for o in deposits:
        o.hide_render = False
    rock.data.materials.clear()
    rock.data.materials.append(A.holdout_material())
    scene.render.use_freestyle = False
    dep_frames = render_tumble(empty, axis)

    pack(rock_frames).save(os.path.abspath(os.path.join(ASTEROID_DIR, f"rock_{idx}.png")))
    pack(dep_frames).save(os.path.abspath(os.path.join(ASTEROID_DIR, f"dep_{idx}.png")))
    print(f"baked shape {idx} (seed {seed})")


def main():
    for idx in range(N_SHAPES):
        bake_shape(seed=100 + idx * 7, idx=idx)
    print(f"done — {N_SHAPES} asteroid shapes → {ASTEROID_DIR}")


if __name__ == "__main__":
    main()
