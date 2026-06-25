"""
bake_atlases.py — bake per-ship rotation/thrust sprite atlases for Bevy.

For each ship we render an 8×8 atlas of TILE×TILE frames:
  * frames  0..N-1   : idle  (no drive flame)
  * frames  N..2N-1  : thrust (drive flame on)
Within each block, frame i is the ship lit as if its heading were i·(2π/N):
the SHIP stays nose-up and the LIGHT RIG is spun by -i·step.  At runtime Bevy
keeps rotating the sprite smoothly by the physics heading and picks
  index = round(heading/step) mod N  (+N while thrusting)
so the baked lighting lands world-fixed → smooth rotation + gliding highlights.

N = 32  →  step 11.25°, packs as 8 cols × 8 rows (32 idle + 32 thrust).

Run:  scripts/.blender_venv/bin/python bake_atlases.py [ship1 ship2 ...]
Out:  assets/sprites/ships/atlas/<ship>.png   (+ scripts/ship3d/out/_atlas_verify.png)
"""

import math
import os
import sys

import bpy
from PIL import Image

from blender_gen import render_to, reset, setup_scene
from fleet_gen import REGISTRY, _set_exhaust_visible

N = 32
COLS, ROWS = 8, 8           # 64 tiles = N idle + N thrust
TILE = 128
ATLAS_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                         "assets", "sprites", "ships", "atlas")
TMP = os.path.join(os.path.dirname(__file__), "out", "_tmp_frame.png")
os.makedirs(ATLAS_DIR, exist_ok=True)


def bake(name):
    builder, ortho, _, _ = REGISTRY[name]
    reset()
    builder()
    pivot = setup_scene(ortho, TILE)
    step = 2 * math.pi / N
    atlas = Image.new("RGBA", (COLS * TILE, ROWS * TILE), (0, 0, 0, 0))
    idx = 0
    for thrust in (False, True):
        _set_exhaust_visible(thrust)
        for i in range(N):
            pivot.rotation_euler = (0, 0, -i * step)
            render_to(TMP)
            frame = Image.open(TMP).convert("RGBA")
            if frame.size != (TILE, TILE):
                frame = frame.resize((TILE, TILE), Image.LANCZOS)
            r, c = divmod(idx, COLS)
            atlas.paste(frame, (c * TILE, r * TILE), frame)
            idx += 1
    out = os.path.abspath(os.path.join(ATLAS_DIR, f"{name}.png"))
    atlas.save(out)
    print("baked", name, "->", out)


def verify(name):
    """Simulate the Bevy runtime: pick frame i=round(h/step) and rotate the
    tile by the heading; the highlight should stay world-fixed while the
    silhouette turns.  Saved to out/_atlas_verify.png."""
    atlas = Image.open(os.path.abspath(os.path.join(ATLAS_DIR, f"{name}.png")))
    step_deg = 360.0 / N
    headings = [0, 45, 90, 135, 180, 270]
    cell = TILE * 2
    strip = Image.new("RGBA", (cell * len(headings), cell + 16), (34, 36, 44, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(strip)
    for j, h in enumerate(headings):
        i = round(h / step_deg) % N            # idle block
        r, c = divmod(i, COLS)
        tile = atlas.crop((c * TILE, r * TILE, c * TILE + TILE, r * TILE + TILE))
        # Bevy rotates the sprite by heading h (CCW in world == CCW in this
        # nose-up image); PIL rotate is CCW for positive angles.
        rot = tile.rotate(h, resample=Image.BICUBIC, expand=False)
        rot = rot.resize((cell, cell), Image.LANCZOS)
        d.text((j * cell + 4, 2), f"{h}°", fill=(235, 235, 245, 255))
        strip.paste(rot, (j * cell, 16), rot)
    strip.save(os.path.join(os.path.dirname(__file__), "out", "_atlas_verify.png"))
    print("saved _atlas_verify.png")


def main():
    names = sys.argv[1:] or list(REGISTRY.keys())
    for name in names:
        bake(name)
    verify(names[0])


if __name__ == "__main__":
    main()
