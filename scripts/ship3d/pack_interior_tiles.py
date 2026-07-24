#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Pack the baked cabinet interior tiles into game-ready atlases.

The offset fix: every wall tile is baked in the SAME 1024px frame with the
ground origin (world 0,0) at the exact centre pixel (512,512). Cropping each
tile to its OWN content box rounds differently per tile, which is what made
adjacent tiles jitter by a pixel. Instead we crop ALL of them to ONE common
box (the union), so they share an identical canvas and an identical
ground-origin pixel → they line up exactly.

Outputs (assets/sprites/worlds/):
  interior_cabinet_walls.png   9-cell strip: N S E W CNW CNE CSW CSE V
  interior_cabinet_floor.png   single floor cell

Prints the cell size + ground-origin anchor for the engine.

Run:  uv run scripts/ship3d/pack_interior_tiles.py
"""
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
TILES = os.path.join(HERE, "out", "tiles")
WORLDS = os.path.join(HERE, "..", "..", "assets", "sprites", "worlds")

ORDER = ["N", "S", "E", "W", "CNW", "CNE", "CSW", "CSE", "V"]
SRC_PXU = 256          # the bake renders 256 px per world unit
ASSET_PXU = 64         # store at 64 px/unit (crisp 4x downscale; game is 32)
SCALE = ASSET_PXU / SRC_PXU
ORIGIN = 512           # world (0,0) pixel in the 1024px bake frame


def content_box(im, thr=12):
    a = np.asarray(im)[:, :, 3]
    ys, xs = np.where(a > thr)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def main():
    os.makedirs(WORLDS, exist_ok=True)
    imgs = {n: Image.open(os.path.join(TILES, f"_it_{n}.png")).convert("RGBA")
            for n in ORDER}

    # union content box across every wall tile → one common canvas
    boxes = [content_box(imgs[n]) for n in ORDER]
    l = min(b[0] for b in boxes)
    t = min(b[1] for b in boxes)
    r = max(b[2] for b in boxes)
    b = max(b[3] for b in boxes)
    cw, ch = r - l, b - t

    # ground origin relative to the common box, then to asset px
    ox = (ORIGIN - l) * SCALE
    oy = (ORIGIN - t) * SCALE
    aw = round(cw * SCALE)
    ah = round(ch * SCALE)

    strip = Image.new("RGBA", (aw * len(ORDER), ah), (0, 0, 0, 0))
    for i, n in enumerate(ORDER):
        cell = imgs[n].crop((l, t, r, b)).resize((aw, ah), Image.LANCZOS)
        strip.paste(cell, (i * aw, 0))
    strip.save(os.path.join(WORLDS, "interior_cabinet_walls.png"))

    floor = Image.open(os.path.join(TILES, "_it_floor.png")).convert("RGBA")
    floor = floor.resize((ASSET_PXU, ASSET_PXU), Image.LANCZOS)
    floor.save(os.path.join(WORLDS, "interior_cabinet_floor.png"))

    # Bevy Anchor is a fraction from centre, +Y up: origin at (ox,oy) top-left px
    ax = ox / aw - 0.5
    ay = 0.5 - oy / ah
    print(f"cell_px        = {aw} x {ah}")
    print(f"ground_origin  = ({ox:.1f}, {oy:.1f}) px  (from top-left of cell)")
    print(f"bevy_anchor    = ({ax:.5f}, {ay:.5f})")
    print(f"units_per_cell = {cw / SRC_PXU:.4f} x {ch / SRC_PXU:.4f}")
    print(f"asset_pxu      = {ASSET_PXU}  (world unit = TILE_PX in game)")
    print("order          =", " ".join(ORDER))


if __name__ == "__main__":
    main()
