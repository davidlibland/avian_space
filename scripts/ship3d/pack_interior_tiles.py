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
# The station shop keeps the bare asset names the engine already loads; the
# maze venues get a suffix. All share ONE bounding box (below), so the
# engine's cell size + anchor constants are identical for every venue.
VENUES = {"interior": "", "mine": "_mine", "warehouse": "_warehouse",
          "substation": "_substation"}
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
    # Load every wall tile of every venue.
    imgs = {(v, n): Image.open(os.path.join(TILES, f"_it_{v}_{n}.png")).convert("RGBA")
            for v in VENUES for n in ORDER}

    # ONE common box across ALL venues → shared canvas + shared ground origin,
    # so the venue-specific greebles never shift the anchor.
    boxes = [content_box(im) for im in imgs.values()]
    l = min(b[0] for b in boxes)
    t = min(b[1] for b in boxes)
    r = max(b[2] for b in boxes)
    b = max(b[3] for b in boxes)
    cw, ch = r - l, b - t

    ox = (ORIGIN - l) * SCALE
    oy = (ORIGIN - t) * SCALE
    aw = round(cw * SCALE)
    ah = round(ch * SCALE)

    floor_counts = {}
    for v, suffix in VENUES.items():
        strip = Image.new("RGBA", (aw * len(ORDER), ah), (0, 0, 0, 0))
        for i, n in enumerate(ORDER):
            cell = imgs[(v, n)].crop((l, t, r, b)).resize((aw, ah), Image.LANCZOS)
            strip.paste(cell, (i * aw, 0))
        strip.save(os.path.join(WORLDS, f"interior_cabinet_walls{suffix}.png"))
        # Floor variants → a horizontal strip (1 cell for a single-tile floor,
        # N interchangeable cells for organic floors like the mine's dirt).
        variants = []
        k = 0
        while os.path.exists(os.path.join(TILES, f"_it_{v}_floor{k}.png")):
            variants.append(Image.open(os.path.join(TILES, f"_it_{v}_floor{k}.png"))
                            .convert("RGBA").resize((ASSET_PXU, ASSET_PXU), Image.LANCZOS))
            k += 1
        floor_counts[v] = len(variants)
        fstrip = Image.new("RGBA", (ASSET_PXU * len(variants), ASSET_PXU), (0, 0, 0, 0))
        for i, fim in enumerate(variants):
            fstrip.paste(fim, (i * ASSET_PXU, 0))
        fstrip.save(os.path.join(WORLDS, f"interior_cabinet_floor{suffix}.png"))

    ax = ox / aw - 0.5
    ay = 0.5 - oy / ah
    print(f"venues         = {', '.join(VENUES)}")
    print(f"floor_variants = {floor_counts}")
    print(f"cell_px        = {aw} x {ah}")
    print(f"ground_origin  = ({ox:.1f}, {oy:.1f}) px  (from top-left of cell)")
    print(f"bevy_anchor    = ({ax:.5f}, {ay:.5f})")
    print(f"units_per_cell = {cw / SRC_PXU:.4f} x {ch / SRC_PXU:.4f}")
    print(f"asset_pxu      = {ASSET_PXU}  (world unit = TILE_PX in game)")
    print("order          =", " ".join(ORDER))


if __name__ == "__main__":
    main()
