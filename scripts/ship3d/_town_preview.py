"""Scratch: composite colony buildings on garden terrain with a tile grid +
footprint shading, to verify footprints land on integer tiles. (underscore =
scratch, not committed)."""
import math
import numpy as np
from PIL import Image, ImageDraw
import terrain_gen as G


class BB:  # mirror buildings3d constants (avoid importing bpy)
    ELEV, RES, ORTHO = 50.0, 320, 17.0
    FOOTPRINTS = {"market": (6, 5), "outfitter": (4, 4), "shipyard": (8, 6),
                  "mechanic": (6, 4), "bar": (6, 5), "pad": (3, 3)}


TILE = G.TILE
BLOB47, r47 = G.BLOB47, G.reduce_to_47
TL, T, TR, L, R_, BL, B, BR = G.TL, G.T, G.TR, G.L, G.R_, G.BL, G.B, G.BR
NB = [(TL, -1, -1), (T, 0, -1), (TR, 1, -1), (L, -1, 0), (R_, 1, 0), (BL, -1, 1), (B, 0, 1), (BR, 1, 1)]

E = math.radians(BB.ELEV)
PXU = BB.RES / BB.ORTHO            # px per world-unit in the raw sprite
S = TILE / PXU                     # scale so 1 unit -> 1 tile
TGT_SY = 2.2 * math.cos(E)         # camera target screen-y


def terrain_img(terr):
    H, W = terr.shape
    atlas = Image.open("../../assets/sprites/worlds/garden_atlas.png").convert("RGB")
    out = Image.new("RGB", (W * TILE, H * TILE))
    for y in range(H):
        for x in range(W):
            t = terr[y, x]; mask = 0
            for bit, dx, dy in NB:
                nx, ny = min(max(x + dx, 0), W - 1), min(max(y + dy, 0), H - 1)
                if terr[ny, nx] >= t:
                    mask |= bit
            col = BLOB47.index(r47(mask))
            if col == G.INTERIOR_COL:
                col = 46 + ((x * 7 + y * 13) % G.N_VARIANTS)
            out.paste(atlas.crop((col * TILE, t * TILE, col * TILE + TILE, t * TILE + TILE)), (x * TILE, y * TILE))
    return out


def anchor_y(D):                   # footprint front-center pixel in the raw RES×RES sprite
    return BB.RES / 2 + (D / 2 * math.sin(E) + TGT_SY) * PXU


def main():
    W, H = 32, 23
    terr = np.full((H, W), 2, int)
    terr[0:2, :] = 3; terr[:, 0:2] = 3; terr[:, W - 2:] = 3; terr[H - 2:, :] = 3
    terr[2:6, 3:8] = 3
    base = terrain_img(terr).convert("RGBA")

    # (fn, front-center grid x, front grid y) — even W → integer x; pad(3) → x.5
    town = [("shipyard", 9, 10), ("market", 19, 9), ("outfitter", 26, 9),
            ("bar", 9, 18), ("mechanic", 19, 18), ("pad", 26, 17)]

    # footprint shading (under everything)
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for fn, cx, cy in town:
        Wt, Dt = BB.FOOTPRINTS[fn]
        x0, x1 = (cx - Wt / 2) * TILE, (cx + Wt / 2) * TILE
        y0, y1 = (cy - Dt) * TILE, cy * TILE
        od.rectangle([x0, y0, x1, y1], fill=(255, 220, 60, 60), outline=(255, 210, 40, 255), width=2)
    base.alpha_composite(ov)

    # tile grid
    gd = ImageDraw.Draw(base)
    for x in range(0, base.width + 1, TILE):
        gd.line([(x, 0), (x, base.height)], fill=(255, 255, 255, 40))
    for y in range(0, base.height + 1, TILE):
        gd.line([(0, y), (base.width, y)], fill=(255, 255, 255, 40))

    # buildings, Y-sorted, placed by analytic footprint anchor
    for fn, cx, cy in sorted(town, key=lambda b: b[2]):
        Dt = BB.FOOTPRINTS[fn][1]
        spr = Image.open(f"out/_b_colony_{fn}.png").convert("RGBA")
        spr = spr.resize((int(spr.width * S), int(spr.height * S)), Image.LANCZOS)
        px = cx * TILE - (BB.RES / 2) * S
        py = cy * TILE - anchor_y(Dt) * S
        base.alpha_composite(spr, (int(px), int(py)))

    # footprint outlines ON TOP (so alignment is visible over the buildings)
    td = ImageDraw.Draw(base)
    for fn, cx, cy in town:
        Wt, Dt = BB.FOOTPRINTS[fn]
        td.rectangle([(cx - Wt / 2) * TILE, (cy - Dt) * TILE, (cx + Wt / 2) * TILE, cy * TILE],
                     outline=(255, 60, 60, 255), width=2)

    base.convert("RGB").resize((base.width * 2, base.height * 2), Image.LANCZOS).save("out/_town_grid.png")
    print("saved out/_town_grid.png")


if __name__ == "__main__":
    main()
