#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Prototype interior tiles: LimeZu-style binary floor/wall with wall height.

Interiors are really BINARY (floor vs wall), not the 6-tier depth field the
exterior autotiler assumes. This generates a 2-row, 50-column atlas —
row 0 flat floor, row 1 wall-TOP (the cap seen from above) — both autotiled
with the same blob47 columns the engine already uses, PLUS a separate
shaded wall-FACE strip that the renderer hangs below south-facing walls to
give the walls real height (the depth trick from limezu/moderninteriors).

Outputs to assets/sprites/worlds/:
  interior_proto_atlas.png   (50*32 x 2*32)
  interior_proto_face.png    (4 variants x 32 wide, 44 tall)

Run:  uv run scripts/ship3d/interior_proto_gen.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw

TILE = 32
FACE_H = 44  # wall face ~1.4 tiles tall
WORLDS = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "sprites", "worlds")
rng = np.random.default_rng(0x1117E)

# ── blob47 (must match terrain_gen.py / blob47_lut.ron column order) ────────
TL, T, TR = 1, 2, 4
L, R_ = 8, 16
BL, B, BR = 32, 64, 128


def reduce_to_47(m):
    if not (m & L) or not (m & T): m &= ~TL
    if not (m & R_) or not (m & T): m &= ~TR
    if not (m & L) or not (m & B): m &= ~BL
    if not (m & R_) or not (m & B): m &= ~BR
    return m


BLOB47 = sorted({reduce_to_47(m) for m in range(256)})
INTERIOR_COL = BLOB47.index(255)  # 46
N_VARIANTS = 4


# ── base tiles (return HxWx3 float arrays) ──────────────────────────────────
def floor_base(variant):
    """Clean station deck: darker cool-grey panels, thin seams, faint sheen.
    Kept clearly darker than the wall cap so floor vs wall reads at a glance."""
    base = np.array([84, 92, 106], float)
    arr = np.tile(base, (TILE, TILE, 1))
    # gentle radial sheen (top-down light)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    r = np.hypot(xx - TILE / 2, yy - TILE * 0.35) / TILE
    arr *= (1.08 - 0.16 * r)[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    seam = (60, 66, 78)
    # 2x2 panel seams
    d.line([(TILE // 2, 0), (TILE // 2, TILE)], fill=seam)
    d.line([(0, TILE // 2), (TILE, TILE // 2)], fill=seam)
    arr = np.asarray(img, float)
    # per-variant grain
    arr += (rng.random((TILE, TILE, 1)) - 0.5) * 7 * (0.6 + 0.4 * variant)
    return arr


def walltop_base(variant):
    """The top CAP of a wall seen from above: brighter brushed metal, a lit
    north edge, panel line. Reads as the top surface of a thick wall."""
    base = np.array([158, 165, 178], float)
    arr = np.tile(base, (TILE, TILE, 1))
    # brushed vertical streaks
    streak = (rng.random((1, TILE)) - 0.5) * 10
    arr += streak[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    # bright top (north) highlight + dark bottom lip
    d.line([(0, 1), (TILE, 1)], fill=(196, 202, 214))
    d.line([(0, TILE - 2), (TILE, TILE - 2)], fill=(96, 102, 116))
    d.line([(0, TILE // 2), (TILE, TILE // 2)], fill=(120, 126, 140))
    arr = np.asarray(img, float)
    return arr


def seam_recess(arr, mask):
    """Darken the 2px edge toward any neighbour that ISN'T the same tier
    (same convention as terrain_gen.render_interior_tile)."""
    for bit, sl in ((T, (slice(0, 2), slice(None))),
                    (B, (slice(TILE - 2, TILE), slice(None))),
                    (R_, (slice(None), slice(TILE - 2, TILE))),
                    (L, (slice(None), slice(0, 2)))):
        if not (mask & bit):
            arr[sl] *= 0.5
    for diag, c1, c2, sl in ((TL, T, L, (slice(0, 2), slice(0, 2))),
                             (TR, T, R_, (slice(0, 2), slice(TILE - 2, TILE))),
                             (BL, B, L, (slice(TILE - 2, TILE), slice(0, 2))),
                             (BR, B, R_, (slice(TILE - 2, TILE), slice(TILE - 2, TILE)))):
        if not (mask & diag) and (mask & c1) and (mask & c2):
            arr[sl] *= 0.5
    return arr


def render_cell(row, col, variant):
    mask = BLOB47[col] if col < 47 else 255
    v = variant if mask == 255 else 0
    arr = (floor_base(v) if row == 0 else walltop_base(v)).copy()
    arr = seam_recess(arr, mask)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB").convert("RGBA")


def build_atlas():
    cols = 47 + (N_VARIANTS - 1)  # 50
    atlas = Image.new("RGBA", (cols * TILE, 2 * TILE), (0, 0, 0, 0))
    for row in range(2):
        for col in range(47):
            atlas.paste(render_cell(row, col, 0), (col * TILE, row * TILE))
        for vv in range(1, N_VARIANTS):
            atlas.paste(render_cell(row, INTERIOR_COL, vv),
                        ((46 + vv) * TILE, row * TILE))
    out = os.path.abspath(os.path.join(WORLDS, "interior_proto_atlas.png"))
    atlas.save(out)
    print(f"  interior_proto_atlas.png  {atlas.size[0]}x{atlas.size[1]}")


def build_face():
    """Vertical wall face: bright under the cap, darkening down, panel seams,
    soft contact shadow at the floor. Four variants for variety."""
    n = 4
    sheet = Image.new("RGBA", (n * TILE, FACE_H), (0, 0, 0, 0))
    for v in range(n):
        col = np.zeros((FACE_H, TILE, 4), float)
        # vertical light ramp: lit at top (just under cap) → shadow at bottom
        ramp = np.linspace(1.12, 0.62, FACE_H)[:, None]
        base = np.array([120, 128, 142], float)
        col[..., :3] = base * ramp[..., None]
        col[..., 3] = 255
        img = Image.fromarray(np.clip(col, 0, 255).astype("uint8"))
        d = ImageDraw.Draw(img)
        # cap shadow line at very top, panel seams, foot contact shadow
        d.line([(0, 0), (TILE, 0)], fill=(70, 76, 88, 255))
        for sx in (TILE // 3, 2 * TILE // 3):
            j = sx + int((rng.random() - 0.5) * 3)
            d.line([(j, 2), (j, FACE_H - 3)], fill=(92, 98, 112, 255))
        # a couple of rivets
        for ry in (5, FACE_H - 8):
            d.point([(4, ry), (TILE - 5, ry)], fill=(150, 156, 168, 255))
        # bottom contact shadow (soft, semi-transparent)
        sh = np.asarray(img, float)
        k = 4
        fade = np.linspace(0.55, 1.0, k)[:, None, None]
        sh[FACE_H - k:, :, :3] *= fade
        img = Image.fromarray(np.clip(sh, 0, 255).astype("uint8"))
        sheet.paste(img, (v * TILE, 0))
    out = os.path.abspath(os.path.join(WORLDS, "interior_proto_face.png"))
    sheet.save(out)
    print(f"  interior_proto_face.png  {sheet.size[0]}x{sheet.size[1]}  ({n} variants)")


if __name__ == "__main__":
    os.makedirs(WORLDS, exist_ok=True)
    build_atlas()
    build_face()
    print("Done.")
