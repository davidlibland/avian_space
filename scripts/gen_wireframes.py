#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pillow",
#     "pyyaml",
# ]
# ///
"""Pre-generate small military-style target wireframes for the HUD.

Outputs to assets/sprites/wireframes/:
  <ship>.png         one per ship (edge-extracted from the baked rotation atlas)
  asteroid.png       generic chunk-of-rock
  pickup.png         generic cargo canister
  planet_<type>.png  one per planet_type (procedural schematic)

Each is a bright-green line drawing with a faint inner fill on transparent — a
"targeting computer" look. Run: .venv/bin/python scripts/gen_wireframes.py
"""
import math
import os

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "assets", "sprites", "wireframes")
os.makedirs(OUT, exist_ok=True)

SIZE = 96
GREEN = (90, 255, 180)
LINE = GREEN + (210,)   # slightly translucent for a lighter, less-bold line
FILL = GREEN + (42,)


# ── shared helpers ───────────────────────────────────────────────────────────
def blank():
    im = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def poly(d, pts, closed=True, w=1):
    p = list(pts) + ([pts[0]] if closed else [])
    d.line(p, fill=LINE, width=w, joint="curve")


def faint_fill(im, shape_pts):
    """Flood a faint green inside a polygon by compositing a filled mask."""
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).polygon(list(shape_pts), fill=FILL[3])
    tint = Image.new("RGBA", (SIZE, SIZE), GREEN + (0,))
    tint.putalpha(mask)
    im.alpha_composite(tint)


# ── ships: edge-extract from the baked atlas (frame 0 = nose-up idle) ─────────
def fit(rgba, pad=9):
    bb = rgba.getbbox()
    if bb:
        rgba = rgba.crop(bb)
    w, h = rgba.size
    s = (SIZE - 2 * pad) / max(w, h)
    rgba = rgba.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(rgba, ((SIZE - rgba.width) // 2, (SIZE - rgba.height) // 2))
    return canvas


def ship_wireframe(atlas_path):
    atlas = Image.open(atlas_path).convert("RGBA")
    tile = atlas.width // 8
    frame = fit(atlas.crop((0, 0, tile, tile)))           # heading-0 idle tile
    a = np.array(frame)[..., 3]
    mask = a > 40
    # internal edges (panel lines / ink) within the silhouette
    edges = np.array(frame.convert("L").filter(ImageFilter.FIND_EDGES))
    edge_line = (edges > 38) & mask
    # silhouette boundary = mask minus its erosion
    am = Image.fromarray((mask * 255).astype("uint8"))
    er = np.array(am.filter(ImageFilter.MinFilter(3))) == 0
    boundary = mask & er
    line = Image.fromarray(((edge_line | boundary) * 255).astype("uint8"))
    line = np.array(line) > 0   # crisp 1px lines (no thickening — lighter)
    out = np.zeros((SIZE, SIZE, 4), dtype="uint8")
    out[mask] = [GREEN[0], GREEN[1], GREEN[2], FILL[3]]
    out[line] = LINE
    return Image.fromarray(out, "RGBA")


# ── generic procedural schematics ────────────────────────────────────────────
def asteroid_wireframe():
    im, d = blank()
    cx = cy = SIZE / 2
    rs = [35, 27, 37, 29, 38, 25, 33, 30]
    verts = [(cx + r * math.cos(i / len(rs) * 2 * math.pi + 0.3),
              cy + r * math.sin(i / len(rs) * 2 * math.pi + 0.3)) for i, r in enumerate(rs)]
    faint_fill(im, verts)
    poly(d, verts)
    d.line([verts[0], verts[4]], fill=LINE, width=1)       # facet creases
    d.line([verts[2], verts[6]], fill=LINE, width=1)
    d.ellipse([40, 30, 52, 42], outline=LINE, width=1)     # craters
    d.ellipse([54, 54, 62, 62], outline=LINE, width=1)
    return im


def pickup_wireframe():
    im, d = blank()
    # an isometric cargo canister (a 3D box)
    top = [(48, 22), (70, 34), (48, 46), (26, 34)]
    faint_fill(im, [(26, 34), (70, 34), (70, 64), (26, 64)])
    poly(d, top)                                            # lid (diamond)
    for sx in (26, 48, 70):
        d.line([(sx, 34 if sx != 48 else 46), (sx, 64 if sx != 48 else 74)], fill=LINE, width=1)
    d.line([(26, 64), (48, 74), (70, 64)], fill=LINE, width=1)   # bottom edges
    d.ellipse([44, 30, 52, 38], outline=LINE, width=1)     # hazard hatch on lid
    return im


def planet_wireframe(ptype):
    im, d = blank()
    cx = cy = SIZE / 2
    r = 38
    # faint disk fill + rim
    n = 48
    circle = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
              for i in range(n)]
    faint_fill(im, circle)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=LINE, width=1)

    def band(yoff, squash=0.32):
        # a latitude arc clipped to the disk
        xs = np.linspace(-r, r, 40)
        pts = [(cx + x, cy + yoff + squash * yoff * math.sin(math.acos(max(-1, min(1, x / r)))))
               for x in xs if abs(x) <= r]
        # simpler: a chord-following arc
        ys = [cy + yoff - 0.18 * yoff * (1 - (x / r) ** 2) for x in xs]
        seg = [(cx + x, y) for x, y in zip(xs, ys) if (x ** 2 + (y - cy) ** 2) <= r * r]
        if len(seg) > 1:
            d.line(seg, fill=LINE, width=1)

    if ptype in ("gas_giant", "ice_giant", "cloud"):
        for yoff in (-20, -7, 7, 20):
            band(yoff)
    elif ptype == "habitable":
        band(0)                                            # equator
        for cxy, rr in [((40, 38), 8), ((58, 52), 9), ((44, 60), 6)]:
            d.ellipse([cxy[0] - rr, cxy[1] - rr, cxy[0] + rr, cxy[1] + rr], outline=LINE, width=1)
    elif ptype == "desert":
        for yoff in (-14, 2, 16):
            band(yoff)
    else:  # rocky, icy_dwarf → cratered
        for cxy, rr in [((40, 36), 7), ((60, 44), 9), ((46, 58), 6), ((58, 62), 5)]:
            d.ellipse([cxy[0] - rr, cxy[1] - rr, cxy[0] + rr, cxy[1] + rr], outline=LINE, width=1)
    return im




# ── outfitter items: procedural schematics per weapon family ─────────────────
def _barrel(d, x0, y, ln, bore=4):
    d.line([(x0, y - bore), (x0 + ln, y - bore)], fill=LINE, width=1)
    d.line([(x0, y + bore), (x0 + ln, y + bore)], fill=LINE, width=1)
    d.line([(x0 + ln, y - bore), (x0 + ln, y + bore)], fill=LINE, width=1)


def item_gun(energy=False):
    im, d = blank()
    body = [(18, 44), (52, 44), (52, 60), (30, 60), (30, 52), (18, 52)]
    faint_fill(im, body)
    poly(d, body)
    _barrel(d, 52, 52, 26)
    if energy:                                             # emitter crystal
        poly(d, [(78, 46), (86, 52), (78, 58), (74, 52)])
        d.ellipse([60, 48, 68, 56], outline=LINE, width=1)  # focus ring
    else:
        d.line([(60, 44), (60, 60)], fill=LINE, width=1)    # breech seam
    d.ellipse([34, 48, 42, 56], outline=LINE, width=1)      # receiver detail
    return im


def item_turret():
    im, d = blank()
    d.ellipse([26, 46, 70, 74], outline=LINE, width=1)      # base ring
    dome = [(32, 56), (38, 40), (58, 40), (64, 56)]
    faint_fill(im, dome + [(64, 62), (32, 62)])
    poly(d, dome + [(64, 62), (32, 62)])
    _barrel(d, 56, 46, 24, bore=3)
    _barrel(d, 54, 36, 18, bore=2)                          # second gun
    d.arc([36, 50, 60, 66], 200, 340, fill=LINE, width=1)   # traverse track
    return im


def item_missile():
    im, d = blank()
    body = [(26, 48), (66, 48), (66, 58), (26, 58)]
    faint_fill(im, body)
    poly(d, body)
    poly(d, [(66, 48), (80, 53), (66, 58)])                 # nose cone
    poly(d, [(26, 48), (18, 40), (26, 52)])                 # upper fin
    poly(d, [(26, 58), (18, 66), (26, 54)])                 # lower fin
    d.line([(38, 48), (38, 58)], fill=LINE, width=1)        # stage seam
    d.line([(10, 53), (22, 53)], fill=LINE, width=1)        # exhaust dashes
    return im


def item_mine():
    im, d = blank()
    d.ellipse([34, 34, 62, 62], outline=LINE, width=1)
    faint_fill(im, [(38, 38), (58, 38), (58, 58), (38, 58)])
    for a in range(8):                                      # contact spikes
        ang = a / 8 * 2 * math.pi
        cx, cy, r0, r1 = 48, 48, 14, 22
        d.line([(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)),
                (cx + r1 * math.cos(ang), cy + r1 * math.sin(ang))], fill=LINE, width=1)
    d.ellipse([44, 44, 52, 52], outline=LINE, width=1)      # core light
    return im


def item_bay():
    im, d = blank()
    frame = [(20, 32), (76, 32), (76, 68), (20, 68)]
    faint_fill(im, frame)
    poly(d, frame)                                          # hangar box
    d.line([(20, 40), (76, 40)], fill=LINE, width=1)        # door track
    poly(d, [(40, 50), (56, 54), (40, 58), (44, 54)])       # tiny fighter glyph
    d.line([(24, 64), (72, 64)], fill=LINE, width=1)        # deck line
    return im


def item_rotary():
    im, d = blank()
    body = [(20, 44), (46, 44), (46, 62), (20, 62)]
    faint_fill(im, body)
    poly(d, body)
    for dy in (-5, 0, 5):                                   # barrel cluster
        d.line([(46, 53 + dy), (78, 53 + dy)], fill=LINE, width=1)
    d.line([(78, 47), (78, 59)], fill=LINE, width=1)
    d.ellipse([26, 48, 38, 60], outline=LINE, width=1)      # drum
    return im


def item_heavy():
    im, d = blank()
    mount = [(18, 42), (44, 42), (44, 64), (18, 64)]
    faint_fill(im, mount)
    poly(d, mount)
    _barrel(d, 44, 50, 36, bore=6)                          # massive bore
    d.line([(56, 44), (56, 56)], fill=LINE, width=1)        # reinforcing bands
    d.line([(66, 44), (66, 56)], fill=LINE, width=1)
    return im


def item_pod():
    im, d = blank()
    d.ellipse([34, 30, 62, 70], outline=LINE, width=1)      # canister
    faint_fill(im, [(38, 34), (58, 34), (58, 66), (38, 66)])
    d.line([(34, 44), (62, 44)], fill=LINE, width=1)
    d.line([(34, 56), (62, 56)], fill=LINE, width=1)
    for x, y in ((70, 36), (74, 50), (70, 64)):             # ejected decoys
        d.ellipse([x, y, x + 4, y + 4], outline=LINE, width=1)
    return im


ITEM_FAMILIES = (
    ("turret", item_turret),
    ("missile", item_missile), ("javelin", item_missile), ("seeker", item_missile),
    ("goose", item_missile),
    ("mine", item_mine), ("censer", item_mine),
    ("bay", item_bay),
    ("chaingun", item_rotary), ("flak", item_rotary),
    ("mass_driver", item_heavy), ("siege", item_heavy),
    ("lure", item_pod), ("chaff", item_pod), ("flare", item_pod),
)


def item_wireframe(key):
    for tag, fn in ITEM_FAMILIES:
        if tag in key:
            return fn()
    energy = any(t in key for t in ("laser", "beam", "lance", "proton", "relic",
                                    "exotic", "precursor"))
    return item_gun(energy=energy)


def weapon_keys():
    """Top-level keys from weapons.yaml (regex — the file uses custom !tags)."""
    import re
    keys = []
    for line in open(os.path.join(ROOT, "assets", "weapons.yaml")):
        m = re.match(r"^([a-z0-9_]+):", line)
        if m:
            keys.append(m.group(1))
    return keys


def main():
    for key in weapon_keys():
        item_wireframe(key).save(os.path.join(OUT, f"item_{key}.png"))
    print(f"item wireframes: {len(weapon_keys())}")
    ships = list(yaml.safe_load(open(os.path.join(ROOT, "assets", "ships.yaml"))).keys())
    n = 0
    for s in ships:
        atlas = os.path.join(ROOT, "assets", "sprites", "ships", "atlas", f"{s}.png")
        if os.path.exists(atlas):
            ship_wireframe(atlas).save(os.path.join(OUT, f"{s}.png"))
            n += 1
    asteroid_wireframe().save(os.path.join(OUT, "asteroid.png"))
    pickup_wireframe().save(os.path.join(OUT, "pickup.png"))
    ptypes = ["habitable", "rocky", "cloud", "desert", "gas_giant", "ice_giant", "icy_dwarf"]
    for pt in ptypes:
        planet_wireframe(pt).save(os.path.join(OUT, f"planet_{pt}.png"))
    print(f"wrote {n} ship + 2 generic + {len(ptypes)} planet wireframes → {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
