#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pillow",
#     "pyyaml",
# ]
# ///
"""Generate faction crests and their placements (assets/sprites/factions/).

For every faction in assets/factions.yaml, four sprites:
  crest_<f>.png   32x32  — the emblem on a shield field (UI use)
  flag_<f>.png    56x36  — waving cloth for the garrison flagpole
  banner_<f>.png  28x44  — hanging wall banner (garrison interior prop)
  pad_<f>.png     96x96  — worn painted ring decal for the landing pad

Field colors derive from the faction's palette color (single source of
truth), motifs are drawn in off-white so they read at 16 px. Independent
is deliberately crestless: plain undyed cloth, no pad/banner marks.

Run:  uv run scripts/gen_crests.py
"""
import math
import os
import random

import yaml
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "assets", "sprites", "factions")
SS = 4  # supersample factor

MOTIF = (238, 232, 216, 255)  # off-white paint/thread
GOLD = (232, 204, 120, 255)


def shade(c, k, a=255):
    return (min(int(c[0] * k), 255), min(int(c[1] * k), 255), min(int(c[2] * k), 255), a)


# ── Motifs ────────────────────────────────────────────────────────────────
# Each draws centered on a transparent S×S canvas (S already supersampled).


def star(d, cx, cy, r, color, points=5, rot=-math.pi / 2):
    pts = []
    for i in range(points * 2):
        rad = r if i % 2 == 0 else r * 0.42
        a = rot + i * math.pi / points
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    d.polygon(pts, fill=color)


def motif_federation(d, s):
    c = s / 2
    # Two crossing orbital rings around a central star.
    w = max(2, s // 24)
    d.ellipse([c - s * 0.42, c - s * 0.20, c + s * 0.42, c + s * 0.20], outline=MOTIF, width=w)
    d.ellipse([c - s * 0.20, c - s * 0.42, c + s * 0.20, c + s * 0.42], outline=MOTIF, width=w)
    star(d, c, c, s * 0.22, MOTIF)


def motif_rebel(d, s):
    c = s / 2
    # Torn diagonal slash with a comet rising across it.
    w = s * 0.13
    d.polygon(
        [(s * 0.08, s * 0.86), (s * 0.08 + w, s * 0.86), (s * 0.92, s * 0.14 + w), (s * 0.92 - w, s * 0.14)],
        fill=shade(MOTIF, 0.85),
    )
    # Comet: head + tapering tail crossing the slash the other way.
    d.polygon([(s * 0.16, s * 0.30), (s * 0.62, s * 0.52), (s * 0.60, s * 0.62)], fill=MOTIF)
    d.ellipse([s * 0.56, s * 0.48, s * 0.74, s * 0.66], fill=MOTIF)


def motif_freefrontier(d, s):
    c = s / 2
    hy = s * 0.62  # horizon
    d.rectangle([s * 0.10, hy, s * 0.90, hy + s * 0.05], fill=MOTIF)
    # Half sun below the horizon line's top edge.
    d.pieslice([c - s * 0.24, hy - s * 0.24, c + s * 0.24, hy + s * 0.24], 180, 360, fill=GOLD)
    for i in range(5):  # rays
        a = math.pi + (i + 0.5) * math.pi / 5
        x0 = c + s * 0.28 * math.cos(a)
        y0 = hy + s * 0.28 * math.sin(a)
        x1 = c + s * 0.38 * math.cos(a)
        y1 = hy + s * 0.38 * math.sin(a)
        d.line([x0, y0, x1, y1], fill=GOLD, width=max(2, s // 20))
    star(d, c, s * 0.22, s * 0.10, MOTIF)


def motif_helios(d, s):
    c = s / 2
    d.ellipse([c - s * 0.17, c - s * 0.17, c + s * 0.17, c + s * 0.17], fill=MOTIF)
    for i in range(8):  # circuit-tick rays
        a = i * math.pi / 4
        x0 = c + s * 0.26 * math.cos(a)
        y0 = c + s * 0.26 * math.sin(a)
        x1 = c + s * 0.40 * math.cos(a)
        y1 = c + s * 0.40 * math.sin(a)
        d.line([x0, y0, x1, y1], fill=MOTIF, width=max(2, s // 16))


def motif_bastion(d, s):
    c = s / 2
    # Crenellated tower over an anvil bar.
    tw = s * 0.34
    d.rectangle([c - tw / 2, s * 0.26, c + tw / 2, s * 0.58], fill=MOTIF)
    for i in range(3):  # battlements
        x = c - tw / 2 + i * tw / 2.6
        d.rectangle([x, s * 0.16, x + tw / 4.2, s * 0.28], fill=MOTIF)
    d.rectangle([s * 0.20, s * 0.64, s * 0.80, s * 0.74], fill=MOTIF)  # anvil face
    d.rectangle([c - s * 0.10, s * 0.74, c + s * 0.10, s * 0.84], fill=MOTIF)  # anvil waist
    # Arrow-slit window in the faction's field color (punched through).
    d.rectangle([c - s * 0.03, s * 0.34, c + s * 0.03, s * 0.48], fill=(0, 0, 0, 0))


def motif_order(d, s):
    c = s / 2
    w = max(2, s // 24)
    d.ellipse([c - s * 0.42, c - s * 0.42, c + s * 0.42, c + s * 0.42], outline=MOTIF, width=w)
    d.ellipse([c - s * 0.34, c - s * 0.34, c + s * 0.34, c + s * 0.34], outline=MOTIF, width=w)
    # Bell: dome + flared skirt + clapper.
    d.pieslice([c - s * 0.18, c - s * 0.26, c + s * 0.18, c + s * 0.10], 180, 360, fill=MOTIF)
    d.polygon(
        [(c - s * 0.18, c - s * 0.08), (c + s * 0.18, c - s * 0.08), (c + s * 0.24, c + s * 0.14), (c - s * 0.24, c + s * 0.14)],
        fill=MOTIF,
    )
    d.ellipse([c - s * 0.045, c + s * 0.16, c + s * 0.045, c + s * 0.25], fill=MOTIF)


def motif_precursor(d, s):
    c = s / 2
    # Incomplete angular triskelion: two L-arms of three, one missing.
    w = max(4, s // 8)
    for rot in (0.0, 2.0 * math.pi / 3.0):  # third arm deliberately absent
        a1 = rot - math.pi / 2
        x1, y1 = c + s * 0.32 * math.cos(a1), c + s * 0.32 * math.sin(a1)
        d.line([c, c, x1, y1], fill=MOTIF, width=w)
        a2 = a1 + math.pi / 2.2
        x2, y2 = x1 + s * 0.24 * math.cos(a2), y1 + s * 0.24 * math.sin(a2)
        d.line([x1, y1, x2, y2], fill=MOTIF, width=w)
    d.ellipse([c - s * 0.09, c - s * 0.09, c + s * 0.09, c + s * 0.09], fill=MOTIF)


def motif_pirate(d, s):
    c = s / 2
    # Broken hull halves beneath a crescent blade.
    d.polygon([(s * 0.14, s * 0.62), (s * 0.46, s * 0.56), (s * 0.44, s * 0.72), (s * 0.20, s * 0.74)], fill=MOTIF)
    d.polygon([(s * 0.54, s * 0.54), (s * 0.86, s * 0.58), (s * 0.82, s * 0.72), (s * 0.56, s * 0.70)], fill=MOTIF)
    # Crescent blade: a disc with an offset disc punched out.
    d.ellipse([c - s * 0.24, s * 0.06, c + s * 0.24, s * 0.54], fill=MOTIF)
    d.ellipse([c - s * 0.22, s * 0.00, c + s * 0.22, s * 0.42], fill=(0, 0, 0, 0))


def motif_merchant(d, s):
    c = s / 2
    w = max(2, s // 20)
    d.line([c, s * 0.18, c, s * 0.62], fill=MOTIF, width=w)  # column
    d.line([s * 0.22, s * 0.26, s * 0.78, s * 0.26], fill=MOTIF, width=w)  # beam
    for x in (s * 0.22, s * 0.78):  # chains + coin pans
        d.line([x, s * 0.26, x, s * 0.46], fill=MOTIF, width=max(1, w // 2))
        d.ellipse([x - s * 0.10, s * 0.44, x + s * 0.10, s * 0.60], fill=GOLD)
    d.rectangle([c - s * 0.14, s * 0.62, c + s * 0.14, s * 0.68], fill=MOTIF)  # base


MOTIFS = {
    "Federation": motif_federation,
    "Rebel": motif_rebel,
    "FreeFrontier": motif_freefrontier,
    "Helios": motif_helios,
    "Bastion": motif_bastion,
    "Order": motif_order,
    "Precursor": motif_precursor,
    "Pirate": motif_pirate,
    "Merchant": motif_merchant,
    # Independent: crestless on purpose — plain undyed cloth.
}


def motif_layer(faction, px):
    """The faction's motif on a transparent px×px canvas (supersampled)."""
    s = px * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    if faction in MOTIFS:
        MOTIFS[faction](ImageDraw.Draw(img), s)
    return img.resize((px, px), Image.LANCZOS)


# ── Variants ──────────────────────────────────────────────────────────────


def make_crest(faction, color):
    """32x32 shield emblem."""
    s = 32 * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    field = shade(color, 0.62)
    # Shield: rounded top, tapered point.
    d.polygon(
        [(s * 0.10, s * 0.08), (s * 0.90, s * 0.08), (s * 0.90, s * 0.55),
         (s * 0.50, s * 0.95), (s * 0.10, s * 0.55)],
        fill=field, outline=shade(color, 1.25), width=max(2, s // 24),
    )
    img = img.resize((32, 32), Image.LANCZOS)
    m = motif_layer(faction, 22)
    img.alpha_composite(m, (5, 3))
    return img


def make_flag(faction, color):
    """56x36 waving cloth, matches the baked flagpole cloth size."""
    w, h = 56 * SS, 36 * SS
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    field = shade(color, 0.72)
    # Cloth with a gently curved fly edge.
    d.polygon(
        [(0, 0), (w - 1, h * 0.06), (w * 0.94, h * 0.5), (w - 1, h * 0.94), (0, h - 1)],
        fill=field,
    )
    d.rectangle([0, 0, w * 0.045, h], fill=shade(color, 0.45))  # hoist shadow
    img = img.resize((56, 36), Image.LANCZOS)
    # Motif sits on the cloth center.
    m = motif_layer(faction, 26)
    img.alpha_composite(m, (16, 5))
    # Wave shading bands (post-motif so the cloth ripples over the crest).
    band = Image.new("RGBA", (56, 36), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    for x in range(56):
        k = math.sin(x / 56.0 * 2.5 * math.pi)
        if k > 0.35:
            bd.line([x, 0, x, 35], fill=(255, 255, 255, 26))
        elif k < -0.35:
            bd.line([x, 0, x, 35], fill=(0, 0, 0, 34))
    img.alpha_composite(band)
    # Clip bands to cloth alpha.
    return img


def make_banner(faction, color):
    """28x44 hanging wall banner with rod and swallowtail cut."""
    w, h = 28 * SS, 44 * SS
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    field = shade(color, 0.62)
    d.polygon(
        [(w * 0.08, h * 0.06), (w * 0.92, h * 0.06), (w * 0.92, h * 0.86),
         (w * 0.5, h * 0.97), (w * 0.08, h * 0.86)],
        fill=field,
    )
    d.rectangle([w * 0.08, h * 0.06, w * 0.92, h * 0.115], fill=shade(color, 1.15))  # top hem
    d.rectangle([0, 0, w, h * 0.05], fill=(96, 78, 56, 255))  # rod
    d.polygon(  # swallowtail notch
        [(w * 0.5, h * 0.80), (w * 0.62, h * 0.97), (w * 0.38, h * 0.97)],
        fill=(0, 0, 0, 0),
    )
    img = img.resize((28, 44), Image.LANCZOS)
    m = motif_layer(faction, 20)
    img.alpha_composite(m, (4, 9))
    return img


def make_pad(faction, color):
    """96x96 worn painted ring decal for the pad center."""
    s = 96 * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ring = shade(color, 1.15, a=235)
    w = s // 14
    d.ellipse([s * 0.05, s * 0.05, s * 0.95, s * 0.95], outline=ring, width=w)
    d.ellipse([s * 0.14, s * 0.14, s * 0.86, s * 0.86], outline=(238, 232, 216, 200), width=max(2, w // 3))
    img = img.resize((96, 96), Image.LANCZOS)
    m = motif_layer(faction, 52)
    img.alpha_composite(m, (22, 22))
    # Worn paint: seeded speckle erosion + overall translucency.
    rnd = random.Random(hash(faction) & 0xFFFF)
    px = img.load()
    for _ in range(900):
        x, y = rnd.randrange(96), rnd.randrange(96)
        r, g, b, a = px[x, y]
        if a:
            px[x, y] = (r, g, b, max(0, a - rnd.randrange(120, 255)))
    alpha = img.getchannel("A").point(lambda a: int(a * 0.82))
    img.putalpha(alpha)
    return img


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(ROOT, "assets", "factions.yaml")) as f:
        factions = yaml.safe_load(f)
    for name, data in factions.items():
        color = tuple(data["color"])
        stem = name.lower()
        make_crest(name, color).save(os.path.join(OUT, f"crest_{stem}.png"))
        make_flag(name, color).save(os.path.join(OUT, f"flag_{stem}.png"))
        make_banner(name, color).save(os.path.join(OUT, f"banner_{stem}.png"))
        make_pad(name, color).save(os.path.join(OUT, f"pad_{stem}.png"))
        print(f"  {name}: crest/flag/banner/pad")
    print(f"Done → {os.path.relpath(OUT, ROOT)}")
