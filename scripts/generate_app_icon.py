#!/usr/bin/env python3
"""
generate_app_icon.py — Build scripts/AppIcon.icns for the macOS bundle.

Renders a habitable planet using the planet sprite generator, redraws the
fighter using the same procedural style as the ship sprite generator, and
composites them onto a 1024x1024 black canvas: planet in the bottom-left
quadrant (50% of the icon width), fighter overlaid on its upper-right rim
flying away at an angle. The 1024 master is then expanded into a macOS
.iconset and packaged via iconutil.

Environment
-----------
  conda activate avian-sprites      (see scripts/sprites_environment.yml)

Run
---
  python3 scripts/generate_app_icon.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_planet_sprites as planet_gen


# ---------------------------------------------------------------------------
# Fighter drawing (scaled copy of generate_sprites.draw_fighter, supersampled
# 4x then downsampled with LANCZOS for clean antialiased edges).
# ---------------------------------------------------------------------------

def draw_fighter(size: int) -> Image.Image:
    SUPER = 4
    S = size * SUPER
    s = S / 28.0  # original sprite is 28x28; scale all coordinates by s
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = S / 2

    hull = [(cx, 2 * s), (cx + 3 * s, 14 * s), (cx, 18 * s), (cx - 3 * s, 14 * s)]
    d.polygon(hull, fill=(180, 200, 230, 255))

    nose = [(cx, 2 * s), (cx + 2 * s, 8 * s), (cx - 2 * s, 8 * s)]
    d.polygon(nose, fill=(220, 235, 255, 255))

    lwing = [(cx - 3 * s, 10 * s), (cx - 11 * s, 22 * s), (cx - 3 * s, 18 * s)]
    rwing = [(cx + 3 * s, 10 * s), (cx + 11 * s, 22 * s), (cx + 3 * s, 18 * s)]
    d.polygon(lwing, fill=(130, 155, 195, 255))
    d.polygon(rwing, fill=(130, 155, 195, 255))

    d.rectangle([cx - 3 * s, 18 * s, cx - 1 * s, 22 * s], fill=(80, 170, 255, 190))
    d.rectangle([cx + 1 * s, 18 * s, cx + 3 * s, 22 * s], fill=(80, 170, 255, 190))
    d.rectangle([cx - 2 * s, 19 * s, cx + 2 * s, 21 * s], fill=(200, 230, 255, 220))

    return img.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

ICON_SIZE = 1024


def crop_to_alpha(img: Image.Image) -> Image.Image:
    bbox = img.getchannel("A").getbbox()
    return img.crop(bbox) if bbox else img


def render_planet_image(diameter: int) -> Image.Image:
    """Render a habitable planet and resize tightly to ``diameter``."""
    planet = planet_gen.habitable_planet(
        seed=20260508,
        ocean_fraction=0.55,
        cloud_cover=0.40,
        radius=2.0,
        subdiv=256,
        name="icon_planet",
    )
    raw = planet_gen.render_sprite(planet, size=max(diameter * 2, 800))
    return crop_to_alpha(raw).resize((diameter, diameter), Image.LANCZOS)


def build_icon() -> Image.Image:
    canvas = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 255))

    # Planet — 512 px diameter (50% of the icon), tucked into bottom-left.
    planet_diam = 512
    planet_img = render_planet_image(planet_diam)
    canvas.alpha_composite(planet_img, (0, ICON_SIZE - planet_diam))

    # Fighter — supersampled draw, then rotated so the nose points to the
    # upper-right (CW from straight-up = negative PIL angle), centred on the
    # planet's upper-right rim so it reads as flying away from the planet.
    fighter = draw_fighter(460)
    fighter_rot = fighter.rotate(-60, resample=Image.BICUBIC, expand=True)

    fw, fh = fighter_rot.size
    cx, cy = 600, 460  # off the planet's upper-right rim
    canvas.alpha_composite(fighter_rot, (cx - fw // 2, cy - fh // 2))

    return canvas


# ---------------------------------------------------------------------------
# .icns packaging via macOS iconutil
# ---------------------------------------------------------------------------

ICONSET_SIZES = [
    ("icon_16x16.png",         16),
    ("icon_16x16@2x.png",      32),
    ("icon_32x32.png",         32),
    ("icon_32x32@2x.png",      64),
    ("icon_128x128.png",      128),
    ("icon_128x128@2x.png",   256),
    ("icon_256x256.png",      256),
    ("icon_256x256@2x.png",   512),
    ("icon_512x512.png",      512),
    ("icon_512x512@2x.png",  1024),
]


def write_icns(master: Image.Image, out_icns: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for filename, sz in ICONSET_SIZES:
            master.resize((sz, sz), Image.LANCZOS).save(iconset / filename)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(out_icns)],
            check=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root     = Path(__file__).resolve().parent.parent
    out_png  = root / "scripts" / "AppIcon.png"
    out_icns = root / "scripts" / "AppIcon.icns"

    print("rendering 1024x1024 icon master...")
    icon = build_icon()
    icon.save(out_png)
    print(f"  saved {out_png.relative_to(root)}")

    print("packaging .icns via iconutil...")
    write_icns(icon, out_icns)
    print(f"  saved {out_icns.relative_to(root)}")


if __name__ == "__main__":
    main()
