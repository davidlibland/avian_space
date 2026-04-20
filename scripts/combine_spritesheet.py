#!/usr/bin/env python3
"""Combine individual character sprites into RPG Maker-style sprite sheets.

Standard layout (3 columns x 4 rows):
    Column:  still  w1  w2
    Row 0:   down   ...
    Row 1:   left   ...
    Row 2:   right  ...
    Row 3:   up     ...

Usage:
    python tools/combine_spritesheet.py

Reads from:  assets/sprites/people/{character}/{direction}_{frame}.png
Writes to:   assets/sprites/people/{character}.png

The individual frame files are preserved (not deleted).
"""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)

# RPG Maker standard ordering
DIRECTIONS = ["down", "left", "right", "up"]
FRAMES = ["still", "w1", "w2"]

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "sprites" / "people"


def combine_character(char_dir: Path) -> bool:
    """Combine individual PNGs in char_dir into a single sprite sheet.

    Returns True on success, False if any frames are missing.
    """
    name = char_dir.name

    # Load all frames and verify they exist + have uniform size.
    tiles: list[list[Image.Image]] = []
    tile_w, tile_h = None, None

    for direction in DIRECTIONS:
        row = []
        for frame in FRAMES:
            path = char_dir / f"{direction}_{frame}.png"
            if not path.exists():
                print(f"  SKIP {name}: missing {path.name}")
                return False
            img = Image.open(path).convert("RGBA")
            if tile_w is None:
                tile_w, tile_h = img.size
            elif img.size != (tile_w, tile_h):
                print(
                    f"  SKIP {name}: {path.name} is {img.size}, "
                    f"expected {(tile_w, tile_h)}"
                )
                return False
            row.append(img)
        tiles.append(row)

    # Assemble the sheet.
    cols, rows = len(FRAMES), len(DIRECTIONS)
    sheet = Image.new("RGBA", (cols * tile_w, rows * tile_h), (0, 0, 0, 0))
    for r, row in enumerate(tiles):
        for c, img in enumerate(row):
            sheet.paste(img, (c * tile_w, r * tile_h))

    out_path = char_dir.parent / f"{name}.png"
    sheet.save(out_path, "PNG")
    print(f"  {name}.png  ({cols * tile_w}x{rows * tile_h}, "
          f"{cols}x{rows} grid of {tile_w}x{tile_h} tiles)")
    return True


def main():
    if not ASSETS_DIR.is_dir():
        print(f"Error: {ASSETS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    chars = sorted(
        p for p in ASSETS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not chars:
        print("No character directories found.")
        sys.exit(0)

    print(f"Combining sprite sheets in {ASSETS_DIR}:")
    ok = 0
    for char_dir in chars:
        if combine_character(char_dir):
            ok += 1

    print(f"\nDone: {ok}/{len(chars)} characters combined.")


if __name__ == "__main__":
    main()
