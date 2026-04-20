"""
buildings_scifi.py  —  Sci-fi building styles for each planet biome
====================================================================
Drop-in replacement for the BUILDING_STYLES list in buildings.py.
Imports the full atlas/template machinery from buildings.py unchanged;
only the visual layer (palettes + texture functions) is new here.

Styles produced:
  colony      — garden world   — terraforming prefab colony modules
  cryo        — ice world      — cryo-research outpost, frosted composites
  extraction  — rocky world    — mining/extraction facility, rusted metal
  station     — city world     — space station interior, glowing conduit walls

Usage (standalone):
    python buildings_scifi.py --tile-size 32 --out-dir output_scifi

Or replace buildings.BUILDING_STYLES before calling tilegen_full.main():
    import buildings, buildings_scifi
    buildings.BUILDING_STYLES = buildings_scifi.BUILDING_STYLES
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import buildings as _b  # so we can monkey-patch texture functions

# Re-use all atlas + template machinery from buildings.py
from buildings import (
    EXT_COLS,
    BuildingStyle,
    ColourPalette,
    _fill,
    make_large_building,
    make_medium_house,
    make_small_house,
    make_station_room,
    make_tall_building,
    make_tower,
    make_wide_house,
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  New texture drawing functions                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _draw_composite_panel(
    img: Image.Image,
    colour: tuple[int, int, int],
    edge: tuple[int, int, int],
    size: int,
    panel_w: int = 10,
    panel_h: int = 10,
    seed: int = 0,
) -> Image.Image:
    """Rectangular composite panels with recessed edges and rivet dots."""
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    for row_y in range(0, size, panel_h):
        for col_x in range(0, size, panel_w):
            jit = int(rng.integers(-8, 9))
            c = tuple(max(0, min(255, v + jit)) for v in colour)
            # Panel fill
            draw.rectangle(
                [col_x + 1, row_y + 1, col_x + panel_w - 2, row_y + panel_h - 2],
                fill=c,
            )
            # Recessed border line
            draw.rectangle(
                [col_x, row_y, col_x + panel_w - 1, row_y + panel_h - 1],
                outline=edge,
                width=1,
            )
            # Corner rivets
            for rx, ry in [(col_x + 2, row_y + 2), (col_x + panel_w - 3, row_y + 2)]:
                if 0 <= rx < size and 0 <= ry < size:
                    bright = tuple(min(255, v + 40) for v in c)
                    draw.ellipse([rx - 1, ry - 1, rx + 1, ry + 1], fill=bright)
    return img


def _draw_geodesic_roof(
    img: Image.Image,
    colour: tuple[int, int, int],
    edge: tuple[int, int, int],
    size: int,
) -> Image.Image:
    """Triangular geodesic facets for sci-fi roofing."""
    draw = ImageDraw.Draw(img)
    step = max(6, size // 4)
    # Build a grid of triangles alternating fill direction
    for row in range(0, size, step):
        for col in range(0, size, step * 2):
            # Up-triangle
            pts_up = [col, row + step, col + step, row, col + step * 2, row + step]
            bright = tuple(min(255, v + 20) for v in colour)
            draw.polygon(pts_up, fill=bright, outline=edge)
            # Down-triangle
            pts_dn = [col, row, col + step, row + step, col + step * 2, row]
            draw.polygon(pts_dn, fill=colour, outline=edge)
    return img


def _draw_corrugated_rust(
    img: Image.Image,
    colour: tuple[int, int, int],
    dark: tuple[int, int, int],
    rust: tuple[int, int, int],
    size: int,
    seed: int = 0,
) -> Image.Image:
    """Corrugated metal with rust streaks for the rocky mining world."""
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    # Corrugation ridges (horizontal)
    ridge_h = max(3, size // 6)
    for y in range(0, size, ridge_h):
        hi = tuple(min(255, v + 25) for v in colour)
        lo = tuple(max(0, v - 20) for v in colour)
        draw.line([(0, y), (size, y)], fill=hi, width=1)
        draw.line([(0, y + 1), (size, y + 1)], fill=colour, width=1)
        draw.line([(0, y + 2), (size, y + 2)], fill=lo, width=1)
    # Bolt heads
    for _ in range(4):
        bx = int(rng.integers(2, size - 2))
        by = int(rng.integers(2, size - 2))
        bright = tuple(min(255, v + 45) for v in colour)
        draw.ellipse([bx - 2, by - 2, bx + 2, by + 2], fill=bright, outline=dark)
    # Rust streaks (vertical drips)
    for _ in range(rng.integers(2, 5)):
        rx = int(rng.integers(0, size))
        ry = int(rng.integers(0, size // 2))
        rlen = int(rng.integers(4, 12))
        for i in range(rlen):
            alpha = max(0, 200 - i * 18)
            px, py = rx + int(rng.integers(-1, 2)), ry + i
            if 0 <= px < size and 0 <= py < size:
                arr = np.array(img)
                arr[py, px] = [
                    int(arr[py, px, 0] * 0.4 + rust[0] * 0.6),
                    int(arr[py, px, 1] * 0.6 + rust[1] * 0.4),
                    int(arr[py, px, 2] * 0.8 + rust[2] * 0.2),
                ]
                img = Image.fromarray(arr)
                draw = ImageDraw.Draw(img)
    return img


def _draw_hex_roof(
    img: Image.Image,
    colour: tuple[int, int, int],
    edge: tuple[int, int, int],
    size: int,
) -> Image.Image:
    """Hexagonal heat-sink / armour-plate pattern."""
    import math

    draw = ImageDraw.Draw(img)
    r = max(4, size // 5)
    h = r * math.sqrt(3)
    cols_n = int(size / (r * 1.5)) + 2
    rows_n = int(size / h) + 2
    for row in range(-1, rows_n):
        for col in range(-1, cols_n):
            cx = col * r * 3 + (r * 1.5 if row % 2 else 0)
            cy = row * h
            pts = [
                (
                    cx + r * math.cos(math.radians(60 * i)),
                    cy + r * math.sin(math.radians(60 * i)),
                )
                for i in range(6)
            ]
            jit = (col * 7 + row * 13) % 15 - 7
            c = tuple(max(0, min(255, v + jit)) for v in colour)
            draw.polygon(pts, fill=c, outline=edge)
    return img


def _draw_emissive_strip(
    img: Image.Image,
    colour: tuple[int, int, int],
    accent: tuple[int, int, int],
    size: int,
    edge: str = "bottom",  # "bottom" | "top" | "left" | "right"
    width: int = 3,
) -> Image.Image:
    """Glowing LED strip along one edge — the key sci-fi genre marker."""
    draw = ImageDraw.Draw(img)
    if edge == "bottom":
        y0 = size - width
        draw.rectangle([0, y0, size, size], fill=accent)
        # Glow bloom: two fainter rows above
        bloom = tuple(
            min(255, int(v * 0.55 + accent_v * 0.45))
            for v, accent_v in zip(colour, accent)
        )
        draw.line([(0, y0 - 1), (size, y0 - 1)], fill=bloom, width=1)
        draw2 = tuple(
            min(255, int(v * 0.75 + accent_v * 0.25))
            for v, accent_v in zip(colour, accent)
        )
        draw.line([(0, y0 - 2), (size, y0 - 2)], fill=draw2, width=1)
    elif edge == "top":
        draw.rectangle([0, 0, size, width], fill=accent)
        bloom = tuple(
            min(255, int(v * 0.55 + a * 0.45)) for v, a in zip(colour, accent)
        )
        draw.line([(0, width + 1), (size, width + 1)], fill=bloom, width=1)
    elif edge == "left":
        draw.rectangle([0, 0, width, size], fill=accent)
    elif edge == "right":
        draw.rectangle([size - width, 0, size, size], fill=accent)
    return img


def _draw_conduit_wall(
    img: Image.Image,
    colour: tuple[int, int, int],
    edge: tuple[int, int, int],
    accent: tuple[int, int, int],
    size: int,
    seed: int = 0,
) -> Image.Image:
    """Station wall with routed conduit channels and emissive node dots."""
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    # Base panel grid
    _draw_composite_panel(
        img, colour, edge, size, panel_w=size // 2, panel_h=size, seed=seed
    )
    draw = ImageDraw.Draw(img)
    # Conduit channel (vertical stripe)
    cx = int(rng.integers(size // 4, 3 * size // 4))
    channel_c = tuple(max(0, v - 30) for v in colour)
    draw.line([(cx, 0), (cx, size)], fill=channel_c, width=2)
    # Emissive node every ~8px along conduit
    for y in range(4, size, 8):
        draw.ellipse([cx - 2, y - 2, cx + 2, y + 2], fill=accent)
    return img


def _draw_warning_stripe(
    img: Image.Image,
    colour: tuple[int, int, int],
    accent: tuple[int, int, int],
    size: int,
    stripe_w: int = 4,
) -> Image.Image:
    """Yellow/black hazard stripe for the base row of mining facilities."""
    draw = ImageDraw.Draw(img)
    black = (20, 20, 20)
    for x in range(0, size * 2, stripe_w * 2):
        pts = [x, size, x + stripe_w, 0, x + stripe_w * 2, 0, x + stripe_w, size]
        draw.polygon(pts, fill=accent)
        pts2 = [
            x + stripe_w,
            size,
            x + stripe_w * 2,
            0,
            x + stripe_w * 3,
            0,
            x + stripe_w * 2,
            size,
        ]
        draw.polygon(pts2, fill=black)
    return img


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Patched texture dispatch                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# We monkey-patch _make_wall_base and _make_roof_base so that the existing
# build_exterior_atlas / build_interior_atlas functions pick up the new types
# transparently.


def _scifi_wall_base(style: BuildingStyle, size: int, seed: int = 0) -> np.ndarray:
    p = style.palette
    arr = _fill(size, p.wall_fill, style.roughness * 0.4, seed)
    img = Image.fromarray(arr)

    wt = style.wall_texture
    if wt == "composite":
        panel_w = max(6, size // 3)
        panel_h = max(6, size // 3)
        img = _draw_composite_panel(
            img,
            p.wall_fill,
            p.wall_edge,
            size,
            panel_w=panel_w,
            panel_h=panel_h,
            seed=seed,
        )
    elif wt == "corrugated_rust":
        rust = (160, 80, 30)
        img = _draw_corrugated_rust(img, p.wall_fill, p.wall_dark, rust, size, seed)
    elif wt == "conduit":
        img = _draw_conduit_wall(
            img,
            p.wall_fill,
            p.wall_edge,
            p.window_fill,  # window_fill doubles as accent
            size,
            seed,
        )
    elif wt == "cryo_panel":
        img = _draw_composite_panel(
            img,
            p.wall_fill,
            p.wall_edge,
            size,
            panel_w=max(8, size // 2),
            panel_h=max(8, size // 2),
            seed=seed,
        )
        # Add frost vignette edges
        arr2 = np.array(img, dtype=np.float32)
        y_idx, x_idx = np.mgrid[0:size, 0:size]
        dist = np.minimum(
            np.minimum(x_idx, size - 1 - x_idx),
            np.minimum(y_idx, size - 1 - y_idx),
        ).astype(np.float32) / (size * 0.25)
        frost = np.array([220, 235, 248], dtype=np.float32)
        w = np.clip(1.0 - dist, 0, 1)[:, :, np.newaxis]
        arr2 = arr2 * (1 - w * 0.35) + frost * (w * 0.35)
        img = Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8))
    else:
        # Fallback to original
        return _b._orig_wall_base(style, size, seed)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return np.array(img)


def _scifi_roof_base(style: BuildingStyle, size: int, seed: int = 0) -> np.ndarray:
    p = style.palette
    arr = _fill(size, p.roof_fill, style.roughness * 0.5, seed)
    img = Image.fromarray(arr)

    rt = style.roof_texture
    if rt == "geodesic":
        img = _draw_geodesic_roof(img, p.roof_fill, p.roof_edge, size)
    elif rt == "hex":
        img = _draw_hex_roof(img, p.roof_fill, p.roof_edge, size)
    elif rt == "flat_sci":
        draw = ImageDraw.Draw(img)
        # Flat roof with antenna / vent details
        draw.rectangle([2, 2, size - 3, size - 3], outline=p.roof_edge, width=1)
        # Vent slots
        slot_w = max(2, size // 8)
        for sx in range(4, size - 4, slot_w * 2):
            draw.rectangle(
                [sx, size // 2 - 1, sx + slot_w - 1, size // 2 + 1], fill=p.roof_dark
            )
    elif rt == "corrugated_rust":
        rust = (160, 80, 30)
        img = _draw_corrugated_rust(img, p.roof_fill, p.roof_dark, rust, size, seed + 1)
    else:
        return _b._orig_roof_base(style, size, seed)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return np.array(img)


# Save originals before patching (so fallback path works)
_b._orig_wall_base = _b._make_wall_base
_b._orig_roof_base = _b._make_roof_base
_b._make_wall_base = _scifi_wall_base
_b._make_roof_base = _scifi_roof_base


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Patched base-row tile (emissive strips + warning stripes)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# We override the door/base rendering in build_exterior_atlas by subclassing
# the palette with extra fields and post-processing the base row.


def _post_process_base_row(
    atlas: Image.Image,
    style: BuildingStyle,
    size: int,
) -> None:
    """Add emissive strip or warning stripe to the base row tiles (row 4) in-place."""
    p = style.palette
    accent = getattr(p, "_emissive", None)
    stripe = getattr(p, "_hazard_stripe", False)
    if accent is None and not stripe:
        return

    # Door tile columns (DOOR_L=3, DOOR=4, DOOR_R=5) — only apply the
    # emissive strip, not the hazard stripe, so the door stays visible.
    door_cols = {3, 4, 5}
    for col in range(EXT_COLS):
        x0, y0 = col * size, 4 * size
        tile = atlas.crop((x0, y0, x0 + size, y0 + size)).convert("RGB")
        img = tile.copy()

        if stripe and col not in door_cols:
            yellow = (210, 180, 20)
            img = _draw_warning_stripe(img, p.wall_fill, yellow, size)
        if accent:
            img = _draw_emissive_strip(
                img, p.wall_fill, accent, size, edge="bottom", width=2
            )

        atlas.paste(img.convert("RGBA"), (x0, y0))


# Patch build_exterior_atlas to call post_process
_b._orig_build_exterior = _b.build_exterior_atlas


def _scifi_build_exterior(style: BuildingStyle, tile_size: int) -> Image.Image:
    atlas = _b._orig_build_exterior(style, tile_size)
    _post_process_base_row(atlas, style, tile_size)
    return atlas


_b.build_exterior_atlas = _scifi_build_exterior


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Sci-fi ColourPalette subclass with emissive fields                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class SciFiPalette(ColourPalette):
    _emissive: tuple[int, int, int] | None = None  # LED strip accent colour
    _hazard_stripe: bool = False  # yellow/black warning stripe


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Sci-fi building styles                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BUILDING_STYLES: list[BuildingStyle] = [
    # ── Colony module — garden world ───────────────────────────────────────
    # Prefab aluminium-composite panels, flat green-house roofs, amber LEDs
    BuildingStyle(
        name="colony",
        biome="garden",
        wall_texture="composite",
        roof_texture="geodesic",
        roughness=0.03,
        seed=10,
        palette=SciFiPalette(
            roof_fill=(82, 98, 72),
            roof_edge=(58, 72, 50),
            roof_dark=(38, 50, 30),
            wall_fill=(155, 168, 148),
            wall_edge=(112, 125, 105),
            wall_dark=(72, 82, 65),
            window_fill=(180, 230, 175),
            window_frame=(65, 78, 55),
            door_fill=(85, 95, 75),
            door_frame=(55, 65, 45),
            door_knob=(220, 180, 60),
            floor_fill=(148, 158, 140),
            floor_dark=(105, 115, 98),
            int_wall=(88, 98, 80),
            int_dark=(55, 62, 48),
            _emissive=(210, 175, 55),  # warm amber LED strip at base
        ),
    ),
    # ── Cryo outpost — ice world ────────────────────────────────────────────
    # Frosted white composite panels, geodesic roofs, cyan cryo-light accents
    BuildingStyle(
        name="cryo",
        biome="ice",
        wall_texture="cryo_panel",
        roof_texture="geodesic",
        roughness=0.02,
        seed=20,
        palette=SciFiPalette(
            roof_fill=(195, 218, 238),
            roof_edge=(150, 180, 210),
            roof_dark=(105, 140, 172),
            wall_fill=(218, 232, 245),
            wall_edge=(172, 195, 220),
            wall_dark=(128, 158, 188),
            window_fill=(145, 220, 255),
            window_frame=(95, 145, 185),
            door_fill=(168, 198, 225),
            door_frame=(115, 152, 185),
            door_knob=(145, 220, 255),
            floor_fill=(195, 215, 235),
            floor_dark=(148, 172, 200),
            int_wall=(138, 165, 195),
            int_dark=(95, 120, 152),
            _emissive=(90, 210, 255),  # cyan cryo-light strip
        ),
    ),
    # ── Extraction facility — rocky world ──────────────────────────────────
    # Rusted corrugated metal, flat roofs with vents, hazard stripes + red LEDs
    BuildingStyle(
        name="extraction",
        biome="rocky",
        wall_texture="corrugated_rust",
        roof_texture="corrugated_rust",
        roughness=0.07,
        seed=30,
        palette=SciFiPalette(
            roof_fill=(88, 72, 55),
            roof_edge=(65, 52, 38),
            roof_dark=(42, 32, 22),
            wall_fill=(118, 95, 68),
            wall_edge=(88, 70, 48),
            wall_dark=(58, 45, 30),
            window_fill=(255, 135, 55),
            window_frame=(55, 45, 32),
            door_fill=(75, 60, 42),
            door_frame=(48, 38, 25),
            door_knob=(255, 185, 45),
            floor_fill=(105, 88, 65),
            floor_dark=(72, 58, 42),
            int_wall=(78, 65, 48),
            int_dark=(48, 38, 28),
            _emissive=(220, 60, 40),  # red emergency LED strip
            _hazard_stripe=True,  # yellow/black warning stripes
        ),
    ),
    # ── Station module — city world ─────────────────────────────────────────
    # Dark composite panels with routed cyan conduit channels, hex roofs
    BuildingStyle(
        name="station",
        biome="city",
        wall_texture="conduit",
        roof_texture="hex",
        roughness=0.02,
        seed=40,
        palette=SciFiPalette(
            roof_fill=(48, 55, 65),
            roof_edge=(35, 40, 50),
            roof_dark=(22, 25, 32),
            wall_fill=(68, 78, 92),
            wall_edge=(48, 56, 68),
            wall_dark=(30, 35, 44),
            window_fill=(55, 195, 235),
            window_frame=(38, 48, 60),
            door_fill=(45, 52, 62),
            door_frame=(28, 34, 44),
            door_knob=(55, 195, 235),
            floor_fill=(85, 95, 110),
            floor_dark=(58, 66, 78),
            int_wall=(52, 60, 72),
            int_dark=(32, 38, 48),
            _emissive=(55, 195, 235),  # cyan conduit glow at base
        ),
    ),
    # ── Outpost — desert world ─────────────────────────────────────────────
    # Heat-bleached composite panels, flat roofs with heat vents, amber glow
    BuildingStyle(
        name="outpost",
        biome="desert",
        wall_texture="composite",
        roof_texture="flat_sci",
        roughness=0.05,
        seed=50,
        palette=SciFiPalette(
            roof_fill=(158, 138, 105),
            roof_edge=(118, 100, 72),
            roof_dark=(78, 65, 45),
            wall_fill=(192, 172, 138),
            wall_edge=(148, 130, 100),
            wall_dark=(105, 90, 65),
            window_fill=(225, 190, 105),
            window_frame=(95, 80, 55),
            door_fill=(128, 108, 78),
            door_frame=(85, 72, 48),
            door_knob=(235, 195, 65),
            floor_fill=(172, 155, 125),
            floor_dark=(125, 112, 88),
            int_wall=(115, 100, 75),
            int_dark=(72, 62, 42),
            _emissive=(225, 170, 45),  # amber heat-vent glow
        ),
    ),
]

# Template assignments for sci-fi styles
_b._TEMPLATE_FACTORIES.update(
    {
        "composite": [make_small_house, make_medium_house, make_large_building, make_wide_house, make_tall_building],
        "cryo_panel": [make_small_house, make_medium_house, make_tower, make_wide_house, make_tall_building],
        "corrugated_rust": [make_small_house, make_medium_house, make_large_building, make_wide_house, make_tall_building],
        "conduit": [make_station_room, make_medium_house, make_large_building, make_wide_house, make_tall_building],
    }
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Standalone entry point                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def generate_all(tile_size: int, out_dir: Path) -> None:
    """Generate all sci-fi building atlases, templates, and manifest."""
    tmpl_dir = out_dir / "buildings"
    tmpl_dir.mkdir(exist_ok=True)

    for style in BUILDING_STYLES:
        print(f"  Sci-fi style: {style.name}  (biome: {style.biome})")

        ext_atlas = _b.build_exterior_atlas(style, tile_size)
        ext_path = out_dir / f"{style.name}_building_exterior.png"
        ext_atlas.save(ext_path)
        print(f"    → {ext_path.name}  {ext_atlas.size[0]}×{ext_atlas.size[1]}px")

        int_atlas = _b.build_interior_atlas(style, tile_size)
        int_path = out_dir / f"{style.name}_building_interior.png"
        int_atlas.save(int_path)
        print(f"    → {int_path.name}  {int_atlas.size[0]}×{int_atlas.size[1]}px")

        pad_atlas = _b.build_landing_pad_atlas(style, tile_size)
        pad_path = out_dir / f"{style.name}_landing_pad.png"
        pad_atlas.save(pad_path)
        print(f"    → {pad_path.name}  {pad_atlas.size[0]}×{pad_atlas.size[1]}px")

        mech_atlas = _b.build_mechanic_atlas(style, tile_size)
        mech_path = out_dir / f"{style.name}_mechanic.png"
        mech_atlas.save(mech_path)
        print(f"    → {mech_path.name}  {mech_atlas.size[0]}×{mech_atlas.size[1]}px")

        factories = _b._TEMPLATE_FACTORIES.get(style.wall_texture, [make_small_house])
        for factory in factories:
            tmpl = factory(style)
            ron_text = _b._template_to_ron(tmpl)
            ron_path = tmpl_dir / f"{style.name}_{tmpl.name}.ron"
            ron_path.write_text(ron_text)
            print(f"    → buildings/{style.name}_{tmpl.name}.ron")

    _b.write_buildings_manifest(BUILDING_STYLES, tile_size, out_dir)
    print("  → buildings_manifest.ron")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sci-fi building tile atlases"
    )
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="output_scifi")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    print(f"Tile size : {args.tile_size}px  →  {out.resolve()}\n")
    generate_all(args.tile_size, out)
    print("\nDone.")


if __name__ == "__main__":
    main()
