"""
buildings.py  —  Building tile atlas + template generator
==========================================================

Generates per-biome building tilesets and procedural building templates.

Exterior tileset layout (30 tiles, 6 cols × 5 rows):
  Row 0: roof_nw    roof_n    roof_ne   roof_w    roof_fill  roof_e
  Row 1: roof_sw    roof_s    roof_se   attic_w   attic      attic_e
  Row 2: wall_nw    wall_n    wall_ne   wall_w    wall_fill  wall_e
  Row 3: wall_sw    wall_s    wall_se   win_w     window     win_e
  Row 4: base_nw    base_n    base_ne   door_l    door       door_r

Interior tileset layout (16 tiles, 4 cols × 4 rows) — blob-4 indexed:
  Cardinals only: N=1, E=2, S=4, W=8  (note: different bit assignment from terrain)
  Row 0 (mask 0–3):  isolated  N-only   E-only   N+E
  Row 1 (mask 4–7):  S-only    N+S      E+S      N+E+S
  Row 2 (mask 8–11): W-only    N+W      E+W      N+E+W
  Row 3 (mask 12–15):S+W       N+S+W    E+S+W    N+E+S+W (fully surrounded = floor)

Output per biome style:
  <style>_building_exterior.png   30-tile exterior sheet
  <style>_building_interior.png   16-tile interior sheet
  buildings_manifest.ron          all style metadata + Blob4 LUT
  buildings/<style>/<name>.ron    individual building templates
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

# Re-use rendering helpers from tilegen_full
from tilegen import (
    CC,
    TerrainSpec,
    render_base_tile,
)


# ── Tile indices into the exterior sheet ─────────────────────────────────────
# (row * 6 + col)  — used in BuildingTemplate.tiles
class EXT:
    ROOF_NW = 0
    ROOF_N = 1
    ROOF_NE = 2
    ROOF_W = 3
    ROOF_FILL = 4
    ROOF_E = 5
    ROOF_SW = 6
    ROOF_S = 7
    ROOF_SE = 8
    ATTIC_W = 9
    ATTIC = 10
    ATTIC_E = 11
    WALL_NW = 12
    WALL_N = 13
    WALL_NE = 14
    WALL_W = 15
    WALL_FILL = 16
    WALL_E = 17
    WALL_SW = 18
    WALL_S = 19
    WALL_SE = 20
    WIN_W = 21
    WINDOW = 22
    WIN_E = 23
    BASE_NW = 24
    BASE_N = 25
    BASE_NE = 26
    DOOR_L = 27
    DOOR = 28
    DOOR_R = 29


EXT_COLS = 6
EXT_ROWS = 5
EXT_TOTAL = EXT_COLS * EXT_ROWS  # 30

# ── Blob-4 LUT for interior walls ─────────────────────────────────────────────
# N=1, E=2, S=4, W=8  (cardinal only — man-made structures, no diagonals)
INT_N, INT_E, INT_S, INT_W = 1, 2, 4, 8
# mask (0–15) → atlas index (same value — blob-4 is already contiguous)
INT_COLS = 4
INT_ROWS = 4
INT_TOTAL = INT_COLS * INT_ROWS  # 16

BLOB4_LUT: list[int] = list(range(16))  # identity mapping: mask == atlas index

# ── Collision codes for building tiles ───────────────────────────────────────
# All exterior shells + interior walls are SOLID.
# Floors, windows (for sight-lines), and doors (when open) are WALKABLE or TRIGGER.
_EXT_COLLISION: dict[int, int] = {
    EXT.ROOF_NW: CC.SOLID,
    EXT.ROOF_N: CC.SOLID,
    EXT.ROOF_NE: CC.SOLID,
    EXT.ROOF_W: CC.SOLID,
    EXT.ROOF_FILL: CC.SOLID,
    EXT.ROOF_E: CC.SOLID,
    EXT.ROOF_SW: CC.SOLID,
    EXT.ROOF_S: CC.SOLID,
    EXT.ROOF_SE: CC.SOLID,
    EXT.ATTIC_W: CC.SOLID,
    EXT.ATTIC: CC.SOLID,
    EXT.ATTIC_E: CC.SOLID,
    EXT.WALL_NW: CC.SOLID,
    EXT.WALL_N: CC.SOLID,
    EXT.WALL_NE: CC.SOLID,
    EXT.WALL_W: CC.SOLID,
    EXT.WALL_FILL: CC.SOLID,
    EXT.WALL_E: CC.SOLID,
    EXT.WALL_SW: CC.SOLID,
    EXT.WALL_S: CC.SOLID,
    EXT.WALL_SE: CC.SOLID,
    EXT.WIN_W: CC.SOLID,
    EXT.WINDOW: CC.SOLID,
    EXT.WIN_E: CC.SOLID,
    EXT.BASE_NW: CC.SOLID,
    EXT.BASE_N: CC.SOLID,
    EXT.BASE_NE: CC.SOLID,
    EXT.DOOR_L: CC.SOLID,
    EXT.DOOR: CC.TRIGGER,
    EXT.DOOR_R: CC.SOLID,
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data classes                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class ColourPalette:
    """All colour roles needed to render a building style."""

    # Roof
    roof_fill: tuple[int, int, int] = (120, 60, 40)
    roof_edge: tuple[int, int, int] = (90, 40, 25)
    roof_dark: tuple[int, int, int] = (60, 25, 15)
    # Wall
    wall_fill: tuple[int, int, int] = (200, 185, 160)
    wall_edge: tuple[int, int, int] = (160, 145, 120)
    wall_dark: tuple[int, int, int] = (110, 100, 85)
    # Window
    window_fill: tuple[int, int, int] = (140, 180, 220)
    window_frame: tuple[int, int, int] = (80, 70, 60)
    # Door
    door_fill: tuple[int, int, int] = (100, 75, 50)
    door_frame: tuple[int, int, int] = (70, 55, 35)
    door_knob: tuple[int, int, int] = (200, 170, 80)
    # Interior floor
    floor_fill: tuple[int, int, int] = (175, 165, 150)
    floor_dark: tuple[int, int, int] = (130, 122, 110)
    # Interior wall
    int_wall: tuple[int, int, int] = (90, 88, 85)
    int_dark: tuple[int, int, int] = (55, 53, 50)


@dataclass
class BuildingStyle:
    name: str
    palette: ColourPalette
    # Texture parameters
    roughness: float = 0.04
    wall_texture: str = "stone"  # "stone" | "brick" | "panel" | "ice" | "log"
    roof_texture: str = "tile"  # "tile"  | "flat"  | "ice"   | "metal"
    seed: int = 0
    # Which biome this style is used in (informational)
    biome: str = "garden"


@dataclass
class BuildingTemplate:
    """
    A specific building layout: a 2-D grid of EXT tile indices.
    0 = transparent (no tile placed).
    entry_points: list of (col, row) door positions in tile-coords.
    interior_offset: (col, row) top-left of the walkable interior rectangle.
    interior_size: (w, h) in tiles.
    """

    name: str
    style: str
    tiles: list[list[int]]  # [row][col], 0 = empty
    width: int
    height: int
    entry_points: list[tuple[int, int]]
    interior_offset: tuple[int, int]
    interior_size: tuple[int, int]
    layer: int = 1  # z-layer above terrain


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Tile rendering helpers                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _fill(
    size: int, colour: tuple[int, int, int], roughness: float = 0.0, seed: int = 0
) -> np.ndarray:
    """Flat colour fill with optional per-pixel jitter."""
    arr = np.full((size, size, 3), colour, dtype=np.float32)
    if roughness > 0:
        rng = np.random.default_rng(seed)
        arr += rng.normal(0, roughness * 255, (size, size, 3)).astype(np.float32)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _draw_brick(
    img: Image.Image,
    colour: tuple[int, int, int],
    mortar: tuple[int, int, int],
    size: int,
    brick_h: int = 5,
    brick_w: int = 10,
    seed: int = 0,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    for row in range(0, size, brick_h):
        offset = (brick_w // 2) if (row // brick_h) % 2 else 0
        for col in range(-brick_w, size + brick_w, brick_w):
            x0 = col + offset
            jit = int(rng.integers(-1, 2))
            r = tuple(max(0, min(255, c + jit * 8)) for c in colour)
            draw.rectangle(
                [x0 + 1, row + 1, x0 + brick_w - 2, row + brick_h - 2], fill=r
            )
        # horizontal mortar line
        draw.line([(0, row), (size, row)], fill=mortar, width=1)
    # vertical mortar
    for row in range(0, size, brick_h):
        offset = (brick_w // 2) if (row // brick_h) % 2 else 0
        for col in range(-brick_w, size + brick_w, brick_w):
            draw.line(
                [(col + offset, row), (col + offset, row + brick_h)],
                fill=mortar,
                width=1,
            )
    return img


def _draw_stone(
    img: Image.Image,
    colour: tuple[int, int, int],
    dark: tuple[int, int, int],
    size: int,
    seed: int = 0,
) -> Image.Image:
    """Irregular stone block texture."""
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        x0 = int(rng.integers(0, size - 6))
        y0 = int(rng.integers(0, size - 4))
        w = int(rng.integers(5, min(14, size - x0)))
        h = int(rng.integers(3, min(8, size - y0)))
        jit = int(rng.integers(-15, 16))
        c = tuple(max(0, min(255, v + jit)) for v in colour)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=c, outline=dark)
    return img


def _draw_panel(
    img: Image.Image,
    colour: tuple[int, int, int],
    edge: tuple[int, int, int],
    size: int,
    grid: int = 8,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x in range(0, size, grid):
        draw.line([(x, 0), (x, size)], fill=edge, width=1)
    for y in range(0, size, grid):
        draw.line([(0, y), (size, y)], fill=edge, width=1)
    # Rivet dots at intersections
    rv = tuple(min(255, c + 30) for c in colour)
    for x in range(0, size, grid):
        for y in range(0, size, grid):
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=rv)
    return img


def _draw_tile_roof(
    img: Image.Image,
    colour: tuple[int, int, int],
    dark: tuple[int, int, int],
    size: int,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    h = max(3, size // 5)
    for row in range(0, size, h):
        col_offset = (size // 3) if (row // h) % 2 else 0
        w = size // 3
        for col in range(-w, size + w, w):
            x0 = col + col_offset
            pts = [x0, row + h, x0 + w // 2, row, x0 + w, row + h]
            draw.polygon(pts, fill=colour, outline=dark)
    return img


def _draw_ice_block(
    img: Image.Image,
    colour: tuple[int, int, int],
    bright: tuple[int, int, int],
    size: int,
    seed: int = 0,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    block = max(6, size // 4)
    for y in range(0, size, block):
        for x in range(0, size, block):
            jit = int(rng.integers(-10, 11))
            c = tuple(max(0, min(255, v + jit)) for v in colour)
            draw.rectangle([x + 1, y + 1, x + block - 1, y + block - 1], fill=c)
            # specular glint
            draw.line([(x + 2, y + 2), (x + 4, y + 2)], fill=bright, width=1)
    return img


def _make_wall_base(style: BuildingStyle, size: int, seed: int = 0) -> np.ndarray:
    """Generate a wall-fill texture array for the given building style."""
    p = style.palette
    arr = _fill(size, p.wall_fill, style.roughness * 0.5, seed)
    img = Image.fromarray(arr)
    if style.wall_texture == "brick":
        img = _draw_brick(img, p.wall_fill, p.wall_dark, size, seed=seed)
    elif style.wall_texture == "stone":
        img = _draw_stone(img, p.wall_fill, p.wall_dark, size, seed=seed)
    elif style.wall_texture == "panel":
        img = _draw_panel(img, p.wall_fill, p.wall_edge, size)
    elif style.wall_texture == "ice":
        img = _draw_ice_block(
            img, p.wall_fill, tuple(min(255, c + 40) for c in p.wall_fill), size, seed
        )
    elif style.wall_texture == "log":
        draw = ImageDraw.Draw(img)
        lc = p.wall_dark
        for y in range(0, size, max(4, size // 5)):
            draw.line([(0, y), (size, y)], fill=lc, width=1)
    return np.array(img)


def _make_roof_base(style: BuildingStyle, size: int, seed: int = 0) -> np.ndarray:
    p = style.palette
    arr = _fill(size, p.roof_fill, style.roughness * 0.6, seed)
    img = Image.fromarray(arr)
    if style.roof_texture == "tile":
        img = _draw_tile_roof(img, p.roof_fill, p.roof_dark, size)
    elif style.roof_texture == "flat":
        draw = ImageDraw.Draw(img)
        draw.rectangle([2, 2, size - 3, size - 3], outline=p.roof_edge, width=1)
    elif style.roof_texture == "ice":
        img = _draw_ice_block(
            img, p.roof_fill, tuple(min(255, c + 50) for c in p.roof_fill), size, seed
        )
    elif style.roof_texture == "metal":
        draw = ImageDraw.Draw(img)
        sp = max(4, size // 6)
        for x in range(0, size, sp):
            draw.line([(x, 0), (x, size)], fill=p.roof_edge, width=1)
    return np.array(img)


def _shadow_edge(
    arr: np.ndarray, edge: str, width: int = 3, alpha: float = 0.55
) -> np.ndarray:
    """Darken a strip along one edge to fake a shadow / depth indicator."""
    out = arr.astype(np.float32)
    if edge == "top":
        out[:width, :] *= alpha
    elif edge == "bottom":
        out[-width:, :] *= alpha
    elif edge == "left":
        out[:, :width] *= alpha
    elif edge == "right":
        out[:, -width:] *= alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _highlight_edge(
    arr: np.ndarray, edge: str, width: int = 2, alpha: float = 1.35
) -> np.ndarray:
    out = arr.astype(np.float32)
    if edge == "top":
        out[:width, :] = np.clip(out[:width, :] * alpha, 0, 255)
    elif edge == "left":
        out[:, :width] = np.clip(out[:, :width] * alpha, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_window(
    draw: ImageDraw.ImageDraw, x: int, y: int, size: int, palette: ColourPalette
) -> None:
    """Draw a window centered in a tile at pixel offset (x,y)."""
    p = palette
    m = max(4, size // 6)
    w = size - 2 * m
    # Frame
    draw.rectangle(
        [x + m - 1, y + m - 1, x + size - m, y + size - m],
        fill=p.window_frame,
        outline=p.window_frame,
    )
    # Glass (split into 4 panes)
    hw = w // 2
    for dy in (0, hw + 1):
        for dx in (0, hw + 1):
            draw.rectangle(
                [x + m + dx, y + m + dy, x + m + dx + hw - 1, y + m + dy + hw - 1],
                fill=p.window_fill,
            )
    # Glint
    draw.line(
        [(x + m + 1, y + m + 1), (x + m + 3, y + m + 1)],
        fill=tuple(min(255, c + 60) for c in p.window_fill),
        width=1,
    )


def _draw_door(
    draw: ImageDraw.ImageDraw, x: int, y: int, size: int, palette: ColourPalette
) -> None:
    p = palette
    m = max(3, size // 8)
    w = size - 2 * m
    h = size - m
    draw.rectangle(
        [x + m, y + m, x + m + w, y + h], fill=p.door_fill, outline=p.door_frame
    )
    # Door panels
    pw = w // 2 - 1
    ph = (h - m) // 2 - 1
    for dy in (m + 1, m + ph + 3):
        for dx in (m + 1, m + pw + 2):
            draw.rectangle(
                [x + dx, y + dy, x + dx + pw, y + dy + ph],
                outline=p.door_frame,
                width=1,
            )
    # Knob
    draw.ellipse(
        [x + size // 2 - 2, y + size // 2, x + size // 2 + 1, y + size // 2 + 3],
        fill=p.door_knob,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Exterior atlas assembly                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def build_exterior_atlas(style: BuildingStyle, tile_size: int) -> Image.Image:
    """
    Generate the 6×5 (30-tile) exterior building atlas for this style.
    """
    S = tile_size
    p = style.palette
    atlas = Image.new("RGBA", (EXT_COLS * S, EXT_ROWS * S), (0, 0, 0, 0))

    wall = _make_wall_base(style, S, style.seed)
    roof = _make_roof_base(style, S, style.seed + 1)

    # Helper: paste an ndarray tile at (col, row) in the atlas
    def paste(arr: np.ndarray, col: int, row: int) -> None:
        img = Image.fromarray(arr, "RGB").convert("RGBA")
        atlas.paste(img, (col * S, row * S))

    def solid(colour: tuple[int, int, int], roughness: float = 0.02) -> np.ndarray:
        return _fill(S, colour, roughness, style.seed)

    # ── Row 0: roof caps ────────────────────────────────────────────────────
    rf = roof.copy()
    paste(_shadow_edge(_highlight_edge(rf, "top"), "left"), 0, 0)  # NW
    paste(_highlight_edge(rf, "top"), 1, 0)  # N
    paste(_shadow_edge(_highlight_edge(rf, "top"), "right"), 2, 0)  # NE
    paste(_shadow_edge(rf, "left"), 3, 0)  # W
    paste(rf, 4, 0)  # fill
    paste(_shadow_edge(rf, "right"), 5, 0)  # E

    # ── Row 1: roof bottom / attic strip ────────────────────────────────────
    attic = solid(p.roof_edge, 0.03)
    attic = _shadow_edge(attic, "bottom", width=4, alpha=0.5)
    paste(_shadow_edge(_shadow_edge(roof, "bottom"), "left"), 0, 1)  # SW
    paste(_shadow_edge(roof, "bottom"), 1, 1)  # S
    paste(_shadow_edge(_shadow_edge(roof, "bottom"), "right"), 2, 1)  # SE
    paste(_shadow_edge(attic, "left"), 3, 1)  # attic_w
    paste(attic, 4, 1)  # attic
    paste(_shadow_edge(attic, "right"), 5, 1)  # attic_e

    # ── Row 2: wall ─────────────────────────────────────────────────────────
    wf = wall.copy()
    paste(_shadow_edge(_highlight_edge(wf, "top"), "left"), 0, 2)  # NW
    paste(_highlight_edge(wf, "top"), 1, 2)  # N
    paste(_shadow_edge(_highlight_edge(wf, "top"), "right"), 2, 2)  # NE
    paste(_shadow_edge(wf, "left"), 3, 2)  # W
    paste(wf, 4, 2)  # fill
    paste(_shadow_edge(wf, "right"), 5, 2)  # E

    # ── Row 3: wall + windows ───────────────────────────────────────────────
    def make_window_tile(shadow_side: str | None = None) -> np.ndarray:
        base = wall.copy()
        img = Image.fromarray(base).convert("RGBA")
        draw = ImageDraw.Draw(img)
        _draw_window(draw, 0, 0, S, p)
        arr = np.array(img.convert("RGB"))
        if shadow_side:
            arr = _shadow_edge(arr, shadow_side)
        return arr

    paste(_shadow_edge(make_window_tile(), "left"), 0, 3)  # win_w  (no window, edge)
    paste(make_window_tile(), 1, 3)  # window (unused col)
    paste(_shadow_edge(make_window_tile(), "right"), 2, 3)  # win_e
    paste(_shadow_edge(wall, "left"), 3, 3)  # WIN_W slot  (plain wall)
    paste(make_window_tile(), 4, 3)  # WINDOW
    paste(_shadow_edge(wall, "right"), 5, 3)  # WIN_E slot (plain wall)

    # ── Row 4: base / door row ──────────────────────────────────────────────
    base_arr = _shadow_edge(wall, "bottom", width=4, alpha=0.6)

    def make_door_tile(part: str) -> np.ndarray:
        b = base_arr.copy()
        img = Image.fromarray(b).convert("RGBA")
        draw = ImageDraw.Draw(img)
        if part == "door":
            _draw_door(draw, 0, 0, S, p)
        elif part == "door_l":
            # Left half of door frame
            m = max(3, S // 8)
            draw.rectangle([S - m, m, S, S], fill=p.door_fill, outline=p.door_frame)
        elif part == "door_r":
            m = max(3, S // 8)
            draw.rectangle([0, m, m, S], fill=p.door_fill, outline=p.door_frame)
        return np.array(img.convert("RGB"))

    paste(_shadow_edge(_shadow_edge(base_arr, "bottom"), "left"), 0, 4)  # base_nw
    paste(_shadow_edge(base_arr, "bottom"), 1, 4)  # base_n
    paste(_shadow_edge(_shadow_edge(base_arr, "bottom"), "right"), 2, 4)  # base_ne
    paste(make_door_tile("door_l"), 3, 4)  # door_l
    paste(make_door_tile("door"), 4, 4)  # door
    paste(make_door_tile("door_r"), 5, 4)  # door_r

    return atlas


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Interior atlas assembly  (blob-4, 16 tiles)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def build_interior_atlas(style: BuildingStyle, tile_size: int) -> Image.Image:
    """
    Generate the 4×4 (16-tile) interior wall atlas for this style.
    Indexed by blob-4 cardinal mask (N=1,E=2,S=4,W=8).
    mask 15 (all neighbours same) = floor tile.
    """
    S = tile_size
    p = style.palette
    atlas = Image.new("RGBA", (INT_COLS * S, INT_ROWS * S), (0, 0, 0, 0))

    floor_spec = TerrainSpec(
        name="floor",
        base=p.floor_fill,
        accent=tuple(min(255, c + 20) for c in p.floor_fill),
        dark=p.floor_dark,
        roughness=0.03,
        macro_scale=8.0,
        detail_scale=16.0,
        micro_scale=24.0,
        macro_w=0.1,
        detail_w=0.3,
        micro_w=0.6,
        grid=(style.wall_texture == "panel"),
        grid_spacing=8,
    )
    wall_arr = _fill(S, p.int_wall, 0.04, style.seed + 50)
    if style.wall_texture == "panel":
        img = Image.fromarray(wall_arr)
        wall_arr = np.array(_draw_panel(img, p.int_wall, p.int_dark, S))

    floor_arr = render_base_tile(floor_spec, S, style.seed + 100)

    for mask in range(16):
        col = mask % INT_COLS
        row = mask // INT_COLS
        px = col * S
        py = row * S

        if mask == 15:
            # Fully surrounded = pure floor
            tile = Image.fromarray(floor_arr).convert("RGBA")
        else:
            # Blend: wall where no same-type neighbour, floor otherwise
            base = wall_arr.astype(np.float32)
            flo = floor_arr.astype(np.float32)

            # For each cardinal that IS same-type (bit set), blend toward floor
            ys, xs = np.mgrid[0:S, 0:S]
            blend = np.zeros((S, S), dtype=np.float32)

            for bit, axis, hi_side in [
                (INT_N, "y", False),
                (INT_S, "y", True),
                (INT_E, "x", True),
                (INT_W, "x", False),
            ]:
                if not (mask & bit):
                    continue
                coords = (
                    (ys / S)
                    if (axis == "y" and hi_side)
                    else ((S - 1 - ys) / S)
                    if (axis == "y")
                    else (xs / S)
                    if hi_side
                    else ((S - 1 - xs) / S)
                )
                t = np.clip(coords * 2.0, 0.0, 1.0)
                w = t * t * (3 - 2 * t)
                blend = np.maximum(blend, w)

            blend3 = blend[:, :, np.newaxis]
            merged = base * (1 - blend3) + flo * blend3
            merged = np.clip(merged, 0, 255).astype(np.uint8)

            # Apply shadow on unshared edges
            if not (mask & INT_N):
                merged = _shadow_edge(merged, "top", 3, 0.55)
            if not (mask & INT_S):
                merged = _shadow_edge(merged, "bottom", 3, 0.55)
            if not (mask & INT_W):
                merged = _shadow_edge(merged, "left", 3, 0.55)
            if not (mask & INT_E):
                merged = _shadow_edge(merged, "right", 3, 0.55)

            tile = Image.fromarray(merged).convert("RGBA")

        atlas.paste(tile, (px, py))

    return atlas


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Building template generation                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _make_exterior_shell(
    width: int,
    height: int,
    door_col: int,
    window_rows: list[int],
    window_cols: list[int],
) -> list[list[int]]:
    """
    Lay out the exterior tile indices for a building of (width × height) tiles.
    width, height are total footprint including walls.
    door_col: which column (0-based) gets the door (must be interior col).
    window_rows / window_cols: which row/col positions get windows.
    Returns grid[row][col] of EXT.* indices.
    """
    grid = [[0] * width for _ in range(height)]

    for r in range(height):
        for c in range(width):
            left = c == 0
            right = c == width - 1
            top = r == 0
            bot = r == height - 1
            roof_rows = 2  # rows 0,1 are roof
            is_roof = r < roof_rows
            is_attic = r == roof_rows - 1
            is_base = bot
            is_door_c = c == door_col
            is_win = r in window_rows and c in window_cols and not left and not right

            if is_roof and not is_attic:
                # Pure roof fill row
                if left and top:
                    grid[r][c] = EXT.ROOF_NW
                elif right and top:
                    grid[r][c] = EXT.ROOF_NE
                elif left:
                    grid[r][c] = EXT.ROOF_W
                elif right:
                    grid[r][c] = EXT.ROOF_E
                else:
                    grid[r][c] = EXT.ROOF_FILL
            elif is_attic:
                # Roof bottom / attic strip
                if left:
                    grid[r][c] = EXT.ROOF_SW
                elif right:
                    grid[r][c] = EXT.ROOF_SE
                else:
                    grid[r][c] = EXT.ROOF_S
            elif is_base:
                # Ground-floor (base/door row)
                if left:
                    grid[r][c] = EXT.BASE_NW
                elif right:
                    grid[r][c] = EXT.BASE_NE
                elif is_door_c - 1 == c:
                    grid[r][c] = EXT.DOOR_L
                elif is_door_c:
                    grid[r][c] = EXT.DOOR
                elif is_door_c + 1 == c:
                    grid[r][c] = EXT.DOOR_R
                else:
                    grid[r][c] = EXT.BASE_N
            else:
                # Mid wall rows
                if left and top:
                    grid[r][c] = EXT.WALL_NW
                elif right and top:
                    grid[r][c] = EXT.WALL_NE
                elif left and bot:
                    grid[r][c] = EXT.WALL_SW
                elif right and bot:
                    grid[r][c] = EXT.WALL_SE
                elif left:
                    grid[r][c] = EXT.WALL_W
                elif right:
                    grid[r][c] = EXT.WALL_E
                elif is_win:
                    grid[r][c] = EXT.WINDOW
                else:
                    grid[r][c] = EXT.WALL_FILL

    return grid


def make_small_house(style: BuildingStyle) -> BuildingTemplate:
    """4-wide × 4-tall exterior shell."""
    W, H = 4, 4
    door_col = 2
    grid = _make_exterior_shell(
        W, H, door_col=door_col, window_rows=[2], window_cols=[1]
    )
    return BuildingTemplate(
        name="small_house",
        style=style.name,
        tiles=grid,
        width=W,
        height=H,
        entry_points=[(door_col, H - 1)],
        interior_offset=(1, 2),
        interior_size=(W - 2, H - 2 - 1),
    )


def make_medium_house(style: BuildingStyle) -> BuildingTemplate:
    """6-wide × 5-tall exterior shell."""
    W, H = 6, 5
    door_col = 3
    grid = _make_exterior_shell(
        W, H, door_col=door_col, window_rows=[2, 3], window_cols=[1, 2, 4]
    )
    return BuildingTemplate(
        name="medium_house",
        style=style.name,
        tiles=grid,
        width=W,
        height=H,
        entry_points=[(door_col, H - 1)],
        interior_offset=(1, 2),
        interior_size=(W - 2, H - 2 - 1),
    )


def make_large_building(style: BuildingStyle) -> BuildingTemplate:
    """8-wide × 6-tall exterior shell — shop / inn / barracks."""
    W, H = 8, 6
    door_col = 4
    grid = _make_exterior_shell(
        W, H, door_col=door_col, window_rows=[2, 3, 4], window_cols=[1, 2, 3, 5, 6]
    )
    return BuildingTemplate(
        name="large_building",
        style=style.name,
        tiles=grid,
        width=W,
        height=H,
        entry_points=[(door_col, H - 1), (door_col - 1, H - 1)],
        interior_offset=(1, 2),
        interior_size=(W - 2, H - 2 - 1),
    )


def make_tower(style: BuildingStyle) -> BuildingTemplate:
    """3-wide × 7-tall tower."""
    W, H = 3, 7
    door_col = 1
    grid = _make_exterior_shell(
        W, H, door_col=door_col, window_rows=[2, 4], window_cols=[1]
    )
    return BuildingTemplate(
        name="tower",
        style=style.name,
        tiles=grid,
        width=W,
        height=H,
        entry_points=[(door_col, H - 1)],
        interior_offset=(1, 2),
        interior_size=(1, H - 2 - 1),
    )


def make_station_room(style: BuildingStyle) -> BuildingTemplate:
    """5-wide × 5-tall interior room for space station / city."""
    W, H = 5, 5
    door_col = 2
    grid = _make_exterior_shell(W, H, door_col=door_col, window_rows=[], window_cols=[])
    return BuildingTemplate(
        name="station_room",
        style=style.name,
        tiles=grid,
        width=W,
        height=H,
        entry_points=[(door_col, H - 1), (door_col, 0)],
        interior_offset=(1, 2),
        interior_size=(W - 2, H - 2 - 1),
    )


# Map style name → list of template-factory functions
_TEMPLATE_FACTORIES: dict[str, list[Callable]] = {
    "stone": [make_small_house, make_medium_house, make_large_building, make_tower],
    "brick": [make_small_house, make_medium_house, make_large_building],
    "ice": [make_small_house, make_medium_house, make_tower],
    "log": [make_small_house, make_medium_house],
    "panel": [make_station_room, make_medium_house, make_large_building],
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RON serialisation                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _template_to_ron(tmpl: BuildingTemplate) -> str:
    rows_ron = (
        "[\n" + "".join(f"            {row!r},\n" for row in tmpl.tiles) + "        ]"
    )
    ep_ron = "[" + ", ".join(f"({c}, {r})" for c, r in tmpl.entry_points) + "]"
    io = tmpl.interior_offset
    is_ = tmpl.interior_size
    return (
        f"// {tmpl.name}.ron — auto-generated by buildings.py\n"
        f"(\n"
        f'    name:            "{tmpl.name}",\n'
        f'    style:           "{tmpl.style}",\n'
        f"    width:           {tmpl.width},\n"
        f"    height:          {tmpl.height},\n"
        f"    layer:           {tmpl.layer},\n"
        f"    tiles:           {rows_ron},\n"
        f"    entry_points:    {ep_ron},\n"
        f"    interior_offset: ({io[0]}, {io[1]}),\n"
        f"    interior_size:   ({is_[0]}, {is_[1]}),\n"
        f")\n"
    )


def write_buildings_manifest(
    styles: list[BuildingStyle],
    tile_size: int,
    out_dir: Path,
) -> None:
    lines = [
        "// buildings_manifest.ron — auto-generated by buildings.py\n",
        "// CollisionType: 0=walkable 1=solid 2=slow 3=damaging 4=trigger\n",
        "(\n",
        f"    tile_size: {tile_size},\n",
        f"    ext_cols: {EXT_COLS},\n",
        f"    ext_rows: {EXT_ROWS},\n",
        f"    int_cols: {INT_COLS},\n",
        f"    int_rows: {INT_ROWS},\n",
        # Blob-4 LUT (trivial identity, included for completeness)
        f"    blob4_lut: {BLOB4_LUT!r},\n",
        # Per-slot collision for the exterior sheet
        f"    ext_collision: {[_EXT_COLLISION[i] for i in range(EXT_TOTAL)]!r},\n",
        "    styles: {\n",
    ]
    for s in styles:
        factories = _TEMPLATE_FACTORIES.get(s.wall_texture, [make_small_house])
        template_names = [f.__name__.replace("make_", "") for f in factories]
        lines += [
            f'        "{s.name}": (\n',
            f'            biome: "{s.biome}",\n',
            f'            exterior_atlas: "{s.name}_building_exterior.png",\n',
            f'            interior_atlas: "{s.name}_building_interior.png",\n',
            f"            templates: {template_names!r},\n",
            "        ),\n",
        ]
    lines += ["    },\n", ")\n"]
    (out_dir / "buildings_manifest.ron").write_text("".join(lines))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Building styles                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BUILDING_STYLES: list[BuildingStyle] = [
    BuildingStyle(
        name="stone",
        biome="garden",
        wall_texture="stone",
        roof_texture="tile",
        seed=10,
        palette=ColourPalette(
            roof_fill=(105, 55, 35),
            roof_edge=(78, 38, 22),
            roof_dark=(52, 24, 12),
            wall_fill=(192, 180, 160),
            wall_edge=(148, 138, 120),
            wall_dark=(100, 92, 78),
            window_fill=(160, 200, 230),
            window_frame=(80, 70, 55),
            door_fill=(110, 78, 48),
            door_frame=(72, 52, 30),
            door_knob=(210, 175, 70),
            floor_fill=(170, 160, 145),
            floor_dark=(125, 118, 105),
            int_wall=(100, 95, 88),
            int_dark=(62, 58, 52),
        ),
    ),
    BuildingStyle(
        name="brick",
        biome="garden",
        wall_texture="brick",
        roof_texture="tile",
        seed=20,
        palette=ColourPalette(
            roof_fill=(80, 45, 30),
            roof_edge=(58, 30, 18),
            roof_dark=(38, 18, 8),
            wall_fill=(178, 100, 72),
            wall_edge=(138, 75, 52),
            wall_dark=(90, 48, 32),
            window_fill=(150, 195, 225),
            window_frame=(65, 42, 28),
            door_fill=(90, 62, 38),
            door_frame=(60, 40, 22),
            door_knob=(200, 165, 60),
            floor_fill=(162, 148, 132),
            floor_dark=(118, 108, 96),
            int_wall=(95, 68, 52),
            int_dark=(60, 42, 30),
        ),
    ),
    BuildingStyle(
        name="log",
        biome="garden",
        wall_texture="log",
        roof_texture="tile",
        seed=30,
        palette=ColourPalette(
            roof_fill=(70, 50, 30),
            roof_edge=(50, 35, 18),
            roof_dark=(32, 22, 10),
            wall_fill=(148, 108, 72),
            wall_edge=(112, 82, 52),
            wall_dark=(78, 55, 32),
            window_fill=(168, 210, 238),
            window_frame=(72, 52, 32),
            door_fill=(95, 68, 42),
            door_frame=(65, 45, 25),
            door_knob=(205, 172, 65),
            floor_fill=(158, 138, 112),
            floor_dark=(115, 100, 80),
            int_wall=(98, 72, 48),
            int_dark=(62, 45, 28),
        ),
    ),
    BuildingStyle(
        name="ice",
        biome="ice",
        wall_texture="ice",
        roof_texture="ice",
        seed=40,
        palette=ColourPalette(
            roof_fill=(148, 188, 218),
            roof_edge=(115, 158, 192),
            roof_dark=(80, 118, 155),
            wall_fill=(198, 225, 245),
            wall_edge=(162, 195, 222),
            wall_dark=(118, 155, 188),
            window_fill=(210, 238, 255),
            window_frame=(130, 168, 200),
            door_fill=(148, 190, 220),
            door_frame=(108, 148, 182),
            door_knob=(240, 248, 255),
            floor_fill=(188, 215, 238),
            floor_dark=(148, 178, 208),
            int_wall=(130, 162, 192),
            int_dark=(90, 118, 148),
        ),
    ),
    BuildingStyle(
        name="panel",
        biome="city",
        wall_texture="panel",
        roof_texture="metal",
        seed=50,
        palette=ColourPalette(
            roof_fill=(68, 75, 85),
            roof_edge=(50, 55, 65),
            roof_dark=(32, 35, 42),
            wall_fill=(105, 112, 122),
            wall_edge=(78, 85, 95),
            wall_dark=(52, 55, 62),
            window_fill=(80, 185, 225),
            window_frame=(45, 50, 58),
            door_fill=(55, 62, 72),
            door_frame=(35, 38, 45),
            door_knob=(80, 185, 225),
            floor_fill=(128, 135, 145),
            floor_dark=(88, 95, 105),
            int_wall=(62, 68, 78),
            int_dark=(38, 42, 50),
        ),
    ),
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Public entry point                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def generate_all(tile_size: int, out_dir: Path) -> None:
    """
    Generate all building atlases, templates, and manifest.
    Called from tilegen_full.main().
    """
    tmpl_dir = out_dir / "buildings"
    tmpl_dir.mkdir(exist_ok=True)

    for style in BUILDING_STYLES:
        print(f"  Building style: {style.name}  (biome: {style.biome})")

        # Exterior atlas
        ext_atlas = build_exterior_atlas(style, tile_size)
        ext_path = out_dir / f"{style.name}_building_exterior.png"
        ext_atlas.save(ext_path)
        print(f"    → {ext_path.name}  {ext_atlas.size[0]}×{ext_atlas.size[1]}px")

        # Interior atlas
        int_atlas = build_interior_atlas(style, tile_size)
        int_path = out_dir / f"{style.name}_building_interior.png"
        int_atlas.save(int_path)
        print(f"    → {int_path.name}  {int_atlas.size[0]}×{int_atlas.size[1]}px")

        # Templates
        factories = _TEMPLATE_FACTORIES.get(style.wall_texture, [make_small_house])
        for factory in factories:
            tmpl = factory(style)
            ron_text = _template_to_ron(tmpl)
            ron_path = tmpl_dir / f"{style.name}_{tmpl.name}.ron"
            ron_path.write_text(ron_text)
            print(f"    → buildings/{style.name}_{tmpl.name}.ron")

    write_buildings_manifest(BUILDING_STYLES, tile_size, out_dir)
    print("  → buildings_manifest.ron")
