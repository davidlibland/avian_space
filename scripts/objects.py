"""
objects.py  —  Animated landscape object sprite generator
==========================================================
Generates per-biome, per-terrain object sprite sheets and a manifest
for the Rust side to look up and spawn.

Each object type has:
  - 4 variations (columns)
  - N animation frames (rows), typically 3-6

Output per biome:
  {biome}_objects.png           Combined sprite sheet
  objects_manifest.ron          Lookup: biome → terrain → [object_types]

Atlas layout per object type:
  Columns = 4 variations
  Rows = N frames (animation)

The combined sheet stacks all object types for a biome vertically.
Each entry in the manifest records the (row_offset, n_frames, n_variants,
tile_w, tile_h) so Rust can build TextureAtlasLayouts on the fly.

Usage:
  python objects.py --tile-size 16 --out-dir output
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data types                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ObjectType:
    """One kind of landscape object (e.g. 'swaying_grass')."""
    name: str
    n_frames: int           # animation frames (rows in the sheet)
    n_variants: int = 4     # visual variations (columns)
    tile_w: int = 16        # sprite width
    tile_h: int = 16        # sprite height
    # Which terrains this object can appear on (within its biome).
    terrains: list[str] = field(default_factory=list)
    # Spawning parameters
    density: float = 0.15       # base probability per tile (before fBm modulation)
    min_distance: float = 2.0   # minimum distance between instances (Poisson disk)
    max_per_tile: int = 1       # how many can share a single tile
    y_offset: float = 0.0       # pixel offset for depth-sorting (positive = visually lower)
    shy: bool = False           # if True, frame 0 = hidden; resets to frame 0 on player proximity
    # Draw function: (img, draw, variant, frame, tile_w, tile_h) -> None
    draw_fn: object = None  # assigned after definition


@dataclass
class BiomeObjects:
    """All object types for one biome."""
    biome: str
    objects: list[ObjectType]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Drawing helpers                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _clear(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _sway(frame: int, n_frames: int, amplitude: float = 1.5) -> float:
    """Sinusoidal sway offset for animation."""
    t = frame / max(n_frames - 1, 1)
    return amplitude * math.sin(t * 2 * math.pi)


def _bob(frame: int, n_frames: int, amplitude: float = 1.0) -> float:
    """Vertical bob."""
    t = frame / max(n_frames - 1, 1)
    return amplitude * math.sin(t * 2 * math.pi)


# ── Garden biome objects ─────────────────────────────────────────────────

def draw_seaweed(img: Image.Image, draw: ImageDraw.ImageDraw,
                 variant: int, frame: int, w: int, h: int) -> None:
    """Swaying seaweed strand."""
    colors = [(30, 120, 60), (25, 100, 50), (40, 130, 70), (20, 90, 45)]
    c = colors[variant % len(colors)]
    sx = _sway(frame, 4, 1.5)
    cx = w // 2
    # Draw a wavy strand from bottom up
    for y in range(h - 2, h // 4, -1):
        frac = (h - y) / h
        x = cx + int(sx * frac * 1.5) + int(math.sin(y * 0.5 + frame) * 0.8)
        draw.line([(x, y), (x, y)], fill=c, width=1)
    # Leaf tips
    tip_c = tuple(min(255, v + 30) for v in c)
    draw.ellipse([cx + int(sx) - 1, h // 4 - 1, cx + int(sx) + 1, h // 4 + 1], fill=tip_c)


def draw_fish(img: Image.Image, draw: ImageDraw.ImageDraw,
              variant: int, frame: int, w: int, h: int) -> None:
    """Small fish that jumps in an arc."""
    colors = [(180, 140, 60), (160, 80, 40), (100, 160, 180), (200, 120, 80)]
    c = colors[variant % len(colors)]
    n = 5
    t = frame / max(n - 1, 1)
    # Arc trajectory
    x = int(w * 0.2 + t * w * 0.6)
    y = int(h * 0.7 - math.sin(t * math.pi) * h * 0.5)
    # Body
    draw.ellipse([x - 2, y - 1, x + 2, y + 1], fill=c)
    # Tail
    tx = x - 3 if t < 0.5 else x + 3
    draw.line([(tx, y - 1), (tx, y + 1)], fill=c, width=1)
    # Splash at entry/exit
    if frame == 0 or frame == n - 1:
        splash = (180, 210, 240)
        draw.line([(x - 1, y + 2), (x + 1, y + 2)], fill=splash, width=1)


def draw_ripple(img: Image.Image, draw: ImageDraw.ImageDraw,
                variant: int, frame: int, w: int, h: int) -> None:
    """Expanding ripple ring."""
    n = 4
    t = frame / max(n - 1, 1)
    cx, cy = w // 2 + (variant - 2), h // 2
    r = int(2 + t * (w // 2 - 2))
    alpha = int(200 * (1.0 - t))
    c = (200, 220, 240, alpha)
    # Draw circle outline (approximate with ellipse)
    if r > 1:
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], outline=c, width=1)


def draw_grass_tuft(img: Image.Image, draw: ImageDraw.ImageDraw,
                    variant: int, frame: int, w: int, h: int) -> None:
    """Small grass tuft swaying in wind."""
    greens = [(60, 140, 50), (70, 155, 55), (50, 125, 45), (65, 150, 60)]
    c = greens[variant % len(greens)]
    sx = _sway(frame, 4, 1.0)
    cx = w // 2
    # 3-5 blades
    n_blades = 3 + variant % 3
    for i in range(n_blades):
        bx = cx + (i - n_blades // 2) * 2
        tip_x = bx + int(sx * (0.8 + i * 0.1))
        draw.line([(bx, h - 2), (tip_x, h // 3 + i)], fill=c, width=1)
    # Base
    draw.line([(cx - 3, h - 1), (cx + 3, h - 1)], fill=(50, 100, 40), width=1)


def draw_bush(img: Image.Image, draw: ImageDraw.ImageDraw,
              variant: int, frame: int, w: int, h: int) -> None:
    """Small bush with subtle sway."""
    greens = [(35, 90, 30), (40, 100, 35), (30, 80, 25), (45, 95, 40)]
    c = greens[variant % len(greens)]
    sx = _sway(frame, 4, 0.5)
    cx = w // 2 + int(sx)
    # Canopy (overlapping circles)
    r = w // 4
    draw.ellipse([cx - r - 1, h // 3, cx + r + 1, h - 3], fill=c)
    highlight = tuple(min(255, v + 25) for v in c)
    draw.ellipse([cx - r + 1, h // 3 - 1, cx + r - 1, h // 2 + 1], fill=highlight)
    # Trunk
    draw.line([(cx, h - 3), (cx, h - 1)], fill=(80, 55, 30), width=1)


def draw_rock(img: Image.Image, draw: ImageDraw.ImageDraw,
              variant: int, frame: int, w: int, h: int) -> None:
    """Static rock with flat bottom and craggy edges."""
    greys = [(130, 125, 118), (110, 108, 100), (145, 140, 130), (95, 92, 85)]
    c = greys[variant % len(greys)]
    dark = tuple(max(0, v - 30) for v in c)
    highlight = tuple(min(255, v + 25) for v in c)
    cx = w // 2
    base_y = h - 3  # flat bottom
    top_y = h // 3 + variant % 3
    rw = w // 3 + variant

    # Craggy outline: irregular polygon
    import random as _rnd
    _rnd.seed(variant * 1000 + 42)
    pts = []
    # Top arc (craggy)
    for i in range(7):
        angle = math.pi + math.pi * i / 6
        r = rw + _rnd.randint(-2, 2)
        ry = (base_y - top_y) // 2 + _rnd.randint(-1, 1)
        px = cx + int(math.cos(angle) * r)
        py = (top_y + base_y) // 2 + int(math.sin(angle) * ry)
        pts.append((max(0, min(w - 1, px)), max(0, min(h - 1, py))))
    # Flat bottom edge
    pts.append((cx + rw, base_y))
    pts.append((cx - rw, base_y))

    draw.polygon(pts, fill=c, outline=dark)
    # Highlight along top
    draw.line([(pts[2][0], pts[2][1]), (pts[4][0], pts[4][1])],
              fill=highlight, width=1)
    # Crack
    crack_c = tuple(max(0, v - 40) for v in c)
    draw.line([(cx - 1, top_y + 3), (cx + 1, base_y - 2)], fill=crack_c, width=1)


def draw_tree(img: Image.Image, draw: ImageDraw.ImageDraw,
              variant: int, frame: int, w: int, h: int) -> None:
    """Tree with swaying canopy."""
    greens = [(30, 85, 25), (25, 75, 20), (35, 95, 30), (28, 80, 22)]
    c = greens[variant % len(greens)]
    sx = _sway(frame, 4, 0.8)
    cx = w // 2
    margin = 2
    # Trunk
    draw.rectangle([cx - 1, h // 2, cx + 1, h - 2], fill=(75, 50, 28))
    # Canopy (clamped to margins)
    r = min(w // 3, w // 2 - margin - abs(int(sx)) - 1)
    top = max(margin, h // 6)
    draw.ellipse([cx - r + int(sx), top, cx + r + int(sx), h // 2 + 2], fill=c)
    highlight = tuple(min(255, v + 20) for v in c)
    draw.ellipse([cx - r + 2 + int(sx), top, cx + int(sx), h // 3], fill=highlight)


def draw_sand_critter(img: Image.Image, draw: ImageDraw.ImageDraw,
                      variant: int, frame: int, w: int, h: int) -> None:
    """Small animal popping head out of sand.
    Frame 0 = just the sand mound (hidden/shy state)."""
    n = 6
    cx = w // 2
    # Sand mound (always visible)
    sand_c = (190, 175, 130)
    draw.ellipse([cx - 4, h - 5, cx + 4, h - 1], fill=sand_c)

    if frame == 0:
        return  # hiding

    # Pop-up trajectory over frames 1–5
    t = (frame - 1) / max(n - 2, 1)
    if t < 0.3:
        pop = t / 0.3
    elif t < 0.7:
        pop = 1.0
    else:
        pop = (1.0 - t) / 0.3
    pop = max(0, min(1, pop))
    head_y = int(h * 0.7 - pop * h * 0.3)
    if pop > 0.1:
        # Head
        body_colors = [(140, 110, 70), (120, 100, 65), (150, 120, 80), (130, 105, 60)]
        bc = body_colors[variant % len(body_colors)]
        draw.ellipse([cx - 2, head_y - 2, cx + 2, head_y + 2], fill=bc)
        # Eyes
        draw.point((cx - 1, head_y - 1), fill=(20, 20, 20))
        draw.point((cx + 1, head_y - 1), fill=(20, 20, 20))


def draw_ice_chunk(img: Image.Image, draw: ImageDraw.ImageDraw,
                   variant: int, frame: int, w: int, h: int) -> None:
    """Jagged ice chunk with glint."""
    blues = [(170, 200, 230), (185, 215, 240), (155, 190, 225), (195, 225, 248)]
    c = blues[variant % len(blues)]
    dark = tuple(max(0, v - 40) for v in c)
    cx, cy = w // 2, h // 2
    # Angular ice shape
    pts = [
        (cx - 3, cy + 3), (cx - 4, cy - 1),
        (cx - 1, cy - 4 - variant % 2), (cx + 2, cy - 3),
        (cx + 4, cy), (cx + 3, cy + 3),
    ]
    draw.polygon(pts, fill=c, outline=dark)
    # Glint animation
    gx = cx - 1 + (frame % 3)
    gy = cy - 3 + (frame % 2)
    draw.point((gx, gy), fill=(255, 255, 255))


def draw_icy_plant(img: Image.Image, draw: ImageDraw.ImageDraw,
                   variant: int, frame: int, w: int, h: int) -> None:
    """Frosty crystal plant."""
    c = (140, 180, 210)
    sx = _sway(frame, 4, 0.5)
    cx = w // 2
    # Crystalline branches
    for i in range(3 + variant % 2):
        angle = -0.8 + i * 0.5 + sx * 0.1
        length = h // 3 + i * 2
        ex = cx + int(math.sin(angle) * length)
        ey = h - 2 - int(math.cos(angle) * length)
        draw.line([(cx, h - 2), (ex, ey)], fill=c, width=1)
        # Ice crystal tip
        tip = (200, 230, 250)
        draw.point((ex, ey), fill=tip)
    # Base
    draw.point((cx, h - 1), fill=(100, 140, 170))


def draw_lava_bubble(img: Image.Image, draw: ImageDraw.ImageDraw,
                     variant: int, frame: int, w: int, h: int) -> None:
    """Bubbling lava pool."""
    n = 5
    t = frame / max(n - 1, 1)
    cx = w // 2 + (variant - 2)
    # Bubble grows and pops
    r = int(1 + t * 3)
    if t < 0.8:
        # Growing bubble
        c = (235, 140 + int(t * 60), 30)
        draw.ellipse([cx - r, h // 2 - r, cx + r, h // 2 + r], fill=c)
        # Highlight
        draw.point((cx - r // 2, h // 2 - r // 2), fill=(255, 220, 100))
    else:
        # Pop — small splatter dots
        for dx, dy in [(-2, -1), (1, -2), (2, 0), (-1, 1)]:
            draw.point((cx + dx, h // 2 + dy), fill=(255, 160, 40))


def draw_lava_spurt(img: Image.Image, draw: ImageDraw.ImageDraw,
                    variant: int, frame: int, w: int, h: int) -> None:
    """Lava spurting upward."""
    n = 6
    t = frame / max(n - 1, 1)
    cx = w // 2
    # Upward spurt arc
    spurt_h = int(t * h * 0.6) if t < 0.5 else int((1.0 - t) * h * 0.6)
    for dy in range(spurt_h):
        y = h - 3 - dy
        wobble = int(math.sin(dy * 0.8 + variant) * 1.5)
        brightness = max(0, 255 - dy * 15)
        c = (brightness, max(0, brightness - 100), 20)
        draw.point((cx + wobble, y), fill=c)
    # Base glow
    draw.ellipse([cx - 2, h - 3, cx + 2, h - 1], fill=(200, 80, 20))


def draw_cactus(img: Image.Image, draw: ImageDraw.ImageDraw,
                variant: int, frame: int, w: int, h: int) -> None:
    """Desert cactus (minimal sway)."""
    greens = [(60, 110, 45), (55, 100, 40), (65, 120, 50), (50, 95, 38)]
    c = greens[variant % len(greens)]
    dark = tuple(max(0, v - 20) for v in c)
    sx = _sway(frame, 4, 0.3)
    cx = w // 2
    # Main trunk
    draw.rectangle([cx - 1, h // 3, cx + 1, h - 2], fill=c, outline=dark)
    # Arms (variant determines arm config)
    if variant % 2 == 0:
        # Left arm
        ay = h // 2
        draw.rectangle([cx - 4, ay - 1, cx - 1, ay + 1], fill=c)
        draw.rectangle([cx - 4 + int(sx * 0.3), ay - 4, cx - 3, ay], fill=c)
    if variant % 3 != 2:
        # Right arm
        ay = h // 2 - 2
        draw.rectangle([cx + 1, ay - 1, cx + 4, ay + 1], fill=c)
        draw.rectangle([cx + 3, ay - 4, cx + 4 + int(sx * 0.3), ay], fill=c)
    # Base
    draw.line([(cx - 2, h - 1), (cx + 2, h - 1)], fill=(160, 140, 100), width=1)


def draw_conifer(img: Image.Image, draw: ImageDraw.ImageDraw,
                 variant: int, frame: int, w: int, h: int) -> None:
    """Coniferous tree (triangular, slight sway)."""
    greens = [(20, 65, 18), (18, 58, 15), (25, 72, 22), (22, 60, 20)]
    c = greens[variant % len(greens)]
    sx = _sway(frame, 4, 0.5)
    cx = w // 2
    # Trunk
    draw.rectangle([cx - 1, h * 2 // 3, cx + 1, h - 1], fill=(60, 40, 22))
    # Triangular canopy layers
    for layer in range(3):
        base_y = h // 4 + layer * h // 6
        tip_y = base_y - h // 5
        half_w = (w // 4) + layer
        tip_x = cx + int(sx * 0.5)
        pts = [(tip_x, tip_y), (cx - half_w, base_y + 2), (cx + half_w, base_y + 2)]
        lc = tuple(max(0, v - layer * 8) for v in c)
        draw.polygon(pts, fill=lc)


# ── Large multi-tile objects ──────────────────────────────────────────────

def draw_tall_tree(img: Image.Image, draw: ImageDraw.ImageDraw,
                   variant: int, frame: int, w: int, h: int) -> None:
    """Large deciduous tree, 2 tiles tall. Trunk from bottom, canopy on top."""
    greens = [(35, 95, 30), (30, 85, 25), (40, 105, 35), (32, 90, 28)]
    c = greens[variant % len(greens)]
    dark = tuple(max(0, v - 20) for v in c)
    sx = _sway(frame, 4, 1.2)
    cx = w // 2
    margin = 3  # top/side margin to prevent clipping

    trunk_bottom = h - 2
    trunk_top = h * 2 // 5
    trunk_w = max(2, w // 6)

    # Trunk
    draw.rectangle([cx - trunk_w, trunk_top, cx + trunk_w, trunk_bottom],
                   fill=(75, 50, 28))
    # Trunk bark lines
    for by in range(trunk_top + 2, trunk_bottom, h // 8):
        draw.line([(cx - trunk_w + 1, by), (cx + trunk_w - 1, by)],
                  fill=(60, 38, 18), width=1)
    # Roots
    draw.line([(cx - trunk_w - 2, trunk_bottom), (cx - trunk_w, trunk_top + h // 5)],
              fill=(65, 42, 22), width=1)
    draw.line([(cx + trunk_w + 2, trunk_bottom), (cx + trunk_w, trunk_top + h // 5)],
              fill=(65, 42, 22), width=1)

    # Canopy: layered circles (keep within margins)
    canopy_cy = h // 4 + margin + int(sx * 0.3)
    canopy_rx = min(w // 3 + variant, w // 2 - margin - 2)
    canopy_ry = min(h // 4, canopy_cy - margin)
    # Shadow layer
    draw.ellipse([cx - canopy_rx - 1 + int(sx), canopy_cy - canopy_ry + 1,
                  cx + canopy_rx + 1 + int(sx), canopy_cy + canopy_ry + 1],
                 fill=dark)
    # Main canopy
    draw.ellipse([cx - canopy_rx + int(sx), canopy_cy - canopy_ry,
                  cx + canopy_rx + int(sx), canopy_cy + canopy_ry],
                 fill=c)
    # Highlight
    highlight = tuple(min(255, v + 25) for v in c)
    draw.ellipse([cx - canopy_rx // 2 + int(sx), canopy_cy - canopy_ry,
                  cx + canopy_rx // 2 + int(sx), canopy_cy - canopy_ry // 2],
                 fill=highlight)


def draw_tall_conifer(img: Image.Image, draw: ImageDraw.ImageDraw,
                      variant: int, frame: int, w: int, h: int) -> None:
    """Tall conifer, 3 tiles. Multiple triangular layers."""
    greens = [(18, 62, 15), (15, 55, 12), (22, 70, 18), (20, 58, 16)]
    c = greens[variant % len(greens)]
    sx = _sway(frame, 4, 0.8)
    cx = w // 2
    margin = 4  # top margin

    # Trunk
    draw.rectangle([cx - 1, h // 2, cx + 1, h - 2], fill=(55, 38, 18))

    # 4-6 canopy layers from bottom to top
    n_layers = 4 + variant % 3
    canopy_top = margin
    canopy_bottom = h // 2 + 2
    layer_h = (canopy_bottom - canopy_top) // n_layers
    for i in range(n_layers):
        base_y = canopy_bottom - i * layer_h
        tip_y = max(canopy_top, base_y - layer_h)
        half_w = min(w // 3 - i * 1, w // 2 - 2)
        half_w = max(2, half_w)
        tip_x = cx + int(sx * 0.4)
        pts = [(tip_x, tip_y), (cx - half_w, base_y + 2), (cx + half_w, base_y + 2)]
        lc = tuple(max(0, v - i * 5) for v in c)
        draw.polygon(pts, fill=lc)

    # Snow cap on top for some variants
    if variant % 3 == 0:
        draw.ellipse([cx - 2 + int(sx * 0.3), margin, cx + 2 + int(sx * 0.3), margin + 3],
                     fill=(220, 235, 248))


def draw_large_rock(img: Image.Image, draw: ImageDraw.ImageDraw,
                    variant: int, frame: int, w: int, h: int) -> None:
    """Large boulder with flat bottom and craggy edges."""
    greys = [(120, 115, 105), (105, 100, 92), (135, 128, 118), (90, 85, 78)]
    c = greys[variant % len(greys)]
    dark = tuple(max(0, v - 30) for v in c)
    highlight = tuple(min(255, v + 20) for v in c)
    cx = w // 2
    base_y = h - 3
    top_y = h // 5 + variant * 2

    rw = w * 2 // 5 + variant

    # Craggy irregular polygon
    import random as _rnd
    _rnd.seed(variant * 2000 + 77)
    pts = []
    for i in range(9):
        angle = math.pi + math.pi * i / 8
        r = rw + _rnd.randint(-2, 3)
        ry = (base_y - top_y) // 2 + _rnd.randint(-2, 2)
        px = cx + int(math.cos(angle) * r)
        py = (top_y + base_y) // 2 + int(math.sin(angle) * ry)
        pts.append((max(1, min(w - 2, px)), max(1, min(h - 2, py))))
    # Flat bottom
    pts.append((cx + rw, base_y))
    pts.append((cx - rw, base_y))

    draw.polygon(pts, fill=c, outline=dark)

    # Secondary bump on top
    bump_pts = [(cx - rw // 3, top_y + 2), (cx, top_y - 2 + _rnd.randint(0, 2)),
                (cx + rw // 3, top_y + 2)]
    draw.polygon(bump_pts, fill=c, outline=dark)

    # Highlight
    draw.line([(pts[3][0], pts[3][1]), (pts[5][0], pts[5][1])],
              fill=highlight, width=1)

    # Cracks
    crack_c = tuple(max(0, v - 45) for v in c)
    draw.line([(cx - 2, top_y + 5), (cx + 2, base_y - 3)], fill=crack_c, width=1)
    draw.line([(cx + 1, top_y + 8), (cx + 4, base_y - 5)], fill=crack_c, width=1)


def draw_ice_spire(img: Image.Image, draw: ImageDraw.ImageDraw,
                   variant: int, frame: int, w: int, h: int) -> None:
    """Tall ice spire, 2 tiles. Glinting crystal."""
    blues = [(160, 195, 230), (175, 210, 240), (145, 185, 225), (190, 220, 248)]
    c = blues[variant % len(blues)]
    dark = tuple(max(0, v - 40) for v in c)
    cx = w // 2
    margin = 3

    # Spire body (narrow triangle)
    base_w = min(w // 3, w // 2 - margin)
    pts = [(cx, margin), (cx - base_w, h - 2), (cx + base_w, h - 2)]
    draw.polygon(pts, fill=c, outline=dark)
    # Highlight stripe
    draw.line([(cx - 1, h // 4), (cx - 1, h * 3 // 4)],
              fill=tuple(min(255, v + 35) for v in c), width=1)
    # Animated glint
    gx = cx + (frame % 3) - 1
    gy = h // 5 + (frame * 3) % (h // 2)
    draw.point((gx, gy), fill=(255, 255, 255))
    if frame % 2 == 0:
        draw.point((gx + 1, gy), fill=(240, 248, 255))


# ── Creatures (appear across biomes) ─────────────────────────────────────

def draw_alien_peek(img: Image.Image, draw: ImageDraw.ImageDraw,
                    variant: int, frame: int, w: int, h: int) -> None:
    """Small alien creature peeking from behind a rock/plant.
    Frame 0 = just the rock (hidden/shy state)."""
    n = 5
    cx = w // 2

    # Rock/cover (always visible)
    rock_c = (100, 95, 88)
    rock_dark = (70, 65, 58)
    draw.ellipse([cx - 4, h - 6, cx + 4, h - 1], fill=rock_c, outline=rock_dark)

    if frame == 0:
        return  # hiding behind the rock

    # Peek trajectory over frames 1–4
    t = (frame - 1) / max(n - 2, 1)
    if t < 0.25:
        peek = t / 0.25
    elif t < 0.75:
        peek = 1.0
    else:
        peek = (1.0 - t) / 0.25
    peek = max(0, min(1, peek))

    if peek > 0.15:
        # Body colours vary by variant (alien species)
        body_colors = [(80, 180, 100), (180, 100, 180), (100, 150, 200), (200, 160, 60)]
        eye_colors = [(255, 220, 30), (255, 80, 80), (80, 255, 200), (255, 180, 255)]
        bc = body_colors[variant % len(body_colors)]
        ec = eye_colors[variant % len(eye_colors)]

        head_y = int(h - 7 - peek * 4)
        head_x = cx + 3  # peeking from the right side of the rock

        # Head (round)
        draw.ellipse([head_x - 2, head_y - 2, head_x + 2, head_y + 2], fill=bc)
        # Eye(s) — big alien eye
        draw.ellipse([head_x - 1, head_y - 1, head_x + 1, head_y], fill=ec)
        # Pupil
        draw.point((head_x, head_y - 1), fill=(10, 10, 10))

        # Blink on one frame
        if frame == 3 and n > 3:
            draw.line([(head_x - 1, head_y - 1), (head_x + 1, head_y - 1)], fill=bc, width=1)

        # Antennae for some variants
        if variant % 2 == 0:
            draw.line([(head_x - 1, head_y - 2), (head_x - 2, head_y - 4)],
                      fill=bc, width=1)
            draw.point((head_x - 2, head_y - 4), fill=ec)


def draw_hole_creature(img: Image.Image, draw: ImageDraw.ImageDraw,
                       variant: int, frame: int, w: int, h: int) -> None:
    """Creature popping out of a hole in the ground.
    Frame 0 = just the hole (hidden/shy state)."""
    n = 6
    cx = w // 2

    # Hole (always visible)
    draw.ellipse([cx - 3, h - 4, cx + 3, h - 1], fill=(30, 28, 25))
    draw.ellipse([cx - 3, h - 5, cx + 3, h - 3], fill=(50, 45, 38))

    if frame == 0:
        return  # hiding — just the empty hole

    # Pop sequence over frames 1–5
    t = (frame - 1) / max(n - 2, 1)
    if t < 0.2:
        pop = t / 0.2
    elif t < 0.7:
        pop = 1.0
    else:
        pop = (1.0 - t) / 0.3
    pop = max(0, min(1, pop))

    if pop > 0.1:
        body_colors = [(140, 200, 80), (200, 120, 60), (80, 160, 180), (180, 80, 140)]
        bc = body_colors[variant % len(body_colors)]
        head_y = int(h - 5 - pop * 5)

        # Body emerging from hole
        draw.rectangle([cx - 2, head_y + 2, cx + 2, h - 4], fill=bc)
        # Head
        draw.ellipse([cx - 3, head_y - 2, cx + 3, head_y + 2], fill=bc)
        # Eyes (two dots)
        draw.point((cx - 1, head_y - 1), fill=(20, 20, 20))
        draw.point((cx + 1, head_y - 1), fill=(20, 20, 20))
        # Mouth
        draw.point((cx, head_y + 1), fill=tuple(max(0, v - 40) for v in bc))

        # Look around: head shifts left/right
        if 0.4 < t < 0.6:
            look_dir = 1 if frame % 2 == 0 else -1
            draw.point((cx + look_dir * 2, head_y - 1),
                       fill=tuple(min(255, v + 30) for v in bc))


def draw_rat(img: Image.Image, draw: ImageDraw.ImageDraw,
             variant: int, frame: int, w: int, h: int) -> None:
    """Station rat scurrying along.
    Frame 0 = hidden (no rat visible)."""
    n = 5
    if frame == 0:
        return  # hiding

    t = (frame - 1) / max(n - 2, 1)
    # Scurry from left to right
    x = int(w * 0.1 + t * w * 0.7)
    y = h - 4

    body_c = (90, 80, 70)
    if variant % 2 == 1:
        body_c = (70, 65, 60)  # darker variant

    # Body
    draw.ellipse([x - 3, y - 1, x + 2, y + 1], fill=body_c)
    # Head
    draw.ellipse([x + 1, y - 2, x + 4, y + 1], fill=body_c)
    # Eye
    draw.point((x + 3, y - 1), fill=(20, 20, 20))
    # Tail
    tail_y = y + int(math.sin(frame * 1.5) * 1)
    draw.line([(x - 3, y), (x - 6, tail_y - 1)], fill=(110, 95, 80), width=1)
    # Legs (animated)
    leg_offset = 1 if frame % 2 == 0 else -1
    draw.point((x - 1, y + 2 + leg_offset), fill=body_c)
    draw.point((x + 1, y + 2 - leg_offset), fill=body_c)
    # Ears
    draw.point((x + 2, y - 2), fill=(120, 100, 85))


# Objects that appear on interior biome terrains
def draw_floor_vent(img: Image.Image, draw: ImageDraw.ImageDraw,
                    variant: int, frame: int, w: int, h: int) -> None:
    """Steam vent in the floor."""
    # Vent grate
    cx, cy = w // 2, h // 2
    draw.rectangle([cx - 3, cy - 1, cx + 3, cy + 1], fill=(60, 62, 68), outline=(40, 42, 48))
    draw.line([(cx - 2, cy), (cx + 2, cy)], fill=(45, 47, 52), width=1)
    # Steam puff (animated)
    n = 4
    t = frame / max(n - 1, 1)
    puff_y = cy - 2 - int(t * 4)
    alpha = int(150 * (1.0 - t))
    if alpha > 20:
        r = 1 + int(t * 2)
        c = (180, 185, 195, alpha)
        draw.ellipse([cx - r, puff_y - r, cx + r, puff_y + r], fill=c)


def draw_wall_pipe(img: Image.Image, draw: ImageDraw.ImageDraw,
                   variant: int, frame: int, w: int, h: int) -> None:
    """Pipes along the wall (static with drip animation)."""
    pipe_c = (75, 80, 88)
    # Horizontal pipe
    py = h // 3 + variant * 2
    draw.rectangle([0, py, w - 1, py + 2], fill=pipe_c)
    # Joint
    jx = w // 3 + variant * 3
    draw.rectangle([jx - 1, py - 1, jx + 1, py + 3], fill=(90, 95, 105))
    # Drip
    n = 4
    t = frame / max(n - 1, 1)
    drip_y = py + 3 + int(t * (h - py - 5))
    if t < 0.9:
        draw.point((jx, drip_y), fill=(100, 180, 220))


def draw_display_kiosk(img: Image.Image, draw: ImageDraw.ImageDraw,
                       variant: int, frame: int, w: int, h: int) -> None:
    """Small display terminal or kiosk with flickering screen."""
    body_c = (55, 58, 65)
    cx = w // 2
    # Stand
    draw.rectangle([cx - 1, h * 2 // 3, cx + 1, h - 1], fill=(50, 52, 58))
    # Screen body
    sw, sh = w // 2, h // 3
    sx = cx - sw // 2
    sy = h // 4
    draw.rectangle([sx, sy, sx + sw, sy + sh], fill=body_c, outline=(40, 42, 48))
    # Screen content (flickers)
    screen_colors = [(50, 200, 120), (60, 180, 220), (200, 180, 50), (180, 60, 60)]
    sc = screen_colors[(variant + frame) % len(screen_colors)]
    draw.rectangle([sx + 1, sy + 1, sx + sw - 1, sy + sh - 1], fill=sc)
    # Scan line
    scan_y = sy + 1 + (frame * 2) % max(1, sh - 2)
    draw.line([(sx + 1, scan_y), (sx + sw - 1, scan_y)],
              fill=tuple(min(255, v + 50) for v in sc), width=1)


def draw_machine(img: Image.Image, draw: ImageDraw.ImageDraw,
                 variant: int, frame: int, w: int, h: int) -> None:
    """Small machine with spinning element."""
    body_c = (65, 68, 75)
    dark = (42, 45, 52)
    cx, cy = w // 2, h // 2
    # Machine body
    draw.rectangle([cx - 4, cy - 2, cx + 4, cy + 3], fill=body_c, outline=dark)
    # Spinning disc/gear
    angle = frame * (math.pi / 2)
    r = 2
    gx = cx + int(math.cos(angle) * r)
    gy = cy + int(math.sin(angle) * r)
    indicator = [(180, 60, 40), (60, 180, 60), (60, 60, 180), (180, 180, 40)]
    draw.ellipse([gx - 1, gy - 1, gx + 1, gy + 1],
                 fill=indicator[variant % len(indicator)])
    # Status LED
    led_colors = [(255, 50, 30), (50, 255, 50), (50, 150, 255), (255, 200, 30)]
    led = led_colors[(variant + frame) % len(led_colors)]
    draw.point((cx + 3, cy - 1), fill=led)


def draw_sewage(img: Image.Image, draw: ImageDraw.ImageDraw,
                variant: int, frame: int, w: int, h: int) -> None:
    """Fluorescent green sewage puddle with bubbles."""
    green = (40, 200, 60)
    dark_green = (25, 140, 35)
    cx, cy = w // 2 + (variant - 2), h * 2 // 3
    # Puddle
    rw = 4 + variant % 2
    rh = 2 + variant % 2
    draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=dark_green)
    draw.ellipse([cx - rw + 1, cy - rh, cx + rw - 1, cy], fill=green)
    # Bubbles (animated)
    n = 4
    t = frame / max(n - 1, 1)
    bx = cx - 1 + (frame % 3)
    by = cy - int(t * 3)
    br = 1 if t < 0.7 else 0
    if br > 0:
        draw.ellipse([bx - br, by - br, bx + br, by + br],
                     outline=(80, 255, 100), width=1)
    # Glow
    glow = (60, 220, 80, int(100 * (1.0 - t)))
    draw.ellipse([cx - 2, cy - 1, cx + 2, cy + 1], fill=glow)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Biome object definitions                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GARDEN_OBJECTS = BiomeObjects(
    biome="garden",
    objects=[
        ObjectType("seaweed", n_frames=4, terrains=["water"],
                   density=0.20, min_distance=1.5, max_per_tile=2, y_offset=4.0,
                   draw_fn=draw_seaweed),
        ObjectType("fish", n_frames=5, terrains=["water"],
                   density=0.06, min_distance=4.0, max_per_tile=1, y_offset=0.0,
                   draw_fn=draw_fish),
        ObjectType("ripple", n_frames=4, terrains=["water"],
                   density=0.10, min_distance=2.0, max_per_tile=2, y_offset=0.0,
                   draw_fn=draw_ripple),
        ObjectType("grass_tuft", n_frames=4, terrains=["sand", "grass"],
                   density=0.40, min_distance=0.8, max_per_tile=3, y_offset=6.0,
                   draw_fn=draw_grass_tuft),
        ObjectType("bush", n_frames=4, terrains=["grass", "forest"],
                   density=0.18, min_distance=2.0, max_per_tile=1, y_offset=4.0,
                   draw_fn=draw_bush),
        ObjectType("rock", n_frames=1, terrains=["sand", "grass", "mountain"],
                   density=0.10, min_distance=2.5, max_per_tile=2, y_offset=6.0,
                   draw_fn=draw_rock),
        ObjectType("tree", n_frames=4, terrains=["grass", "forest"],
                   density=0.18, min_distance=2.5, max_per_tile=1, y_offset=8.0,
                   draw_fn=draw_tree),
        ObjectType("conifer", n_frames=4, terrains=["forest", "mountain"],
                   density=0.18, min_distance=2.5, max_per_tile=1, y_offset=8.0,
                   draw_fn=draw_conifer),
        ObjectType("tall_tree", n_frames=4, tile_h=32,
                   terrains=["grass", "forest"],
                   density=0.12, min_distance=3.0, max_per_tile=1, y_offset=14.0,
                   draw_fn=draw_tall_tree),
        ObjectType("tall_conifer", n_frames=4, tile_h=48,
                   terrains=["forest", "mountain"],
                   density=0.08, min_distance=4.0, max_per_tile=1, y_offset=22.0,
                   draw_fn=draw_tall_conifer),
        ObjectType("large_rock", n_frames=1, tile_h=32,
                   terrains=["grass", "mountain", "sand"],
                   density=0.04, min_distance=5.0, max_per_tile=1, y_offset=12.0,
                   draw_fn=draw_large_rock),
        ObjectType("alien_peek", n_frames=5, terrains=["grass", "forest", "sand"],
                   density=0.03, min_distance=8.0, max_per_tile=1, y_offset=-4.0,
                   shy=True, draw_fn=draw_alien_peek),
        ObjectType("hole_creature", n_frames=6, terrains=["grass", "sand"],
                   density=0.03, min_distance=8.0, max_per_tile=1, y_offset=-6.0,
                   shy=True, draw_fn=draw_hole_creature),
    ],
)

ICE_OBJECTS = BiomeObjects(
    biome="ice",
    objects=[
        ObjectType("ice_chunk", n_frames=3, terrains=["deep_ice", "ice", "snow"],
                   density=0.12, min_distance=2.5, max_per_tile=2, y_offset=4.0,
                   draw_fn=draw_ice_chunk),
        ObjectType("icy_plant", n_frames=4, terrains=["snow", "ice"],
                   density=0.10, min_distance=2.0, max_per_tile=2, y_offset=4.0,
                   draw_fn=draw_icy_plant),
        ObjectType("rock", n_frames=1, terrains=["ice_rock", "snow"],
                   density=0.08, min_distance=3.0, max_per_tile=2, y_offset=6.0,
                   draw_fn=draw_rock),
        ObjectType("ice_spire", n_frames=3, tile_h=32,
                   terrains=["deep_ice", "ice", "ice_rock"],
                   density=0.05, min_distance=4.0, max_per_tile=1, y_offset=14.0,
                   draw_fn=draw_ice_spire),
        ObjectType("large_rock", n_frames=1, tile_h=32,
                   terrains=["ice_rock", "snow"],
                   density=0.04, min_distance=5.0, max_per_tile=1, y_offset=12.0,
                   draw_fn=draw_large_rock),
        ObjectType("alien_peek", n_frames=5, terrains=["snow", "ice"],
                   density=0.02, min_distance=10.0, max_per_tile=1, y_offset=-4.0,
                   shy=True, draw_fn=draw_alien_peek),
    ],
)

ROCKY_OBJECTS = BiomeObjects(
    biome="rocky",
    objects=[
        ObjectType("lava_bubble", n_frames=5, terrains=["lava"],
                   density=0.18, min_distance=2.0, max_per_tile=2, y_offset=2.0,
                   draw_fn=draw_lava_bubble),
        ObjectType("lava_spurt", n_frames=6, terrains=["lava"],
                   density=0.08, min_distance=3.5, max_per_tile=1, y_offset=0.0,
                   draw_fn=draw_lava_spurt),
        ObjectType("rock", n_frames=1, terrains=["basalt", "rock", "dust"],
                   density=0.10, min_distance=2.5, max_per_tile=2, y_offset=6.0,
                   draw_fn=draw_rock),
        ObjectType("large_rock", n_frames=1, tile_h=32,
                   terrains=["basalt", "rock", "cliff"],
                   density=0.05, min_distance=4.0, max_per_tile=1, y_offset=12.0,
                   draw_fn=draw_large_rock),
        ObjectType("bush", n_frames=4, terrains=["dust", "rock"],
                   density=0.08, min_distance=3.0, max_per_tile=1, y_offset=4.0,
                   draw_fn=draw_bush),
        ObjectType("hole_creature", n_frames=6, terrains=["basalt", "dust"],
                   density=0.03, min_distance=8.0, max_per_tile=1, y_offset=-6.0,
                   shy=True, draw_fn=draw_hole_creature),
    ],
)

DESERT_OBJECTS = BiomeObjects(
    biome="desert",
    objects=[
        ObjectType("sand_critter", n_frames=6, terrains=["dunes", "hard_sand"],
                   density=0.04, min_distance=6.0, max_per_tile=1, y_offset=-4.0,
                   shy=True, draw_fn=draw_sand_critter),
        ObjectType("cactus", n_frames=4, terrains=["dunes", "hard_sand"],
                   density=0.08, min_distance=4.0, max_per_tile=1, y_offset=6.0,
                   draw_fn=draw_cactus),
        ObjectType("grass_tuft", n_frames=4, terrains=["hard_sand", "dunes"],
                   density=0.25, min_distance=1.0, max_per_tile=3, y_offset=6.0,
                   draw_fn=draw_grass_tuft),
        ObjectType("rock", n_frames=1, terrains=["hard_sand", "sandstone"],
                   density=0.08, min_distance=3.0, max_per_tile=2, y_offset=6.0,
                   draw_fn=draw_rock),
        ObjectType("large_rock", n_frames=1, tile_h=32,
                   terrains=["sandstone", "mesa"],
                   density=0.04, min_distance=5.0, max_per_tile=1, y_offset=12.0,
                   draw_fn=draw_large_rock),
        ObjectType("alien_peek", n_frames=5, terrains=["dunes", "hard_sand"],
                   density=0.02, min_distance=10.0, max_per_tile=1, y_offset=-4.0,
                   shy=True, draw_fn=draw_alien_peek),
    ],
)

INTERIOR_OBJECTS = BiomeObjects(
    biome="interior",
    objects=[
        ObjectType("floor_vent", n_frames=4, terrains=["floor", "plating", "grate"],
                   density=0.06, min_distance=4.0, max_per_tile=1, y_offset=2.0,
                   draw_fn=draw_floor_vent),
        ObjectType("wall_pipe", n_frames=4, terrains=["plating", "grate"],
                   density=0.08, min_distance=3.0, max_per_tile=1, y_offset=0.0,
                   draw_fn=draw_wall_pipe),
        ObjectType("display_kiosk", n_frames=4, terrains=["floor", "plating"],
                   density=0.05, min_distance=5.0, max_per_tile=1, y_offset=4.0,
                   draw_fn=draw_display_kiosk),
        ObjectType("machine", n_frames=4, terrains=["plating", "grate"],
                   density=0.06, min_distance=4.0, max_per_tile=1, y_offset=4.0,
                   draw_fn=draw_machine),
        ObjectType("sewage", n_frames=4, terrains=["grate", "conduit"],
                   density=0.08, min_distance=3.0, max_per_tile=2, y_offset=2.0,
                   draw_fn=draw_sewage),
        ObjectType("rat", n_frames=5, terrains=["floor", "plating", "grate"],
                   density=0.03, min_distance=8.0, max_per_tile=1, y_offset=6.0,
                   shy=True, draw_fn=draw_rat),
    ],
)

ALL_BIOME_OBJECTS = [GARDEN_OBJECTS, ICE_OBJECTS, ROCKY_OBJECTS,
                     DESERT_OBJECTS, INTERIOR_OBJECTS]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Atlas assembly                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_objects_atlas(biome_objs: BiomeObjects,
                        tile_size: int) -> tuple[Image.Image, list[dict]]:
    """
    Build a combined sprite sheet for all objects in a biome.

    Each object uses its own tile_w × tile_h (which may differ from the
    base tile_size for multi-tile objects like tall trees).

    Returns (atlas_image, metadata_list).
    """
    S = tile_size
    objects = biome_objs.objects
    if not objects:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), []

    # Each object occupies (n_variants * obj.tile_w) wide, (n_frames * obj.tile_h) tall.
    # Atlas width = max across all objects of (n_variants * tile_w).
    atlas_w = max(o.n_variants * o.tile_w for o in objects)
    total_h = sum(o.n_frames * o.tile_h for o in objects)
    atlas = Image.new("RGBA", (atlas_w, total_h), (0, 0, 0, 0))

    meta = []
    y_offset_px = 0  # current pixel y in the atlas

    for obj in objects:
        tw, th = obj.tile_w, obj.tile_h
        for frame in range(obj.n_frames):
            for var in range(obj.n_variants):
                px = var * tw
                py = y_offset_px + frame * th
                tile_img, tile_draw = _clear(tw, th)
                if obj.draw_fn:
                    obj.draw_fn(tile_img, tile_draw, var, frame, tw, th)
                atlas.paste(tile_img, (px, py), tile_img)

        meta.append({
            "name": obj.name,
            "y_px": y_offset_px,            # pixel offset in the atlas
            "n_frames": obj.n_frames,
            "n_variants": obj.n_variants,
            "tile_w": tw,
            "tile_h": th,
            "terrains": obj.terrains,
            "density": obj.density,
            "min_distance": obj.min_distance,
            "max_per_tile": obj.max_per_tile,
            "y_offset": obj.y_offset,
            "shy": obj.shy,
        })
        y_offset_px += obj.n_frames * th

    return atlas, meta


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RON manifest                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def write_objects_manifest(all_meta: dict[str, list[dict]],
                           tile_size: int, out_dir: Path) -> None:
    """Write objects_manifest.ron with biome → terrain → object lookup."""
    lines = [
        "// objects_manifest.ron — auto-generated by objects.py\n",
        "(\n",
        f"    tile_size: {tile_size},\n",
        "    biomes: {\n",
    ]
    for biome_name, obj_list in all_meta.items():
        lines.append(f'        "{biome_name}": (\n')
        lines.append(f'            atlas: "{biome_name}_objects.png",\n')
        lines.append("            objects: [\n")
        for obj in obj_list:
            terrains_ron = "[" + ", ".join(f'"{t}"' for t in obj["terrains"]) + "]"
            lines.append("                (\n")
            lines.append(f'                    name: "{obj["name"]}",\n')
            lines.append(f"                    y_px: {obj['y_px']},\n")
            lines.append(f"                    n_frames: {obj['n_frames']},\n")
            lines.append(f"                    n_variants: {obj['n_variants']},\n")
            lines.append(f"                    tile_w: {obj['tile_w']},\n")
            lines.append(f"                    tile_h: {obj['tile_h']},\n")
            lines.append(f"                    terrains: {terrains_ron},\n")
            lines.append(f"                    density: {obj['density']},\n")
            lines.append(f"                    min_distance: {obj['min_distance']},\n")
            lines.append(f"                    max_per_tile: {obj['max_per_tile']},\n")
            lines.append(f"                    y_offset: {obj['y_offset']},\n")
            lines.append(f"                    shy: {'true' if obj['shy'] else 'false'},\n")
            lines.append("                ),\n")
        lines.append("            ],\n")
        lines.append("        ),\n")
    lines.append("    },\n")
    lines.append(")\n")
    (out_dir / "objects_manifest.ron").write_text("".join(lines))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_all(tile_size: int, out_dir: Path) -> None:
    """Generate all object atlases and manifest."""
    all_meta: dict[str, list[dict]] = {}

    for biome_objs in ALL_BIOME_OBJECTS:
        name = biome_objs.biome
        print(f"  Objects: {name}  ({len(biome_objs.objects)} types)")

        atlas, meta = build_objects_atlas(biome_objs, tile_size)
        atlas_path = out_dir / f"{name}_objects.png"
        atlas.save(atlas_path)
        print(f"    → {atlas_path.name}  {atlas.size[0]}×{atlas.size[1]}px")

        all_meta[name] = meta

    write_objects_manifest(all_meta, tile_size, out_dir)
    print("  → objects_manifest.ron")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate animated landscape object sprites"
    )
    parser.add_argument("--tile-size", type=int, default=16,
                        help="Sprite size in pixels (default 16)")
    parser.add_argument("--out-dir", type=str, default="output")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    print(f"Tile size : {args.tile_size}px  →  {out.resolve()}\n")
    generate_all(args.tile_size, out)
    print("\nDone.")


if __name__ == "__main__":
    main()
