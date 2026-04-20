"""
tilegen.py  —  Blob-47 tile atlas generator
================================================================
Outputs (per biome, into --out-dir):

  <biome>_atlas.png          Visual tile sheet
                             Layout: rows = terrain types, cols = 48
                             TileTextureIndex = terrain_row * 48 + blob47_lut[reduce_mask]

  blob47_lut.ron             The 256-entry mask->column LUT as a RON asset
  world_manifest.ron         All biome metadata in one file for Bevy AssetServer

Terrain map generation and collision derivation are handled at runtime
in Rust (see src/fbm.rs and src/surface.rs).

Usage:
  pip install numpy pillow opensimplex
  python tilegen.py [--tile-size 32] [--out-dir output]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opensimplex
from PIL import Image, ImageDraw, ImageFilter

# ── output directory ────────────────────────────────────────────────────────
ATLAS_COLS = 48  # 47 blob tiles + 1 padding column


# ── collision codes (matched in Bevy via CollisionType enum) ─────────────────
class CC:
    WALKABLE = 0
    SOLID = 1
    SLOW = 2  # e.g. shallow water, deep snow  — slows movement
    DAMAGING = 3  # e.g. lava                      — deals tick damage
    TRIGGER = 4  # e.g. warp zone, interaction     — fires an event


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Blob-47 LUT  —  Boris The Brave's standard ordering                    ║
# ║  https://www.boristhebrave.com/2013/07/14/tileset-roundup/               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TL, T, TR = 1, 2, 4
L, R = 8, 16
BL, B, BR = 32, 64, 128

# Aliases used by render_transition_tile (cardinal directions)
N, E, S, W = T, R, B, L


def reduce_to_47(mask: int) -> int:
    if not (mask & L) or not (mask & T):
        mask &= ~TL
    if not (mask & R) or not (mask & T):
        mask &= ~TR
    if not (mask & L) or not (mask & B):
        mask &= ~BL
    if not (mask & R) or not (mask & B):
        mask &= ~BR
    return mask


# The 47 valid reduced masks, sorted numerically.
# Atlas column = index in this list.
_BLOB47_SORTED: list[int] = sorted({
    reduce_to_47(m) for m in range(256)
})
assert len(_BLOB47_SORTED) == 47

# 256-entry LUT: reduced_mask → atlas column (255 = unmapped)
BLOB47_LUT: list[int] = [255] * 256
for _col, _mask in enumerate(_BLOB47_SORTED):
    BLOB47_LUT[_mask] = _col
assert sum(1 for v in BLOB47_LUT if v != 255) == 47

# Verify against Boris The Brave's canonical pick_tile dict
_PICK_TILE = {
    0: 0, 2: 1, 8: 2, 10: 3, 11: 4, 16: 5, 18: 6,
    22: 7, 24: 8, 26: 9, 27: 10, 30: 11, 31: 12, 64: 13,
    66: 14, 72: 15, 74: 16, 75: 17, 80: 18, 82: 19, 86: 20,
    88: 21, 90: 22, 91: 23, 94: 24, 95: 25, 104: 26, 106: 27,
    107: 28, 120: 29, 122: 30, 123: 31, 126: 32, 127: 33,
    208: 34, 210: 35, 214: 36, 216: 37, 218: 38, 219: 39,
    222: 40, 223: 41, 248: 42, 250: 43, 251: 44, 254: 45, 255: 46,
}
for _mask, _expected_col in _PICK_TILE.items():
    assert BLOB47_LUT[_mask] == _expected_col, (
        f"LUT mismatch for mask {_mask}: got {BLOB47_LUT[_mask]}, expected {_expected_col}"
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data classes                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class TerrainSpec:
    name: str
    # Visual
    base: tuple[int, int, int]
    accent: tuple[int, int, int]
    dark: tuple[int, int, int]
    roughness: float = 0.06
    macro_scale: float = 3.0
    detail_scale: float = 8.0
    micro_scale: float = 16.0
    macro_w: float = 0.30
    detail_w: float = 0.50
    micro_w: float = 0.20
    specular: float = 0.0
    cracks: bool = False
    grid: bool = False
    grid_spacing: int = 8
    # Game logic
    threshold: float = 0.5  # preserved in manifest for Rust-side terrain assignment
    collision: int = CC.WALKABLE
    movement_cost: float = 1.0  # multiplier (1.0 = normal, 2.0 = half speed)
    damage_per_sec: float = 0.0  # HP/s when standing on tile


@dataclass
class BiomeConfig:
    name: str
    terrains: list[TerrainSpec]
    seed: int = 42  # used for tile texture rendering
    soft_boundaries: bool = True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Tile texture rendering                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _tiling_noise(size: int, scale: float, seed: int) -> np.ndarray:
    """Seamless noise via torus-wrapped OpenSimplex sampling."""
    opensimplex.seed(seed)
    arr = np.array(
        [
            [
                opensimplex.noise2((gx / size) * scale, (gy / size) * scale)
                for gx in range(size)
            ]
            for gy in range(size)
        ],
        dtype=np.float32,
    )
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def render_base_tile(spec: TerrainSpec, size: int, seed: int) -> np.ndarray:
    """Return (size, size, 3) uint8 solid tile for this terrain."""
    tile_h = tile_w = size
    macro = _tiling_noise(size, spec.macro_scale, seed)
    detail = _tiling_noise(size, spec.detail_scale, seed + 100)
    micro = _tiling_noise(size, spec.micro_scale, seed + 200)
    hmap = macro * spec.macro_w + detail * spec.detail_w + micro * spec.micro_w
    lo, hi = hmap.min(), hmap.max()
    hmap = (hmap - lo) / (hi - lo + 1e-9)

    base = np.array(spec.base, dtype=np.float32) / 255.0
    accent = np.array(spec.accent, dtype=np.float32) / 255.0
    dark = np.array(spec.dark, dtype=np.float32) / 255.0

    img = np.zeros((tile_h, tile_w, 3), dtype=np.float32)
    for ch in range(3):
        img[:, :, ch] = np.where(
            hmap < 0.5,
            dark[ch] + (base[ch] - dark[ch]) * (hmap * 2.0),
            base[ch] + (accent[ch] - base[ch]) * ((hmap - 0.5) * 2.0),
        )

    if spec.specular > 0.0:
        dx = np.gradient(hmap, axis=1)
        dy = np.gradient(hmap, axis=0)
        nz = 1.0 / np.sqrt(dx**2 + dy**2 + 1.0)
        lv = np.array([0.6, -0.8, 1.0], dtype=np.float32)
        lv /= np.linalg.norm(lv)
        diffuse = np.clip(-dx * nz * lv[0] + -dy * nz * lv[1] + nz * lv[2], 0.0, 1.0)
        for ch in range(3):
            img[:, :, ch] = np.clip(
                img[:, :, ch] * (0.7 + 0.3 * diffuse) + spec.specular * diffuse**4,
                0.0,
                1.0,
            )

    rng = np.random.default_rng(seed + 999)
    img = np.clip(
        img + rng.normal(0, spec.roughness, (tile_h, tile_w, 3)).astype(np.float32),
        0.0,
        1.0,
    )
    pil = Image.fromarray((img * 255).astype(np.uint8), "RGB")

    if spec.cracks:
        opensimplex.seed(seed + 300)
        cn = np.array(
            [
                [
                    opensimplex.noise2(gx / tile_w * 8, gy / tile_h * 8)
                    for gx in range(tile_w)
                ]
                for gy in range(tile_h)
            ],
            dtype=np.float32,
        )
        ca = np.array(pil, dtype=np.float32)
        dc = np.array(spec.dark, dtype=np.float32) * 0.45
        for ch in range(3):
            ca[:, :, ch] = np.where(cn < -0.62, dc[ch], ca[:, :, ch])
        pil = Image.fromarray(ca.astype(np.uint8))

    if spec.grid:
        draw = ImageDraw.Draw(pil)
        gc = tuple(max(0, c - 35) for c in spec.dark)
        sp = spec.grid_spacing
        for x in range(0, tile_w, sp):
            draw.line([(x, 0), (x, tile_h - 1)], fill=gc, width=1)
        for y in range(0, tile_h, sp):
            draw.line([(0, y), (tile_w - 1, y)], fill=gc, width=1)

    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.35))
    return np.array(pil, dtype=np.uint8)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def render_transition_tile(
    base_arr: np.ndarray,
    neighbours: dict[int, np.ndarray],
    mask: int,
    size: int,
    hard_edge: bool = False,
) -> np.ndarray:
    """
    Blend base_arr with neighbour textures at unshared edges.

    When a cardinal bit is CLEAR in *mask*, the neighbour in that direction
    is a different terrain, so we blend from the base texture (tile centre)
    to the neighbour texture (tile edge).

    Transition width = near half of tile for a consistent look.
    """
    # Use tile_h / tile_w to avoid shadowing the global W = L = 8.
    tile_h = tile_w = size
    result = base_arr.astype(np.float32)
    ys, xs = np.mgrid[0:tile_h, 0:tile_w]

    if hard_edge:
        # Interior biome: darkened 2px border instead of smooth blend
        for bit, (sy, sx, ey, ex) in [
            (N, (0, 0, 2, tile_w)),
            (S, (tile_h - 2, 0, tile_h, tile_w)),
            (E, (0, tile_w - 2, tile_h, tile_w)),
            (W, (0, 0, tile_h, 2)),
        ]:
            if not (mask & bit) and bit in neighbours:
                nbr = neighbours[bit].astype(np.float32)
                avg = (result[sy:ey, sx:ex] + nbr[sy:ey, sx:ex]) * 0.5
                result[sy:ey, sx:ex] = avg * 0.65
        # Corner darkening for diagonals
        for diag, card1, card2, (sy, sx, ey, ex) in [
            (TL, T, L, (0, 0, 2, 2)),
            (TR, T, R, (0, tile_w - 2, 2, tile_w)),
            (BL, B, L, (tile_h - 2, 0, tile_h, 2)),
            (BR, B, R, (tile_h - 2, tile_w - 2, tile_h, tile_w)),
        ]:
            if (mask & diag) or not (mask & card1) or not (mask & card2):
                continue
            if diag not in neighbours:
                continue
            nbr = neighbours[diag].astype(np.float32)
            avg = (result[sy:ey, sx:ex] + nbr[sy:ey, sx:ex]) * 0.5
            result[sy:ey, sx:ex] = avg * 0.65
        return np.clip(result, 0, 255).astype(np.uint8)

    # Organic blend (natural biomes).
    #
    # Step 1: Cardinal edges — for each cardinal where the neighbour is
    # different (bit clear), blend from base (centre) to neighbour (edge).
    #
    # direction=+1 → blend at top / left  (small ys / xs values)
    # direction=-1 → blend at bottom / right (large ys / xs values)
    for bit, axis, direction in [
        (N, "y", +1),   # top neighbour different → blend at top edge
        (S, "y", -1),   # bottom neighbour different → blend at bottom edge
        (E, "x", -1),   # right neighbour different → blend at right edge
        (W, "x", +1),   # left neighbour different → blend at left edge
    ]:
        if (mask & bit) or (bit not in neighbours):
            continue
        nbr = neighbours[bit].astype(np.float32)
        if axis == "y":
            coords = ys / tile_h if direction > 0 else (tile_h - 1 - ys) / tile_h
        else:
            coords = xs / tile_w if direction > 0 else (tile_w - 1 - xs) / tile_w
        t = 1.0 - np.clip(coords * 2.0, 0.0, 1.0)
        blend = _smoothstep(t)[:, :, np.newaxis]
        result = result * (1.0 - blend) + nbr * blend

    # Step 2: Diagonal corners — when a diagonal bit is clear but both
    # adjacent cardinals are set, there's a concave corner notch.  Blend
    # a quarter-circle patch in that corner toward the lower terrain.
    corner_radius = 0.7  # fraction of half-tile
    for diag, card1, card2, cx, cy in [
        (TL, T, L, 0, 0),                          # top-left
        (TR, T, R, tile_w - 1, 0),                  # top-right
        (BL, B, L, 0, tile_h - 1),                  # bottom-left
        (BR, B, R, tile_w - 1, tile_h - 1),          # bottom-right
    ]:
        if (mask & diag) or not (mask & card1) or not (mask & card2):
            continue
        if diag not in neighbours:
            continue
        nbr = neighbours[diag].astype(np.float32)
        dx = np.abs(xs - cx).astype(np.float32) / (tile_w * 0.5)
        dy = np.abs(ys - cy).astype(np.float32) / (tile_h * 0.5)
        dist = np.sqrt(dx ** 2 + dy ** 2) / corner_radius
        t = 1.0 - np.clip(dist, 0.0, 1.0)
        blend = _smoothstep(t)[:, :, np.newaxis]
        result = result * (1.0 - blend) + nbr * blend

    return np.clip(result, 0, 255).astype(np.uint8)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Atlas assembly                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def build_atlas(biome: BiomeConfig, tile_size: int) -> Image.Image:
    """
    Return a PIL Image: (ATLAS_COLS * tile_size) wide x (n_terrains * tile_size) tall.
    Each row = one terrain; each column = one blob-47 tile variant.
    Column 47 = padding (copy of interior tile, col 46).
    """
    n = len(biome.terrains)
    atlas = Image.new("RGBA", (ATLAS_COLS * tile_size, n * tile_size), (0, 0, 0, 0))

    base_tiles = [
        render_base_tile(spec, tile_size, seed=i * 500 + biome.seed)
        for i, spec in enumerate(biome.terrains)
    ]

    hard = not biome.soft_boundaries

    for row, spec in enumerate(biome.terrains):
        base = base_tiles[row]

        for col, mask_r in enumerate(_BLOB47_SORTED):
            # For each unshared direction (cardinal or diagonal): provide
            # the lower-terrain texture for blending.
            nbr: dict[int, np.ndarray] = {}
            nbr_row = (row - 1) if row > 0 else 1
            nbr_tile = base_tiles[nbr_row % n]
            for bit in (N, E, S, W, TL, TR, BL, BR):
                if not (mask_r & bit):
                    nbr[bit] = nbr_tile

            tile_arr = render_transition_tile(base, nbr, mask_r, tile_size, hard)
            tile_img = Image.fromarray(tile_arr, "RGB").convert("RGBA")
            atlas.paste(tile_img, (col * tile_size, row * tile_size))

        # Column 47 padding: copy interior tile (col 46 = mask 255 = all neighbours same)
        interior = Image.fromarray(base, "RGB").convert("RGBA")
        atlas.paste(interior, (47 * tile_size, row * tile_size))

    return atlas


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RON serialisation                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _ron_comment(text: str) -> str:
    return f"// {text}\n"


def write_blob47_lut(out_dir: Path) -> None:
    lines = [
        _ron_comment("blob47_lut.ron — auto-generated by tilegen.py"),
        _ron_comment("Boris The Brave's standard ordering"),
        _ron_comment("Usage: lut[reduce_to_47(bitmask) as usize] -> atlas column"),
        _ron_comment("255 = unmapped (should never occur after reduce_to_47)"),
        "(\n",
        "    atlas_cols: 48,\n",
        f"    lut: {BLOB47_LUT!r},\n",
        ")\n",
    ]
    (out_dir / "blob47_lut.ron").write_text("".join(lines))


def write_world_manifest(
    biomes: list[BiomeConfig], tile_size: int, out_dir: Path
) -> None:
    lines = [
        _ron_comment("world_manifest.ron — auto-generated by tilegen.py"),
        _ron_comment("Deserialize into WorldManifest in Bevy"),
        "(\n",
        f"    tile_size: {tile_size},\n",
        f"    atlas_cols: {ATLAS_COLS},\n",
        "    biomes: {\n",
    ]
    for biome in biomes:
        lines += [
            f'        "{biome.name}": (\n',
            f'            atlas: "{biome.name}_atlas.png",\n',
            f"            soft_boundaries: {str(biome.soft_boundaries).lower()},\n",
            "            terrains: [\n",
        ]
        for i, t in enumerate(biome.terrains):
            lines += [
                "                (\n",
                f'                    name: "{t.name}",\n',
                f"                    row: {i},\n",
                f"                    threshold: {t.threshold},\n",
                f"                    collision: {t.collision},\n",
                f"                    movement_cost: {t.movement_cost},\n",
                f"                    damage_per_sec: {t.damage_per_sec},\n",
                "                ),\n",
            ]
        lines += [
            "            ],\n",
            "        ),\n",
        ]
    lines += ["    },\n", ")\n"]
    (out_dir / "world_manifest.ron").write_text("".join(lines))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Biome definitions                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GARDEN = BiomeConfig(
    name="garden",
    seed=13,
    soft_boundaries=True,
    terrains=[
        TerrainSpec(
            "water",
            (55, 118, 182),
            (100, 162, 215),
            (28, 72, 130),
            roughness=0.04,
            macro_scale=2.0,
            detail_scale=5.0,
            micro_scale=12.0,
            macro_w=0.50,
            detail_w=0.35,
            micro_w=0.15,
            specular=0.08,
            threshold=0.20,
            collision=CC.SLOW,
            movement_cost=2.0,
        ),
        TerrainSpec(
            "sand",
            (205, 188, 138),
            (228, 212, 162),
            (158, 140, 92),
            roughness=0.05,
            macro_scale=3.5,
            detail_scale=9.0,
            micro_scale=18.0,
            macro_w=0.25,
            detail_w=0.50,
            micro_w=0.25,
            threshold=0.30,
            collision=CC.WALKABLE,
            movement_cost=1.2,
        ),
        TerrainSpec(
            "grass",
            (72, 145, 72),
            (108, 178, 82),
            (38, 90, 38),
            roughness=0.05,
            macro_scale=3.0,
            detail_scale=8.0,
            micro_scale=16.0,
            macro_w=0.30,
            detail_w=0.50,
            micro_w=0.20,
            threshold=0.62,
            collision=CC.WALKABLE,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "forest",
            (30, 88, 38),
            (55, 118, 52),
            (15, 52, 20),
            roughness=0.06,
            macro_scale=2.5,
            detail_scale=6.0,
            micro_scale=14.0,
            macro_w=0.40,
            detail_w=0.40,
            micro_w=0.20,
            threshold=0.82,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "mountain",
            (122, 114, 104),
            (158, 150, 138),
            (74, 68, 60),
            roughness=0.07,
            macro_scale=2.8,
            detail_scale=7.0,
            micro_scale=15.0,
            macro_w=0.40,
            detail_w=0.40,
            micro_w=0.20,
            specular=0.04,
            cracks=True,
            threshold=1.00,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
    ],
)

ICE = BiomeConfig(
    name="ice",
    seed=7,
    soft_boundaries=True,
    terrains=[
        TerrainSpec(
            "deep_ice",
            (118, 162, 208),
            (165, 200, 232),
            (72, 108, 152),
            roughness=0.04,
            macro_scale=2.5,
            detail_scale=7.0,
            micro_scale=14.0,
            macro_w=0.40,
            detail_w=0.40,
            micro_w=0.20,
            specular=0.12,
            cracks=True,
            threshold=0.25,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "ice",
            (188, 215, 238),
            (222, 238, 252),
            (132, 168, 200),
            roughness=0.03,
            macro_scale=3.0,
            detail_scale=8.0,
            micro_scale=16.0,
            macro_w=0.30,
            detail_w=0.50,
            micro_w=0.20,
            specular=0.15,
            threshold=0.52,
            collision=CC.SLOW,
            movement_cost=1.6,
        ),
        TerrainSpec(
            "snow",
            (228, 240, 252),
            (248, 252, 255),
            (188, 208, 228),
            roughness=0.02,
            macro_scale=4.0,
            detail_scale=10.0,
            micro_scale=20.0,
            macro_w=0.20,
            detail_w=0.55,
            micro_w=0.25,
            specular=0.05,
            threshold=0.72,
            collision=CC.WALKABLE,
            movement_cost=1.3,
        ),
        TerrainSpec(
            "ice_rock",
            (108, 128, 148),
            (140, 162, 182),
            (68, 82, 98),
            roughness=0.06,
            macro_scale=2.0,
            detail_scale=6.0,
            micro_scale=12.0,
            macro_w=0.45,
            detail_w=0.35,
            micro_w=0.20,
            specular=0.06,
            cracks=True,
            threshold=0.88,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "crevasse",
            (58, 85, 118),
            (88, 122, 158),
            (28, 48, 72),
            roughness=0.05,
            macro_scale=1.8,
            detail_scale=5.0,
            micro_scale=10.0,
            macro_w=0.50,
            detail_w=0.35,
            micro_w=0.15,
            specular=0.18,
            cracks=True,
            threshold=1.00,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
    ],
)

ROCKY = BiomeConfig(
    name="rocky",
    seed=31,
    soft_boundaries=True,
    terrains=[
        TerrainSpec(
            "lava",
            (192, 62, 22),
            (235, 118, 28),
            (112, 28, 8),
            roughness=0.04,
            macro_scale=2.2,
            detail_scale=6.0,
            micro_scale=12.0,
            macro_w=0.50,
            detail_w=0.35,
            micro_w=0.15,
            specular=0.12,
            threshold=0.18,
            collision=CC.SOLID,
            movement_cost=1.0,
            damage_per_sec=10.0,
        ),
        TerrainSpec(
            "basalt",
            (68, 65, 62),
            (98, 95, 90),
            (38, 36, 34),
            roughness=0.06,
            macro_scale=2.5,
            detail_scale=7.0,
            micro_scale=14.0,
            macro_w=0.40,
            detail_w=0.40,
            micro_w=0.20,
            specular=0.06,
            cracks=True,
            threshold=0.45,
            collision=CC.WALKABLE,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "rock",
            (128, 120, 108),
            (165, 158, 142),
            (78, 72, 65),
            roughness=0.07,
            macro_scale=3.0,
            detail_scale=8.0,
            micro_scale=16.0,
            macro_w=0.35,
            detail_w=0.45,
            micro_w=0.20,
            specular=0.04,
            cracks=True,
            threshold=0.70,
            collision=CC.WALKABLE,
            movement_cost=1.1,
        ),
        TerrainSpec(
            "dust",
            (172, 160, 138),
            (200, 190, 168),
            (118, 108, 88),
            roughness=0.05,
            macro_scale=4.0,
            detail_scale=10.0,
            micro_scale=20.0,
            macro_w=0.25,
            detail_w=0.50,
            micro_w=0.25,
            threshold=0.88,
            collision=CC.WALKABLE,
            movement_cost=1.2,
        ),
        TerrainSpec(
            "cliff",
            (45, 32, 28),
            (72, 48, 38),
            (22, 14, 12),
            roughness=0.12,
            macro_scale=1.5,
            detail_scale=4.0,
            micro_scale=8.0,
            macro_w=0.55,
            detail_w=0.30,
            micro_w=0.15,
            specular=0.03,
            cracks=True,
            threshold=1.00,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
    ],
)

DESERT = BiomeConfig(
    name="desert",
    seed=53,
    soft_boundaries=True,
    terrains=[
        TerrainSpec(
            "quicksand",
            (178, 148, 98),
            (200, 172, 118),
            (132, 108, 68),
            roughness=0.03,
            macro_scale=2.0,
            detail_scale=5.0,
            micro_scale=10.0,
            macro_w=0.50,
            detail_w=0.35,
            micro_w=0.15,
            threshold=0.15,
            collision=CC.SLOW,
            movement_cost=2.5,
        ),
        TerrainSpec(
            "dunes",
            (218, 195, 142),
            (238, 218, 168),
            (172, 152, 108),
            roughness=0.04,
            macro_scale=3.5,
            detail_scale=8.0,
            micro_scale=16.0,
            macro_w=0.35,
            detail_w=0.45,
            micro_w=0.20,
            specular=0.03,
            threshold=0.42,
            collision=CC.WALKABLE,
            movement_cost=1.3,
        ),
        TerrainSpec(
            "hard_sand",
            (202, 182, 128),
            (222, 205, 155),
            (162, 142, 98),
            roughness=0.05,
            macro_scale=4.0,
            detail_scale=10.0,
            micro_scale=20.0,
            macro_w=0.25,
            detail_w=0.50,
            micro_w=0.25,
            threshold=0.65,
            collision=CC.WALKABLE,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "sandstone",
            (168, 132, 88),
            (192, 158, 112),
            (118, 92, 58),
            roughness=0.06,
            macro_scale=2.8,
            detail_scale=7.0,
            micro_scale=14.0,
            macro_w=0.40,
            detail_w=0.40,
            micro_w=0.20,
            specular=0.04,
            cracks=True,
            threshold=0.85,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "mesa",
            (142, 108, 68),
            (172, 138, 92),
            (98, 72, 42),
            roughness=0.07,
            macro_scale=2.2,
            detail_scale=6.0,
            micro_scale=12.0,
            macro_w=0.45,
            detail_w=0.38,
            micro_w=0.17,
            specular=0.05,
            cracks=True,
            threshold=1.00,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
    ],
)

INTERIOR = BiomeConfig(
    name="interior",
    seed=19,
    soft_boundaries=False,
    terrains=[
        TerrainSpec(
            "floor",
            (145, 152, 158),
            (170, 178, 185),
            (98, 105, 112),
            roughness=0.02,
            macro_scale=8.0,
            detail_scale=16.0,
            micro_scale=24.0,
            macro_w=0.10,
            detail_w=0.30,
            micro_w=0.60,
            specular=0.03,
            grid=True,
            grid_spacing=8,
            threshold=0.30,
            collision=CC.WALKABLE,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "plating",
            (88, 94, 102),
            (112, 118, 128),
            (58, 62, 70),
            roughness=0.03,
            macro_scale=8.0,
            detail_scale=16.0,
            micro_scale=24.0,
            macro_w=0.10,
            detail_w=0.25,
            micro_w=0.65,
            specular=0.05,
            grid=True,
            grid_spacing=16,
            threshold=0.55,
            collision=CC.WALKABLE,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "grate",
            (108, 118, 128),
            (132, 142, 155),
            (68, 76, 86),
            roughness=0.03,
            macro_scale=8.0,
            detail_scale=16.0,
            micro_scale=24.0,
            macro_w=0.05,
            detail_w=0.20,
            micro_w=0.75,
            specular=0.03,
            grid=True,
            grid_spacing=6,
            threshold=0.70,
            collision=CC.WALKABLE,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "wall",
            (52, 56, 62),
            (74, 78, 86),
            (32, 34, 40),
            roughness=0.04,
            macro_scale=8.0,
            detail_scale=16.0,
            micro_scale=24.0,
            macro_w=0.05,
            detail_w=0.20,
            micro_w=0.75,
            specular=0.02,
            threshold=0.88,
            collision=CC.SOLID,
            movement_cost=1.0,
        ),
        TerrainSpec(
            "conduit",
            (25, 78, 112),
            (38, 142, 182),
            (12, 44, 70),
            roughness=0.02,
            macro_scale=8.0,
            detail_scale=16.0,
            micro_scale=24.0,
            macro_w=0.05,
            detail_w=0.20,
            micro_w=0.75,
            specular=0.10,
            grid=True,
            grid_spacing=12,
            threshold=1.00,
            collision=CC.TRIGGER,
            movement_cost=1.0,
        ),
    ],
)

ALL_BIOMES = [GARDEN, ICE, ROCKY, DESERT, INTERIOR]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Blob-47 tile atlases + world manifest"
    )
    parser.add_argument(
        "--tile-size", type=int, default=32, help="Tile size in pixels (default 32)"
    )
    parser.add_argument("--out-dir", type=str, default="output")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    print(f"Tile size : {args.tile_size}px")
    print(f"Output    : {out.resolve()}\n")

    for biome in ALL_BIOMES:
        print(f"── {biome.name} ──────────────────────────────")

        print(
            f"  Building atlas ({len(biome.terrains)} terrains × {ATLAS_COLS} cols) …"
        )
        atlas = build_atlas(biome, args.tile_size)
        atlas.save(out / f"{biome.name}_atlas.png")
        print(f"  → {biome.name}_atlas.png  {atlas.size[0]}×{atlas.size[1]}px")

    write_blob47_lut(out)
    print("\n  → blob47_lut.ron")

    write_world_manifest(ALL_BIOMES, args.tile_size, out)
    print("  → world_manifest.ron")

    print("\nDone.")


if __name__ == "__main__":
    main()
