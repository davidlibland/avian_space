#!/usr/bin/env python3
"""
generate_planet_sprites.py — Procedural planet sprite generator for avian_space.

Reads assets/star_systems.yaml and generates a sprite for each planet based on
its ``planet_type`` field.  Sprites are saved to assets/sprites/planets/.

Planet types (set per planet in star_systems.yaml)
--------------------------------------------------
  habitable       Earth-like: oceans, continents, polar caps, clouds
  rocky           Mercury/Moon-like: grey cratered rock
  cloud           Venus-like: thick creamy cloud deck
  desert          Mars-like: rust/ochre desert, polar ice, dust
  gas_giant       Jupiter-like: turbulent colour bands, storm vortex
  gas_giant_rings Saturn-like: gas giant body + tilted ring disc
  ice_giant       Uranus/Neptune-like: pale blue-green banding
  icy_dwarf       Pluto-like: brown-grey icy rock, nitrogen plain

The ``color`` field already present in the YAML (an [r, g, b] triple of 0–1
floats) is used to derive the palette for each planet, so every sprite's
colour palette stays consistent with the existing in-game colour.

Geometry: 3D simplex noise sampled at unit-sphere coordinates avoids seam and
pole artefacts.  Vertex colours are baked directly onto the mesh so pyrender
doesn't need texture uploads.

Environment
-----------
  conda activate avian-sprites   (see scripts/sprites_environment.yml)

Run
---
  python3 scripts/generate_planet_sprites.py
"""

from __future__ import annotations

import hashlib
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
import trimesh
import trimesh.creation as tc
import trimesh.transformations as tf
import pyrender
from PIL import Image
from noise import snoise3

# Headless rendering: EGL on Linux, pyglet (requires display) on macOS.
# Override with PYOPENGL_PLATFORM env var if needed (e.g. osmesa).
if platform.system() == "Linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ---------------------------------------------------------------------------
# Noise and colour utilities
# ---------------------------------------------------------------------------

def sphere_noise(
    unit: np.ndarray,
    scale: float,
    octaves: int,
    persistence: float = 0.5,
    lacunarity: float  = 2.0,
    offset: float      = 0.0,
) -> np.ndarray:
    """3D simplex noise sampled at unit-sphere coordinates (N×3 → N)."""
    return np.array([
        snoise3(v[0]*scale + offset, v[1]*scale + offset, v[2]*scale + offset,
                octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        for v in unit
    ])


def lerp(c0: np.ndarray, c1: np.ndarray, t) -> np.ndarray:
    return np.clip(c0 * (1 - t) + c1 * t, 0, 255)


def remap(x: np.ndarray, lo: float = None, hi: float = None) -> np.ndarray:
    lo = x.min() if lo is None else lo
    hi = x.max() if hi is None else hi
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def apply_vertex_colors(sphere: trimesh.Trimesh, rgb: np.ndarray) -> None:
    rgba = np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)])
    sphere.visual = trimesh.visual.ColorVisuals(mesh=sphere, vertex_colors=rgba)


def make_sphere(radius: float, subdiv: int) -> tuple[trimesh.Trimesh, np.ndarray]:
    sphere = tc.uv_sphere(radius=radius, count=[subdiv, subdiv // 2])
    return sphere, sphere.vertices / radius


# ---------------------------------------------------------------------------
# Planet dataclass
# ---------------------------------------------------------------------------

@dataclass
class Planet:
    name:   str
    mesh:   trimesh.Trimesh        = field(default=None)
    rings:  trimesh.Trimesh | None = field(default=None)
    radius: float                  = field(default=2.0)


# ---------------------------------------------------------------------------
# Ring disc geometry (for Saturn-like planets)
# ---------------------------------------------------------------------------

def make_ring_disc(
    r_inner:   float,
    r_outer:   float,
    n_radial:  int        = 120,
    n_theta:   int        = 360,
    tilt_deg:  float      = 27.0,
    band_spec: list[dict] = None,
) -> trimesh.Trimesh:
    if band_spec is None:
        band_spec = [
            {"r_start": 0.00, "r_end": 0.28, "color": [148, 132, 108]},
            {"r_start": 0.28, "r_end": 0.63, "color": [218, 202, 168]},
            {"r_start": 0.63, "r_end": 0.71, "color": [ 14,  10,   8]},
            {"r_start": 0.71, "r_end": 0.88, "color": [192, 178, 148]},
            {"r_start": 0.88, "r_end": 1.00, "color": [130, 120,  98]},
        ]

    radii  = np.linspace(r_inner, r_outer, n_radial)
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(radii, thetas)
    r_norm = (rr - r_inner) / (r_outer - r_inner)

    x = (rr * np.cos(tt)).ravel()
    z = (rr * np.sin(tt)).ravel()
    y = np.zeros_like(x)
    verts = np.column_stack([x, y, z])

    faces_front = []
    for ti in range(n_theta):
        for ri in range(n_radial - 1):
            a = ti * n_radial + ri
            b = ti * n_radial + ri + 1
            c = ((ti + 1) % n_theta) * n_radial + ri + 1
            d = ((ti + 1) % n_theta) * n_radial + ri
            faces_front += [[a, b, c], [a, c, d]]

    faces_front = np.array(faces_front)
    faces_back  = faces_front[:, ::-1]
    mesh = trimesh.Trimesh(vertices=verts,
                           faces=np.vstack([faces_front, faces_back]))

    r_flat = r_norm.ravel()
    colors = np.zeros((len(verts), 4), dtype=np.uint8)
    colors[:, 3] = 255
    for band in band_spec:
        mask = (r_flat >= band["r_start"]) & (r_flat < band["r_end"])
        colors[mask, :3] = band["color"]

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    mesh.apply_transform(tf.rotation_matrix(np.radians(tilt_deg), [1, 0, 0]))
    return mesh


# ---------------------------------------------------------------------------
# Planet factory functions
# ---------------------------------------------------------------------------

def habitable_planet(
    seed=42, ocean_fraction=0.55, cloud_cover=0.42, polar_cap_lat=68.0,
    radius=2.0, subdiv=256, name="habitable",
) -> Planet:
    rng   = np.random.default_rng(seed)
    off_h = float(rng.uniform(0, 1000))
    off_c = float(rng.uniform(0, 1000))

    sphere, unit = make_sphere(radius, subdiv)
    lat = np.arcsin(np.clip(unit[:, 2], -1, 1))

    height = sphere_noise(unit, scale=3.0, octaves=8, persistence=0.52,
                          lacunarity=2.1, offset=off_h)
    sea_level = np.percentile(height, ocean_fraction * 100)
    h = height - sea_level

    deep_ocean = np.array([ 15,  50, 140], dtype=float)
    mid_ocean  = np.array([ 25,  80, 170], dtype=float)
    shallow    = np.array([ 35, 105, 185], dtype=float)
    beach      = np.array([185, 170, 115], dtype=float)
    lowland    = np.array([ 55, 125,  45], dtype=float)
    midland    = np.array([ 70, 110,  50], dtype=float)
    highland   = np.array([ 95, 100,  65], dtype=float)
    rock       = np.array([110,  95,  80], dtype=float)
    snow       = np.array([220, 225, 235], dtype=float)

    rgb = np.zeros((len(unit), 3), dtype=np.uint8)
    rgb[h < -0.25]                       = deep_ocean.astype(np.uint8)
    rgb[(h >= -0.25) & (h < -0.05)]      = mid_ocean.astype(np.uint8)
    rgb[(h >= -0.05) & (h <  0.00)]      = shallow.astype(np.uint8)
    rgb[(h >=  0.00) & (h <  0.04)]      = beach.astype(np.uint8)
    rgb[(h >=  0.04) & (h <  0.20)]      = lowland.astype(np.uint8)
    rgb[(h >=  0.20) & (h <  0.40)]      = midland.astype(np.uint8)
    rgb[(h >=  0.40) & (h <  0.60)]      = highland.astype(np.uint8)
    rgb[(h >=  0.60) & (h <  0.75)]      = rock.astype(np.uint8)
    rgb[h >= 0.75]                       = snow.astype(np.uint8)

    mid_mask = (h >= -0.25) & (h < -0.05)
    t = remap(h[mid_mask], -0.25, -0.05)
    rgb[mid_mask] = lerp(deep_ocean, mid_ocean, t[:, None]).astype(np.uint8)

    cap_rad  = np.radians(polar_cap_lat)
    soft_rad = np.radians(polar_cap_lat - 8)
    soft = (np.abs(lat) > soft_rad) & (np.abs(lat) <= cap_rad)
    t_p  = remap(np.abs(lat[soft]), polar_cap_lat - 8, polar_cap_lat)
    rgb[soft] = lerp(rgb[soft].astype(float), snow, t_p[:, None]).astype(np.uint8)
    rgb[np.abs(lat) > cap_rad] = snow.astype(np.uint8)

    clouds = sphere_noise(unit, scale=4.5, octaves=5, persistence=0.60,
                          lacunarity=2.2, offset=off_c)
    cloud_thr = np.percentile(clouds, (1 - cloud_cover) * 100)
    cloud_mask = clouds > cloud_thr
    cloud_color = np.array([230, 235, 245], dtype=float)
    t_c = np.clip((clouds[cloud_mask] - cloud_thr) / 0.3, 0, 1)
    rgb[cloud_mask] = lerp(rgb[cloud_mask].astype(float),
                           cloud_color, (t_c * 0.85)[:, None]).astype(np.uint8)

    apply_vertex_colors(sphere, rgb)
    return Planet(name=name, mesh=sphere, radius=radius)


def rocky_barren(
    seed=1, dark=(52, 48, 45), light=(158, 148, 138),
    radius=2.0, subdiv=256, name="rocky_barren",
) -> Planet:
    rng = np.random.default_rng(seed)
    o1, o2, o3 = [float(rng.uniform(0, 1000)) for _ in range(3)]

    sphere, unit = make_sphere(radius, subdiv)

    base    = sphere_noise(unit, scale=2.0, octaves=4, persistence=0.50, offset=o1)
    craters = sphere_noise(unit, scale=11.0, octaves=3, persistence=0.42, offset=o2)
    fine    = sphere_noise(unit, scale=26.0, octaves=2, persistence=0.35, offset=o3)
    combined = remap(0.50 * base + 0.35 * craters + 0.15 * fine)

    d = np.array(dark,  dtype=float)
    l = np.array(light, dtype=float)
    rgb = lerp(d, l, combined[:, None]).astype(np.uint8)

    apply_vertex_colors(sphere, rgb)
    return Planet(name=name, mesh=sphere, radius=radius)


def thick_cloud(
    seed=2, light=(238, 222, 168), dark=(188, 168, 108),
    radius=2.0, subdiv=256, name="thick_cloud",
) -> Planet:
    rng = np.random.default_rng(seed)
    o1, o2 = [float(rng.uniform(0, 1000)) for _ in range(2)]

    sphere, unit = make_sphere(radius, subdiv)

    n1 = sphere_noise(unit, scale=3.5, octaves=6, persistence=0.55, offset=o1)
    n2 = sphere_noise(unit, scale=8.0, octaves=3, persistence=0.40, offset=o2)
    t  = remap(0.70 * n1 + 0.30 * n2)

    l = np.array(light, dtype=float)
    d = np.array(dark,  dtype=float)
    rgb = lerp(d, l, t[:, None]).astype(np.uint8)

    apply_vertex_colors(sphere, rgb)
    return Planet(name=name, mesh=sphere, radius=radius)


def desert_planet(
    seed=3, polar_cap_lat=72.0,
    radius=2.0, subdiv=256, name="desert",
) -> Planet:
    rng = np.random.default_rng(seed)
    o1, o2 = [float(rng.uniform(0, 1000)) for _ in range(2)]

    sphere, unit = make_sphere(radius, subdiv)
    lat = np.arcsin(np.clip(unit[:, 2], -1, 1))

    height = sphere_noise(unit, scale=3.2, octaves=7, persistence=0.54, offset=o1)
    dust   = sphere_noise(unit, scale=6.0, octaves=4, persistence=0.45, offset=o2)
    h  = remap(height)
    du = remap(dust)

    lowland  = np.array([155,  78,  48], dtype=float)
    midland  = np.array([188, 108,  62], dtype=float)
    highland = np.array([198, 130,  80], dtype=float)
    dusty    = np.array([210, 158, 105], dtype=float)
    ice      = np.array([220, 225, 235], dtype=float)

    rgb = np.zeros((len(unit), 3), dtype=np.uint8)
    rgb[h < 0.35]                  = lowland.astype(np.uint8)
    rgb[(h >= 0.35) & (h < 0.65)] = midland.astype(np.uint8)
    rgb[h >= 0.65]                 = highland.astype(np.uint8)

    heavy_dust = du > 0.65
    t_d = remap(du[heavy_dust], 0.65, 1.0)
    rgb[heavy_dust] = lerp(rgb[heavy_dust].astype(float),
                           dusty, (t_d * 0.55)[:, None]).astype(np.uint8)

    cap_rad  = np.radians(polar_cap_lat)
    soft_rad = np.radians(polar_cap_lat - 10)
    soft = (np.abs(lat) > soft_rad) & (np.abs(lat) <= cap_rad)
    t_p  = remap(np.abs(lat[soft]), polar_cap_lat - 10, polar_cap_lat)
    rgb[soft] = lerp(rgb[soft].astype(float), ice, (t_p * 0.9)[:, None]).astype(np.uint8)
    rgb[np.abs(lat) > cap_rad] = ice.astype(np.uint8)

    apply_vertex_colors(sphere, rgb)
    return Planet(name=name, mesh=sphere, radius=radius)


def gas_giant(
    seed=7, band_colors=None, spot_color=(180, 80, 55),
    radius=2.0, subdiv=256, name="gas_giant",
) -> Planet:
    if band_colors is None:
        band_colors = [
            (200, 170, 130), (165, 115,  75), (190, 150, 110),
            (130,  80,  55), (210, 185, 150), (150, 100,  65),
            (195, 165, 125), (110,  65,  45),
        ]

    rng = np.random.default_rng(seed)
    o1  = float(rng.uniform(0, 1000))

    sphere, unit = make_sphere(radius, subdiv)
    lat = np.arcsin(np.clip(unit[:, 2], -1, 1))
    lon = np.arctan2(unit[:, 1], unit[:, 0])

    turb      = sphere_noise(unit, scale=1.8, octaves=5, persistence=0.65, offset=o1)
    turb_fine = sphere_noise(unit, scale=6.0, octaves=3, persistence=0.40, offset=o1+500)
    lat_p     = lat + turb * 0.35 + turb_fine * 0.08

    n_bands  = len(band_colors)
    phase    = (lat_p / (np.pi / 2)) * n_bands * np.pi
    band_t   = (np.sin(phase) + 1) / 2
    band_f   = band_t * (n_bands - 1)
    band_lo  = np.floor(band_f).astype(int).clip(0, n_bands - 2)
    band_hi  = band_lo + 1
    t_interp = band_f - band_lo

    palette = np.array(band_colors, dtype=float)
    rgb = np.array([lerp(palette[band_lo[i]], palette[band_hi[i]], t_interp[i])
                    for i in range(len(unit))], dtype=np.uint8)

    detail = sphere_noise(unit, scale=10.0, octaves=4, persistence=0.45, offset=o1+250)
    rgb = np.clip(rgb.astype(int) + (detail * 18).astype(int)[:, None],
                  0, 255).astype(np.uint8)

    spot_lat = np.radians(float(rng.uniform(-25, -15)))
    spot_lon = np.radians(float(rng.uniform(-60, 60)))
    d_oval   = np.sqrt(((lat - spot_lat) / np.radians(12))**2
                     + ((lon - spot_lon) / np.radians(22))**2)
    spot_mask = d_oval < 1.0
    t_spot    = np.clip(1.0 - d_oval[spot_mask], 0, 1) ** 2
    sc        = np.array(spot_color, dtype=float)
    rgb[spot_mask] = lerp(rgb[spot_mask].astype(float), sc,
                          (t_spot * 0.75)[:, None]).astype(np.uint8)

    polar_t = np.clip((np.abs(lat) - np.radians(55)) / np.radians(20), 0, 1)
    dark    = np.array([60, 40, 30], dtype=float)
    mask    = polar_t > 0
    rgb[mask] = lerp(rgb[mask].astype(float), dark,
                     (polar_t[mask] * 0.6)[:, None]).astype(np.uint8)

    apply_vertex_colors(sphere, rgb)
    return Planet(name=name, mesh=sphere, radius=radius)


def gas_giant_rings(
    seed=99, band_colors=None, ring_r_inner=2.30, ring_r_outer=4.10,
    ring_tilt=27.0, radius=2.0, subdiv=256, name="gas_giant_rings",
) -> Planet:
    if band_colors is None:
        band_colors = [
            (210, 190, 155), (188, 162, 118), (215, 198, 162),
            (170, 148, 108), (222, 208, 175), (182, 158, 118),
            (208, 188, 152), (165, 142, 102),
        ]
    body = gas_giant(seed=seed, band_colors=band_colors,
                     spot_color=(195, 155, 105),
                     radius=radius, subdiv=subdiv, name=name)
    body.rings = make_ring_disc(r_inner=ring_r_inner, r_outer=ring_r_outer,
                                tilt_deg=ring_tilt)
    return body


def ice_giant(
    seed=5, base_color=(155, 210, 210), dark_color=(128, 182, 195),
    band_strength=0.30, radius=2.0, subdiv=256, name="ice_giant",
) -> Planet:
    rng = np.random.default_rng(seed)
    o1, o2 = [float(rng.uniform(0, 1000)) for _ in range(2)]

    sphere, unit = make_sphere(radius, subdiv)
    lat = np.arcsin(np.clip(unit[:, 2], -1, 1))

    turb = sphere_noise(unit, scale=1.5, octaves=4, persistence=0.50, offset=o1)
    fine = sphere_noise(unit, scale=8.0, octaves=3, persistence=0.35, offset=o2)
    lat_p = lat + turb * 0.15

    band_t = (np.sin(lat_p * 6) * 0.5 + 0.5) * band_strength + fine * 0.08

    b = np.array(base_color, dtype=float)
    d = np.array(dark_color, dtype=float)
    rgb = lerp(d, b, np.clip(band_t + 0.5, 0, 1)[:, None]).astype(np.uint8)

    apply_vertex_colors(sphere, rgb)
    return Planet(name=name, mesh=sphere, radius=radius)


def icy_dwarf(
    seed=11, dark=(85, 68, 58), mid=(130, 110, 95), light=(185, 175, 165),
    heart_lat=-20.0, heart_lon=180.0,
    radius=2.0, subdiv=256, name="icy_dwarf",
) -> Planet:
    rng = np.random.default_rng(seed)
    o1  = float(rng.uniform(0, 1000))

    sphere, unit = make_sphere(radius, subdiv)
    lat = np.arcsin(np.clip(unit[:, 2], -1, 1))
    lon = np.arctan2(unit[:, 1], unit[:, 0])

    base = sphere_noise(unit, scale=4.0, octaves=5, persistence=0.50, offset=o1)
    t    = remap(base)

    d = np.array(dark,  dtype=float)
    m = np.array(mid,   dtype=float)
    l = np.array(light, dtype=float)
    rgb = np.where(
        t[:, None] < 0.4,
        lerp(d, m, (t / 0.4)[:, None]),
        lerp(m, l, ((t - 0.4) / 0.6)[:, None]),
    ).astype(np.uint8)

    hl  = np.radians(heart_lat)
    hln = np.radians(heart_lon)
    d_oval = np.sqrt(((lat - hl)  / np.radians(28))**2
                   + ((lon - hln) / np.radians(40))**2)
    heart_mask = d_oval < 1.0
    nitrogen   = np.array([210, 205, 195], dtype=float)
    t_h = np.clip(1 - d_oval[heart_mask], 0, 1) ** 1.5
    rgb[heart_mask] = lerp(rgb[heart_mask].astype(float),
                           nitrogen, (t_h * 0.9)[:, None]).astype(np.uint8)

    apply_vertex_colors(sphere, rgb)
    return Planet(name=name, mesh=sphere, radius=radius)


# ---------------------------------------------------------------------------
# Factory registry and colour-to-palette derivation
# ---------------------------------------------------------------------------

FACTORIES = {
    "habitable":       habitable_planet,
    "rocky":           rocky_barren,
    "cloud":           thick_cloud,
    "desert":          desert_planet,
    "gas_giant":       gas_giant,
    "gas_giant_rings": gas_giant_rings,
    "ice_giant":       ice_giant,
    "icy_dwarf":       icy_dwarf,
}


def _name_seed(name: str) -> int:
    """Deterministic seed from planet name."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _f32_to_u8(c: list[float]) -> tuple[int, int, int]:
    """Convert [0-1] float RGB to [0-255] int tuple."""
    return tuple(max(0, min(255, int(v * 255))) for v in c)


def _scale(c: tuple, factor: float) -> tuple[int, int, int]:
    """Scale an RGB tuple, clamping to [0, 255]."""
    return tuple(max(0, min(255, int(v * factor))) for v in c)


URANUS_RING_BANDS = [
    {"r_start": 0.00, "r_end": 0.20, "color": [ 6,  6,  8]},
    {"r_start": 0.20, "r_end": 0.35, "color": [38, 34, 30]},
    {"r_start": 0.35, "r_end": 0.50, "color": [ 6,  6,  8]},
    {"r_start": 0.50, "r_end": 0.62, "color": [44, 38, 34]},
    {"r_start": 0.62, "r_end": 0.72, "color": [ 6,  6,  8]},
    {"r_start": 0.72, "r_end": 0.82, "color": [48, 42, 38]},
    {"r_start": 0.82, "r_end": 0.90, "color": [ 6,  6,  8]},
    {"r_start": 0.90, "r_end": 1.00, "color": [58, 50, 44]},
]


def build_planet(name: str, pdata: dict) -> Planet:
    """
    Build a Planet using the factory for ``planet_type``, seeded from ``name``.

    The YAML ``color`` field (0-1 floats) is converted to a palette that drives
    the procedural generation, keeping each sprite visually consistent with the
    colour already assigned in the game data.

    If a ``rings`` block is present in the YAML, a ring disc is added to the
    planet after the base mesh is created.  Supported keys::

        rings:
          inner_radius: 2.30    # in mesh units (planet radius = 2.0)
          outer_radius: 4.10
          tilt_deg: 27.0
          style: wide           # "wide" (Saturn, default) or "thin" (Uranus)
    """
    planet_type = pdata.get("planet_type", "rocky")
    color_f32   = pdata.get("color", [0.5, 0.5, 0.5])

    seed = _name_seed(name)
    c    = _f32_to_u8(color_f32)
    kwargs: dict = {"seed": seed, "name": name}

    if planet_type == "rocky":
        kwargs["dark"]  = _scale(c, 0.4)
        kwargs["light"] = _scale(c, 1.4)

    elif planet_type == "cloud":
        kwargs["light"] = _scale(c, 1.15)
        kwargs["dark"]  = _scale(c, 0.75)

    elif planet_type == "desert":
        pass  # seed alone gives good variety with default Mars palette

    elif planet_type == "habitable":
        blue_ratio = color_f32[2] / (sum(color_f32) + 1e-9)
        kwargs["ocean_fraction"] = np.clip(0.3 + blue_ratio * 0.6, 0.15, 0.85)

    elif planet_type == "ice_giant":
        kwargs["base_color"] = _scale(c, 1.1)
        kwargs["dark_color"] = _scale(c, 0.85)

    elif planet_type == "icy_dwarf":
        kwargs["dark"]  = _scale(c, 0.5)
        kwargs["mid"]   = c
        kwargs["light"] = _scale(c, 1.5)

    elif planet_type in ("gas_giant", "gas_giant_rings"):
        bands = []
        for i in range(8):
            f = 0.55 + 0.50 * ((i + 1) % 3) / 2
            bands.append(_scale(c, f))
        kwargs["band_colors"] = bands

    planet = FACTORIES[planet_type](**kwargs)

    # Attach rings if the YAML specifies a rings block
    rings_cfg = pdata.get("rings")
    if rings_cfg and planet.rings is None:
        style     = rings_cfg.get("style", "wide")
        band_spec = URANUS_RING_BANDS if style == "thin" else None
        planet.rings = make_ring_disc(
            r_inner  = rings_cfg.get("inner_radius", 2.30),
            r_outer  = rings_cfg.get("outer_radius", 4.10),
            tilt_deg = rings_cfg.get("tilt_deg", 27.0),
            band_spec = band_spec,
        )

    return planet


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _orbit_pose(elev_rad: float, azim_rad: float, dist: float) -> np.ndarray:
    eye   = dist * np.array([np.cos(elev_rad) * np.sin(azim_rad),
                              np.cos(elev_rad) * np.cos(azim_rad),
                              np.sin(elev_rad)])
    fwd   = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)
    mat   = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -fwd
    mat[:3, 3] = eye
    return mat


def sprite_size(radius: float) -> int:
    """Sprite pixel size from game radius, matching the ship convention (r * 2.2)."""
    return max(40, round(radius * 2.2 / 2) * 2)


def render_sprite(planet: Planet, size: int = 256) -> Image.Image:
    """Render a planet to an RGBA PIL Image with transparent background."""
    fov = 35.0
    # Distance so the planet body just fits: r / tan(half_fov) + margin
    dist = planet.radius / np.tan(np.radians(fov / 2)) * 1.15

    if planet.rings is not None:
        # Use the full bounding box of the ring mesh (post-tilt) to find
        # the maximum extent in any axis, then push the camera back enough
        # that the rings fit inside the field of view with some padding.
        ring_extent = np.abs(planet.rings.bounds).max()
        needed_dist = ring_extent / np.tan(np.radians(fov / 2)) * 1.25
        dist = max(dist, needed_dist)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 1.0],
                           ambient_light=[0.04, 0.04, 0.06])
    scene.add(pyrender.Mesh.from_trimesh(planet.mesh, smooth=True))
    if planet.rings is not None:
        scene.add(pyrender.Mesh.from_trimesh(planet.rings, smooth=False))

    # Key light (warm, sun-like)
    scene.add(pyrender.DirectionalLight(color=[1.0, 0.96, 0.88], intensity=9.0),
              pose=_orbit_pose(np.radians(20), np.radians(-35), 1.0))
    # Fill light (cool, reflected)
    scene.add(pyrender.DirectionalLight(color=[0.25, 0.35, 0.80], intensity=0.8),
              pose=_orbit_pose(np.radians(-10), np.radians(155), 1.0))
    # Camera
    scene.add(pyrender.PerspectiveCamera(yfov=np.radians(fov)),
              pose=_orbit_pose(np.radians(20), np.radians(30), dist))

    r = pyrender.OffscreenRenderer(size, size)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    # Build alpha from depth buffer (background pixels have depth == 0)
    alpha = np.where(depth > 0, 255, 0).astype(np.uint8)
    rgba  = np.dstack([color[:, :, :3], alpha])
    return Image.fromarray(rgba, "RGBA")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root      = Path(__file__).resolve().parent.parent
    yaml_path = root / "assets" / "star_systems.yaml"
    out_dir   = root / "assets" / "sprites" / "planets"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_path) as f:
        systems = yaml.safe_load(f)

    count = 0
    for sys_name, sys_data in systems.items():
        planets = sys_data.get("planets") or {}
        for planet_name, pdata in planets.items():
            ptype = pdata.get("planet_type", "rocky")

            game_radius = pdata.get("radius", 30.0)
            planet = build_planet(planet_name, pdata)

            # For ringed planets, scale the sprite to the ring outer extent
            # so rings aren't clipped in-game.  The ratio ring_outer/body
            # (both in mesh units where body=2.0) maps to the game-pixel
            # scale factor.
            if planet.rings is not None:
                ring_extent = np.abs(planet.rings.bounds).max()
                scale = ring_extent / planet.radius   # mesh-space ratio
                effective_radius = game_radius * scale
            else:
                effective_radius = game_radius
            sz = sprite_size(effective_radius)

            print(f"  {sys_name}/{planet_name} ({ptype}, {sz}x{sz})...",
                  end=" ", flush=True)
            img    = render_sprite(planet, size=max(sz, 128)).resize(
                (sz, sz), Image.LANCZOS)
            out    = out_dir / f"{planet_name}.png"
            img.save(out)
            print(f"-> {out.relative_to(root)}")
            count += 1

    print(f"\nDone — {count} planet sprites written to {out_dir.relative_to(root)}/")


if __name__ == "__main__":
    main()
