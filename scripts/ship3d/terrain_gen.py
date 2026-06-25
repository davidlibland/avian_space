"""
terrain_gen.py — 3D-lit terrain atlases for all biomes.

Replaces the visual {biome}_atlas.png produced by tilegen.py with 3D-lit tiles:
real inter-terrain elevation (transition tiles slope from the higher terrain to
the lower one), organic noise-warped boundaries, and K seamless interior
variants to break the repeating grid.  Same atlas format as before (rows =
terrains, cols 0..46 = blob47) PLUS extra columns 47..(46+K) for the interior
variants.  Also bumps `atlas_cols` in blob47_lut.ron / world_manifest.ron.

Everything is periodic (FFT noise) → seamless; variant deltas are windowed to 0
at the tile edge → any variant tiles against any other.

Run:  scripts/.sprite3d_venv/bin/python terrain_gen.py
Out:  assets/sprites/worlds/{biome}_atlas.png  (+ updated *.ron col counts)
"""
import os
import re
import numpy as np
from PIL import Image

WORLDS = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "sprites", "worlds")
R = 96            # internal render res per tile (downscaled to TILE)
TILE = 32         # game tile size (don't change)
ELEV = 24.0       # px of height per elevation unit (drives transition slopes)
N_VARIANTS = 4    # interior variants per terrain (cols 46..46+N-1)

# ── blob47 (ported from tilegen.py) ─────────────────────────────────────────
TL, T, TR = 1, 2, 4
L, R_, = 8, 16
BL, B, BR = 32, 64, 128


def reduce_to_47(mask):
    if not (mask & L) or not (mask & T): mask &= ~TL
    if not (mask & R_) or not (mask & T): mask &= ~TR
    if not (mask & L) or not (mask & B): mask &= ~BL
    if not (mask & R_) or not (mask & B): mask &= ~BR
    return mask


BLOB47 = sorted({reduce_to_47(m) for m in range(256)})   # col -> reduced mask
assert len(BLOB47) == 47
INTERIOR_COL = BLOB47.index(255)                          # 46


# ── noise / shading ─────────────────────────────────────────────────────────
def pnoise(beta, seed, n=R):
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n) * n
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy); fr[0, 0] = 1.0
    a = np.fft.ifft2(fr ** (-beta) * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, n)))).real
    return (a - a.mean()) / (a.std() + 1e-9)


def pblur(a, sigma):
    n = a.shape[0]
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    g = np.exp(-0.5 * (2 * np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.fft.ifft2(np.fft.fft2(a) * g).real


def n01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def interior_window():
    m = 0.28
    x = np.linspace(0, 1, R)
    w = smoothstep(np.clip(np.minimum(x, 1 - x) / m, 0, 1))
    return w[None, :] * w[:, None]


WIN = interior_window()


def detail_height(spec, variant=0):
    h = pnoise(spec["beta"], spec["seed"]) + 0.45 * pnoise(spec["beta"] * 0.7, spec["seed"] + 9)
    h = n01(h)
    if spec.get("cracks"):
        c = spec["cracks"]
        f = n01(pnoise(c["beta"], spec["seed"] + 13))
        h = n01(h - np.exp(-((f - 0.5) ** 2) / c["eps"] ** 2) * c["carve"])
    if variant:
        d = pnoise(spec["beta"] * 0.8, spec["seed"] + 700 + variant)
        h = np.clip(h + 0.5 * d * WIN, 0, 1)
    return h


def albedo(spec, h):
    c0 = np.array(spec["low"], float); c1 = np.array(spec["high"], float)
    t = np.clip(h * spec.get("ccon", 1.3) + spec.get("cbias", 0.05), 0, 1)[..., None]
    col = c0 * (1 - t) + c1 * t
    g = pnoise(0.6, spec["seed"] + 5)
    return col * (1 + spec.get("grain", 0.05) * g[..., None])


def shade_field(hz, spec):
    gx = np.gradient(hz, axis=1); gy = np.gradient(hz, axis=0)
    nl = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    nx, ny, nz = -gx / nl, -gy / nl, 1.0 / nl
    Lt = np.array((-0.55, -0.6, 0.65)); Lt /= np.linalg.norm(Lt)
    diff = np.clip(nx * Lt[0] + ny * Lt[1] + nz * Lt[2], 0, 1) ** 0.8
    cav = hz - pblur(hz, 7.0)
    ao = np.clip(1.0 + cav * 0.06, 0.4, 1.2)
    return (spec.get("ambient", 0.5) + spec.get("key", 0.9) * diff) * ao


# ── coverage from blob47 mask (organic, warped) ─────────────────────────────
def coverage(mask, seed):
    """1 = centre terrain, 0 = lower terrain, with noise-warped boundaries.
    Replicates tilegen's cardinal+diagonal blob shapes."""
    yy, xx = np.mgrid[0:R, 0:R] / R
    wx = (n01(pnoise(2.4, seed)) - 0.5)
    wy = (n01(pnoise(2.4, seed + 1)) - 0.5)
    cov = np.ones((R, R))
    # cardinal edges: bit clear → neighbour at that edge
    for bit, coord, warp in ((T, yy, wy), (B, 1 - yy, wy), (R_, 1 - xx, wx), (L, xx, wx)):
        if mask & bit:
            continue
        c = np.clip(coord + 0.3 * warp, 0, 1)
        edge = 1.0 - smoothstep(np.clip(c * 2.0, 0, 1))   # 1 at edge → 0 mid
        cov = np.minimum(cov, 1.0 - edge)
    # diagonal concave corners: diag clear but both cardinals set
    for diag, c1, c2, cx, cy in ((TL, T, L, 0, 0), (TR, T, R_, 1, 0),
                                 (BL, B, L, 0, 1), (BR, B, R_, 1, 1)):
        if (mask & diag) or not (mask & c1) or not (mask & c2):
            continue
        dx = np.abs(xx - cx) + 0.18 * wx
        dy = np.abs(yy - cy) + 0.18 * wy
        dist = np.sqrt(dx ** 2 + dy ** 2) / 0.55
        corner = smoothstep(1.0 - np.clip(dist, 0, 1))
        cov = np.minimum(cov, 1.0 - corner)
    return cov


# ── tile render ─────────────────────────────────────────────────────────────
def render_tile(terrains, row, col, variant, max_elev, downscale=True):
    spec = terrains[row]
    lower = terrains[(row - 1) % len(terrains)] if row > 0 else terrains[min(1, len(terrains) - 1)]
    mask = BLOB47[col] if col < 47 else 255
    interior = (mask == 255)

    h_hi = detail_height(spec, variant if interior else 0)
    if interior:
        cov = np.ones((R, R))
    else:
        cov = coverage(mask, spec["seed"] * 7 + lower["seed"] * 13 + col * 5)
    h_lo = detail_height(lower)
    h = cov * h_hi + (1 - cov) * h_lo
    e = lower["elev"] + cov * (spec["elev"] - lower["elev"])
    hz = e * ELEV + h * (cov * spec["relief"] + (1 - cov) * lower["relief"])
    shade = shade_field(hz, spec) * (0.9 + 0.18 * e / max(max_elev, 1e-6))
    col_arr = cov[..., None] * albedo(spec, h_hi) + (1 - cov[..., None]) * albedo(lower, h_lo)
    img = np.clip(col_arr * shade[..., None], 0, 255).astype("uint8")
    out = Image.fromarray(img, "RGB")
    return out.resize((TILE, TILE), Image.LANCZOS) if downscale else out


# ── interior biome: geometric station tiles (man-made, hard edges) ──────────
from PIL import ImageDraw  # noqa: E402

INTERIOR_COLORS = {
    "floor": (150, 156, 162),
    "plating": (92, 98, 106),
    "grate": (116, 126, 136),
    "wall": (58, 62, 68),
    "conduit": (44, 72, 100),
}


def _interior_base(name, variant=0, S=64):
    base = np.array(INTERIOR_COLORS[name], float)
    img = Image.new("RGB", (S, S), tuple(base.astype(int)))
    d = ImageDraw.Draw(img)
    seam = tuple((base * 0.5).astype(int))
    hi = tuple(np.clip(base * 1.26, 0, 255).astype(int))
    lo = tuple((base * 0.74).astype(int))

    if name == "floor":
        P = S // 2                                  # 2×2 panels per tile
        for cx in range(0, S, P):
            for cy in range(0, S, P):
                d.line([(cx, cy), (cx + P - 1, cy)], fill=hi)
                d.line([(cx, cy), (cx, cy + P - 1)], fill=hi)
                d.line([(cx + P - 1, cy), (cx + P - 1, cy + P - 1)], fill=lo)
                d.line([(cx, cy + P - 1), (cx + P - 1, cy + P - 1)], fill=lo)
        for c in range(0, S, P):
            d.line([(c, 0), (c, S)], fill=seam)
            d.line([(0, c), (S, c)], fill=seam)
        for bx in range(0, S + 1, P):              # corner bolts
            for by in range(0, S + 1, P):
                d.ellipse([bx - 2, by - 2, bx + 1, by + 1], fill=lo)
    elif name == "plating":
        for i in range(3):                          # beveled plate border
            d.line([(i, i), (S - 1 - i, i)], fill=hi)
            d.line([(i, i), (i, S - 1 - i)], fill=hi)
            d.line([(S - 1 - i, i), (S - 1 - i, S - 1 - i)], fill=lo)
            d.line([(i, S - 1 - i), (S - 1 - i, S - 1 - i)], fill=lo)
        d.rectangle([0, 0, S - 1, S - 1], outline=seam)
        for rx, ry in ((10, 10), (S - 10, 10), (10, S - 10), (S - 10, S - 10)):
            d.ellipse([rx - 3, ry - 3, rx + 3, ry + 3], fill=lo)
            d.ellipse([rx - 2, ry - 2, rx + 1, ry + 1], fill=hi)
    elif name == "grate":
        H = 8
        for cx in range(0, S, H):                   # mesh of holes
            for cy in range(0, S, H):
                d.rectangle([cx + 2, cy + 2, cx + H - 2, cy + H - 2], fill=seam)
                d.line([(cx + 2, cy + 2), (cx + H - 3, cy + 2)], fill=hi)
                d.line([(cx + 2, cy + H - 3), (cx + H - 2, cy + H - 3)], fill=lo)
    elif name == "wall":
        for i in range(6):                          # raised solid block
            f = hi if i < 4 else lo
            d.line([(i, i), (S - 1 - i, i)], fill=hi)
            d.line([(i, i), (i, S - 1 - i)], fill=hi)
            d.line([(S - 1 - i, i), (S - 1 - i, S - 1 - i)], fill=lo)
            d.line([(i, S - 1 - i), (S - 1 - i, S - 1 - i)], fill=lo)
    elif name == "conduit":
        pipe = tuple(np.clip(base * 1.15, 0, 255).astype(int))
        glow = tuple(np.clip(base * 1.6 + np.array([30, 30, 60]), 0, 255).astype(int))
        for py in range(8, S, 16):                  # horizontal conduit runs
            d.rectangle([0, py - 5, S, py + 5], fill=pipe)
            d.line([(0, py - 4), (S, py - 4)], fill=glow)
            d.line([(0, py + 5), (S, py + 5)], fill=seam)

    arr = np.asarray(img, float)
    nz = n01(pnoise(0.7, (abs(hash(name)) % 9000) + variant * 5, S))
    arr = arr * (1 + 0.05 * (nz - 0.5) * 2)[..., None]
    if variant:                                     # subtle interior scuff (edges untouched)
        sc = n01(pnoise(1.2, 4000 + variant, S))
        arr = arr * (1 - 0.12 * (sc * WIN_S(S))[..., None])
    small = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).resize((TILE, TILE), Image.LANCZOS)
    return np.asarray(small, float)


def WIN_S(S):
    m = 0.3
    x = np.linspace(0, 1, S)
    w = smoothstep(np.clip(np.minimum(x, 1 - x) / m, 0, 1))
    return w[None, :] * w[:, None]


def render_interior_tile(terrains, row, col, variant):
    spec = terrains[row]
    mask = BLOB47[col] if col < 47 else 255
    arr = _interior_base(spec["name"], variant if mask == 255 else 0)
    # hard recessed seams where the neighbour is a different (lower) terrain
    for bit, sl in ((T, (slice(0, 2), slice(None))), (B, (slice(TILE - 2, TILE), slice(None))),
                    (R_, (slice(None), slice(TILE - 2, TILE))), (L, (slice(None), slice(0, 2)))):
        if not (mask & bit):
            arr[sl] *= 0.45
    for diag, c1, c2, sl in ((TL, T, L, (slice(0, 2), slice(0, 2))),
                             (TR, T, R_, (slice(0, 2), slice(TILE - 2, TILE))),
                             (BL, B, L, (slice(TILE - 2, TILE), slice(0, 2))),
                             (BR, B, R_, (slice(TILE - 2, TILE), slice(TILE - 2, TILE)))):
        if not (mask & diag) and (mask & c1) and (mask & c2):
            arr[sl] *= 0.45
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


def build_atlas(name, terrains):
    max_elev = max(t["elev"] for t in terrains)
    cols = 47 + (N_VARIANTS - 1)        # blob47 + extra interior variants
    is_interior = name == "interior"    # geometric station tiles, not heightfield

    def tile(row, col, v):
        return (render_interior_tile(terrains, row, col, v) if is_interior
                else render_tile(terrains, row, col, v, max_elev))

    atlas = Image.new("RGBA", (cols * TILE, len(terrains) * TILE), (0, 0, 0, 0))
    for row in range(len(terrains)):
        for col in range(47):
            atlas.paste(tile(row, col, 0).convert("RGBA"), (col * TILE, row * TILE))
        # interior variants: variant 0 already lives at INTERIOR_COL (46).
        for v in range(1, N_VARIANTS):
            atlas.paste(tile(row, INTERIOR_COL, v).convert("RGBA"),
                        ((46 + v) * TILE, row * TILE))
    atlas.save(os.path.abspath(os.path.join(WORLDS, f"{name}_atlas.png")))
    print(f"  {name}_atlas.png  {atlas.size[0]}×{atlas.size[1]}  ({cols} cols)")
    return cols


# ── biome specs (palette base from tilegen; elev = row order) ───────────────
def terr(name, base, elev, relief, beta, **kw):
    b = np.array(base, float)
    return dict(name=name, low=tuple(np.clip(b * 0.78, 0, 255)),
                high=tuple(np.clip(b * 1.16, 0, 255)), elev=elev, relief=relief,
                beta=beta, **kw)


BIOMES = {
    "garden": [
        terr("water", (55, 118, 182), 0.00, 2, 2.9, grain=0.03, ambient=0.6, key=0.7, ccon=1.2),
        terr("sand", (205, 188, 138), 0.30, 3, 2.7, grain=0.05),
        terr("grass", (72, 145, 72), 0.50, 4, 2.5, grain=0.06),
        terr("forest", (30, 88, 38), 0.66, 6, 2.4, grain=0.07),
        terr("mountain", (122, 114, 104), 0.95, 12, 2.2, grain=0.08,
             cracks=dict(beta=1.8, eps=0.07, carve=0.3)),
    ],
    "ice": [
        terr("deep_ice", (118, 162, 208), 0.00, 2, 2.9, grain=0.03, ambient=0.62, key=0.7),
        terr("ice", (188, 215, 238), 0.30, 3, 2.7, grain=0.03,
             cracks=dict(beta=2.1, eps=0.05, carve=0.25)),
        terr("snow", (228, 240, 252), 0.52, 5, 2.8, grain=0.04),
        terr("ice_rock", (108, 128, 148), 0.72, 9, 2.3, grain=0.07,
             cracks=dict(beta=1.9, eps=0.06, carve=0.3)),
        terr("crevasse", (58, 85, 118), 0.95, 12, 2.2, grain=0.06,
             cracks=dict(beta=2.0, eps=0.04, carve=0.5)),
    ],
    "rocky": [
        terr("lava", (192, 62, 22), 0.00, 3, 2.7, grain=0.06, ambient=0.7, key=0.6, ccon=1.4),
        terr("basalt", (68, 65, 62), 0.30, 6, 2.3, grain=0.08,
             cracks=dict(beta=1.9, eps=0.06, carve=0.3)),
        terr("rock", (128, 120, 108), 0.52, 9, 2.2, grain=0.09,
             cracks=dict(beta=1.8, eps=0.07, carve=0.3)),
        terr("dust", (172, 160, 138), 0.70, 5, 2.6, grain=0.06),
        terr("cliff", (45, 32, 28), 0.95, 14, 2.1, grain=0.09,
             cracks=dict(beta=1.7, eps=0.06, carve=0.45)),
    ],
    "desert": [
        terr("quicksand", (178, 148, 98), 0.00, 2, 2.8, grain=0.04, ambient=0.55),
        terr("dunes", (218, 195, 142), 0.32, 5, 2.7, grain=0.05),
        terr("hard_sand", (202, 182, 128), 0.52, 4, 2.6, grain=0.06),
        terr("sandstone", (168, 132, 88), 0.72, 7, 2.3, grain=0.08,
             cracks=dict(beta=1.9, eps=0.07, carve=0.25)),
        terr("mesa", (142, 108, 68), 0.95, 12, 2.2, grain=0.08,
             cracks=dict(beta=1.8, eps=0.06, carve=0.35)),
    ],
    # interior = space-station floor: flat (no elevation), low relief.
    "interior": [
        terr("floor", (145, 152, 158), 0.5, 1.5, 3.0, grain=0.03, key=0.6, ambient=0.6),
        terr("plating", (88, 94, 102), 0.5, 2.0, 2.8, grain=0.04, key=0.6, ambient=0.6),
        terr("grate", (108, 118, 128), 0.5, 2.5, 2.6, grain=0.05, key=0.6, ambient=0.6),
        terr("wall", (52, 56, 62), 0.5, 3.0, 2.5, grain=0.05, key=0.6, ambient=0.55),
        terr("conduit", (25, 78, 112), 0.5, 2.0, 2.7, grain=0.05, key=0.6, ambient=0.6),
    ],
}
# stable per-terrain seeds
for bi, (bname, ts) in enumerate(BIOMES.items()):
    for ti, t in enumerate(ts):
        t["seed"] = 1000 + bi * 137 + ti * 17


def update_col_count(cols):
    for fn in ("blob47_lut.ron", "world_manifest.ron"):
        p = os.path.abspath(os.path.join(WORLDS, fn))
        s = open(p).read()
        s = re.sub(r"atlas_cols:\s*\d+", f"atlas_cols: {cols}", s)
        open(p, "w").write(s)
        print(f"  updated atlas_cols={cols} in {fn}")


def main():
    cols = 0
    for name, terrains in BIOMES.items():
        cols = build_atlas(name, terrains)
    update_col_count(cols)
    print("done")


if __name__ == "__main__":
    main()
