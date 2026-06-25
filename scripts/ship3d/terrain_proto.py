"""
terrain_proto.py — prototype of richer, seamless terrain tiles.

Renders each terrain as a tileable heightfield with a proper lighting model
(normal-based directional light + ambient + cavity AO) and layered materials,
instead of the current flat noise + gradient.  Periodic by construction (FFT
noise) so it tiles seamlessly — shown as 2×2 to prove it.

Run:  scripts/.sprite3d_venv/bin/python terrain_proto.py
Out:  out/_terrain.png
"""
import os
import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), "out")
N = 160  # render resolution per tile (downscaled in game to ~32-64)


def periodic_noise(n, beta, seed):
    """Tileable fractal noise via 1/f^beta spectrum + random phase."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n) * n
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    fr[0, 0] = 1.0
    amp = fr ** (-beta)
    spec = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, n)))
    field = np.fft.ifft2(spec).real
    return (field - field.mean()) / (field.std() + 1e-9)


def pblur(a, sigma):
    """Periodic gaussian blur (FFT)."""
    n = a.shape[0]
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    g = np.exp(-0.5 * (2 * np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.fft.ifft2(np.fft.fft2(a) * g).real


def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


def crack_field(seed, beta, eps):
    """Network of thin crevice lines (iso-contours of a periodic noise)."""
    f = norm01(periodic_noise(N, beta, seed))
    return np.exp(-((f - 0.5) ** 2) / (eps ** 2))  # 1 on a crack, 0 elsewhere


def render(spec, seed=1):
    h = periodic_noise(N, spec["beta"], seed)
    h = h + 0.45 * periodic_noise(N, spec["beta"] * 0.7, seed + 9)
    h = norm01(h)
    if spec.get("ripple"):                          # anisotropic dunes
        rip = np.sin((np.arange(N) / N) * 2 * np.pi * spec["ripple"])[:, None] * np.ones((1, N))
        warp = norm01(periodic_noise(N, 3.0, seed + 3))
        h = norm01(h * 0.7 + 0.3 * norm01(rip * 0.5 + 0.5 + 0.4 * warp))

    crk = None
    if spec.get("cracks"):
        c = spec["cracks"]
        crk = crack_field(seed + 13, c["beta"], c["eps"])
        h = norm01(h - crk * c["carve"])            # carve crevices into height

    hz = h * spec["relief"]
    gx = np.gradient(hz, axis=1)
    gy = np.gradient(hz, axis=0)
    nl = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    nx, ny, nz = -gx / nl, -gy / nl, 1.0 / nl
    L = np.array(spec["light"], float)
    L /= np.linalg.norm(L)
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    diff = diff ** spec.get("light_gamma", 0.8)     # punch up the light

    cav = hz - pblur(hz, spec["ao_sigma"])
    ao = np.clip(1.0 + cav * spec["ao_strength"], 0.35, 1.2)
    shade = (spec["ambient"] + spec["key"] * diff) * ao

    c0 = np.array(spec["low"], float)
    c1 = np.array(spec["high"], float)
    t = np.clip(h * spec["color_contrast"] + spec["color_bias"], 0, 1)[..., None]
    col = c0 * (1 - t) + c1 * t
    grain = periodic_noise(N, 0.6, seed + 5)
    col = col * (1.0 + spec["grain"] * grain[..., None])
    if crk is not None:
        col = col * (1.0 - (crk * spec["cracks"]["dark"])[..., None])

    img = np.clip(col * shade[..., None], 0, 255).astype("uint8")
    return Image.fromarray(img, "RGB")


SPECS = {
    "snow": dict(low=(188, 200, 220), high=(248, 250, 255), relief=10, beta=2.6,
                 light=(-0.55, -0.6, 0.7), key=0.9, ambient=0.5, ao_sigma=9,
                 ao_strength=0.08, grain=0.05, color_contrast=1.4, color_bias=0.05),
    "sand": dict(low=(176, 144, 92), high=(232, 204, 150), relief=8, beta=2.7,
                 light=(-0.55, -0.6, 0.65), key=1.0, ambient=0.45, ao_sigma=8,
                 ao_strength=0.09, grain=0.06, color_contrast=1.3, color_bias=0.1,
                 ripple=6),
    "rock": dict(low=(58, 55, 52), high=(155, 146, 134), relief=20, beta=2.15,
                 light=(-0.55, -0.6, 0.65), key=1.05, ambient=0.38, ao_sigma=6,
                 ao_strength=0.14, grain=0.09, color_contrast=1.15, color_bias=0.0,
                 cracks=dict(beta=2.0, eps=0.05, carve=0.5, dark=0.45)),
    "ice": dict(low=(96, 134, 178), high=(212, 230, 245), relief=8, beta=2.7,
                light=(-0.55, -0.6, 0.7), key=0.85, ambient=0.55, ao_sigma=8,
                ao_strength=0.07, grain=0.03, color_contrast=1.45, color_bias=0.05,
                cracks=dict(beta=2.3, eps=0.04, carve=0.6, dark=0.3)),
}


def main():
    names = list(SPECS)
    cell = 256  # 2x2 of 128
    pad = 12
    cv = Image.new("RGB", (len(names) * (cell + pad) + pad, cell + 40), (24, 26, 32))
    d = ImageDraw.Draw(cv)
    d.text((pad, 6), "IMPROVED TERRAIN (each shown 2×2 to prove seamless tiling)",
           fill=(235, 235, 245))
    for i, name in enumerate(names):
        tile = render(SPECS[name]).resize((cell // 2, cell // 2), Image.LANCZOS)
        block = Image.new("RGB", (cell, cell))
        for ox in (0, cell // 2):
            for oy in (0, cell // 2):
                block.paste(tile, (ox, oy))
        x = pad + i * (cell + pad)
        cv.paste(block, (x, 28))
        d.text((x + 4, 28 + cell - 16), name, fill=(245, 245, 255))
    cv.save(os.path.join(OUT, "_terrain.png"))
    print("saved _terrain.png")


if __name__ == "__main__":
    main()
