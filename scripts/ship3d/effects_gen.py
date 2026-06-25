"""
effects_gen.py — procedural VFX textures (PIL/numpy only, no Blender).

  smoke.png — soft radial alpha-gradient puff used for ship damage smoke
              (tinted grey + faded at runtime in src/explosions.rs).

Run:  scripts/.sprite3d_venv/bin/python effects_gen.py
Out:  assets/sprites/effects/smoke.png
"""

import os

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "assets", "sprites", "effects")


def smoke_puff(n=64, seed=3):
    c = (n - 1) / 2
    y, x = np.mgrid[0:n, 0:n]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c
    rng = np.random.default_rng(seed)
    # soft falloff + a little noise so it reads as a smoke puff, not a disc
    alpha = np.clip(1.0 - r, 0, 1) ** 1.7 * (0.85 + 0.15 * rng.random((n, n)))
    img = np.zeros((n, n, 4))
    img[..., :3] = 255
    img[..., 3] = np.clip(alpha, 0, 1) * 255
    return Image.fromarray(img.astype("uint8"), "RGBA")


def main():
    os.makedirs(OUT, exist_ok=True)
    smoke_puff().save(os.path.join(OUT, "smoke.png"))
    print("wrote smoke.png")


if __name__ == "__main__":
    main()
