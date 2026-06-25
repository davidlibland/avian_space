"""
weapons_gen.py — 3D top-down sprites for the projectile weapons.

Weapons rotate in-engine to face their heading (nose = +Y), so each just needs
one nose-up 3D sprite (no rotation atlas).  Rendered big then downsampled to a
small target size (projectiles must stay small relative to ships).

Replaces the 2D sprites in assets/sprites/ships/:
  ir_missile.png · javelin.png · goose.png · space_mine.png

Run:  scripts/.blender_venv/bin/python weapons_gen.py
"""

import math
import os

from PIL import Image

from blender_gen import (add_box, add_cylinder, add_sphere, glow_material,
                         render_to, reset, setup_scene, toon_material,
                         _obj_from_pydata)

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "assets", "sprites", "weapons")
os.makedirs(OUT, exist_ok=True)


def C(*rgb):
    return tuple(v / 255.0 for v in rgb)


def _exhaust(name, y_base, r, length, mat):
    """Cone tapering aft from the tail (wide at the body, point behind)."""
    add_cylinder(name, (0, y_base - length / 2, 0), 0.004, length, mat,
                 axis="y", r2=r)


def _fins(name, y, span, mat):
    add_box(f"{name}_l", (-span * 0.6, y, 0), (span, 0.16, 0.03), mat, bevel=0.005)
    add_box(f"{name}_r", (span * 0.6, y, 0), (span, 0.16, 0.03), mat, bevel=0.005)
    # thin dorsal fin (reads as a centre ridge from top-down)
    add_box(f"{name}_d", (0, y, 0.05), (0.03, 0.18, 0.12), mat, bevel=0.005)


def build_ir_missile():
    body = toon_material("irm", C(230, 120, 35), spec=0.6)
    dark = toon_material("irm_d", C(140, 72, 22))
    nose = toon_material("irm_n", C(255, 215, 130), spec=0.9)
    glow = glow_material("irm_e", C(255, 170, 60), 1.4)
    add_cylinder("irm_body", (0, 0.0, 0), 0.11, 1.0, body, axis="y", seg=20)
    add_cylinder("irm_nose", (0, 0.67, 0), 0.11, 0.34, nose, axis="y", r2=0.004, seg=20)
    _fins("irm_fin", -0.4, 0.16, dark)
    _exhaust("irm_ex", -0.5, 0.08, 0.22, glow)


def build_javelin():
    body = toon_material("jav", C(80, 165, 230), spec=0.9)
    dark = toon_material("jav_d", C(40, 95, 150))
    nose = toon_material("jav_n", C(200, 235, 255), spec=1.0)
    glow = glow_material("jav_e", C(140, 210, 255), 1.6)
    # slimmer, longer, sharper than the ir_missile (a fast dart)
    add_cylinder("jav_body", (0, 0.0, 0), 0.075, 1.05, body, axis="y", seg=18)
    add_cylinder("jav_nose", (0, 0.72, 0), 0.075, 0.45, nose, axis="y", r2=0.003, seg=18)
    _fins("jav_fin", -0.42, 0.13, dark)
    _exhaust("jav_ex", -0.52, 0.055, 0.26, glow)


def build_goose():
    # cheap, stubby pirate missile — fatter, cruder, asymmetric fins
    body = toon_material("goo", C(225, 110, 45), spec=0.4)
    dark = toon_material("goo_d", C(120, 70, 35))
    nose = toon_material("goo_n", C(245, 175, 90))
    glow = glow_material("goo_e", C(255, 150, 50), 1.4)
    add_cylinder("goo_body", (0, 0.0, 0), 0.15, 0.8, body, axis="y", seg=16)
    add_cylinder("goo_nose", (0, 0.5, 0), 0.15, 0.26, nose, axis="y", r2=0.02, seg=16)
    # crude mismatched fins
    add_box("goo_fl", (-0.2, -0.32, 0), (0.22, 0.18, 0.03), dark, bevel=0.005)
    add_box("goo_fr", (0.17, -0.34, 0), (0.16, 0.16, 0.03), dark, bevel=0.005)
    add_box("goo_band", (0, 0.18, 0.12), (0.26, 0.06, 0.04), dark)  # paint band
    _exhaust("goo_ex", -0.4, 0.11, 0.2, glow)


def build_space_mine():
    shell = toon_material("sm", C(90, 45, 45), spec=0.7)
    bolt = toon_material("sm_b", C(60, 30, 30))
    spike_m = toon_material("sm_s", C(150, 60, 55), spec=0.8)
    core = glow_material("sm_e", C(255, 70, 55), 2.2)
    # central armoured sphere
    add_sphere("sm_core", (0, 0, 0), (0.42, 0.42, 0.42), shell)
    # radiating in-plane spikes (4-sided pyramids)
    n = 8
    for i in range(n):
        a = 2 * math.pi * i / n
        cx, cy = math.cos(a), math.sin(a)
        px, py = -math.sin(a), math.cos(a)
        r0, r1, w = 0.36, 0.78, 0.07
        bx, by = r0 * cx, r0 * cy
        verts = [
            (bx + w * px, by + w * py, 0),
            (bx - w * px, by - w * py, 0),
            (bx, by, w),
            (bx, by, -w),
            (r1 * cx, r1 * cy, 0),
        ]
        faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (0, 1, 2), (1, 0, 3)]
        _obj_from_pydata(f"sm_spike{i}", verts, faces, spike_m, smooth=False)
    # a couple of bolt details + glowing red eye
    for a in (0.4, 2.5, 4.6):
        add_sphere(f"sm_bolt{a}", (0.28 * math.cos(a), 0.28 * math.sin(a), 0.28),
                   (0.06, 0.06, 0.06), bolt)
    add_sphere("sm_eye", (0, 0, 0.4), (0.13, 0.13, 0.1), core)


WEAPONS = [
    # name, builder, ortho, target px — matches the original 2D sprite sizes so
    # on-screen scale (and gameplay proportions) are unchanged.
    ("ir_missile", build_ir_missile, 2.0, 18),
    ("javelin", build_javelin, 2.1, 16),
    ("goose", build_goose, 1.8, 18),
    ("space_mine", build_space_mine, 1.7, 21),
]


def main():
    preview = Image.new("RGBA", (4 * 140, 160), (28, 30, 38, 255))
    for i, (name, builder, ortho, target) in enumerate(WEAPONS):
        reset()
        builder()
        setup_scene(ortho, 128)
        tmp = os.path.join(os.path.dirname(__file__), "out", f"_w_{name}.png")
        render_to(tmp)
        big = Image.open(tmp).convert("RGBA")
        small = big.resize((target, target), Image.LANCZOS)
        small.save(os.path.join(OUT, f"{name}.png"))
        print(f"wrote {name}.png ({target}x{target})")
        # preview tile (scaled up to inspect)
        disp = big.resize((120, 120), Image.LANCZOS)
        preview.paste(disp, (i * 140 + 10, 20), disp)
    preview.save(os.path.join(os.path.dirname(__file__), "out", "_weapons.png"))
    print("saved _weapons.png")


if __name__ == "__main__":
    main()
