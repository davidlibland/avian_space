"""
fleet_gen.py — BESPOKE, purpose-built ship silhouettes for the whole fleet.

Every ship is hand-shaped to the role in docs/ship_design_bible.md; no shared
template.  A ship must be unmistakable in silhouette alone.

Pipeline hooks (for the two requested features):
  * EXHAUST LAYER  — all drive flames go into a separate "exhaust" collection so
    each ship renders an idle frame (no flame) and a thrust frame (flame).  The
    game composites the flame layer only while the drive is firing.
  * YAW / ANGLES   — render_ship(yaw=θ) rotates the SHIP against a FIXED world
    light, so baking a ring of headings gives subtly shifting highlights/shadows
    (rotating a single sprite in-engine would not).

Run:  scripts/.blender_venv/bin/python fleet_gen.py
"""

import math
import os

import bpy
from PIL import Image, ImageDraw

from blender_gen import (add_box, add_cylinder, add_sphere, elliptical_wing,
                         glow_material, loft_hull, render_to, reset,
                         setup_scene, toon_material, _obj_from_pydata)

OUT = os.path.join(os.path.dirname(__file__), "out")
EXHAUST = "exhaust"


def C(*rgb):
    return tuple(v / 255.0 for v in rgb)


# ─────────────────────── exhaust layer + yaw pipeline ──────────────────────
def _exhaust_coll():
    if EXHAUST not in bpy.data.collections:
        c = bpy.data.collections.new(EXHAUST)
        bpy.context.scene.collection.children.link(c)
    return bpy.data.collections[EXHAUST]


def plume(name, x, y_nozzle, z, r, length, width, mat):
    """Exhaust cone (wide ring at the nozzle, point aft) — registered in the
    separate EXHAUST collection so it can be toggled for thrust."""
    seg = 16
    verts, faces = [], []
    rr = r * width
    y0 = y_nozzle - 0.02
    for k in range(seg):
        a = 2 * math.pi * k / seg
        verts.append((x + rr * math.cos(a), y0, z + rr * math.sin(a)))
    tip = len(verts); verts.append((x, y0 - length, z))
    base = len(verts); verts.append((x, y0, z))
    for j in range(seg):
        jn = (j + 1) % seg
        faces.append((j, jn, tip))
        faces.append((base, jn, j))
    ob = _obj_from_pydata(name, verts, faces, mat, smooth=True)
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    _exhaust_coll().objects.link(ob)
    return ob


def _set_exhaust_visible(vis):
    if EXHAUST in bpy.data.collections:
        for ob in bpy.data.collections[EXHAUST].objects:
            ob.hide_render = not vis


def _rotate_ship(yaw):
    """Parent every ship mesh to a root empty and spin it; camera+lights stay
    fixed in world so the lighting on the hull changes with heading."""
    if abs(yaw) < 1e-6:
        return
    root = bpy.data.objects.new("ship_root", None)
    bpy.context.scene.collection.objects.link(root)
    for o in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
        o.parent = root
    root.rotation_euler = (0, 0, yaw)


# ════════════════════ Independent / Merchant civilian ══════════════════════
def build_shuttle():
    hull = toon_material("sh", C(95, 150, 175), spec=0.8)
    dark = toon_material("sh_d", C(60, 110, 135))
    glass = toon_material("sh_g", C(150, 210, 235), spec=1.6, glass=True)
    glow = glow_material("sh_e", C(255, 175, 90), 4)
    loft_hull("sh_h", [
        dict(y=0.72, w=0.09, h=0.11, cz=0.02),
        dict(y=0.45, w=0.25, h=0.2, cz=0.03),
        dict(y=0.08, w=0.3, h=0.22, cz=0.03),
        dict(y=-0.28, w=0.25, h=0.19, cz=0.02),
        dict(y=-0.56, w=0.15, h=0.13, cz=0.01),
        dict(y=-0.7, w=0.11, h=0.11, cz=0),
    ], hull, m=18, n=2.2, flatten=0.7, subsurf=2)
    add_sphere("sh_can", (0, 0.34, 0.18), (0.13, 0.17, 0.13), glass, zclip=0.18)
    elliptical_wing("sh_fin", dark, span=0.36, root_chord=0.26, tip_chord=0.07,
                    root_y=-0.12, thick=0.04, cz=0.0, sweep=0.55, sections=5)
    add_cylinder("sh_nz", (0, -0.72, 0.0), 0.09, 0.1, dark, r2=0.1)
    plume("sh_pl", 0, -0.78, 0, 0.075, 0.28, 1.0, glow)


def build_courier():
    hull = toon_material("co", C(165, 150, 115), spec=0.9)
    dark = toon_material("co_d", C(120, 108, 82))
    glass = toon_material("co_g", C(150, 205, 230), spec=1.6, glass=True)
    glow = glow_material("co_e", C(255, 175, 95), 4)
    loft_hull("co_h", [
        dict(y=1.05, w=0.015, h=0.02, cz=0),
        dict(y=0.7, w=0.07, h=0.08, cz=0.01),
        dict(y=0.2, w=0.1, h=0.11, cz=0.02),
        dict(y=-0.3, w=0.09, h=0.1, cz=0.02),
        dict(y=-0.8, w=0.07, h=0.08, cz=0.01),
        dict(y=-1.02, w=0.05, h=0.06, cz=0),
    ], hull, m=16, n=2.2, flatten=0.75, subsurf=2)
    add_sphere("co_can", (0, 0.4, 0.1), (0.06, 0.16, 0.07), glass, zclip=0.1)
    elliptical_wing("co_fin", dark, span=0.46, root_chord=0.5, tip_chord=0.03,
                    root_y=-0.2, thick=0.035, cz=0.0, sweep=0.7, sections=6)
    for sx in (-0.05, 0.05):
        add_cylinder("co_nz", (sx, -1.02, 0.0), 0.04, 0.1, dark, r2=0.05)
        plume("co_pl", sx, -1.08, 0, 0.032, 0.34, 1.1, glow)


def build_prospector():
    hull = toon_material("pr", C(150, 120, 80), spec=0.6)
    dark = toon_material("pr_d", C(108, 86, 56))
    drill = toon_material("pr_dr", C(220, 175, 80), spec=1.1)
    glass = toon_material("pr_g", C(130, 180, 205), spec=1.4, glass=True)
    glow = glow_material("pr_e", C(255, 170, 90), 4)
    # compact arrowhead — nimbler than the heavy miner
    loft_hull("pr_h", [
        dict(y=0.7, w=0.05, h=0.06, cz=0.01),
        dict(y=0.35, w=0.22, h=0.16, cz=0.03),
        dict(y=0.0, w=0.24, h=0.17, cz=0.03),
        dict(y=-0.35, w=0.2, h=0.14, cz=0.02),
        dict(y=-0.62, w=0.13, h=0.11, cz=0.01),
    ], hull, m=14, n=2.6, flatten=0.6, subsurf=2)
    add_sphere("pr_can", (0, 0.28, 0.17), (0.1, 0.12, 0.09), glass, zclip=0.17)
    # forward mining drill spike jutting from the nose
    add_cylinder("pr_drl", (0, 0.95, 0.04), 0.07, 0.4, drill, r2=0.004)
    add_cylinder("pr_col", (0, 0.72, 0.04), 0.09, 0.06, dark)
    # swept agility fins
    elliptical_wing("pr_fin", dark, span=0.42, root_chord=0.28, tip_chord=0.05,
                    root_y=-0.18, thick=0.04, cz=0.0, sweep=0.55, sections=5)
    for sx in (-0.13, 0.13):
        add_cylinder("pr_nz", (sx, -0.62, 0.0), 0.07, 0.1, dark, r2=0.08)
        plume("pr_pl", sx, -0.68, 0, 0.058, 0.24, 1.0, glow)


def build_fighter():
    hull = toon_material("fi", C(150, 170, 205), spec=1.0, spec_sharp=0.9)
    dark = toon_material("fi_d", C(96, 116, 150))
    accent = toon_material("fi_a", C(210, 70, 70))
    glass = toon_material("fi_g", C(150, 215, 250), spec=1.6, glass=True)
    glow = glow_material("fi_e", C(120, 190, 255), 6)
    loft_hull("fi_h", [
        dict(y=1.02, w=0.015, h=0.015, cz=0),
        dict(y=0.6, w=0.15, h=0.16, cz=0.03),
        dict(y=0.2, w=0.18, h=0.17, cz=0.03),
        dict(y=-0.2, w=0.15, h=0.14, cz=0.02),
        dict(y=-0.6, w=0.11, h=0.12, cz=0.01),
        dict(y=-0.95, w=0.09, h=0.1, cz=0),
    ], hull, m=14, n=2.4, flatten=0.7, subsurf=2)
    add_sphere("fi_can", (0, 0.32, 0.16), (0.085, 0.2, 0.12), glass, zclip=0.16)
    add_box("fi_str", (0, 0.5, 0.16), (0.2, 0.04, 0.05), accent, bevel=0.01)
    # swept-back delta + forward canards (the approved "A" look)
    elliptical_wing("fi_wing", hull, span=0.66, root_chord=0.6, tip_chord=0.06,
                    root_y=-0.2, thick=0.055, cz=0.02, sweep=0.62, dihedral=0.04)
    elliptical_wing("fi_can", dark, span=0.26, root_chord=0.16, tip_chord=0.03,
                    root_y=0.42, thick=0.03, cz=0.02, sweep=0.18, sections=5)
    for sx in (-0.06, 0.06):
        add_cylinder("fi_nz", (sx, -0.96, 0.0), 0.06, 0.12, dark, r2=0.072)
        plume("fi_pl", sx, -1.02, 0, 0.05, 0.32, 0.8, glow)


def build_corvette():
    hull = toon_material("cv", C(135, 158, 198), spec=1.0)
    dark = toon_material("cv_d", C(90, 112, 152))
    glass = toon_material("cv_g", C(150, 215, 250), spec=1.6, glass=True)
    glow = glow_material("cv_e", C(120, 190, 255), 6)
    # needle blade — narrow and tall, almost no wing
    loft_hull("cv_h", [
        dict(y=1.0, w=0.015, h=0.02, cz=0),
        dict(y=0.6, w=0.06, h=0.07, cz=0.01),
        dict(y=0.1, w=0.08, h=0.09, cz=0.02),
        dict(y=-0.4, w=0.07, h=0.08, cz=0.01),
        dict(y=-0.85, w=0.05, h=0.06, cz=0),
    ], hull, m=14, n=2.3, flatten=0.7, subsurf=2)
    add_sphere("cv_can", (0, 0.42, 0.09), (0.04, 0.13, 0.05), glass, zclip=0.09)
    # small delta tail tabs low on the hull
    elliptical_wing("cv_fin", dark, span=0.26, root_chord=0.22, tip_chord=0.03,
                    root_y=-0.55, thick=0.03, cz=0.0, sweep=0.5, sections=5)
    add_cylinder("cv_nz", (0, -0.85, 0.0), 0.055, 0.1, dark, r2=0.065)
    plume("cv_pl", 0, -0.91, 0, 0.045, 0.3, 0.8, glow)


def build_frigate():
    hull = toon_material("fr", C(120, 135, 162), spec=0.7)
    dark = toon_material("fr_d", C(80, 94, 118))
    nac = toon_material("fr_n", C(95, 108, 132))
    glass = toon_material("fr_g", C(140, 195, 225), spec=1.4, glass=True)
    glow = glow_material("fr_e", C(120, 185, 255), 5)
    # practical surplus warship — chunky wedge, greebly
    loft_hull("fr_h", [
        dict(y=0.95, w=0.08, h=0.1, cz=0.02),
        dict(y=0.55, w=0.24, h=0.18, cz=0.03),
        dict(y=0.05, w=0.26, h=0.19, cz=0.03),
        dict(y=-0.45, w=0.22, h=0.16, cz=0.02),
        dict(y=-0.8, w=0.16, h=0.13, cz=0.01),
    ], hull, m=14, n=3.0, flatten=0.55, subsurf=2)
    add_box("fr_can", (0, 0.45, 0.16), (0.12, 0.16, 0.07), glass, taper=0.7, bevel=0.02)
    # side weapon nacelles with barrel stubs
    for sx in (-1, 1):
        add_box("fr_nac", (sx * 0.34, -0.05, 0.02), (0.13, 0.6, 0.18), nac,
                taper=0.85, bevel=0.02)
        add_cylinder("fr_brl", (sx * 0.34, 0.35, 0.05), 0.02, 0.24, dark)
    # center spine detail + panel greebles
    add_box("fr_spine", (0, -0.05, 0.2), (0.1, 0.7, 0.08), dark, bevel=0.01)
    for yc in (0.1, -0.2):
        add_box("fr_pan", (0, yc, 0.21), (0.4, 0.04, 0.03), dark)
    for sx in (-0.16, 0.16):
        add_cylinder("fr_nz", (sx, -0.82, 0.0), 0.07, 0.12, dark, r2=0.085)
        plume("fr_pl", sx, -0.88, 0, 0.06, 0.3, 0.8, glow)


def build_freighter():
    hull = toon_material("ft", C(150, 142, 120), spec=0.5)
    dark = toon_material("ft_d", C(105, 100, 84))
    cargo = toon_material("ft_c", C(120, 128, 138))
    glass = toon_material("ft_g", C(130, 180, 205), spec=1.3, glass=True)
    glow = glow_material("ft_e", C(255, 170, 90), 4)
    # boxy medium hauler with paired side cargo modules
    loft_hull("ft_h", [
        dict(y=0.9, w=0.08, h=0.1, cz=0.02),
        dict(y=0.5, w=0.2, h=0.18, cz=0.03),
        dict(y=0.0, w=0.22, h=0.2, cz=0.03),
        dict(y=-0.5, w=0.2, h=0.18, cz=0.02),
        dict(y=-0.85, w=0.15, h=0.14, cz=0.01),
    ], hull, m=12, n=3.2, flatten=0.55, subsurf=2)
    add_box("ft_cock", (0, 0.62, 0.12), (0.16, 0.2, 0.16), hull, taper=0.6)
    add_sphere("ft_can", (0, 0.7, 0.2), (0.06, 0.07, 0.05), glass)
    # paired side cargo modules
    for sx in (-1, 1):
        add_box("ft_cargo", (sx * 0.36, -0.05, 0.04), (0.18, 0.7, 0.3), cargo,
                taper=0.92, bevel=0.015)
        for yc in (0.15, -0.2):
            add_box("ft_seam", (sx * 0.36, yc, 0.2), (0.18, 0.03, 0.03), dark)
    for sx in (-0.18, 0.0, 0.18):
        add_cylinder("ft_nz", (sx, -0.86, 0.0), 0.06, 0.1, dark, r2=0.075)
        plume("ft_pl", sx, -0.92, 0, 0.05, 0.24, 1.0, glow)


def build_cargo_transport():
    hull = toon_material("ct", C(138, 150, 128), spec=0.5)
    dark = toon_material("ct_d", C(96, 105, 90))
    cargo = toon_material("ct_c", C(126, 134, 120))
    glass = toon_material("ct_g", C(130, 185, 205), spec=1.4, glass=True)
    glow = glow_material("ct_e", C(255, 170, 90), 4)
    # a stubby "box truck": short cab up front, a big ribbed container on the deck
    loft_hull("ct_h", [
        dict(y=0.76, w=0.11, h=0.13, cz=0.02),
        dict(y=0.48, w=0.21, h=0.18, cz=0.03),
        dict(y=0.08, w=0.24, h=0.2, cz=0.02),
        dict(y=-0.42, w=0.22, h=0.18, cz=0.02),
        dict(y=-0.7, w=0.16, h=0.13, cz=0.01),
    ], hull, m=12, n=3.4, flatten=0.5, subsurf=2)
    add_box("ct_cab", (0, 0.5, 0.13), (0.19, 0.22, 0.18), hull, taper=0.62)
    add_sphere("ct_can", (0, 0.58, 0.22), (0.07, 0.08, 0.05), glass)
    # the defining feature: a ribbed container box riding the rear deck
    add_box("ct_box", (0, -0.16, 0.17), (0.34, 0.6, 0.27), cargo, taper=0.97, bevel=0.02)
    for yc in (0.04, -0.16, -0.36):
        add_box("ct_rib", (0, yc, 0.31), (0.36, 0.035, 0.04), dark)
    # side rails along the deck
    for sx in (-1, 1):
        add_box("ct_rail", (sx * 0.29, -0.16, 0.03), (0.05, 0.58, 0.13), dark, taper=0.95)
    # twin engines
    for sx in (-0.12, 0.12):
        add_cylinder("ct_nz", (sx, -0.72, 0.0), 0.06, 0.1, dark, r2=0.075)
        plume("ct_pl", sx, -0.78, 0, 0.05, 0.24, 1.0, glow)


def build_hauler():
    spine = toon_material("ha_s", C(140, 142, 150))
    spine_d = toon_material("ha_sd", C(98, 100, 110))
    crates = [toon_material("ha_c0", C(150, 132, 95)),
              toon_material("ha_c1", C(108, 120, 132)),
              toon_material("ha_c2", C(140, 110, 90))]
    glass = toon_material("ha_g", C(130, 180, 205), spec=1.3, glass=True)
    glow = glow_material("ha_e", C(255, 170, 90), 4)
    add_box("ha_spine", (0, 0.0, 0.0), (0.2, 1.8, 0.26), spine, taper=0.88)
    add_box("ha_cock", (0, 0.85, 0.08), (0.18, 0.24, 0.2), spine, taper=0.6)
    add_sphere("ha_can", (0, 0.92, 0.18), (0.07, 0.08, 0.06), glass)
    # three rows of paired detachable containers
    for row, yc in enumerate((0.42, 0.0, -0.42)):
        for sx in (-0.34, 0.34):
            add_box("ha_crate", (sx, yc, 0.05), (0.32, 0.36, 0.34),
                    crates[row], taper=0.96, bevel=0.012)
            add_box("ha_latch", (sx, yc, 0.23), (0.28, 0.32, 0.04), spine_d, bevel=0.01)
    add_box("ha_eng", (0, -0.94, 0.0), (0.46, 0.2, 0.26), spine_d)
    for sx in (-0.16, 0.16):
        add_cylinder("ha_nz", (sx, -1.06, 0.0), 0.075, 0.12, spine_d, r2=0.09)
        plume("ha_pl", sx, -1.12, 0, 0.06, 0.22, 1.0, glow)


def build_bulk_carrier():
    spine = toon_material("bc_s", C(120, 118, 112))
    spine_d = toon_material("bc_sd", C(85, 84, 80))
    pods = [toon_material("bc_p0", C(150, 132, 95)),
            toon_material("bc_p1", C(110, 120, 130)),
            toon_material("bc_p2", C(140, 110, 90))]
    glass = toon_material("bc_g", C(120, 170, 200), spec=1.3, glass=True)
    glow = glow_material("bc_e", C(255, 170, 90), 4)
    add_box("bc_spine", (0, -0.05, 0.0), (0.22, 1.9, 0.28), spine, taper=0.9)
    add_box("bc_cock", (0, 0.92, 0.08), (0.2, 0.22, 0.2), spine, taper=0.6)
    add_sphere("bc_can", (0, 0.98, 0.18), (0.07, 0.08, 0.06), glass)
    for ci, cx in enumerate((-0.5, -0.28, 0.28, 0.5)):
        for r, yc in enumerate((0.55, 0.22, -0.11, -0.44)):
            add_box("bc_pod", (cx, yc, 0.04), (0.2, 0.3, 0.34),
                    pods[(ci + r) % 3], taper=0.96, bevel=0.012)
    for yc in (0.55, -0.44):
        add_box("bc_bar", (0, yc, -0.06), (1.2, 0.08, 0.1), spine_d, bevel=0.01)
    add_box("bc_eng", (0, -0.92, 0.0), (0.7, 0.2, 0.26), spine_d)
    for sx in (-0.28, -0.1, 0.1, 0.28):
        add_cylinder("bc_nz", (sx, -1.04, 0.0), 0.07, 0.12, spine_d, r2=0.085)
        plume("bc_pl", sx, -1.1, 0, 0.055, 0.22, 1.0, glow)


def build_asteroid_miner():
    hull = toon_material("am", C(168, 120, 58), spec=0.5)
    dark = toon_material("am_d", C(118, 84, 42))
    steel = toon_material("am_s", C(120, 120, 128), spec=0.7)
    drill = toon_material("am_dr", C(220, 170, 70), spec=1.1)
    glass = toon_material("am_g", C(120, 170, 200), spec=1.4, glass=True)
    glow = glow_material("am_e", C(255, 165, 80), 4)
    loft_hull("am_h", [
        dict(y=0.45, w=0.22, h=0.16, cz=0.03),
        dict(y=0.15, w=0.4, h=0.24, cz=0.04),
        dict(y=-0.15, w=0.42, h=0.25, cz=0.04),
        dict(y=-0.5, w=0.34, h=0.2, cz=0.02),
        dict(y=-0.72, w=0.24, h=0.16, cz=0.01),
    ], hull, m=14, n=3.0, flatten=0.6, subsurf=2)
    add_sphere("am_can", (0, 0.3, 0.2), (0.13, 0.15, 0.1), glass, zclip=0.2)
    for sx in (-1, 1):
        add_box("am_arm", (sx * 0.36, 0.45, 0.03), (0.13, 0.6, 0.18), steel,
                taper=0.8, bevel=0.02)
        add_cylinder("am_col", (sx * 0.36, 0.68, 0.03), 0.12, 0.06, dark)
        add_cylinder("am_drl", (sx * 0.36, 0.84, 0.03), 0.1, 0.3, drill, r2=0.004)
        add_box("am_pod", (sx * 0.46, -0.25, 0.0), (0.12, 0.4, 0.2), dark, taper=0.85)
    for sx in (-0.16, 0.16):
        add_cylinder("am_nz", (sx, -0.72, 0.0), 0.1, 0.14, steel, r2=0.12)
        plume("am_pl", sx, -0.8, 0, 0.08, 0.26, 1.1, glow)


# ════════════════════════════ Federation ═══════════════════════════════════
def build_fed_patrol():
    hull = toon_material("fp", C(92, 94, 102), spec=0.7)   # visible charcoal
    dark = toon_material("fp_d", C(58, 60, 68))
    red = glow_material("fp_r", C(228, 40, 34), 1.12)       # vivid emissive red
    glass = toon_material("fp_g", C(140, 160, 185), spec=1.4, glass=True)
    glow = glow_material("fp_e", C(110, 175, 255), 6)
    # angular armored dart
    loft_hull("fp_h", [
        dict(y=1.0, w=0.02, h=0.03, cz=0),
        dict(y=0.62, w=0.14, h=0.13, cz=0.03),
        dict(y=0.2, w=0.17, h=0.16, cz=0.03),
        dict(y=-0.25, w=0.14, h=0.13, cz=0.02),
        dict(y=-0.7, w=0.1, h=0.11, cz=0.01),
        dict(y=-0.9, w=0.08, h=0.1, cz=0),
    ], hull, m=12, n=3.4, flatten=0.55, subsurf=2)
    add_box("fp_can", (0, 0.4, 0.14), (0.09, 0.16, 0.06), glass, taper=0.7)
    add_box("fp_str", (0, 0.5, 0.15), (0.24, 0.1, 0.05), red)
    # bold red swept wings against the charcoal body — unmistakable
    elliptical_wing("fp_wing", red, span=0.5, root_chord=0.5, tip_chord=0.05,
                    root_y=-0.18, thick=0.05, cz=0.02, sweep=0.6, dihedral=0.03)
    for sx in (-0.07, 0.07):
        add_cylinder("fp_nz", (sx, -0.9, 0.0), 0.06, 0.12, dark, r2=0.072)
        plume("fp_pl", sx, -0.96, 0, 0.05, 0.34, 0.8, glow)


def build_fed_destroyer():
    hull = toon_material("fd", C(92, 94, 102), spec=0.6)   # visible charcoal
    dark = toon_material("fd_d", C(58, 60, 68))
    plate = toon_material("fd_p", C(76, 78, 86))
    red = glow_material("fd_r", C(228, 40, 34), 1.12)       # vivid emissive red
    glass = toon_material("fd_g", C(140, 160, 185), spec=1.3, glass=True)
    glow = glow_material("fd_e", C(110, 175, 255), 5)
    loft_hull("fd_h", [
        dict(y=1.0, w=0.06, h=0.1, cz=0.02),
        dict(y=0.62, w=0.34, h=0.2, cz=0.04),
        dict(y=0.1, w=0.42, h=0.22, cz=0.05),
        dict(y=-0.45, w=0.4, h=0.21, cz=0.04),
        dict(y=-0.82, w=0.32, h=0.18, cz=0.03),
        dict(y=-0.95, w=0.3, h=0.17, cz=0.03),
    ], hull, m=12, n=3.8, flatten=0.5, subsurf=2)
    for sx in (-1, 1):
        # big red side armour belts — the dominant red mass
        add_box("fd_flank", (sx * 0.44, -0.15, 0.03), (0.12, 0.95, 0.22), red,
                taper=0.75, bevel=0.02)
    add_box("fd_can", (0, 0.5, 0.16), (0.12, 0.18, 0.07), glass, taper=0.7, bevel=0.02)
    add_box("fd_brow", (0, 0.62, 0.17), (0.28, 0.05, 0.05), red, bevel=0.01)
    for yc in (0.2, -0.1, -0.4):
        add_box("fd_seam", (0, yc, 0.24), (0.7, 0.035, 0.04), red, bevel=0)
    for (tx, ty) in ((0, 0.0), (-0.2, -0.55), (0.2, -0.55)):
        add_cylinder("fd_tr", (tx, ty, 0.26), 0.08, 0.05, plate)
        add_sphere("fd_td", (tx, ty, 0.3), (0.07, 0.07, 0.055), dark)
        for dx in (-0.025, 0.025):
            add_cylinder("fd_brl", (tx + dx, ty + 0.12, 0.3), 0.012, 0.18, dark)
    for sx in (-0.27, -0.09, 0.09, 0.27):
        add_cylinder("fd_nz", (sx, -0.98, 0.0), 0.07, 0.12, dark, r2=0.085)
        plume("fd_pl", sx, -1.04, 0, 0.06, 0.4, 0.8, glow)


def build_fed_missile_cruiser():
    hull = toon_material("mc", C(90, 92, 100))             # visible charcoal
    dark = toon_material("mc_d", C(56, 58, 66))
    pod = toon_material("mc_p", C(76, 78, 86))
    red = glow_material("mc_r", C(228, 40, 34), 1.12)       # vivid emissive red
    tube = toon_material("mc_t", C(28, 29, 35))
    glass = toon_material("mc_g", C(140, 160, 185), spec=1.3, glass=True)
    glow = glow_material("mc_e", C(110, 175, 255), 5)
    loft_hull("mc_h", [
        dict(y=1.0, w=0.05, h=0.08, cz=0.02),
        dict(y=0.6, w=0.14, h=0.15, cz=0.03),
        dict(y=0.0, w=0.15, h=0.16, cz=0.03),
        dict(y=-0.6, w=0.13, h=0.14, cz=0.02),
        dict(y=-0.95, w=0.1, h=0.11, cz=0.01),
    ], hull, m=14, n=3.0, flatten=0.6, subsurf=2)
    add_box("mc_can", (0, 0.45, 0.14), (0.1, 0.16, 0.06), glass, taper=0.7)
    add_box("mc_brow", (0, 0.56, 0.15), (0.18, 0.04, 0.04), red)
    for sx in (-1, 1):
        add_box("mc_pod", (sx * 0.4, -0.05, 0.04), (0.34, 0.95, 0.26), pod,
                taper=0.92, bevel=0.02)
        for yc in (0.25, 0.0, -0.25):
            for tx in (-0.09, 0.0, 0.09):
                add_box("mc_tube", (sx * 0.4 + tx, yc, 0.2),
                        (0.05, 0.05, 0.1), tube, bevel=0)
        add_box("mc_trim", (sx * 0.24, -0.05, 0.18), (0.06, 0.92, 0.05), red)
        # red caps across the front + outer edge of each missile pod
        add_box("mc_cap", (sx * 0.4, 0.42, 0.19), (0.34, 0.06, 0.06), red)
        add_box("mc_edge", (sx * 0.565, -0.05, 0.18), (0.04, 0.92, 0.05), red)
    for sx in (-0.1, 0.0, 0.1):
        add_cylinder("mc_nz", (sx, -0.96, 0.0), 0.06, 0.1, dark, r2=0.07)
        plume("mc_pl", sx, -1.02, 0, 0.05, 0.36, 0.8, glow)


def build_fed_carrier():
    hull = toon_material("fc", C(90, 92, 100))             # visible charcoal
    dark = toon_material("fc_d", C(56, 58, 66))
    deck = toon_material("fc_dk", C(38, 40, 46))
    plate = toon_material("fc_p", C(76, 78, 86))
    red = glow_material("fc_r", C(228, 40, 34), 1.12)       # vivid emissive red
    amber = glow_material("fc_l", C(255, 180, 70), 3)
    glow = glow_material("fc_e", C(110, 175, 255), 5)
    loft_hull("fc_h", [
        dict(y=1.0, w=0.18, h=0.1, cz=0.0),
        dict(y=0.6, w=0.55, h=0.14, cz=0.0),
        dict(y=0.1, w=0.66, h=0.15, cz=0.0),
        dict(y=-0.5, w=0.62, h=0.14, cz=0.0),
        dict(y=-0.9, w=0.48, h=0.12, cz=0.0),
        dict(y=-1.0, w=0.42, h=0.11, cz=0.0),
    ], hull, m=12, n=3.4, flatten=0.5, subsurf=2)
    add_box("fc_ram", (0, 0.95, 0.02), (0.18, 0.2, 0.12), red, taper=0.4, bevel=0.02)
    add_box("fc_deck", (-0.04, -0.1, 0.13), (0.7, 1.3, 0.04), deck, bevel=0)
    # bold red stripes along the flight-deck edges (very visible top-down)
    for sx in (-0.4, 0.32):
        add_box("fc_dredge", (sx, -0.1, 0.15), (0.04, 1.3, 0.035), red)
    for y in (0.4, 0.1, -0.2, -0.5):
        add_box("fc_line", (-0.04, y, 0.16), (0.02, 0.18, 0.02), amber)
    add_box("fc_isl", (0.46, 0.3, 0.18), (0.14, 0.34, 0.18), plate, taper=0.8, bevel=0.02)
    add_cylinder("fc_mast", (0.46, 0.4, 0.32), 0.012, 0.22, dark, axis="z")
    add_box("fc_isl_r", (0.46, 0.44, 0.2), (0.12, 0.04, 0.05), red)
    for sx in (-1, 1):
        add_box("fc_spon", (sx * 0.66, -0.3, 0.04), (0.08, 0.4, 0.14), red, taper=0.8)
    for sx in (-0.34, -0.2, -0.07, 0.07, 0.2, 0.34):
        add_cylinder("fc_nz", (sx, -1.02, 0.0), 0.055, 0.1, dark, r2=0.07)
        plume("fc_pl", sx, -1.08, 0, 0.045, 0.34, 0.8, glow)


# ══════════════════════════════ Rebels ═════════════════════════════════════
def build_rebel_fighter():
    hull = toon_material("rf", C(48, 145, 230), spec=0.9)   # vivid Rebel blue
    dark = toon_material("rf_d", C(32, 105, 185))
    green = toon_material("rf_gr", C(70, 235, 130))          # bright green
    glass = toon_material("rf_g", C(140, 230, 205), spec=1.6, glass=True)
    glow = glow_material("rf_e", C(110, 245, 150), 6)
    loft_hull("rf_h", [
        dict(y=1.05, w=0.02, h=0.02, cz=0),
        dict(y=0.75, w=0.08, h=0.09, cz=0.02),
        dict(y=0.35, w=0.13, h=0.14, cz=0.03),
        dict(y=-0.05, w=0.12, h=0.13, cz=0.02),
        dict(y=-0.5, w=0.1, h=0.11, cz=0.01),
        dict(y=-0.95, w=0.08, h=0.09, cz=0),
    ], hull, m=16, n=2.1, flatten=0.7, subsurf=2)
    add_sphere("rf_can", (0, 0.38, 0.14), (0.08, 0.2, 0.11), glass, zclip=0.14)
    add_sphere("rf_nose", (0, 0.66, 0.03), (0.05, 0.13, 0.05), green)
    elliptical_wing("rf_wing", hull, span=0.66, root_chord=0.55, tip_chord=0.04,
                    root_y=-0.05, thick=0.05, cz=0.02, sweep=0.34, dihedral=0.07)
    elliptical_wing("rf_tip", green, span=0.66, root_chord=0.12, tip_chord=0.03,
                    root_y=-0.42, thick=0.045, cz=0.02, sweep=0.34, sections=5)
    elliptical_wing("rf_can", dark, span=0.28, root_chord=0.15, tip_chord=0.03,
                    root_y=0.46, thick=0.03, cz=0.02, sweep=0.16, sections=5)
    add_cylinder("rf_nz", (0, -0.95, 0.0), 0.08, 0.12, dark, r2=0.09)
    plume("rf_pl", 0, -1.01, 0, 0.07, 0.36, 0.8, glow)


def build_rebel_gunboat():
    hull = toon_material("rg", C(46, 140, 225), spec=0.8)   # vivid Rebel blue
    dark = toon_material("rg_d", C(30, 100, 178))
    green = toon_material("rg_gr", C(70, 235, 130))          # bright green
    glass = toon_material("rg_g", C(130, 225, 200), spec=1.5, glass=True)
    glow = glow_material("rg_e", C(110, 245, 150), 5)
    # stockier hull, weapon-forward
    loft_hull("rg_h", [
        dict(y=0.7, w=0.1, h=0.12, cz=0.02),
        dict(y=0.35, w=0.2, h=0.18, cz=0.03),
        dict(y=-0.05, w=0.22, h=0.19, cz=0.03),
        dict(y=-0.45, w=0.18, h=0.15, cz=0.02),
        dict(y=-0.8, w=0.13, h=0.12, cz=0.01),
    ], hull, m=16, n=2.4, flatten=0.6, subsurf=2)
    add_sphere("rg_can", (0, 0.28, 0.17), (0.11, 0.13, 0.1), glass, zclip=0.17)
    add_box("rg_str", (0, 0.42, 0.16), (0.32, 0.04, 0.05), green, bevel=0.01)
    # twin forward gun booms projecting ahead of the hull
    for sx in (-0.16, 0.16):
        add_cylinder("rg_boom", (sx, 0.55, 0.04), 0.045, 0.6, dark, r2=0.05)
        add_cylinder("rg_muz", (sx, 0.86, 0.04), 0.03, 0.1, green, r2=0.02)
    # wing plates
    elliptical_wing("rg_wing", hull, span=0.5, root_chord=0.42, tip_chord=0.06,
                    root_y=-0.2, thick=0.05, cz=0.02, sweep=0.4, sections=6)
    for sx in (-0.12, 0.0, 0.12):
        add_cylinder("rg_nz", (sx, -0.82, 0.0), 0.06, 0.1, dark, r2=0.072)
        plume("rg_pl", sx, -0.88, 0, 0.05, 0.3, 0.8, glow)


def build_rebel_frigate():
    hull = toon_material("rfr", C(44, 135, 218), spec=0.7)  # vivid Rebel blue
    dark = toon_material("rfr_d", C(28, 95, 170))
    green = toon_material("rfr_gr", C(70, 235, 130))         # bright green
    bay = toon_material("rfr_b", C(20, 40, 60))
    glass = toon_material("rfr_g", C(130, 220, 198), spec=1.4, glass=True)
    glow = glow_material("rfr_e", C(110, 245, 150), 5)
    # angular wedge with forward-swept nacelles
    loft_hull("rfr_h", [
        dict(y=0.95, w=0.06, h=0.1, cz=0.02),
        dict(y=0.55, w=0.2, h=0.17, cz=0.03),
        dict(y=0.05, w=0.22, h=0.18, cz=0.03),
        dict(y=-0.45, w=0.18, h=0.15, cz=0.02),
        dict(y=-0.82, w=0.13, h=0.12, cz=0.01),
    ], hull, m=14, n=2.7, flatten=0.55, subsurf=2)
    add_box("rfr_can", (0, 0.45, 0.16), (0.1, 0.16, 0.07), glass, taper=0.7, bevel=0.02)
    add_box("rfr_str", (0, 0.58, 0.16), (0.22, 0.04, 0.05), green)
    # forward-swept nacelles (negative sweep)
    elliptical_wing("rfr_nac", hull, span=0.46, root_chord=0.42, tip_chord=0.08,
                    root_y=0.1, thick=0.06, cz=0.02, sweep=-0.4, sections=6)
    for sx in (-1, 1):
        add_cylinder("rfr_brl", (sx * 0.42, 0.42, 0.04), 0.022, 0.22, dark)
    # rear fighter bay (open dark recess)
    add_box("rfr_bay", (0, -0.55, 0.16), (0.18, 0.3, 0.06), bay, bevel=0)
    for sx in (-0.13, 0.13):
        add_cylinder("rfr_nz", (sx, -0.84, 0.0), 0.065, 0.1, dark, r2=0.078)
        plume("rfr_pl", sx, -0.9, 0, 0.055, 0.32, 0.8, glow)


def build_rebel_carrier():
    hull = toon_material("rc", C(46, 138, 220), spec=0.6)   # vivid Rebel blue
    dark = toon_material("rc_d", C(30, 98, 172))
    green = toon_material("rc_gr", C(70, 235, 130))          # bright green
    frame = toon_material("rc_f", C(60, 155, 235))
    glass = toon_material("rc_g", C(130, 220, 198), spec=1.4, glass=True)
    glow = glow_material("rc_e", C(110, 245, 150), 5)
    # long narrow hull
    loft_hull("rc_h", [
        dict(y=1.0, w=0.1, h=0.12, cz=0.0),
        dict(y=0.6, w=0.22, h=0.17, cz=0.0),
        dict(y=0.0, w=0.24, h=0.18, cz=0.0),
        dict(y=-0.6, w=0.22, h=0.16, cz=0.0),
        dict(y=-1.0, w=0.16, h=0.13, cz=0.0),
    ], hull, m=14, n=2.6, flatten=0.5, subsurf=2)
    add_sphere("rc_nose", (0, 0.9, 0.05), (0.07, 0.12, 0.06), green)
    # open lattice flight deck: two side rails + cross spars, gaps between
    for sx in (-0.26, 0.26):
        add_box("rc_rail", (sx, 0.0, 0.18), (0.05, 1.5, 0.06), frame, bevel=0.01)
    for yc in (0.55, 0.28, 0.0, -0.28, -0.55):
        add_box("rc_spar", (0, yc, 0.18), (0.5, 0.05, 0.05), frame, bevel=0.01)
    # bridge fin offset
    add_box("rc_brg", (0.2, 0.45, 0.2), (0.1, 0.24, 0.16), dark, taper=0.8, bevel=0.02)
    add_box("rc_brg_g", (0.2, 0.5, 0.24), (0.06, 0.04, 0.05), glass)
    for sx in (-0.16, 0.0, 0.16):
        add_cylinder("rc_nz", (sx, -1.02, 0.0), 0.07, 0.12, dark, r2=0.085)
        plume("rc_pl", sx, -1.08, 0, 0.06, 0.3, 0.8, glow)


# ══════════════════════════════ Pirates ════════════════════════════════════
def build_pirate_corvette():
    hull = toon_material("pc", C(150, 112, 64), spec=0.5)
    dark = toon_material("pc_d", C(95, 70, 40))
    rust = toon_material("pc_ru", C(120, 70, 40))
    glass = toon_material("pc_g", C(120, 150, 110), spec=1.3, glass=True)
    glow = glow_material("pc_e", C(255, 140, 50), 5)
    # asymmetric scrap blade
    loft_hull("pc_h", [
        dict(y=0.95, w=0.03, h=0.04, cz=0),
        dict(y=0.55, w=0.1, h=0.1, cz=0.02),
        dict(y=0.1, w=0.12, h=0.12, cz=0.02),
        dict(y=-0.4, w=0.1, h=0.1, cz=0.01),
        dict(y=-0.82, w=0.07, h=0.08, cz=0),
    ], hull, m=12, n=2.6, flatten=0.6, subsurf=2)
    add_sphere("pc_can", (0.02, 0.35, 0.13), (0.07, 0.11, 0.08), glass, zclip=0.13)
    # mismatched wings (left bigger, forward; right small)
    elliptical_wing("pc_wR", hull, span=0.34, root_chord=0.3, tip_chord=0.05,
                    root_y=-0.15, thick=0.05, cz=0.0, sweep=0.45, sections=5,
                    side=1, mirror=False)
    elliptical_wing("pc_wL", dark, span=0.46, root_chord=0.34, tip_chord=0.05,
                    root_y=-0.05, thick=0.05, cz=0.0, sweep=0.3, sections=5,
                    side=-1, mirror=False)
    add_box("pc_rust", (-0.12, 0.1, 0.13), (0.08, 0.14, 0.02), rust, bevel=0)
    add_cylinder("pc_ant", (0.08, 0.2, 0.2), 0.008, 0.2, dark, axis="z")
    add_cylinder("pc_nz", (0.02, -0.82, 0.0), 0.06, 0.1, dark, r2=0.07)
    plume("pc_pl", 0.02, -0.88, 0, 0.05, 0.28, 1.2, glow)
    plume("pc_plh", 0.02, -0.88, 0, 0.03, 0.16, 1.1, glow_material("pc_ph", C(255, 210, 120), 7))


def build_pirate_missile_boat():
    hull = toon_material("pm", C(150, 112, 64), spec=0.5)
    dark = toon_material("pm_d", C(95, 70, 40))
    rack = toon_material("pm_r", C(110, 95, 78))
    tube = toon_material("pm_t", C(45, 35, 22))
    rust = toon_material("pm_ru", C(120, 70, 40))
    glass = toon_material("pm_g", C(120, 150, 110), spec=1.3, glass=True)
    glow = glow_material("pm_e", C(255, 140, 50), 5)
    loft_hull("pm_h", [
        dict(y=0.85, w=0.08, h=0.1, cz=0.01),
        dict(y=0.45, w=0.2, h=0.18, cz=0.03),
        dict(y=0.0, w=0.22, h=0.19, cz=0.03),
        dict(y=-0.45, w=0.2, h=0.17, cz=0.02),
        dict(y=-0.8, w=0.15, h=0.13, cz=0.01),
    ], hull, m=12, n=3.0, flatten=0.55, subsurf=2)
    add_sphere("pm_can", (0.03, 0.4, 0.17), (0.1, 0.13, 0.1), glass, zclip=0.17)
    racks = [(-0.34, 0.0, 0.5, 0.34), (0.34, -0.08, 0.42, 0.3)]
    for ri, (rx, ry, rl, rw_) in enumerate(racks):
        add_box("pm_rack", (rx, ry, 0.06), (rw_, rl, 0.2), rack, bevel=0.01, taper=0.9)
        rows = 3 if ri == 0 else 2
        for k in range(rows):
            yy = ry + rl * 0.5 - 0.08 - k * 0.14
            add_box("pm_tube", (rx, yy, 0.17), (rw_ * 0.7, 0.05, 0.1), tube)
    for (rx, ry) in ((-0.1, 0.2), (0.12, -0.3), (-0.05, -0.1)):
        add_box("pm_rust", (rx, ry, 0.2), (0.08, 0.1, 0.02), rust, bevel=0)
    add_cylinder("pm_ant", (0.12, 0.25, 0.28), 0.01, 0.26, dark, axis="z")
    for i, (sx, er) in enumerate(((-0.11, 0.09), (0.13, 0.11))):
        add_cylinder("pm_nz", (sx, -0.82, 0.0), er, 0.14, dark, r2=er * 1.2)
        plume("pm_pl", sx, -0.9, 0, er * 0.9, 0.26, 1.25, glow)
        plume("pm_plh", sx, -0.9, 0, er * 0.5, 0.15, 1.1,
              glow_material(f"pm_ph{i}", C(255, 210, 120), 7))


def build_pirate_carrier():
    hull = toon_material("prc", C(145, 110, 64), spec=0.4)
    dark = toon_material("prc_d", C(92, 68, 40))
    deck = toon_material("prc_dk", C(70, 56, 38))
    plate_a = toon_material("prc_pa", C(120, 122, 126))  # scavenged fed plate
    plate_b = toon_material("prc_pb", C(60, 110, 120))   # scavenged rebel plate
    rust = toon_material("prc_ru", C(120, 70, 40))
    glow = glow_material("prc_e", C(255, 140, 50), 5)
    # big chunky asymmetric hulk
    loft_hull("prc_h", [
        dict(y=1.0, w=0.14, h=0.13, cz=0.0),
        dict(y=0.55, w=0.34, h=0.2, cz=0.0),
        dict(y=0.0, w=0.4, h=0.22, cz=0.0),
        dict(y=-0.55, w=0.36, h=0.2, cz=0.0),
        dict(y=-0.95, w=0.26, h=0.16, cz=0.0),
    ], hull, m=12, n=3.0, flatten=0.5, subsurf=2)
    # mismatched welded deck plates — raised ABOVE the bulbous hull crest
    # (~0.22) so they're not occluded from top-down.
    add_box("prc_deck", (-0.03, -0.05, 0.25), (0.5, 1.2, 0.04), deck, bevel=0)
    add_box("prc_plA", (-0.18, 0.35, 0.29), (0.22, 0.34, 0.05), plate_a, bevel=0.01)
    add_box("prc_plB", (0.16, -0.1, 0.29), (0.26, 0.5, 0.05), plate_b, bevel=0.01)
    add_box("prc_plC", (-0.1, -0.5, 0.29), (0.3, 0.26, 0.05), dark, bevel=0.01)
    # trophy spikes + antennae + rust
    for (sx, sy) in ((-0.42, 0.3), (0.44, -0.2)):
        add_cylinder("prc_spike", (sx, sy, 0.1), 0.04, 0.4, dark, r2=0.005)
    add_cylinder("prc_ant", (0.3, 0.5, 0.34), 0.012, 0.3, dark, axis="z")
    for (rx, ry) in ((0.0, 0.5), (-0.3, -0.3)):
        add_box("prc_rust", (rx, ry, 0.32), (0.12, 0.14, 0.02), rust, bevel=0)
    # mismatched engine bank, orange fire
    for i, (sx, er) in enumerate(((-0.22, 0.085), (-0.02, 0.1), (0.2, 0.075))):
        add_cylinder("prc_nz", (sx, -0.96, 0.0), er, 0.12, dark, r2=er * 1.2)
        plume("prc_pl", sx, -1.02, 0, er * 0.9, 0.26, 1.25, glow)


# ════════════════════════ Cross-faction carrier ════════════════════════════
def build_surplus_carrier():
    hull = toon_material("sc", C(95, 105, 95), spec=0.5)   # neutral olive
    dark = toon_material("sc_d", C(62, 70, 64))
    deck = toon_material("sc_dk", C(40, 44, 40))
    olive = toon_material("sc_o", C(120, 125, 90))
    glass = toon_material("sc_g", C(130, 190, 175), spec=1.4, glass=True)
    glow = glow_material("sc_e", C(110, 175, 255), 5)
    # streamlined long wedge
    loft_hull("sc_h", [
        dict(y=1.0, w=0.06, h=0.1, cz=0.02),
        dict(y=0.55, w=0.26, h=0.18, cz=0.02),
        dict(y=0.0, w=0.3, h=0.19, cz=0.02),
        dict(y=-0.55, w=0.26, h=0.17, cz=0.02),
        dict(y=-0.95, w=0.18, h=0.13, cz=0.01),
    ], hull, m=14, n=2.8, flatten=0.55, subsurf=2)
    add_box("sc_nose", (0, 0.78, 0.18), (0.12, 0.2, 0.08), hull, taper=0.5)
    add_box("sc_str", (0, 0.5, 0.22), (0.34, 0.04, 0.04), olive)
    # swept-back fins (fast carrier feel)
    elliptical_wing("sc_fin", dark, span=0.42, root_chord=0.3, tip_chord=0.05,
                    root_y=0.2, thick=0.04, cz=0.02, sweep=0.5, sections=5)
    # compact rear flight deck — raised above the hull crest (~0.21) so it shows
    add_box("sc_deck", (0, -0.5, 0.24), (0.34, 0.5, 0.04), deck, bevel=0)
    for y in (-0.35, -0.5, -0.65):
        add_box("sc_line", (0, y, 0.27), (0.02, 0.1, 0.02),
                glow_material("sc_l", C(255, 180, 70), 3))
    # single offset bridge fin
    add_box("sc_brg", (0.2, 0.15, 0.27), (0.1, 0.22, 0.16), dark, taper=0.8, bevel=0.02)
    add_box("sc_brg_g", (0.2, 0.2, 0.32), (0.06, 0.04, 0.05), glass)
    for sx in (-0.14, 0.14):
        add_cylinder("sc_nz", (sx, -0.96, 0.0), 0.07, 0.12, dark, r2=0.085)
        plume("sc_pl", sx, -1.02, 0, 0.06, 0.32, 0.8, glow)


# ══════════════════════ Free Frontier (solar sails) ════════════════════════
# Defensive farm-militia. SOLAR SAILS are the primary drive — large pale film
# panels on ASYMMETRIC outrigger booms (designed asymmetry, painted with pride;
# not the pirates' salvage-asymmetry). Sun-bleached yellow/white, warm-white
# auxiliary plasma. Converted workboats that fight.
def _sail(name, cx, cy, cz, w, l, sail, spar, spars=2):
    """A taut solar sail: a gently BILLOWED translucent membrane (arched across
    its width so the cel light gradients over it = reads as cloth catching sun),
    in a visible outer frame with a few light spars."""
    segs = 10
    verts, faces = [], []
    for i in range(segs + 1):
        fx = i / segs - 0.5
        x = cx + fx * w
        zz = cz + 0.07 * math.cos(fx * math.pi)      # belly bulges sunward (z+)
        verts.append((x, cy - l / 2, zz))
        verts.append((x, cy + l / 2, zz))
    for i in range(segs):
        a = 2 * i
        faces.append((a, a + 2, a + 3, a + 1))
    _obj_from_pydata(name, verts, faces, sail, smooth=True)
    add_box(f"{name}_fl", (cx - w / 2, cy, cz + 0.004), (0.02, l, 0.03), spar)
    add_box(f"{name}_fr", (cx + w / 2, cy, cz + 0.004), (0.02, l, 0.03), spar)
    add_box(f"{name}_ft", (cx, cy + l / 2, cz + 0.05), (w, 0.022, 0.03), spar)
    add_box(f"{name}_fb", (cx, cy - l / 2, cz + 0.05), (w, 0.022, 0.03), spar)
    for k in range(spars):
        fx = (-0.5 + (k + 1) / (spars + 1))
        zz = cz + 0.07 * math.cos(fx * math.pi)
        add_box(f"{name}_s", (cx + fx * w, cy, zz + 0.004), (0.009, l * 0.94, 0.02), spar)


def _boom(name, x0, x1, y, z, r, mat):
    """A visible box strut from hull (x0) out to the sail (x1) along X — must
    overlap both so the sail never looks detached."""
    add_box(name, ((x0 + x1) / 2, y, z), (abs(x1 - x0), r * 2.0, r * 2.4), mat,
            bevel=0.006)


# warm sun-catching film — same recipe across the fleet so the faction reads
_FSAIL = lambda tag: toon_material(tag, C(250, 242, 205), spec=1.7, spec_sharp=0.8, glass=True)


def _hex_sail(name, cx, cy, cz, r, sail, spar, billow=0.08):
    """VERTICAL hexagonal solar sail: a six-sided membrane standing UPRIGHT in the
    X-Z plane at Y=cy, its face pointing FORWARD (+Y). Upright line from the side,
    full hexagon from the front. cz is the sail centre height; centre bulges sunward."""
    verts = [(cx, cy + billow, cz)]
    for k in range(6):
        a = k * math.pi / 3
        verts.append((cx + r * math.cos(a), cy, cz + r * math.sin(a)))
    faces = [(0, 1 + k, 1 + (k + 1) % 6) for k in range(6)]
    _obj_from_pydata(name, verts, faces, sail, smooth=True)
    sv, sf = [], []
    for k in range(6):
        a = k * math.pi / 3
        dx, dz = math.cos(a), math.sin(a)
        px, pz = -dz * 0.014, dx * 0.014
        b = len(sv)
        sv += [(cx + px, cy + billow, cz + pz), (cx - px, cy + billow, cz - pz),
               (cx + r * dx - px, cy, cz + r * dz - pz), (cx + r * dx + px, cy, cz + r * dz + pz)]
        sf.append((b, b + 1, b + 2, b + 3))
    _obj_from_pydata(name + "_sp", sv, sf, spar, smooth=False)
    add_cylinder(name + "_hub", (cx, cy, cz), 0.045, 0.07, spar, axis="y")


def _parabolic_sail(name, cx, cy, cz, r, depth, sail, spar):
    """VERTICAL parabolic dish standing in the X-Z plane at Y=cy, bulging FORWARD
    (+Y) toward the sun (convex face ahead, like the hex sails). Radial struts + hub."""
    rings, seg = 4, 22
    verts = [(cx, cy + depth, cz)]                    # centre bulges forward toward the sun
    for i in range(1, rings + 1):
        rr = r * i / rings
        yy = cy + depth * (1 - (i / rings) ** 2)
        for k in range(seg):
            a = 2 * math.pi * k / seg
            verts.append((cx + rr * math.cos(a), yy, cz + rr * math.sin(a)))
    faces = [(0, 1 + k, 1 + (k + 1) % seg) for k in range(seg)]
    for i in range(rings - 1):
        b0, b1 = 1 + i * seg, 1 + (i + 1) * seg
        for k in range(seg):
            kn = (k + 1) % seg
            faces.append((b0 + k, b1 + k, b1 + kn, b0 + kn))
    _obj_from_pydata(name, verts, faces, sail, smooth=True)
    sv, sf = [], []
    for k in range(8):
        a = 2 * math.pi * k / 8
        dx, dz = math.cos(a), math.sin(a)
        px, pz = -dz * 0.013, dx * 0.013
        b = len(sv)
        sv += [(cx + px, cy + depth, cz + pz), (cx - px, cy + depth, cz - pz),
               (cx + r * dx - px, cy, cz + r * dz - pz), (cx + r * dx + px, cy, cz + r * dz + pz)]
        sf.append((b, b + 1, b + 2, b + 3))
    _obj_from_pydata(name + "_sp", sv, sf, spar, smooth=False)
    add_cylinder(name + "_hub", (cx, cy + depth, cz), 0.06, 0.08, spar, axis="y")


def _canopy(name, cx, cy, w, l, cz, amp, sail, spar):
    """VERTICAL billowing canopy standing at Y=cy: spans width w in X and height l
    in Z, its face FORWARD and bulging forward (toward the sun) like a sail
    reaching ahead. cz is the sail centre height."""
    nx, nz = 10, 7
    verts = []
    for j in range(nz + 1):
        fz = j / nz - 0.5
        for i in range(nx + 1):
            fx = i / nx - 0.5
            dome = amp * (1 - (2 * fx) ** 2) * (1 - (1.3 * fz) ** 2)
            verts.append((cx + fx * w, cy + max(0.0, dome), cz + fz * l))
    faces = []
    for j in range(nz):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    _obj_from_pydata(name, verts, faces, sail, smooth=True)
    for fx in (-0.34, -0.11, 0.11, 0.34):            # vertical battens
        xx = cx + fx * w
        yb = cy + amp * (1 - (2 * fx) ** 2) * 0.5
        add_box(name + "_rib", (xx, yb, cz), (0.018, 0.018, l * 0.96), spar)


def _tether3(name, p0, p1, mat, wdt=0.008):
    """A thin rigging line between two 3-D points (sail edge -> hull)."""
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    L = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    px, py = -dy / L * wdt, dx / L * wdt
    _obj_from_pydata(name, [(p0[0] + px, p0[1] + py, p0[2]), (p0[0] - px, p0[1] - py, p0[2]),
                            (p1[0] - px, p1[1] - py, p1[2]), (p1[0] + px, p1[1] + py, p1[2])],
                     [(0, 1, 2, 3)], mat, smooth=False)


def _tether(name, x0, y0, x1, y1, z, mat, wdt=0.009):
    """A taut rigging line from a sail edge to the hull (any direction)."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    px, py = -dy / L * wdt, dx / L * wdt
    _obj_from_pydata(name, [(x0 + px, y0 + py, z), (x0 - px, y0 - py, z),
                            (x1 - px, y1 - py, z), (x1 + px, y1 + py, z)],
                     [(0, 1, 2, 3)], mat, smooth=False)


def build_frontier_skiff():
    hull = toon_material("fk", C(228, 205, 128), spec=0.9)    # sun-bleached yellow
    dark = toon_material("fk_d", C(165, 142, 86))
    chev = toon_material("fk_c", C(120, 150, 90))             # hand-painted chevron
    sail = _FSAIL("fk_sl"); spar = toon_material("fk_sp", C(120, 108, 76))
    glass = toon_material("fk_g", C(150, 210, 200), spec=1.5, glass=True)
    glow = glow_material("fk_e", C(255, 240, 195), 5)         # warm-white plasma
    loft_hull("fk_h", [
        dict(y=0.62, w=0.05, h=0.06, cz=0.02),
        dict(y=0.30, w=0.13, h=0.13, cz=0.03),
        dict(y=-0.08, w=0.12, h=0.12, cz=0.02),
        dict(y=-0.52, w=0.085, h=0.09, cz=0.01),
    ], hull, m=14, n=2.4, flatten=0.65, subsurf=2)
    add_sphere("fk_can", (0, 0.18, 0.13), (0.075, 0.1, 0.08), glass, zclip=0.13)
    add_box("fk_chev", (0, 0.0, 0.14), (0.12, 0.06, 0.04), chev, bevel=0.01)
    # VERTICAL hexagonal sail standing directly in FRONT (offset to starboard)
    add_cylinder("fk_bsprit", (0.08, 0.62, 0.08), 0.024, 0.34, spar, axis="y")
    add_cylinder("fk_mast", (0.16, 0.8, 0.2), 0.022, 0.4, spar, axis="z")
    _hex_sail("fk_sail", 0.16, 0.82, 0.38, 0.36, sail, spar)
    add_cylinder("fk_gun", (-0.06, 0.5, 0.0), 0.022, 0.2, dark, r2=0.015)
    add_cylinder("fk_nz", (0, -0.52, 0.0), 0.05, 0.08, dark, r2=0.06)
    plume("fk_pl", 0, -0.56, 0, 0.04, 0.22, 0.8, glow)


def build_frontier_harvester():
    # GUN-forward gunboat: prominent reaper-boom arms + hexagonal outrigger sails.
    hull = toon_material("fh", C(222, 198, 120), spec=0.8)
    dark = toon_material("fh_d", C(158, 136, 82))
    green = toon_material("fh_gr", C(120, 165, 80))
    sail = _FSAIL("fh_sl"); spar = toon_material("fh_sp", C(118, 106, 74))
    glass = toon_material("fh_g", C(150, 205, 195), spec=1.4, glass=True)
    glow = glow_material("fh_e", C(255, 238, 190), 5)
    loft_hull("fh_h", [
        dict(y=0.72, w=0.1, h=0.12, cz=0.02),
        dict(y=0.35, w=0.22, h=0.19, cz=0.03),
        dict(y=-0.05, w=0.24, h=0.2, cz=0.03),
        dict(y=-0.45, w=0.19, h=0.16, cz=0.02),
        dict(y=-0.82, w=0.12, h=0.12, cz=0.01),
    ], hull, m=14, n=2.6, flatten=0.6, subsurf=2)
    add_sphere("fh_can", (0, 0.26, 0.18), (0.1, 0.13, 0.1), glass, zclip=0.18)
    add_box("fh_str", (0, 0.42, 0.16), (0.3, 0.04, 0.05), green, bevel=0.01)
    for sx in (-0.22, 0.22):
        add_box("fh_arm", (sx, 0.5, 0.05), (0.09, 0.66, 0.16), dark, taper=0.8)
        add_cylinder("fh_drum", (sx, 0.78, 0.05), 0.06, 0.12, green, r2=0.04)
        add_cylinder("fh_muz", (sx, 0.9, 0.05), 0.022, 0.1, dark, r2=0.016)
    # asymmetric VERTICAL hex sails standing in FRONT, flanking the guns (port bigger)
    add_cylinder("fh_bsl", (-0.34, 0.42, 0.1), 0.024, 0.4, spar, axis="y")
    add_cylinder("fh_msl", (-0.34, 0.62, 0.22), 0.022, 0.44, spar, axis="z")
    _hex_sail("fh_sail_l", -0.34, 0.64, 0.42, 0.3, sail, spar)
    add_cylinder("fh_bsr", (0.34, 0.28, 0.1), 0.02, 0.34, spar, axis="y")
    add_cylinder("fh_msr", (0.34, 0.44, 0.16), 0.02, 0.32, spar, axis="z")
    _hex_sail("fh_sail_r", 0.34, 0.46, 0.3, 0.22, sail, spar)
    for sx in (-0.12, 0.12):
        add_cylinder("fh_nz", (sx, -0.84, 0.0), 0.06, 0.1, dark, r2=0.072)
        plume("fh_pl", sx, -0.9, 0, 0.05, 0.3, 0.8, glow)


def build_frontier_sailtender():
    # SOLAR-KITE: a huge billowing canopy LEADS, the slim hull TRAILS behind on
    # rigging tethers — the most dramatic sail rig in the fleet.
    hull = toon_material("ft", C(225, 202, 124), spec=0.8)
    dark = toon_material("ft_d", C(160, 138, 84))
    sail = _FSAIL("ft_sl"); spar = toon_material("ft_sp", C(116, 104, 72))
    glass = toon_material("ft_g", C(150, 205, 195), spec=1.4, glass=True)
    glow = glow_material("ft_e", C(255, 238, 190), 5)
    # slim hull set well AFT (it trails the sail)
    loft_hull("ft_h", [
        dict(y=-0.05, w=0.05, h=0.06, cz=0.02),
        dict(y=-0.34, w=0.12, h=0.12, cz=0.03),
        dict(y=-0.7, w=0.11, h=0.11, cz=0.02),
        dict(y=-1.05, w=0.08, h=0.09, cz=0.01),
    ], hull, m=14, n=2.4, flatten=0.6, subsurf=2)
    add_sphere("ft_can", (0, -0.3, 0.13), (0.075, 0.1, 0.08), glass, zclip=0.13)
    # huge VERTICAL billowing canopy standing in FRONT; hull trails on 3-D tethers
    add_cylinder("ft_mast", (0, 0.6, 0.14), 0.024, 0.66, spar, axis="z")
    _canopy("ft_sail", 0.0, 0.6, 1.05, 0.66, 0.46, 0.2, sail, spar)
    for sx in (-0.46, -0.16, 0.16, 0.46):
        _tether3("ft_t", (sx, 0.58, 0.16), (0.0, -0.04, 0.13), spar)
    add_cylinder("ft_yoke", (0, 0.0, 0.06), 0.05, 0.04, dark, axis="z")
    for sx in (-0.09, 0.09):
        add_cylinder("ft_nz", (sx, -1.07, 0.0), 0.05, 0.1, dark, r2=0.06)
        plume("ft_pl", sx, -1.13, 0, 0.045, 0.28, 0.8, glow)


def build_frontier_monitor():
    hull = toon_material("fm", C(218, 195, 116), spec=0.7)
    dark = toon_material("fm_d", C(152, 132, 80))
    plate = toon_material("fm_p", C(180, 162, 100))
    chev = toon_material("fm_c", C(120, 150, 90))
    sail = _FSAIL("fm_sl"); spar = toon_material("fm_sp", C(114, 102, 70))
    glass = toon_material("fm_g", C(150, 205, 195), spec=1.3, glass=True)
    glow = glow_material("fm_e", C(255, 238, 188), 5)
    loft_hull("fm_h", [
        dict(y=0.85, w=0.16, h=0.16, cz=0.02),
        dict(y=0.4, w=0.34, h=0.24, cz=0.03),
        dict(y=-0.1, w=0.38, h=0.26, cz=0.03),
        dict(y=-0.55, w=0.32, h=0.22, cz=0.02),
        dict(y=-0.95, w=0.2, h=0.16, cz=0.01),
    ], hull, m=14, n=2.8, flatten=0.55, subsurf=2)
    for yc in (0.2, -0.25):
        add_box("fm_belt", (0, yc, 0.24), (0.74, 0.06, 0.05), plate, bevel=0.01)
    add_box("fm_chev", (0, -0.05, 0.26), (0.3, 0.08, 0.04), chev, bevel=0.01)
    add_box("fm_brg", (0, 0.32, 0.26), (0.2, 0.26, 0.16), dark, taper=0.7, bevel=0.02)
    add_sphere("fm_g", (0, 0.38, 0.34), (0.07, 0.08, 0.05), glass)
    # huge VERTICAL parabolic dish sun-shield standing directly in FRONT
    add_cylinder("fm_bsprit", (0.0, 0.9, 0.12), 0.04, 0.4, spar, axis="y")
    add_cylinder("fm_mast", (0.0, 1.08, 0.3), 0.035, 0.6, spar, axis="z")
    _parabolic_sail("fm_sail", 0.0, 1.12, 0.5, 0.5, 0.18, sail, spar)
    # flak-turret blisters on the flanks (visible barrels)
    for sx in (-0.36, 0.36):
        add_cylinder("fm_turr", (sx, 0.0, 0.28), 0.085, 0.1, dark, axis="z")
        add_cylinder("fm_bbl", (sx, 0.14, 0.32), 0.02, 0.2, plate, r2=0.014)
    for sx in (-0.22, 0.0, 0.22):
        add_cylinder("fm_nz", (sx, -0.98, 0.0), 0.07, 0.12, dark, r2=0.085)
        plume("fm_pl", sx, -1.05, 0, 0.06, 0.32, 0.8, glow)


# ════════════════════════ Helios Combine (drones) ══════════════════════════
# Megacorp product design: smooth seamless GLOSS-WHITE shells, gold brand trim,
# CYAN accents & near-silent cyan drives. NO cockpits — automated, so a cyan
# sensor "eye" instead of a canopy. Recessed weapons; drone swarms + drone-bays.
# The deliberate opposite of the Free Frontier's flat painted sails.
def _helios_mats(tag):
    return dict(
        white=toon_material(tag + "w", C(236, 240, 245), spec=1.7, spec_sharp=0.82),
        wd=toon_material(tag + "wd", C(180, 190, 202)),
        gold=glow_material(tag + "au", C(232, 196, 96), 1.6),     # bright brand trim
        dark=toon_material(tag + "k", C(38, 48, 60)),
        bay=toon_material(tag + "b", C(24, 34, 46)),
        seam=toon_material(tag + "s", C(150, 162, 176)),          # panel seam
        cyan=glow_material(tag + "e", C(55, 215, 255), 14),       # saturated brand cyan
    )


def _hnz(tag, sx, y, r, m):
    """Recessed cyan drive ring."""
    add_cylinder(tag + "nz", (sx, y, 0.0), r, r * 1.4, m["dark"], r2=r * 1.1)
    add_cylinder(tag + "ring", (sx, y - r * 0.7, 0.0), r * 0.8, r * 0.18, m["cyan"])
    plume(tag + "pl", sx, y - r * 0.9, 0, r * 0.8, r * 4.5, 0.85, m["cyan"])


def build_helios_drone():
    m = _helios_mats("hd")
    # a crisp flat delta wedge (minimal, purposeful — not an egg)
    loft_hull("hd_h", [
        dict(y=0.52, w=0.03, h=0.04, cz=0.0),
        dict(y=0.1, w=0.14, h=0.08, cz=0.0),
        dict(y=-0.28, w=0.22, h=0.09, cz=0.0),
        dict(y=-0.42, w=0.16, h=0.07, cz=0.0),
    ], m["white"], m=12, n=3.0, flatten=0.7, subsurf=2)
    add_box("hd_tr", (0, -0.28, 0.06), (0.4, 0.03, 0.015), m["gold"], bevel=0.006)
    add_sphere("hd_eye", (0, 0.18, 0.05), (0.07, 0.14, 0.045), m["cyan"])  # big camera eye
    add_cylinder("hd_tip", (0, 0.46, 0.02), 0.02, 0.06, m["cyan"], r2=0.004)
    _hnz("hd_", 0, -0.42, 0.05, m)


def build_helios_enforcer():
    m = _helios_mats("he")
    loft_hull("he_h", [
        dict(y=0.98, w=0.04, h=0.06, cz=0.0),
        dict(y=0.5, w=0.18, h=0.14, cz=0.0),
        dict(y=0.0, w=0.22, h=0.16, cz=0.0),
        dict(y=-0.5, w=0.17, h=0.13, cz=0.0),
        dict(y=-0.95, w=0.06, h=0.07, cz=0.0),
    ], m["white"], m=16, n=2.9, flatten=0.55, subsurf=2)
    add_box("he_seam", (0, 0.05, 0.165), (0.06, 1.3, 0.012), m["gold"], bevel=0.004)
    add_sphere("he_eye", (0, 0.52, 0.1), (0.085, 0.16, 0.05), m["cyan"])  # bigger eye
    # cyan running lights along the flanks
    for sx in (-0.16, 0.16):
        add_box("he_run", (sx, 0.18, 0.14), (0.018, 0.34, 0.018), m["cyan"])
    for sy in (0.25, -0.25):
        add_box("he_pan", (0, sy, 0.155), (0.34, 0.012, 0.012), m["seam"])
    for sx in (-1, 1):
        elliptical_wing("he_fin", m["wd"], span=0.42, root_chord=0.36, tip_chord=0.05,
                        root_y=-0.12, thick=0.04, cz=0.0, sweep=0.34, sections=6,
                        side=sx, mirror=False)
        add_box("he_port", (sx * 0.17, 0.36, 0.09), (0.035, 0.22, 0.04), m["dark"], bevel=0.008)
    _hnz("he_", 0, -0.95, 0.075, m)


def build_helios_overseer():
    m = _helios_mats("ho")
    loft_hull("ho_h", [
        dict(y=1.0, w=0.07, h=0.1, cz=0.0),
        dict(y=0.5, w=0.26, h=0.16, cz=0.0),
        dict(y=0.0, w=0.3, h=0.18, cz=0.0),
        dict(y=-0.55, w=0.24, h=0.15, cz=0.0),
        dict(y=-1.0, w=0.09, h=0.1, cz=0.0),
    ], m["white"], m=16, n=2.9, flatten=0.5, subsurf=2)
    add_box("ho_seam", (0, 0.0, 0.185), (0.06, 1.6, 0.012), m["gold"], bevel=0.004)
    add_sphere("ho_eye", (0, 0.58, 0.12), (0.11, 0.18, 0.07), m["cyan"])
    add_box("ho_pan", (0, 0.78, 0.16), (0.22, 0.012, 0.012), m["seam"])
    # recessed drone berths with bright cyan launch strips
    for sy in (0.22, -0.08, -0.4):
        for sx in (-0.22, 0.22):
            add_box("ho_bay", (sx, sy, 0.15), (0.12, 0.18, 0.06), m["bay"], bevel=0.0)
            add_box("ho_lt", (sx, sy + 0.09, 0.2), (0.1, 0.025, 0.025), m["cyan"])
    for sx in (-0.14, 0.14):
        _hnz("ho_", sx, -1.0, 0.072, m)


def build_helios_titan():
    m = _helios_mats("ht")
    loft_hull("ht_h", [
        dict(y=1.2, w=0.09, h=0.12, cz=0.0),
        dict(y=0.6, w=0.34, h=0.2, cz=0.0),
        dict(y=0.0, w=0.4, h=0.24, cz=0.0),
        dict(y=-0.6, w=0.34, h=0.2, cz=0.0),
        dict(y=-1.2, w=0.13, h=0.12, cz=0.0),
    ], m["white"], m=18, n=3.0, flatten=0.5, subsurf=2)
    for yc in (0.45, 0.0, -0.45):
        add_box("ht_band", (0, yc, 0.24), (0.74, 0.06, 0.018), m["gold"], bevel=0.005)
    add_sphere("ht_eye", (0, 0.74, 0.16), (0.12, 0.19, 0.08), m["cyan"])
    # central drone-launch trench with bright cyan rails (the factory ship)
    add_box("ht_trench", (0, -0.1, 0.22), (0.18, 1.0, 0.08), m["bay"], bevel=0.0)
    for yc in (0.3, 0.0, -0.3, -0.6):
        add_box("ht_rail", (0, yc, 0.28), (0.2, 0.035, 0.025), m["cyan"])
    for sx in (-0.24, 0.0, 0.24):
        _hnz("ht%d_" % int(sx * 10), sx, -1.2, 0.08, m)


# ═══════════════════ Fed Bastion (all-angle iron siege) ════════════════════
# Hardliner splinter: ALL ANGLES, nothing smooth — faceted iron built from boxes
# and wedges (no subsurf lofts). Iron-grey + black + heavy red, ram prows, banner
# masts, LONG kinetic slug/chaingun barrels (no missiles), red-orange ion drives.
def _wedge(name, cx, cy, cz, w, l, h, mat, flat=1.0):
    """Faceted ram-wedge: rectangular base tapering to a point forward (y+)."""
    hw, hh = w / 2, h / 2
    v = [(cx - hw, cy - l / 2, cz - hh * flat), (cx + hw, cy - l / 2, cz - hh * flat),
         (cx + hw, cy - l / 2, cz + hh), (cx - hw, cy - l / 2, cz + hh),
         (cx, cy + l / 2, cz)]
    f = [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    _obj_from_pydata(name, v, f, mat, smooth=False)


def _bastion_mats(tag):
    return dict(
        iron=toon_material(tag + "i", C(92, 95, 102), spec=0.5),
        dark=toon_material(tag + "k", C(44, 47, 54)),
        steel=toon_material(tag + "s", C(126, 128, 134), spec=0.7),
        red=toon_material(tag + "r", C(176, 44, 36)),
        slit=glow_material(tag + "sl", C(220, 70, 40), 3),
        glow=glow_material(tag + "e", C(255, 95, 32), 5),    # red-orange ion
    )


def _slug_drive(tag, sx, y, r, m):
    add_box(tag + "nz", (sx, y, 0.0), (r * 2.2, r * 1.6, r * 2.0), m["dark"], bevel=0.01)
    plume(tag + "pl", sx, y - r * 0.9, 0, r * 0.9, r * 4.0, 0.85, m["glow"])


def _iron_plates(tag, ys, w, pl, ztop, m):
    """Raised faceted armour plates with a dark bolt-seam — breaks up the slab."""
    for i, yy in enumerate(ys):
        add_box(tag + "ap%d" % i, (0, yy, ztop), (w, pl, 0.03), m["steel"], bevel=0.01)
        add_box(tag + "bs%d" % i, (0, yy, ztop + 0.014), (w * 0.86, 0.012, 0.02), m["dark"])


def build_bastion_guard():
    m = _bastion_mats("bg")
    add_box("bg_body", (0, -0.12, 0.0), (0.34, 1.0, 0.22), m["iron"], taper=0.7)
    _wedge("bg_prow", 0, 0.62, 0.0, 0.34, 0.5, 0.22, m["iron"])
    for sx in (-1, 1):
        add_box("bg_plate", (sx * 0.21, -0.1, 0.02), (0.08, 0.72, 0.28), m["dark"], taper=0.85)
    add_box("bg_str", (0, -0.1, 0.13), (0.1, 0.66, 0.04), m["red"], bevel=0.004)
    add_box("bg_slit", (0, 0.16, 0.14), (0.07, 0.16, 0.04), m["slit"])
    _iron_plates("bg", (0.3, 0.02, -0.26), 0.2, 0.16, 0.12, m)
    for sx in (-0.08, 0.08):                                  # twin forward chaingun
        add_cylinder("bg_gun", (sx, 0.5, 0.06), 0.028, 0.42, m["steel"], r2=0.022)
    for sx in (-0.1, 0.1):
        _slug_drive("bg%d" % int(sx * 10), sx, -0.62, 0.06, m)


def build_bastion_lance():
    m = _bastion_mats("bl")
    add_box("bl_body", (0, -0.1, 0.0), (0.4, 1.3, 0.26), m["iron"], taper=0.72)
    _wedge("bl_prow", 0, 0.82, 0.0, 0.4, 0.6, 0.26, m["iron"])
    for sx in (-1, 1):                                        # stepped angular armour
        add_box("bl_p1", (sx * 0.25, 0.0, 0.0), (0.1, 0.9, 0.3), m["dark"], taper=0.8)
        add_box("bl_p2", (sx * 0.3, -0.2, 0.04), (0.07, 0.5, 0.24), m["steel"], taper=0.85)
    add_box("bl_str", (0, -0.1, 0.15), (0.12, 0.8, 0.04), m["red"], bevel=0.004)
    add_box("bl_slit", (0, 0.28, 0.16), (0.08, 0.18, 0.04), m["slit"])
    _iron_plates("bl", (0.34, 0.04, -0.3), 0.24, 0.18, 0.15, m)
    # long forward chainguns + a dorsal chaingun turret
    for sx in (-0.12, 0.12):
        add_cylinder("bl_gun", (sx, 0.7, 0.07), 0.032, 0.6, m["steel"], r2=0.026)
    add_box("bl_turr", (0, -0.3, 0.18), (0.18, 0.2, 0.12), m["dark"], bevel=0.01)
    for sx in (-0.05, 0.05):
        add_cylinder("bl_tg", (sx, -0.12, 0.2), 0.022, 0.34, m["steel"], r2=0.018)
    for sx in (-0.12, 0.12):
        _slug_drive("bl%d" % int(sx * 10), sx, -0.78, 0.07, m)


def build_siege_monitor():
    m = _bastion_mats("sm")
    # a hull built AROUND one enormous forward siege cannon
    add_box("sm_body", (0, -0.3, 0.0), (0.42, 1.0, 0.3), m["iron"], taper=0.7)
    for sx in (-1, 1):
        add_box("sm_sponson", (sx * 0.3, -0.2, 0.0), (0.16, 0.6, 0.34), m["dark"], taper=0.8)
        add_cylinder("sm_sg", (sx * 0.3, 0.2, 0.06), 0.03, 0.5, m["steel"], r2=0.024)
    add_box("sm_str", (0, -0.3, 0.17), (0.14, 0.5, 0.04), m["red"], bevel=0.004)
    add_box("sm_slit", (0, -0.05, 0.18), (0.09, 0.16, 0.04), m["slit"])
    _iron_plates("sm", (-0.12, -0.42), 0.26, 0.18, 0.16, m)
    # the giant siege cannon: a long heavy barrel down the spine, outranging all
    add_box("sm_mount", (0, 0.25, 0.1), (0.18, 0.5, 0.2), m["dark"], taper=0.85)
    add_cylinder("sm_cannon", (0, 0.85, 0.12), 0.06, 1.0, m["steel"], r2=0.05)
    add_cylinder("sm_muzzle", (0, 1.3, 0.12), 0.07, 0.12, m["dark"], r2=0.065)
    for sx in (-0.14, 0.14):
        _slug_drive("sm%d" % int(sx * 10), sx, -0.82, 0.08, m)


def build_iron_dreadnought():
    m = _bastion_mats("id")
    # broad faceted battlewagon, ram prow, dorsal siege turrets, banner masts
    add_box("id_body", (0, -0.1, 0.0), (0.62, 1.5, 0.36), m["iron"], taper=0.66)
    _wedge("id_prow", 0, 0.95, 0.0, 0.62, 0.7, 0.36, m["iron"])
    for sx in (-1, 1):                                        # layered belt armour
        add_box("id_belt", (sx * 0.36, -0.1, 0.0), (0.12, 1.1, 0.42), m["dark"], taper=0.8)
        add_box("id_belt2", (sx * 0.42, -0.15, 0.05), (0.08, 0.7, 0.3), m["steel"], taper=0.85)
    for yc in (0.35, -0.05, -0.45):
        add_box("id_str", (0, yc, 0.2), (0.4, 0.05, 0.05), m["red"], bevel=0.005)
    # red imperial heraldry emblem amidships
    _wedge("id_herald", 0, 0.18, 0.24, 0.2, 0.26, 0.08, m["red"])
    add_box("id_cit", (0, -0.05, 0.24), (0.26, 0.4, 0.16), m["dark"], taper=0.7)
    add_box("id_slit", (0, 0.1, 0.34), (0.1, 0.16, 0.04), m["slit"])
    # dorsal siege turrets (boxy) with twin long barrels
    for yc in (0.45, -0.35):
        add_box("id_turr", (0, yc, 0.3), (0.24, 0.22, 0.14), m["dark"], bevel=0.012)
        for sx in (-0.06, 0.06):
            add_cylinder("id_tg", (sx, yc + 0.34, 0.32), 0.028, 0.5, m["steel"], r2=0.022)
    # banner masts on the flanks
    for sx in (-0.42, 0.42):
        add_box("id_mast", (sx, -0.5, 0.34), (0.03, 0.04, 0.4), m["steel"])
        add_box("id_banner", (sx, -0.5, 0.5), (0.02, 0.16, 0.18), m["red"])
    for sx in (-0.3, 0.0, 0.3):
        _slug_drive("id%d" % int(sx * 10 + 5), sx, -1.0, 0.09, m)


# ═══════════════════ Artifact Order (steampunk-gothic) ═════════════════════
# Reliquary ships: cream liturgical hulls built around an exposed, venerated
# violet Precursor relic-core. Brass/bronze pipes, gears & buttress ribs
# (steampunk); symmetric, ornate, banner masts & stained glass (gothic). Drives
# bleed gold->violet (toward the Precursor palette). Relic lances + martyr rams.
def _order_mats(tag):
    return dict(
        cream=toon_material(tag + "c", C(236, 229, 211), spec=0.8),
        brass=toon_material(tag + "br", C(194, 152, 80), spec=1.3, spec_sharp=0.8),
        bronze=toon_material(tag + "bz", C(138, 100, 52)),
        gold=toon_material(tag + "au", C(222, 186, 96), spec=1.3),
        violet=glow_material(tag + "v", C(150, 75, 228), 8),     # relic core
        vglass=toon_material(tag + "g", C(160, 110, 214), spec=1.6, glass=True),
        drive=glow_material(tag + "e", C(200, 120, 210), 6),     # gold-violet
    )


def _relic(name, cx, cy, cz, r, m, ribs=8):
    """Exposed violet relic-core haloed by a brass disc + gold-studded rose-window
    ribs — the venerated heart of every Order hull."""
    add_cylinder(name + "halo", (cx, cy, cz - 0.05), r * 1.7, 0.035, m["brass"], axis="z", seg=24)
    add_cylinder(name + "ring", (cx, cy, cz - 0.02), r * 1.25, 0.05, m["bronze"], axis="z", seg=24)
    add_sphere(name, (cx, cy, cz), (r, r * 1.15, r * 0.85), m["violet"])
    for a in range(ribs):
        ang = 2 * math.pi * a / ribs
        rx, ry = cx + r * 1.45 * math.cos(ang), cy + r * 1.45 * math.sin(ang)
        add_box(name + "rib%d" % a, (rx, ry, cz - 0.01), (0.04, 0.04, 0.1), m["brass"], bevel=0.006)
        add_sphere(name + "st%d" % a, (rx, ry, cz + 0.05), (0.024, 0.024, 0.02), m["gold"])


def _pipes(tag, cy, l, m, xs=(-0.09, 0.09), z=0.08, r=0.024):
    for i, sx in enumerate(xs):
        add_cylinder(tag + "pp%d" % i, (sx, cy, z), r, l, m["brass"], axis="y", seg=10)


def _gear(name, cx, cy, cz, r, m, teeth=8):
    add_cylinder(name, (cx, cy, cz), r, 0.04, m["brass"], axis="z", seg=20)
    add_cylinder(name + "h", (cx, cy, cz + 0.01), r * 0.4, 0.05, m["bronze"], axis="z", seg=12)
    for a in range(teeth):
        ang = 2 * math.pi * a / teeth
        add_box(name + "t%d" % a, (cx + r * 1.1 * math.cos(ang), cy + r * 1.1 * math.sin(ang), cz),
                (0.02, 0.02, 0.03), m["brass"])


def build_order_acolyte():
    m = _order_mats("oa")
    loft_hull("oa_h", [
        dict(y=0.95, w=0.03, h=0.04, cz=0),
        dict(y=0.55, w=0.1, h=0.11, cz=0.02),
        dict(y=0.1, w=0.13, h=0.13, cz=0.02),
        dict(y=-0.45, w=0.1, h=0.1, cz=0.01),
        dict(y=-0.85, w=0.06, h=0.07, cz=0),
    ], m["cream"], m=14, n=2.4, flatten=0.6, subsurf=2)
    add_sphere("oa_tip", (0, 0.82, 0.03), (0.05, 0.11, 0.05), m["violet"])   # ram relic-tip
    add_cylinder("oa_collar", (0, 0.64, 0.03), 0.065, 0.06, m["brass"])
    add_sphere("oa_can", (0, 0.3, 0.13), (0.07, 0.09, 0.07), m["vglass"], zclip=0.13)
    _pipes("oa", 0.0, 0.6, m)
    _relic("oa_core", 0, -0.12, 0.11, 0.06, m, ribs=6)
    add_box("oa_fil", (0, 0.05, 0.14), (0.02, 0.5, 0.02), m["gold"])
    add_cylinder("oa_nz", (0, -0.85, 0.0), 0.05, 0.08, m["bronze"], r2=0.06)
    plume("oa_pl", 0, -0.9, 0, 0.045, 0.28, 0.8, m["drive"])


def build_order_censer():
    m = _order_mats("oc")
    loft_hull("oc_h", [
        dict(y=0.85, w=0.06, h=0.08, cz=0),
        dict(y=0.4, w=0.18, h=0.15, cz=0.02),
        dict(y=-0.05, w=0.2, h=0.16, cz=0.02),
        dict(y=-0.5, w=0.15, h=0.13, cz=0.01),
        dict(y=-0.85, w=0.08, h=0.08, cz=0),
    ], m["cream"], m=16, n=2.5, flatten=0.55, subsurf=2)
    add_sphere("oc_can", (0, 0.34, 0.16), (0.09, 0.12, 0.08), m["vglass"], zclip=0.16)
    add_box("oc_filig", (0, 0.5, 0.15), (0.16, 0.04, 0.04), m["gold"], bevel=0.01)
    _relic("oc_core", 0, 0.0, 0.16, 0.07, m)
    # swinging brass censer-pods on side arms (the guided-bomb ordnance)
    for sx in (-1, 1):
        add_box("oc_arm", (sx * 0.26, 0.18, 0.05), (0.18, 0.05, 0.05), m["bronze"], bevel=0.01)
        add_sphere("oc_censer", (sx * 0.36, 0.16, 0.02), (0.07, 0.08, 0.07), m["brass"])
        add_sphere("oc_cglow", (sx * 0.36, 0.16, 0.06), (0.03, 0.03, 0.03), m["violet"])
    _pipes("oc", -0.1, 0.7, m, xs=(-0.13, 0.13))
    add_box("oc_fil", (0, -0.2, 0.16), (0.02, 0.5, 0.02), m["gold"])
    _gear("oc_g", 0, -0.5, 0.14, 0.05, m)
    for sx in (-0.1, 0.1):
        add_cylinder("oc_nz", (sx, -0.86, 0.0), 0.055, 0.1, m["bronze"], r2=0.066)
        plume("oc_pl", sx, -0.92, 0, 0.05, 0.28, 0.8, m["drive"])


def build_order_reliquary():
    m = _order_mats("or")
    loft_hull("or_h", [
        dict(y=0.95, w=0.08, h=0.1, cz=0),
        dict(y=0.45, w=0.24, h=0.17, cz=0.02),
        dict(y=-0.05, w=0.28, h=0.18, cz=0.02),
        dict(y=-0.55, w=0.22, h=0.15, cz=0.01),
        dict(y=-0.95, w=0.1, h=0.1, cz=0),
    ], m["cream"], m=16, n=2.6, flatten=0.55, subsurf=2)
    add_sphere("or_can", (0, 0.5, 0.18), (0.08, 0.1, 0.07), m["vglass"], zclip=0.18)
    # BIG exposed relic-core amidships, brass rose-window frame (the showcase)
    _relic("or_core", 0, 0.0, 0.18, 0.13, m, ribs=8)
    for sx in (-1, 1):
        for yc in (0.3, -0.05, -0.4):
            add_box("or_butt", (sx * 0.26, yc, 0.12), (0.06, 0.14, 0.16), m["brass"], bevel=0.01)
    _pipes("or", -0.1, 0.9, m, xs=(-0.17, 0.17), z=0.1)
    add_box("or_fil", (0, -0.45, 0.2), (0.025, 0.4, 0.025), m["gold"])
    for _sx in (-0.18, 0.18):
        _gear("or_g%d" % int(_sx*10), _sx, 0.45, 0.16, 0.05, m)
    for sx in (-0.26, 0.26):
        add_box("or_mast", (sx, -0.6, 0.2), (0.02, 0.03, 0.24), m["bronze"])
        add_box("or_banner", (sx, -0.6, 0.32), (0.015, 0.12, 0.14), m["violet"])
    for sx in (-0.13, 0.13):
        add_cylinder("or_nz", (sx, -0.96, 0.0), 0.07, 0.12, m["bronze"], r2=0.085)
        plume("or_pl", sx, -1.02, 0, 0.06, 0.3, 0.8, m["drive"])


def build_order_cathedral():
    m = _order_mats("ot")
    loft_hull("ot_h", [
        dict(y=1.15, w=0.08, h=0.1, cz=0),
        dict(y=0.6, w=0.3, h=0.2, cz=0.02),
        dict(y=0.0, w=0.36, h=0.22, cz=0.02),
        dict(y=-0.6, w=0.3, h=0.19, cz=0.01),
        dict(y=-1.15, w=0.12, h=0.11, cz=0),
    ], m["cream"], m=18, n=2.7, flatten=0.5, subsurf=2)
    # three relic-cores down the nave (the temple-ship)
    _relic("ot_c0", 0, 0.45, 0.2, 0.1, m)
    _relic("ot_c1", 0, 0.0, 0.22, 0.14, m, ribs=8)
    _relic("ot_c2", 0, -0.5, 0.2, 0.1, m)
    add_sphere("ot_can", (0, 0.78, 0.2), (0.08, 0.11, 0.07), m["vglass"], zclip=0.2)
    add_box("ot_spine", (0, 0.0, 0.26), (0.05, 1.7, 0.03), m["gold"])
    for sx in (-1, 1):
        for yc in (0.5, 0.1, -0.3, -0.7):
            add_box("ot_col", (sx * 0.32, yc, 0.14), (0.07, 0.1, 0.2), m["brass"], bevel=0.01)
    _pipes("ot", 0.0, 1.1, m, xs=(-0.22, 0.22), z=0.12)
    for _sx in (-0.22, 0.22):
        _gear("ot_g%d" % int(_sx*10), _sx, -0.85, 0.16, 0.055, m)
    for sx in (-0.34, 0.34):
        add_box("ot_mast", (sx, -0.2, 0.26), (0.025, 0.035, 0.34), m["bronze"])
        add_box("ot_banner", (sx, -0.2, 0.44), (0.018, 0.18, 0.2), m["violet"])
    for sx in (-0.26, 0.0, 0.26):
        add_cylinder("ot_nz", (sx, -1.16, 0.0), 0.08, 0.14, m["bronze"], r2=0.1)
        plume("ot_pl", sx, -1.24, 0, 0.07, 0.32, 0.8, m["drive"])


# ════════════════════════ Precursors (alien geometry) ══════════════════════
# Not human: NO cockpit, NO drive plume, NO bilateral nose-tail. Radial
# crystalline forms grown from exotic matter — glowing violet cores, faceted
# shards that HOVER apart with gaps, a gravitic HALO ring instead of exhaust.
# Iridescent black + violet. Enemy/boss ships only (never sold). "Wrong & beautiful."
def _prec_mats(tag):
    return dict(
        black=toon_material(tag + "k", C(36, 30, 54), spec=1.6, spec_sharp=0.7),
        crystal=toon_material(tag + "cr", C(120, 86, 210), spec=1.9, spec_sharp=0.7, glass=True),
        core=glow_material(tag + "co", C(160, 55, 250), 7),
        halo=glow_material(tag + "ha", C(150, 75, 245), 7),
    )


def _crystal(name, cx, cy, cz, w, l, h, mat):
    """A faceted crystal shard (octahedron) — angular, non-human."""
    hw, hl, hh = w / 2, l / 2, h / 2
    v = [(cx, cy + hl, cz), (cx, cy - hl, cz), (cx + hw, cy, cz), (cx - hw, cy, cz),
         (cx, cy, cz + hh), (cx, cy, cz - hh)]
    f = [(4, 0, 2), (4, 2, 1), (4, 1, 3), (4, 3, 0),
         (5, 2, 0), (5, 1, 2), (5, 3, 1), (5, 0, 3)]
    _obj_from_pydata(name, v, f, mat, smooth=False)


def _halo(name, cx, cy, cz, r, mat, seg=28):
    """A gravitic halo ring of glowing segments (no exhaust)."""
    for a in range(seg):
        ang = 2 * math.pi * a / seg
        add_box(name + "%d" % a, (cx + r * math.cos(ang), cy + r * math.sin(ang), cz),
                (0.028, 0.028, 0.02), mat)


def build_precursor_seeker():
    m = _prec_mats("pk")
    add_sphere("pk_core", (0, 0, 0.0), (0.08, 0.08, 0.07), m["core"])
    _halo("pk_h", 0, 0, 0.0, 0.2, m["halo"], seg=18)
    for a in range(3):                       # three hovering shards (radial)
        ang = 2 * math.pi * a / 3
        _crystal("pk_s%d" % a, 0.3 * math.cos(ang), 0.3 * math.sin(ang), 0.0,
                 0.1, 0.26, 0.1, m["crystal"])


def build_precursor_sentinel():
    m = _prec_mats("ps")
    add_sphere("ps_core", (0, 0, 0.0), (0.13, 0.13, 0.1), m["core"])
    _halo("ps_h", 0, 0, 0.0, 0.34, m["halo"], seg=26)
    for a in range(6):                       # outer ring of radial shards (gaps)
        ang = 2 * math.pi * a / 6
        _crystal("ps_s%d" % a, 0.52 * math.cos(ang), 0.52 * math.sin(ang), 0.0,
                 0.11, 0.28, 0.11, m["crystal"])
    for a in range(3):                       # inner floating monolith segments
        ang = 2 * math.pi * a / 3 + 0.52
        _crystal("ps_m%d" % a, 0.24 * math.cos(ang), 0.24 * math.sin(ang), 0.04,
                 0.09, 0.16, 0.16, m["black"])


def build_precursor_harbinger():
    m = _prec_mats("ph")
    add_sphere("ph_core", (0, 0.0, 0.0), (0.16, 0.18, 0.12), m["core"])
    _halo("ph_h1", 0, 0, 0.0, 0.4, m["halo"], seg=30)
    _halo("ph_h2", 0, 0, 0.06, 0.26, m["halo"], seg=20)
    # a hunter-form: longer fore shard (the "lance") + radial shards
    _crystal("ph_lance", 0, 0.7, 0.02, 0.12, 0.6, 0.12, m["crystal"])
    for a in range(5):
        ang = 2 * math.pi * a / 5 + math.pi / 2
        _crystal("ph_s%d" % a, 0.58 * math.cos(ang), 0.58 * math.sin(ang), 0.0,
                 0.12, 0.3, 0.12, m["crystal"])
    for a in range(4):
        ang = 2 * math.pi * a / 4 + 0.4
        _crystal("ph_m%d" % a, 0.3 * math.cos(ang), 0.3 * math.sin(ang), 0.05,
                 0.1, 0.2, 0.18, m["black"])


def build_precursor_sleeper():
    m = _prec_mats("pz")
    # the leviathan boss: concentric halos, many cores, a vast shard mandala
    add_sphere("pz_core", (0, 0, 0.0), (0.24, 0.26, 0.16), m["core"])
    _halo("pz_h1", 0, 0, 0.0, 0.95, m["halo"], seg=44)
    _halo("pz_h2", 0, 0, 0.05, 0.62, m["halo"], seg=34)
    _halo("pz_h3", 0, 0, 0.1, 0.34, m["halo"], seg=22)
    for a in range(8):                       # outer mandala of great shards
        ang = 2 * math.pi * a / 8
        _crystal("pz_o%d" % a, 0.85 * math.cos(ang), 0.85 * math.sin(ang), 0.0,
                 0.16, 0.42, 0.16, m["crystal"])
    for a in range(6):                       # mid satellite cores
        ang = 2 * math.pi * a / 6 + 0.3
        add_sphere("pz_sc%d" % a, (0.5 * math.cos(ang), 0.5 * math.sin(ang), 0.02),
                   (0.07, 0.07, 0.06), m["core"])
    for a in range(5):                       # inner black monoliths
        ang = 2 * math.pi * a / 5 + 0.6
        _crystal("pz_m%d" % a, 0.3 * math.cos(ang), 0.3 * math.sin(ang), 0.08,
                 0.13, 0.24, 0.22, m["black"])


# ════════════════════════════════ registry ═════════════════════════════════
REGISTRY = {
    # name: (builder, ortho, group, label)
    "shuttle": (build_shuttle, 1.9, "merchant", "r10 · starter pod"),
    "courier": (build_courier, 2.5, "merchant", "r12 · parcel needle"),
    "prospector": (build_prospector, 2.2, "merchant", "r14 · drill-spike dart"),
    "asteroid_miner": (build_asteroid_miner, 2.4, "merchant", "r20 · drill-arm crab"),
    "cargo_transport": (build_cargo_transport, 2.2, "merchant", "r20 · boxy container truck"),
    "freighter": (build_freighter, 2.5, "merchant", "r32 · side-module hauler"),
    "hauler": (build_hauler, 2.6, "merchant", "r40 · container spine"),
    "bulk_carrier": (build_bulk_carrier, 2.7, "merchant", "r55 · cargo-pod brick"),
    "fighter": (build_fighter, 2.5, "independent", "r12 · delta + canards"),
    "corvette": (build_corvette, 2.2, "independent", "r9 · needle blade"),
    "frigate": (build_frigate, 2.6, "independent", "r32 · surplus warship"),
    "surplus_carrier": (build_surplus_carrier, 2.7, "independent", "r45 · neutral wedge"),
    "fed_patrol": (build_fed_patrol, 2.3, "federation", "r15 · armored dart"),
    "fed_destroyer": (build_fed_destroyer, 2.6, "federation", "r42 · slab battlewagon"),
    "fed_missile_cruiser": (build_fed_missile_cruiser, 2.6, "federation", "r35 · all-tubes cruiser"),
    "fed_carrier": (build_fed_carrier, 2.7, "federation", "r60 · flat-top flagship"),
    "rebel_fighter": (build_rebel_fighter, 2.6, "rebels", "r12 · winged dart"),
    "rebel_gunboat": (build_rebel_gunboat, 2.5, "rebels", "r22 · twin gun booms"),
    "rebel_frigate": (build_rebel_frigate, 2.6, "rebels", "r28 · fwd-swept nacelles"),
    "rebel_carrier": (build_rebel_carrier, 2.8, "rebels", "r55 · lattice deck"),
    "pirate_corvette": (build_pirate_corvette, 2.1, "pirates", "r11 · scrap blade"),
    "pirate_missile_boat": (build_pirate_missile_boat, 2.4, "pirates", "r18 · bolt-on racks"),
    "pirate_carrier": (build_pirate_carrier, 2.8, "pirates", "r50 · welded hulk"),
    "frontier_skiff": (build_frontier_skiff, 2.1, "frontier", "r11 · sail interceptor"),
    "frontier_harvester": (build_frontier_harvester, 2.6, "frontier", "r24 · boom gunboat"),
    "frontier_sailtender": (build_frontier_sailtender, 2.7, "frontier", "r20 · outrigger sails"),
    "frontier_monitor": (build_frontier_monitor, 2.9, "frontier", "r40 · sun-shield fort"),
    "helios_drone": (build_helios_drone, 1.5, "helios", "r7 · swarm wedge"),
    "helios_enforcer": (build_helios_enforcer, 2.5, "helios", "r16 · security gunship"),
    "helios_overseer": (build_helios_overseer, 2.8, "helios", "r28 · drone command"),
    "helios_titan": (build_helios_titan, 3.0, "helios", "r50 · factory carrier"),
    "bastion_guard": (build_bastion_guard, 2.3, "bastion", "r15 · iron patrol"),
    "bastion_lance": (build_bastion_lance, 2.6, "bastion", "r24 · chaingun frigate"),
    "siege_monitor": (build_siege_monitor, 2.7, "bastion", "r34 · spinal siege cannon"),
    "iron_dreadnought": (build_iron_dreadnought, 3.0, "bastion", "r48 · battlewagon"),
    "order_acolyte": (build_order_acolyte, 2.3, "order", "r12 · martyr ram-dart"),
    "order_censer": (build_order_censer, 2.5, "order", "r20 · censer gunship"),
    "order_reliquary": (build_order_reliquary, 2.7, "order", "r30 · relic-core cruiser"),
    "order_cathedral": (build_order_cathedral, 3.0, "order", "r48 · temple flagship"),
    "precursor_seeker": (build_precursor_seeker, 1.8, "precursor", "r10 · shard drone"),
    "precursor_sentinel": (build_precursor_sentinel, 2.4, "precursor", "r18 · radial guardian"),
    "precursor_harbinger": (build_precursor_harbinger, 2.8, "precursor", "r30 · hunter-form"),
    "precursor_sleeper": (build_precursor_sleeper, 3.2, "precursor", "r60 · leviathan boss"),
}


def render_ship(name, yaw=0.0, thrust=True, suffix=""):
    builder, ortho, _, _ = REGISTRY[name]
    reset()
    builder()
    _rotate_ship(yaw)
    setup_scene(ortho, 256)
    _set_exhaust_visible(thrust)
    render_to(os.path.join(OUT, f"fleet_{name}{suffix}.png"))


def _cell(path, cell):
    im = Image.open(path).convert("RGBA")
    im.thumbnail((cell, cell), Image.LANCZOS)
    c = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
    c.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2), im)
    return c


def montage(group, names, cols=4):
    cell = 210; pad = 12; lblh = 20
    rows = (len(names) + cols - 1) // cols
    W = cell * cols + pad * (cols + 1)
    H = (cell + lblh) * rows + pad * (rows + 1) + 24
    cv = Image.new("RGBA", (W, H), (34, 36, 44, 255))
    d = ImageDraw.Draw(cv)
    d.text((pad, 6), group.upper(), fill=(240, 240, 250, 255))
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = 24 + pad + r * (cell + lblh + pad)
        d.text((x + 4, y), name, fill=(235, 235, 245, 255))
        d.text((x + 4, y + 10), REGISTRY[name][3], fill=(165, 170, 185, 255))
        cv.paste(_cell(os.path.join(OUT, f"fleet_{name}.png"), cell), (x, y + lblh),
                 _cell(os.path.join(OUT, f"fleet_{name}.png"), cell))
    cv.save(os.path.join(OUT, f"_fleet_{group}.png"))
    print("saved", f"_fleet_{group}.png")


def feature_demos():
    """Prove the two pipeline features on one ship."""
    # idle vs thrust (exhaust layer toggle)
    render_ship("rebel_fighter", thrust=False, suffix="_idle")
    render_ship("rebel_fighter", thrust=True, suffix="_thrust")
    # ring of headings (dynamic light) — note exhaust off so we read the hull
    angles = [0, 30, 60, 90]
    for a in angles:
        render_ship("fed_destroyer", yaw=math.radians(a), thrust=False,
                    suffix=f"_yaw{a}")
    # compose demo strips
    cell = 210; pad = 12
    pair = Image.new("RGBA", (cell * 2 + pad * 3, cell + pad * 2 + 20), (34, 36, 44, 255))
    dd = ImageDraw.Draw(pair)
    for j, (sfx, lbl) in enumerate(((("_idle"), "drive OFF"), (("_thrust"), "drive ON"))):
        dd.text((pad + j * (cell + pad), 4), lbl, fill=(235, 235, 245, 255))
        pair.paste(_cell(os.path.join(OUT, f"fleet_rebel_fighter{sfx}.png"), cell),
                   (pad + j * (cell + pad), 20),
                   _cell(os.path.join(OUT, f"fleet_rebel_fighter{sfx}.png"), cell))
    pair.save(os.path.join(OUT, "_demo_exhaust.png"))
    strip = Image.new("RGBA", (cell * 4 + pad * 5, cell + pad * 2 + 20), (34, 36, 44, 255))
    ds = ImageDraw.Draw(strip)
    for j, a in enumerate(angles):
        ds.text((pad + j * (cell + pad), 4), f"heading {a}°", fill=(235, 235, 245, 255))
        strip.paste(_cell(os.path.join(OUT, f"fleet_fed_destroyer_yaw{a}.png"), cell),
                    (pad + j * (cell + pad), 20),
                    _cell(os.path.join(OUT, f"fleet_fed_destroyer_yaw{a}.png"), cell))
    strip.save(os.path.join(OUT, "_demo_angles.png"))
    print("saved _demo_exhaust.png _demo_angles.png")


def main():
    for name in REGISTRY:
        render_ship(name)
        print("rendered", name)
    groups = {}
    for name, (_, _, g, _) in REGISTRY.items():
        groups.setdefault(g, []).append(name)
    for g, names in groups.items():
        montage(g, names)
    feature_demos()


if __name__ == "__main__":
    main()
