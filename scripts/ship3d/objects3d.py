"""
objects3d.py — 3D toon world-object prototype (trees, rocks…), same toon + ink
language as the ships/buildings. Forward-facing 3/4 by default (matches the
buildings) so objects "stand up" on the top-down surface.

Run:  scripts/.blender_venv/bin/python objects3d.py
"""
import math
import os

import bpy  # noqa
from PIL import Image

import blender_gen as B

OUT = os.path.join(os.path.dirname(__file__), "out")
RES = 192


def C(*v):
    return tuple(x / 255.0 for x in v)


def mats():
    return dict(  # matte (spec=0): the toon banding carries the form, no white specks
        bark=B.toon_material("bark", C(108, 74, 44)),
        bark_d=B.toon_material("bark_d", C(78, 50, 30)),
        birchbark=B.toon_material("birchbark", C(198, 200, 192)),
        stem=B.toon_material("stem", C(214, 206, 188)),
        leaf=B.toon_material("leaf", C(64, 134, 54)),
        leaf_d=B.toon_material("leaf_d", C(42, 98, 42)),
        leaf_l=B.toon_material("leaf_l", C(112, 176, 84)),
        pine=B.toon_material("pine", C(42, 98, 52)),
        pine_l=B.toon_material("pine_l", C(70, 130, 76)),
        stone=B.toon_material("stone", C(108, 104, 96)),
        stone_d=B.toon_material("stone_d", C(80, 76, 70)),
        moss=B.toon_material("moss", C(78, 128, 58)),
        dead=B.toon_material("dead", C(124, 100, 74)),
        mush=B.toon_material("mush", C(190, 92, 72)),
        dark=B.toon_material("dark", C(40, 40, 44)),
        # water
        reedg=B.toon_material("reedg", C(96, 150, 70)),
        cattail=B.toon_material("cattail", C(120, 84, 50)),
        lily=B.toon_material("lily", C(232, 168, 196)),
        fishc=B.toon_material("fishc", C(150, 168, 196)),
        seaw=B.toon_material("seaw", C(46, 110, 70)),
        # sand
        shell=B.toon_material("shell", C(224, 196, 178)),
        drift=B.toon_material("drift", C(176, 166, 150)),
        # grass flowers
        fl_r=B.toon_material("fl_r", C(226, 92, 104)),
        fl_y=B.toon_material("fl_y", C(236, 206, 96)),
        fl_p=B.toon_material("fl_p", C(186, 132, 222)),
        # mountain
        scrub=B.toon_material("scrub", C(96, 120, 70)),
        # creatures
        fur=B.toon_material("fur", C(150, 96, 58)),
        fur_l=B.toon_material("fur_l", C(186, 134, 92)),
        nest=B.toon_material("nest", C(132, 100, 64)),
        nest_d=B.toon_material("nest_d", C(70, 52, 34)),
        bird=B.toon_material("bird", C(96, 130, 196)),
        bird_d=B.toon_material("bird_d", C(64, 96, 160)),
        belly=B.toon_material("belly", C(196, 150, 110)),
        beak=B.toon_material("beak", C(228, 176, 80)),
        frog=B.toon_material("frog", C(96, 160, 78)),
        frog_l=B.toon_material("frog_l", C(150, 196, 110)),
        alien=B.toon_material("alien", C(140, 196, 130)),
        # ── rocky / volcanic ──
        # bubbles read as molten lava (same red-orange as the lava tile, just a
        # hair brighter/raised) — no bright highlight that pops off the surface.
        lava=B.glow_material("lava", C(214, 76, 28), strength=1.2),
        lava2=B.glow_material("lava2", C(238, 106, 42), strength=1.4),
        ember=B.glow_material("ember", C(200, 62, 24), strength=1.2),
        vrock=B.toon_material("vrock", C(92, 84, 80)),
        vrock_d=B.toon_material("vrock_d", C(58, 52, 50)),
        vrock_l=B.toon_material("vrock_l", C(122, 112, 106)),
        basalt=B.toon_material("basalt", C(64, 60, 62)),
        ore=B.glow_material("ore", C(120, 230, 240), strength=2.2),
        oremat=B.toon_material("oremat", C(150, 130, 70)),
        rmetal=B.toon_material("rmetal", C(120, 96, 78), spec=0.8, spec_sharp=0.85),
        rmetal_d=B.toon_material("rmetal_d", C(80, 64, 52)),
        steam=B.toon_material("steam", C(176, 176, 182)),
        ash=B.toon_material("ash", C(120, 122, 88)),
        sulfur=B.toon_material("sulfur", C(206, 196, 96)),
        crfur=B.toon_material("crfur", C(70, 60, 58)),
        sal=B.toon_material("sal", C(220, 120, 60)),
        sal_l=B.toon_material("sal_l", C(255, 180, 90)),
    )


def build_oak(m):
    B.add_cylinder("trunk", (0, 0, 0.55), 0.16, 1.15, m["bark"], axis="z", r2=0.12)
    for i, (x, y, z, r, k) in enumerate(   # deeper-green than the birch/willow
        [(0, 0, 1.55, 0.92, "leaf_d"), (-0.55, 0.15, 1.45, 0.6, "leaf_d"),
         (0.55, -0.1, 1.45, 0.6, "leaf"), (0.1, 0.35, 2.0, 0.5, "leaf"),
         (-0.15, -0.35, 1.7, 0.55, "leaf_d"), (0.35, 0.25, 1.85, 0.45, "leaf")]):
        B.add_sphere(f"oak{i}", (x, y, z), (r, r * 0.9, r), m[k])


def build_conifer(m):
    B.add_cylinder("ctrunk", (0, 0, 0.4), 0.12, 0.8, m["bark"], axis="z")
    for i, (z, r, k) in enumerate([(0.9, 0.85, "pine"), (1.4, 0.62, "pine_l"),
                                   (1.85, 0.4, "pine"), (2.2, 0.2, "pine_l")]):
        B.add_cylinder(f"cone{i}", (0, 0, z), r, 0.55, m[k], axis="z", r2=0.02, seg=18)


def build_birch(m):
    B.add_cylinder("btrunk", (0, 0, 0.7), 0.12, 1.5, m["birchbark"], axis="z")
    B.add_box("bm1", (0, -0.13, 0.9), (0.25, 0.04, 0.05), m["bark_d"])    # bark marks
    B.add_box("bm2", (0, -0.13, 1.3), (0.25, 0.04, 0.05), m["bark_d"])
    for i, (x, y, z, r, k) in enumerate(
        [(0, 0, 1.8, 0.68, "leaf_l"), (-0.35, 0.12, 1.68, 0.45, "leaf"),
         (0.35, -0.1, 1.72, 0.45, "leaf_l"), (0, 0.18, 2.05, 0.4, "leaf_l")]):
        B.add_sphere(f"bch{i}", (x, y, z), (r, r * 0.9, r), m[k])


def build_willow(m):
    B.add_cylinder("wtrunk", (0, 0, 0.5), 0.15, 1.0, m["bark"], axis="z")
    B.add_sphere("wcan", (0, 0, 1.5), (0.82, 0.78, 0.78), m["leaf_l"])    # rounded weeping crown
    for a in range(10):                                                   # long hanging fronds
        th = a / 10 * 2 * math.pi
        x, y = 0.78 * math.cos(th), 0.7 * math.sin(th)
        B.add_cylinder(f"wf{a}", (x, y, 1.0), 0.05, 1.15, m["leaf"], axis="z", r2=0.02)


def build_dead_tree(m):
    B.add_cylinder("dtr", (0, 0, 0.78), 0.14, 1.56, m["dead"], axis="z", r2=0.04)
    for i, (z, ln, sx) in enumerate(
        [(0.95, 0.62, -1), (1.12, 0.52, 1), (1.32, 0.46, -1), (1.46, 0.36, 1), (1.56, 0.3, -1)]):
        B.add_cylinder(f"db{i}", (sx * ln * 0.45, 0, z), 0.045, ln, m["dead"], axis="x", r2=0.004)


def build_bush(m):
    for i, (x, y, z, r, k) in enumerate(
        [(0, 0, 0.32, 0.42, "leaf"), (-0.3, 0.1, 0.28, 0.3, "leaf_d"),
         (0.3, -0.05, 0.3, 0.32, "leaf"), (0.05, 0.2, 0.45, 0.28, "leaf_l")]):
        B.add_sphere(f"bush{i}", (x, y, z), (r, r * 0.9, r), m[k])


def build_fern(m):
    for a in range(6):
        th = a / 6 * 2 * math.pi
        x, y = 0.35 * math.cos(th), 0.35 * math.sin(th)
        B.add_cylinder(f"frond{a}", (x * 0.6, y * 0.6, 0.32), 0.05, 0.7, m["leaf_d"],
                       axis="z", r2=0.005)
    B.add_sphere("ferncore", (0, 0, 0.2), (0.18, 0.18, 0.12), m["leaf"])


def build_mushroom(m):
    for i, (x, y, h, r) in enumerate([(0, 0, 0.34, 0.2), (0.22, 0.05, 0.24, 0.14), (-0.15, 0.1, 0.2, 0.12)]):
        B.add_cylinder(f"stem{i}", (x, y, h * 0.5), 0.05, h, m["birchbark"], axis="z")
        B.add_sphere(f"cap{i}", (x, y, h), (r, r, r * 0.7), m["mush"], zclip=h - 0.01)


def build_rock(m):                                                       # angular faceted stone
    B.add_box("rk1", (0, 0, 0.24), (0.72, 0.56, 0.46), m["stone"], bevel=0.1)
    B.add_box("rk2", (0.3, 0.1, 0.13), (0.34, 0.3, 0.26), m["stone_d"], bevel=0.08)
    B.add_box("rk3", (-0.28, -0.12, 0.16), (0.26, 0.24, 0.32), m["stone"], bevel=0.08)
    B.add_box("moss", (-0.05, 0.05, 0.47), (0.32, 0.24, 0.04), m["moss"], bevel=0.02)  # thin moss cap


# ── water ──
def build_reed(m):
    for a, (x, y) in enumerate([(-0.15, 0), (0.1, 0.1), (0.0, -0.12), (0.2, -0.05)]):
        B.add_cylinder(f"reed{a}", (x, y, 0.6), 0.045, 1.2, m["reedg"], axis="z", r2=0.02)
        B.add_cylinder(f"cat{a}", (x, y, 1.12), 0.08, 0.3, m["cattail"], axis="z", r2=0.06)


def build_lilypad(m):
    B.add_cylinder("pad", (0, 0, 0.05), 0.5, 0.06, m["leaf"], axis="z")
    B.add_cylinder("pad2", (0, 0, 0.08), 0.3, 0.04, m["leaf_l"], axis="z")
    B.add_sphere("flower", (0.12, 0.1, 0.16), (0.12, 0.12, 0.1), m["lily"])


def build_fish(m):                                                       # mid-jump arc
    B.add_sphere("fbody", (0, 0, 0.45), (0.32, 0.15, 0.2), m["fishc"])
    B.add_box("ftail", (0, 0.3, 0.5), (0.2, 0.03, 0.2), m["fishc"], bevel=0.04, taper=0.3)


def build_seaweed(m):
    for a, x in enumerate((-0.13, 0.0, 0.13)):
        B.add_cylinder(f"sw{a}", (x, 0, 0.42), 0.05, 0.85, m["seaw"], axis="z", r2=0.02)


# ── sand ──
def build_shell(m):
    B.add_sphere("shell", (0, 0, 0.13), (0.2, 0.17, 0.15), m["shell"], zclip=-0.02)
    B.add_cylinder("spire", (0, 0, 0.24), 0.07, 0.12, m["shell"], axis="z", r2=0.0)


def build_driftwood(m):
    B.add_cylinder("dw", (0, 0, 0.13), 0.13, 0.8, m["drift"], axis="x", r2=0.09)


# ── grass meadow ──
def build_wildflower(m):
    cols = ["fl_r", "fl_y", "fl_p"]
    for a, (x, y) in enumerate([(-0.12, 0), (0.12, 0.06), (0.0, -0.1), (0.02, 0.12)]):
        B.add_cylinder(f"stem{a}", (x, y, 0.26), 0.03, 0.52, m["leaf_d"], axis="z")
        B.add_sphere(f"fl{a}", (x, y, 0.52), (0.09, 0.09, 0.08), m[cols[a % 3]])


# ── mountain ──
def build_boulder(m):
    B.add_box("bld1", (0, 0, 0.42), (1.12, 0.86, 0.82), m["stone"], bevel=0.14)
    B.add_box("bld2", (0.42, 0.2, 0.2), (0.5, 0.46, 0.42), m["stone_d"], bevel=0.1)
    B.add_box("bld3", (-0.4, -0.18, 0.26), (0.4, 0.38, 0.5), m["stone"], bevel=0.1)


def build_alpine_scrub(m):
    for a, (x, y, z, r) in enumerate([(0, 0, 0.2, 0.3), (-0.22, 0.1, 0.17, 0.2), (0.22, -0.06, 0.18, 0.2)]):
        B.add_sphere(f"as{a}", (x, y, z), (r, r * 0.8, r * 0.7), m["scrub"])


# ── shy peekers (static + animated; n_frames emerge, shy resets to hidden) ──
def build_squirrel(m, peek=1.0):
    B.add_sphere("sqtuft", (0, 0.1, 0.08), (0.32, 0.22, 0.1), m["leaf_d"], zclip=-0.02)  # cover
    dz = (peek - 1) * 0.36
    B.add_sphere("sqbody", (0, -0.05, 0.2 + dz), (0.15, 0.13, 0.18), m["fur"], zclip=0.05)
    B.add_sphere("sqbelly", (0, -0.14, 0.18 + dz), (0.1, 0.07, 0.13), m["belly"], zclip=0.05)
    B.add_sphere("sqhead", (0, -0.14, 0.4 + dz), (0.13, 0.12, 0.12), m["fur"], zclip=0.05)
    for i, (z, y, r) in enumerate([(0.2, 0.22, 0.12), (0.4, 0.27, 0.14), (0.6, 0.22, 0.14), (0.72, 0.08, 0.11)]):
        B.add_sphere(f"sqt{i}", (0.16, y, z + dz), (r, r * 0.7, r), m["fur_l"], zclip=0.05)
    for sx in (-1, 1):
        B.add_sphere(f"sqear{sx}", (sx * 0.08, -0.14, 0.51 + dz), (0.045, 0.04, 0.06), m["fur"], zclip=0.05)
        B.add_sphere(f"sqeye{sx}", (sx * 0.05, -0.24, 0.42 + dz), (0.028, 0.028, 0.028), m["dark"], zclip=0.05)
    B.add_sphere("sqnose", (0, -0.26, 0.38 + dz), (0.03, 0.03, 0.03), m["dark"], zclip=0.05)


def build_bird_nest(m, peek=1.0):
    B.add_sphere("nest", (0, 0.06, 0.12), (0.36, 0.32, 0.12), m["nest"], zclip=0.0)
    B.add_sphere("nesth", (0, 0.06, 0.16), (0.24, 0.2, 0.08), m["nest_d"])
    dz = (peek - 1) * 0.24
    B.add_sphere("bird", (0, -0.02, 0.26 + dz), (0.15, 0.14, 0.16), m["bird"], zclip=0.12)
    B.add_sphere("bwing", (0.12, 0.06, 0.27 + dz), (0.07, 0.11, 0.11), m["bird_d"], zclip=0.12)
    if peek > 0.35:
        B.add_sphere("bhead", (0, -0.1, 0.43 + dz), (0.1, 0.1, 0.1), m["bird"])
        B.add_box("beak", (0, -0.2, 0.43 + dz), (0.03, 0.09, 0.03), m["beak"], bevel=0.01, taper=0.2)
        for sx in (-1, 1):
            B.add_sphere(f"beye{sx}", (sx * 0.045, -0.16, 0.45 + dz), (0.022, 0.022, 0.022), m["dark"])


def build_frog(m, peek=1.0):
    dz = (peek - 1) * 0.1
    B.add_sphere("frbody", (0, 0, 0.15 + dz), (0.21, 0.23, 0.14), m["frog"], zclip=-0.02)
    for sx in (-1, 1):
        B.add_sphere(f"freye{sx}", (sx * 0.1, -0.12, 0.26 + dz * 1.5), (0.07, 0.07, 0.07), m["frog_l"])
        B.add_sphere(f"frpup{sx}", (sx * 0.1, -0.17, 0.27 + dz * 1.5), (0.028, 0.028, 0.028), m["dark"])
    for sx in (-1, 1):
        B.add_sphere(f"frleg{sx}", (sx * 0.18, 0.12, 0.08), (0.08, 0.1, 0.06), m["frog"])


def build_grass_tuft(m):
    for a, x in enumerate((-0.11, -0.04, 0.03, 0.1, 0.0)):
        B.add_cylinder(f"bl{a}", (x, 0, 0.18), 0.028, 0.36, m["reedg"], axis="z", r2=0.004)


# Shy peekers take peek∈[0,1]: 0 = hidden behind/below its base, 1 = fully emerged.
def build_alien_peek(m, peek=1.0):
    B.add_sphere("amound", (0, 0, 0.1), (0.42, 0.32, 0.13), m["leaf_d"], zclip=-0.02)
    if peek > 0.12:
        z = -0.05 + peek * 0.35
        B.add_sphere("ahead", (0, -0.08, z), (0.13, 0.12, 0.16), m["alien"], zclip=0.05)
        for sx in (-1, 1):
            B.add_sphere(f"aeye{sx}", (sx * 0.06, -0.16, z + 0.04), (0.045, 0.055, 0.045), m["dark"])
        B.add_cylinder("aant", (0, -0.02, z + 0.16), 0.012, 0.12, m["alien"], axis="z")


def build_hole_creature(m, peek=1.0):
    B.add_cylinder("hole", (0, 0, 0.04), 0.3, 0.06, m["dark"], axis="z")
    B.add_cylinder("holerim", (0, 0, 0.05), 0.34, 0.05, m["bark_d"], axis="z", r2=0.34)
    if peek > 0.12:
        z = -0.05 + peek * 0.22
        B.add_sphere("crhead", (0, -0.04, z), (0.15, 0.14, 0.14), m["fur"], zclip=0.05)
        for sx in (-1, 1):
            B.add_sphere(f"creye{sx}", (sx * 0.06, -0.13, z + 0.04), (0.03, 0.03, 0.03), m["dark"])


def build_giant_oak(m):                                                  # emergent canopy giant
    B.add_cylinder("gtrunk", (0, 0, 0.85), 0.22, 1.7, m["bark"], axis="z", r2=0.16)
    for i, (x, y, z, r, k) in enumerate(
        [(0, 0, 2.3, 1.15, "leaf_d"), (-0.7, 0.2, 2.1, 0.7, "leaf_d"), (0.7, -0.15, 2.1, 0.7, "leaf"),
         (0.15, 0.45, 2.85, 0.6, "leaf"), (-0.2, -0.45, 2.45, 0.65, "leaf_d"), (0.45, 0.3, 2.6, 0.55, "leaf")]):
        B.add_sphere(f"goak{i}", (x, y, z), (r, r * 0.9, r), m[k])


def build_tall_pine(m):                                                  # tall emergent conifer
    B.add_cylinder("tptrunk", (0, 0, 0.5), 0.15, 1.0, m["bark"], axis="z")
    for i, (z, r, k) in enumerate([(1.0, 1.0, "pine"), (1.6, 0.78, "pine_l"), (2.2, 0.56, "pine"),
                                   (2.7, 0.36, "pine_l"), (3.1, 0.18, "pine")]):
        B.add_cylinder(f"tpc{i}", (0, 0, z), r, 0.7, m[k], axis="z", r2=0.02, seg=18)


def build_undergrowth(m):                                                # low leafy forest-floor spread
    for i, (x, y, r) in enumerate([(0, 0, 0.22), (-0.2, 0.12, 0.16), (0.22, -0.05, 0.18),
                                   (0.08, 0.2, 0.14), (-0.15, -0.15, 0.15)]):
        B.add_sphere(f"ug{i}", (x, y, 0.12), (r, r * 0.85, r * 0.55), m[["leaf_d", "leaf", "leaf_d"][i % 3]])


# ── rocky / volcanic biome ──
def build_lava_bubble(m):
    B.add_cylinder("lcrust", (0, 0, 0.04), 0.42, 0.06, m["vrock_d"], axis="z")
    B.add_sphere("lbub", (0, 0, 0.08), (0.32, 0.32, 0.24), m["lava"], zclip=0.0)
    B.add_sphere("lbub2", (0.06, -0.06, 0.2), (0.13, 0.13, 0.1), m["lava2"])


def build_lava_spurt(m):
    B.add_cylinder("lcol", (0, 0, 0.32), 0.13, 0.64, m["lava"], axis="z", r2=0.06)
    B.add_sphere("ltop", (0, 0, 0.7), (0.17, 0.17, 0.17), m["lava2"])
    for a in range(4):
        th = a / 4 * 2 * math.pi
        B.add_sphere(f"ldrop{a}", (0.26 * math.cos(th), 0.26 * math.sin(th), 0.55), (0.05, 0.05, 0.05), m["ember"])


def build_vrock(m):                                                      # faceted angular chunk
    B.add_cylinder("vr1", (0, 0, 0.2), 0.42, 0.42, m["vrock"], axis="z", seg=6, r2=0.3)
    B.add_cylinder("vr2", (0.22, 0.12, 0.34), 0.22, 0.34, m["vrock_l"], axis="z", seg=6, r2=0.12)
    B.add_cylinder("vr3", (-0.2, -0.1, 0.16), 0.2, 0.3, m["vrock_d"], axis="z", seg=5, r2=0.14)


def build_vrock_slab(m):                                                 # tilted layered slab
    B.add_cylinder("vs1", (0, 0, 0.14), 0.55, 0.24, m["vrock"], axis="z", seg=6, r2=0.5)
    B.add_cylinder("vs2", (0.05, 0.05, 0.32), 0.4, 0.2, m["vrock_l"], axis="z", seg=6, r2=0.34)
    B.add_cylinder("vs3", (-0.1, -0.05, 0.46), 0.24, 0.16, m["vrock_d"], axis="z", seg=5, r2=0.18)


def build_vrock_shard(m):                                                # angular pointed shard
    B.add_cylinder("vsh1", (0, 0, 0.45), 0.26, 0.9, m["vrock"], axis="z", seg=6, r2=0.04)
    B.add_cylinder("vsh2", (0.22, 0.06, 0.22), 0.18, 0.44, m["vrock_d"], axis="z", seg=5, r2=0.06)
    B.add_cylinder("vsh3", (-0.18, -0.04, 0.18), 0.14, 0.36, m["vrock_l"], axis="z", seg=6, r2=0.05)


def build_basalt_column(m):
    B.add_cylinder("bcol", (0, 0, 0.7), 0.32, 1.4, m["basalt"], axis="z", seg=6)
    B.add_cylinder("bcol2", (0.36, 0.1, 0.5), 0.22, 1.0, m["vrock_d"], axis="z", seg=6)


def build_ore_deposit(m):
    B.add_box("orerk", (0, 0, 0.2), (0.5, 0.42, 0.38), m["vrock"], bevel=0.06)
    for a, (x, y, z) in enumerate([(0, -0.1, 0.42), (0.18, 0.05, 0.38), (-0.15, 0.05, 0.36)]):
        B.add_cylinder(f"crys{a}", (x, y, z), 0.06, 0.24, m["ore"], axis="z", r2=0.0, seg=6)


def build_vboulder(m):
    B.add_box("vb1", (0, 0, 0.42), (1.12, 0.86, 0.82), m["vrock"], bevel=0.12)
    B.add_box("vb2", (0.42, 0.2, 0.22), (0.5, 0.46, 0.44), m["vrock_d"], bevel=0.1)
    B.add_box("vb3", (-0.4, -0.16, 0.26), (0.4, 0.38, 0.5), m["vrock_l"], bevel=0.1)


def build_mining_drill(m):
    B.add_box("mbase", (0, 0, 0.16), (0.74, 0.62, 0.32), m["rmetal_d"], bevel=0.05)
    for sx in (-1, 1):
        for sy in (-1, 1):
            B.add_cylinder(f"mleg{sx}{sy}", (sx * 0.3, sy * 0.26, 0.65), 0.05, 1.05, m["rmetal"], axis="z")
    B.add_box("mtop", (0, 0, 1.18), (0.55, 0.46, 0.12), m["rmetal_d"], bevel=0.04)
    B.add_cylinder("mdrill", (0, 0, 0.5), 0.1, 0.95, m["rmetal"], axis="z", r2=0.02)
    B.add_box("mmotor", (0.22, 0, 0.95), (0.32, 0.3, 0.3), m["rmetal"], bevel=0.06)
    B.add_box("mhaz", (0, -0.32, 0.16), (0.5, 0.04, 0.12), m["lava2"], bevel=0.0)


def build_fumarole(m):
    B.add_cylinder("fring", (0, 0, 0.1), 0.28, 0.2, m["vrock_d"], axis="z", r2=0.16)
    for i, (z, r) in enumerate([(0.4, 0.14), (0.66, 0.18), (0.92, 0.14)]):
        B.add_sphere(f"steam{i}", (0.04 * i, 0, z), (r, r, r), m["steam"])


def build_ash_scrub(m):
    for a in range(5):
        th = a / 5 * 2 * math.pi
        B.add_cylinder(f"ash{a}", (0.1 * math.cos(th), 0.1 * math.sin(th), 0.2), 0.022, 0.4, m["ash"], axis="z", r2=0.0)


def build_rock_dweller(m, peek=1.0):
    B.add_box("rdrk", (0, 0.06, 0.14), (0.42, 0.3, 0.26), m["vrock"], bevel=0.06)
    if peek > 0.12:
        z = -0.05 + peek * 0.22
        B.add_sphere("rdhead", (0, -0.12, z + 0.1), (0.13, 0.12, 0.12), m["crfur"], zclip=0.04)
        for sx in (-1, 1):
            B.add_sphere(f"rdeye{sx}", (sx * 0.06, -0.2, z + 0.14), (0.038, 0.042, 0.038), m["ore"])


def build_salamander(m, peek=1.0):
    B.add_box("salrk", (0, 0.12, 0.1), (0.42, 0.3, 0.18), m["vrock_d"], bevel=0.06)
    if peek > 0.12:
        z = -0.05 + peek * 0.13
        B.add_sphere("salbody", (0, -0.04, z + 0.12), (0.22, 0.16, 0.12), m["sal"], zclip=0.04)
        B.add_sphere("salhead", (0, -0.22, z + 0.12), (0.1, 0.1, 0.09), m["sal"])
        for sx in (-1, 1):
            B.add_sphere(f"saleye{sx}", (sx * 0.06, -0.29, z + 0.16), (0.025, 0.025, 0.025), m["dark"])
        for i in range(3):
            B.add_sphere(f"salsp{i}", (-0.06 + i * 0.08, -0.04, z + 0.2), (0.03, 0.03, 0.02), m["sal_l"])


BUILDERS = {
    "grass_tuft": build_grass_tuft,
    "oak": build_oak, "conifer": build_conifer, "birch": build_birch, "willow": build_willow,
    "dead_tree": build_dead_tree, "bush": build_bush, "fern": build_fern, "mushroom": build_mushroom,
    "rock": build_rock, "boulder": build_boulder, "alpine_scrub": build_alpine_scrub,
    "reed": build_reed, "lilypad": build_lilypad, "fish": build_fish, "seaweed": build_seaweed,
    "shell": build_shell, "driftwood": build_driftwood, "wildflower": build_wildflower,
    "squirrel": build_squirrel, "bird_nest": build_bird_nest, "frog": build_frog,
    "alien_peek": build_alien_peek, "hole_creature": build_hole_creature,
    "giant_oak": build_giant_oak, "tall_pine": build_tall_pine, "undergrowth": build_undergrowth,
    # rocky
    "lava_bubble": build_lava_bubble, "lava_spurt": build_lava_spurt, "vrock": build_vrock,
    "vrock_slab": build_vrock_slab, "vrock_shard": build_vrock_shard,
    "basalt_column": build_basalt_column, "ore_deposit": build_ore_deposit, "vboulder": build_vboulder,
    "mining_drill": build_mining_drill, "fumarole": build_fumarole, "ash_scrub": build_ash_scrub,
    "rock_dweller": build_rock_dweller, "salamander": build_salamander,
}
TOP = {"grass_tuft": 0.6, "oak": 2.4, "conifer": 2.4, "birch": 2.3, "willow": 2.0, "dead_tree": 2.2,
       "bush": 0.9, "fern": 1.0, "mushroom": 0.6, "rock": 0.7, "boulder": 1.0, "alpine_scrub": 0.6,
       "reed": 1.6, "lilypad": 0.4, "fish": 0.8, "seaweed": 1.0, "shell": 0.5, "driftwood": 0.5,
       "wildflower": 0.7, "squirrel": 0.7, "bird_nest": 0.5, "frog": 0.4, "alien_peek": 0.6,
       "hole_creature": 0.4,
       "giant_oak": 3.5, "tall_pine": 3.3, "undergrowth": 0.5,
       "lava_bubble": 0.4, "lava_spurt": 0.9, "vrock": 0.7, "vrock_slab": 0.7, "vrock_shard": 1.0,
       "basalt_column": 1.4, "ore_deposit": 0.7,
       "vboulder": 1.0, "mining_drill": 1.3, "fumarole": 1.1, "ash_scrub": 0.7, "rock_dweller": 0.6,
       "salamander": 0.5}

# Garden placement + animation roster for the bake.
# (name, terrains, density, min_distance, max_per_tile, n_frames, n_variants, tile, anim, shy)
#   anim: "sway" (rotate), "static", "peek" (emerge, shy)
GARDEN_BAKE = [
    ("grass_tuft", ["sand", "grass", "forest"], 0.95, 0.8, 5, 4, 3, 22, "sway", False),
    ("wildflower", ["grass"], 0.50, 0.8, 2, 4, 3, 22, "sway", False),
    ("fern", ["forest"], 0.80, 0.8, 3, 4, 3, 22, "sway", False),
    ("undergrowth", ["forest"], 0.55, 0.8, 2, 4, 3, 22, "sway", False),
    ("bush", ["grass", "forest"], 0.50, 1.5, 2, 4, 3, 26, "sway", False),
    ("mushroom", ["forest"], 0.35, 1.0, 2, 1, 3, 20, "static", False),
    ("seaweed", ["water"], 0.18, 1.5, 2, 4, 3, 22, "sway", False),
    ("reed", ["water"], 0.14, 2.0, 1, 4, 3, 30, "sway", False),
    ("lilypad", ["water"], 0.12, 2.0, 1, 1, 3, 26, "static", False),
    ("fish", ["water"], 0.05, 4.0, 1, 4, 2, 24, "sway", False),
    ("shell", ["sand"], 0.10, 1.5, 1, 1, 3, 18, "static", False),
    ("driftwood", ["sand"], 0.06, 3.0, 1, 1, 2, 26, "static", False),
    ("rock", ["sand", "grass", "mountain"], 0.10, 2.5, 2, 1, 3, 22, "static", False),
    ("alpine_scrub", ["mountain"], 0.15, 1.5, 2, 4, 3, 22, "sway", False),
    ("oak", ["grass", "forest"], 0.40, 2.5, 1, 4, 3, 40, "sway", False),
    ("birch", ["forest"], 0.50, 1.8, 1, 4, 3, 40, "sway", False),
    ("conifer", ["forest", "mountain"], 0.50, 2.0, 1, 4, 3, 40, "sway", False),
    ("willow", ["forest"], 0.20, 2.5, 1, 4, 3, 40, "sway", False),
    ("dead_tree", ["forest"], 0.12, 3.0, 1, 4, 3, 40, "sway", False),
    ("giant_oak", ["forest"], 0.14, 4.0, 1, 4, 3, 56, "sway", False),
    ("tall_pine", ["forest", "mountain"], 0.14, 4.0, 1, 4, 3, 56, "sway", False),
    ("boulder", ["mountain"], 0.06, 5.0, 1, 1, 2, 36, "static", False),
    ("frog", ["water", "grass"], 0.04, 4.0, 1, 4, 2, 18, "peek", True),
    ("squirrel", ["forest"], 0.03, 6.0, 1, 5, 2, 24, "peek", True),
    ("bird_nest", ["forest", "grass"], 0.03, 6.0, 1, 5, 2, 24, "peek", True),
    ("alien_peek", ["grass", "forest", "sand"], 0.02, 8.0, 1, 5, 1, 24, "peek", True),
    ("hole_creature", ["grass", "sand"], 0.02, 8.0, 1, 5, 2, 22, "peek", True),
]

# Rocky / volcanic: lava terrains busy with bubbles; basalt/rock = stone + ore +
# rigs; dust the only (barely) living tiles; cliff bare.
ROCKY_BAKE = [
    ("lava_bubble", ["lava"], 0.55, 0.8, 2, 4, 3, 18, "sway", False),
    ("lava_spurt", ["lava"], 0.18, 2.0, 1, 4, 2, 24, "sway", False),
    ("vrock", ["basalt", "rock", "dust"], 0.24, 1.2, 2, 1, 3, 22, "static", False),
    ("vrock_slab", ["basalt", "rock"], 0.16, 1.5, 1, 1, 3, 24, "static", False),
    ("vrock_shard", ["basalt", "rock", "dust"], 0.14, 2.0, 1, 1, 3, 26, "static", False),
    ("ore_deposit", ["basalt", "rock"], 0.18, 2.5, 1, 1, 3, 22, "static", False),
    ("basalt_column", ["basalt", "cliff"], 0.10, 3.0, 1, 1, 2, 38, "static", False),
    ("vboulder", ["rock", "cliff"], 0.10, 4.0, 1, 1, 2, 36, "static", False),
    ("fumarole", ["basalt", "dust"], 0.14, 3.0, 1, 4, 2, 30, "sway", False),
    ("mining_drill", ["basalt", "rock", "dust"], 0.06, 6.0, 1, 1, 2, 34, "static", False),
    ("ash_scrub", ["dust"], 0.10, 1.5, 2, 4, 2, 22, "sway", False),
    ("rock_dweller", ["basalt", "dust"], 0.03, 6.0, 1, 5, 2, 22, "peek", True),
    ("salamander", ["lava", "basalt"], 0.02, 8.0, 1, 5, 2, 22, "peek", True),
]

ROSTERS = {"garden": GARDEN_BAKE, "rocky": ROCKY_BAKE}


def setup_cam(elev, azim, ortho, target_z):
    B.setup_scene(ortho, RES, freestyle_thick=1.1)
    cam = bpy.context.scene.camera
    cam.constraints.clear()
    E, A = math.radians(elev), math.radians(azim)
    dist = 24
    cam.location = (dist * math.cos(E) * math.sin(A), -dist * math.cos(E) * math.cos(A), dist * math.sin(E))
    tgt = bpy.data.objects.new("tgt", None)
    tgt.location = (0, 0, target_z)
    bpy.context.scene.collection.objects.link(tgt)
    c = cam.constraints.new("TRACK_TO")
    c.target = tgt


def render(fn, elev=50, azim=0, ortho=3.2, suffix=""):
    B.reset()
    BUILDERS[fn](mats())
    setup_cam(elev, azim, ortho, TOP[fn] * 0.5)
    path = os.path.join(OUT, f"_o_{fn}{suffix}.png")
    B.render_to(path)
    return Image.open(path).convert("RGBA")


import numpy as np  # noqa: E402
from PIL import ImageDraw, ImageFilter  # noqa: E402

ORTHO_B = 4.2
# final sprite scale per biome (px per world-unit). Rocky props are short, so
# they need a larger scale to read as objects rather than speckle.
PX_PER_UNIT = {"garden": 12.0, "rocky": 17.0}
NO_SHADOW = {"lava_bubble", "lava_spurt", "fish", "seaweed"}   # sit in lava/water, not on ground
TMP = os.path.join(OUT, "_obj_bake_tmp.png")


def _render_frames(name, anim, n_frames):
    builder = BUILDERS[name]
    imgs = []
    if anim == "peek":
        for i in range(n_frames):
            B.reset(); builder(mats(), peek=i / max(n_frames - 1, 1))
            setup_cam(50, 0, ORTHO_B, TOP[name] * 0.5)
            B.render_to(TMP); imgs.append(Image.open(TMP).convert("RGBA"))
    else:
        B.reset(); builder(mats()); setup_cam(50, 0, ORTHO_B, TOP[name] * 0.5)
        if anim == "static" or n_frames == 1:
            B.render_to(TMP); imgs.append(Image.open(TMP).convert("RGBA"))
        else:                                                   # sway: parent meshes, rotate
            scene = bpy.context.scene
            piv = bpy.data.objects.new("piv", None); scene.collection.objects.link(piv)
            for o in list(scene.objects):
                if o.type == "MESH" and o.parent is None:
                    o.parent = piv
            for i in range(n_frames):
                piv.rotation_euler = (0, math.radians(3.0 * math.sin(2 * math.pi * i / n_frames)), 0)
                B.render_to(TMP); imgs.append(Image.open(TMP).convert("RGBA"))
    return imgs


def _jitter(img, var):
    f = [1.0, 0.9, 1.12][var % 3]
    a = np.array(img).astype(float); a[..., :3] = np.clip(a[..., :3] * f, 0, 255)
    return Image.fromarray(a.astype("uint8"))


def bake(biome="garden"):
    out_dir = os.path.abspath(os.path.join(OUT, "..", "..", "..", "assets", "sprites", "worlds"))
    ce = math.cos(math.radians(50)); ppt = RES / ORTHO_B; D = PX_PER_UNIT.get(biome, 12.0) / ppt
    cells = []   # per object: dict(name, frames[var][frame]=PIL, tw, th, y_off, params)
    for (name, terrains, dens, mind, mx, nf, nv, _tile, anim, shy) in ROSTERS[biome]:
        frames = _render_frames(name, anim, nf)
        # union bbox across frames so the animation is aligned
        x0 = y0 = 10**9; x1 = y1 = 0
        for im in frames:
            a = np.asarray(im); ys, xs = np.where(a[..., 3] > 8)
            x0, y0, x1, y1 = min(x0, xs.min()), min(y0, ys.min()), max(x1, xs.max() + 1), max(y1, ys.max() + 1)
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        base_row = RES / 2 + (TOP[name] * 0.5) * ce * ppt - y0  # base (z=0) screen row in the crop
        tw = max(1, int((x1 - x0) * D)); th = max(1, int((y1 - y0) * D))
        base_in_tile = base_row * D
        y_off = round(base_in_tile - th / 2, 1)
        var_frames = []
        for v in range(nv):
            ff = []
            for im in frames:
                c = im.crop((x0, y0, x1, y1))
                if name in NO_SHADOW:               # in-surface (lava/water) → no ground shadow
                    sh = c
                else:                               # contact shadow at the base
                    sh = Image.new("RGBA", c.size, (0, 0, 0, 0)); dd = ImageDraw.Draw(sh)
                    ew = int((x1 - x0) * 0.5); eh = max(3, int((x1 - x0) * 0.13)); cx = (x1 - x0) // 2
                    br = int(base_row)
                    dd.ellipse([cx - ew // 2, br - eh // 2, cx + ew // 2, br + eh // 2], fill=(12, 26, 12, 120))
                    sh = sh.filter(ImageFilter.GaussianBlur(int(max(1, (x1 - x0) // 30))))
                    sh.alpha_composite(c)
                ff.append(_jitter(sh.resize((tw, th), Image.LANCZOS), v))
            var_frames.append(ff)
        cells.append(dict(name=name, vf=var_frames, tw=tw, th=th, nf=nf, nv=nv, y_off=y_off,
                          terrains=terrains, dens=dens, mind=mind, mx=mx, shy=shy))
    # assemble atlas (per object: nv*tw wide × nf*th tall, stacked vertically)
    atlas_w = max(c["nv"] * c["tw"] for c in cells)
    atlas_h = sum(c["nf"] * c["th"] for c in cells)
    atlas = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
    meta = []; ypx = 0
    for c in cells:
        for fr in range(c["nf"]):
            for v in range(c["nv"]):
                atlas.paste(c["vf"][v][fr], (v * c["tw"], ypx + fr * c["th"]))
        meta.append((c["name"], ypx, c["nf"], c["nv"], c["tw"], c["th"], c["terrains"],
                     c["dens"], c["mind"], c["mx"], c["y_off"], c["shy"]))
        ypx += c["nf"] * c["th"]
    atlas.save(os.path.join(out_dir, f"{biome}_objects.png"))
    _merge_manifest(out_dir, meta, biome)
    print(f"baked {biome}: {len(meta)} objects → {biome}_objects.png ({atlas_w}×{atlas_h})")


def _merge_manifest(out_dir, meta, biome):
    import re
    path = os.path.join(out_dir, "objects_manifest.ron")
    txt = open(path).read()
    lines = [f'        "{biome}": (', f'            atlas: "{biome}_objects.png",', "            objects: ["]
    for (nm, ypx, nf, nv, tw, th, terr, dens, mind, mx, yo, shy) in meta:
        tr = "[" + ", ".join(f'"{t}"' for t in terr) + "]"
        lines += ["                (",
                  f'                    name: "{nm}",', f"                    y_px: {ypx},",
                  f"                    n_frames: {nf},", f"                    n_variants: {nv},",
                  f"                    tile_w: {tw},", f"                    tile_h: {th},",
                  f"                    terrains: {tr},", f"                    density: {dens},",
                  f"                    min_distance: {mind},", f"                    max_per_tile: {mx},",
                  f"                    y_offset: {yo},", f"                    shy: {str(shy).lower()},",
                  "                ),"]
    lines += ["            ],", "        ),"]
    block = "\n".join(lines)
    new = re.sub(rf'        "{biome}": \(.*?\n        \),', block, txt, count=1, flags=re.S)
    open(path, "w").write(new)


if __name__ == "__main__":
    import sys
    if "bake" in sys.argv:
        bake(sys.argv[2] if len(sys.argv) > 2 else "garden")
    else:
        for fn in BUILDERS:
            render(fn, 50, 0, suffix="_34")
        print(f"rendered {len(BUILDERS)} objects at 3/4")
