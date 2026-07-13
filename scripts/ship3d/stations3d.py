"""
stations3d.py — BESPOKE space-station sprites for every orbital "planet".

Station-flavoured entries in the star-system YAMLs currently reuse the
planet generator, so they read as tiny worlds instead of built structures.
Each builder here is a purpose-built silhouette in the fleet_gen mold —
top-down ortho, toon shading, Freestyle ink — flavoured by owner:

  Federation  — clean naval rings: white hull, navy trim, blue lights
  Rebel       — open truss yards: rust plate, gantry arms, ochre worklights
  Helios      — automated science works: white sphere, solar wings, cyan
  Bastion     — armored drydock: gunmetal octagon, orange floodlights
  Eco-hotel   — glass garden torus with gold trim (Halcyon's free port)
  Steampunk   — riveted brass wheel, copper boilers, amber gaslight
  Rock        — modules bolted to a captured asteroid (Drift, the Marches)

Sprites land directly in assets/sprites/planets/<name>.png at the SAME
pixel size as the sprite they replace (in-game scale is size-derived), and
are supersampled 8-10x then LANCZOS-downscaled so the ink survives 40-96px.

Run:  scripts/.blender_venv/bin/python stations3d.py [preview]
      preview -> only a montage in out/, nothing written to assets/.
"""

import math
import os
import sys

import bpy
from PIL import Image

from blender_gen import (add_box, add_cylinder, add_sphere, glow_material,
                         render_to, reset, setup_scene, toon_material)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
PLANETS = os.path.join(HERE, "..", "..", "assets", "sprites", "planets")
RES = 560  # supersample master; downscaled per-station below


def C(*rgb):
    return tuple(v / 255.0 for v in rgb)


# ── shared palette bits ────────────────────────────────────────────────────
def palettes():
    return {
        "fed_hull": toon_material("fed_hull", C(208, 216, 228)),
        "fed_trim": toon_material("fed_trim", C(42, 58, 108)),
        "fed_glow": glow_material("fed_glow", C(90, 160, 255), 2.6),
        "reb_plate": toon_material("reb_plate", C(140, 72, 42)),
        "reb_truss": toon_material("reb_truss", C(70, 66, 62)),
        "reb_glow": glow_material("reb_glow", C(255, 150, 50), 2.6),
        "hel_hull": toon_material("hel_hull", C(228, 232, 238)),
        "hel_panel": toon_material("hel_panel", C(26, 52, 92)),
        "hel_glow": glow_material("hel_glow", C(60, 230, 255), 3.0),
        "bas_armor": toon_material("bas_armor", C(78, 82, 90)),
        "bas_plate": toon_material("bas_plate", C(112, 104, 94)),
        "bas_glow": glow_material("bas_glow", C(255, 130, 30), 2.6),
        "eco_glass": toon_material("eco_glass", C(140, 216, 205), spec=0.5),
        "eco_green": toon_material("eco_green", C(64, 150, 76)),
        "eco_gold": toon_material("eco_gold", C(216, 166, 66), spec=0.6),
        "stm_brass": toon_material("stm_brass", C(184, 132, 56), spec=0.5),
        "stm_copper": toon_material("stm_copper", C(150, 88, 50)),
        "stm_iron": toon_material("stm_iron", C(60, 58, 60)),
        "stm_glow": glow_material("stm_glow", C(255, 190, 80), 2.2),
        "rock": toon_material("rock", C(106, 96, 86)),
        "rock_mod": toon_material("rock_mod", C(152, 156, 164)),
        "rock_glow": glow_material("rock_glow", C(255, 210, 120), 2.4),
    }


def ring(name, r_major, r_minor, mat, *, z=0.0, seg=48):
    """Flat torus in the XY plane (a ring seen face-on from the camera)."""
    bpy.ops.mesh.primitive_torus_add(
        major_radius=r_major, minor_radius=r_minor,
        major_segments=seg, minor_segments=12, location=(0, 0, z))
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


def spokes(prefix, n, r_in, r_out, w, mat, *, z=0.0, phase=0.0, thick=0.16):
    """N radial struts from an inner to an outer radius."""
    for i in range(n):
        a = phase + i * 2 * math.pi / n
        mid = (r_in + r_out) / 2
        o = add_box(f"{prefix}{i}", (mid * math.cos(a), mid * math.sin(a), z),
                    (r_out - r_in, w, thick), mat, bevel=0.02)
        o.rotation_euler = (0, 0, a)


def lights(prefix, n, r, s, mat, *, z=0.3, phase=0.0):
    """Ring of small glow studs (docking / running lights)."""
    for i in range(n):
        a = phase + i * 2 * math.pi / n
        add_sphere(f"{prefix}{i}", (r * math.cos(a), r * math.sin(a), z),
                   (s, s, s * 0.6), mat)


# ── Federation: clean naval rings ──────────────────────────────────────────
def build_fed(kind):
    P = palettes()
    ring("hull", 2.3, 0.42, P["fed_hull"])
    ring("trim", 2.3, 0.16, P["fed_trim"], z=0.34)
    add_sphere("hub", (0, 0, 0), (1.05, 1.05, 0.7), P["fed_hull"])
    add_cylinder("hub_band", (0, 0, 0.28), 0.78, 0.5, P["fed_trim"], axis="z")
    spokes("spoke", 4, 0.9, 2.0, 0.34, P["fed_hull"], phase=math.pi / 4)
    lights("dock", 8, 2.3, 0.13, P["fed_glow"], z=0.45)
    if kind == "naval":  # gun pods + sensor mast
        for sx in (-1, 1):
            add_cylinder("pod", (sx * 2.9, 0, 0.1), 0.3, 1.1, P["fed_trim"],
                         axis="y")
            add_cylinder("gun", (sx * 2.9, 0.85, 0.1), 0.08, 0.7,
                         P["fed_hull"], axis="y")
        add_cylinder("mast", (0, 0, 0.5), 0.07, 1.6, P["fed_hull"], axis="y")
    elif kind == "trade":  # docking arms with berthed containers
        for i, a in enumerate((0.5, 2.1, 3.6, 5.2)):
            x, y = 3.0 * math.cos(a), 3.0 * math.sin(a)
            arm = add_box(f"arm{i}", ((2.3 + 3.0) / 2 * math.cos(a),
                                      (2.3 + 3.0) / 2 * math.sin(a), 0),
                          (0.8, 0.2, 0.14), P["fed_trim"], bevel=0.02)
            arm.rotation_euler = (0, 0, a)
            add_box(f"crate{i}", (x, y, 0.1), (0.55, 0.42, 0.3),
                    P["fed_trim" if i % 2 else "fed_hull"], bevel=0.05)
    elif kind == "refinery":  # tank cluster on one flank
        for i, (dx, dy) in enumerate(((2.9, 0.6), (3.15, -0.35), (2.6, -1.1))):
            add_cylinder(f"tank{i}", (dx, dy, 0), 0.34, 1.0, P["fed_trim"],
                         axis="y")
    # customs: bare small ring — handled by scale at render time


# ── Rebel: open truss yards ────────────────────────────────────────────────
def build_rebel(kind):
    P = palettes()
    # rectangular truss frame
    for (x, y, w, h) in ((0, 2.0, 5.6, 0.3), (0, -2.0, 5.6, 0.3),
                         (-2.65, 0, 0.3, 4.3), (2.65, 0, 0.3, 4.3)):
        add_box(f"frame_{x}_{y}", (x, y, 0), (w, h, 0.34), P["reb_plate"],
                bevel=0.02)
    # cross-bracing
    for i, x in enumerate((-1.4, 0, 1.4)):
        add_box(f"brace{i}", (x, 0, -0.05), (0.2, 4.0, 0.2), P["reb_truss"])
    for i, y in enumerate((-1.0, 1.0)):
        add_box(f"xbrace{i}", (0, y, -0.05), (5.2, 0.18, 0.18),
                P["reb_truss"])
    # habitat cluster, off-centre (nothing symmetric about a rebel yard)
    add_sphere("hab", (-1.9, 1.2, 0.15), (0.8, 0.8, 0.55), P["reb_plate"])
    add_box("hab2", (-0.9, 1.35, 0.1), (0.9, 0.6, 0.45), P["reb_plate"],
            bevel=0.06)
    lights("work", 6, 2.35, 0.12, P["reb_glow"], z=0.35, phase=0.4)
    if kind == "shipyard":  # a hull under construction in the cradle
        add_cylinder("keel", (0.7, -0.5, 0.1), 0.36, 2.6, P["reb_plate"],
                     axis="x", r2=0.16)
        for i, x in enumerate((-0.3, 0.5, 1.3)):
            add_box(f"rib{i}", (x, -0.5, 0.15), (0.12, 1.1, 0.5),
                    P["reb_truss"])
        for sy in (-1, 1):
            add_box("gantry", (0.7, -0.5 + sy * 0.9, 0.45), (2.4, 0.14, 0.12),
                    P["reb_truss"])
    else:  # depot: fuel drums lashed inside the frame
        for i, (dx, dy) in enumerate(
                ((0.4, -0.9), (1.3, -0.6), (0.9, 0.35), (1.9, 0.5))):
            add_cylinder(f"drum{i}", (dx, dy, 0.1), 0.3, 0.8, P["reb_plate"],
                         axis="y")


# ── Helios: automated science works ────────────────────────────────────────
def build_helios(kind):
    P = palettes()
    add_sphere("core", (0, 0, 0), (1.15, 1.15, 0.85), P["hel_hull"])
    ring("collar", 1.35, 0.14, P["hel_panel"], z=0.1)
    # big solar wings
    for sx in (-1, 1):
        add_box("wing", (sx * 3.0, 0, -0.05), (3.2, 1.9, 0.08),
                P["hel_panel"], bevel=0.02)
        for i in range(3):  # panel seams read as glowing conduits
            add_box(f"seam{sx}{i}", (sx * (1.9 + i * 1.05), 0, 0.06),
                    (0.14, 1.85, 0.12), P["hel_glow"])
        add_box("boom", (sx * 1.35, 0, 0), (0.9, 0.22, 0.16), P["hel_hull"])
    add_sphere("eye", (0, 0, 0.75), (0.34, 0.34, 0.25), P["hel_glow"])
    if kind == "refinery":  # smelter ring + slag glow
        ring("smelt", 1.8, 0.22, P["hel_panel"], z=-0.1)
        lights("slag", 6, 1.8, 0.14, P["bas_glow" if False else "hel_glow"],
               z=0.2)
        for i, a in enumerate((0.9, 2.4, 4.1)):
            add_cylinder(f"silo{i}", (1.9 * math.cos(a), 1.9 * math.sin(a),
                                      0.05), 0.3, 0.9, P["hel_hull"],
                         axis="y")
    elif kind == "annex":  # bunkhouse pods strung under a dorsal spine
        add_box("spine", (0, 1.6, 0.1), (4.6, 0.2, 0.16), P["hel_hull"])
        for i, x in enumerate((-1.8, -0.6, 0.6, 1.8)):
            add_sphere(f"pod{i}", (x, 1.15, 0.05), (0.42, 0.5, 0.4),
                       P["hel_hull"])
            add_sphere(f"win{i}", (x, 0.75, 0.2), (0.1, 0.1, 0.08),
                       P["hel_glow"])
    else:  # robotics: manipulator arms mid-task
        for i, a in enumerate((0.7, 2.8, 4.4)):
            x, y = 1.5 * math.cos(a), 1.5 * math.sin(a)
            arm = add_box(f"arm{i}", (x, y, 0.25), (1.5, 0.13, 0.1),
                          P["hel_hull"])
            arm.rotation_euler = (0, 0, a + 0.5)


# ── Bastion: armored drydock ───────────────────────────────────────────────
def build_bastion():
    P = palettes()
    ring("wall", 2.4, 0.5, P["bas_armor"], seg=8)  # octagonal: low seg count
    ring("rim", 2.4, 0.2, P["bas_plate"], z=0.4, seg=8)
    add_cylinder("keep", (0, 0, 0.1), 1.0, 0.7, P["bas_plate"], axis="z",
                 seg=8)
    add_cylinder("keep_top", (0, 0, 0.5), 0.6, 0.4, P["bas_armor"], axis="z",
                 seg=8)
    spokes("arm", 3, 0.9, 2.1, 0.42, P["bas_plate"], phase=0.5, thick=0.3)
    # cradled hull being refitted, bright against the gunmetal
    add_cylinder("refit", (0.2, 1.3, 0.45), 0.3, 2.0, P["fed_hull"],
                 axis="x", r2=0.12)
    lights("flood", 8, 2.4, 0.13, P["bas_glow"], z=0.55, phase=0.39)
    for sx in (-1, 1):  # heavy gun bastions
        add_cylinder("turret", (sx * 2.75, -1.2, 0.2), 0.32, 0.5,
                     P["bas_armor"], axis="z")
        add_cylinder("barrel", (sx * 2.75, -1.9, 0.25), 0.07, 0.9,
                     P["bas_plate"], axis="y")


# ── Halcyon: eco-station hotel ─────────────────────────────────────────────
def build_eco():
    P = palettes()
    ring("glass", 2.5, 0.5, P["eco_glass"])
    ring("garden", 2.5, 0.3, P["eco_green"], z=0.25)
    ring("gold", 2.5, 0.09, P["eco_gold"], z=0.52)
    ring("inner_gold", 1.0, 0.08, P["eco_gold"], z=0.2)
    add_sphere("atrium", (0, 0, 0), (0.9, 0.9, 0.75), P["eco_glass"])
    add_sphere("canopy", (0, 0, 0.45), (0.55, 0.55, 0.4), P["eco_green"])
    spokes("prom", 6, 0.85, 2.15, 0.18, P["eco_gold"], phase=0.3, thick=0.1)
    lights("lamp", 12, 2.5, 0.09, P["stm_glow"], z=0.6)
    # yacht pier
    add_box("pier", (0, -3.2, 0), (0.3, 1.2, 0.12), P["eco_gold"])
    add_sphere("pier_end", (0, -3.9, 0.05), (0.28, 0.28, 0.2), P["eco_glass"])


# ── Lowmark: steampunk wheel ───────────────────────────────────────────────
def build_steampunk():
    P = palettes()
    ring("wheel", 2.3, 0.45, P["stm_brass"])
    ring("tyre", 2.3, 0.18, P["stm_iron"], z=0.36)
    lights("rivet", 16, 2.3, 0.07, P["stm_copper"], z=0.5)
    add_cylinder("boiler", (0, 0, 0), 0.85, 1.4, P["stm_copper"], axis="y")
    add_cylinder("boiler_band", (0, 0.3, 0), 0.88, 0.16, P["stm_iron"],
                 axis="y")
    add_cylinder("boiler_band2", (0, -0.3, 0), 0.88, 0.16, P["stm_iron"],
                 axis="y")
    spokes("spoke", 6, 0.8, 2.0, 0.16, P["stm_iron"], phase=0.2, thick=0.12)
    lights("porthole", 8, 1.55, 0.11, P["stm_glow"], z=0.25, phase=0.26)
    # vent stacks blowing off steam (little grey puffs)
    steam = toon_material("steam", C(210, 210, 214))
    for i, a in enumerate((0.8, 2.9)):
        x, y = 2.75 * math.cos(a), 2.75 * math.sin(a)
        add_cylinder(f"stack{i}", (x, y, 0.1), 0.14, 0.7, P["stm_copper"],
                     axis="z")
        add_sphere(f"puff{i}", (x * 1.15, y * 1.15, 0.4), (0.3, 0.3, 0.22),
                   steam)


# ── Rock stations: bolted to a captured asteroid ───────────────────────────
def build_rock(kind):
    P = palettes()
    # The rock is a backdrop, NOT the subject: keep it small and let the
    # bolted structure dominate the silhouette or it reads as a planet.
    add_sphere("rock", (-0.9, -0.5, -0.2), (1.9, 1.5, 1.0), P["rock"])
    add_sphere("lump", (-2.2, 0.6, 0.0), (0.8, 0.65, 0.5), P["rock"])
    # spine truss driven into the rock, modules strung along it
    add_box("spine", (1.0, 0.3, 0.5), (3.8, 0.28, 0.22), P["rock_mod"],
            bevel=0.03)
    add_box("mod_a", (0.4, 0.75, 0.6), (1.5, 0.9, 0.5), P["rock_mod"],
            bevel=0.08)
    add_box("mod_b", (1.9, -0.15, 0.6), (1.1, 0.75, 0.45), P["rock_mod"],
            bevel=0.08)
    add_cylinder("tank", (0.6, -0.55, 0.55), 0.3, 1.2, P["stm_iron"],
                 axis="x")
    lights("win", 5, 0.0, 0.11, P["rock_glow"], z=0.9)
    for i, x in enumerate((0.0, 0.8, 1.6)):
        add_sphere(f"winrow{i}", (x, 0.78, 0.9), (0.1, 0.1, 0.07),
                   P["rock_glow"])
    add_cylinder("mast", (2.7, 0.9, 0.7), 0.05, 1.4, P["rock_mod"], axis="z")
    add_sphere("dish", (2.7, 0.9, 1.45), (0.34, 0.34, 0.12), P["rock_mod"])
    if kind == "freeport":  # glowing hangar maw carved into the rock face
        add_cylinder("maw_rim", (-0.9, -0.5, 0.55), 0.75, 0.5, P["stm_iron"],
                     axis="z")
        add_cylinder("maw", (-0.9, -0.5, 0.62), 0.55, 0.5, P["rock_glow"],
                     axis="z")
        lights("berth", 6, 1.05, 0.09, P["rock_glow"], z=0.85, phase=0.5)
    elif kind == "mine":  # drill gantry + ore conveyor off the rock
        add_cylinder("drill", (-0.9, -0.5, 0.6), 0.2, 1.3, P["rock_mod"],
                     axis="z", r2=0.05)
        belt = add_box("belt", (-2.0, -1.4, 0.4), (2.2, 0.3, 0.12),
                       P["stm_iron"])
        belt.rotation_euler = (0, 0, 0.5)
        add_box("hopper", (-3.0, -1.9, 0.45), (0.7, 0.6, 0.4), P["rock_mod"],
                bevel=0.06)
    else:  # relay: oversized comm dishes are the silhouette
        add_sphere("bigdish", (-0.9, 1.3, 0.6), (1.0, 1.0, 0.3),
                   P["rock_mod"])
        add_sphere("dishcore", (-0.9, 1.3, 0.85), (0.2, 0.2, 0.2),
                   P["rock_glow"])


# ── catalog: name → (builder, ortho zoom, out px) ──────────────────────────
STATIONS = [
    ("kepler_22_station", lambda: build_fed("naval"), 8.6, 48),
    ("eridani_station", lambda: build_fed("trade"), 8.8, 88),
    ("barnard_station", lambda: build_fed("refinery"), 8.8, 44),
    ("centauri_post", lambda: build_fed("customs"), 7.4, 48),
    ("rigel_station", lambda: build_rebel("depot"), 8.0, 48),
    ("deneb_station", lambda: build_rebel("shipyard"), 8.2, 56),
    ("helios_orbital", lambda: build_helios("robotics"), 9.6, 52),
    ("foundry_station", lambda: build_helios("refinery"), 9.6, 62),
    ("drift_annex", lambda: build_helios("annex"), 9.6, 62),
    ("bastion_yard", build_bastion, 8.6, 52),
    ("halcyon_port", build_eco, 9.4, 96),
    ("lowmark_port", build_steampunk, 8.2, 66),
    ("drift_station", lambda: build_rock("relay"), 7.6, 40),
    ("marches_freeport", lambda: build_rock("freeport"), 7.6, 52),
    ("verge_station", lambda: build_rock("mine"), 7.6, 58),
]


def render_station(name, builder, ortho, px, out_dir):
    reset()
    setup_scene(ortho, RES, freestyle_thick=2.2)
    builder()
    tmp = os.path.join(OUT, f"_station_{name}.png")
    render_to(tmp)
    img = Image.open(tmp).convert("RGBA")
    img.resize((px, px), Image.LANCZOS).save(os.path.join(out_dir,
                                                          f"{name}.png"))
    return img


def main():
    preview = "preview" in sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    out_dir = OUT if preview else PLANETS
    masters = []
    for name, builder, ortho, px in STATIONS:
        print(f"— {name}")
        masters.append((name, render_station(name, builder, ortho, px,
                                             out_dir)))
    # montage of the supersampled masters for review
    cell, pad, cols = 190, 8, 5
    rows = (len(masters) + cols - 1) // cols
    cv = Image.new("RGBA", (pad + cols * (cell + pad),
                            pad + rows * (cell + 22 + pad)), (16, 18, 26, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(cv)
    for i, (name, img) in enumerate(masters):
        x = pad + (i % cols) * (cell + pad)
        y = pad + (i // cols) * (cell + 22 + pad)
        cv.alpha_composite(img.resize((cell, cell), Image.LANCZOS), (x, y))
        d.text((x + 4, y + cell + 4), name, fill=(220, 224, 232, 255))
    cv.save(os.path.join(OUT, "stations_montage.png"))
    print(f"montage -> {os.path.join(OUT, 'stations_montage.png')}")


if __name__ == "__main__":
    main()
