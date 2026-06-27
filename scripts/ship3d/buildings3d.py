"""
buildings3d.py — 3/4-perspective 3D building sprites (toon + ink).

Per docs/building_design_bible.md: six civic FUNCTIONS (market / outfitter /
shipyard / mechanic / bar / pad) whose silhouette is constant across biomes,
re-skinned by five STYLES (colony / cryo / extraction / station / outpost) that
change material, roof logic and glow palette.

Rendered from an oblique orthographic camera (high-ish 3/4 so it sits on the
top-down terrain), world-fixed light, Freestyle ink — same family as the ships.

Run:  scripts/.blender_venv/bin/python buildings3d.py
Out:  out/_buildings3d.png  (prototype montage)
"""
import math
import os

import bpy  # noqa
import numpy as np
from PIL import Image, ImageDraw

import blender_gen as B

OUT = os.path.join(os.path.dirname(__file__), "out")
RES = 576       # → px_per_tile ≈ 33.9 ≥ tile_px(32), so the game DOWN-scales the
                # sprite (crisp) instead of up-scaling 1.7× (the old soft look)
ELEV, AZIM = 50.0, 0.0       # forward-facing oblique: tilted down ~midway
                             # between front-on (0) and top-down (90), no azimuth
ORTHO = 17.0                 # fixed across the montage so relative sizes show

# footprint W×D in tiles (base/collision box); roofs/awnings may overhang
FOOTPRINTS = {"market": (6, 5), "outfitter": (4, 4), "shipyard": (8, 6),
              "mechanic": (6, 4), "bar": (6, 5), "pad": (3, 3)}


def C(*v):
    return tuple(x / 255.0 for x in v)


# ── styles (palettes from buildings_manifest.ron + biome theme) ─────────────
STYLES = {
    "colony": dict(biome="garden", roof="gable",
                   wall=C(156, 170, 138), wall_d=C(98, 112, 82),
                   beam=C(94, 70, 48), roof_c=C(168, 104, 74),
                   trim=C(206, 212, 196), glow=C(190, 236, 180)),
    "cryo": dict(biome="ice", roof="dome",
                 wall=C(195, 218, 238), wall_d=C(150, 180, 210),
                 roof_c=C(220, 234, 246), trim=C(238, 246, 252), glow=C(145, 220, 255)),
    "extraction": dict(biome="rocky", roof="flat_industrial",
                       wall=C(118, 95, 68), wall_d=C(78, 62, 46),
                       roof_c=C(92, 78, 64), trim=C(150, 130, 100), glow=C(255, 135, 55)),
    "station": dict(biome="city", roof="flat_tech",
                    wall=C(78, 88, 102), wall_d=C(48, 55, 65),
                    roof_c=C(60, 68, 80), trim=C(150, 162, 178), glow=C(55, 195, 235)),
    "outpost": dict(biome="desert", roof="vault",
                    wall=C(192, 172, 138), wall_d=C(150, 130, 98),
                    roof_c=C(178, 158, 124), trim=C(214, 198, 168), glow=C(225, 190, 105)),
}


def textured(mat, kind, base, scale=3.0):
    """Inject a procedural texture into the toon base-colour multiply, so parts
    read as timber / masonry / brushed metal instead of flat fills."""
    nt = mat.node_tree
    mul = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeMixRGB" and n.blend_type == "MULTIPLY"), None)
    if mul is None:
        return mat
    tc = nt.nodes.new("ShaderNodeTexCoord")
    dark = tuple(c * 0.6 for c in base)
    if kind == "wood":                                   # plank grain
        t = nt.nodes.new("ShaderNodeTexWave"); t.wave_type = "BANDS"; t.bands_direction = "Z"
        t.inputs["Scale"].default_value = scale
        t.inputs["Distortion"].default_value = 2.6
        t.inputs["Detail"].default_value = 2.0
        nt.links.new(tc.outputs["Object"], t.inputs["Vector"])
        cr = nt.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (*dark, 1); cr.color_ramp.elements[1].color = (*base, 1)
        nt.links.new(t.outputs["Fac"], cr.inputs[0]); nt.links.new(cr.outputs["Color"], mul.inputs[1])
    elif kind == "brick":                                # cut-stone masonry
        t = nt.nodes.new("ShaderNodeTexBrick")
        t.inputs["Scale"].default_value = scale
        t.inputs["Color1"].default_value = (*base, 1)
        t.inputs["Color2"].default_value = (*tuple(c * 0.86 for c in base), 1)
        t.inputs["Mortar"].default_value = (*dark, 1)
        if "Mortar Size" in t.inputs:
            t.inputs["Mortar Size"].default_value = 0.025
        nt.links.new(tc.outputs["Object"], t.inputs["Vector"]); nt.links.new(t.outputs["Color"], mul.inputs[1])
    elif kind == "shingle":                              # overlapping roof tiles / shingles
        t = nt.nodes.new("ShaderNodeTexBrick")
        try:
            t.offset = 0.5
        except Exception:
            pass
        t.inputs["Scale"].default_value = scale
        t.inputs["Color1"].default_value = (*base, 1)
        t.inputs["Color2"].default_value = (*tuple(c * 0.66 for c in base), 1)   # strong row contrast
        t.inputs["Mortar"].default_value = (*tuple(c * 0.4 for c in base), 1)
        for _k, _v in (("Mortar Size", 0.06), ("Row Height", 0.3), ("Brick Width", 0.6)):
            if _k in t.inputs:
                t.inputs[_k].default_value = _v
        nt.links.new(tc.outputs["Object"], t.inputs["Vector"]); nt.links.new(t.outputs["Color"], mul.inputs[1])
    else:                                                # subtle noise (brushed metal / adobe / ice)
        t = nt.nodes.new("ShaderNodeTexNoise"); t.inputs["Scale"].default_value = scale
        nt.links.new(tc.outputs["Object"], t.inputs["Vector"])
        cr = nt.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (*tuple(c * 0.9 for c in base), 1)
        cr.color_ramp.elements[1].color = (*tuple(min(1, c * 1.1) for c in base), 1)
        nt.links.new(t.outputs["Fac"], cr.inputs[0]); nt.links.new(cr.outputs["Color"], mul.inputs[1])
    return mat


WALL_TEX = {"garden": "wood", "ice": "noise", "rocky": "brick", "city": "noise", "desert": "noise"}
# roof material per biome: shingles for pitched timber roofs, clay-tile shingles
# for the desert vault, brushed-panel noise for the tech/industrial flat roofs.
ROOF_TEX = {"garden": "shingle", "ice": "noise", "rocky": "noise", "city": "noise", "desert": "shingle"}


def mats(s):
    tex = WALL_TEX[s["biome"]]
    return dict(
        # timber/stone/adobe = matte (spec~0); metal = shiny; glow = emissive
        wall=textured(B.toon_material("wall", s["wall"], spec=0.05), tex, s["wall"], scale=2.6),
        wall_d=textured(B.toon_material("wall_d", s["wall_d"], spec=0.05), tex, s["wall_d"], scale=2.6),
        roof=textured(B.toon_material("roof", s["roof_c"], spec=0.12), ROOF_TEX[s["biome"]],
                      s["roof_c"], scale=5.5),
        trim=B.toon_material("trim", s["trim"], spec=0.2),
        beam=textured(B.toon_material("beam", s.get("beam", s["wall_d"]), spec=0.0), "wood",
                      s.get("beam", s["wall_d"]), scale=4.0),
        plant=B.toon_material("plant", s.get("plant", C(96, 152, 74)), spec=0.0),
        stone=textured(B.toon_material("stone", C(140, 136, 128), spec=0.05), "brick", C(140, 136, 128), scale=6.0),
        dark=B.toon_material("dark", C(40, 42, 48), spec=0.0),
        metal=textured(B.toon_material("metal", C(122, 128, 136), spec=1.3, spec_sharp=0.82), "noise",
                       C(122, 128, 136), scale=9.0),
        glow=B.glow_material("glow", s["glow"], strength=5.0),
    )


# ── primitives ──────────────────────────────────────────────────────────────
def gable(cx, cy, z0, w, d, rh, mat):
    v = [(cx - w / 2, cy - d / 2, z0), (cx + w / 2, cy - d / 2, z0),
         (cx + w / 2, cy + d / 2, z0), (cx - w / 2, cy + d / 2, z0),
         (cx, cy - d / 2, z0 + rh), (cx, cy + d / 2, z0 + rh)]
    f = [(0, 3, 5, 4), (1, 4, 5, 2), (0, 4, 1), (3, 2, 5)]
    B._obj_from_pydata("gable", v, f, mat, smooth=False, bevel=0.02)


def windows(cx, cy, w, d, z0, z1, glow, cols=3, rows=2):
    """Glowing window grid on the camera-facing front wall (-Y). Forward-facing
    view is azimuth 0, so side faces are edge-on / invisible — front only."""
    zs = np.linspace(z0, z1, rows + 2)[1:-1]
    xs = np.linspace(cx - w / 2, cx + w / 2, cols + 2)[1:-1]
    for z in zs:
        for x in xs:
            B.add_box("win", (x, cy - d / 2 - 0.03, z), (w / (cols + 2) * 0.9, 0.06, 0.55), glow, bevel=0.0)


def door(cx, cy, d, m, glow):
    B.add_box("door", (cx, cy - d / 2 - 0.04, 0.9), (1.1, 0.12, 1.8), m["dark"], bevel=0.03)
    B.add_box("lintel", (cx, cy - d / 2 - 0.06, 1.85), (1.3, 0.16, 0.18), glow, bevel=0.0)


def roof_for(style, cx, cy, z0, w, d, m):
    rt = style["roof"]
    if rt == "gable":
        gable(cx, cy, z0 - 0.05, w + 0.9, d + 0.9, 1.9, m["roof"])   # deep eaves
        B.add_box("ridge", (cx, cy, z0 + 1.85), (0.2, d + 0.9, 0.12), m["beam"], bevel=0.0)
    elif rt == "dome":                          # steep snow-shedding dome (within the walls) + finial
        B.add_sphere("dome", (cx, cy, z0 - 0.9), (w / 2 - 0.15, d / 2 - 0.15, 2.7), m["roof"], zclip=z0 - 0.05)
        B.add_cylinder("finial", (cx, cy, z0 + 1.6), 0.1, 0.5, m["trim"], axis="z")
    elif rt == "vault":
        B.add_sphere("vault", (cx, cy, z0 - 0.6), (w / 2 + 0.2, d / 2 + 0.2, 1.5), m["roof"], zclip=z0 - 0.05)
        B.add_box("parapet", (cx, cy, z0 + 0.1), (w + 0.5, d + 0.5, 0.25), m["trim"], bevel=0.04)
        B.add_cylinder("tank", (cx + w / 4, cy + d / 4, z0 + 0.9), 0.5, 0.9, m["metal"], axis="z")
    elif rt == "flat_industrial":
        B.add_box("deck", (cx, cy, z0 + 0.12), (w + 0.3, d + 0.3, 0.25), m["roof"], bevel=0.03)
        B.add_cylinder("stack", (cx - w / 4, cy + d / 5, z0 + 1.3), 0.32, 2.4, m["metal"], axis="z")
        B.add_cylinder("tank", (cx + w / 4, cy - d / 6, z0 + 0.9), 0.6, 1.1, m["wall_d"], axis="z")
        B.add_box("vent", (cx + w / 5, cy + d / 4, z0 + 0.5), (0.8, 0.8, 0.5), m["metal"], bevel=0.05)
    elif rt == "flat_tech":
        B.add_box("deck", (cx, cy, z0 + 0.1), (w + 0.2, d + 0.2, 0.2), m["roof"], bevel=0.03)
        B.add_box("para", (cx, cy, z0 + 0.35), (w + 0.2, d + 0.2, 0.12), m["trim"], bevel=0.02)
        # stacked upper module (the city builds up) with its own holo sign
        B.add_box("setback", (cx, cy + 0.4, z0 + 1.2), (w * 0.6, d * 0.55, 2.1), m["wall"], bevel=0.06)
        B.add_box("setsign", (cx, cy + 0.4 - d * 0.28, z0 + 1.8), (w * 0.42, 0.1, 0.7), m["glow"], bevel=0.0)
        B.add_box("setpara", (cx, cy + 0.4, z0 + 2.3), (w * 0.6 + 0.15, d * 0.55 + 0.15, 0.12), m["trim"], bevel=0.02)
        B.add_cylinder("mast", (cx - w / 4, cy + d / 3, z0 + 2.8), 0.08, 2.6, m["metal"], axis="z")
        B.add_box("sign", (cx, cy - d / 2 - 0.05, z0 + 0.6), (w * 0.7, 0.1, 0.7), m["glow"], bevel=0.0)


# ── functional silhouettes (footprint w×d in ~tile units, biome-agnostic) ───
def base_block(cx, cy, w, d, h, m, plinth=True):
    if plinth:
        B.add_box("fl_plinth", (cx, cy, 0.18), (w + 0.4, d + 0.4, 0.36), m["stone"], bevel=0.05)
    B.add_box("body", (cx, cy, 0.36 + h / 2), (w, d, h), m["wall"], bevel=0.06)
    return 0.36 + h


# ── biome shell: base + wall cladding + entry per material economy & hazard ──
def base_and_walls(s, m, w, d, h):
    """Biome base treatment + body. Returns (z0 = floor height, top)."""
    b = s["biome"]
    if b == "rocky":                       # raised on stone piers over bad ground
        for ox in (-w / 2 + 0.5, w / 2 - 0.5):
            for oy in (-d / 2 + 0.5, d / 2 - 0.5):
                B.add_box("fl_pier", (ox, oy, 0.28), (0.55, 0.55, 0.56), m["stone"], bevel=0.04)
        B.add_box("fl_deck", (0, 0, 0.6), (w + 0.2, d + 0.2, 0.28), m["metal"], bevel=0.03)
        z0 = 0.74
    elif b == "ice":                       # bermed insulation skirt vs drift + wind
        B.add_box("fl_berm", (0, 0, 0.3), (w + 1.1, d + 1.1, 0.6), m["wall"], bevel=0.55)
        z0 = 0.5
    elif b == "desert":                    # flared dust-skirt vs storms + burial
        B.add_box("fl_skirt", (0, 0, 0.26), (w + 0.9, d + 0.9, 0.5), m["wall_d"], bevel=0.4)
        z0 = 0.46
    else:                                  # garden / city: cut-stone plinth
        B.add_box("fl_plinth", (0, 0, 0.18), (w + 0.4, d + 0.4, 0.36), m["stone"], bevel=0.05)
        z0 = 0.36
    B.add_box("body", (0, 0, z0 + h / 2), (w, d, h), m["wall"], bevel=0.06)
    return z0, z0 + h


def wall_skin(s, m, w, d, h, z0, top, win=(3, 2), front=True):
    """Biome cladding + openings + a hazard feature on the front wall. Corner
    features always; centre-front openings only when `front` (garage/bay omit)."""
    b = s["biome"]; fz = -d / 2
    if b == "garden":                      # timber frame, generous windows
        for sx in (-w / 2, w / 2):
            B.add_box("post", (sx, fz - 0.02, z0 + h / 2), (0.2, 0.16, h), m["beam"], bevel=0.02)
        B.add_box("plate", (0, fz - 0.03, top - 0.08), (w, 0.1, 0.16), m["beam"], bevel=0.0)
        if front:
            windows(0, 0, w, d, z0 + 0.7, top - 0.6, m["glow"], cols=win[0], rows=win[1])
    elif b == "ice":                       # padded buttress corners, heat-trace seams, deep windows
        for sx in (-w / 2, w / 2):
            B.add_cylinder("butt", (sx, fz + 0.16, z0 + h / 2), 0.46, h, m["wall"], axis="z")
        if front:
            for zz in (z0 + h * 0.34, z0 + h * 0.72):
                B.add_box("trace", (0, fz - 0.05, zz), (w * 0.8, 0.05, 0.07), m["glow"], bevel=0.0)
            for sx in (-w * 0.24, w * 0.24):
                B.add_box("win", (sx, fz - 0.04, z0 + h * 0.62), (0.5, 0.06, 0.55), m["glow"], bevel=0.0)
    elif b == "rocky":                     # riveted metal frame, side scrubber, sealed seam-glow
        for sx in (-w / 2, w / 2):
            B.add_box("ibeam", (sx, fz - 0.02, z0 + h / 2), (0.22, 0.18, h), m["metal"], bevel=0.02)
        B.add_cylinder("scrub", (w / 2 + 0.42, 0.4, z0 + h * 0.55), 0.42, h * 0.95, m["metal"], axis="z")
        B.add_cylinder("scrubcap", (w / 2 + 0.42, 0.4, top + 0.25), 0.47, 0.32, m["dark"], axis="z")
        if front:
            B.add_box("xbeam", (0, fz - 0.04, z0 + h * 0.5), (w, 0.12, 0.16), m["metal"], bevel=0.0)
            for sx in (-w * 0.24, w * 0.24):
                B.add_box("seam", (sx, fz - 0.04, z0 + h * 0.58), (0.42, 0.06, 0.8), m["glow"], bevel=0.0)
    elif b == "city":                      # vertical module seams, curtain-wall glow, holo sign
        B.add_box("signband", (0, fz - 0.07, top - 0.5), (w * 0.85, 0.08, 0.42), m["glow"], bevel=0.0)
        if front:
            for x in np.linspace(-w / 2 + 0.3, w / 2 - 0.3, 4):
                B.add_box("seam", (x, fz - 0.02, z0 + h / 2), (0.08, 0.1, h), m["trim"], bevel=0.0)
            B.add_box("curtain", (0, fz - 0.05, z0 + h * 0.5), (w * 0.78, 0.05, h * 0.62), m["glow"], bevel=0.0)
    elif b == "desert":                    # thick rounded thermal-mass corners + small deep windows
        for sx in (-w / 2, w / 2):
            B.add_cylinder("corner", (sx, fz + 0.1, z0 + h / 2), 0.4, h, m["wall"], axis="z")
        if front:
            for sx in (-w * 0.2, w * 0.2):
                B.add_box("win", (sx, fz - 0.03, z0 + h * 0.62), (0.42, 0.06, 0.42), m["glow"], bevel=0.0)


def entry(s, m, w, d, z0):
    b = s["biome"]; fz = -d / 2
    if b == "ice":                         # storm-porch airlock vestibule
        B.add_box("vest", (0, fz - 0.7, z0 + 0.85), (1.9, 1.4, 1.7), m["wall"], bevel=0.12)
        B.add_box("vroof", (0, fz - 0.7, z0 + 1.78), (2.1, 1.62, 0.18), m["roof"], bevel=0.04)
        B.add_box("vdoor", (0, fz - 1.42, z0 + 0.7), (1.0, 0.1, 1.3), m["dark"], bevel=0.03)
        B.add_box("vglow", (0, fz - 1.45, z0 + 1.42), (1.1, 0.1, 0.13), m["glow"], bevel=0.0)
    else:
        B.add_box("door", (0, fz - 0.04, z0 + 0.55), (1.1, 0.12, 1.45), m["dark"], bevel=0.03)
        B.add_box("lintel", (0, fz - 0.06, z0 + 1.34), (1.3, 0.16, 0.16), m["glow"], bevel=0.0)


def ivy_climb(w, d, z0, top, m):
    """Climbing ivy up the front-wall corners (garden) — clusters of leaves."""
    fy = -d / 2 - 0.06
    for sx in (-w / 2 + 0.35, w / 2 - 0.35):
        inward = -1 if sx > 0 else 1
        n = 10
        for i in range(n):
            t = i / (n - 1)
            zz = z0 + (top - z0 + 0.4) * t
            xx = sx + inward * (0.05 + 0.5 * t) * abs(math.sin(i * 1.2))
            r = 0.17 - 0.06 * t
            B.add_sphere(f"ivy{'L' if sx < 0 else 'R'}{i}", (xx, fy, zz), (r, 0.05, r * 0.85), m["plant"])


def shell(s, m, w, d, h, win=(3, 2), front=True, door=True):
    if s["biome"] == "city":
        h *= 1.3                           # the city builds UP (floor area is scarce)
    z0, top = base_and_walls(s, m, w, d, h)
    wall_skin(s, m, w, d, h, z0, top, win, front)
    roof_for(s, 0, 0, top, w, d, m)
    if door:
        entry(s, m, w, d, z0)
    if front and s["biome"] == "garden":
        ivy_climb(w, d, z0, top, m)
    return z0, top


def build_market(s, m):
    w, d, h = 6, 5, 3.0       # market → medium_house 6×5
    z0, top = shell(s, m, w, d, h, win=(4, 1))
    # attached market porch (function cue): a roof sloping from the eave to a
    # front beam on grounded posts; produce crates pulled against the posts.
    wp, yf, zf = w + 0.2, -d / 2 - 1.5, z0 + 1.9
    v = [(-wp / 2, -d / 2, top - 0.15), (wp / 2, -d / 2, top - 0.15), (wp / 2, yf, zf), (-wp / 2, yf, zf)]
    B._obj_from_pydata("porch", v, [(0, 1, 2, 3)], m["roof"], smooth=False, bevel=0.0)
    B.add_box("pbeam", (0, yf, zf - 0.05), (wp, 0.16, 0.18), m["beam"], bevel=0.0)
    for sx in (-wp / 2 + 0.25, wp / 2 - 0.25):
        B.add_box("ppost", (sx, yf, zf / 2), (0.16, 0.16, zf), m["beam"], bevel=0.02)
    for i, (bx, by) in enumerate([(-wp / 2 + 0.55, yf + 0.2), (-wp / 2 + 1.0, yf + 0.3), (wp / 2 - 0.6, yf + 0.2)]):
        B.add_box("crate", (bx, by, 0.42), (0.8, 0.8, 0.8), m["wall_d"] if i % 2 else m["beam"], bevel=0.04)


def build_outfitter(s, m):
    w, d, h = 4, 4, 3.4        # outfitter → small_house 4×4
    z0, top = shell(s, m, w, d, h, win=(2, 1))
    # weapon rack out front (arms dealer) + rooftop test-mount
    B.add_box("rackbar", (0, -d / 2 - 1.3, 1.2), (3.0, 0.16, 0.16), m["metal"], bevel=0.0)
    for rx in (-1.0, 0.0, 1.0):
        B.add_cylinder("missile", (rx, -d / 2 - 1.3, 0.7), 0.12, 1.0, m["dark"], axis="z", r2=0.02)
    for rx in (-1.2, 1.2):
        B.add_cylinder("rackleg", (rx, -d / 2 - 1.3, 0.6), 0.07, 1.2, m["metal"], axis="z")
    B.add_box("rack", (0, 0, top + 0.5), (2.4, 0.5, 0.5), m["metal"], bevel=0.05)
    B.add_cylinder("barrel", (0, -1.0, top + 0.9), 0.18, 2.2, m["dark"], axis="y")


def build_shipyard(s, m):
    w, d, h = 8, 6, 6.5        # shipyard → large_building 8×6
    B.add_box("fl_plinth", (0, 0, 0.18), (w + 0.5, d + 0.5, 0.36), m["stone"], bevel=0.05)
    # three walls (front open hull-bay)
    B.add_box("backw", (0, d / 2 - 0.2, 0.36 + h / 2), (w, 0.4, h), m["wall"], bevel=0.04)
    for sx in (-w / 2 + 0.2, w / 2 - 0.2):
        B.add_box("sidew", (sx, 0, 0.36 + h / 2), (0.4, d, h), m["wall"], bevel=0.04)
    # lit panels on the rear interior wall (one row → less clutter at top-down)
    for x in np.linspace(-w / 2 + 1.4, w / 2 - 1.4, 4):
        B.add_box("win", (x, d / 2 - 0.42, 2.5), (0.8, 0.06, 1.0), m["glow"], bevel=0.0)
    # overhead gantry: stout gusseted legs to a top beam (reads connected from above)
    beam_z = 0.36 + h + 0.7
    leg_h = beam_z - 0.36
    for sx in (-w / 2 + 0.6, w / 2 - 0.6):
        B.add_box("leg", (sx, -d / 2 + 0.7, 0.36 + leg_h / 2), (0.45, 0.45, leg_h), m["metal"], bevel=0.03)
        B.add_box("gusset", (sx * 0.82, -d / 2 + 0.7, beam_z - 0.55), (0.8, 0.45, 0.7), m["metal"], bevel=0.12)
    B.add_box("beam", (0, -d / 2 + 0.7, beam_z + 0.08), (w - 0.4, 0.5, 0.5), m["metal"], bevel=0.04)
    B.add_box("trolley", (0.6, -d / 2 + 0.7, beam_z - 0.2), (0.6, 0.6, 0.35), m["dark"], bevel=0.04)
    B.add_box("hook", (0.6, -d / 2 + 0.7, beam_z - 1.0), (0.16, 0.16, 1.5), m["dark"], bevel=0.02)
    # a small ship under construction, up on service cradles in the bay
    for sx in (-1.2, 1.2):
        B.add_box("cradle", (sx, 0.3, 0.7), (0.4, 2.6, 1.0), m["metal"], bevel=0.06)
    small_ship(0, 0.2, 1.95, m, scale=1.2)
    roof_for(s, 0, d / 2 - 0.2, 0.36 + h, w, 0.6, m)


def build_mechanic(s, m):
    w, d, h = 6, 4, 3.2        # mechanic → 6×4 garage
    z0, top = shell(s, m, w, d, h, front=False, door=False)
    bw = w * 0.6
    # OPEN roll-up door (rolled to a drum at the lintel) → dark bay behind
    B.add_box("bay", (0, -d / 2 + 0.25, z0 + 1.1), (bw, 0.1, 2.2), m["dark"], bevel=0.0)
    B.add_cylinder("drum", (0, -d / 2 - 0.05, top - 0.35), 0.32, bw + 0.3, m["metal"], axis="x")
    for sx in (-bw / 2 - 0.18, bw / 2 + 0.18):
        B.add_box("jamb", (sx, -d / 2 - 0.05, z0 + 1.2), (0.18, 0.2, 2.5), m["wall_d"], bevel=0.03)
    B.add_box("hazard", (0, -d / 2 - 0.12, z0 + 0.08), (bw, 0.06, 0.18), m["glow"], bevel=0.0)
    # a ship engine pulled out front on a stand, opened up for repair
    repair_engine(-0.8, -d / 2 - 1.9, m)
    B.add_box("plate", (w / 2 - 0.5, -d / 2 - 1.1, 1.1), (1.4, 0.14, 2.0), m["wall_d"], bevel=0.05)
    B.add_box("toolbox", (w / 2 - 1.5, -d / 2 - 1.4, 0.35), (0.9, 0.6, 0.5), m["roof"], bevel=0.05)
    B.add_cylinder("jib_p", (w / 2 - 0.3, d / 2 - 0.3, top + 0.8), 0.14, 2.0, m["metal"], axis="z")
    B.add_box("jib_a", (w / 2 - 1.2, d / 2 - 0.3, top + 1.4), (2.0, 0.2, 0.2), m["metal"], bevel=0.03)


def build_bar(s, m):
    w, d, h = 6, 5, 3.2        # bar → medium_house 6×5
    z0, top = shell(s, m, w, d, h, win=(3, 2))
    # big glowing hanging sign on a bracket arm + roof chimney
    B.add_box("bracket", (-w / 2 + 0.2, -d / 2 - 0.5, top - 0.3), (0.1, 1.0, 0.1), m["beam"], bevel=0.0)
    B.add_box("signpost", (-w / 2 + 0.2, -d / 2 - 1.0, top - 1.1), (0.12, 0.12, 1.0), m["dark"], bevel=0.0)
    B.add_box("signface", (-w / 2 + 0.2, -d / 2 - 1.06, top - 1.6), (1.4, 0.1, 0.85), m["glow"], bevel=0.02)
    B.add_cylinder("chimney", (w / 4, d / 4, top + 0.9), 0.3, 1.5, m["wall_d"], axis="z")
    B.add_cylinder("chimcap", (w / 4, d / 4, top + 1.7), 0.36, 0.2, m["beam"], axis="z")
    if s["biome"] == "garden":
        planters(w, d, m)


def build_pad(s, m):
    # pad → 3×3 tiles (matches PAD_TILES); apron just inside the footprint
    B.add_cylinder("apron", (0, 0, 0.12), 1.5, 0.24, m["wall_d"], axis="z")
    B.add_cylinder("ring", (0, 0, 0.26), 1.3, 0.12, m["trim"], axis="z")
    B.add_cylinder("inner", (0, 0, 0.3), 0.85, 0.06, m["dark"], axis="z")
    B.add_cylinder("glowpad", (0, 0, 0.33), 0.5, 0.05, m["glow"], axis="z")          # centre touchdown light
    for sy in (-1, 1):                                                                # approach chevrons
        B.add_box("chev", (0, sy * 0.95, 0.34), (0.8, 0.14, 0.06), m["glow"], bevel=0.0)
    for a in range(8):
        th = a / 8 * 2 * math.pi
        B.add_box("light", (1.4 * math.cos(th), 1.4 * math.sin(th), 0.34), (0.16, 0.16, 0.16), m["glow"], bevel=0.0)


def planters(w, d, m):
    """Garden beds flanking the entrance — colony agrarian life (bible)."""
    for sx in (-w / 2 + 0.7, w / 2 - 0.7):
        B.add_box("planter", (sx, -d / 2 - 0.35, 0.32), (1.0, 0.5, 0.34), m["beam"], bevel=0.05)
        B.add_box("plants", (sx, -d / 2 - 0.35, 0.62), (0.9, 0.45, 0.34), m["plant"], bevel=0.2)


def frame_detail(w, d, h, base_z, m):
    """Corner posts + a single top plate under the eave. (No mid/sill bands —
    they compressed into 'black stripes' near top-down; texture carries the wall.)"""
    for sx in (-w / 2, w / 2):
        B.add_box("post", (sx, -d / 2 - 0.02, base_z + h / 2), (0.2, 0.16, h), m["beam"], bevel=0.02)
    B.add_box("plate", (0, -d / 2 - 0.03, base_z + h - 0.08), (w, 0.1, 0.16), m["beam"], bevel=0.0)


def small_ship(cx, cy, cz, m, scale=1.0):
    """A little delta-wing hull that reads from above, nose toward -Y (front)."""
    s = scale
    B.add_box("sf", (cx, cy + 0.1 * s, cz), (0.68 * s, 2.6 * s, 0.6 * s), m["wall"], bevel=0.3, taper=0.55)
    B.add_cylinder("snose", (cx, cy - 1.5 * s, cz), 0.3 * s, 1.0 * s, m["wall"], axis="y", r2=0.04 * s, seg=14)
    for sx in (-1, 1):                                   # big swept delta wings (read top-down)
        B.add_box("swing", (cx + sx * 1.15 * s, cy + 0.55 * s, cz),
                  (1.7 * s, 1.9 * s, 0.12 * s), m["wall_d"], bevel=0.05, taper=0.35)
    B.add_sphere("scock", (cx, cy - 0.55 * s, cz + 0.36 * s), (0.26 * s, 0.6 * s, 0.3 * s), m["glow"])
    B.add_box("stail", (cx, cy + 1.45 * s, cz + 0.45 * s), (0.12 * s, 0.6 * s, 0.75 * s), m["wall_d"], bevel=0.04)
    for sx in (-0.34, 0.34):
        B.add_cylinder("snoz", (cx + sx * s, cy + 1.55 * s, cz), 0.16 * s, 0.4 * s, m["dark"], axis="y")


def repair_engine(cx, cy, m):
    """A ship engine pulled onto a service stand, opened up — exposed glowing core."""
    B.add_box("estand", (cx, cy, 0.08), (1.5, 1.1, 0.16), m["dark"], bevel=0.03)
    for ox, oy in ((-0.55, -0.4), (0.55, -0.4), (-0.55, 0.4), (0.55, 0.4)):
        B.add_cylinder("eleg", (cx + ox, cy + oy, 0.5), 0.06, 0.8, m["metal"], axis="z")
    B.add_cylinder("ecore", (cx, cy, 1.2), 0.5, 1.7, m["metal"], axis="y", r2=0.42, seg=18)
    B.add_cylinder("ebell", (cx, cy + 1.0, 1.2), 0.44, 0.5, m["dark"], axis="y")          # exhaust bell
    B.add_box("eintern", (cx, cy - 0.25, 1.2), (0.7, 0.5, 0.7), m["glow"], bevel=0.02)    # exposed internals
    B.add_cylinder("epipe", (cx - 0.55, cy - 0.1, 1.5), 0.07, 1.1, m["dark"], axis="y")
    B.add_box("epanel", (cx + 0.62, cy - 0.3, 1.55), (0.1, 0.6, 0.5), m["wall_d"], bevel=0.03)  # open access hatch


BUILDERS = {"market": build_market, "outfitter": build_outfitter, "shipyard": build_shipyard,
            "mechanic": build_mechanic, "bar": build_bar, "pad": build_pad}


# ── camera / render ─────────────────────────────────────────────────────────
def setup_iso(elev=ELEV, azim=AZIM):
    scene = bpy.context.scene
    cam = scene.camera
    cam.constraints.clear()
    E, A = math.radians(elev), math.radians(azim)
    dist = 40
    cam.location = (dist * math.cos(E) * math.sin(A), -dist * math.cos(E) * math.cos(A), dist * math.sin(E))
    tgt = bpy.data.objects.new("tgt", None)
    tgt.location = (0, 0, 2.2)
    scene.collection.objects.link(tgt)
    c = cam.constraints.new("TRACK_TO")
    c.target = tgt


def render(style_name, fn, elev=ELEV, azim=AZIM, suffix=""):
    B.reset()
    s = STYLES[style_name]
    m = mats(s)
    BUILDERS[fn](s, m)
    B.setup_scene(ORTHO, RES, freestyle_thick=1.5)
    setup_iso(elev, azim)
    path = os.path.join(OUT, f"_b_{style_name}_{fn}{suffix}.png")
    B.render_to(path)
    return Image.open(path).convert("RGBA")


def debug_angles(style="colony", fns=("shipyard", "market", "bar", "outfitter", "mechanic")):
    """COHERENCE CHECK ONLY (not used in game): render each building from several
    angles so floating / detached parts become obvious."""
    angles = [(50, 0, "game"), (32, -34, "L"), (32, 34, "R"), (12, 0, "eye")]
    cell, pad = RES, 6
    cv = Image.new("RGBA", (pad + len(angles) * (cell + pad), 24 + len(fns) * (cell + 22)), (26, 28, 34, 255))
    d = ImageDraw.Draw(cv)
    d.text((pad, 6), f"COHERENCE / multi-angle ({style}) — not in-game", fill=(240, 240, 250))
    for ri, fn in enumerate(fns):
        for ci, (e, a, lab) in enumerate(angles):
            img = render(style, fn, e, a, suffix=f"_dbg")
            x, y = pad + ci * (cell + pad), 24 + ri * (cell + 22)
            cv.alpha_composite(img, (x, y))
            d.text((x + 4, y + cell + 2), f"{fn} {lab} ({e},{a})", fill=(230, 230, 245))
    cv.convert("RGB").save(os.path.join(OUT, "_angles.png"))
    print("saved _angles.png")


def montage(rows, title, fname):
    cell = RES
    pad = 8
    maxc = max(len(r[1]) for r in rows)
    W = pad + maxc * (cell + pad)
    H = 30 + len(rows) * (cell + 24)
    cv = Image.new("RGBA", (W, H), (26, 28, 34, 255))
    d = ImageDraw.Draw(cv)
    d.text((pad, 8), title, fill=(240, 240, 250))
    for ri, (label, items) in enumerate(rows):
        y = 30 + ri * (cell + 24)
        d.text((pad, y - 2), label, fill=(210, 220, 235))
        for ci, (sty, fn, img) in enumerate(items):
            x = pad + ci * (cell + pad)
            cv.alpha_composite(img, (x, y))
            d.text((x + 4, y + cell + 4), f"{fn} · {sty}", fill=(235, 235, 245))
    cv.convert("RGB").save(os.path.join(OUT, fname))
    print("saved", fname)


def z0_for_biome(b):
    return {"rocky": 0.74, "ice": 0.5, "desert": 0.46}.get(b, 0.36)


def bake():
    """Bake committed game sprites, depth-split into _back (behind the player)
    and _front (over the player) layers so a player stands framed in a doorway.
    The split is at the player's door-plane depth via camera clip planes;
    door-having buildings get their door cut open in the _front layer so the
    player shows through. Engine scale = tile_px / px_per_tile."""
    out_dir = os.path.abspath(os.path.join(OUT, "..", "..", "..", "assets", "sprites", "worlds", "buildings3d"))
    os.makedirs(out_dir, exist_ok=True)
    E = math.radians(ELEV); ppt = RES / ORTHO; tgt = 2.2 * math.cos(E); ce = math.cos(E)
    dist = 40
    cam_loc = (0.0, -dist * math.cos(E), dist * math.sin(E))
    vv = (-cam_loc[0], -cam_loc[1], 2.2 - cam_loc[2])
    vl = math.sqrt(sum(c * c for c in vv)); view = tuple(c / vl for c in vv)
    # (half-width, height-above-floor) of the doorway to cut open in _front
    door_funcs = {"market": (0.62, 1.25), "outfitter": (0.62, 1.25),
                  "bar": (0.62, 1.25), "mechanic": (1.95, 2.5)}
    tmp = lambda n: os.path.join(OUT, f"_bk_{n}.png")
    entries = []
    for st in STYLES:
        for fn in BUILDERS:
            B.reset(); s = STYLES[st]; m = mats(s); BUILDERS[fn](s, m)
            w_t, d_t = FOOTPRINTS[fn]
            # explicit walkable floor slab over the footprint (drawn BELOW the
            # player in-game; the structure is drawn separately, sorted normally)
            B.add_box("floor", (0, 0, 0.03), (w_t - 0.12, d_t - 0.12, 0.06), m["wall_d"], bevel=0.0)
            B.setup_scene(ORTHO, RES, freestyle_thick=1.5); setup_iso()
            cam = bpy.context.scene.camera
            meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
            pp = (0.0, -d_t / 2 + 0.5, 0.0)                     # player door-plane point
            depth = sum((pp[i] - cam_loc[i]) * view[i] for i in range(3))
            # FLOOR pass: floor slab only
            for o in meshes:
                o.hide_render = not o.name.startswith(("floor", "fl_"))
            cam.data.clip_start = 0.05; cam.data.clip_end = 1000.0
            B.render_to(tmp("floor")); floor_im = Image.open(tmp("floor")).convert("RGBA")
            # STRUCTURE passes: hide the floor slab, split by the door plane
            for o in meshes:
                o.hide_render = o.name.startswith(("floor", "fl_"))
            cam.data.clip_start = 0.05; cam.data.clip_end = 1000.0
            B.render_to(tmp("full")); full = Image.open(tmp("full")).convert("RGBA")
            cam.data.clip_end = depth
            B.render_to(tmp("front")); front = Image.open(tmp("front")).convert("RGBA")
            cam.data.clip_start = depth; cam.data.clip_end = 1000.0
            B.render_to(tmp("back")); back = Image.open(tmp("back")).convert("RGBA")
            # union bbox (structure ∪ floor — the floor sits lower on screen)
            a = np.asarray(full)[..., 3]; afl = np.asarray(floor_im)[..., 3]
            ys, xs = np.where((a > 8) | (afl > 8))
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            box = (x0, y0, x1, y1)
            full_c = full.crop(box); fr = np.array(front.crop(box)); bk = back.crop(box)
            floor_c = floor_im.crop(box)
            cw, ch = full_c.size
            ax = RES / 2 - x0
            ay = (RES / 2 + (d_t / 2 * math.sin(E) + tgt) * ppt) - y0
            fx, fy = ax / cw - 0.5, 0.5 - ay / ch
            if fn in door_funcs:                               # cut the doorway open in _front
                z0 = z0_for_biome(s["biome"]); hw, zt = door_funcs[fn]
                lx = int(ax - hw * ppt); rx = int(ax + hw * ppt)
                # Cut from the lintel down to the FLOOR (z0), not the ground — the
                # plinth below the floor stays opaque so the doorway never exposes
                # the terrain behind/below the building.
                yt = int(ay - (z0 + zt) * ce * ppt); yb = int(ay) + 2
                fr[max(yt, 0):max(yb, 0), max(lx, 0):max(rx, 0), 3] = 0
            full_c.save(os.path.join(out_dir, f"{st}_{fn}.png"))
            Image.fromarray(fr).save(os.path.join(out_dir, f"{st}_{fn}_front.png"))
            bk.save(os.path.join(out_dir, f"{st}_{fn}_back.png"))
            floor_c.save(os.path.join(out_dir, f"{st}_{fn}_floor.png"))
            entries.append((st, fn, w_t, d_t, round(fx, 4), round(fy, 4)))
    lines = ["// buildings3d_manifest.ron — auto-generated by buildings3d.py bake",
             "// anchor = footprint front-centre on the ground, as Bevy Anchor::Custom fraction",
             "(", f"    px_per_tile: {round(ppt, 4)},", "    sprites: ["]
    for st, fn, w, d, fx, fy in entries:
        lines.append(f'        (style: "{st}", func: "{fn}", w: {w}, d: {d}, anchor: ({fx}, {fy})),')
    lines += ["    ],", ")", ""]
    mpath = os.path.abspath(os.path.join(out_dir, "..", "buildings3d_manifest.ron"))
    open(mpath, "w").write("\n".join(lines))
    print(f"baked {len(entries)} sprites → {out_dir}")
    print(f"manifest → {mpath}")


def main():
    # Full matrix: rows = function (recognisable across biomes), cols = biome.
    rows = [(fn, [(st, fn, render(st, fn)) for st in STYLES]) for fn in BUILDERS]
    montage(rows, f"FULL BIOME SET — rows=function, cols=biome  (elev {ELEV:.0f})", "_buildings3d.png")


if __name__ == "__main__":
    import sys
    if "angles" in sys.argv:
        debug_angles(sys.argv[2] if len(sys.argv) > 2 else "colony")
    elif "bake" in sys.argv:
        bake()
    else:
        main()
