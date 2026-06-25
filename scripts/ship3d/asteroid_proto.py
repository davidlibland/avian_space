"""
asteroid_proto.py — prototype of the tumble + tinted-deposit-mask asteroid.

Iterating on the look: craggy multi-scale displacement, per-vertex rock colour
(noise mottling + crater darkening), smoother shading ramp, angular ore crystals.

Run:  scripts/.blender_venv/bin/python asteroid_proto.py
Out:  out/_asteroid_rock.png, _asteroid_deposit.png, _asteroid_preview.png
"""

import math
import os
import random

import bpy  # noqa: must precede bmesh
import bmesh
import numpy as np
from PIL import Image, ImageDraw

from blender_gen import render_to, reset, setup_scene, toon_material

OUT = os.path.join(os.path.dirname(__file__), "out")
T = 16
TILE = 128
COLS = 4

DARK = (0.10, 0.09, 0.08)     # crevice rock (near-black)
LIGHT = (0.46, 0.40, 0.33)    # ridge rock (warm grey-brown — asteroid albedo)


def C(*rgb):
    return tuple(v / 255.0 for v in rgb)


# ───────────────────────── procedural geometry ─────────────────────────────
def _rand_unit(rng):
    while True:
        x, y, z = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)
        l = math.sqrt(x * x + y * y + z * z)
        if 0.1 < l <= 1.0:
            return (x / l, y / l, z / l)


def gen_bumps(rng):
    """Multi-scale lumps (+) and craters (-) → craggy, not a smooth potato."""
    bumps = []
    for _ in range(9):                    # large lobes / big craters
        bumps.append((_rand_unit(rng), rng.uniform(-0.38, 0.40), rng.uniform(0.30, 0.6)))
    for _ in range(16):                   # medium relief
        bumps.append((_rand_unit(rng), rng.uniform(-0.15, 0.15), rng.uniform(0.12, 0.28)))
    for _ in range(34):                   # fine surface roughness
        bumps.append((_rand_unit(rng), rng.uniform(-0.07, 0.07), rng.uniform(0.05, 0.13)))
    return bumps


def _disp(d, bumps):
    r = 1.0
    for (c, amp, width) in bumps:
        ang = 1.0 - (d[0] * c[0] + d[1] * c[1] + d[2] * c[2])
        r += amp * math.exp(-(ang / width) ** 2)
    return max(0.4, r)


def max_disp(bumps, samples=600):
    """Largest displaced radius over the whole sphere (fibonacci sampling).
    Used to normalize the asteroid so it never crops at any tumble angle."""
    m, ga = 0.0, math.pi * (3 - math.sqrt(5))
    for i in range(samples):
        z = 1 - 2 * (i + 0.5) / samples
        rad = math.sqrt(max(0.0, 1 - z * z))
        th = ga * i
        m = max(m, _disp((rad * math.cos(th), rad * math.sin(th), z), bumps))
    return m


def gen_waves(rng, n=14):
    """Sum-of-sines noise field for surface colour mottling."""
    return [(_rand_unit(rng), rng.uniform(2.0, 7.0), rng.uniform(0, 6.28),
             rng.uniform(0.4, 1.0)) for _ in range(n)]


def _noise(p, waves):
    s = tot = 0.0
    for (d, f, ph, a) in waves:
        s += a * math.sin(f * (p[0] * d[0] + p[1] * d[1] + p[2] * d[2]) + ph)
        tot += a
    return s / tot


def make_asteroid(name, bumps, waves, radius, mat, rng, subdiv=4):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    for v in bm.verts:
        n = v.co.normalized()
        d = _disp((n.x, n.y, n.z), bumps) + rng.uniform(-0.03, 0.03)
        v.co = n * radius * d
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = False                 # faceted = craggy rock
    # per-vertex rock colour: mottle by noise, darken craters (disp<1)
    col = me.color_attributes.new(name="rockcol", type="FLOAT_COLOR", domain="POINT")
    for i, v in enumerate(me.vertices):
        p = v.co
        rr = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z) / radius
        dirn = (p.x / (rr * radius + 1e-9), p.y / (rr * radius + 1e-9), p.z / (rr * radius + 1e-9))
        nz = _noise(dirn, waves)
        t = 0.40 + 0.52 * nz + 1.3 * (rr - 1.0)   # deep crater darkening = relief
        t = max(0.0, min(1.0, t))
        col.data[i].color = (DARK[0] * (1 - t) + LIGHT[0] * t,
                             DARK[1] * (1 - t) + LIGHT[1] * t,
                             DARK[2] * (1 - t) + LIGHT[2] * t, 1.0)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def make_deposits(bumps, radius, mat, rng, clusters=6):
    objs = []
    for i in range(clusters):
        d = _rand_unit(rng)
        rr = _disp(d, bumps) * radius
        up = (0, 0, 1) if abs(d[2]) < 0.9 else (1, 0, 0)
        tx = np.cross(d, up); tx = tx / (np.linalg.norm(tx) + 1e-9)
        ty = np.cross(d, tx)
        for j in range(rng.randint(3, 5)):
            s = rng.uniform(0.10, 0.17) * radius
            off = (rng.uniform(-0.2, 0.2) * radius * tx
                   + rng.uniform(-0.2, 0.2) * radius * ty)
            center = np.array(d) * (rr + s * 0.3) + off
            bm = bmesh.new()
            bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
            for v in bm.verts:                       # angular crystal
                v.co = v.co * s * (1 + rng.uniform(-0.45, 0.45))
            me = bpy.data.meshes.new(f"dep{i}_{j}")
            bm.to_mesh(me); bm.free()
            for p in me.polygons:
                p.use_smooth = False
            ob = bpy.data.objects.new(f"dep{i}_{j}", me)
            ob.location = (center[0], center[1], center[2])
            bpy.context.scene.collection.objects.link(ob)
            ob.data.materials.append(mat)
            objs.append(ob)
    return objs


# ───────────────────────────── materials ───────────────────────────────────
def crack_scale_for(seed):
    """Per-asteroid Voronoi crack scale so they don't all look identical
    (derived from the seed without disturbing the shape RNG)."""
    return random.Random(seed * 1009 + 17).uniform(4.5, 8.5)


def rock_material(name, attr="rockcol", crack_scale=6.0):
    """Shading ramp × per-vertex colour × procedural grain, with a noise bump
    for fine roughness and irregular Voronoi cracks.  All textures use Object
    coords so they tumble with the rock."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    texco = nt.nodes.new("ShaderNodeTexCoord")

    # ── micro surface roughness: noise → bump
    nbump = nt.nodes.new("ShaderNodeTexNoise")
    nbump.inputs["Scale"].default_value = 13.0
    nbump.inputs["Detail"].default_value = 6.0
    nt.links.new(texco.outputs["Object"], nbump.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.30
    nt.links.new(nbump.outputs["Fac"], bump.inputs["Height"])

    # ── Voronoi cracks: TWO scales mixed (min of edge-distances) → irregular
    # cell sizes; subtler crevice bump than before.
    def _vor(scale):
        v = nt.nodes.new("ShaderNodeTexVoronoi")
        v.feature = "DISTANCE_TO_EDGE"
        v.inputs["Scale"].default_value = scale
        v.inputs["Randomness"].default_value = 1.0
        nt.links.new(texco.outputs["Object"], v.inputs["Vector"])
        return v

    vor1 = _vor(crack_scale)
    vor2 = _vor(crack_scale * 2.3)
    cmin = nt.nodes.new("ShaderNodeMath")
    cmin.operation = "MINIMUM"
    nt.links.new(vor1.outputs["Distance"], cmin.inputs[0])
    nt.links.new(vor2.outputs["Distance"], cmin.inputs[1])
    bump2 = nt.nodes.new("ShaderNodeBump")
    bump2.inputs["Strength"].default_value = 0.24      # subtler crevices
    nt.links.new(cmin.outputs[0], bump2.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bump2.inputs["Normal"])  # chain bumps

    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs[0].default_value = (1, 1, 1, 1)
    nt.links.new(bump2.outputs["Normal"], diff.inputs["Normal"])
    s2 = nt.nodes.new("ShaderNodeShaderToRGB")
    nt.links.new(diff.outputs[0], s2.inputs[0])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "LINEAR"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.12, 0.12, 0.12, 1)
    e = cr.elements.new(0.5); e.color = (0.5, 0.5, 0.5, 1)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.88, 0.88, 0.88, 1)
    nt.links.new(s2.outputs[0], ramp.inputs[0])

    attr_n = nt.nodes.new("ShaderNodeAttribute")
    attr_n.attribute_name = attr
    mul = nt.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs[0].default_value = 1.0
    nt.links.new(ramp.outputs[0], mul.inputs[1])
    nt.links.new(attr_n.outputs[0], mul.inputs[2])

    # ── fine albedo grain (speckled light/dark rock)
    ngrain = nt.nodes.new("ShaderNodeTexNoise")
    ngrain.inputs["Scale"].default_value = 8.5
    ngrain.inputs["Detail"].default_value = 4.0
    nt.links.new(texco.outputs["Object"], ngrain.inputs["Vector"])
    gramp = nt.nodes.new("ShaderNodeValToRGB")
    gcr = gramp.color_ramp
    gcr.interpolation = "LINEAR"
    gcr.elements[0].color = (0.68, 0.68, 0.68, 1)
    gcr.elements[1].color = (1.22, 1.22, 1.22, 1)
    nt.links.new(ngrain.outputs["Fac"], gramp.inputs[0])
    grain = nt.nodes.new("ShaderNodeMixRGB")
    grain.blend_type = "MULTIPLY"
    grain.inputs[0].default_value = 1.0
    nt.links.new(mul.outputs[0], grain.inputs[1])
    nt.links.new(gramp.outputs[0], grain.inputs[2])

    # ── dark crack lines along Voronoi cell edges (albedo)
    crackramp = nt.nodes.new("ShaderNodeValToRGB")
    ccr = crackramp.color_ramp
    ccr.interpolation = "LINEAR"
    ccr.elements[0].position = 0.0
    ccr.elements[0].color = (0.5, 0.5, 0.5, 1)     # lighter crack line (subtler)
    ccr.elements[1].position = 0.06
    ccr.elements[1].color = (1, 1, 1, 1)
    nt.links.new(cmin.outputs[0], crackramp.inputs[0])
    crack = nt.nodes.new("ShaderNodeMixRGB")
    crack.blend_type = "MULTIPLY"
    crack.inputs[0].default_value = 1.0
    nt.links.new(grain.outputs[0], crack.inputs[1])
    nt.links.new(crackramp.outputs[0], crack.inputs[2])

    emis = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(crack.outputs[0], emis.inputs[0])
    nt.links.new(emis.outputs[0], out.inputs[0])
    return mat


def holdout_material():
    mat = bpy.data.materials.new("holdout")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    ho = nt.nodes.new("ShaderNodeHoldout")
    nt.links.new(ho.outputs[0], out.inputs[0])
    return mat


# ─────────────────────────────── render ────────────────────────────────────
def render_tumble(empty, prefix):
    axis = np.array([1.0, 0.4, 0.25]); axis /= np.linalg.norm(axis)
    frames = []
    empty.rotation_mode = "AXIS_ANGLE"
    for t in range(T):
        ang = 2 * math.pi * t / T
        empty.rotation_axis_angle = (ang, axis[0], axis[1], axis[2])
        tmp = os.path.join(OUT, f"_at_{prefix}.png")
        render_to(tmp)
        frames.append(Image.open(tmp).convert("RGBA"))
    return frames


def pack(frames):
    atlas = Image.new("RGBA", (COLS * TILE, COLS * TILE), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        if f.size != (TILE, TILE):
            f = f.resize((TILE, TILE), Image.LANCZOS)
        r, c = divmod(i, COLS)
        atlas.paste(f, (c * TILE, r * TILE), f)
    return atlas


def tint(mask, color):
    a = np.array(mask.convert("RGBA"), float)
    for k in range(3):
        a[..., k] *= color[k] / 255.0
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"), "RGBA")


def main():
    reset()
    seed = 7
    rng = random.Random(seed)
    rock_mat = rock_material("rock", crack_scale=crack_scale_for(seed))
    dep_mat = toon_material("dep", C(242, 240, 236), spec=0.4)

    bumps = gen_bumps(rng)
    waves = gen_waves(rng)
    rock = make_asteroid("rock", bumps, waves, 1.0, rock_mat, rng, subdiv=3)
    deposits = make_deposits(bumps, 1.0, dep_mat, rng, clusters=6)

    empty = bpy.data.objects.new("tumble", None)
    bpy.context.scene.collection.objects.link(empty)
    for o in [rock] + deposits:
        o.parent = empty

    setup_scene(2.9, TILE)
    scene = bpy.context.scene
    # silhouette outline only — internal crease lines look like scribbles on a
    # faceted rock.
    ls = scene.view_layers[0].freestyle_settings.linesets[0]
    ls.select_crease = False

    for o in deposits:
        o.hide_render = True
    rock_frames = render_tumble(empty, "rock")

    for o in deposits:
        o.hide_render = False
    rock.data.materials.clear()
    rock.data.materials.append(holdout_material())
    scene.render.use_freestyle = False
    dep_frames = render_tumble(empty, "dep")

    pack(rock_frames).save(os.path.join(OUT, "_asteroid_rock.png"))
    pack(dep_frames).save(os.path.join(OUT, "_asteroid_deposit.png"))

    cell = 120
    strip_n = 8
    W = max(strip_n, 4) * (cell + 8) + 8
    H = (cell + 26) * 2 + 30
    cv = Image.new("RGBA", (W, H), (22, 24, 30, 255))
    d = ImageDraw.Draw(cv)

    def put(img, x, y):
        im = img.convert("RGBA").resize((cell, cell), Image.LANCZOS)
        bg = Image.new("RGBA", (cell, cell), (40, 42, 50, 255))
        bg.alpha_composite(im)
        cv.paste(bg, (x, y))

    d.text((8, 2), "TUMBLE (rock atlas, every-other frame)", fill=(240, 240, 250, 255))
    for k in range(strip_n):
        put(rock_frames[k * 2], 8 + k * (cell + 8), 28)

    y2 = 28 + cell + 30
    GOLD = (235, 190, 70); IRON = (150, 92, 66)
    f0, m0 = rock_frames[0], dep_frames[0]
    rg = f0.copy(); rg.alpha_composite(tint(m0, GOLD))
    ri = f0.copy(); ri.alpha_composite(tint(m0, IRON))
    d.text((8, y2 - 14), "COMPOSITE: rock | mask | +GOLD | +IRON", fill=(240, 240, 250, 255))
    for k, im in enumerate((f0, m0, rg, ri)):
        put(im, 8 + k * (cell + 8), y2)
    cv.save(os.path.join(OUT, "_asteroid_preview.png"))
    print("saved _asteroid_preview.png")


if __name__ == "__main__":
    main()
