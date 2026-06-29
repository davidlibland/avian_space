"""
blender_gen.py — Blender (bpy) prototype of the top-down ship sprites.

Renders one ship per class with toon/cel shading, subsurf-smoothed hulls, a
key/fill/rim light rig, and a Freestyle ink outline, from a straight-down
orthographic camera on a transparent film.  Offline asset bake — committed as
PNGs, never shipped in the game.

Run:  scripts/.blender_venv/bin/python blender_gen.py
Out:  scripts/ship3d/out/bl_<name>.png  (+ _preview)
"""

import math
import os

import bpy

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


# ───────────────────────── scene / material helpers ────────────────────────
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def toon_material(name, rgb, *, bands=((0.0, 0.5), (0.5, 0.82), (0.84, 1.0)),
                  spec=0.0, spec_sharp=0.92, glass=False):
    """Cel material: white Diffuse -> ShaderToRGB -> constant ColorRamp (bands)
    -> multiply by base colour -> Emission.  Optional toon specular blob."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs[0].default_value = (1, 1, 1, 1)
    s2 = nt.nodes.new("ShaderNodeShaderToRGB")
    nt.links.new(diff.outputs[0], s2.inputs[0])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "CONSTANT"
    # first element already exists at 0
    cr.elements[0].position = bands[0][0]
    cr.elements[0].color = (bands[0][1],) * 3 + (1,)
    for pos, val in bands[1:]:
        e = cr.elements.new(pos)
        e.color = (val,) * 3 + (1,)
    nt.links.new(s2.outputs[0], ramp.inputs[0])

    mul = nt.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs[0].default_value = 1.0
    mul.inputs[1].default_value = (*rgb, 1)
    nt.links.new(ramp.outputs[0], mul.inputs[2])

    shaded = mul.outputs[0]

    if spec > 0:
        gl = nt.nodes.new("ShaderNodeBsdfGlossy")
        gl.inputs[0].default_value = (1, 1, 1, 1)
        if "Roughness" in gl.inputs:
            gl.inputs["Roughness"].default_value = 0.25
        gs2 = nt.nodes.new("ShaderNodeShaderToRGB")
        nt.links.new(gl.outputs[0], gs2.inputs[0])
        gramp = nt.nodes.new("ShaderNodeValToRGB")
        gcr = gramp.color_ramp
        gcr.interpolation = "CONSTANT"
        gcr.elements[0].position = 0.0
        gcr.elements[0].color = (0, 0, 0, 1)
        e = gcr.elements.new(spec_sharp)
        e.color = (spec, spec, spec, 1)
        # Blender seeds every new ColorRamp with a white stop at pos 1.0. We never
        # recoloured it, so any glossy reflection >=1.0 returned pure white instead
        # of `spec` — and under the ortho camera a flat forward-facing face catches
        # the specular lobe uniformly (reflection >=1.0 across the whole face), so
        # the entire face went white. Recolour that top stop to the spec value so a
        # saturated highlight is the intended faint (or, for metal, bright) spec.
        gcr.elements[-1].color = (spec, spec, spec, 1)
        nt.links.new(gs2.outputs[0], gramp.inputs[0])
        add = nt.nodes.new("ShaderNodeMixRGB")
        add.blend_type = "ADD"
        add.inputs[0].default_value = 1.0
        nt.links.new(mul.outputs[0], add.inputs[1])
        nt.links.new(gramp.outputs[0], add.inputs[2])
        shaded = add.outputs[0]

    emis = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(shaded, emis.inputs[0])
    nt.links.new(emis.outputs[0], out.inputs[0])

    if glass:
        mat.use_backface_culling = False
    return mat


def glow_material(name, rgb, strength=6.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    emis.inputs[0].default_value = (*rgb, 1)
    emis.inputs[1].default_value = strength
    nt.links.new(emis.outputs[0], out.inputs[0])
    return mat


# ───────────────────────────── geometry helpers ────────────────────────────
def _obj_from_pydata(name, verts, faces, mat, smooth=True, subsurf=0, bevel=0.0,
                     mirror=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    if smooth:
        for p in ob.data.polygons:
            p.use_smooth = True
    if bevel > 0:
        m = ob.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = 2
        m.limit_method = "ANGLE"
        m.angle_limit = math.radians(35)
    if mirror:
        m = ob.modifiers.new("mirror", "MIRROR")
        m.use_axis[0] = True
    if subsurf:
        m = ob.modifiers.new("subsurf", "SUBSURF")
        m.levels = subsurf
        m.render_levels = subsurf
    return ob


def _superellipse(m, n, flatten_bottom):
    pts = []
    for k in range(m):
        t = 2 * math.pi * k / m
        ct, st = math.cos(t), math.sin(t)
        x = math.copysign(abs(ct) ** (2.0 / n), ct)
        z = math.copysign(abs(st) ** (2.0 / n), st)
        if z < 0:
            z *= flatten_bottom
        pts.append((x, z))
    return pts


def loft_hull(name, stations, mat, *, m=12, n=2.6, flatten=0.6, subsurf=2,
              cap=True):
    prof = _superellipse(m, n, flatten)
    verts = []
    for s in stations:
        for (px, pz) in prof:
            verts.append((px * s["w"], s["y"], s.get("cz", 0.0) + pz * s["h"]))
    faces = []
    ns = len(stations)
    for i in range(ns - 1):
        b0, b1 = i * m, (i + 1) * m
        for j in range(m):
            jn = (j + 1) % m
            faces.append((b0 + j, b0 + jn, b1 + jn, b1 + j))
    if cap:
        # fan caps to ring centroids
        def centroid(i):
            xs = [verts[i * m + j][0] for j in range(m)]
            ys = [verts[i * m + j][1] for j in range(m)]
            zs = [verts[i * m + j][2] for j in range(m)]
            return (sum(xs) / m, sum(ys) / m, sum(zs) / m)

        f_c = len(verts); verts.append(centroid(ns - 1))
        b_c = len(verts); verts.append(centroid(0))
        bN = (ns - 1) * m
        for j in range(m):
            jn = (j + 1) % m
            faces.append((f_c, bN + j, bN + jn))
            faces.append((b_c, jn, j))
    return _obj_from_pydata(name, verts, faces, mat, smooth=True, subsurf=subsurf)


def elliptical_wing(name, mat, *, span, root_chord, tip_chord, root_y, thick,
                    cz, sweep=0.2, dihedral=0.05, sections=8, subsurf=1,
                    side=1, mirror=True):
    rings = []
    for i in range(sections):
        x = span * (0.04 + 0.96 * i / (sections - 1)) * side
        frac = abs(x) / span
        chord = max(root_chord * math.sqrt(max(0.0, 1 - frac ** 2)), tip_chord)
        cy = root_y - sweep * abs(x)
        z = cz + dihedral * frac
        th = thick * (1 - 0.55 * frac)
        rings.append([
            (x, cy + chord * 0.5, z),
            (x, cy + chord * 0.04, z + th),
            (x, cy - chord * 0.5, z),
            (x, cy + chord * 0.04, z - th * 0.5),
        ])
    verts = [v for r in rings for v in r]
    faces = []
    for i in range(len(rings) - 1):
        b0, b1 = i * 4, (i + 1) * 4
        for j in range(4):
            jn = (j + 1) % 4
            faces.append((b0 + j, b0 + jn, b1 + jn, b1 + j))
    faces.append((0, 1, 2, 3))
    # tip cap
    t = (len(rings) - 1) * 4
    faces.append((t, t + 3, t + 2, t + 1))
    return _obj_from_pydata(name, verts, faces, mat, smooth=True, subsurf=subsurf,
                            mirror=mirror)


def add_box(name, loc, size, mat, bevel=0.04, taper=1.0, subsurf=0):
    cx, cy, cz = loc
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    tx = hx * taper
    v = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - tx, cy - hy, cz + hz), (cx + tx, cy - hy, cz + hz),
        (cx + tx, cy + hy, cz + hz), (cx - tx, cy + hy, cz + hz),
    ]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    return _obj_from_pydata(name, v, f, mat, smooth=False, bevel=bevel,
                            subsurf=subsurf)


def add_cylinder(name, loc, r, length, mat, *, axis="y", r2=None, seg=20,
                 bevel=0.0):
    if r2 is None:
        r2 = r
    cx, cy, cz = loc
    verts, faces = [], []
    for end, rr in ((-length / 2, r), (length / 2, r2)):
        for k in range(seg):
            a = 2 * math.pi * k / seg
            if axis == "y":
                verts.append((cx + rr * math.cos(a), cy + end, cz + rr * math.sin(a)))
            else:
                verts.append((cx + rr * math.cos(a), cy + rr * math.sin(a), cz + end))
    for j in range(seg):
        jn = (j + 1) % seg
        faces.append((j, jn, seg + jn, seg + j))
    c0 = len(verts)
    verts.append((cx, cy - length / 2, cz) if axis == "y" else (cx, cy, cz - length / 2))
    c1 = len(verts)
    verts.append((cx, cy + length / 2, cz) if axis == "y" else (cx, cy, cz + length / 2))
    for j in range(seg):
        jn = (j + 1) % seg
        faces.append((c0, jn, j))
        faces.append((c1, seg + j, seg + jn))
    return _obj_from_pydata(name, verts, faces, mat, smooth=True, bevel=bevel)


def add_sphere(name, loc, radii, mat, zclip=-1e9):
    cx, cy, cz = loc
    rx, ry, rz = radii
    stacks, slices = 12, 18
    verts, idx = [], {}
    for i in range(stacks + 1):
        v = math.pi * i / stacks
        for j in range(slices):
            u = 2 * math.pi * j / slices
            verts.append((cx + rx * math.sin(v) * math.cos(u),
                          cy + ry * math.sin(v) * math.sin(u),
                          cz + rz * math.cos(v)))
            idx[(i, j)] = len(verts) - 1
    faces = []
    for i in range(stacks):
        for j in range(slices):
            jn = (j + 1) % slices
            a, b, c, d = idx[(i, j)], idx[(i, jn)], idx[(i + 1, jn)], idx[(i + 1, j)]
            if all(verts[k][2] < zclip for k in (a, b, c, d)):
                continue
            faces.append((a, b, c, d))
    return _obj_from_pydata(name, verts, faces, mat, smooth=True)


# ───────────────────────────── camera / lights ─────────────────────────────
def setup_scene(ortho, res, freestyle_thick=1.6):
    scene = bpy.context.scene
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    cam_d.ortho_scale = ortho
    cam = bpy.data.objects.new("cam", cam_d)
    scene.collection.objects.link(cam)
    cam.location = (0, 0, 10)
    cam.rotation_euler = (0, 0, 0)
    scene.camera = cam

    # Light pivot: all suns parent to this empty so the whole rig can be spun
    # about the view axis (Z) to bake per-heading lighting frames.
    pivot = bpy.data.objects.new("light_pivot", None)
    scene.collection.objects.link(pivot)

    def sun(name, energy, rot, color=(1, 1, 1)):
        d = bpy.data.lights.new(name, "SUN")
        d.energy = energy
        d.color = color
        o = bpy.data.objects.new(name, d)
        scene.collection.objects.link(o)
        o.rotation_euler = rot
        o.parent = pivot
        return o

    # key from upper-left-front, warm; fill from lower-right, cool & soft; rim
    sun("key", 4.6, (math.radians(46), math.radians(-24), math.radians(18)),
        (1.0, 0.94, 0.85))
    sun("fill", 1.6, (math.radians(-35), math.radians(30), math.radians(-20)),
        (0.80, 0.88, 1.0))
    sun("rim", 2.2, (math.radians(122), math.radians(0), math.radians(0)),
        (1.0, 1.0, 1.0))

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True
    # Standard view transform: AgX (the default) desaturates the stylised cel
    # colours — Standard keeps faction colours (esp. reds) clean and punchy.
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    # Freestyle ink outline
    scene.render.use_freestyle = True
    vl = scene.view_layers[0]
    vl.use_freestyle = True
    fs = vl.freestyle_settings
    ls = fs.linesets[0] if fs.linesets else fs.linesets.new("ink")
    ls.select_silhouette = True
    ls.select_border = True
    ls.select_crease = True
    if ls.linestyle is None:
        ls.linestyle = bpy.data.linestyles.new("ink")
    lst = ls.linestyle
    lst.color = (0.07, 0.08, 0.11)
    lst.thickness = freestyle_thick
    # gentle AO + soft look
    try:
        scene.eevee.use_gtao = True
    except Exception:
        pass
    return pivot


def render_to(path):
    bpy.context.scene.render.filepath = os.path.abspath(path)
    bpy.ops.render.render(write_still=True)


# ════════════════════════════════ SHIPS ════════════════════════════════════
def build_fighter(variant="back"):
    """variant 'back' = swept-back wings (jet fighter); 'fwd' = forward wings
    carrying engine nacelles (Phantom-style)."""
    hull = toon_material("f_hull", (0.45, 0.55, 0.74), spec=1.0, spec_sharp=0.9)
    hull_d = toon_material("f_hull_d", (0.30, 0.38, 0.55))
    accent = toon_material("f_accent", (0.80, 0.22, 0.22))
    glass = toon_material("f_glass", (0.30, 0.62, 0.85),
                          bands=((0.0, 0.45), (0.5, 0.8), (0.8, 1.0)),
                          spec=1.6, spec_sharp=0.8, glass=True)
    nac = toon_material("f_nac", (0.34, 0.40, 0.52), spec=0.9, spec_sharp=0.88)
    intake = toon_material("f_intake", (0.14, 0.17, 0.22))
    glow = glow_material("f_glow", (0.5, 0.75, 1.0), 7)

    loft_hull("f_fuse", [
        dict(y=1.02, w=0.015, h=0.015, cz=0.0),
        dict(y=0.80, w=0.09, h=0.10, cz=0.01),
        dict(y=0.55, w=0.16, h=0.16, cz=0.02),
        dict(y=0.30, w=0.20, h=0.19, cz=0.03),
        dict(y=0.00, w=0.19, h=0.17, cz=0.02),
        dict(y=-0.38, w=0.15, h=0.14, cz=0.01),
        dict(y=-0.72, w=0.115, h=0.12, cz=0.0),
        dict(y=-0.96, w=0.095, h=0.10, cz=0.0),
    ], hull, m=14, n=2.4, flatten=0.7, subsurf=2)

    add_sphere("f_canopy", (0, 0.34, 0.15), (0.095, 0.23, 0.13), glass, zclip=0.15)

    # nose accent ring (shared)
    loft_hull("f_stripe", [
        dict(y=0.52, w=0.165, h=0.165, cz=0.02),
        dict(y=0.46, w=0.175, h=0.175, cz=0.02),
    ], accent, m=14, n=2.4, flatten=0.7, subsurf=1, cap=False)

    # vertical fin (shared)
    fin_v = [(0, -0.62, 0.04), (0, -0.95, 0.04), (0, -0.85, 0.30), (0, -0.66, 0.20)]
    _obj_from_pydata("f_fin", fin_v, [(0, 1, 2, 3)], hull_d, smooth=False)

    if variant == "back":
        # ── jet-fighter: sharply swept delta wings set well aft, small canard
        # foreplanes up front, twin central afterburners.
        elliptical_wing("f_wing", hull, span=0.70, root_chord=0.62,
                        tip_chord=0.05, root_y=-0.20, thick=0.05, cz=0.02,
                        sweep=0.62, dihedral=0.05)
        # forward canards for a modern delta-canard silhouette
        elliptical_wing("f_canard", hull_d, span=0.26, root_chord=0.16,
                        tip_chord=0.03, root_y=0.46, thick=0.03, cz=0.02,
                        sweep=0.18, sections=5)
        noz = toon_material("f_noz", (0.22, 0.26, 0.34))
        for sx in (-0.06, 0.06):
            add_cylinder("f_nz", (sx, -0.99, 0.0), 0.06, 0.14, noz, r2=0.072)
            add_cylinder("f_gl", (sx, -1.07, 0.0), 0.045, 0.02, glow)

    else:  # "fwd" — forward straight wings holding engine nacelles
        # near-straight wings set forward so the nacelles ride alongside the
        # cockpit; engines exhaust behind the trailing edge.
        elliptical_wing("f_wing", hull, span=0.60, root_chord=0.40,
                        tip_chord=0.12, root_y=0.26, thick=0.05, cz=0.0,
                        sweep=-0.04, dihedral=0.03, sections=6)
        # twin engine nacelles mounted under the wings
        for sx in (-0.40, 0.40):
            add_cylinder("f_nacelle", (sx, 0.10, -0.01), 0.075, 0.78, nac,
                         r2=0.07, seg=20)
            # dark intake ring at the front
            add_cylinder("f_intake", (sx, 0.50, -0.01), 0.066, 0.06, intake,
                         seg=20)
            # afterburner glow at the rear
            add_cylinder("f_burn", (sx, -0.30, -0.01), 0.05, 0.03, glow, seg=16)
        # a single small central tail nozzle keeps the fuselage tail resolved
        noz = toon_material("f_noz", (0.22, 0.26, 0.34))
        add_cylinder("f_nz", (0, -0.99, 0.0), 0.07, 0.12, noz, r2=0.08)
        add_cylinder("f_gl", (0, -1.06, 0.0), 0.05, 0.02, glow)


def build_hauler():
    spine = toon_material("h_spine", (0.56, 0.56, 0.60))
    spine_d = toon_material("h_spine_d", (0.40, 0.40, 0.46))
    c_a = toon_material("h_a", (0.66, 0.58, 0.42))
    c_b = toon_material("h_b", (0.47, 0.51, 0.55))
    c_c = toon_material("h_c", (0.58, 0.47, 0.37))
    glass = toon_material("h_glass", (0.35, 0.65, 0.85), spec=1.4, glass=True)

    add_box("h_spine", (0, 0.0, 0.0), (0.30, 1.9, 0.26), spine, taper=0.85)
    add_box("h_rail", (0, -0.05, 0.18), (0.14, 1.7, 0.10), spine_d, bevel=0.02)
    add_box("h_cockpit", (0, 0.92, 0.10), (0.26, 0.30, 0.22), spine, taper=0.6)
    add_sphere("h_glass", (0, 1.02, 0.20), (0.085, 0.10, 0.07), glass)

    # mismatched industrial cargo: a palette of crate colours, assigned per
    # crate so the stack looks like real salvaged/mixed freight.
    c_olive = toon_material("h_olive", (0.45, 0.48, 0.34))
    c_rust = toon_material("h_rust", (0.55, 0.40, 0.32))
    c_steel = toon_material("h_steel", (0.50, 0.52, 0.56))
    palette = [c_a, c_b, c_c, c_olive, c_rust, c_steel]
    hatch = toon_material("h_hatch", (0.30, 0.30, 0.34))
    k = 0
    for yc in (0.46, 0.0, -0.46):
        for sx in (-0.34, 0.34):
            col = palette[(k * 5) % len(palette)]   # spread colours around
            k += 1
            add_box("h_cont", (sx, yc, 0.05), (0.34, 0.40, 0.34), col, taper=0.96)
            # raised corner posts (framework)
            for px in (-0.15, 0.15):
                add_box("h_post", (sx + px, yc, 0.12), (0.05, 0.42, 0.30), spine_d, bevel=0.01)
            # top hatch + cross strut break up the flat lid
            add_box("h_hatch", (sx, yc, 0.245), (0.16, 0.16, 0.05), hatch, bevel=0.015)
            add_box("h_strut", (sx, yc, 0.235), (0.30, 0.05, 0.045), spine_d, bevel=0.01)

    # spine greebles: a couple of pipes running fore-aft (industrial plumbing)
    pipe = toon_material("h_pipe", (0.34, 0.36, 0.40), spec=0.8)
    for px in (-0.10, 0.10):
        add_cylinder("h_pipe", (px, 0.0, 0.20), 0.028, 1.5, pipe, axis="y", seg=10)

    add_box("h_eng", (0, -1.02, 0.0), (0.5, 0.22, 0.30), spine_d)
    noz = toon_material("h_noz", (0.31, 0.31, 0.36))
    glow = glow_material("h_glow", (1.0, 0.6, 0.25), 6)
    for sx in (-0.26, -0.09, 0.09, 0.26):
        add_cylinder("h_nz", (sx, -1.16, 0.0), 0.075, 0.16, noz, r2=0.09)
        add_cylinder("h_gl", (sx, -1.25, 0.0), 0.05, 0.02, glow)


def build_miner():
    # richer, warmer browns so the cool fill light doesn't grey them out
    hull = toon_material("m_hull", (0.66, 0.46, 0.22))
    hull_d = toon_material("m_hull_d", (0.46, 0.31, 0.16))
    steel = toon_material("m_steel", (0.46, 0.45, 0.48))
    drill = toon_material("m_drill", (0.86, 0.66, 0.28), spec=1.3)
    glass = toon_material("m_glass", (0.35, 0.6, 0.8), spec=1.3, glass=True)

    loft_hull("m_hull", [
        dict(y=0.56, w=0.16, h=0.15, cz=0.02),
        dict(y=0.30, w=0.34, h=0.24, cz=0.03),
        dict(y=0.0, w=0.40, h=0.27, cz=0.03),
        dict(y=-0.35, w=0.36, h=0.25, cz=0.02),
        dict(y=-0.66, w=0.27, h=0.20, cz=0.01),
    ], hull, m=14, n=3.2, flatten=0.6, subsurf=2)

    for yc in (0.15, -0.15, -0.42):
        add_box("m_seam", (0, yc, 0.26), (0.66, 0.05, 0.05), hull_d, bevel=0.01)

    add_sphere("m_glass", (0, 0.42, 0.17), (0.12, 0.16, 0.11), glass, zclip=0.17)

    for sx in (-1, 1):
        add_box("m_arm", (sx * 0.40, 0.55, 0.04), (0.14, 0.7, 0.20), steel, taper=0.8)
        add_cylinder("m_collar", (sx * 0.40, 0.80, 0.04), 0.13, 0.06, hull_d)
        add_cylinder("m_drill", (sx * 0.40, 0.95, 0.04), 0.11, 0.28, drill, r2=0.004)
        add_box("m_pod", (sx * 0.46, -0.18, 0.0), (0.16, 0.5, 0.24), hull_d, taper=0.9)

    noz = toon_material("m_noz", (0.38, 0.38, 0.40))
    glow = glow_material("m_glow", (1.0, 0.6, 0.25), 6)
    for sx in (-0.16, 0.16):
        add_cylinder("m_nz", (sx, -0.72, 0.0), 0.10, 0.18, noz, r2=0.12)
        add_cylinder("m_gl", (sx, -0.82, 0.0), 0.07, 0.02, glow)


SHIPS = [
    ("fighter_back", lambda: build_fighter("back"), 28, 2.55),
    ("fighter_fwd", lambda: build_fighter("fwd"), 28, 2.55),
    ("hauler", build_hauler, 88, 2.95),
    ("miner", build_miner, 44, 2.6),
]


def main():
    import sys
    from PIL import Image
    only = sys.argv[1:] or None
    for name, builder, game_size, ortho in SHIPS:
        if only and name not in only:
            continue
        reset()
        builder()
        # render big once; the 256px frame is both the preview and the source
        # for the game sprite (supersampled down so the pixel-based Freestyle
        # outline stays proportional and anti-aliased at small sizes).
        setup_scene(ortho, 256)
        prev = os.path.join(OUT, f"bl_{name}_preview.png")
        render_to(prev)
        big = Image.open(prev).convert("RGBA")
        big.resize((game_size, game_size), Image.LANCZOS).save(
            os.path.join(OUT, f"bl_{name}.png"))
        print("rendered", name)


if __name__ == "__main__":
    main()
