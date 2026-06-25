"""3D toon roaming-fauna walk-sheets for the planet surface.

Produces RPG-Maker-style sheets: 3 cols (still, w1, w2) x 4 rows
(down, left, right, up), matching the civilian/walker layout so the game can
reuse CharacterAnim/Facing/sprite_index unchanged. Each animal is modelled
facing -Y (toward the 3/4 camera = "down"); the other facings are the same mesh
spun about Z. Walk frames shuffle diagonal leg pairs + bob the body.

    scripts/.blender_venv/bin/python fauna3d.py          # render all -> assets/sprites/fauna/
    scripts/.blender_venv/bin/python fauna3d.py deer     # one species preview -> out/

Writes assets/sprites/fauna/<name>.png + fauna_manifest.ron.
"""
import os, sys, math
import bpy
import blender_gen as B

OUT = os.path.join(os.path.dirname(__file__), "out")
FAUNA = os.path.abspath(os.path.join(OUT, "..", "..", "..", "assets", "sprites", "fauna"))
RES = 160


def C(r, g, b, **kw):
    return B.toon_material(f"m{r}{g}{b}{kw}", (r / 255, g / 255, b / 255), **kw)


def mats():
    return dict(
        dark=B.toon_material("fdark", (0.15, 0.15, 0.17)),
        eye=B.toon_material("feye", (0.1, 0.1, 0.12)),
        # garden
        deer=C(170, 120, 78), deer_d=C(126, 86, 54), deer_l=C(206, 162, 116),
        cream=C(232, 214, 188), antler=C(150, 130, 96),
        rab=C(178, 168, 156), rab_d=C(132, 122, 110), rabbel=C(232, 224, 214),
        fox=C(196, 110, 56), fox_d=C(150, 78, 38), foxw=C(236, 224, 210),
        # ice / rocky / desert / station signatures
        snowf=C(238, 240, 246), snowf_d=C(198, 208, 220),
        icem=C(150, 200, 224), icem_d=C(104, 158, 192),
        rockm=C(96, 88, 82), rockm_d=C(64, 58, 54), rockm_l=C(128, 118, 110),
        ember=B.glow_material("fember", (1.0, 0.5, 0.2), strength=2.0),
        liz=C(150, 140, 90), liz_d=C(108, 96, 62), liz_l=C(190, 178, 120),
        ratf=C(120, 110, 104), ratf_d=C(84, 76, 72), ratbel=C(176, 166, 158),
        dmetal=C(150, 156, 166, spec=0.8, spec_sharp=0.85), dmetal_d=C(96, 100, 112),
        dlens=B.glow_material("fdlens", (0.4, 0.85, 0.95), strength=2.2),
    )


# ── leg helper: 4 legs that shuffle by walk frame ─────────────────────────
def _legs(m, mat, hx, hy, z0, ln, r, frame):
    stride = {0: 0.0, 1: 0.11, 2: -0.11}[frame]
    for (sx, sy) in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        # diagonal pairs step in opposite phase (front-left + back-right together)
        ph = stride if (sx * sy < 0) else -stride
        B.add_cylinder(f"leg{sx}{sy}", (sx * hx, sy * hy + ph, z0), r, ln, m[mat], axis="z", r2=r * 0.8)


def _bob(frame):
    return 0.02 if frame != 0 else 0.0


# ── garden roamers ────────────────────────────────────────────────────────
def build_deer(m, frame):
    b = _bob(frame)
    B.add_sphere("body", (0, 0.12, 0.66 + b), (0.2, 0.4, 0.2), m["deer"])           # leaner, longer
    B.add_sphere("haunch", (0, 0.36, 0.68 + b), (0.22, 0.22, 0.24), m["deer"])
    B.add_sphere("chest", (0, -0.16, 0.62 + b), (0.19, 0.2, 0.2), m["deer"])
    B.add_sphere("belly", (0, 0.08, 0.56 + b), (0.16, 0.34, 0.12), m["cream"])
    B.add_cylinder("neck", (0, -0.3, 0.92 + b), 0.08, 0.5, m["deer"], axis="z", r2=0.06)  # tall upright neck
    B.add_sphere("head", (0, -0.4, 1.2 + b), (0.11, 0.15, 0.12), m["deer"])
    B.add_sphere("snout", (0, -0.52, 1.14 + b), (0.06, 0.1, 0.06), m["deer_d"])
    for sx in (-1, 1):
        B.add_box(f"ear{sx}", (sx * 0.08, -0.34, 1.32 + b), (0.04, 0.03, 0.09), m["deer_l"])
        B.add_cylinder(f"ant{sx}", (sx * 0.05, -0.36, 1.4 + b), 0.018, 0.2, m["antler"], axis="z", r2=0.0)
        B.add_cylinder(f"antb{sx}", (sx * 0.11, -0.36, 1.46 + b), 0.014, 0.1, m["antler"], axis="z", r2=0.0)
        B.add_sphere(f"eye{sx}", (sx * 0.06, -0.5, 1.22 + b), (0.022, 0.022, 0.022), m["eye"])
    B.add_sphere("tail", (0, 0.52, 0.78 + b), (0.06, 0.06, 0.08), m["cream"])
    _legs(m, "deer_d", 0.13, 0.3, 0.24, 0.56, 0.042, frame)                          # long thin legs


def build_rabbit(m, frame):
    b = _bob(frame)
    B.add_sphere("body", (0, 0.06, 0.28 + b), (0.2, 0.28, 0.2), m["rab"])
    B.add_sphere("belly", (0, 0, 0.2 + b), (0.16, 0.2, 0.1), m["rabbel"])
    B.add_sphere("head", (0, -0.22, 0.34 + b), (0.14, 0.14, 0.14), m["rab"])
    B.add_sphere("snout", (0, -0.34, 0.3 + b), (0.07, 0.07, 0.06), m["rabbel"])
    for sx in (-1, 1):
        B.add_box(f"ear{sx}", (sx * 0.06, -0.16, 0.56 + b), (0.05, 0.04, 0.18), m["rab_d"])
        B.add_sphere(f"eye{sx}", (sx * 0.08, -0.3, 0.36 + b), (0.025, 0.025, 0.025), m["eye"])
    B.add_sphere("tail", (0, 0.3, 0.3 + b), (0.08, 0.08, 0.08), m["rabbel"])
    _legs(m, "rab_d", 0.12, 0.16, 0.12, 0.22, 0.05, frame)


def build_fox(m, frame):
    b = _bob(frame)
    B.add_sphere("body", (0, 0.08, 0.42 + b), (0.2, 0.36, 0.2), m["fox"])
    B.add_sphere("belly", (0, 0.02, 0.32 + b), (0.16, 0.3, 0.1), m["foxw"])
    B.add_sphere("head", (0, -0.34, 0.5 + b), (0.14, 0.15, 0.13), m["fox"])
    B.add_box("snout", (0, -0.5, 0.46 + b), (0.07, 0.12, 0.07), m["foxw"])
    for sx in (-1, 1):
        B.add_box(f"ear{sx}", (sx * 0.09, -0.3, 0.66 + b), (0.06, 0.03, 0.1), m["fox_d"])
        B.add_sphere(f"eye{sx}", (sx * 0.07, -0.42, 0.52 + b), (0.024, 0.024, 0.024), m["eye"])
    B.add_sphere("tail", (0, 0.42, 0.42 + b), (0.13, 0.16, 0.13), m["foxw"])
    _legs(m, "fox_d", 0.12, 0.22, 0.18, 0.34, 0.045, frame)


# species → (builder, ortho, target_z, tile, manifest fields)
SPECIES = [
    ("deer", build_deer, 2.2, 0.55, 34, dict(biome="garden", terrains=["grass", "forest"],
        speed=34.0, flee_speed=78.0, group=3, flier=False)),
    ("rabbit", build_rabbit, 1.5, 0.3, 28, dict(biome="garden", terrains=["grass"],
        speed=46.0, flee_speed=96.0, group=1, flier=False)),
    ("fox", build_fox, 1.8, 0.4, 30, dict(biome="garden", terrains=["grass", "forest"],
        speed=42.0, flee_speed=84.0, group=1, flier=False)),
]
BUILDERS = {n: b for (n, b, *_r) in SPECIES}

# row order = Facing enum: Down, Left, Right, Up. Animal models face -Y (= Down).
FACINGS = [("down", 0.0), ("left", 270.0), ("right", 90.0), ("up", 180.0)]


def setup_cam(ortho, target_z):
    B.setup_scene(ortho, RES, freestyle_thick=1.0)
    cam = bpy.context.scene.camera
    cam.constraints.clear()
    E, A = math.radians(50), 0.0
    dist = 24
    cam.location = (0.0, -dist * math.cos(E), dist * math.sin(E))
    tgt = bpy.data.objects.new("tgt", None); tgt.location = (0, 0, target_z)
    bpy.context.scene.collection.objects.link(tgt)
    c = cam.constraints.new("TRACK_TO"); c.target = tgt


def render_cell(builder, frame, facing_deg, ortho, target_z):
    from PIL import Image
    B.reset()
    builder(mats(), frame)
    scene = bpy.context.scene
    facer = bpy.data.objects.new("facer", None); scene.collection.objects.link(facer)
    for o in list(scene.objects):
        if o.type == "MESH" and o.parent is None:
            o.parent = facer
    facer.rotation_euler = (0, 0, math.radians(facing_deg))
    setup_cam(ortho, target_z)
    tmp = os.path.join(OUT, "_fauna_tmp.png")
    B.render_to(tmp)
    return Image.open(tmp).convert("RGBA")


def render_sheet(name):
    from PIL import Image
    builder, ortho, target_z, tile, _meta = next((b, o, t, ti, mt) for (n, b, o, t, ti, mt) in SPECIES if n == name)
    sheet = Image.new("RGBA", (3 * tile, 4 * tile), (0, 0, 0, 0))
    for row, (_fname, deg) in enumerate(FACINGS):
        for col in range(3):                         # still, w1, w2
            cell = render_cell(builder, col, deg, ortho, target_z)
            cell = cell.resize((tile, tile), Image.LANCZOS)
            sheet.paste(cell, (col * tile, row * tile), cell)
    os.makedirs(FAUNA, exist_ok=True)
    sheet.save(os.path.join(FAUNA, f"{name}.png"))
    print(f"{name}: sheet {3 * tile}x{4 * tile} (tile {tile})")
    return tile


def write_manifest(tiles):
    lines = ["// fauna_manifest.ron — roaming animals (surface_fauna).",
             "// Sheet layout: 3 cols (still, w1, w2) x 4 rows (down, left, right, up).", "("]
    lines.append("    species: [")
    for (n, _b, _o, _t, tile, mt) in SPECIES:
        tr = "[" + ", ".join(f'"{t}"' for t in mt["terrains"]) + "]"
        lines += ["        (",
                  f'            name: "{n}", sheet: "sprites/fauna/{n}.png",',
                  f'            biome: "{mt["biome"]}", terrains: {tr},',
                  f'            tile_w: {tile}, tile_h: {tile},',
                  f'            speed: {mt["speed"]}, flee_speed: {mt["flee_speed"]},',
                  f'            group: {mt["group"]}, flier: {str(mt["flier"]).lower()},',
                  "        ),"]
    lines += ["    ],", ")"]
    open(os.path.join(FAUNA, "fauna_manifest.ron"), "w").write("\n".join(lines))
    print(f"manifest: {len(SPECIES)} species")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    names = args if args else [n for (n, *_r) in SPECIES]
    tiles = {n: render_sheet(n) for n in names}
    if not args:
        write_manifest(tiles)
