"""
terrain_3d_proto.py — Blender 3D terrain tiles, to compare vs the 2D approach.

A periodic heightmap (so it tiles) displaces a subdivided plane; rendered
straight-down ortho with a no-shadow sun + ambient occlusion for real depth.
No cast shadows / only periodic inputs → seamless.

Run:  scripts/.blender_venv/bin/python terrain_3d_proto.py
Out:  out/_terrain3d.png
"""
import math
import os

import bpy  # noqa
import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), "out")
HN = 512  # heightmap resolution


def periodic_noise(n, beta, seed):
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n) * n
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy); fr[0, 0] = 1.0
    spec = fr ** (-beta) * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, n)))
    a = np.fft.ifft2(spec).real
    return (a - a.mean()) / (a.std() + 1e-9)


def make_heightmap(name, beta, seed, cracks=None):
    h = periodic_noise(HN, beta, seed) + 0.45 * periodic_noise(HN, beta * 0.7, seed + 9)
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    if cracks:
        f = periodic_noise(HN, cracks["beta"], seed + 13)
        f = (f - f.min()) / (f.max() - f.min() + 1e-9)
        cr = np.exp(-((f - 0.5) ** 2) / (cracks["eps"] ** 2))
        h = h - cr * cracks["carve"]
        h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    path = os.path.join(OUT, f"_hm_{name}.png")
    Image.fromarray((h * 255).astype("uint8"), "L").save(path)
    return path


def terrain_material(name, ramp, relief):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.95
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -relief * 0.55
    mr.inputs["From Max"].default_value = relief * 0.55
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    cr = nt.nodes.new("ShaderNodeValToRGB")
    el = cr.color_ramp
    el.elements[0].position = ramp[0][0]
    el.elements[0].color = (*ramp[0][1], 1)
    for pos, col in ramp[1:-1]:
        e = el.elements.new(pos); e.color = (*col, 1)
    el.elements[-1].position = ramp[-1][0]
    el.elements[-1].color = (*ramp[-1][1], 1)
    nt.links.new(mr.outputs[0], cr.inputs[0])
    nt.links.new(cr.outputs[0], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return mat


def C(*v):
    return tuple(x / 255.0 for x in v)


SPECS = {
    # softer / finer features so a constant-terrain field doesn't show an
    # obvious repeating motif.
    "rock": dict(beta=2.3, relief=0.34, seed=2,
                 cracks=dict(beta=1.7, eps=0.07, carve=0.28),
                 ramp=[(0.0, C(54, 50, 47)), (0.45, C(98, 92, 85)), (1.0, C(158, 150, 139))]),
    "sand": dict(beta=2.7, relief=0.24, seed=4,
                 ramp=[(0.0, C(170, 140, 94)), (0.5, C(206, 178, 128)), (1.0, C(234, 208, 158))]),
    "ice": dict(beta=2.7, relief=0.22, seed=6,
                cracks=dict(beta=2.0, eps=0.06, carve=0.22),
                ramp=[(0.0, C(120, 152, 188)), (0.5, C(176, 202, 224)), (1.0, C(216, 232, 245))]),
    "snow": dict(beta=2.8, relief=0.26, seed=8,
                 ramp=[(0.0, C(204, 214, 230)), (0.5, C(232, 238, 248)), (1.0, C(250, 251, 255))]),
}


def setup_terrain_scene():
    scene = bpy.context.scene
    cam_d = bpy.data.cameras.new("cam"); cam_d.type = "ORTHO"; cam_d.ortho_scale = 2.0
    cam = bpy.data.objects.new("cam", cam_d); scene.collection.objects.link(cam)
    cam.location = (0, 0, 6); cam.rotation_euler = (0, 0, 0); scene.camera = cam
    # key sun (NO shadow → no cross-tile shadow seams) + fill
    for nm, e, rot, col, sh in (
        ("key", 3.2, (math.radians(42), math.radians(-26), 0), (1, 0.97, 0.92), False),
        ("fill", 1.0, (math.radians(-30), math.radians(28), 0), (0.85, 0.9, 1.0), False),
    ):
        d = bpy.data.lights.new(nm, "SUN"); d.energy = e; d.color = col
        d.use_shadow = sh
        o = bpy.data.objects.new(nm, d); scene.collection.objects.link(o); o.rotation_euler = rot
    if scene.world is None:
        scene.world = bpy.data.worlds.new("world")
    scene.world.use_nodes = False
    scene.world.color = (0.16, 0.16, 0.18)   # ambient fill
    scene.render.engine = "BLENDER_EEVEE"
    try:
        scene.eevee.use_gtao = True
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256


def render_terrain(name, spec):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=300, y_subdivisions=300, size=2)
    obj = bpy.context.active_object
    hm = make_heightmap(name, spec["beta"], spec["seed"], spec.get("cracks"))
    img = bpy.data.images.load(hm)
    tex = bpy.data.textures.new(f"d_{name}", "IMAGE"); tex.image = img; tex.extension = "REPEAT"
    m = obj.modifiers.new("disp", "DISPLACE")
    m.texture = tex; m.texture_coords = "UV"; m.strength = spec["relief"]; m.mid_level = 0.5
    obj.data.materials.append(terrain_material(name, spec["ramp"], spec["relief"]))
    bpy.ops.object.shade_smooth()
    setup_terrain_scene()
    tmp = os.path.join(OUT, f"_t3_{name}.png")
    bpy.context.scene.render.filepath = os.path.abspath(tmp)
    bpy.ops.render.render(write_still=True)
    return Image.open(tmp).convert("RGB")


def main():
    # Repetition test: tile the game-size (32px) tile in a big grid, the way
    # the game actually renders a constant-terrain region.
    names = list(SPECS)
    R = 8           # 8×8 field of repeats
    g = 32          # game tile px
    field = R * g
    pad = 14
    cv = Image.new("RGB", (len(names) * (field + pad) + pad, field + 44), (24, 26, 32))
    d = ImageDraw.Draw(cv)
    d.text((pad, 6), "BLENDER 3D TERRAIN — 8×8 repetition test at the 32px game tile size",
           fill=(235, 235, 245))
    for i, name in enumerate(names):
        full = render_terrain(name, SPECS[name])
        tile = full.resize((g, g), Image.LANCZOS)    # downscale to game size
        block = Image.new("RGB", (field, field))
        for rx in range(R):
            for ry in range(R):
                block.paste(tile, (rx * g, ry * g))
        x = pad + i * (field + pad)
        cv.paste(block, (x, 30))
        d.text((x + 4, 30 + field - 16), name, fill=(245, 245, 255))
    cv.save(os.path.join(OUT, "_terrain3d.png"))
    print("saved _terrain3d.png")


if __name__ == "__main__":
    main()
