import buildings3d as bd, blender_gen as B, bpy
from PIL import Image, ImageDraw

STYLE = "colony"
ANGLES = [("3-4 front", 50, 0), ("3-4 turned", 45, 38), ("side", 16, 90),
          ("front-on", 6, 0), ("below", -30, 12), ("top-down", 80, 0)]

def build(remove_back):
    B.reset(); s = bd.STYLES[STYLE]; m = bd.mats(s)
    w, d, h = 6, 4, 3.2
    z0, top = bd.shell(s, m, w, d, h, front=False, door=False)
    fz = -d/2; dw = 1.8
    if remove_back:                      # debevel body so the through-cut booleans clean
        body = bpy.data.objects.get("body")
        for mod in list(body.modifiers):
            if mod.type == "BEVEL": body.modifiers.remove(mod)
    B.add_box("doorpanel", (0, fz-0.07, z0+1.1), (dw, 0.1, 2.1), m["wall_d"], bevel=0.02)
    if remove_back:
        bd.cut_doorway(m, ["body"], 0, fz, dw, 2.15, z0+1.1, d+1.0, solver="MANIFOLD")
    else:
        bd.cut_doorway(m, ["body"], 0, fz, dw, 2.15, z0+1.1, d/2+0.3)
    B.add_box("lintel", (0, fz-0.05, z0+2.2), (dw+0.3, 0.16, 0.18), m["glow"], bevel=0.0)
    for sx in (-dw/2-0.16, dw/2+0.16):
        B.add_box("jamb", (sx, fz-0.05, z0+1.15), (0.16, 0.18, 2.2), m["wall_d"], bevel=0.03)
    B.add_cylinder("chimney", (1.7, fz-0.05, z0+1.75), 0.2, 3.5, m["metal"], axis="z")
    B.add_cylinder("chimcap", (1.7, fz-0.05, z0+3.55), 0.27, 0.22, m["dark"], axis="z")
    B.add_box("fl_hazard", (0, fz-0.12, z0+0.04), (dw, 0.06, 0.12), m["glow"], bevel=0.0)
    bd.repair_engine(-2.0, -d/2-1.5, m)
    B.add_box("toolbox", (w/2-1.5, -d/2-1.4, 0.35), (0.9, 0.6, 0.5), m["roof"], bevel=0.05)
    B.add_cylinder("jib_p", (w/2-0.3, d/2-0.3, top+0.8), 0.14, 2.0, m["metal"], axis="z")
    B.add_box("jib_a", (w/2-1.2, d/2-0.3, top+1.4), (2.0, 0.2, 0.2), m["metal"], bevel=0.03)

def crop(im):
    a = im.split()[3]; bb = a.getbbox(); return im.crop(bb) if bb else im

cells = {}
for rb in (False, True):
    build(rb)
    B.setup_scene(bd.ORTHO, bd.RES)
    for nm, e, a in ANGLES:
        bd.setup_iso(e, a)
        p = f"out/_ang_{int(rb)}_{nm}.png"
        B.render_to(p)
        cells[(rb, nm)] = crop(Image.open(p).convert("RGBA"))

CELL = 230; pad = 8; lbl = 16
W = CELL*2 + pad*3 + 70
H = (CELL+lbl)*len(ANGLES) + pad*(len(ANGLES)+1) + 24
canvas = Image.new("RGBA", (W, H), (200, 200, 206, 255)); dr = ImageDraw.Draw(canvas)
dr.text((90, 4), "WITH back wall", fill=(20,20,24,255)); dr.text((90+CELL+pad, 4), "WITHOUT back wall", fill=(20,20,24,255))
for r, (nm, e, a) in enumerate(ANGLES):
    y = 24 + r*(CELL+lbl+pad)
    dr.text((4, y+CELL//2), nm, fill=(20,20,24,255))
    for c, rb in enumerate((False, True)):
        im = cells[(rb, nm)]
        s = min((CELL-4)/im.width, (CELL-4)/im.height)
        im2 = im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.LANCZOS)
        x = 70 + c*(CELL+pad)
        canvas.alpha_composite(im2, (x+(CELL-im2.width)//2, y+(CELL-im2.height)//2))
canvas.convert("RGB").save("../../docs/previews/mechanic_angles.png")
print("saved", canvas.size)
