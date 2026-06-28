#!/usr/bin/env python3
"""Render a flow chart of the mission graph in assets/missions.yaml.

* Node COLOUR  = the faction the mission is associated with (its employer).
* Node SHAPE   = the objective/stage type (combat, pickup, meet, catch, delivery).
* A GOLD ring + caption marks missions that grant an unlock (ship/weapon licence).
* Arrows follow the `completed` preconditions (what unlocks what), so the whole
  branching campaign — including the cross-faction gates — reads at a glance.

Pure-PIL (no graphviz needed). Layout = left-to-right layered DAG (x = dependency
depth) with a barycentre pass to reduce edge crossings.

Run:  .venv/bin/python scripts/mission_flowchart.py
Out:  docs/mission_flowchart.png
"""
import os
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MISSIONS = os.path.join(ROOT, "assets", "missions.yaml")
OUT = os.path.join(ROOT, "docs", "mission_flowchart.png")

# ── faction (employer) by mission-name prefix → (label, colour) ──────────────
FACTIONS = {
    "deliver":   ("Intro",          (150, 152, 158)),
    "scavenge":  ("Intro",          (150, 152, 158)),
    "mining":    ("Mining Guild",   (176, 132, 78)),
    "merchant":  ("Merchant Guild", (214, 170, 70)),
    "pirate":    ("Federation",     (78, 120, 196)),
    "bounty":    ("Federation",     (78, 120, 196)),
    "espionage": ("Fed Intel",      (70, 150, 165)),
    "frontier":  ("Free Frontier",  (228, 198, 92)),
    "helios":    ("Helios Combine", (84, 200, 230)),
    "bastion":   ("Fed Bastion",    (188, 64, 56)),
    "order":     ("Artifact Order", (162, 110, 214)),
    "rift":      ("Precursor Rift", (196, 84, 206)),
}
FACTION_ORDER = ["Intro", "Mining Guild", "Merchant Guild", "Federation", "Fed Intel",
                 "Free Frontier", "Helios Combine", "Fed Bastion", "Artifact Order", "Precursor Rift"]

# ── objective kind → stage shape + label ─────────────────────────────────────
STAGES = {
    "destroy_ships":             ("combat",   "rect"),
    "catch_npc":                 ("apprehend", "hex"),
    "meet_npc":                  ("meet",     "diamond"),
    "collect_pickups":           ("salvage",  "ellipse"),
    "collect_from_asteroid_field": ("mine",   "ellipse"),
    "land_on_planet":            ("deliver",  "para"),
}

NW, NH = 158, 46          # node box size
COLGAP, ROWGAP = 96, 30   # gaps
MARGIN_L, MARGIN_T = 24, 150


def faction_of(name):
    pre = name.split("_")[0]
    return FACTIONS.get(pre, ("Other", (120, 120, 120)))


def stage_of(m):
    return STAGES.get(m.get("objective", {}).get("kind"), ("?", "rect"))


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw_shape(d, kind, cx, cy, w, h, fill, outline, ow=2):
    l, r, t, b = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    if kind == "rect":
        d.rounded_rectangle([l, t, r, b], radius=6, fill=fill, outline=outline, width=ow)
    elif kind == "ellipse":
        d.ellipse([l, t, r, b], fill=fill, outline=outline, width=ow)
    elif kind == "diamond":
        d.polygon([(cx, t), (r, cy), (cx, b), (l, cy)], fill=fill, outline=outline, width=ow)
    elif kind == "hex":
        i = w * 0.22
        d.polygon([(l + i, t), (r - i, t), (r, cy), (r - i, b), (l + i, b), (l, cy)],
                  fill=fill, outline=outline, width=ow)
    elif kind == "para":
        s = w * 0.16
        d.polygon([(l + s, t), (r, t), (r - s, b), (l, b)], fill=fill, outline=outline, width=ow)


def main():
    M = yaml.safe_load(open(MISSIONS))
    # edges from `completed` preconditions
    preds = {n: [] for n in M}
    succ = {n: [] for n in M}
    for n, v in M.items():
        for pc in v.get("preconditions", []) or []:
            if pc.get("kind") == "completed" and pc.get("mission") in M:
                preds[n].append(pc["mission"]); succ[pc["mission"]].append(n)
    # longest-path depth
    depth = {}
    def d_of(n):
        if n in depth:
            return depth[n]
        depth[n] = 0 if not preds[n] else 1 + max(d_of(p) for p in preds[n])
        return depth[n]
    for n in M:
        d_of(n)
    maxd = max(depth.values())
    cols = {c: [n for n in M if depth[n] == c] for c in range(maxd + 1)}
    # initial y-order within a column: by faction then name
    fidx = {f: i for i, f in enumerate(FACTION_ORDER)}
    order = {}
    for c in range(maxd + 1):
        cols[c].sort(key=lambda n: (fidx.get(faction_of(n)[0], 99), n))
        for i, n in enumerate(cols[c]):
            order[n] = i
    # barycentre passes to reduce crossings
    for _ in range(4):
        for c in range(1, maxd + 1):
            cols[c].sort(key=lambda n: (sum(order[p] for p in preds[n]) / len(preds[n])) if preds[n] else order[n])
            for i, n in enumerate(cols[c]):
                order[n] = i
        for c in range(maxd - 1, -1, -1):
            cols[c].sort(key=lambda n: (sum(order[s] for s in succ[n]) / len(succ[n])) if succ[n] else order[n])
            for i, n in enumerate(cols[c]):
                order[n] = i
    rows = max(len(cols[c]) for c in cols)
    # positions
    pos = {}
    for c in range(maxd + 1):
        x = MARGIN_L + c * (NW + COLGAP) + NW / 2
        for i, n in enumerate(cols[c]):
            y = MARGIN_T + i * (NH + ROWGAP) + NH / 2
            pos[n] = (x, y)
    W = MARGIN_L * 2 + (maxd + 1) * (NW + COLGAP)
    Hh = MARGIN_T + rows * (NH + ROWGAP) + 40
    img = Image.new("RGB", (W, Hh), (28, 30, 38))
    dr = ImageDraw.Draw(img)
    f_node = font(12, True); f_sub = font(10); f_title = font(22, True); f_leg = font(13)
    # edges (under nodes)
    for n, v in M.items():
        x2, y2 = pos[n]
        for p in preds[n]:
            x1, y1 = pos[p]
            col = (110, 114, 126)
            dr.line([(x1 + NW / 2, y1), (x2 - NW / 2, y2)], fill=col, width=2)
            # arrowhead
            import math
            ang = math.atan2(y2 - y1, (x2 - NW / 2) - (x1 + NW / 2))
            ax, ay = x2 - NW / 2, y2
            dr.polygon([(ax, ay), (ax - 9 * math.cos(ang - 0.4), ay - 9 * math.sin(ang - 0.4)),
                        (ax - 9 * math.cos(ang + 0.4), ay - 9 * math.sin(ang + 0.4))], fill=col)
    # nodes
    for n, v in M.items():
        cx, cy = pos[n]
        flabel, fcol = faction_of(n)
        slabel, shape = stage_of(v)
        grants = [e["name"] for e in v.get("completion_effects", []) or [] if e.get("kind") == "grant_unlock"]
        outline = (235, 200, 90) if grants else (18, 19, 24)
        ow = 4 if grants else 2
        draw_shape(dr, shape, cx, cy, NW, NH, fcol, outline, ow)
        # text: mission name (wrapped-ish) + stage
        name = n if len(n) <= 21 else n[:20] + "…"
        tc = (15, 16, 20)
        dr.text((cx, cy - 7), name, font=f_node, fill=tc, anchor="mm")
        dr.text((cx, cy + 9), "▸ " + slabel, font=f_sub, fill=tc, anchor="mm")
        if grants:
            g = ", ".join(x.replace("_license", "").replace("_", " ") for x in grants)
            dr.text((cx, cy + NH / 2 + 9), "unlocks " + g, font=f_sub, fill=(235, 205, 110), anchor="mm")
    # title + legend
    dr.text((MARGIN_L, 16), "avian_space — Mission Flow", font=f_title, fill=(240, 240, 248))
    dr.text((MARGIN_L, 48), "colour = faction · shape = stage type · gold ring = grants a ship/weapon licence · arrows = prerequisite",
            font=f_leg, fill=(170, 174, 186))
    # faction swatches
    lx = MARGIN_L; ly = 78
    for f in FACTION_ORDER:
        col = next(c for (lab, c) in FACTIONS.values() if lab == f)
        dr.rounded_rectangle([lx, ly, lx + 16, ly + 16], radius=3, fill=col)
        dr.text((lx + 22, ly + 8), f, font=f_leg, fill=(210, 213, 222), anchor="lm")
        lx += 30 + dr.textlength(f, font=f_leg) + 22
    # stage shapes
    lx = MARGIN_L; ly = 108
    seen = []
    for kind, (slab, shp) in STAGES.items():
        if slab in seen:
            continue
        seen.append(slab)
        draw_shape(dr, shp, lx + 12, ly + 8, 24, 16, (90, 92, 100), (210, 213, 222), 2)
        dr.text((lx + 30, ly + 8), slab, font=f_leg, fill=(210, 213, 222), anchor="lm")
        lx += 30 + dr.textlength(slab, font=f_leg) + 22
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT)
    print(f"wrote {OUT}  ({W}x{Hh}, {len(M)} missions, depth {maxd})")


if __name__ == "__main__":
    main()
