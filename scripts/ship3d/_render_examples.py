"""Scratch: render full-res populated biome examples for review (docs/<biome>_objects_example.png).
Z-sort matches the game: foot image-y = wy + y_offset (front/south drawn on top)."""
import re, random, sys
import numpy as np
from PIL import Image
import terrain_gen as G

TILE = G.TILE; BLOB47 = G.BLOB47; r47 = G.reduce_to_47
TL, T, TR, L, R_, BL, B, BR = G.TL, G.T, G.TR, G.L, G.R_, G.BL, G.B, G.BR
NB = [(TL, -1, -1), (T, 0, -1), (TR, 1, -1), (L, -1, 0), (R_, 1, 0), (BL, -1, 1), (B, 0, 1), (BR, 1, 1)]
W3 = "../../assets/sprites/worlds"


def parse():
    txt = open(f"{W3}/objects_manifest.ron").read(); biome = None; cur = {}; objs = {}
    for line in txt.splitlines():
        s = line.strip(); mb = re.match(r'"(\w+)":\s*\($', s)
        if mb: biome = mb.group(1); objs[biome] = []
        for k, cast in [("name", str), ("y_px", int), ("n_frames", int), ("n_variants", int),
                        ("tile_w", int), ("tile_h", int), ("terrains", str), ("density", float),
                        ("max_per_tile", int), ("y_offset", float), ("shy", str), ("grid", str)]:
            m = re.match(rf'{k}:\s*(.+?),\s*$', s)
            if m: cur[k] = cast(m.group(1).strip('"')) if cast != str else m.group(1)
        if s.startswith(')') and 'name' in cur and biome: objs[biome].append(cur); cur = {}
    return objs


OBJS = parse()


def render_biome(biome, tnames, terr, seed):
    random.seed(seed); np.random.seed(seed); MH, MW = terr.shape
    atlas = Image.open(f"{W3}/{biome}_atlas.png").convert("RGB"); tim = Image.new("RGB", (MW * TILE, MH * TILE))
    for y in range(MH):
        for x in range(MW):
            t = terr[y, x]; mask = 0
            for bit, dx, dy in NB:
                nx, ny = min(max(x + dx, 0), MW - 1), min(max(y + dy, 0), MH - 1)
                if terr[ny, nx] >= t: mask |= bit
            c = BLOB47.index(r47(mask)); c = 46 + ((x * 7 + y * 13) % G.N_VARIANTS) if c == G.INTERIOR_COL else c
            tim.paste(atlas.crop((c * TILE, t * TILE, c * TILE + TILE, t * TILE + TILE)), (x * TILE, y * TILE))
    cv = tim.convert("RGBA"); oatlas = Image.open(f"{W3}/{biome}_objects.png").convert("RGBA")
    df = G.n01(G.pnoise(2.2, seed + 1, 64))
    df = np.asarray(Image.fromarray((df * 255).astype('uint8')).resize((MW, MH), Image.BICUBIC), float) / 255
    TN = {i: n for i, n in enumerate(tnames)}; placed = []
    for o in OBJS[biome]:
        tn = [t.strip().strip('"') for t in o["terrains"].strip("[]").split(",")]
        for y in range(MH):
            for x in range(MW):
                if TN[terr[y, x]] not in tn: continue
                sp = o["density"] * df[y, x]
                for _ in range(o["max_per_tile"]):
                    if random.random() < sp:
                        var = random.randrange(o["n_variants"]); tw, th = o["tile_w"], o["tile_h"]
                        fr = 0 if o["shy"] == "true" else random.randrange(o["n_frames"])
                        spr = oatlas.crop((var * tw, o["y_px"] + fr * th, var * tw + tw, o["y_px"] + (fr + 1) * th))
                        j = 0.0 if o.get("grid") == "true" else 0.4   # grid → snap to tile centre
                        wx = x * TILE + TILE / 2 + random.uniform(-j, j) * TILE
                        wy = y * TILE + TILE / 2 + random.uniform(-j, j) * TILE
                        placed.append((wy + o["y_offset"], spr, wx, wy, th))   # foot image-y = wy + y_offset
    for key, spr, wx, wy, th in sorted(placed, key=lambda s: s[0]):
        cv.alpha_composite(spr, (int(wx - spr.width / 2), int(wy - th // 2)))
    out = cv.convert("RGB").resize((cv.width * 2, cv.height * 2), Image.NEAREST)
    out.save(f"../../docs/{biome}_objects_example.png")
    print(f"{biome}: {len(placed)} objects -> {out.size}")


MW, MH = 34, 26; yy, xx = np.mgrid[0:MH, 0:MW]
gn = G.n01(G.pnoise(2.6, 30, 64)); gn = np.asarray(Image.fromarray((gn * 255).astype('uint8')).resize((MW, MH), Image.BICUBIC), float) / 255
gt = np.full((MH, MW), 2, int); gt[(yy < 7) | ((yy < 9) & (gn > 0.58))] = 0
gt[((yy >= 6) & (yy < 9)) | ((yy < 10) & (gt == 2) & (gn < 0.32))] = 1
gt[(yy >= 14) & (gn > 0.42)] = 3; gt[yy >= 22] = 4; gt[(yy >= 22) & (gn > 0.58)] = 3
render_biome("garden", ["water", "sand", "grass", "forest", "mountain"], gt, 30)
rn = G.n01(G.pnoise(2.6, 40, 64)); rn = np.asarray(Image.fromarray((rn * 255).astype('uint8')).resize((MW, MH), Image.BICUBIC), float) / 255
rt = np.full((MH, MW), 2, int); rt[rn > 0.63] = 0; rt[(rn > 0.46) & (rn <= 0.63)] = 1
rt[rn < 0.28] = 3; rt[yy >= 23] = 4; rt[(yy >= 23) & (rn > 0.55)] = 1
render_biome("rocky", ["lava", "basalt", "rock", "dust", "cliff"], rt, 40)

# ice: snow base, deep_ice/ice patches, ice_rock outcrops, a crevasse
inz = G.n01(G.pnoise(2.6, 50, 64)); inz = np.asarray(Image.fromarray((inz * 255).astype('uint8')).resize((MW, MH), Image.BICUBIC), float) / 255
it = np.full((MH, MW), 2, int); it[inz > 0.62] = 0; it[(inz > 0.46) & (inz <= 0.62)] = 1; it[inz < 0.27] = 3
it[(yy >= 12) & (yy <= 13) & (xx > 7) & (xx < 27)] = 4
render_biome("ice", ["deep_ice", "ice", "snow", "ice_rock", "crevasse"], it, 50)

# desert: hard_sand base, dunes, sandstone, mesa band, a quicksand patch
dnz = G.n01(G.pnoise(2.6, 60, 64)); dnz = np.asarray(Image.fromarray((dnz * 255).astype('uint8')).resize((MW, MH), Image.BICUBIC), float) / 255
dt = np.full((MH, MW), 2, int); dt[dnz > 0.56] = 1; dt[dnz < 0.30] = 3; dt[yy >= 21] = 4
dt[(yy < 4) & (dnz > 0.55)] = 0
render_biome("desert", ["quicksand", "dunes", "hard_sand", "sandstone", "mesa"], dt, 60)

# interior: floor base, a plating room, grate strips, wall border, conduit lines
ip = np.full((MH, MW), 0, int)
ip[6:18, 4:16] = 1
ip[(yy % 8 == 4)] = 2
ip[(yy % 8 == 5) & (xx > 18)] = 4
ip[(xx == 0) | (xx == MW - 1) | (yy == 0) | (yy == MH - 1)] = 3
render_biome("interior", ["floor", "plating", "grate", "wall", "conduit"], ip, 70)
