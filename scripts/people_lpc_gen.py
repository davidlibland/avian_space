#!/usr/bin/env python3
"""LPC -> avian_space character-layer pipeline.

Reads curated items from people_lpc_config.py, extracts walk frames from the
vendored LPC generator repo (scripts/lpc), assembles game-format 3x4 sheets at
32x32, and writes:

  assets/sprites/people/layers/<slot>/<item>[_<sex>][_<layerN>].png
  assets/sprites/people/layers.ron       (items + z-order + palette ramps)
  assets/CREDITS-SPRITES.md              (aggregated artist credits)

Color variation is applied at RUNTIME by src/character_compositor.rs using the
palette ramps embedded in layers.ron (exact-color remap of the material's base
ramp), so recolorable items ship exactly one base PNG per (item, sex, layer).
Variant-style LPC items (pre-baked colors, no palettes) ship one PNG per
curated variant with no runtime remap.

Usage:
  .venv/bin/python scripts/people_lpc_gen.py            # generate
  .venv/bin/python scripts/people_lpc_gen.py --preview  # contact sheet -> /tmp
  .venv/bin/python scripts/people_lpc_gen.py --strict   # skip CC-BY-SA/GPL-only items

Deterministic: re-running produces byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import people_lpc_config as cfg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # repo root
LPC = ROOT / "scripts" / "lpc"
OUT_LAYERS = ROOT / "assets" / "sprites" / "people" / "layers"
OUT_RON = ROOT / "assets" / "sprites" / "people" / "layers.ron"
OUT_CREDITS = ROOT / "assets" / "CREDITS-SPRITES.md"

PERMISSIVE = {"CC0", "OGA-BY 3.0", "OGA-BY 3.0+", "CC-BY 3.0", "CC-BY 3.0+", "CC-BY 4.0"}


# ── palette loading ──────────────────────────────────────────────────────────

def load_palettes():
    """material -> {"base": [rgb...], "ramps": {name: [rgb...]}, "alien_ramps": [...]}"""
    out = {}
    for material, groups in cfg.RAMPS.items():
        pdir = LPC / "palette_definitions" / material
        meta = json.loads((pdir / f"meta_{material}.json").read_text())
        base_name = meta["base"]
        # Collect ramp definitions across every palette file for the material.
        # The material's *default* palette file (usually ulpc) is what the
        # source art is painted in, so it takes priority on name collisions.
        default_file = pdir / f"{material}_{meta.get('default', 'ulpc')}.json"
        files = [default_file] + [p for p in sorted(pdir.glob(f"{material}_*.json"))
                                  if p != default_file]
        defs = {}
        for pf in files:
            if not pf.exists():
                continue
            for name, colors in json.loads(pf.read_text()).items():
                defs.setdefault(name, colors)
        if base_name not in defs:
            sys.exit(f"[palette] base ramp {base_name!r} not found for {material}")
        base = defs[base_name]
        ramps, missing = {}, []
        for group, names in groups.items():
            for n in names:
                if n in defs:
                    ramps[n] = (group, defs[n])
                else:
                    missing.append(n)
        if missing:
            print(f"[palette] WARN {material}: missing ramps {missing}; "
                  f"available: {sorted(defs)[:40]}")
        out[material] = {"base": base, "ramps": ramps}
    return out


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ── LPC definition handling ──────────────────────────────────────────────────

def load_def(def_path):
    p = LPC / def_path
    if not p.exists():
        return None
    return json.loads(p.read_text())


def def_layers(d):
    """Yield (layer_index, zPos, {sex: srcdir}) for layer_1..layer_9."""
    for k in sorted(d.keys()):
        m = re.fullmatch(r"layer_(\d+)", k)
        if not m:
            continue
        layer = d[k]
        paths = {s: layer[s] for s in cfg.SEXES if layer.get(s)}
        yield int(m.group(1)), layer.get("zPos"), paths


def def_materials(d):
    """All recolor materials of the item ([] for variant-style).

    Handles both single-material (`recolors: {material: hair}`) and multi-slot
    (`recolors: {color_1: {material: body}, color_2: {material: eye}}`) defs.
    """
    rec = d.get("recolors")
    if not rec:
        return []
    if "material" in rec:
        return [rec["material"]]
    mats = []
    for k in sorted(rec.keys()):
        if re.fullmatch(r"color_\d+", k) and isinstance(rec[k], dict):
            m = rec[k].get("material")
            if m:
                mats.append(m)
    return mats


def def_licenses(d):
    lic = set()
    for c in d.get("credits", []):
        lic.update(c.get("licenses", []))
    return lic


# ── frame extraction ─────────────────────────────────────────────────────────

def load_walk_source(srcdir, variant=None):
    """Return the 576x256 walk sheet Image or None.

    Recolor-style: <srcdir>/walk.png ; variant-style: <srcdir>/walk/<variant>.png
    """
    base = LPC / "spritesheets" / srcdir
    p = base / "walk" / f"{variant}.png" if variant else base / "walk.png"
    if not p.exists():
        walkdir = base / "walk"
        if walkdir.is_dir():
            avail = sorted(f.stem for f in walkdir.glob("*.png"))
            print(f"[gen] HINT {srcdir}: variant-style; available: {avail}")
        return None
    return Image.open(p).convert("RGBA")


def assemble_game_sheet(walk_img):
    """LPC walk sheet (9x4 of 64) -> game sheet (3x4 of TILE)."""
    f = cfg.LPC_FRAME
    sheet = Image.new("RGBA", (cfg.GAME_COLS * f, cfg.GAME_ROWS * f), (0, 0, 0, 0))
    for game_row, lpc_row in enumerate(cfg.LPC_ROW_FOR_GAME_ROW):
        for game_col, lpc_col in enumerate(cfg.LPC_WALK_COLS):
            frame = walk_img.crop((lpc_col * f, lpc_row * f, (lpc_col + 1) * f, (lpc_row + 1) * f))
            sheet.paste(frame, (game_col * f, game_row * f))
    return sheet.resize((cfg.GAME_COLS * cfg.TILE, cfg.GAME_ROWS * cfg.TILE), Image.NEAREST)


# ── main generation ──────────────────────────────────────────────────────────

def slug(s):
    return re.sub(r"[^a-z0-9_]+", "_", s.lower())


def generate(strict=False, preview=False):
    palettes = load_palettes()
    entries = []          # layers.ron item entries
    credits = {}          # (authors tuple, licenses tuple, url) -> [files]
    skipped, warned = [], []

    for item in cfg.ITEMS:
        d = load_def(item["def_path"])
        if d is None:
            warned.append(f"missing def: {item['def_path']}")
            continue
        if "walk" not in d.get("animations", ["walk"]):
            warned.append(f"no walk animation: {item['id']}")
            continue
        lic = def_licenses(d)
        if strict and lic and not (lic & PERMISSIVE):
            skipped.append(f"{item['id']} ({sorted(lic)})")
            continue

        materials = def_materials(d)
        variants = item["variants"] or [None]
        for c in d.get("credits", []):
            key = (tuple(c.get("authors", [])), tuple(c.get("licenses", [])),
                   tuple(c.get("urls", [])))
            credits.setdefault(key, set()).add(c.get("file", item["def_path"]))

        for layer_idx, zpos, sex_paths in def_layers(d):
            z = zpos if zpos is not None else cfg.SLOTS[item["slot"]]["z"]
            # Hats draw over whatever hair remains (close-cropped styles).
            if item["slot"] == "hat":
                z = max(z, 126)
            # Optional curation-level sex restriction (beards, sexed heads).
            if item["sexes"]:
                sex_paths = {s: p for s, p in sex_paths.items() if s in item["sexes"]}
            # Collapse identical per-sex paths (e.g. hair 'adult' shared).
            by_src = {}
            for sex, src in sex_paths.items():
                by_src.setdefault(src, []).append(sex)
            for src, sexes in sorted(by_src.items()):
                for variant in variants:
                    img = load_walk_source(src, variant)
                    if img is None:
                        warned.append(f"no walk src: {item['id']} {src} variant={variant}")
                        continue
                    sheet = assemble_game_sheet(img)
                    suffix = ""
                    if len(by_src) > 1:
                        suffix += "_" + "".join(s[0] for s in sexes)  # _m / _f
                    if variant:
                        suffix += f"_{slug(variant)}"
                    if layer_idx > 1:
                        suffix += f"_l{layer_idx}"
                    fname = f"{item['id']}{suffix}.png"
                    rel = f"sprites/people/layers/{item['slot']}/{fname}"
                    out = OUT_LAYERS / item["slot"] / fname
                    out.parent.mkdir(parents=True, exist_ok=True)
                    sheet.save(out, optimize=True)
                    entries.append({
                        "id": item["id"] + (f"_{slug(variant)}" if variant else ""),
                        "slot": item["slot"],
                        "layer": layer_idx,
                        "z": z,
                        "path": rel,
                        "materials": materials if not variant else [],
                        "sexes": sorted(sexes),
                        "roles": item["roles"],
                        "tags": item["tags"],
                    })

    write_ron(entries, palettes)
    write_credits(credits)

    print(f"[gen] wrote {len(entries)} layer sheets under {OUT_LAYERS}")
    if skipped:
        print(f"[gen] --strict skipped: {', '.join(skipped)}")
    for w in warned:
        print(f"[gen] WARN {w}")
    if preview:
        write_preview(entries, palettes)
    return entries


# ── outputs ──────────────────────────────────────────────────────────────────

def ron_str_list(xs):
    return "[" + ", ".join(f'"{x}"' for x in xs) + "]"


def write_ron(entries, palettes):
    lines = []
    lines.append("// GENERATED by scripts/people_lpc_gen.py — do not edit by hand.")
    lines.append("// LPC-derived character layers; see assets/CREDITS-SPRITES.md.")
    lines.append("(")
    lines.append(f"    tile_w: {cfg.TILE}, tile_h: {cfg.TILE}, "
                 f"cols: {cfg.GAME_COLS}, rows: {cfg.GAME_ROWS}, "
                 f"walk_speed: {cfg.WALK_SPEED},")
    lines.append("    palettes: {")
    for material, p in sorted(palettes.items()):
        lines.append(f'        "{material}": (')
        lines.append(f"            base: {ron_str_list(p['base'])},")
        lines.append("            ramps: {")
        for name, (group, colors) in sorted(p["ramps"].items()):
            lines.append(f'                "{name}": (group: "{group}", '
                         f"colors: {ron_str_list(colors)}),")
        lines.append("            },")
        lines.append("        ),")
    lines.append("    },")
    lines.append("    items: [")
    for e in sorted(entries, key=lambda e: (e["slot"], e["id"], e["layer"], e["sexes"])):
        lines.append(
            f'        (id: "{e["id"]}", slot: "{e["slot"]}", layer: {e["layer"]}, '
            f'z: {e["z"]}, path: "{e["path"]}", materials: {ron_str_list(e["materials"])}, '
            f'sexes: {ron_str_list(e["sexes"])}, roles: {ron_str_list(e["roles"])}, '
            f'tags: {ron_str_list(e["tags"])}),'
        )
    lines.append("    ],")
    lines.append(")")
    OUT_RON.write_text("\n".join(lines) + "\n")
    print(f"[gen] wrote {OUT_RON}")


def write_credits(credits):
    lines = [
        "# Character Sprite Credits",
        "",
        "NPC and player character sprites are composited from the Liberated",
        "Pixel Cup (LPC) asset collection, via the",
        "[Universal LPC Spritesheet Character Generator]"
        "(https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator).",
        "",
        "Per the LPC terms these art assets (and our derived layer sheets under",
        "`assets/sprites/people/layers/`) are licensed under the licenses listed",
        "for each entry below (in general CC-BY-SA 3.0 / GPL 3.0, some CC0 /",
        "OGA-BY 3.0 / CC-BY). This applies to the ART ONLY; game code is not",
        "covered by these licenses.",
        "",
    ]
    for (authors, licenses, urls), files in sorted(
            credits.items(), key=lambda kv: (kv[0][0], sorted(kv[1]))):
        lines.append(f"## {', '.join(authors) if authors else 'unknown'}")
        lines.append(f"- Licenses: {', '.join(licenses) if licenses else 'unknown'}")
        for u in urls:
            lines.append(f"- URL: {u}")
        lines.append(f"- Files: {', '.join(sorted(files))}")
        lines.append("")
    OUT_CREDITS.write_text("\n".join(lines))
    print(f"[gen] wrote {OUT_CREDITS}")


# ── preview ──────────────────────────────────────────────────────────────────

def remap(img, base, target):
    """Exact-color remap base ramp -> target ramp (rank-interpolated lengths)."""
    b = [hex_to_rgb(c) for c in base]
    t = [hex_to_rgb(c) for c in target]
    lut = {}
    for i, c in enumerate(b):
        j = round(i * (len(t) - 1) / max(len(b) - 1, 1))
        lut[c] = t[j]
    out = img.copy()
    px = out.load()
    w, h = out.size
    for y in range(h):
        for x in range(w):
            r, g, bl, a = px[x, y]
            if a and (r, g, bl) in lut:
                nr, ng, nb = lut[(r, g, bl)]
                px[x, y] = (nr, ng, nb, a)
    return out


def write_preview(entries, palettes, n=24):
    """Composite n random characters into a contact sheet at 4x zoom."""
    rng = random.Random(7)
    by_slot = {}
    for e in entries:
        by_slot.setdefault(e["slot"], []).append(e)

    def pick(slot, sex, allow_none, exclude_tags=()):
        cands = [e for e in by_slot.get(slot, [])
                 if sex in e["sexes"] and e["layer"] == 1 and "alien" not in e["tags"]
                 and not (set(e["tags"]) & set(exclude_tags))]
        if not cands:
            return None
        if allow_none and rng.random() < 0.35:
            return None
        return rng.choice(cands)

    tile_w, tile_h = cfg.GAME_COLS * cfg.TILE, cfg.GAME_ROWS * cfg.TILE
    cols = 8
    rows = (n + cols - 1) // cols
    board = Image.new("RGBA", (cols * tile_w, rows * tile_h), (40, 40, 48, 255))

    for i in range(n):
        sex = rng.choice(cfg.SEXES)
        chosen = []
        hat = pick("hat", sex, allow_none=True)
        for slot, meta in cfg.SLOTS.items():
            if slot == "hat":
                e = hat
            elif slot == "hair":
                # A hat hides bulky hair; only close-cropped styles fit under.
                e = pick(slot, sex, allow_none=True,
                         exclude_tags=("bulky",) if hat else ())
            else:
                e = pick(slot, sex, allow_none=not meta["required"])
            if e is None:
                continue
            # include extra layers of the same item id (behind-body parts)
            same = [x for x in by_slot[slot]
                    if x["id"] == e["id"] and sex in x["sexes"]]
            chosen.extend(same)
        # material -> ramp choice, consistent across this character's layers
        ramp_pick = {}
        for mat, p in palettes.items():
            human = [k for k, (g, _) in p["ramps"].items() if g != "alien"]
            if human:
                ramp_pick[mat] = p["ramps"][rng.choice(human)][1]
        comp = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
        for e in sorted(chosen, key=lambda e: e["z"]):
            img = Image.open(ROOT / "assets" / e["path"]).convert("RGBA")
            for mat in e["materials"]:
                if mat in ramp_pick:
                    img = remap(img, palettes[mat]["base"], ramp_pick[mat])
            comp.alpha_composite(img)
        board.paste(comp, ((i % cols) * tile_w, (i // cols) * tile_h))

    board = board.resize((board.width * 4, board.height * 4), Image.NEAREST)
    out = Path("/tmp/people_lpc_preview.png")
    board.save(out)
    print(f"[gen] preview -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="skip items with only copyleft (CC-BY-SA/GPL) licensing")
    ap.add_argument("--preview", action="store_true",
                    help="also write a composited contact sheet to /tmp")
    args = ap.parse_args()
    generate(strict=args.strict, preview=args.preview)
