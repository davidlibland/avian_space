"""Curation tables for the LPC -> avian_space character-layer pipeline.

Edit this file to change which LPC items ship as character layers, which
palette ramps are exposed, and which roles/tags each item carries.
`people_lpc_gen.py` reads these tables, validates every referenced
definition against the vendored LPC repo (scripts/lpc), and warns about
anything missing rather than failing hard.

Item tuple fields:
  def_path : sheet_definitions/... json path (relative to scripts/lpc)
  slot     : game slot (body/head/hair/beard/torso/legs/feet/hat)
  id       : stable item id used in layers.ron and filenames
  roles    : role affinities ([] = generic civilian-usable)
  tags     : free tags ("alien", "fancy", ...)
  variants : for variant-style items (no recolor palettes): which baked
             color variants to ship (each becomes its own item id `id_<v>`)
"""

# Body types we ship (LPC also has muscular/pregnant/teen).
SEXES = ["male", "female"]

# ── Palette ramps exposed to the game ────────────────────────────────────────
# material -> list of ramp names looked up across the LPC palette files
# (body_ulpc.json / body_lpcr.json etc). The material's `base` ramp (from
# meta_<material>.json) is what the source art is painted in.
RAMPS = {
    "body": {
        "human": [
            "ivory", "porcelain", "peach", "tan", "tawny",
            "honey", "bronze", "brown", "coffee",
        ],
        # Distinct alien set: never mixed into the civilian human pool.
        "alien": ["blue", "green", "lavender", "pale_green", "zombie_green"],
    },
    "hair": {
        "all": [
            "black", "charcoal", "brown", "chestnut", "ash_brown",
            "blonde", "platinum", "gray", "orange", "red" ,
            "blue", "green", "purple", "pink",
        ],
    },
    "cloth": {
        "all": [
            "white", "black", "gray", "charcoal", "brown", "leather",
            "tan", "red", "maroon", "orange", "yellow", "forest",
            "green", "teal", "sky", "blue", "navy", "purple",
            "lavender", "pink", "rose", "bluegray",
        ],
    },
    "metal": {
        "all": ["steel", "iron", "brass", "bronze", "gold", "copper"],
    },
    # Eye color (heads carry eyes painted in the eye base ramp).
    "eye": {
        "all": ["blue", "brown", "green", "gray", "hazel", "purple"],
    },
}

# ── Curated items ────────────────────────────────────────────────────────────
def I(def_path, slot, id, roles=(), tags=(), variants=(), sexes=()):
    """sexes: optional restriction, e.g. ("male",) for beards / sexed heads."""
    return {
        "def_path": def_path, "slot": slot, "id": id,
        "roles": list(roles), "tags": list(tags), "variants": list(variants),
        "sexes": list(sexes),
    }

ITEMS = [
    # Slot: body (required, material=body) ------------------------------------
    I("sheet_definitions/body/body.json", "body", "body"),

    # Slot: head (required, material=body + eye) -------------------------------
    I("sheet_definitions/head/heads/human/heads_human_male.json", "head", "head_m",
      sexes=("male",)),
    I("sheet_definitions/head/heads/human/heads_human_female.json", "head", "head_f",
      sexes=("female",)),

    # Slot: hair (optional, material=hair). "bulky" hair is excluded when a hat
    # is worn (rule mirrored in the preview and in character_compositor.rs).
    I("sheet_definitions/hair/afro/hair_afro.json", "hair", "afro", tags=("bulky",)),
    I("sheet_definitions/hair/afro/hair_cornrows.json", "hair", "cornrows"),
    I("sheet_definitions/hair/afro/hair_dreadlocks_short.json", "hair", "dreads_short"),
    I("sheet_definitions/hair/afro/hair_flat_top_fade.json", "hair", "flat_top"),
    I("sheet_definitions/hair/bald/hair_buzzcut.json", "hair", "buzzcut"),
    I("sheet_definitions/hair/short/hair_bedhead.json", "hair", "bedhead"),
    I("sheet_definitions/hair/short/hair_cowlick.json", "hair", "cowlick"),
    I("sheet_definitions/hair/bob/hair_bob.json", "hair", "bob", tags=("bulky",)),
    I("sheet_definitions/hair/curly/hair_curly_short.json", "hair", "curly_short", tags=("bulky",)),
    I("sheet_definitions/hair/short/hair_bangs.json", "hair", "bangs"),
    I("sheet_definitions/hair/long/hair_bangslong.json", "hair", "bangs_long", tags=("bulky",)),
    I("sheet_definitions/hair/braids/hair_braid.json", "hair", "braid", tags=("bulky",)),
    I("sheet_definitions/hair/short/hair_curtains.json", "hair", "curtains"),

    # Slot: beard (optional, material=hair, male-only by convention) ----------
    I("sheet_definitions/hair/beards/beards_beard.json", "beard", "beard", sexes=("male",)),
    I("sheet_definitions/hair/mustaches/beards_mustache.json", "beard", "mustache", sexes=("male",)),
    I("sheet_definitions/hair/beards/beards_5oclock_shadow.json", "beard", "stubble", sexes=("male",)),

    # Slot: torso (required; these shirts are variant-style = baked colors) ---
    I("sheet_definitions/torso/shirts/torso_clothes_tunic.json", "torso", "tunic",
      variants=("blue", "forest", "red", "charcoal", "tan", "purple")),
    I("sheet_definitions/torso/shirts/torso_clothes_blouse.json", "torso", "blouse",
      variants=("white", "blue", "rose", "forest")),
    I("sheet_definitions/torso/shirts/torso_clothes_blouse_longsleeve.json", "torso", "blouse_ls",
      variants=("white", "charcoal", "navy")),
    I("sheet_definitions/torso/shirts/sleeves/torso_clothes_longsleeves.json", "torso", "longsleeve"),
    I("sheet_definitions/torso/shirts/torso_clothes_corset.json", "torso", "corset", tags=("fancy",),
      variants=("maroon", "brown")),
    I("sheet_definitions/torso/jacket/torso_jacket_collared.json", "torso", "jacket_collared",
      roles=("merchant",), variants=("charcoal", "navy", "brown")),
    I("sheet_definitions/torso/jacket/torso_jacket_trench.json", "torso", "jacket_trench",
      roles=("merchant",), variants=("dark_gray", "gray")),
    I("sheet_definitions/torso/armour/torso_armour_leather.json", "torso", "armour_leather", roles=("guard",)),
    I("sheet_definitions/torso/torso_chainmail.json", "torso", "chainmail", roles=("guard",)),
    I("sheet_definitions/torso/aprons/torso_aprons_overalls.json", "torso", "overalls",
      roles=("miner", "mechanic"), variants=("blue", "brown", "charcoal", "forest")),
    I("sheet_definitions/torso/aprons/torso_aprons_apron.json", "torso", "apron",
      roles=("merchant",), variants=("white", "brown", "forest")),

    # Slot: legs (required, material cloth/metal) ------------------------------
    I("sheet_definitions/legs/pants/legs_pants.json", "legs", "pants"),
    I("sheet_definitions/legs/pants/legs_cuffed.json", "legs", "pants_cuffed"),
    I("sheet_definitions/legs/pants/legs_formal.json", "legs", "pants_formal", roles=("merchant",)),
    I("sheet_definitions/legs/shorts/legs_shorts.json", "legs", "shorts"),
    I("sheet_definitions/legs/skirts/legs_skirts_plain.json", "legs", "skirt"),
    I("sheet_definitions/legs/legs_armour.json", "legs", "legs_armour", roles=("guard",)),

    # Slot: feet (required; mostly variant-style) ------------------------------
    I("sheet_definitions/feet/shoes/feet_shoes_basic.json", "feet", "shoes",
      variants=("black", "brown", "maroon")),
    I("sheet_definitions/feet/boots/feet_boots_basic.json", "feet", "boots",
      variants=("black", "brown", "charcoal", "leather")),
    I("sheet_definitions/feet/feet_sandals.json", "feet", "sandals",
      variants=("brown",)),

    # Slot: hat (optional; recolor-style, cloth material at runtime) -----------
    I("sheet_definitions/headwear/coverings/bandana/hat_bandana.json", "hat", "bandana"),
    I("sheet_definitions/headwear/coverings/headbands/hat_headband_tied.json", "hat", "headband"),
]

# Slots, their fallback z (used only if a def lacks zPos), and whether a
# character must have one ("required") or may omit it ("optional").
SLOTS = {
    "body":  {"z": 10,  "required": True},
    "head":  {"z": 100, "required": True},
    "hair":  {"z": 120, "required": False},
    "beard": {"z": 118, "required": False},
    "torso": {"z": 40,  "required": True},
    "legs":  {"z": 20,  "required": True},
    "feet":  {"z": 25,  "required": True},
    "hat":   {"z": 130, "required": False},
}

# Game sheet geometry (matches surface_character.rs people layout).
# Columns per facing row: 2 idle (breathing) + 8 walk cycle + 8 run cycle.
TILE = 32                 # output tile size (LPC 64 halved)
IDLE_FRAMES = 2
WALK_FRAMES = 8
RUN_FRAMES = 8
GAME_COLS = IDLE_FRAMES + WALK_FRAMES + RUN_FRAMES  # 18
GAME_ROWS = 4             # down, left, right, up
WALK_SPEED = 40.0

# LPC source geometry: 64px frames; row order up,left,down,right.
#   walk.png: 9 cols (col 0 = standing pose, cols 1-8 = walk cycle)
#   idle.png: 2 cols (breathing); run.png: 8 cols
LPC_FRAME = 64
LPC_ROW_FOR_GAME_ROW = [2, 1, 3, 0]  # game rows down,left,right,up -> LPC rows
