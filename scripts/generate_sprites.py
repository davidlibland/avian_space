"""
Ship sprite generator for avian_space.

Environment setup (one-time):
  conda env create -f environment.yml
  conda activate avian-sprites

Run with:  python3 generate_sprites.py
Requires:  Pillow  (pinned in environment.yml)

─── Style guide ────────────────────────────────────────────────────────────────

All sprites are RGBA PNGs.  Transparent pixels use (0,0,0,0).

Orientation
  The ship nose points UP (negative-Y in image space).  The game reads the
  sprite as-is and rotates it at runtime.

Size convention
  Sprite size ≈ radius × 2.2  (rounded to a sensible even number).
  Keep sizes consistent with ships.yaml so the visual matches the collision
  radius used by the physics engine.

  shuttle        radius 10  →  22 × 22
  fighter        radius 12  →  28 × 28
  asteroid_miner radius 20  →  44 × 44
  hauler         radius 40  →  88 × 88
  corvette       radius  9  →  20 × 40  (tall rectangular canvas for 2× fighter length)
  frigate        radius 32  →  70 × 70
  courier        radius 12  →  26 × 26
  freighter      radius 32  →  70 × 70
  bulk_carrier   radius 55  → 120 × 120

Colour palette conventions
  Fighters / military  – cool blue-grays  (100-140, 130-160, 175-200, 255)
  Traders / cargo      – warm gray-tans   (95-165, 90-160, 80-150, 255)
  Mining               – earthy browns    (80-160, 80-130, 50-90,  255)
  Engine glows         – blue-white       (80, 160-180, 255, 180-210)
                       – or warm orange   (255, 160-180, 80-100, 180-210)
                         (fighters use blue; traders use orange)

Drawing approach
  Use PIL ImageDraw.polygon / rectangle primitives.  Shapes are defined with
  float coordinates relative to cx = S/2 (horizontal centre) so they stay
  symmetric and are easy to tweak.

  Typical layer order (back to front):
    1. Side wings / nacelles   – darkest shade
    2. Main hull polygon       – mid shade
    3. Nose highlight          – lightest shade
    4. Detail panels / lines   – darkest shade
    5. Engine glow rectangles  – translucent

Adding a new ship
  1. Decide the sprite size from the radius (size = round(radius * 2.2 / 2) * 2).
  2. Copy one of the draw_* functions below as a template.
  3. Follow the colour conventions above for the ship's role.
  4. Call your function at the bottom of the file.
  5. Update ships.yaml: set sprite_path to ship_sprites/<name>.png.
"""

import os

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "ship_sprites")


def make_img(size):
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


# ─── SHUTTLE (22×22) – starter general-purpose ship ──────────────────────────
# Design note: the shuttle is the player's starting vessel and doubles as a
# generic civilian transport seen in all systems.  It deliberately uses a
# brighter blue-teal body rather than the warm gray-tan trader palette, so the
# player can pick it out visually in a crowd of AI ships.  Engine glow is warm
# orange, consistent with the civilian / trader convention.


def draw_shuttle():
    S = 22
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Compact tapered hull – slightly wider at the midpoint than at the base
    # so it reads as a rounded capsule rather than a sharp dart.
    hull = [(cx, 2), (cx + 5, 11), (cx + 4, 18), (cx - 4, 18), (cx - 5, 11)]
    d.polygon(hull, fill=(80, 150, 190, 255))

    # Nose highlight – lighter band to suggest a curved forward section
    nose = [(cx, 2), (cx + 3, 9), (cx - 3, 9)]
    d.polygon(nose, fill=(140, 200, 230, 255))

    # Short swept wings – kept stubby to signal "not a fighter"
    lwing = [(cx - 5, 11), (cx - 9, 17), (cx - 6, 18), (cx - 4, 13)]
    rwing = [(cx + 5, 11), (cx + 9, 17), (cx + 6, 18), (cx + 4, 13)]
    d.polygon(lwing, fill=(55, 115, 155, 255))
    d.polygon(rwing, fill=(55, 115, 155, 255))

    # Cockpit window – small ellipse near the nose
    d.ellipse([int(cx) - 2, 5, int(cx) + 2, 10], fill=(180, 230, 255, 200))

    # Single engine slot (warm orange – civilian convention)
    d.rectangle([int(cx) - 3, 18, int(cx) + 3, 20], fill=(255, 165, 80, 180))

    img.save(os.path.join(OUT, "shuttle.png"))


# ─── FIGHTER (28×28) – fast military interceptor ─────────────────────────────
# Design note: slim central spine to convey speed; highly swept wings that
# start mid-hull and angle sharply back — common "fighter" silhouette cue at
# small pixel sizes.  Military cool blue-gray palette throughout.  Dual blue
# engine glow slots rather than a single slot reinforce the "powerful" read.


def draw_fighter():
    S = 28
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Slim hull – narrow diamond shape, nose-heavy
    hull = [(cx, 2), (cx + 3, 14), (cx, 18), (cx - 3, 14)]
    d.polygon(hull, fill=(180, 200, 230, 255))

    # Nose highlight – bright tip to sharpen the silhouette
    nose = [(cx, 2), (cx + 2, 8), (cx - 2, 8)]
    d.polygon(nose, fill=(220, 235, 255, 255))

    # Highly swept wings – attach mid-hull, tip at 22 y (near bottom of sprite)
    lwing = [(cx - 3, 10), (cx - 11, 22), (cx - 3, 18)]
    rwing = [(cx + 3, 10), (cx + 11, 22), (cx + 3, 18)]
    d.polygon(lwing, fill=(130, 155, 195, 255))
    d.polygon(rwing, fill=(130, 155, 195, 255))

    # Dual engine glow slots (blue-white – military convention)
    d.rectangle([int(cx) - 3, 18, int(cx) - 1, 22], fill=(80, 170, 255, 190))
    d.rectangle([int(cx) + 1, 18, int(cx) + 3, 22], fill=(80, 170, 255, 190))
    # Bright inner core
    d.rectangle([int(cx) - 2, 19, int(cx) + 2, 21], fill=(200, 230, 255, 220))

    img.save(os.path.join(OUT, "fighter.png"))


# ─── ASTEROID MINER (44×44) – industrial mining vessel ───────────────────────
# Design note: wide, squat silhouette signals slow but tough.  The defining
# visual feature is the pair of lateral drill arms that extend beyond the hull
# width, each tipped with a triangular drill bit.  Earthy brown palette marks
# it as a mining class.  Two separate engine pods sit low on the hull.
# Engine glow is warm orange (civilian worker, not military).


def draw_asteroid_miner():
    S = 44
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Wide hexagonal hull – broad shoulders taper toward tail
    hull = [
        (cx, 3),
        (cx + 10, 14),
        (cx + 12, 30),
        (cx, 34),
        (cx - 12, 30),
        (cx - 10, 14),
    ]
    d.polygon(hull, fill=(155, 125, 85, 255))

    # Armour plating seams – horizontal lines that break up the hull
    d.line([(cx - 10, 14), (cx + 10, 14)], fill=(100, 78, 48, 255), width=2)
    d.line([(cx - 11, 22), (cx + 11, 22)], fill=(100, 78, 48, 255), width=2)

    # Lateral drill arms – rectangular struts, set behind the nose
    d.rectangle([int(cx) - 18, 16, int(cx) - 13, 32], fill=(115, 95, 65, 255))
    d.rectangle([int(cx) + 13, 16, int(cx) + 18, 32], fill=(115, 95, 65, 255))

    # Drill bits – pointed triangles at the arm tips; golden colour for worn
    # hardened steel, contrasting with the brown hull
    d.polygon([(cx - 18, 32), (cx - 13, 32), (cx - 15, 38)], fill=(195, 155, 75, 255))
    d.polygon([(cx + 13, 32), (cx + 18, 32), (cx + 15, 38)], fill=(195, 155, 75, 255))

    # Cockpit blister near the nose
    d.ellipse([int(cx) - 4, 6, int(cx) + 4, 13], fill=(80, 155, 200, 200))

    # Engine pods – two rectangular nacelles symmetrically placed
    d.rectangle([int(cx) - 9, 30, int(cx) - 4, 38], fill=(78, 78, 95, 255))
    d.rectangle([int(cx) + 4, 30, int(cx) + 9, 38], fill=(78, 78, 95, 255))

    # Engine glow (warm orange – civilian / worker convention, not military blue)
    d.ellipse([int(cx) - 9, 35, int(cx) - 4, 41], fill=(255, 160, 80, 180))
    d.ellipse([int(cx) + 4, 35, int(cx) + 9, 41], fill=(255, 160, 80, 180))

    img.save(os.path.join(OUT, "asteroid_miner.png"))


# ─── HAULER (88×88) – heavy cargo transport ───────────────────────────────────
# Design note: the hauler is defined by its cargo containers rather than by
# any aerodynamic shape.  Three rows of paired containers flank a central
# spine; horizontal latch lines make each container read as a discrete unit.
# The tiny nose/cockpit area at the top is deliberately undersized relative to
# the cargo mass — the ship is "all hold, no hull".
# Warm gray-tan body; four-nozzle engine bank with orange glow (trader class).


def draw_hauler():
    S = 88
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Central spine – runs almost the full height, slightly recessed colour
    d.rectangle([int(cx) - 8, 8, int(cx) + 8, 72], fill=(138, 143, 152, 255))

    # Narrow nose cap – small triangle so the cockpit reads tiny vs. cargo bulk
    d.polygon([(cx, 4), (cx - 8, 14), (cx + 8, 14)], fill=(168, 173, 183, 255))

    # Cockpit window – centred in the nose triangle
    d.polygon([(cx, 5), (cx - 5, 13), (cx + 5, 13)], fill=(80, 155, 200, 200))

    # Three rows of cargo containers, two per side per row.
    # Colour steps slightly toward darker / cooler tones lower down, so the
    # stacked rows don't look uniform.
    for row, y in enumerate([14, 30, 46]):
        dark = (88 + row * 12, 108, 128 - row * 8, 255)
        light = (100 + row * 12, 118, 138 - row * 8, 255)
        latch = (58 + row * 4, 62, 68 - row * 4, 255)

        # Left container
        d.rectangle([int(cx) - 30, y, int(cx) - 10, y + 14], fill=dark)
        d.rectangle([int(cx) - 29, y + 1, int(cx) - 11, y + 13], fill=light)
        d.line([(int(cx) - 30, y + 7), (int(cx) - 10, y + 7)], fill=latch, width=1)

        # Right container
        d.rectangle([int(cx) + 10, y, int(cx) + 30, y + 14], fill=dark)
        d.rectangle([int(cx) + 11, y + 1, int(cx) + 29, y + 13], fill=light)
        d.line([(int(cx) + 10, y + 7), (int(cx) + 30, y + 7)], fill=latch, width=1)

    # Engine block – wide rectangular housing for the four nozzles
    d.rectangle([int(cx) - 14, 64, int(cx) + 14, 78], fill=(98, 98, 112, 255))

    # Four individual nozzles with orange glow (warm trader convention)
    for ex in [int(cx) - 11, int(cx) - 3, int(cx) + 3, int(cx) + 11]:
        d.rectangle([ex - 3, 74, ex + 3, 82], fill=(68, 68, 82, 255))
        d.ellipse([ex - 3, 79, ex + 3, 86], fill=(255, 160, 80, 170))

    img.save(os.path.join(OUT, "hauler.png"))


# ─── CORVETTE (20×20) – needle-slim fast interceptor ─────────────────────────
# Design note: the corvette is physically smaller (radius 9) than the fighter
# (radius 12) but needs to read as clearly distinct from it.  The key
# differentiator is aspect ratio: the corvette uses a very narrow (3 px) needle
# hull that fills almost the full 20 px height, giving it a tall-thin silhouette
# versus the fighter's stockier diamond.  The fins are small delta tabs placed
# near the tail rather than mid-ship swept wings, further distinguishing the
# two outlines at a glance.


def draw_corvette():
    S = 20
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Needle hull – only 3 px wide at the shoulder, uses y 1→18
    hull = [
        (cx, 1),
        (cx + 2, 7),
        (cx + 3, 14),
        (cx + 2, 18),
        (cx - 2, 18),
        (cx - 3, 14),
        (cx - 2, 7),
    ]
    d.polygon(hull, fill=(135, 158, 198, 255))

    # Bright needle tip
    nose = [(cx, 1), (cx + 1, 5), (cx - 1, 5)]
    d.polygon(nose, fill=(215, 230, 255, 255))

    # Small delta tail-fins – low on the hull, near-horizontal sweep
    lfin = [(cx - 3, 13), (cx - 9, 18), (cx - 6, 18), (cx - 2, 15)]
    rfin = [(cx + 3, 13), (cx + 9, 18), (cx + 6, 18), (cx + 2, 15)]
    d.polygon(lfin, fill=(95, 122, 165, 255))
    d.polygon(rfin, fill=(95, 122, 165, 255))

    # Single-slot engine glow (blue – military)
    d.rectangle([int(cx) - 2, 18, int(cx) + 2, 19], fill=(80, 165, 255, 190))

    img.save(os.path.join(OUT, "corvette.png"))


# ─── FRIGATE (70×70) – medium warship ─────────────────────────────────────────


def draw_frigate():
    S = 70
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Main spine / hull
    hull = [(cx, 3), (cx + 14, 20), (cx + 16, 52), (cx - 16, 52), (cx - 14, 20)]
    d.polygon(hull, fill=(90, 105, 125, 255))

    # Nose highlight band
    nose = [(cx, 3), (cx + 8, 18), (cx - 8, 18)]
    d.polygon(nose, fill=(150, 170, 200, 255))

    # Center panel
    center = [(cx - 8, 20), (cx + 8, 20), (cx + 10, 48), (cx - 10, 48)]
    d.polygon(center, fill=(110, 125, 148, 255))

    # Weapon pod stubs (horizontal bars)
    d.rectangle([int(cx) - 24, 26, int(cx) - 14, 32], fill=(70, 85, 105, 255))
    d.rectangle([int(cx) + 14, 26, int(cx) + 24, 32], fill=(70, 85, 105, 255))
    # Barrel tips
    d.rectangle([int(cx) - 26, 29, int(cx) - 24, 31], fill=(40, 55, 75, 255))
    d.rectangle([int(cx) + 24, 29, int(cx) + 26, 31], fill=(40, 55, 75, 255))

    # Side nacelles
    lnac = [(cx - 18, 34), (cx - 22, 38), (cx - 20, 52), (cx - 16, 52)]
    rnac = [(cx + 18, 34), (cx + 22, 38), (cx + 20, 52), (cx + 16, 52)]
    d.polygon(lnac, fill=(75, 90, 110, 255))
    d.polygon(rnac, fill=(75, 90, 110, 255))

    # Engine exhausts
    d.rectangle([int(cx) - 12, 52, int(cx) - 6, 55], fill=(80, 160, 255, 200))
    d.rectangle([int(cx) + 6, 52, int(cx) + 12, 55], fill=(80, 160, 255, 200))
    d.rectangle([int(cx) - 3, 52, int(cx) + 3, 56], fill=(80, 180, 255, 200))
    d.rectangle([int(cx) - 20, 52, int(cx) - 16, 54], fill=(80, 140, 220, 180))
    d.rectangle([int(cx) + 16, 52, int(cx) + 20, 54], fill=(80, 140, 220, 180))

    img.save(os.path.join(OUT, "frigate.png"))


# ─── COURIER (26×26) – fast trader ────────────────────────────────────────────

# ─── COURIER (26×26) – thin streamlined fast trader ──────────────────────────
# Design note: the courier is fast (max_speed 160) and that should be legible
# in the silhouette.  The previous design had a wide hull and a cargo blister
# that made it look sluggish.  This version uses a narrow cigar fuselage (4 px
# at widest) with long swept-back fins starting near mid-hull — a shape that
# reads as "built for speed, carries a little on the side".  The cargo blister
# is removed; the courier trades cargo capacity for agility.  Orange engine
# glow, trader convention.


def draw_courier():
    S = 26
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Slim cigar fuselage – only 4 px wide, pointed at both ends
    hull = [
        (cx, 2),
        (cx + 2, 8),
        (cx + 4, 16),
        (cx + 3, 22),
        (cx - 3, 22),
        (cx - 4, 16),
        (cx - 2, 8),
    ]
    d.polygon(hull, fill=(158, 143, 112, 255))

    # Sharp nose highlight
    nose = [(cx, 2), (cx + 2, 7), (cx - 2, 7)]
    d.polygon(nose, fill=(205, 190, 158, 255))

    # Long swept-back fins – start at mid-hull, sweep far aft
    # The wide span (to x±11) signals speed; the shallow angle (nearly parallel
    # to hull axis) reads as "streamlined" rather than "aggressive".
    lfin = [(cx - 4, 12), (cx - 11, 22), (cx - 7, 22), (cx - 3, 15)]
    rfin = [(cx + 4, 12), (cx + 11, 22), (cx + 7, 22), (cx + 3, 15)]
    d.polygon(lfin, fill=(125, 112, 85, 255))
    d.polygon(rfin, fill=(125, 112, 85, 255))

    # Twin engine glows (warm orange – trader convention)
    d.rectangle([int(cx) - 3, 22, int(cx) - 1, 25], fill=(255, 162, 80, 180))
    d.rectangle([int(cx) + 1, 22, int(cx) + 3, 25], fill=(255, 162, 80, 180))

    img.save(os.path.join(OUT, "courier.png"))


# ─── FREIGHTER (70×70) – medium cargo ─────────────────────────────────────────


def draw_freighter():
    S = 70
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Boxy main hull
    hull = [(cx, 4), (cx + 12, 18), (cx + 18, 58), (cx - 18, 58), (cx - 12, 18)]
    d.polygon(hull, fill=(130, 125, 115, 255))

    # Nose section
    nose = [(cx, 4), (cx + 8, 16), (cx - 8, 16)]
    d.polygon(nose, fill=(165, 160, 150, 255))

    # Cargo modules (side boxes)
    d.rectangle([int(cx) - 26, 22, int(cx) - 14, 50], fill=(110, 105, 98, 255))
    d.rectangle([int(cx) + 14, 22, int(cx) + 26, 50], fill=(110, 105, 98, 255))

    # Center cargo hold indicator
    d.rectangle([int(cx) - 9, 20, int(cx) + 9, 55], fill=(115, 110, 102, 255))

    # Panel lines on cargo modules
    d.rectangle([int(cx) - 26, 32, int(cx) - 14, 33], fill=(85, 80, 73, 255))
    d.rectangle([int(cx) + 14, 32, int(cx) + 26, 33], fill=(85, 80, 73, 255))
    d.rectangle([int(cx) - 26, 42, int(cx) - 14, 43], fill=(85, 80, 73, 255))
    d.rectangle([int(cx) + 14, 42, int(cx) + 26, 43], fill=(85, 80, 73, 255))

    # Engine exhausts (warm orange – trader colour convention)
    d.rectangle([int(cx) - 16, 58, int(cx) - 10, 62], fill=(255, 160, 80, 180))
    d.rectangle([int(cx) - 5, 58, int(cx) + 5, 63], fill=(255, 170, 90, 200))
    d.rectangle([int(cx) + 10, 58, int(cx) + 16, 62], fill=(255, 160, 80, 180))

    img.save(os.path.join(OUT, "freighter.png"))


# ─── BULK CARRIER (120×120) – massive cargo ───────────────────────────────────


def draw_bulk_carrier():
    S = 120
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Massive boxy main hull
    hull = [(cx, 6), (cx + 18, 28), (cx + 35, 90), (cx - 35, 90), (cx - 18, 28)]
    d.polygon(hull, fill=(115, 112, 108, 255))

    # Nose cap
    nose = [(cx, 6), (cx + 14, 26), (cx - 14, 26)]
    d.polygon(nose, fill=(148, 145, 140, 255))

    # Four cargo pod columns (two on each side)
    for lx in [int(cx) - 50, int(cx) - 32]:
        d.rectangle([lx, 30, lx + 16, 80], fill=(95, 93, 88, 255))
        for py in [40, 55, 68]:
            d.rectangle([lx, py, lx + 16, py + 1], fill=(65, 63, 60, 255))

    for rx in [int(cx) + 16, int(cx) + 34]:
        d.rectangle([rx, 30, rx + 16, 80], fill=(95, 93, 88, 255))
        for py in [40, 55, 68]:
            d.rectangle([rx, py, rx + 16, py + 1], fill=(65, 63, 60, 255))

    # Central spine
    d.rectangle([int(cx) - 14, 28, int(cx) + 14, 90], fill=(105, 102, 98, 255))
    for py in [38, 55, 72]:
        d.rectangle([int(cx) - 12, py, int(cx) + 12, py + 1], fill=(75, 72, 68, 255))

    # Bottom connector bar
    d.rectangle([int(cx) - 50, 80, int(cx) + 50, 92], fill=(100, 97, 92, 255))

    # Engine bank (warm orange – trader colour convention)
    for ex in range(int(cx) - 42, int(cx) + 46, 14):
        d.rectangle([ex, 92, ex + 10, 98], fill=(255, 160, 80, 190))
    d.rectangle([int(cx) - 6, 92, int(cx) + 6, 100], fill=(255, 180, 100, 210))

    img.save(os.path.join(OUT, "bulk_carrier.png"))


# ─── FED PATROL (34×34) – Federation fast patrol interceptor ─────────────────
# Design note: Narrow angular dart, distinctly military. Charcoal-black hull
# with a grey nose section and red accent stripe across the mid-ship. Swept
# wings tipped in red. Dual blue engine glow (military convention).


def draw_fed_patrol():
    S = 34
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Angular charcoal hull
    hull = [(cx, 2), (cx + 6, 12), (cx + 5, 26), (cx - 5, 26), (cx - 6, 12)]
    d.polygon(hull, fill=(42, 44, 48, 255))

    # Grey nose section
    nose = [(cx, 2), (cx + 4, 10), (cx - 4, 10)]
    d.polygon(nose, fill=(115, 118, 126, 255))

    # Centre panel (slightly lighter dark)
    center = [(cx - 3, 10), (cx + 3, 10), (cx + 4, 24), (cx - 4, 24)]
    d.polygon(center, fill=(62, 65, 70, 255))

    # Red accent stripe
    d.rectangle([int(cx) - 6, 13, int(cx) + 6, 15], fill=(210, 28, 28, 255))

    # Swept wings
    lwing = [(cx - 6, 12), (cx - 13, 24), (cx - 9, 26), (cx - 5, 16)]
    rwing = [(cx + 6, 12), (cx + 13, 24), (cx + 9, 26), (cx + 5, 16)]
    d.polygon(lwing, fill=(52, 54, 59, 255))
    d.polygon(rwing, fill=(52, 54, 59, 255))

    # Red wing tips
    d.polygon([(cx - 13, 24), (cx - 11, 21), (cx - 9, 26)], fill=(210, 28, 28, 255))
    d.polygon([(cx + 13, 24), (cx + 11, 21), (cx + 9, 26)], fill=(210, 28, 28, 255))

    # Dual engine glow (blue – military)
    d.rectangle([int(cx) - 4, 26, int(cx) - 1, 30], fill=(80, 160, 255, 195))
    d.rectangle([int(cx) + 1, 26, int(cx) + 4, 30], fill=(80, 160, 255, 195))

    img.save(os.path.join(OUT, "fed_patrol.png"))


# ─── FED DESTROYER (92×92) – Federation heavy capital ship ───────────────────
# Design note: Massive and imposing. Very dark hull with grey nose highlight
# and horizontal red armour-seam lines. Wide flanking armour plates. Two
# heavy weapon turrets with barrel stubs. Four-nozzle blue engine bank.


def draw_fed_destroyer():
    S = 92
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Side armour flanks (behind main hull so they go on first)
    lflank = [(cx - 22, 28), (cx - 30, 38), (cx - 28, 68), (cx - 20, 68)]
    rflank = [(cx + 22, 28), (cx + 30, 38), (cx + 28, 68), (cx + 20, 68)]
    d.polygon(lflank, fill=(44, 46, 50, 255))
    d.polygon(rflank, fill=(44, 46, 50, 255))

    # Massive angular main hull
    hull = [(cx, 4), (cx + 20, 24), (cx + 24, 68), (cx - 24, 68), (cx - 20, 24)]
    d.polygon(hull, fill=(36, 38, 42, 255))

    # Grey nose cap
    nose = [(cx, 4), (cx + 12, 22), (cx - 12, 22)]
    d.polygon(nose, fill=(82, 85, 92, 255))

    # Red stripe below nose
    d.rectangle([int(cx) - 12, 22, int(cx) + 12, 25], fill=(210, 25, 25, 255))

    # Centre armour panel
    center = [(cx - 9, 25), (cx + 9, 25), (cx + 11, 64), (cx - 11, 64)]
    d.polygon(center, fill=(50, 52, 58, 255))

    # Horizontal armour seam lines (red)
    for y in [34, 46, 57]:
        d.line([(int(cx) - 20, y), (int(cx) + 20, y)], fill=(200, 22, 22, 200), width=1)

    # Heavy weapon turrets
    d.rectangle([int(cx) - 26, 30, int(cx) - 14, 40], fill=(28, 29, 33, 255))
    d.rectangle([int(cx) - 30, 33, int(cx) - 26, 37], fill=(18, 18, 22, 255))
    d.rectangle([int(cx) + 14, 30, int(cx) + 26, 40], fill=(28, 29, 33, 255))
    d.rectangle([int(cx) + 26, 33, int(cx) + 30, 37], fill=(18, 18, 22, 255))

    # Engine block
    d.rectangle([int(cx) - 16, 68, int(cx) + 16, 78], fill=(28, 29, 34, 255))

    # Four engine nozzles with blue glow
    for ex in [int(cx) - 12, int(cx) - 4, int(cx) + 4, int(cx) + 12]:
        d.rectangle([ex - 3, 74, ex + 3, 82], fill=(22, 22, 28, 255))
        d.ellipse([ex - 3, 78, ex + 3, 88], fill=(80, 155, 255, 188))

    img.save(os.path.join(OUT, "fed_destroyer.png"))


# ─── FED MISSILE CRUISER (78×78) – Federation long-range missile platform ─────
# Design note: Long narrow spine flanked by two large missile pod wings.
# Three rows of tube openings on each pod, trimmed in red. Dark hull.
# The silhouette reads as "all ordnance, no hull".


def draw_fed_missile_cruiser():
    S = 78
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Missile pod wings (behind spine)
    lpod = [(cx - 10, 22), (cx - 28, 28), (cx - 28, 55), (cx - 10, 52)]
    rpod = [(cx + 10, 22), (cx + 28, 28), (cx + 28, 55), (cx + 10, 52)]
    d.polygon(lpod, fill=(45, 47, 52, 255))
    d.polygon(rpod, fill=(45, 47, 52, 255))

    # Missile tube rows on each pod (three rows, red opening trim)
    for y in [30, 38, 46]:
        d.rectangle([int(cx) - 27, y, int(cx) - 11, y + 6], fill=(28, 28, 33, 255))
        d.rectangle([int(cx) - 27, y, int(cx) - 25, y + 6], fill=(185, 20, 20, 255))
        d.rectangle([int(cx) + 11, y, int(cx) + 27, y + 6], fill=(28, 28, 33, 255))
        d.rectangle([int(cx) + 25, y, int(cx) + 27, y + 6], fill=(185, 20, 20, 255))

    # Long narrow spine hull
    spine = [(cx, 4), (cx + 8, 18), (cx + 8, 62), (cx - 8, 62), (cx - 8, 18)]
    d.polygon(spine, fill=(40, 42, 46, 255))

    # Grey nose
    nose = [(cx, 4), (cx + 6, 17), (cx - 6, 17)]
    d.polygon(nose, fill=(92, 95, 102, 255))

    # Red nose stripe
    d.rectangle([int(cx) - 6, 17, int(cx) + 6, 20], fill=(210, 25, 25, 255))

    # Engine section
    d.rectangle([int(cx) - 10, 62, int(cx) + 10, 68], fill=(28, 28, 34, 255))

    # Three nozzles with blue glow
    for ex in [int(cx) - 7, int(cx), int(cx) + 7]:
        d.rectangle([ex - 3, 66, ex + 3, 72], fill=(20, 20, 26, 255))
        d.ellipse([ex - 3, 69, ex + 3, 76], fill=(80, 155, 255, 188))

    img.save(os.path.join(OUT, "fed_missile_cruiser.png"))


# ─── REBEL FIGHTER (28×28) – fast Rebel interceptor ──────────────────────────
# Design note: Deep blue hull with bright green nose and wing tips — the
# Rebel colour pair. Slightly broader than the Federation fighter, conveying
# a more organic build. Single bright blue-green engine glow.


def draw_rebel_fighter():
    S = 28
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Blue hull
    hull = [(cx, 2), (cx + 4, 10), (cx + 4, 20), (cx - 4, 20), (cx - 4, 10)]
    d.polygon(hull, fill=(30, 82, 145, 255))

    # Green nose highlight
    nose = [(cx, 2), (cx + 3, 9), (cx - 3, 9)]
    d.polygon(nose, fill=(55, 190, 105, 255))

    # Cockpit window (teal)
    d.ellipse([int(cx) - 2, 5, int(cx) + 2, 10], fill=(100, 215, 185, 200))

    # Swept delta wings
    lwing = [(cx - 4, 9), (cx - 12, 20), (cx - 7, 21), (cx - 4, 14)]
    rwing = [(cx + 4, 9), (cx + 12, 20), (cx + 7, 21), (cx + 4, 14)]
    d.polygon(lwing, fill=(22, 62, 115, 255))
    d.polygon(rwing, fill=(22, 62, 115, 255))

    # Green wing tips
    d.polygon([(cx - 12, 20), (cx - 10, 17), (cx - 7, 21)], fill=(50, 175, 80, 255))
    d.polygon([(cx + 12, 20), (cx + 10, 17), (cx + 7, 21)], fill=(50, 175, 80, 255))

    # Engine glow (blue-green)
    d.rectangle([int(cx) - 3, 20, int(cx) + 3, 24], fill=(60, 205, 185, 200))
    d.rectangle([int(cx) - 1, 21, int(cx) + 1, 26], fill=(155, 240, 220, 230))

    img.save(os.path.join(OUT, "rebel_fighter.png"))


# ─── REBEL GUNBOAT (48×48) – Rebel medium attack vessel ──────────────────────
# Design note: Wider and stockier than the rebel fighter. Twin forward gun
# barrels read as a weapon-forward design. Blue hull with a single green
# accent stripe across the shoulder. Three-nozzle engine cluster.


def draw_rebel_gunboat():
    S = 48
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Side wing plates (behind hull)
    lplate = [(cx - 12, 18), (cx - 20, 26), (cx - 18, 38), (cx - 10, 38)]
    rplate = [(cx + 12, 18), (cx + 20, 26), (cx + 18, 38), (cx + 10, 38)]
    d.polygon(lplate, fill=(24, 68, 102, 255))
    d.polygon(rplate, fill=(24, 68, 102, 255))

    # Main hull
    hull = [(cx, 3), (cx + 10, 18), (cx + 12, 38), (cx - 12, 38), (cx - 10, 18)]
    d.polygon(hull, fill=(35, 92, 135, 255))

    # Nose / cockpit section
    nose = [(cx, 3), (cx + 7, 16), (cx - 7, 16)]
    d.polygon(nose, fill=(65, 162, 152, 255))

    # Cockpit window
    d.ellipse([int(cx) - 4, 6, int(cx) + 4, 13], fill=(120, 222, 202, 200))

    # Center panel
    d.rectangle([int(cx) - 5, 17, int(cx) + 5, 36], fill=(44, 112, 158, 255))

    # Green accent stripe
    d.rectangle([int(cx) - 12, 20, int(cx) + 12, 22], fill=(55, 192, 95, 255))

    # Forward gun barrels (twin)
    d.rectangle([int(cx) - 9, 3, int(cx) - 6, 13], fill=(18, 52, 78, 255))
    d.rectangle([int(cx) + 6, 3, int(cx) + 9, 13], fill=(18, 52, 78, 255))

    # Three engine nozzles (blue-green)
    for ex in [int(cx) - 7, int(cx), int(cx) + 7]:
        d.rectangle([ex - 2, 38, ex + 2, 42], fill=(60, 182, 202, 192))
    d.rectangle([int(cx) - 1, 39, int(cx) + 1, 45], fill=(140, 230, 210, 215))

    img.save(os.path.join(OUT, "rebel_gunboat.png"))


# ─── REBEL FRIGATE (62×62) – Rebel medium warship ────────────────────────────
# Design note: Angular wedge hull with forward-swept side nacelles that hold
# extra gun hardpoints. Faster than the Federation frigate — the lines are
# sharper and there's less bulk. Blue hull, green accent stripe, blue-green
# engine cluster.


def draw_rebel_frigate():
    S = 62
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Side nacelles (behind main hull)
    lnac = [(cx - 16, 28), (cx - 22, 36), (cx - 20, 50), (cx - 14, 50)]
    rnac = [(cx + 16, 28), (cx + 22, 36), (cx + 20, 50), (cx + 14, 50)]
    d.polygon(lnac, fill=(22, 66, 102, 255))
    d.polygon(rnac, fill=(22, 66, 102, 255))

    # Main hull – wedge
    hull = [(cx, 3), (cx + 14, 22), (cx + 16, 50), (cx - 16, 50), (cx - 14, 22)]
    d.polygon(hull, fill=(30, 86, 128, 255))

    # Nose highlight
    nose = [(cx, 3), (cx + 9, 20), (cx - 9, 20)]
    d.polygon(nose, fill=(58, 158, 182, 255))

    # Centre panel
    center = [(cx - 7, 22), (cx + 7, 22), (cx + 8, 46), (cx - 8, 46)]
    d.polygon(center, fill=(42, 110, 150, 255))

    # Green accent stripe
    d.rectangle([int(cx) - 14, 26, int(cx) + 14, 28], fill=(52, 192, 88, 255))

    # Gun barrel stubs on nacelles
    d.rectangle([int(cx) - 24, 38, int(cx) - 20, 42], fill=(15, 48, 75, 255))
    d.rectangle([int(cx) + 20, 38, int(cx) + 24, 42], fill=(15, 48, 75, 255))

    # Engine exhausts (blue-green)
    d.rectangle([int(cx) - 10, 50, int(cx) - 5, 54], fill=(58, 188, 202, 196))
    d.rectangle([int(cx) + 5, 50, int(cx) + 10, 54], fill=(58, 188, 202, 196))
    d.rectangle([int(cx) - 3, 50, int(cx) + 3, 56], fill=(100, 222, 202, 212))
    d.rectangle([int(cx) - 18, 50, int(cx) - 14, 52], fill=(48, 158, 172, 180))
    d.rectangle([int(cx) + 14, 50, int(cx) + 18, 52], fill=(48, 158, 172, 180))

    img.save(os.path.join(OUT, "rebel_frigate.png"))


# ─── PIRATE CORVETTE (24×24) – scrappy fast Pirate attacker ──────────────────
# Design note: Brown/tan narrow blade, slightly asymmetric wings to suggest
# field repairs and mismatched salvage. A rust patch on one wing reinforces
# the cobbled-together aesthetic. Warm orange engine glow (not military).


def draw_pirate_corvette():
    S = 24
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Narrow blade hull
    hull = [(cx, 1), (cx + 3, 8), (cx + 3, 18), (cx - 3, 18), (cx - 3, 8)]
    d.polygon(hull, fill=(128, 96, 62, 255))

    # Tan nose tip
    nose = [(cx, 1), (cx + 2, 7), (cx - 2, 7)]
    d.polygon(nose, fill=(188, 158, 112, 255))

    # Slightly mismatched wings (pirate patchwork)
    lwing = [(cx - 3, 8), (cx - 10, 17), (cx - 7, 18), (cx - 3, 12)]
    rwing = [(cx + 3, 8), (cx + 9, 18), (cx + 6, 18), (cx + 3, 11)]
    d.polygon(lwing, fill=(102, 76, 48, 255))
    d.polygon(rwing, fill=(94, 70, 44, 255))

    # Rust/wear patch on the larger wing
    d.rectangle([int(cx) - 9, 13, int(cx) - 7, 17], fill=(85, 56, 32, 255))

    # Engine glow (warm orange – not military)
    d.rectangle([int(cx) - 2, 18, int(cx) + 2, 22], fill=(255, 152, 58, 178))

    img.save(os.path.join(OUT, "pirate_corvette.png"))


# ─── PIRATE MISSILE BOAT (40×40) – improvised Pirate missile platform ─────────
# Design note: A converted hauler with crude missile racks bolted to the
# sides — rectangular boxes with no streamlining. Boxy brown hull, tan nose,
# visible tube openings. Warm orange engine glow. The whole ship reads as
# "dangerous but cheap".


def draw_pirate_missile_boat():
    S = 40
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Side missile racks (behind hull so hull overlaps them slightly)
    d.rectangle([int(cx) - 16, 12, int(cx) - 9, 30], fill=(98, 74, 46, 255))
    d.rectangle([int(cx) + 9, 12, int(cx) + 16, 30], fill=(98, 74, 46, 255))

    # Missile tube openings (three per side)
    for y in [14, 19, 24]:
        d.rectangle([int(cx) - 15, y, int(cx) - 10, y + 3], fill=(68, 50, 30, 255))
        d.rectangle([int(cx) + 10, y, int(cx) + 15, y + 3], fill=(68, 50, 30, 255))

    # Boxy main hull
    hull = [(cx, 3), (cx + 8, 14), (cx + 8, 30), (cx - 8, 30), (cx - 8, 14)]
    d.polygon(hull, fill=(125, 96, 60, 255))

    # Tan nose
    nose = [(cx, 3), (cx + 6, 13), (cx - 6, 13)]
    d.polygon(nose, fill=(178, 148, 102, 255))

    # Cockpit window
    d.ellipse([int(cx) - 3, 5, int(cx) + 3, 11], fill=(80, 140, 172, 180))

    # Engine
    d.rectangle([int(cx) - 5, 30, int(cx) + 5, 34], fill=(88, 64, 38, 255))
    d.ellipse([int(cx) - 5, 32, int(cx) + 5, 38], fill=(255, 155, 62, 182))

    img.save(os.path.join(OUT, "pirate_missile_boat.png"))


# ─── IR MISSILE (14×14) – guided heat-seeking missile ────────────────────────
# Design note: narrow dart shape with warm orange body matching the weapon
# colour.  Tiny exhaust plume at the rear.  Must be smaller than the smallest
# ship (corvette 20×20).


def draw_ir_missile():
    S = 14
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Slim pointed body – orange/red (narrow profile)
    body = [(cx, 1), (cx + 1.5, 5), (cx + 1.25, 11), (cx - 1.25, 11), (cx - 1.5, 5)]
    d.polygon(body, fill=(220, 120, 30, 255))

    # Nose tip – bright yellow-white
    nose = [(cx, 1), (cx + 0.75, 4), (cx - 0.75, 4)]
    d.polygon(nose, fill=(255, 220, 120, 255))

    # Small tail fins
    lfin = [(cx - 1.25, 9), (cx - 3.5, 12), (cx - 1, 12)]
    rfin = [(cx + 1.25, 9), (cx + 3.5, 12), (cx + 1, 12)]
    d.polygon(lfin, fill=(180, 80, 15, 255))
    d.polygon(rfin, fill=(180, 80, 15, 255))

    # Engine plume – small warm glow
    d.rectangle([int(cx) - 1, 11, int(cx) + 1, 13], fill=(255, 180, 60, 200))

    img.save(os.path.join(OUT, "ir_missile.png"))


# ─── JAVELIN (12×12) – fast long-range guided missile ────────────────────────
# Design note: slimmer and shorter than the ir_missile, with a cool blue body
# matching the weapon colour.  Sleek dart profile for a faster projectile.


def draw_javelin():
    S = 12
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2

    # Very slim pointed body – cool blue
    body = [(cx, 1), (cx + 0.9, 4), (cx + 0.75, 9), (cx - 0.75, 9), (cx - 0.9, 4)]
    d.polygon(body, fill=(80, 170, 230, 255))

    # Nose tip – bright white-blue
    nose = [(cx, 1), (cx + 0.5, 3), (cx - 0.5, 3)]
    d.polygon(nose, fill=(220, 240, 255, 255))

    # Small tail fins (narrow)
    lfin = [(cx - 0.75, 8), (cx - 2.25, 10), (cx - 0.5, 10)]
    rfin = [(cx + 0.75, 8), (cx + 2.25, 10), (cx + 0.5, 10)]
    d.polygon(lfin, fill=(40, 110, 180, 255))
    d.polygon(rfin, fill=(40, 110, 180, 255))

    # Engine plume – cool blue glow
    d.rectangle([int(cx), 9, int(cx) + 1, 11], fill=(150, 210, 255, 200))

    img.save(os.path.join(OUT, "javelin.png"))


# ─── SPACE MINE (16×16) – slow guided proximity mine ────────────────────────
# Design note: round spiky shape suggesting danger.  Dark red body with
# brighter red spikes radiating outward.  Must be smaller than the smallest
# ship (corvette 20×20).


def draw_space_mine():
    S = 16
    img = make_img(S)
    d = ImageDraw.Draw(img)
    cx = S / 2
    cy = S / 2

    import math

    # Central sphere – dark red/maroon
    r_core = 4
    d.ellipse(
        [int(cx) - r_core, int(cy) - r_core, int(cx) + r_core, int(cy) + r_core],
        fill=(140, 35, 35, 255),
    )

    # Spikes radiating outward (8 spikes)
    num_spikes = 8
    for i in range(num_spikes):
        angle = (i / num_spikes) * 2 * math.pi
        # Spike base at core edge, tip extends outward
        bx1 = cx + math.cos(angle - 0.3) * 3
        by1 = cy + math.sin(angle - 0.3) * 3
        bx2 = cx + math.cos(angle + 0.3) * 3
        by2 = cy + math.sin(angle + 0.3) * 3
        tx = cx + math.cos(angle) * 7
        ty = cy + math.sin(angle) * 7
        d.polygon([(bx1, by1), (tx, ty), (bx2, by2)], fill=(200, 55, 55, 255))

    # Bright centre dot – menacing glow
    d.ellipse(
        [int(cx) - 2, int(cy) - 2, int(cx) + 2, int(cy) + 2],
        fill=(255, 100, 80, 230),
    )

    img.save(os.path.join(OUT, "space_mine.png"))


if __name__ == "__main__":
    draw_shuttle()
    draw_fighter()
    draw_asteroid_miner()
    draw_hauler()
    draw_corvette()
    draw_frigate()
    draw_courier()
    draw_freighter()
    draw_bulk_carrier()
    draw_fed_patrol()
    draw_fed_destroyer()
    draw_fed_missile_cruiser()
    draw_rebel_fighter()
    draw_rebel_gunboat()
    draw_rebel_frigate()
    draw_pirate_corvette()
    draw_pirate_missile_boat()
    draw_ir_missile()
    draw_javelin()
    draw_space_mine()
    print("Sprites written to", OUT)
