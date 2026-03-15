"""
Ship sprite generator for avian_space.

Run with:  python3 generate_sprites.py
Requires:  Pillow  (pip install Pillow)

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

OUT = os.path.dirname(__file__)  # write into the same directory as this script


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
    print("Sprites written to", OUT)
