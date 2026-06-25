"""
ships3d.py — procedural 3D ship meshes for the top-down sprite renderer.

One representative ship per class is built here as a prototype:
  * fighter        — sleek (Spitfire / 50s-streamlined "Phantom 1" feel)
  * hauler         — industrial merchant (boxy spine + cargo container stacks)
  * asteroid_miner — industrial mining rig (heavy hull + forward drill arms)

Object space: +Y forward, +X starboard, +Z up.  Hulls are built on the
starboard side / centre and mirrored so they stay perfectly symmetric.

Run via generate_3d_ships.py.
"""

import numpy as np

from render3d import (Mesh, box, cylinder, ellipsoid, loft, merge, mirror_x,
                      render, superellipse_profile)


# ── shared colours (0..1) ───────────────────────────────────────────────────
def c(*rgb):
    return np.array(rgb, float) / 255.0


GLASS = c(150, 215, 250)
ENGINE_BLUE = c(150, 210, 255)
ENGINE_WARM = c(255, 180, 110)


def elliptical_wing(span, root_chord, tip_chord, root_y, thick, cz,
                    color, sweep=0.18, dihedral=0.0, side=1, sections=7,
                    ao=0.92):
    """Spitfire-style elliptical-planform wing, lofted across the span so both
    the leading and trailing edges curve.  side=+1 starboard."""
    m = Mesh()
    xs = np.linspace(0.06 * span, span, sections)
    rings = []
    for x in xs:
        frac = x / span
        # elliptical chord distribution
        chord = root_chord * np.sqrt(max(0.0, 1 - frac ** 2))
        chord = max(chord, tip_chord)
        cy = root_y - sweep * x          # leading edge sweeps back
        z = cz + dihedral * frac
        th = thick * (1 - 0.6 * frac)
        # lens cross-section: LE, top, TE, bottom
        rings.append([
            (side * x, cy + chord * 0.5, z),
            (side * x, cy + chord * 0.05, z + th),
            (side * x, cy - chord * 0.5, z),
            (side * x, cy + chord * 0.05, z - th * 0.5),
        ])
    verts = [v for r in rings for v in r]
    faces = []
    for i in range(len(rings) - 1):
        b0, b1 = i * 4, (i + 1) * 4
        for j in range(4):
            jn = (j + 1) % 4
            faces.append((b0 + j, b0 + jn, b1 + jn, b1 + j))
    # root cap
    faces.append((0, 1, 2, 3))
    m.add_indexed(verts, faces, color, ao=ao)
    return m


# ════════════════════════════ FIGHTER ══════════════════════════════════════
def build_fighter():
    hull_c = c(150, 170, 205)
    hull_dark = c(96, 116, 150)
    accent = c(210, 70, 70)

    prof = superellipse_profile(m=28, n=2.4, flatten_bottom=0.7)

    # Slender, smoothly-tapered fuselage.  Stations run nose(+Y) -> tail(-Y).
    stations = [
        dict(y= 1.00, w=0.02, h=0.02, cz=0.00, ao=1.0),   # nose tip
        dict(y= 0.78, w=0.10, h=0.11, cz=0.01),
        dict(y= 0.55, w=0.17, h=0.17, cz=0.02),
        dict(y= 0.30, w=0.21, h=0.20, cz=0.03),           # cockpit area (widest)
        dict(y= 0.00, w=0.20, h=0.18, cz=0.02),
        dict(y=-0.35, w=0.16, h=0.15, cz=0.01),
        dict(y=-0.70, w=0.12, h=0.13, cz=0.00),
        dict(y=-0.95, w=0.10, h=0.11, cz=0.00, ao=0.85),  # tail / engine mount
    ]
    fuse = loft(stations, hull_c, profile=prof)

    # Bubble canopy — tinted glass ellipsoid sitting on the cockpit hump
    canopy = ellipsoid(0, 0.34, 0.16, rx=0.10, ry=0.24, rz=0.13, color=GLASS,
                       ao=1.0, stacks=12, slices=18, zclip=0.13)

    # Spitfire elliptical wings (low/mid mount, gentle dihedral)
    rw = elliptical_wing(span=0.62, root_chord=0.50, tip_chord=0.05,
                         root_y=0.02, thick=0.045, cz=0.02, color=hull_c,
                         sweep=0.22, dihedral=0.06, side=1)
    lw = mirror_x(rw)

    # Slim tailplane (horizontal stabiliser) near the tail
    rtail = elliptical_wing(span=0.22, root_chord=0.20, tip_chord=0.03,
                            root_y=-0.78, thick=0.03, cz=0.01, color=hull_dark,
                            sweep=0.25, side=1, sections=5)
    ltail = mirror_x(rtail)

    # Single dorsal fin (vertical stabiliser) — a thin upright blade
    fin = Mesh()
    fv = [(0, -0.62, 0.04), (0, -0.95, 0.04), (0, -0.85, 0.30), (0, -0.66, 0.20)]
    fin.add_indexed(fv, [(0, 1, 2, 3)], hull_dark, ao=0.95)
    fin2 = Mesh()  # give the fin a tiny thickness so it shades
    fv2 = [(0.012, -0.62, 0.04), (0.012, -0.95, 0.04), (0.012, -0.85, 0.30), (0.012, -0.66, 0.20)]
    fin2.add_indexed(fv2, [(0, 3, 2, 1)], hull_dark, ao=0.95)

    # Nose accent stripe (thin band) — a flattened ring slightly proud of hull
    stripe = loft([
        dict(y=0.50, w=0.175, h=0.175, cz=0.02),
        dict(y=0.44, w=0.185, h=0.185, cz=0.02),
    ], accent, profile=prof)

    # Twin engine nozzles + glow at the tail
    noz_r = 0.062
    nozzles = Mesh()
    for sx in (-0.06, 0.06):
        n = cylinder(sx, -0.99, 0.0, r=noz_r, length=0.14, color=hull_dark,
                     axis="y", seg=14, ao=0.8, r2=noz_r * 1.15)
        glow = cylinder(sx, -1.07, 0.0, r=noz_r * 0.7, length=0.02,
                        color=ENGINE_BLUE * 2.2, axis="y", seg=12, ao=1.0)
        nozzles = merge(nozzles, n, glow)

    return merge(fuse, stripe, rw, lw, rtail, ltail, fin, fin2, canopy, nozzles)


# ════════════════════════════ HAULER ═══════════════════════════════════════
def build_hauler():
    spine = c(150, 150, 158)
    spine_d = c(110, 110, 120)
    cargo_a = c(168, 150, 110)   # warm tan containers
    cargo_b = c(120, 130, 140)
    cargo_c = c(150, 120, 95)
    cockpit_col = c(170, 175, 182)

    parts = []

    # Central structural spine (long boxy keel)
    parts.append(box(0, 0.0, 0.0, sx=0.30, sy=1.9, sz=0.26, color=spine,
                     ao=0.95, taper_top=0.85))
    # raised dorsal rail running the length (industrial framework)
    parts.append(box(0, -0.05, 0.18, sx=0.14, sy=1.7, sz=0.10, color=spine_d, ao=0.8))

    # Small forward cockpit module (deliberately tiny vs cargo mass)
    parts.append(box(0, 0.92, 0.10, sx=0.26, sy=0.30, sz=0.22, color=cockpit_col,
                     ao=1.0, taper_top=0.6))
    parts.append(ellipsoid(0, 1.02, 0.20, rx=0.09, ry=0.10, rz=0.07,
                           color=GLASS, ao=1.0, stacks=8, slices=12))

    # Three rows of paired cargo containers flanking the spine.
    container_cols = [cargo_a, cargo_b, cargo_c]
    for row, yc in enumerate([0.46, 0.0, -0.46]):
        col = container_cols[row]
        for sx in (-0.34, 0.34):
            cont = box(sx, yc, 0.05, sx=0.34, sy=0.40, sz=0.34, color=col,
                       ao=0.9, taper_top=0.96)
            parts.append(cont)
            # latch ridge across each container (slightly proud, darker)
            parts.append(box(sx, yc, 0.23, sx=0.30, sy=0.36, sz=0.04,
                             color=col * 0.7, ao=0.7))
            # corner posts (industrial greeble)
            for px in (-0.15, 0.15):
                parts.append(box(sx + px, yc, 0.12, sx=0.05, sy=0.42, sz=0.30,
                                 color=spine_d, ao=0.7))

    # Engine block at the stern + four nozzles with warm glow
    parts.append(box(0, -1.02, 0.0, sx=0.5, sy=0.22, sz=0.30, color=spine_d, ao=0.85))
    for sx in (-0.26, -0.09, 0.09, 0.26):
        parts.append(cylinder(sx, -1.16, 0.0, r=0.075, length=0.16,
                              color=c(80, 80, 92), axis="y", seg=14, ao=0.75,
                              r2=0.09))
        parts.append(cylinder(sx, -1.25, 0.0, r=0.05, length=0.02,
                              color=ENGINE_WARM * 2.0, axis="y", seg=12, ao=1.0))

    return merge(*parts)


# ════════════════════════ ASTEROID MINER ═══════════════════════════════════
def build_asteroid_miner():
    hull = c(150, 120, 82)
    hull_d = c(110, 88, 58)
    steel = c(120, 120, 128)
    drill = c(200, 165, 90)

    prof = superellipse_profile(m=24, n=3.4, flatten_bottom=0.6)  # boxy, tough
    parts = []

    # Heavy squat central hull (short and wide)
    stations = [
        dict(y= 0.55, w=0.18, h=0.16, cz=0.02, ao=0.95),
        dict(y= 0.30, w=0.34, h=0.24, cz=0.03),
        dict(y= 0.00, w=0.40, h=0.27, cz=0.03),   # broad shoulders
        dict(y=-0.35, w=0.36, h=0.25, cz=0.02),
        dict(y=-0.65, w=0.28, h=0.20, cz=0.01, ao=0.85),
    ]
    parts.append(loft(stations, hull, profile=prof))

    # Armour seam ridges across the back (industrial plating)
    for yc in (0.15, -0.15, -0.42):
        parts.append(box(0, yc, 0.26, sx=0.66, sy=0.05, sz=0.04,
                         color=hull_d, ao=0.7))

    # Cockpit blister forward
    parts.append(ellipsoid(0, 0.42, 0.18, rx=0.12, ry=0.16, rz=0.10,
                           color=GLASS, ao=1.0, stacks=10, slices=14, zclip=0.18))

    # Forward drill arms (mandibles): heavy struts projecting past the nose,
    # each tipped with a rotating drill cone.
    drill_parts = []
    arm = box(0.40, 0.55, 0.04, sx=0.14, sy=0.7, sz=0.20, color=steel,
              ao=0.85, taper_top=0.8)
    drill_parts.append(arm)
    # drill cone (cylinder tapering to a point)
    drill_parts.append(cylinder(0.40, 0.95, 0.04, r=0.11, length=0.26,
                                color=drill, axis="y", seg=14, ao=0.9, r2=0.005))
    # mounting collar
    drill_parts.append(cylinder(0.40, 0.80, 0.04, r=0.13, length=0.06,
                                color=hull_d, axis="y", seg=14, ao=0.8))
    right_arm = merge(*drill_parts)
    parts.append(right_arm)
    parts.append(mirror_x(right_arm))

    # Side ore pods (cargo) low on the flanks
    for sx in (-0.46, 0.46):
        parts.append(box(sx, -0.18, 0.0, sx=0.16, sy=0.5, sz=0.24,
                         color=hull_d, ao=0.8, taper_top=0.9))

    # Twin engine pods + warm glow at the stern
    for sx in (-0.16, 0.16):
        parts.append(cylinder(sx, -0.72, 0.0, r=0.10, length=0.18,
                              color=steel * 0.8, axis="y", seg=14, ao=0.75, r2=0.12))
        parts.append(cylinder(sx, -0.82, 0.0, r=0.07, length=0.02,
                              color=ENGINE_WARM * 2.0, axis="y", seg=12, ao=1.0))

    return merge(*parts)
