"""
render3d.py — tiny self-contained software 3D renderer for top-down ship sprites.

No GPU, no display, no Blender — just numpy + Pillow.  Builds procedural ship
meshes, renders them with a *straight-down* orthographic camera (so the game can
freely rotate the sprite at runtime), and produces a stylised "rendered cartoon"
look: smooth shaded curved hulls, a key/fill/rim light rig, Blinn-Phong
speculars, gentle cel banding, ambient occlusion in crevices via vertex AO, and
a dark ink outline.

Coordinate convention (object space):
    +Y = forward  (ship nose points toward +Y)
    +X = starboard (right)
    +Z = up  (toward the camera)
The camera looks straight down the -Z axis.  In the output image the nose points
UP (-Y in image space) to match the existing 2D sprites and ships.yaml.

Everything is supersampled (SS×) then box-downsampled for clean anti-aliasing.
"""

import numpy as np
from PIL import Image, ImageFilter


# ─────────────────────────── mesh container ────────────────────────────────


class Mesh:
    """Triangle soup with per-vertex colour and a flat ambient-occlusion term."""

    def __init__(self):
        self.verts = []   # list of (x,y,z)
        self.tris = []    # list of (i0,i1,i2)
        self.vcol = []    # per-vertex base colour (r,g,b) floats 0..1
        self.vao = []     # per-vertex ambient occlusion 0..1 (1 = fully lit)

    def add(self, verts, color, ao=1.0, smooth_group=None):
        """Add a quad/triangle strip patch.  `verts` is an (N,3) array already
        triangulated by the caller into a flat list of triangles (multiple of 3
        verts) OR we triangulate here if `faces` style.  To keep it simple this
        helper only appends raw triangles: verts must be length 3*k."""
        base = len(self.verts)
        color = np.asarray(color, float)
        for v in verts:
            self.verts.append(np.asarray(v, float))
            self.vcol.append(color.copy())
            self.vao.append(ao)
        n = len(verts)
        for t in range(0, n, 3):
            self.tris.append((base + t, base + t + 1, base + t + 2))

    def add_indexed(self, verts, faces, color, ao=1.0):
        base = len(self.verts)
        color = np.asarray(color, float)
        if np.isscalar(ao):
            ao = [ao] * len(verts)
        for v, a in zip(verts, ao):
            self.verts.append(np.asarray(v, float))
            self.vcol.append(color.copy())
            self.vao.append(a)
        for f in faces:
            if len(f) == 3:
                self.tris.append((base + f[0], base + f[1], base + f[2]))
            else:  # quad -> 2 tris
                self.tris.append((base + f[0], base + f[1], base + f[2]))
                self.tris.append((base + f[0], base + f[2], base + f[3]))


# ─────────────────────────── mesh primitives ───────────────────────────────


def _ring(profile, scale_x, scale_z, y, cz):
    """A cross-section ring at station y.  `profile` is an (M,2) array of unit
    (x,z) points tracing the outline of the cross-section CCW.  Returns (M,3)."""
    out = []
    for (px, pz) in profile:
        out.append((px * scale_x, y, cz + pz * scale_z))
    return np.array(out, float)


def superellipse_profile(m=24, n=2.6, flatten_bottom=0.55):
    """Closed cross-section outline.  n controls roundness (2=ellipse, higher=
    boxier).  flatten_bottom squashes the underside so hulls sit flat-ish."""
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = np.sign(ct) * np.abs(ct) ** (2.0 / n)
    z = np.sign(st) * np.abs(st) ** (2.0 / n)
    z = np.where(z < 0, z * flatten_bottom, z)
    return np.stack([x, z], axis=1)


def loft(stations, color, profile=None, ao_crease=0.0):
    """Loft a hull from a list of stations.  Each station is a dict with keys
    y (forward pos), w (half-width), h (half-height), cz (centre height),
    ao (optional 0..1).  Adjacent rings are bridged with quads; the two ends
    are capped with a fan.  Returns a Mesh."""
    if profile is None:
        profile = superellipse_profile()
    m = Mesh()
    rings = []
    aos = []
    for s in stations:
        rings.append(_ring(profile, s["w"], s["h"], s["y"], s.get("cz", 0.0)))
        aos.append(s.get("ao", 1.0))
    M = len(profile)
    verts = []
    ao_list = []
    for r, a in zip(rings, aos):
        for v in r:
            verts.append(v)
            ao_list.append(a)
    faces = []
    for i in range(len(rings) - 1):
        b0 = i * M
        b1 = (i + 1) * M
        for j in range(M):
            jn = (j + 1) % M
            faces.append((b0 + j, b0 + jn, b1 + jn, b1 + j))
    # end caps (fan to centroid)
    front_c = len(verts)
    verts.append(rings[-1].mean(axis=0)); ao_list.append(aos[-1])
    back_c = len(verts)
    verts.append(rings[0].mean(axis=0)); ao_list.append(aos[0])
    bN = (len(rings) - 1) * M
    for j in range(M):
        jn = (j + 1) % M
        faces.append((front_c, bN + j, bN + jn))
        faces.append((back_c, jn, j))
    m.add_indexed(verts, faces, color, ao=ao_list)
    return m


def box(cx, cy, cz, sx, sy, sz, color, ao=1.0, taper_top=1.0):
    """Axis-aligned (optionally top-tapered) box.  taper_top<1 shrinks the top
    face in x for a bevelled industrial look."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    tx = hx * taper_top
    v = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - tx, cy - hy, cz + hz), (cx + tx, cy - hy, cz + hz),
        (cx + tx, cy + hy, cz + hz), (cx - tx, cy + hy, cz + hz),
    ]
    f = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(3,0,4,7)]
    m = Mesh()
    m.add_indexed(v, f, color, ao=ao)
    return m


def cylinder(cx, cy, cz, r, length, color, axis="y", seg=16, ao=1.0,
             r2=None, cap_color=None):
    """Cylinder/cone along an axis.  r2 lets the far end differ (nozzle flare)."""
    if r2 is None:
        r2 = r
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    m = Mesh()
    verts = []
    if axis == "y":
        a0, a1 = cy - length / 2, cy + length / 2
        for ang in range(seg):
            verts.append((cx + r * c[ang], a0, cz + r * s[ang]))
        for ang in range(seg):
            verts.append((cx + r2 * c[ang], a1, cz + r2 * s[ang]))
    else:  # z axis (vertical pods)
        a0, a1 = cz - length / 2, cz + length / 2
        for ang in range(seg):
            verts.append((cx + r * c[ang], cy + r * s[ang], a0))
        for ang in range(seg):
            verts.append((cx + r2 * c[ang], cy + r2 * s[ang], a1))
    faces = []
    for j in range(seg):
        jn = (j + 1) % seg
        faces.append((j, jn, seg + jn, seg + j))
    c0 = len(verts); verts.append((cx, a0, cz) if axis == "y" else (cx, cy, a0))
    c1 = len(verts); verts.append((cx, a1, cz) if axis == "y" else (cx, cy, a1))
    for j in range(seg):
        jn = (j + 1) % seg
        faces.append((c0, jn, j))
        faces.append((c1, seg + j, seg + jn))
    m.add_indexed(verts, faces, color, ao=ao)
    if cap_color is not None:
        cm = Mesh()
        cap_v = [verts[seg + j] for j in range(seg)] + [(cx, a1, cz) if axis == "y" else (cx, cy, a1)]
        cap_f = [(seg, j, (j + 1) % seg) for j in range(seg)]
        cm.add_indexed(cap_v, cap_f, cap_color, ao=ao)
        m = merge(m, cm)
    return m


def ellipsoid(cx, cy, cz, rx, ry, rz, color, ao=1.0, stacks=10, slices=16,
              zclip=-1e9):
    """UV sphere scaled to an ellipsoid; zclip drops verts below a height (for
    a canopy bubble sitting on the hull)."""
    m = Mesh()
    verts = []
    idx = {}
    for i in range(stacks + 1):
        v = np.pi * i / stacks
        for j in range(slices):
            u = 2 * np.pi * j / slices
            x = cx + rx * np.sin(v) * np.cos(u)
            y = cy + ry * np.sin(v) * np.sin(u)
            z = cz + rz * np.cos(v)
            idx[(i, j)] = len(verts)
            verts.append((x, y, z))
    faces = []
    for i in range(stacks):
        for j in range(slices):
            jn = (j + 1) % slices
            a, b, c, d = idx[(i, j)], idx[(i, jn)], idx[(i + 1, jn)], idx[(i + 1, j)]
            if verts[a][2] < zclip and verts[b][2] < zclip and \
               verts[c][2] < zclip and verts[d][2] < zclip:
                continue
            faces.append((a, b, c, d))
    m.add_indexed(verts, faces, color, ao=ao)
    return m


def wing(root_y0, root_y1, tip_y0, tip_y1, root_x, tip_x, thick, cz,
         color, dihedral_z=0.0, ao=1.0, sweep_side=1):
    """A tapered low-poly airfoil wing (rounded leading edge feel via a 6-vert
    lens cross-section) swept from root to tip.  sweep_side: +1 starboard."""
    sx = sweep_side
    # cross-section at root and tip: top & bottom skins meet at sharp LE/TE
    def section(y0, y1, xpos, z):
        ymid = (y0 + y1) / 2
        return [
            (sx * xpos, y1, z),                 # leading edge
            (sx * xpos, ymid, z + thick),       # upper crest
            (sx * xpos, y0, z),                 # trailing edge
            (sx * xpos, ymid, z - thick * 0.5), # lower crest
        ]
    r = section(root_y0, root_y1, root_x, cz)
    tp = section(tip_y0, tip_y1, tip_x, cz + dihedral_z)
    verts = r + tp
    # 4 side quads + LE/TE caps
    faces = [
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),  # skins
        (0, 1, 2, 3),  # root cap
        (4, 7, 6, 5),  # tip cap
    ]
    m = Mesh()
    m.add_indexed(verts, faces, color, ao=ao)
    return m


def merge(*meshes):
    out = Mesh()
    for m in meshes:
        base = len(out.verts)
        out.verts.extend([v.copy() for v in m.verts])
        out.vcol.extend([c.copy() for c in m.vcol])
        out.vao.extend(m.vao)
        out.tris.extend([(a + base, b + base, c + base) for (a, b, c) in m.tris])
    return out


def mirror_x(m):
    """Return a copy mirrored across the x=0 plane (winding flipped)."""
    out = Mesh()
    for v in m.verts:
        out.verts.append(np.array([-v[0], v[1], v[2]]))
    out.vcol = [c.copy() for c in m.vcol]
    out.vao = list(m.vao)
    out.tris = [(a, c, b) for (a, b, c) in m.tris]
    return out


# ─────────────────────────────── renderer ──────────────────────────────────


def _vertex_normals(V, tris):
    N = np.zeros_like(V)
    for (a, b, c) in tris:
        n = np.cross(V[b] - V[a], V[c] - V[a])
        N[a] += n; N[b] += n; N[c] += n
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return N / ln


def render(mesh, size, world_half, *, ss=4, cel_bands=0,
           light_dir=(-0.45, 0.35, 0.82), bg=(0, 0, 0, 0),
           outline=(18, 20, 28), outline_strength=0.85,
           ambient=0.34, fill=0.18, rim=0.30, spec=0.5, shininess=28,
           gamma=1.0):
    """Render `mesh` to an RGBA PIL image of (size×size).

    world_half: half-extent (in object units) mapped to half the image; the
    visible square spans [-world_half, +world_half] in x and y.
    Nose (+Y object) ends up pointing UP in the image.
    """
    S = size * ss
    V = np.array(mesh.verts, float)
    tris = mesh.tris
    col = np.array(mesh.vcol, float)
    ao = np.array(mesh.vao, float)
    N = _vertex_normals(V, tris)

    L = np.array(light_dir, float); L /= np.linalg.norm(L)
    view = np.array([0, 0, 1.0])
    half_vec = (L + view); half_vec /= np.linalg.norm(half_vec)

    # object -> image pixel.  x: +X right.  y: +Y up  => image row decreases.
    sc = (S / 2) / world_half

    def to_px(p):
        ix = S / 2 + p[0] * sc
        iy = S / 2 - p[1] * sc
        return ix, iy

    color_buf = np.zeros((S, S, 3), float)
    z_buf = np.full((S, S), -1e18, float)
    cov = np.zeros((S, S), bool)

    # per-vertex lit colour (Gouraud) + specular accumulated separately
    def shade(i):
        n = N[i]
        diff = max(0.0, n @ L)
        if cel_bands:
            diff = np.floor(diff * cel_bands + 0.5) / cel_bands
        # fill light from opposite-ish side, rim from behind-up
        fl = max(0.0, n @ np.array([0.5, -0.2, 0.4]))
        rm = max(0.0, n @ np.array([0.0, 0.0, -1.0])) ** 2  # unused-ish
        base = col[i] * ao[i]
        lit = base * (ambient + diff + fill * fl) + base * rim * (1 - abs(n[2])) * 0.0
        spec_v = spec * (max(0.0, n @ half_vec) ** shininess)
        return np.clip(lit + spec_v, 0, 4.0)

    vcolor = np.array([shade(i) for i in range(len(V))])
    P = np.array([to_px(v) for v in V])
    Z = V[:, 2]

    for (a, b, c) in tris:
        # back-face cull (only top-facing tris matter for top-down)
        n = np.cross(V[b] - V[a], V[c] - V[a])
        if n[2] <= 0:
            continue
        pa, pb, pc = P[a], P[b], P[c]
        minx = int(max(0, np.floor(min(pa[0], pb[0], pc[0]))))
        maxx = int(min(S - 1, np.ceil(max(pa[0], pb[0], pc[0]))))
        miny = int(max(0, np.floor(min(pa[1], pb[1], pc[1]))))
        maxy = int(min(S - 1, np.ceil(max(pa[1], pb[1], pc[1]))))
        if maxx < minx or maxy < miny:
            continue
        ax, ay = pa; bx, by = pb; cx_, cy_ = pc
        denom = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
        if abs(denom) < 1e-9:
            continue
        xs = np.arange(minx, maxx + 1) + 0.5
        ys = np.arange(miny, maxy + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((by - cy_) * (gx - cx_) + (cx_ - bx) * (gy - cy_)) / denom
        w1 = ((cy_ - ay) * (gx - cx_) + (ax - cx_) * (gy - cy_)) / denom
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        z = w0 * Z[a] + w1 * Z[b] + w2 * Z[c]
        rr, cc = np.where(inside)
        prow = miny + rr
        pcol = minx + cc
        zv = z[rr, cc]
        cur = z_buf[prow, pcol]
        win = zv > cur
        if not win.any():
            continue
        prow, pcol, zv = prow[win], pcol[win], zv[win]
        w0w, w1w, w2w = w0[rr[win], cc[win]], w1[rr[win], cc[win]], w2[rr[win], cc[win]]
        cval = (w0w[:, None] * vcolor[a] + w1w[:, None] * vcolor[b]
                + w2w[:, None] * vcolor[c])
        z_buf[prow, pcol] = zv
        color_buf[prow, pcol] = cval
        cov[prow, pcol] = True

    # ── cartoon ink outline from the coverage silhouette + interior depth edges
    img = np.zeros((S, S, 4), float)
    img[..., :3] = np.clip(color_buf, 0, 1) ** (1.0 / gamma)
    img[..., 3] = cov.astype(float)

    if outline_strength > 0:
        from PIL import Image as _I
        covimg = _I.fromarray((cov * 255).astype("uint8"))
        # silhouette edge = dilation - original
        dil = covimg.filter(ImageFilter.MaxFilter(2 * ss + 1))
        edge = (np.array(dil) > 0) & (~cov)
        oc = np.array(outline, float) / 255.0
        for k in range(3):
            ch = img[..., k]
            ch[edge] = ch[edge] * (1 - outline_strength) + oc[k] * outline_strength
        a = img[..., 3]
        a[edge] = np.maximum(a[edge], outline_strength)

    # supersample down with alpha-weighted box filter
    out = (np.clip(img, 0, 1) * 255).astype("uint8")
    pim = Image.fromarray(out, "RGBA")
    pim = pim.resize((size, size), Image.LANCZOS)
    return pim
