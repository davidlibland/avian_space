//! Man-made terrain generation for station/interior biomes.
//!
//! Where organic biomes sample fBm noise ([`crate::fbm`]), interiors build a
//! *designed* floor plan — a corridor/room skeleton — and derive the terrain
//! field from the Chebyshev (L∞) distance to it. Because L∞ contours around
//! rectangles are rectangular, the mandatory ±1 tier rings (required by the
//! blob47 autotiler) come out as neat, deliberate-looking trim instead of
//! organic blobs: floor → plating margin → grate trim → wall → conduit → void.
//!
//! Features (all jittered per seed, so layouts are man-made but never grid-
//! repetitive): a central plaza under the landing pad, corridor spines (one
//! widened into a hall), rooms of varied size hung off the corridors (some
//! with chamfered corners, some floored in plating, some with inset panels),
//! town squares — large octagonal plazas with a solid central pedestal — and
//! a grated service walkway down one corridor.
//!
//! The output honours the same contract as [`crate::fbm::generate_terrain_map`]:
//! indices in `0..n_terrain_types`, borders pinned to the highest tier, and no
//! two 8-connected neighbours differing by more than one (enforced by a final
//! [`crate::fbm::clamp_terrain_indices`] pass with all designed tiles pinned).

use rand::{Rng, SeedableRng};

// Tier meanings for interior biomes (see world_manifest.ron):
const T_FLOOR: u32 = 0;
const T_PLATING: u32 = 1;
const T_GRATE: u32 = 2;
const T_WALL: u32 = 3;

/// Keep all walkable features at least this far from the map border so the
/// wall/conduit/void bands have room to ramp down to the border tier.
const MARGIN: i32 = 6;

#[derive(Clone, Copy)]
struct Rect {
    x0: i32,
    y0: i32,
    x1: i32, // exclusive
    y1: i32, // exclusive
}

impl Rect {
    fn clamped(cx: i32, cy: i32, w: i32, h: i32, map_w: i32, map_h: i32) -> Self {
        let x0 = (cx - w / 2).clamp(MARGIN, map_w - MARGIN - 1);
        let y0 = (cy - h / 2).clamp(MARGIN, map_h - MARGIN - 1);
        Rect {
            x0,
            y0,
            x1: (x0 + w).min(map_w - MARGIN),
            y1: (y0 + h).min(map_h - MARGIN),
        }
    }
    fn w(&self) -> i32 {
        self.x1 - self.x0
    }
    fn h(&self) -> i32 {
        self.y1 - self.y0
    }
}

/// A corridor spine: a full-width/height stroke through the map.
#[derive(Clone, Copy)]
struct Spine {
    horizontal: bool,
    /// Row (horizontal) or column (vertical).
    pos: i32,
    /// Half-width of the open stroke (1 = 3 tiles wide, 3 = 7-wide hall).
    half: i32,
}

/// Generate a station-interior terrain map. Same contract as
/// [`crate::fbm::generate_terrain_map`].
pub fn generate_station_map(width: u32, height: u32, n_terrain_types: u32, seed: u64) -> Vec<u32> {
    assert!(
        n_terrain_types >= 4,
        "station generator needs floor/plating/grate/wall tiers"
    );
    let (w, h) = (width as i32, height as i32);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_mul(0x9e37_79b9).wrapping_add(7));
    let top = n_terrain_types - 1;

    // `open[idx]`: walkable designed floor. `over[idx]`: walkable material
    // override (plating rooms, panels, walkways). `ped[idx]`: pedestal tier
    // override (may be solid).
    let n = (w * h) as usize;
    let mut open = vec![false; n];
    let mut over: Vec<Option<u32>> = vec![None; n];
    let mut ped: Vec<Option<u32>> = vec![None; n];
    let idx = |x: i32, y: i32| (y * w + x) as usize;

    let mut carve_rect = |open: &mut Vec<bool>, r: Rect, chamfer: i32| {
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                if chamfer > 0 {
                    // Octagonal corner cut: L1 distance to the nearest corner.
                    let dx = (x - r.x0).min(r.x1 - 1 - x);
                    let dy = (y - r.y0).min(r.y1 - 1 - y);
                    if dx + dy < chamfer {
                        continue;
                    }
                }
                open[idx(x, y)] = true;
            }
        }
    };

    // ── Central plaza (the landing pad spawns at map center) ─────────────
    let (cx, cy) = (w / 2, h / 2);
    let plaza = Rect::clamped(cx, cy, rng.gen_range(12..=16), rng.gen_range(12..=16), w, h);
    carve_rect(&mut open, plaza, 0);

    // ── Corridor spines ──────────────────────────────────────────────────
    // One horizontal + one vertical pass close to the plaza (guaranteeing it
    // connects), plus 1-2 more at jittered offsets. One spine becomes a hall.
    let mut spines: Vec<Spine> = vec![
        Spine {
            horizontal: true,
            pos: cy + rng.gen_range(-4..=4),
            half: 1,
        },
        Spine {
            horizontal: false,
            pos: cx + rng.gen_range(-4..=4),
            half: 1,
        },
    ];
    for _ in 0..rng.gen_range(1..=2) {
        let horizontal = rng.r#gen::<bool>();
        let extent = if horizontal { h } else { w };
        spines.push(Spine {
            horizontal,
            pos: rng.gen_range(MARGIN + 4..extent - MARGIN - 4),
            half: 1,
        });
    }
    // Widen one spine into a hall (7 tiles) with a plating centre band.
    let hall_idx = rng.gen_range(0..spines.len());
    spines[hall_idx].half = 3;

    for s in &spines {
        let r = if s.horizontal {
            Rect {
                x0: MARGIN,
                y0: s.pos - s.half,
                x1: w - MARGIN,
                y1: s.pos + s.half + 1,
            }
        } else {
            Rect {
                x0: s.pos - s.half,
                y0: MARGIN,
                x1: s.pos + s.half + 1,
                y1: h - MARGIN,
            }
        };
        carve_rect(&mut open, r, 0);
    }

    /// A jittered point on a random spine (for hanging rooms/squares off it).
    fn spine_point(rng: &mut impl Rng, spines: &[Spine], w: i32, h: i32) -> (i32, i32) {
        let s = spines[rng.gen_range(0..spines.len())];
        let along_extent = if s.horizontal { w } else { h };
        let along = rng.gen_range(MARGIN + 4..along_extent - MARGIN - 4);
        if s.horizontal {
            (along, s.pos)
        } else {
            (s.pos, along)
        }
    }

    // ── Town squares: big octagonal plazas with a central pedestal ───────
    let mut squares: Vec<Rect> = Vec::new();
    for _ in 0..rng.gen_range(1..=2) {
        let (sx, sy) = spine_point(&mut rng, &spines, w, h);
        let size = rng.gen_range(15..=20);
        let r = Rect::clamped(sx, sy, size, size, w, h);
        carve_rect(&mut open, r, r.w().min(r.h()) / 3);
        squares.push(r);
    }

    // ── Rooms hung off the corridors ─────────────────────────────────────
    let mut rooms: Vec<(Rect, u32)> = Vec::new(); // (rect, floor material)
    for _ in 0..rng.gen_range(8..=12) {
        let (sx, sy) = spine_point(&mut rng, &spines, w, h);
        let rw = rng.gen_range(5..=14);
        let rh = rng.gen_range(5..=12);
        let r = Rect::clamped(sx, sy, rw, rh, w, h);
        if r.w() < 4 || r.h() < 4 {
            continue;
        }
        let chamfer = if rng.gen_bool(0.35) {
            r.w().min(r.h()) / 4
        } else {
            0
        };
        carve_rect(&mut open, r, chamfer);
        // Vary the floor material: most rooms are floor, some full plating.
        let material = if rng.gen_bool(0.35) {
            T_PLATING
        } else {
            T_FLOOR
        };
        rooms.push((r, material));
    }

    // ── Walkable material variety (overrides apply only to open tiles) ───
    // Plating-floored rooms + inset panels one tier up from the room floor.
    for &(r, material) in &rooms {
        if material != T_FLOOR {
            for y in r.y0..r.y1 {
                for x in r.x0..r.x1 {
                    over[idx(x, y)] = Some(material);
                }
            }
        }
        if r.w() >= 8 && r.h() >= 7 && rng.gen_bool(0.7) {
            let panel = Rect {
                x0: r.x0 + 2,
                y0: r.y0 + 2,
                x1: r.x1 - 2,
                y1: r.y1 - 2,
            };
            for y in panel.y0..panel.y1 {
                for x in panel.x0..panel.x1 {
                    over[idx(x, y)] = Some(material + 1); // floor→plating / plating→grate
                }
            }
        }
    }
    // Squares: plating border ring inset one tile from the edge (a "paved"
    // promenade look), floor inside.
    for r in &squares {
        if rng.gen_bool(0.6) {
            for y in r.y0 + 1..r.y1 - 1 {
                for x in r.x0 + 1..r.x1 - 1 {
                    let on_ring = y == r.y0 + 1 || y == r.y1 - 2 || x == r.x0 + 1 || x == r.x1 - 2;
                    if on_ring {
                        over[idx(x, y)] = Some(T_PLATING);
                    }
                }
            }
        }
    }
    // Grated service walkway with plating shoulders down one non-hall spine.
    // The plating band extends one tile past the grate line at both ends so
    // the grate (tier 2) never sits directly against corridor floor (tier 0).
    if let Some(s) = spines.iter().find(|s| s.half == 1) {
        let extent = if s.horizontal { w } else { h };
        for along in MARGIN + 2..extent - MARGIN - 2 {
            let centre_ok = along > MARGIN + 2 && along < extent - MARGIN - 3;
            for off in -1..=1 {
                let (x, y) = if s.horizontal {
                    (along, s.pos + off)
                } else {
                    (s.pos + off, along)
                };
                let o = &mut over[idx(x, y)];
                // Don't overwrite room interiors the walkway passes through.
                if o.is_none() {
                    *o = Some(if off == 0 && centre_ok {
                        T_GRATE
                    } else {
                        T_PLATING
                    });
                }
            }
        }
    }
    // Hall centre band: plating strip down the middle of the hall.
    {
        let s = spines[hall_idx];
        let extent = if s.horizontal { w } else { h };
        for along in MARGIN + 3..extent - MARGIN - 3 {
            let (x, y) = if s.horizontal {
                (along, s.pos)
            } else {
                (s.pos, along)
            };
            if over[idx(x, y)].is_none() {
                over[idx(x, y)] = Some(T_PLATING);
            }
        }
    }

    // ── Pedestals: concentric L∞ rings rising to a solid core ────────────
    // (wall core, grate ring, plating ring — nesting keeps the ±1 invariant.)
    for r in &squares {
        let (px, py) = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2);
        let core: i32 = if rng.gen_bool(0.5) { 1 } else { 0 }; // 3×3 or 1×1 core
        for dy in -(core + 2)..=(core + 2) {
            for dx in -(core + 2)..=(core + 2) {
                let d = dx.abs().max(dy.abs());
                let tier = match d - core {
                    v if v <= 0 => T_WALL,
                    1 => T_GRATE,
                    _ => T_PLATING,
                };
                ped[idx(px + dx, py + dy)] = Some(tier);
            }
        }
    }

    // ── Distance field (two-pass L∞ chamfer) → base tiers ────────────────
    let mut dist = vec![i32::MAX / 2; n];
    for i in 0..n {
        if open[i] {
            dist[i] = 0;
        }
    }
    for pass in 0..2 {
        let (ys, xs): (Vec<i32>, Vec<i32>) = if pass == 0 {
            ((0..h).collect(), (0..w).collect())
        } else {
            ((0..h).rev().collect(), (0..w).rev().collect())
        };
        for &y in &ys {
            for &x in &xs {
                let mut best = dist[idx(x, y)];
                for (dx, dy) in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                    (1, 1),
                ] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < w && ny < h {
                        best = best.min(dist[idx(nx, ny)] + 1);
                    }
                }
                dist[idx(x, y)] = best;
            }
        }
    }

    let mut terrain: Vec<u32> = dist
        .iter()
        .map(|&d| match d {
            0 => T_FLOOR,
            1 => T_PLATING,
            2 => T_GRATE,
            3..=5 => T_WALL,
            6..=8 => (T_WALL + 1).min(top), // conduit band behind walls
            _ => top,                       // void
        })
        .collect();

    // Apply walkable overrides and pedestals.
    let mut pinned = vec![false; n];
    for i in 0..n {
        if open[i] {
            terrain[i] = over[i].unwrap_or(T_FLOOR);
            pinned[i] = true;
        }
        if let Some(t) = ped[i] {
            terrain[i] = t;
            pinned[i] = true;
        }
    }

    // Pin borders to the top tier.
    for x in 0..w {
        for y in [0, h - 1] {
            terrain[idx(x, y)] = top;
            pinned[idx(x, y)] = true;
        }
    }
    for y in 0..h {
        for x in [0, w - 1] {
            terrain[idx(x, y)] = top;
            pinned[idx(x, y)] = true;
        }
    }

    // Safety net: smooth the non-designed bands (wall/conduit/void ramps) so
    // the ±1 autotiler invariant holds everywhere, exactly as fbm does.
    crate::fbm::clamp_terrain_indices(&mut terrain, w as usize, h as usize, &pinned);

    terrain
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N: u32 = 6;

    fn map(seed: u64) -> Vec<u32> {
        generate_station_map(64, 64, N, seed)
    }

    #[test]
    fn deterministic_and_in_range() {
        assert_eq!(map(42), map(42));
        assert!(map(42).iter().all(|&t| t < N));
        assert_ne!(map(1), map(2));
    }

    #[test]
    fn borders_are_top_tier() {
        let m = map(7);
        for x in 0..64usize {
            assert_eq!(m[x], N - 1);
            assert_eq!(m[63 * 64 + x], N - 1);
            assert_eq!(m[x * 64], N - 1);
            assert_eq!(m[x * 64 + 63], N - 1);
        }
    }

    #[test]
    fn no_adjacent_jump_greater_than_one() {
        for seed in [1, 42, 99, 1337] {
            let m = generate_station_map(64, 64, N, seed);
            for y in 0..64i32 {
                for x in 0..64i32 {
                    let t = m[(y * 64 + x) as usize] as i32;
                    for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= 64 || ny >= 64 {
                            continue;
                        }
                        let nt = m[(ny * 64 + nx) as usize] as i32;
                        assert!(
                            (t - nt).abs() <= 1,
                            "seed {seed}: ({x},{y})={t} vs ({nx},{ny})={nt}"
                        );
                    }
                }
            }
        }
    }

    /// Every walkable tile must be reachable from the map centre (the
    /// landing pad) — corridors guarantee this by construction.
    #[test]
    fn walkable_area_is_connected() {
        for seed in [1, 42, 99, 1337] {
            let m = generate_station_map(64, 64, N, seed);
            let walkable = |i: usize| m[i] <= T_GRATE;
            let start = 32 * 64 + 32;
            assert!(walkable(start), "seed {seed}: centre not walkable");
            let mut seen = vec![false; m.len()];
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(i) = stack.pop() {
                let (x, y) = ((i % 64) as i32, (i / 64) as i32);
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= 64 || ny >= 64 {
                        continue;
                    }
                    let j = (ny * 64 + nx) as usize;
                    if !seen[j] && walkable(j) {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            let unreachable = (0..m.len()).filter(|&i| walkable(i) && !seen[i]).count();
            assert_eq!(
                unreachable, 0,
                "seed {seed}: {unreachable} stranded walkable tiles"
            );
        }
    }

    #[test]
    fn has_variety_and_solid_structure() {
        let m = map(42);
        for t in 0..N {
            assert!(m.contains(&t), "tier {t} missing");
        }
        // A meaningful share of the map is walkable, and some walkable
        // variety exists (plating/grate underfoot, not just bare floor).
        let open = m.iter().filter(|&&t| t <= T_GRATE).count();
        assert!(open > 64 * 64 / 8, "too little walkable area: {open}");
        assert!(m.iter().filter(|&&t| t == T_PLATING).count() > 40);
        assert!(m.iter().filter(|&&t| t == T_GRATE).count() > 20);
    }

    /// Dump a PPM visualisation to /tmp for eyeballing:
    /// `cargo test --features dev station_layout::tests::dump_ppm -- --ignored`
    #[test]
    #[ignore]
    fn dump_ppm() {
        let colors: [[u8; 3]; 6] = [
            [145, 152, 158], // floor
            [88, 94, 102],   // plating
            [108, 118, 128], // grate
            [52, 56, 62],    // wall
            [25, 78, 112],   // conduit
            [8, 8, 10],      // void
        ];
        for seed in [7u64, 42, 99] {
            let m = generate_station_map(64, 64, N, seed);
            let mut out = format!("P3\n{} {}\n255\n", 64, 64);
            for &t in &m {
                let c = colors[t as usize];
                out.push_str(&format!("{} {} {}\n", c[0], c[1], c[2]));
            }
            std::fs::write(format!("/tmp/station_{seed}.ppm"), out).unwrap();
        }
    }
}
