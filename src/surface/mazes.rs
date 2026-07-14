//! Maze-layout interiors (docs/maze_interiors_plan.md).
//!
//! Three venues, three DEDICATED generation algorithms — the layout *is*
//! the venue's character:
//!
//! * **Mine** — a recursive-backtracker (depth-first spanning tree) maze on
//!   a coarse cell grid: winding tunnels, real dead ends, a few carved-out
//!   galleries. Trees maximise "lostness"; a couple of braids are added so
//!   backtracking isn't the only option. Multi-level: shafts descend.
//! * **Warehouse** — not a maze algorithm at all: one big hall filled with
//!   an aisle grid of container blocks (some merged, some missing). Long
//!   sightlines, right-angle ambushes; connectivity holds by construction
//!   because every block is ringed by aisles.
//! * **Substation** — binary-space-partition rooms joined by L-shaped
//!   service corridors along the BSP tree (guaranteed connected), plus a
//!   few extra links so the level reads as a braided pipe network.
//!
//! All venues emit a designed FLOOR MASK; `terrain_from_floor` turns it
//! into blob47-safe tiers via an L∞ distance transform (tier = distance,
//! clamped), so the ±1 gradient contract holds everywhere by construction.
//! In maze venues everything above the plating apron is SOLID
//! (`solid_min_tier` = grate), which lets walls be 3 tiles thin.

use super::{BuildingKind, N_TERRAIN_TYPES, WORLD_HEIGHT, WORLD_WIDTH};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Everything a maze level needs to be spawned by `setup_interior`.
pub(crate) struct MazeLevel {
    /// Designed walkable core (row-major, WORLD_WIDTH × WORLD_HEIGHT).
    pub floor: Vec<bool>,
    /// Extra solid tiles INSIDE the floor (warehouse containers).
    pub solid: Vec<bool>,
    /// Exterior door (level 0 only): south end of the entrance corridor.
    pub door: Option<(u32, u32)>,
    /// Where the walker appears.
    pub entry: (u32, u32),
    /// Stairs back up (levels > 0).
    pub stairs_up: Option<(u32, u32)>,
    /// Stairs further down (levels below this one exist).
    pub stairs_down: Option<(u32, u32)>,
    /// Farthest walkable tile from the entry — hunt targets hide here.
    pub hunt_spot: (u32, u32),
    /// Decorative prop anchors (kind name, tile). Sprites resolved later.
    pub props: Vec<(&'static str, (u32, u32))>,
}

/// Is this kind a maze venue (vs. a walk-in shop)?
pub(crate) fn is_maze(kind: BuildingKind) -> bool {
    matches!(
        kind,
        BuildingKind::Mine | BuildingKind::Warehouse | BuildingKind::Substation
    )
}

/// How many levels this venue descends on this planet. Warehouses are one
/// sprawling floor; mines and substations go down.
pub(crate) fn levels_for(kind: BuildingKind, seed: u64) -> u8 {
    match kind {
        BuildingKind::Mine => 2 + (seed % 2) as u8,
        BuildingKind::Substation => 1 + (seed % 2) as u8,
        _ => 1,
    }
}

/// Deterministic per-planet seed (same fold the surface generator uses).
pub(crate) fn planet_seed(planet: &str) -> u64 {
    planet
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

pub(crate) fn build_maze_level(kind: BuildingKind, seed: u64, level: u8) -> MazeLevel {
    let mut rng = StdRng::seed_from_u64(
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(level as u64),
    );
    match kind {
        BuildingKind::Mine => mine_level(&mut rng, level, levels_for(kind, seed)),
        BuildingKind::Warehouse => warehouse_level(&mut rng),
        BuildingKind::Substation => substation_level(&mut rng, level, levels_for(kind, seed)),
        _ => unreachable!("not a maze venue"),
    }
}

const W: u32 = WORLD_WIDTH;
const H: u32 = WORLD_HEIGHT;

fn idx(x: u32, y: u32) -> usize {
    (y * W + x) as usize
}

/// Fill a rect of floor.
fn carve(floor: &mut [bool], x0: u32, y0: u32, w: u32, h: u32) {
    for y in y0..(y0 + h).min(H) {
        for x in x0..(x0 + w).min(W) {
            floor[idx(x, y)] = true;
        }
    }
}

/// BFS distances over the floor mask (u32::MAX = unreachable).
fn bfs(floor: &[bool], solid: &[bool], start: (u32, u32)) -> Vec<u32> {
    let mut dist = vec![u32::MAX; floor.len()];
    let mut q = std::collections::VecDeque::new();
    if floor[idx(start.0, start.1)] && !solid[idx(start.0, start.1)] {
        dist[idx(start.0, start.1)] = 0;
        q.push_back(start);
    }
    while let Some((x, y)) = q.pop_front() {
        let d = dist[idx(x, y)];
        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if nx < 0 || ny < 0 || nx >= W as i32 || ny >= H as i32 {
                continue;
            }
            let (nx, ny) = (nx as u32, ny as u32);
            let ni = idx(nx, ny);
            if floor[ni] && !solid[ni] && dist[ni] == u32::MAX {
                dist[ni] = d + 1;
                q.push_back((nx, ny));
            }
        }
    }
    dist
}

/// The farthest reachable tile from `start`.
fn farthest_from(floor: &[bool], solid: &[bool], start: (u32, u32)) -> (u32, u32) {
    let dist = bfs(floor, solid, start);
    let mut best = (start, 0u32);
    for y in 0..H {
        for x in 0..W {
            let d = dist[idx(x, y)];
            if d != u32::MAX && d > best.1 {
                best = ((x, y), d);
            }
        }
    }
    best.0
}

/// Carve the level-0 entrance: a 2-wide corridor from the southmost floor
/// tile nearest the map's centre column straight south, ending at the door.
fn carve_entrance(floor: &mut [bool]) -> ((u32, u32), (u32, u32)) {
    let mut best: Option<(u32, u32)> = None;
    for y in 0..H {
        for x in 0..W {
            if floor[idx(x, y)] {
                let better = match best {
                    None => true,
                    Some((bx, by)) => (y, x.abs_diff(W / 2)) < (by, bx.abs_diff(W / 2)),
                };
                if better {
                    best = Some((x, y));
                }
            }
        }
    }
    let (sx, sy) = best.expect("maze has floor");
    let door_y = sy.saturating_sub(4);
    for y in door_y..=sy {
        carve(floor, sx, y, 2, 1);
    }
    ((sx, door_y), (sx, sy)) // (door, entry just inside)
}

/// Scatter `count` props of `name` onto random free floor tiles, skipping
/// reserved tiles (entry/stairs/hunt) and tiles already claimed.
fn scatter(
    rng: &mut StdRng,
    floor: &[bool],
    solid: &[bool],
    reserved: &[(u32, u32)],
    used: &mut std::collections::HashSet<(u32, u32)>,
    props: &mut Vec<(&'static str, (u32, u32))>,
    name: &'static str,
    count: usize,
) {
    let mut placed = 0;
    for _ in 0..count * 12 {
        if placed >= count {
            break;
        }
        let t = (rng.gen_range(2..W - 2), rng.gen_range(2..H - 2));
        let i = idx(t.0, t.1);
        if floor[i] && !solid[i] && !reserved.contains(&t) && used.insert(t) {
            props.push((name, t));
            placed += 1;
        }
    }
}

// ── Mine: recursive backtracker ────────────────────────────────────────────

fn mine_level(rng: &mut StdRng, level: u8, n_levels: u8) -> MazeLevel {
    // Coarse cell grid: 2-wide corridors, 3-tile wall bands (pitch 5).
    const MARGIN: u32 = 5;
    const PITCH: u32 = 5;
    const CW: u32 = 2;
    let cells_w = ((W - 2 * MARGIN - 3) / PITCH) as usize; // ≈ 10
    let cells_h = ((H - 2 * MARGIN - 3) / PITCH) as usize;
    let cell_org = |cx: usize, cy: usize| -> (u32, u32) {
        (
            MARGIN + 3 + cx as u32 * PITCH,
            MARGIN + 3 + cy as u32 * PITCH,
        )
    };

    // Depth-first spanning tree over the cells.
    let mut visited = vec![false; cells_w * cells_h];
    let mut carved: Vec<(usize, usize)> = Vec::new(); // edges (cell a, cell b)
    let mut stack = vec![rng.gen_range(0..cells_w * cells_h)];
    visited[stack[0]] = true;
    while let Some(&cur) = stack.last() {
        let (cx, cy) = (cur % cells_w, cur / cells_w);
        let mut nbrs = Vec::new();
        if cx > 0 {
            nbrs.push(cur - 1);
        }
        if cx + 1 < cells_w {
            nbrs.push(cur + 1);
        }
        if cy > 0 {
            nbrs.push(cur - cells_w);
        }
        if cy + 1 < cells_h {
            nbrs.push(cur + cells_w);
        }
        nbrs.retain(|&n| !visited[n]);
        if nbrs.is_empty() {
            stack.pop();
        } else {
            let next = nbrs[rng.gen_range(0..nbrs.len())];
            visited[next] = true;
            carved.push((cur, next));
            stack.push(next);
        }
    }
    // Braid: a few extra passages so backtracking isn't the only way out.
    for _ in 0..(cells_w * cells_h / 12).max(2) {
        let a = rng.gen_range(0..cells_w * cells_h);
        let (ax, ay) = (a % cells_w, a / cells_w);
        let b = if rng.r#gen::<bool>() && ax + 1 < cells_w {
            a + 1
        } else if ay + 1 < cells_h {
            a + cells_w
        } else {
            continue;
        };
        carved.push((a, b));
    }

    // Rasterise: cell floors + carved passages.
    let mut floor = vec![false; (W * H) as usize];
    for cy in 0..cells_h {
        for cx in 0..cells_w {
            let (x, y) = cell_org(cx, cy);
            carve(&mut floor, x, y, CW, CW);
        }
    }
    for &(a, b) in &carved {
        let (ax, ay) = cell_org(a % cells_w, a / cells_w);
        let (bx, by) = cell_org(b % cells_w, b / cells_w);
        let (x0, y0) = (ax.min(bx), ay.min(by));
        let (x1, y1) = (ax.max(bx) + CW, ay.max(by) + CW);
        carve(&mut floor, x0, y0, x1 - x0, y1 - y0);
    }
    // Galleries: blow out a few cells into 2×2-cell chambers by carving the
    // full block spanned by a cell and its south-east neighbours.
    for _ in 0..3 {
        let cx = rng.gen_range(0..cells_w.saturating_sub(1));
        let cy = rng.gen_range(0..cells_h.saturating_sub(1));
        let (x, y) = cell_org(cx, cy);
        carve(&mut floor, x, y, PITCH + CW, PITCH + CW);
    }

    let solid = vec![false; (W * H) as usize];
    let (door, entry) = if level == 0 {
        let (d, e) = carve_entrance(&mut floor);
        (Some(d), e)
    } else {
        // Arrive at the up-shaft: a fixed corner cell.
        let (x, y) = cell_org(0, 0);
        (None, (x, y))
    };
    let stairs_up = (level > 0).then(|| {
        let (x, y) = cell_org(0, 0);
        (x, y)
    });
    // The down-shaft sits at the farthest cell from the entry.
    let stairs_down = (level + 1 < n_levels).then(|| farthest_from(&floor, &solid, entry));
    let hunt_spot = farthest_from(&floor, &solid, entry);

    // Dressing: the tunnels are a WORKED dig — braces and lanterns along
    // the ways, pebbles and ore everywhere, a dropped pickaxe; the deepest
    // level grows glow-crystals.
    let mut props = Vec::new();
    let reserved: Vec<(u32, u32)> = [Some(entry), stairs_up, stairs_down, Some(hunt_spot)]
        .into_iter()
        .flatten()
        .collect();
    let mut used = std::collections::HashSet::new();
    let sc = &mut |rng: &mut StdRng, props: &mut Vec<_>, used: &mut _, name, n| {
        scatter(rng, &floor, &solid, &reserved, used, props, name, n)
    };
    sc(rng, &mut props, &mut used, "timber_brace", 5);
    sc(rng, &mut props, &mut used, "lantern", 5);
    sc(rng, &mut props, &mut used, "ore_cart", 2);
    sc(rng, &mut props, &mut used, "pebbles_a", 8);
    sc(rng, &mut props, &mut used, "pebbles_b", 8);
    sc(rng, &mut props, &mut used, "ore_chunk", 6);
    sc(rng, &mut props, &mut used, "ore_pile", 3);
    sc(rng, &mut props, &mut used, "pickaxe", 1);
    if level + 1 == n_levels {
        sc(rng, &mut props, &mut used, "crystal", 4);
    }

    MazeLevel {
        floor,
        solid,
        door,
        entry,
        stairs_up,
        stairs_down,
        hunt_spot,
        props,
    }
}

// ── Warehouse: aisle grid of container blocks ──────────────────────────────

fn warehouse_level(rng: &mut StdRng) -> MazeLevel {
    // One big hall...
    let (hw, hh) = (44u32.min(W - 12), 32u32.min(H - 12));
    let (x0, y0) = ((W - hw) / 2, (H - hh) / 2);
    let mut floor = vec![false; (W * H) as usize];
    carve(&mut floor, x0, y0, hw, hh);

    // ...filled with container blocks: 4×2 blocks, 2-wide aisles, a 2-wide
    // perimeter lane. Some blocks merge into longer runs, some are missing.
    let mut solid = vec![false; (W * H) as usize];
    let (bw, bh, aisle) = (4u32, 2u32, 2u32);
    let mut props = Vec::new();
    let mut by = y0 + aisle;
    while by + bh + aisle <= y0 + hh {
        let mut bx = x0 + aisle;
        while bx + bw + aisle <= x0 + hw {
            let roll: f32 = rng.r#gen();
            if roll < 0.72 {
                // A block, possibly merged with the next slot to its east.
                let merged = roll < 0.2 && bx + 2 * bw + 2 * aisle <= x0 + hw;
                let w = if merged { bw * 2 + aisle } else { bw };
                for y in by..by + bh {
                    for x in bx..bx + w {
                        solid[idx(x, y)] = true;
                    }
                }
                props.push((
                    if rng.r#gen::<bool>() {
                        "container_a"
                    } else {
                        "container_b"
                    },
                    (bx, by),
                ));
                if merged {
                    props.push(("container_a", (bx + bw + aisle, by)));
                    bx += bw + aisle; // skip the absorbed slot
                }
            } else if roll > 0.94 {
                props.push(("crate_stack", (bx + 1, by)));
            }
            bx += bw + aisle;
        }
        by += bh + aisle;
    }

    let (door, entry) = carve_entrance(&mut floor);
    let hunt_spot = farthest_from(&floor, &solid, entry);
    // Aisle clutter: pallets, drums, and the odd burst crate.
    let reserved = vec![entry, hunt_spot];
    let mut used = std::collections::HashSet::new();
    scatter(
        rng, &floor, &solid, &reserved, &mut used, &mut props, "pallet", 5,
    );
    scatter(
        rng, &floor, &solid, &reserved, &mut used, &mut props, "barrel", 5,
    );
    scatter(
        rng,
        &floor,
        &solid,
        &reserved,
        &mut used,
        &mut props,
        "box_spill",
        3,
    );
    MazeLevel {
        floor,
        solid,
        door: Some(door),
        entry,
        stairs_up: None,
        stairs_down: None,
        hunt_spot,
        props,
    }
}

// ── Substation: BSP rooms + service corridors ──────────────────────────────

fn substation_level(rng: &mut StdRng, level: u8, n_levels: u8) -> MazeLevel {
    const MARGIN: u32 = 5;
    // BSP split of the usable area into leaves ≥ 14 tiles across.
    #[derive(Clone, Copy)]
    struct Leaf(u32, u32, u32, u32); // x, y, w, h
    let mut leaves = vec![Leaf(MARGIN, MARGIN, W - 2 * MARGIN, H - 2 * MARGIN)];
    loop {
        let mut split_any = false;
        let mut next = Vec::new();
        for &Leaf(x, y, w, h) in &leaves {
            if w.max(h) >= 28 {
                split_any = true;
                if w >= h {
                    let cut = w / 2 + rng.gen_range(0..w / 8 + 1) - w / 16;
                    next.push(Leaf(x, y, cut, h));
                    next.push(Leaf(x + cut, y, w - cut, h));
                } else {
                    let cut = h / 2 + rng.gen_range(0..h / 8 + 1) - h / 16;
                    next.push(Leaf(x, y, w, cut));
                    next.push(Leaf(x, y + cut, w, h - cut));
                }
            } else {
                next.push(Leaf(x, y, w, h));
            }
        }
        leaves = next;
        if !split_any {
            break;
        }
    }

    // A pump room centred in each leaf.
    let mut floor = vec![false; (W * H) as usize];
    let mut rooms = Vec::new();
    for &Leaf(x, y, w, h) in &leaves {
        let (rw, rh) = (
            rng.gen_range(5..=7).min(w.saturating_sub(6)).max(4),
            rng.gen_range(5..=7).min(h.saturating_sub(6)).max(4),
        );
        let (rx, ry) = (x + (w - rw) / 2, y + (h - rh) / 2);
        carve(&mut floor, rx, ry, rw, rh);
        rooms.push((rx + rw / 2, ry + rh / 2));
    }
    // Corridors: chain the rooms (sorted by leaf position = BSP order) with
    // L-shaped 2-wide runs — a connected spine — plus a few extra links.
    let link = |floor: &mut Vec<bool>, a: (u32, u32), b: (u32, u32)| {
        let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
        carve(floor, x0, a.1, x1 - x0 + 2, 2);
        let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
        carve(floor, b.0, y0, 2, y1 - y0 + 2);
    };
    for i in 1..rooms.len() {
        link(&mut floor, rooms[i - 1], rooms[i]);
    }
    for _ in 0..rooms.len() / 3 {
        let (a, b) = (rng.gen_range(0..rooms.len()), rng.gen_range(0..rooms.len()));
        if a != b {
            link(&mut floor, rooms[a], rooms[b]);
        }
    }

    let solid = vec![false; (W * H) as usize];
    let (door, entry) = if level == 0 {
        let (d, e) = carve_entrance(&mut floor);
        (Some(d), e)
    } else {
        (None, rooms[0])
    };
    let stairs_up = (level > 0).then(|| rooms[0]);
    let stairs_down = (level + 1 < n_levels).then(|| farthest_from(&floor, &solid, entry));
    let hunt_spot = farthest_from(&floor, &solid, entry);

    let mut props = Vec::new();
    for (i, &(cx, cy)) in rooms.iter().enumerate() {
        props.push((
            ["pump_unit", "pipe_valve"][i % 2],
            (cx.saturating_sub(1), cy.saturating_sub(1)),
        ));
    }
    // Service-level clutter: coils, spare pipe, cones round the works,
    // coolant pooling on the deck, gauge panels ticking away.
    let reserved: Vec<(u32, u32)> = [Some(entry), stairs_up, stairs_down, Some(hunt_spot)]
        .into_iter()
        .flatten()
        .collect();
    let mut used = std::collections::HashSet::new();
    scatter(
        rng,
        &floor,
        &solid,
        &reserved,
        &mut used,
        &mut props,
        "cable_coil",
        4,
    );
    scatter(
        rng,
        &floor,
        &solid,
        &reserved,
        &mut used,
        &mut props,
        "pipe_segment",
        3,
    );
    scatter(
        rng,
        &floor,
        &solid,
        &reserved,
        &mut used,
        &mut props,
        "warning_cone",
        3,
    );
    scatter(
        rng,
        &floor,
        &solid,
        &reserved,
        &mut used,
        &mut props,
        "coolant_puddle",
        6,
    );
    scatter(
        rng,
        &floor,
        &solid,
        &reserved,
        &mut used,
        &mut props,
        "gauge_panel",
        2,
    );

    MazeLevel {
        floor,
        solid,
        door,
        entry,
        stairs_up,
        stairs_down,
        hunt_spot,
        props,
    }
}

// ── Floor mask → blob47-safe terrain ───────────────────────────────────────

/// L∞ distance transform from the floor set → tier = min(d, top). Neighbours
/// (incl. diagonals) differ by ≤1 in distance, so the ±1 contract holds by
/// construction. Two-pass chamfer, exactly as fbm/station_layout do it.
pub(crate) fn terrain_from_floor(floor: &[bool]) -> Vec<u32> {
    let top = N_TERRAIN_TYPES - 1;
    let (w, h) = (W as i32, H as i32);
    let mut d = vec![u32::MAX - 1; floor.len()];
    for (i, &f) in floor.iter().enumerate() {
        if f {
            d[i] = 0;
        }
    }
    let at = |d: &Vec<u32>, x: i32, y: i32| -> u32 {
        if x < 0 || y < 0 || x >= w || y >= h {
            u32::MAX - 1
        } else {
            d[(y * w + x) as usize]
        }
    };
    for y in 0..h {
        for x in 0..w {
            let m = at(&d, x - 1, y)
                .min(at(&d, x, y - 1))
                .min(at(&d, x - 1, y - 1))
                .min(at(&d, x + 1, y - 1))
                .saturating_add(1);
            let i = (y * w + x) as usize;
            d[i] = d[i].min(m);
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let m = at(&d, x + 1, y)
                .min(at(&d, x, y + 1))
                .min(at(&d, x + 1, y + 1))
                .min(at(&d, x - 1, y + 1))
                .saturating_add(1);
            let i = (y * w + x) as usize;
            d[i] = d[i].min(m);
        }
    }
    d.into_iter().map(|v| v.min(top)).collect()
}
