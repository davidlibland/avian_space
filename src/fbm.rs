//! fBm terrain map generation.
//!
//! Produces a flat `Vec<u32>` of terrain indices in `0..n_terrain_types`,
//! row-major (index = `y * width + x`), using fractional Brownian motion.

// ── Public API ────────────────────────────────────────────────────────────────

/// Generate a terrain map using fractional Brownian motion.
///
/// Each element is a terrain index in `0..n_terrain_types`, where 0 is the
/// terrain that appears at the lowest noise values (e.g. water) and
/// `n_terrain_types - 1` at the highest (e.g. mountain).
///
/// # Parameters
/// - `width`, `height`     — map dimensions in tiles
/// - `n_terrain_types`     — number of ordered terrain types
/// - `scale`               — noise coordinate scale; larger → bigger patches
/// - `octaves`             — number of fBm octaves (4–6 is typical)
/// - `lacunarity`          — frequency multiplier per octave (typically 2.0)
/// - `gain`                — amplitude multiplier per octave (typically 0.5)
/// - `seed`                — RNG seed; same seed → same map
pub fn generate_terrain_map(
    width: u32,
    height: u32,
    n_terrain_types: u32,
    scale: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    seed: u64,
) -> Vec<u32> {
    assert!(n_terrain_types >= 1);

    // Sample the fBm field at every tile coordinate.
    let mut field: Vec<f32> = (0..height)
        .flat_map(|ty| {
            (0..width).map(move |tx| {
                let x = tx as f32 / width as f32 * scale;
                let y = ty as f32 / height as f32 * scale;
                fbm(x, y, seed, octaves, lacunarity, gain)
            })
        })
        .collect();

    // Normalise to [0, 1].
    normalise(&mut field);

    // Pin borders to 1.0 (highest terrain = impassable) and mark them
    // as fixed so the gradient clamp never modifies them.  The clamp
    // will create a smooth ramp inward from the borders.
    let w = width as usize;
    let h = height as usize;
    let mut pinned = vec![false; w * h];
    for x in 0..w {
        field[x] = 1.0;                 // top row
        field[(h - 1) * w + x] = 1.0;   // bottom row
        pinned[x] = true;
        pinned[(h - 1) * w + x] = true;
    }
    for y in 0..h {
        field[y * w] = 1.0;             // left column
        field[y * w + (w - 1)] = 1.0;   // right column
        pinned[y * w] = true;
        pinned[y * w + (w - 1)] = true;
    }

    // Clamp the gradient so adjacent tiles can differ by at most one
    // terrain bin after quantisation.  Do NOT re-normalise afterwards —
    // rescaling would amplify differences and break the constraint.
    let max_step = 1.0 / n_terrain_types as f32;
    clamp_gradient(&mut field, w, h, max_step, &pinned);

    // Bin into terrain indices.  The field is still within [0, 1]
    // (clamping only pulls values inward), so binning is safe.
    let n = n_terrain_types as f32;
    field
        .iter()
        .map(|&v| {
            ((v * n) as u32).min(n_terrain_types - 1)
        })
        .collect()
}

// ── Field post-processing ─────────────────────────────────────────────────────

/// Normalise `field` in-place to [0, 1].
fn normalise(field: &mut [f32]) {
    let min = field.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = field.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-9);
    for v in field.iter_mut() {
        *v = (*v - min) / range;
    }
}

/// Clamp the gradient of a 2D field so no two 8-connected neighbours
/// (cardinal + diagonal) differ by more than `max_step`.
///
/// Cells where `pinned[idx]` is true are never modified (their values
/// act as fixed constraints that propagate inward).
///
/// Uses alternating forward/backward raster passes (like a distance
/// transform) until the field converges.
fn clamp_gradient(field: &mut [f32], w: usize, h: usize, max_step: f32, pinned: &[bool]) {
    loop {
        let mut changed = false;

        // Forward pass: top-left → bottom-right.
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if pinned[idx] {
                    continue;
                }
                let v = field[idx];
                let mut c = v;
                if x > 0 {
                    c = c.clamp(field[idx - 1] - max_step, field[idx - 1] + max_step);
                }
                if y > 0 {
                    c = c.clamp(field[idx - w] - max_step, field[idx - w] + max_step);
                    if x > 0 {
                        let tl = field[idx - w - 1];
                        c = c.clamp(tl - max_step, tl + max_step);
                    }
                    if x + 1 < w {
                        let tr = field[idx - w + 1];
                        c = c.clamp(tr - max_step, tr + max_step);
                    }
                }
                if c != v {
                    field[idx] = c;
                    changed = true;
                }
            }
        }

        // Backward pass: bottom-right → top-left.
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let idx = y * w + x;
                if pinned[idx] {
                    continue;
                }
                let v = field[idx];
                let mut c = v;
                if x + 1 < w {
                    c = c.clamp(field[idx + 1] - max_step, field[idx + 1] + max_step);
                }
                if y + 1 < h {
                    c = c.clamp(field[idx + w] - max_step, field[idx + w] + max_step);
                    if x > 0 {
                        let bl = field[idx + w - 1];
                        c = c.clamp(bl - max_step, bl + max_step);
                    }
                    if x + 1 < w {
                        let br = field[idx + w + 1];
                        c = c.clamp(br - max_step, br + max_step);
                    }
                }
                if c != v {
                    field[idx] = c;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

// ── fBm ───────────────────────────────────────────────────────────────────────

/// Fractional Brownian motion: a sum of `octaves` Perlin noise octaves.
///
/// Each octave uses a different seed so successive octaves are uncorrelated.
fn fbm(x: f32, y: f32, seed: u64, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;

    for octave in 0..octaves {
        let octave_seed = seed.wrapping_add((octave as u64).wrapping_mul(0x9e3779b97f4a7c15));
        value += amplitude * perlin(x * frequency, y * frequency, octave_seed);
        amplitude *= gain;
        frequency *= lacunarity;
    }

    value
}

// ── Perlin noise ──────────────────────────────────────────────────────────────

/// Single octave of Perlin noise. Returns a value in approximately [-1, 1].
fn perlin(x: f32, y: f32, seed: u64) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();

    let u = fade(xf);
    let v = fade(yf);

    let g00 = gradient(xi, yi, seed);
    let g10 = gradient(xi + 1, yi, seed);
    let g01 = gradient(xi, yi + 1, seed);
    let g11 = gradient(xi + 1, yi + 1, seed);

    lerp(
        lerp(dot(g00, xf, yf), dot(g10, xf - 1.0, yf), u),
        lerp(dot(g01, xf, yf - 1.0), dot(g11, xf - 1.0, yf - 1.0), u),
        v,
    )
}

/// Pseudo-random gradient vector for lattice point `(ix, iy)`.
///
/// Uses a fast hash instead of a full RNG to pick one of 8 unit gradient
/// directions. The same `(ix, iy, seed)` always produces the same gradient.
fn gradient(ix: i32, iy: i32, seed: u64) -> (f32, f32) {
    // Hash the coordinates into a pseudo-random u64.
    // Cast through u32 to avoid sign-extension issues with negative coords.
    let h = (ix as u32 as u64)
        .wrapping_mul(1_000_003)
        .wrapping_add((iy as u32 as u64).wrapping_mul(999_983))
        .wrapping_add(seed);
    // Mix bits (splitmix64-style).
    let h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    let h = h ^ (h >> 31);

    match h & 7 {
        0 => (1.0, 1.0),
        1 => (-1.0, 1.0),
        2 => (1.0, -1.0),
        3 => (-1.0, -1.0),
        4 => (1.0, 0.0),
        5 => (-1.0, 0.0),
        6 => (0.0, 1.0),
        _ => (0.0, -1.0),
    }
}

// ── Math helpers ──────────────────────────────────────────────────────────────

/// Perlin's quintic fade: 6t^5 - 15t^4 + 10t^3
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}
#[inline]
fn dot((gx, gy): (f32, f32), dx: f32, dy: f32) -> f32 {
    gx * dx + gy * dy
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_map() -> Vec<u32> {
        generate_terrain_map(64, 64, 5, 4.0, 5, 2.0, 0.5, 42)
    }

    #[test]
    fn correct_length() {
        assert_eq!(default_map().len(), 64 * 64);
    }

    #[test]
    fn values_in_range() {
        assert!(default_map().iter().all(|&t| t < 5));
    }

    #[test]
    fn deterministic() {
        assert_eq!(default_map(), default_map());
    }

    #[test]
    fn different_seeds_differ() {
        let a = generate_terrain_map(64, 64, 5, 4.0, 5, 2.0, 0.5, 1);
        let b = generate_terrain_map(64, 64, 5, 4.0, 5, 2.0, 0.5, 2);
        assert_ne!(a, b);
    }

    fn assert_no_jumps(map: &[u32], w: u32, h: u32, label: &str) {
        let offsets: &[(i32, i32)] = &[(1, 0), (0, 1), (1, 1), (1, -1)];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let t = map[(y as u32 * w + x as u32) as usize] as i32;
                for &(dx, dy) in offsets {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let n = map[(ny as u32 * w + nx as u32) as usize] as i32;
                    assert!(
                        (t - n).abs() <= 1,
                        "{label}: ({x},{y})→({nx},{ny}): terrain {t}→{n}",
                    );
                }
            }
        }
    }

    #[test]
    fn no_adjacent_jump_greater_than_one() {
        for seed in [1, 42, 99, 1337] {
            // Large map
            let map = generate_terrain_map(64, 64, 5, 4.0, 5, 2.0, 0.5, seed);
            assert_no_jumps(&map, 64, 64, &format!("64x64 seed={seed}"));
            // Small map (matches in-game WORLD_WIDTH/HEIGHT)
            let map = generate_terrain_map(16, 16, 5, 4.0, 5, 2.0, 0.5, seed);
            assert_no_jumps(&map, 16, 16, &format!("16x16 seed={seed}"));
        }
    }

    #[test]
    fn borders_are_highest_terrain() {
        for seed in [1, 42, 99] {
            for (w, h) in [(16, 16), (64, 64)] {
                let n = 5u32;
                let map = generate_terrain_map(w, h, n, 4.0, 5, 2.0, 0.5, seed);
                let max_t = n - 1;
                for x in 0..w {
                    assert_eq!(
                        map[x as usize], max_t,
                        "seed={seed} top border ({x},0) = {}, expected {max_t}",
                        map[x as usize]
                    );
                    assert_eq!(
                        map[((h - 1) * w + x) as usize], max_t,
                        "seed={seed} bottom border ({x},{}) = {}",
                        h - 1,
                        map[((h - 1) * w + x) as usize]
                    );
                }
                for y in 0..h {
                    assert_eq!(
                        map[(y * w) as usize], max_t,
                        "seed={seed} left border (0,{y}) = {}",
                        map[(y * w) as usize]
                    );
                    assert_eq!(
                        map[(y * w + w - 1) as usize], max_t,
                        "seed={seed} right border ({},{y}) = {}",
                        w - 1,
                        map[(y * w + w - 1) as usize]
                    );
                }
            }
        }
    }

    #[test]
    fn all_terrain_types_present() {
        let map = generate_terrain_map(128, 128, 5, 4.0, 5, 2.0, 0.5, 7);
        for t in 0..5u32 {
            assert!(map.contains(&t), "terrain type {t} missing from map");
        }
    }
}
