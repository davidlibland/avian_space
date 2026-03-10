use avian2d::prelude::*;
use bevy::prelude::*;
use rand::Rng;

use crate::BOUNDS; // your world half-size constant

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct StarfieldPlugin;

impl Plugin for StarfieldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_starfield).add_systems(
            Update,
            (shift_starfield, wrap_starfield)
                .chain()
                .before(PhysicsSystems::Writeback),
        );
    }
}

// ── Components ────────────────────────────────────────────────────────────────

/// Marks a star sprite. `parallax` ∈ (0, 1]: fraction of world offset applied
/// each frame. Values close to 0 feel very distant; 1.0 moves with the world.
#[derive(Component)]
pub struct Star {
    pub parallax: f32,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Describes one layer of the starfield.
struct StarLayer {
    count: usize,
    parallax: f32,
    size_range: (f32, f32),
    brightness: f32, // alpha, so distant layers appear dimmer
}

const STAR_LAYERS: &[StarLayer] = &[
    StarLayer {
        count: 4000,
        parallax: 0.1,
        size_range: (0.5, 1.0),
        brightness: 0.3,
    },
    StarLayer {
        count: 750,
        parallax: 0.3,
        size_range: (1.0, 1.5),
        brightness: 0.5,
    },
    StarLayer {
        count: 160,
        parallax: 0.6,
        size_range: (1.5, 2.5),
        brightness: 0.8,
    },
    StarLayer {
        count: 40,
        parallax: 1.0,
        size_range: (2.5, 4.0),
        brightness: 1.0,
    },
];

// z-depth for stars (behind everything else)
const STAR_Z: f32 = -10.0;

// ── Systems ───────────────────────────────────────────────────────────────────

fn spawn_starfield(mut commands: Commands) {
    let mut rng = rand::thread_rng();

    for layer in STAR_LAYERS {
        for _ in 0..layer.count {
            let effective_bounds = BOUNDS / layer.parallax;
            let x = rng.gen_range(-effective_bounds..effective_bounds);
            let y = rng.gen_range(-effective_bounds..effective_bounds);
            let size = rng.gen_range(layer.size_range.0..layer.size_range.1);

            let red_shift = layer.parallax + (1.0 - layer.parallax) * 0.5;
            commands.spawn((
                Star {
                    parallax: layer.parallax,
                },
                Sprite {
                    color: Color::srgba(1.0, 1.0 * red_shift, 1.0 * red_shift, layer.brightness),
                    custom_size: Some(Vec2::splat(size)),
                    ..default()
                },
                Transform::from_xyz(x, y, STAR_Z),
            ));
        }
    }
}

/// Called from outside (e.g. your `recenter_world` system) to shift stars by
/// `offset * parallax`. Exposed as a public resource so `recenter_world` can
/// write the offset once and this system reads it.
///
/// We store the world offset in a resource so the two systems stay decoupled.
#[derive(Resource, Default)]
pub struct WorldOffset(pub Vec2);

/// Your `recenter_world` system should insert this resource and write to it
/// each frame before `shift_starfield` runs. See the example at the bottom
/// of this file.
fn shift_starfield(offset: Res<WorldOffset>, mut query: Query<(&Star, &mut Transform)>) {
    if offset.0 == Vec2::ZERO {
        return;
    }
    for (star, mut transform) in query.iter_mut() {
        transform.translation -= (offset.0 * star.parallax).extend(0.0);
    }
}

/// Wraps stars back onto the torus. Because each layer moves at a different
/// speed, its effective world size is `BOUNDS / parallax`; we wrap at that
/// boundary so stars tile seamlessly within their own layer.
fn wrap_starfield(mut query: Query<(&Star, &mut Transform)>) {
    for (star, mut transform) in query.iter_mut() {
        // Each parallax layer lives on a torus scaled by 1/parallax.
        // Wrapping at ±(BOUNDS / parallax) keeps the density uniform.
        let effective_bounds = BOUNDS / star.parallax;
        let diameter = 2.0 * effective_bounds;

        transform.translation.x =
            ((transform.translation.x + effective_bounds).rem_euclid(diameter)) - effective_bounds;
        transform.translation.y =
            ((transform.translation.y + effective_bounds).rem_euclid(diameter)) - effective_bounds;
    }
}

// ── Integration example ───────────────────────────────────────────────────────
//
// In main.rs:
//
//   mod starfield;
//   use starfield::{StarfieldPlugin, WorldOffset};
//
//   App::new()
//       .add_plugins((DefaultPlugins, PhysicsPlugins::default(), StarfieldPlugin))
//       .init_resource::<WorldOffset>()   // ← add this
//       ...
//       .add_systems(Update, (
//           player_wrapping_system,
//           recenter_world,
//           camera_follow_player,
//       ).chain().before(PhysicsSystems::Writeback))
//       .run();
//
// Your recenter_world system should write the offset before this plugin's
// systems run:
//
//   fn recenter_world(
//       mut player_query: Query<(&mut Position, &mut Transform), With<Player>>,
//       mut physics_query: Query<(&mut Position, &mut Transform), (With<RigidBody>, Without<Player>)>,
//       mut visual_query: Query<&mut Transform, (Without<RigidBody>, Without<Player>, Without<Camera2d>, Without<Star>)>,
//       mut camera_query: Query<&mut Transform, (With<Camera2d>, Without<RigidBody>, Without<Player>)>,
//       mut world_offset: ResMut<WorldOffset>,
//   ) {
//       let Ok((mut player_pos, mut player_transform)) = player_query.single_mut() else { return };
//
//       let offset = player_pos.0;
//       world_offset.0 = offset;           // ← write offset for starfield
//
//       if offset == Vec2::ZERO { return; }
//
//       player_pos.0 = Vec2::ZERO;
//       player_transform.translation = player_transform.translation.with_xy(Vec2::ZERO);
//
//       for (mut pos, mut transform) in physics_query.iter_mut() {
//           pos.0 -= offset;
//           transform.translation -= offset.extend(0.0);
//       }
//       for mut transform in visual_query.iter_mut() {
//           transform.translation -= offset.extend(0.0);
//       }
//       if let Ok(mut cam) = camera_query.single_mut() {
//           cam.translation -= offset.extend(0.0);
//       }
//   }
//
// Note: `Without<Star>` is added to `visual_query` so stars are NOT shifted
// by the full offset — the starfield plugin handles them at their own parallax.
