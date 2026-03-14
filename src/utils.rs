use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};

/// Despawn an entity safely — silently does nothing if it was already despawned.
///
/// Use this everywhere instead of `commands.entity(e).despawn()` for entities
/// that also carry `DespawnOnExit`, because a state transition and a manual
/// despawn queued in the same frame would otherwise both try to remove the
/// entity, producing a warning.
pub fn safe_despawn(commands: &mut Commands, entity: Entity) {
    commands.queue(move |world: &mut World| {
        let _ = world.try_despawn(entity);
    });
}
use rand::Rng;
use std::f32::consts::PI;

pub fn polygon_mesh(verts: &[Vec2]) -> Mesh {
    // use bevy::render::mesh::{Indices, PrimitiveTopology};
    // use bevy::render::render_asset::RenderAssetUsages;

    let positions: Vec<[f32; 3]> = verts.iter().map(|v| [v.x, v.y, 0.0]).collect();
    let n = verts.len();
    // Fan triangulation from vertex 0
    let mut indices: Vec<u32> = Vec::new();
    for i in 1..(n as u32 - 1) {
        indices.push(0);
        indices.push(i);
        indices.push(i + 1);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

pub fn random_velocity(speed: f32) -> Vec2 {
    let mut rng = rand::thread_rng();
    let angle = rng.gen_range(0.0_f32..(2.0 * std::f32::consts::PI));
    Vec2::new(angle.cos(), angle.sin()) * speed
}

pub fn angle_to_hit(proj_vel: f32, obj_pos: &Vec2, obj_vel: &Vec2) -> Option<f32> {
    let a = obj_vel.length_squared() - proj_vel.powi(2);
    if a == 0.0 {
        return None;
    }
    let b = 2.0 * (obj_pos.x * obj_vel.x + obj_pos.y * obj_vel.y);
    let c = obj_pos.length_squared();
    let disc_sq = b.powi(2) - 4.0 * a * c;
    if disc_sq < 0.0 {
        return None;
    }
    let disc = disc_sq.sqrt();
    let t1 = (-b + disc) / (2.0 * a);
    let t2 = (-b - disc) / (2.0 * a);
    if t1 < 0.0 && t2 < 0.0 {
        return None;
    }
    let t = if t1 < 0.0 {
        t2
    } else if t2 < 0.0 {
        t1
    } else if t1 < t2 {
        t1
    } else {
        t2
    };
    let contact_pos = obj_pos + t * obj_vel;
    return Some(contact_pos.y.atan2(contact_pos.x));
}

/// Tests for despawn safety and collision deduplication logic.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// `world.try_despawn` must not panic if the entity is already gone.
    /// This is the core property that `safe_despawn` relies on.
    #[test]
    fn try_despawn_is_idempotent() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();

        // First despawn — entity exists, should succeed.
        assert!(world.try_despawn(entity).is_ok(), "first try_despawn should succeed");

        // Second despawn — entity is gone, must not panic and return an error.
        assert!(world.try_despawn(entity).is_err(), "second try_despawn should return Err, not panic");
    }

    /// `safe_despawn` must apply correctly via the command queue and survive
    /// being called twice for the same entity (the second call is a silent no-op).
    #[test]
    fn safe_despawn_via_commands() {
        use bevy::ecs::system::SystemState;

        let mut world = World::new();
        let entity = world.spawn_empty().id();

        // Queue safe_despawn twice for the same entity.
        let mut system_state: SystemState<Commands> = SystemState::new(&mut world);
        {
            let mut commands = system_state.get_mut(&mut world);
            safe_despawn(&mut commands, entity);
            safe_despawn(&mut commands, entity); // second call should be a silent no-op
        }
        system_state.apply(&mut world); // flush queued commands

        assert!(
            !world.entities().contains(entity),
            "entity should be despawned after safe_despawn"
        );
    }

    /// The HashSet deduplication used in collision_system must prevent the same
    /// entity from being processed more than once per frame.
    ///
    /// We use u32 IDs here to mirror the logic without needing real Entity handles.
    #[test]
    fn collision_dedup_prevents_double_shatter() {
        // Simulate two CollisionStart events both referencing asteroid A.
        let asteroid_a: u32 = 1;
        let weapon_1: u32 = 2;
        let weapon_2: u32 = 3;

        let collision_pairs = vec![(asteroid_a, weapon_1), (asteroid_a, weapon_2)];

        let mut shattered: HashSet<u32> = HashSet::new();
        let mut shatter_count = 0usize;

        for (asteroid, _weapon) in &collision_pairs {
            if shattered.insert(*asteroid) {
                shatter_count += 1; // would call shatter_asteroid here
            }
        }

        assert_eq!(shatter_count, 1, "asteroid should only be shattered once even when hit by two weapons");
    }
}

pub fn angle_indicator(maybe_angle: Option<f32>) -> f32 {
    maybe_angle
        .map(|a| {
            // Wrap to (-π, π] so that e.g. 2π is treated the same as 0.
            ((a + PI).rem_euclid(2.0 * PI) - PI).abs()
        })
        .map(|angle_error| {
            if angle_error < (PI / 2.0) {
                4.0 * (angle_error - PI / 2.0).powi(2) / PI.powi(2)
            } else {
                0.0
            }
        })
        .unwrap_or(0.0)
}
