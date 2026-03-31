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
