use super::*;
use crate::consts::*;
use crate::ship::{Personality, Target};
use bevy::ecs::entity::Entity;
use std::collections::HashMap;

/// Helper: create a fake Entity from a raw index (for test purposes only).
fn fake_entity(idx: u32) -> Entity {
    Entity::from_bits(idx as u64)
}

/// Helper: zero reward array.
fn zero_rewards() -> [f32; N_REWARD_TYPES] {
    [0.0; N_REWARD_TYPES]
}

/// Helper: reward array with a specific channel set.
fn rewards_with(channel: usize, value: f32) -> [f32; N_REWARD_TYPES] {
    let mut r = zero_rewards();
    r[channel] = value;
    r
}

/// Helper: build a reward_snapshots map from a list of (entity, rewards, faction).
fn make_snapshots(
    entries: &[(Entity, [f32; N_REWARD_TYPES], Option<&str>)],
) -> HashMap<Entity, ([f32; N_REWARD_TYPES], Option<String>)> {
    entries
        .iter()
        .map(|(e, r, f)| (*e, (*r, f.map(|s| s.to_string()))))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Fighter sees allied Merchant → rewards mixed at 0.3
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fighter_sees_allied_merchant_rewards_mixed() {
    let self_e = fake_entity(1);
    let ally_e = fake_entity(2);

    let ally_rewards = rewards_with(REWARD_LANDING, 10.0);
    let snapshots = make_snapshots(&[(ally_e, ally_rewards, Some("Merchant"))]);

    let slot_targets = vec![Some(Target::Ship(ally_e)), None, None];
    let allies = vec!["Federation".to_string(), "Merchant".to_string()];

    let mut rewards = zero_rewards();
    mix_ally_rewards(
        &mut rewards,
        &slot_targets,
        &allies,
        self_e,
        &Personality::Fighter,
        &snapshots,
    );

    // Fighter α = 0.3, 1 ally → rewards[LANDING] += 0.3 * 10.0 / 1 = 3.0
    assert!(
        (rewards[REWARD_LANDING] - 3.0).abs() < 1e-5,
        "expected 3.0, got {}",
        rewards[REWARD_LANDING]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Non-ally ship visible → no sharing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn non_ally_faction_no_sharing() {
    let self_e = fake_entity(1);
    let enemy_e = fake_entity(3);

    let enemy_rewards = rewards_with(REWARD_SHIP_HIT, 5.0);
    // Pirate is NOT in Federation's allies list.
    let snapshots = make_snapshots(&[(enemy_e, enemy_rewards, Some("Pirate"))]);

    let slot_targets = vec![Some(Target::Ship(enemy_e))];
    let allies = vec!["Federation".to_string(), "Merchant".to_string()];

    let mut rewards = zero_rewards();
    mix_ally_rewards(
        &mut rewards,
        &slot_targets,
        &allies,
        self_e,
        &Personality::Fighter,
        &snapshots,
    );

    // No ally match → rewards unchanged.
    assert_eq!(rewards, zero_rewards());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: No visible allies → no sharing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn no_visible_allies_no_sharing() {
    let self_e = fake_entity(1);
    let allies = vec!["Federation".to_string(), "Merchant".to_string()];

    // slot_targets has no ships — just planets/asteroids/None.
    let slot_targets: Vec<Option<Target>> = vec![None, None, None];

    let snapshots = HashMap::new();

    let mut rewards = rewards_with(REWARD_SHIP_HIT, 1.0);
    let original = rewards;
    mix_ally_rewards(
        &mut rewards,
        &slot_targets,
        &allies,
        self_e,
        &Personality::Fighter,
        &snapshots,
    );

    assert_eq!(rewards, original);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Self entity in slot_targets → not counted
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn self_entity_excluded() {
    let self_e = fake_entity(1);

    let self_rewards = rewards_with(REWARD_LANDING, 100.0);
    let snapshots = make_snapshots(&[(self_e, self_rewards, Some("Federation"))]);

    let slot_targets = vec![Some(Target::Ship(self_e))];
    let allies = vec!["Federation".to_string()];

    let mut rewards = zero_rewards();
    mix_ally_rewards(
        &mut rewards,
        &slot_targets,
        &allies,
        self_e,
        &Personality::Fighter,
        &snapshots,
    );

    // Self is skipped → rewards unchanged.
    assert_eq!(rewards, zero_rewards());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: health_raw channel excluded from sharing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn health_raw_excluded() {
    let self_e = fake_entity(1);
    let ally_e = fake_entity(2);

    // Ally has reward ONLY in health_raw channel.
    let ally_rewards = rewards_with(REWARD_HEALTH_RAW, 50.0);
    let snapshots = make_snapshots(&[(ally_e, ally_rewards, Some("Merchant"))]);

    let slot_targets = vec![Some(Target::Ship(ally_e))];
    let allies = vec!["Federation".to_string(), "Merchant".to_string()];

    let mut rewards = zero_rewards();
    mix_ally_rewards(
        &mut rewards,
        &slot_targets,
        &allies,
        self_e,
        &Personality::Fighter,
        &snapshots,
    );

    // health_raw is excluded → all channels should remain 0.
    for ch in 0..N_REWARD_TYPES {
        assert!(
            rewards[ch].abs() < 1e-8,
            "channel {ch} should be 0, got {}",
            rewards[ch]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Personality scaling — Miner uses lower alpha
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn miner_uses_lower_alpha() {
    let self_e = fake_entity(1);
    let ally_e = fake_entity(2);

    let ally_rewards = rewards_with(REWARD_LANDING, 10.0);
    let snapshots = make_snapshots(&[(ally_e, ally_rewards, Some("Merchant"))]);

    let slot_targets = vec![Some(Target::Ship(ally_e))];
    let allies = vec!["Merchant".to_string()]; // Merchant faction (Miner is Merchant)

    let mut rewards = zero_rewards();
    mix_ally_rewards(
        &mut rewards,
        &slot_targets,
        &allies,
        self_e,
        &Personality::Miner,
        &snapshots,
    );

    // Miner α = 0.05, 1 ally → rewards[LANDING] += 0.05 * 10.0 / 1 = 0.5
    assert!(
        (rewards[REWARD_LANDING] - 0.5).abs() < 1e-5,
        "expected 0.5, got {}",
        rewards[REWARD_LANDING]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Multiple allies → rewards averaged
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn multiple_allies_averaged() {
    let self_e = fake_entity(1);
    let ally_a = fake_entity(2);
    let ally_b = fake_entity(3);

    let rewards_a = rewards_with(REWARD_LANDING, 10.0);
    let rewards_b = rewards_with(REWARD_LANDING, 20.0);
    let snapshots = make_snapshots(&[
        (ally_a, rewards_a, Some("Merchant")),
        (ally_b, rewards_b, Some("Federation")),
    ]);

    let slot_targets = vec![
        Some(Target::Ship(ally_a)),
        Some(Target::Ship(ally_b)),
    ];
    let allies = vec!["Federation".to_string(), "Merchant".to_string()];

    let mut rewards = zero_rewards();
    mix_ally_rewards(
        &mut rewards,
        &slot_targets,
        &allies,
        self_e,
        &Personality::Fighter,
        &snapshots,
    );

    // Fighter α = 0.3, 2 allies, sum = 30.0
    // rewards[LANDING] += 0.3 * 30.0 / 2 = 4.5
    assert!(
        (rewards[REWARD_LANDING] - 4.5).abs() < 1e-5,
        "expected 4.5, got {}",
        rewards[REWARD_LANDING]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Ship with no faction in snapshot → not counted as ally
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn no_faction_ship_not_ally() {
    let self_e = fake_entity(1);
    let player_e = fake_entity(99);

    // Player ship has no faction (faction = None).
    let player_rewards = rewards_with(REWARD_LANDING, 100.0);
    let snapshots = make_snapshots(&[(player_e, player_rewards, None)]);

    let slot_targets = vec![Some(Target::Ship(player_e))];
    let allies = vec!["Federation".to_string(), "Merchant".to_string()];

    let mut rewards = zero_rewards();
    mix_ally_rewards(
        &mut rewards,
        &slot_targets,
        &allies,
        self_e,
        &Personality::Fighter,
        &snapshots,
    );

    // No faction → not an ally → no sharing.
    assert_eq!(rewards, zero_rewards());
}
