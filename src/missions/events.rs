use bevy::prelude::*;

// ── General gameplay events ────────────────────────────────────────────────
// These describe things that happened in the game world. They are emitted by
// non-mission modules (planets, main, pickups) and consumed here. Keeping them
// generic means non-mission code never needs to know about missions.

#[derive(Event, Message, Clone, Debug)]
pub struct PlayerLandedOnPlanet {
    pub planet: String,
}

#[derive(Event, Message, Clone, Debug)]
pub struct PlayerEnteredSystem {
    pub system: String,
}

#[derive(Event, Message, Clone, Debug)]
pub struct PickupCollected {
    pub commodity: String,
    pub quantity: u16,
    pub system: String,
}

/// Emitted from `ship.rs::apply_damage` when any AI ship's health reaches 0.
#[derive(Event, Message, Clone, Debug)]
pub struct ShipDestroyed {
    pub entity: Entity,
}

// ── Mission-specific events (UI <-> logic) ─────────────────────────────────

#[derive(Event, Message, Clone, Debug)]
pub struct AcceptMission(pub String);

#[derive(Event, Message, Clone, Debug)]
pub struct DeclineMission(pub String);

#[derive(Event, Message, Clone, Debug)]
pub struct AbandonMission(pub String);

#[derive(Event, Message, Clone, Debug)]
pub struct MissionStarted(pub String);

#[derive(Event, Message, Clone, Debug)]
pub struct MissionCompleted(pub String);

#[derive(Event, Message, Clone, Debug)]
pub struct MissionFailed(pub String);
