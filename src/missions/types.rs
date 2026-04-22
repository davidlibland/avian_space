use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct MissionDef {
    pub briefing: String,
    pub success_text: String,
    pub failure_text: String,
    #[serde(default)]
    pub preconditions: Vec<Precondition>,
    pub offer: OfferKind,
    #[serde(default)]
    pub start_effects: Vec<StartEffect>,
    pub objective: Objective,
    /// Checked the instant the objective would otherwise complete. If any
    /// requirement fails, the mission transitions to Failed instead of
    /// Completed (with the mission's `failure_text`).
    #[serde(default)]
    pub requires: Vec<CompletionRequirement>,
    #[serde(default)]
    pub completion_effects: Vec<CompletionEffect>,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompletionRequirement {
    /// The player must be holding at least `quantity` units of `commodity`
    /// when the objective completes.
    HasCargo { commodity: String, quantity: u16 },
    /// The player must hold a given unlock when the objective completes.
    HasUnlock { name: String },
    // Future: MinCredits, Alive { ship_id }, etc.
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Precondition {
    /// Another mission must be completed before this one is offered.
    Completed { mission: String },
    /// Another mission must have been failed.
    Failed { mission: String },
    /// A named unlock flag must be set in `PlayerUnlocks`.
    HasUnlock { name: String },
    // Future: FactionStandingAtLeast { faction: String, value: i32 }, etc.
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OfferKind {
    /// Starts automatically as soon as preconditions are met.
    Auto,
    /// Appears in the "Missions" tab on any landed planet with given weight.
    Tab { weight: f32 },
    /// An NPC on the planet surface offers this mission.  The NPC either
    /// seeks the player or waits near a building.
    NpcOffer {
        planet: String,
        weight: f32,
        /// Which building the NPC starts at (e.g. "bar", "market").
        /// If absent or not present on this planet, a random building is chosen.
        #[serde(default)]
        building: Option<String>,
        /// Whether the NPC walks toward the player (`seek`) or stands
        /// still near the building (`wait`).  Defaults to `wait`.
        #[serde(default)]
        approach: NpcApproach,
    },
}

/// How a mission-giver NPC approaches the player.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NpcApproach {
    /// The NPC walks toward the player.
    Seek,
    /// The NPC stands near the building and waits.
    #[default]
    Wait,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StartEffect {
    /// Load cargo into the player's hold when the mission starts. If `reserved`
    /// is true, the cargo is locked and can't be sold or dropped by the player
    /// until the mission completes or fails.
    LoadCargo {
        commodity: String,
        quantity: u16,
        #[serde(default)]
        reserved: bool,
    },
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Objective {
    /// Player must arrive in a given star system.
    TravelToSystem { system: String },
    /// Player must land on a specific planet (system is inferred from the
    /// planet name by looking it up in the item universe).
    LandOnPlanet { planet: String },
    /// Player must collect a given quantity of a commodity from pickups in a
    /// specific star system. Purchased cargo does NOT count.
    CollectPickups {
        commodity: String,
        system: String,
        quantity: u16,
    },
    /// Player must meet an NPC on a planet surface.  The NPC either seeks
    /// the player or waits at a location.  Completes when adjacent.
    MeetNpc {
        planet: String,
        /// Display name shown in objective text (e.g. "the informant").
        npc_name: String,
        /// Which building the NPC starts near.
        #[serde(default)]
        building: Option<String>,
        /// Whether the NPC walks toward the player or waits.
        #[serde(default)]
        approach: NpcApproach,
    },
    /// Player must catch a fleeing NPC on a planet surface.  The NPC runs
    /// away; objective completes when the player gets adjacent.
    CatchNpc {
        planet: String,
        /// Display name (e.g. "the pirate").
        npc_name: String,
        /// Which building the NPC starts near.
        #[serde(default)]
        building: Option<String>,
    },
    /// Player must destroy a group of ships in a specific system. The ships
    /// are spawned when the player enters the system with this mission active
    /// and tagged with `MissionTarget`. Optionally also collect cargo from
    /// the wreckage.
    DestroyShips {
        system: String,
        /// Ship type id (key in `ItemUniverse.ships`).
        ship_type: String,
        count: u8,
        /// Display name shown in objective text (e.g. "Pirate Raiders").
        target_name: String,
        /// If true, spawned ships always target the player.
        #[serde(default)]
        hostile: bool,
        /// If set, the player must also collect this cargo from wreckage.
        #[serde(default)]
        collect: Option<CollectRequirement>,
    },
}

/// Sub-requirement for DestroyShips: also collect cargo from wreckage.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct CollectRequirement {
    pub commodity: String,
    pub quantity: u16,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompletionEffect {
    /// Remove some quantity of a commodity from the player's hold (reserved
    /// units count as removable here).
    RemoveCargo { commodity: String, quantity: u16 },
    /// Pay the player on successful completion.
    Pay { credits: i64 },
    /// Add a named unlock flag to `PlayerUnlocks` (e.g. to gate the
    /// shipyard / outfitter on story progress).
    GrantUnlock { name: String },
    // Future: AdjustFactionStanding, etc.
}

impl MissionDef {
    /// Total cargo-space units the player needs free at accept time.
    /// Sum of:
    ///   • every `LoadCargo` start effect (loaded the instant you accept)
    ///   • the target quantity of a `CollectPickups` objective (must all
    ///     fit in the hold simultaneously to complete the mission)
    pub fn required_cargo_space(&self) -> u16 {
        let loaded: u16 = self
            .start_effects
            .iter()
            .map(|e| match e {
                StartEffect::LoadCargo { quantity, .. } => *quantity,
            })
            .sum();
        let objective_room = match &self.objective {
            Objective::CollectPickups { quantity, .. } => *quantity,
            Objective::DestroyShips {
                collect: Some(req), ..
            } => req.quantity,
            _ => 0,
        };
        loaded.saturating_add(objective_room)
    }
}

/// Procedural mission "recipe" — rolled at landing time into a concrete
/// `MissionDef` with randomised commodity / planet / system / pay fields.
///
/// Template text fields support the following placeholders:
///   {commodity}, {quantity}, {pay}, {planet}, {planet_display},
///   {system}, {system_display}
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MissionTemplate {
    /// "Take X units of Y to planet Z." Destination is a random planet
    /// (other than the offer planet).
    Delivery {
        briefing: String,
        success_text: String,
        failure_text: String,
        offer: OfferKind,
        #[serde(default)]
        preconditions: Vec<Precondition>,
        /// Pool of commodity ids to pick from (one is chosen per instance).
        commodity_pool: Vec<String>,
        quantity_range: (u16, u16),
        pay_range: (i64, i64),
        #[serde(default = "default_true")]
        reserved: bool,
    },
    /// "Bring back N units of C from the asteroid fields in system S."
    /// Commodity and system are chosen from systems that have at least one
    /// asteroid field containing the commodity.
    CollectFromAsteroidField {
        briefing: String,
        success_text: String,
        failure_text: String,
        offer: OfferKind,
        #[serde(default)]
        preconditions: Vec<Precondition>,
        quantity_range: (u16, u16),
        pay_range: (i64, i64),
    },
    /// Two-stage: collect pickups from a random asteroid-field system, then
    /// deliver them to a random destination planet. Emitted as two linked
    /// missions: stage 2 auto-starts when stage 1 completes.
    CollectThenDeliver {
        stage1_briefing: String,
        stage1_success_text: String,
        stage1_failure_text: String,
        stage2_briefing: String,
        stage2_success_text: String,
        stage2_failure_text: String,
        offer: OfferKind,
        #[serde(default)]
        preconditions: Vec<Precondition>,
        quantity_range: (u16, u16),
        /// Pay on final delivery (stage 2).
        pay_range: (i64, i64),
    },
    /// "Destroy N ships of type T in a random system." Spawns hostile
    /// mission targets. System chosen from systems that contain at least
    /// one planet (so the player can land afterward).
    BountyHunt {
        briefing: String,
        success_text: String,
        failure_text: String,
        offer: OfferKind,
        #[serde(default)]
        preconditions: Vec<Precondition>,
        /// Pool of ship types to spawn as targets.
        ship_type_pool: Vec<String>,
        count_range: (u8, u8),
        pay_range: (i64, i64),
        /// Display name for targets in the HUD / objective text.
        target_name: String,
    },
    /// Three-stage thief pursuit:
    /// 1. Destroy the thief's ship + collect stolen goods.
    /// 2. Deliver the goods to a planet.
    /// 3. Catch the fleeing thief (who escaped via pod) on a nearby planet.
    CatchThief {
        stage1_briefing: String,
        stage1_success_text: String,
        stage1_failure_text: String,
        stage2_briefing: String,
        stage2_success_text: String,
        stage2_failure_text: String,
        stage3_briefing: String,
        stage3_success_text: String,
        stage3_failure_text: String,
        offer: OfferKind,
        #[serde(default)]
        preconditions: Vec<Precondition>,
        ship_type_pool: Vec<String>,
        target_name: String,
        commodity_pool: Vec<String>,
        quantity_range: (u16, u16),
        pay_range: (i64, i64),
    },
}

impl MissionTemplate {
    pub fn preconditions(&self) -> &[Precondition] {
        match self {
            MissionTemplate::Delivery { preconditions, .. } => preconditions,
            MissionTemplate::CollectFromAsteroidField { preconditions, .. } => preconditions,
            MissionTemplate::CollectThenDeliver { preconditions, .. } => preconditions,
            MissionTemplate::BountyHunt { preconditions, .. } => preconditions,
            MissionTemplate::CatchThief { preconditions, .. } => preconditions,
        }
    }
}

fn default_true() -> bool {
    true
}


#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum MissionStatus {
    #[default]
    Locked,
    Available,
    Active(ObjectiveProgress),
    Completed,
    Failed,
}

/// Per-objective progress counters. Each field is only meaningful for
/// objectives that use it; others stay at 0.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ObjectiveProgress {
    #[serde(default)]
    pub collected: u16,
    #[serde(default)]
    pub destroyed: u8,
}

/// Marker component attached to ships spawned for a mission's `DestroyShips`
/// objective. The missions module owns this component; nothing outside
/// `missions/` needs to read or write it.
#[derive(Component, Clone, Debug)]
pub struct MissionTarget {
    pub mission_id: String,
    pub display_name: String,
    pub always_targets_player: bool,
}
