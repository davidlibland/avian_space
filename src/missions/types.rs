use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct MissionDef {
    pub briefing: String,
    pub success_text: String,
    pub failure_text: String,
    /// Faction this storyline belongs to (key into factions.yaml). Colors
    /// the mission in the story chart; None = neutral/unaffiliated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faction: Option<String>,
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
    /// Friendly support wing spawned for the player in the mission's battle
    /// system while the mission is active (squadron escorts: they follow,
    /// take B/N/M orders, cannot dock, despawn when the mission ends).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub squadron: Vec<String>,
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
        /// Recurring character id (key in `assets/npcs.yaml`). Gives the
        /// giver a consistent name + appearance across a storyline.
        #[serde(default)]
        npc: Option<String>,
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
        /// Recurring character id (key in `assets/npcs.yaml`) for a
        /// consistent name + appearance.
        #[serde(default)]
        npc: Option<String>,
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
        /// Recurring character id (key in `assets/npcs.yaml`).
        #[serde(default)]
        npc: Option<String>,
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
    /// Shift the player's standing with a faction (applied by the standing
    /// module, which watches MissionCompleted — missions stay decoupled).
    AdjustStanding { faction: String, delta: f32 },
    /// Shift a faction's influence share in a (contestable) system — the
    /// galactic-war lever. Applied by the galaxy module, same pattern.
    ShiftInfluence {
        system: String,
        faction: String,
        delta: f32,
    },
}

impl MissionDef {
    /// The system this mission's ShiftInfluence effect targets, if any
    /// (used by the war generator to avoid stacking missions on one front).
    pub fn shift_target(&self) -> Option<&str> {
        self.completion_effects.iter().find_map(|e| match e {
            CompletionEffect::ShiftInfluence { system, .. } => Some(system.as_str()),
            _ => None,
        })
    }

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
/// What a covert-ops mission asks of the player (see MissionTemplate::Covert).
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CovertAction {
    /// Deliver contraband to a landable enemy world.
    Smuggle { commodity: String, quantity: u16 },
    /// Meet a contact on an enemy world (they seek you out).
    MeetContact { npc_name: String },
    /// Catch a fleeing official on an enemy world.
    CatchOfficial { npc_name: String },
    /// Destroy the enemy's supply freighters in the target system — an
    /// influence shift without a battle, at a cost in Merchant standing.
    CutSupply { count: u8 },
    /// Meet the official's bagman on an enemy world and pay them off. The
    /// template's pay_range is NEGATIVE: the bribe comes out of your pocket.
    Bribe { npc_name: String },
    /// Two-stage: meet someone on an enemy world, then carry them (or what
    /// they know) home — the follow-up mission auto-starts on the first
    /// meet and completes on landing at the sponsor's front system.
    Extract { npc_name: String },
    /// Two-stage: smuggle a cargo onto an enemy world, then meet the cell
    /// organizer who takes delivery — the meet auto-starts on landing.
    Propaganda {
        commodity: String,
        quantity: u16,
        npc_name: String,
    },
}

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
    /// A galactic-war battle mission, generated ad-hoc by the front
    /// generator (src/war.rs) at faction worlds bordering enemy space —
    /// never offered by the landing rolls. Substitution vars: {faction}
    /// {enemy} {system_display} {count} {pay}.
    /// `attack: true` fights in the ENEMY's system, else defends the
    /// sponsor's own; `squadron_size > 0` musters a support wing of the
    /// sponsor's fighters. `tier` picks the mission by how contested the
    /// front is (1 = lopsided → raids, 3 = near the threshold → decisive).
    War {
        briefing: String,
        success_text: String,
        failure_text: String,
        attack: bool,
        #[serde(default)]
        squadron_size: u8,
        count_range: (u8, u8),
        influence_delta: f32,
        pay_range: (i64, i64),
        min_standing: f32,
        tier: u8,
    },
    /// A battleless covert-ops mission that shifts system loyalty: smuggle
    /// arms to partisans, meet a dissident cell, or snatch an official from
    /// an ENEMY world near the front. Same generator/vars as `War`, plus
    /// {planet_display} {commodity} {quantity}. Executing it costs standing
    /// with the target faction (`enemy_standing_penalty`).
    Covert {
        briefing: String,
        success_text: String,
        failure_text: String,
        action: CovertAction,
        influence_delta: f32,
        pay_range: (i64, i64),
        min_standing: f32,
        tier: u8,
        #[serde(default)]
        enemy_standing_penalty: f32,
        /// Standing cost with the Merchant guild (CutSupply's moral price).
        #[serde(default)]
        merchant_standing_penalty: f32,
        /// Stage-2 texts for two-stage actions (Extract, Propaganda).
        #[serde(default)]
        stage2_briefing: String,
        #[serde(default)]
        stage2_success: String,
    },
    /// The arrest flow generated by the standing system when the player lands
    /// on a world whose faction they've antagonized past the arrest
    /// threshold. Never offered by the landing rolls — `standing.rs`
    /// instantiates it ad-hoc, substituting {faction}, {fine},
    /// {planet_display}, {dest_display}, {count}, {quantity}, {commodity}.
    /// The meet stage (enforcers seek the player) opens three parallel
    /// resolution missions: pay the fine, fly a penal bounty, or run a
    /// community-service delivery. Completing any one closes the case.
    Arrest {
        meet_briefing: String,
        meet_success: String,
        meet_failure: String,
        fine_briefing: String,
        fine_success: String,
        fine_failure: String,
        bounty_briefing: String,
        bounty_success: String,
        bounty_failure: String,
        service_briefing: String,
        service_success: String,
        service_failure: String,
        /// Fine = fine_base + fine_per_standing × |standing|.
        fine_base: i64,
        fine_per_standing: i64,
        bounty_count: u8,
        service_commodity: String,
        service_quantity: u16,
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
            MissionTemplate::Arrest { .. } => &[],
            MissionTemplate::War { .. } | MissionTemplate::Covert { .. } => &[],
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
    /// Whether this mission's support squadron has already mustered
    /// (never respawned — losses are real).
    #[serde(default)]
    pub squadron_spawned: bool,
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
