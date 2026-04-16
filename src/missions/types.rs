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
    /// Appears in the Bar on a specific planet with given weight.
    Bar { planet: String, weight: f32 },
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
    // Future: DestroyShip { mission_target_tag: String }, etc.
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
}

impl MissionTemplate {
    pub fn preconditions(&self) -> &[Precondition] {
        match self {
            MissionTemplate::Delivery { preconditions, .. } => preconditions,
            MissionTemplate::CollectFromAsteroidField { preconditions, .. } => preconditions,
            MissionTemplate::CollectThenDeliver { preconditions, .. } => preconditions,
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
}
