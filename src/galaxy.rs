//! Galactic control: the live faction-influence simplex per system.
//!
//! See docs/galactic_war_design.md. The static assets define the INITIAL
//! galaxy; this module owns the LIVE one, as a persisted session resource.
//! Influence moves through `CompletionEffect::ShiftInfluence` on missions
//! (watched here — missions stay decoupled, the `AdjustStanding` pattern).
//! Everything else is DERIVED:
//!
//!   * a system's *controller* (with hysteresis: gained at ≥0.6, lost <0.5),
//!   * every planet's *effective faction* = its system's controller
//!     (None while contested → markets degrade to universal-only stock),
//!   * planet market catalogs (re-derived on control change),
//!   * ship traffic (see `derive_ship_presence` in item_universe.rs —
//!     re-derived whenever influence changes anywhere).

use bevy::prelude::*;
use std::collections::HashMap;

use crate::item_universe::ItemUniverse;
use crate::missions::MissionCompleted;
use crate::missions::MissionCatalog;
use crate::missions::types::CompletionEffect;
use crate::PlayState;

// ── Tuning ───────────────────────────────────────────────────────────────────

/// A faction gains control of a system at this influence share…
pub const CONTROL_GAIN: f32 = 0.6;
/// …and only loses it below this one (hysteresis — fronts see-saw).
pub const CONTROL_KEEP: f32 = 0.5;
/// Neighbor-influence decay for ship-presence propagation (per jump).
/// Higher = enemy patrols roam deeper past the border, so fronts (and the
/// systems behind them) see more cross-faction contact.
pub const PRESENCE_LAMBDA: f32 = 0.5;

// ── Resource ─────────────────────────────────────────────────────────────────

/// Live faction influence per system: `system → faction → share`, entries in
/// [0, 1] summing to ≤ 1 (the remainder is "unaligned"). `controllers` is the
/// hysteresis memory: who currently holds each system.
#[derive(Resource, Default, Clone)]
pub struct GalaxyControl {
    pub influence: HashMap<String, HashMap<String, f32>>,
    pub controllers: HashMap<String, String>,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct GalaxyControlSave {
    pub influence: HashMap<String, HashMap<String, f32>>,
    pub controllers: HashMap<String, String>,
}

impl crate::session::SessionResource for GalaxyControl {
    type SaveData = GalaxyControlSave;
    // NB: SAVE_KEY defaults to None (ephemeral)! Without this line the whole
    // war state — every front the player moved — reset on every load.
    const SAVE_KEY: Option<&'static str> = Some("galaxy_control");
    fn new_session(iu: &ItemUniverse) -> Self {
        Self::seeded_from(iu)
    }
    fn to_save(&self) -> Self::SaveData {
        GalaxyControlSave {
            influence: self.influence.clone(),
            controllers: self.controllers.clone(),
        }
    }
    fn from_save(data: Self::SaveData, iu: &ItemUniverse) -> Self {
        // Old saves (empty map) fall back to the seeded galaxy.
        if data.influence.is_empty() {
            Self::seeded_from(iu)
        } else {
            Self {
                influence: data.influence,
                controllers: data.controllers,
            }
        }
    }
}

impl GalaxyControl {
    /// Seed from the static assets: each system starts fully held by its
    /// authored faction (explicit `faction:` or planet majority).
    pub fn seeded_from(iu: &ItemUniverse) -> Self {
        let mut g = Self::default();
        for (name, _) in &iu.star_systems {
            if ItemUniverse::TRAINING_SYSTEM_KEYS.contains(&name.as_str()) {
                continue;
            }
            if let Some(f) = ItemUniverse::static_system_faction(iu, name) {
                g.influence
                    .insert(name.clone(), HashMap::from([(f.clone(), 1.0)]));
                g.controllers.insert(name.clone(), f);
            } else {
                g.influence.insert(name.clone(), HashMap::new());
            }
        }
        g
    }

    /// Influence share of `faction` in `system` (0 if unknown).
    pub fn influence_of(&self, system: &str, faction: &str) -> f32 {
        self.influence
            .get(system)
            .and_then(|m| m.get(faction))
            .copied()
            .unwrap_or(0.0)
    }

    /// The current controller of `system`, per the hysteresis memory.
    pub fn controller(&self, system: &str) -> Option<&str> {
        self.controllers.get(system).map(String::as_str)
    }

    /// Shift `faction`'s share in `system` by `delta`, keeping the simplex
    /// valid: an increase draws first from the unaligned remainder, then
    /// proportionally from the other factions; a decrease releases share back
    /// to unaligned.
    pub fn apply_shift(&mut self, system: &str, faction: &str, delta: f32) {
        let m = self.influence.entry(system.to_string()).or_default();
        let old = m.get(faction).copied().unwrap_or(0.0);
        let new = (old + delta).clamp(0.0, 1.0);
        m.insert(faction.to_string(), new);
        let sum: f32 = m.values().sum();
        if sum > 1.0 {
            let excess = sum - 1.0;
            let others: f32 = sum - new;
            if others > 1e-6 {
                for (k, v) in m.iter_mut() {
                    if k != faction {
                        *v -= excess * (*v / others);
                    }
                }
            } else {
                m.insert(faction.to_string(), 1.0);
            }
        }
        m.retain(|_, v| *v > 1e-4);
    }

    /// Recompute a single system's controller with hysteresis. Returns the
    /// new controller if it CHANGED (Some(None) = became contested).
    #[allow(clippy::option_option)]
    fn recompute_controller(&mut self, system: &str) -> Option<Option<String>> {
        let top = self
            .influence
            .get(system)
            .and_then(|m| {
                m.iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|(f, v)| (f.clone(), *v));
        let current = self.controllers.get(system).cloned();
        let next: Option<String> = match (&current, &top) {
            // Holding: keep unless we fell below the keep threshold.
            (Some(cur), _) if self.influence_of(system, cur) >= CONTROL_KEEP => {
                Some(cur.clone())
            }
            // Lost grip (or never held): a challenger takes over only at GAIN.
            (_, Some((f, v))) if *v >= CONTROL_GAIN => Some(f.clone()),
            _ => None,
        };
        if next != current {
            match &next {
                Some(f) => {
                    self.controllers.insert(system.to_string(), f.clone());
                }
                None => {
                    self.controllers.remove(system);
                }
            }
            Some(next)
        } else {
            None
        }
    }
}

/// A planet's effective faction under live galactic control: the controller
/// of its system (None while contested). Independent enclaves stay
/// independent regardless of who holds the system.
pub fn effective_planet_faction(
    galaxy: &GalaxyControl,
    iu: &ItemUniverse,
    planet: &str,
) -> Option<String> {
    let (sys_name, pd) = iu.find_gameplay_planet(planet)?;
    if !pd.faction.is_empty() && !iu.faction_takes_sides(&pd.faction) {
        return None;
    }
    galaxy.controller(sys_name).map(String::from)
}

// ── Messages ─────────────────────────────────────────────────────────────────

/// A system's controller changed (None = became contested). Consumers
/// re-derive whatever they own: markets, standings displays, traffic.
#[derive(Event, Message, Clone, Debug)]
pub struct SystemControlChanged {
    pub system: String,
    pub controller: Option<String>,
}

// ── Systems ──────────────────────────────────────────────────────────────────

/// Apply `ShiftInfluence` completion effects (missions stay decoupled — this
/// module watches MissionCompleted, like the standing module does for
/// AdjustStanding). Shifts against non-contestable systems are ignored.
fn influence_on_mission_complete(
    mut reader: MessageReader<MissionCompleted>,
    catalog: Res<MissionCatalog>,
    iu: Res<ItemUniverse>,
    mut galaxy: ResMut<GalaxyControl>,
) {
    for MissionCompleted(id) in reader.read() {
        let Some(def) = catalog.defs.get(id) else {
            continue;
        };
        for effect in &def.completion_effects {
            if let CompletionEffect::ShiftInfluence {
                system,
                faction,
                delta,
            } = effect
            {
                let contestable = iu
                    .star_systems
                    .get(system)
                    .map(|s| s.contestable)
                    .unwrap_or(false);
                if !contestable {
                    warn!("ShiftInfluence on non-contestable system '{system}' ignored");
                    continue;
                }
                galaxy.apply_shift(system, faction, *delta);
            }
        }
    }
}

/// Recompute controllers whenever influence changed; emit SystemControlChanged
/// per flip. (Its own controller writes re-trigger it once; it settles.)
fn update_controllers(
    mut galaxy: ResMut<GalaxyControl>,
    mut changed: MessageWriter<SystemControlChanged>,
) {
    let systems: Vec<String> = galaxy.influence.keys().cloned().collect();
    for system in systems {
        if let Some(new_controller) = galaxy.recompute_controller(&system) {
            changed.write(SystemControlChanged {
                system,
                controller: new_controller,
            });
        }
    }
}

/// On a control flip: re-derive the system's planet markets under the new
/// effective faction. (Ship traffic is handled separately by
/// `rederive_traffic_on_influence_change`, which covers every influence
/// movement — flips included.)
fn rederive_on_control_change(
    mut reader: MessageReader<SystemControlChanged>,
    mut iu: ResMut<ItemUniverse>,
) {
    for SystemControlChanged { system, controller } in reader.read() {
        iu.rederive_system_market(system, controller.as_deref());
    }
}

/// Refresh derived traffic whenever influence moved (presence tracks the raw
/// simplex λ-propagated across jumps, not just controller flips).
fn rederive_traffic_on_influence_change(galaxy: Res<GalaxyControl>, mut iu: ResMut<ItemUniverse>) {
    iu.rederive_ship_presence(&galaxy);
}

/// Galactic news: surface control flips to the player through the mission
/// toast queue ("Federation seizes the Drift"). Headless-safe: the toast
/// resource only exists with the UI plugins.
fn control_change_news(
    mut reader: MessageReader<SystemControlChanged>,
    iu: Res<ItemUniverse>,
    toast: Option<ResMut<crate::missions::ui::MissionToast>>,
) {
    let Some(mut toast) = toast else {
        return;
    };
    for SystemControlChanged { system, controller } in reader.read() {
        let display = iu
            .star_systems
            .get(system)
            .map(|s| s.display_name.clone())
            .unwrap_or_else(|| system.clone());
        toast.push(match controller {
            Some(f) => format!("GALACTIC NEWS — {f} seizes control of {display}."),
            None => format!("GALACTIC NEWS — {display} descends into contested chaos."),
        });
    }
}

pub fn galaxy_plugin(app: &mut App) {
    use crate::session::SessionResourceExt;
    app.init_session_resource::<GalaxyControl>()
        .add_message::<SystemControlChanged>()
        .add_systems(
            Update,
            (
                influence_on_mission_complete,
                update_controllers.run_if(resource_changed::<GalaxyControl>),
                rederive_on_control_change,
                control_change_news,
                rederive_traffic_on_influence_change.run_if(resource_changed::<GalaxyControl>),
            )
                .chain()
                .run_if(not(in_state(PlayState::MainMenu))),
        );
}

#[cfg(test)]
#[path = "tests/galaxy_tests.rs"]
mod tests;
