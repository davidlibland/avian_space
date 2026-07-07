//! Mission system. See `README.md` in this directory for how to add new
//! objective kinds, effects, templates, or preconditions.

use bevy::prelude::*;

use crate::PlayState;
use crate::session::SessionResourceExt;

#[cfg(test)]
mod tests;

mod events;
mod log;
pub(crate) mod progress;
pub mod types;
pub mod ui;

pub use events::*;
pub use log::{
    MissionCatalog, MissionCatalogSave, MissionLog, MissionLogSave, MissionOffers, PlayerUnlocks,
};
pub use types::{MissionDef, MissionStatus, MissionTarget, MissionTemplate, NpcApproach, Objective, OfferKind};
pub use ui::{missions_ui_plugin, render_missions_tab};

pub fn missions_plugin(app: &mut App) {
    app.init_session_resource::<MissionLog>()
        .init_session_resource::<MissionCatalog>()
        .init_session_resource::<MissionOffers>()
        .init_session_resource::<PlayerUnlocks>()
        .add_message::<PlayerLandedOnPlanet>()
        // spawn_mission_targets writes arrival flashes; registered here (as
        // well as in explosions_plugin) so headless worlds stay self-contained.
        .add_message::<crate::explosions::TriggerJumpFlash>()
        .add_message::<PlayerEnteredSystem>()
        .add_message::<PickupCollected>()
        .add_message::<ShipDestroyed>()
        .add_message::<AcceptMission>()
        .add_message::<DeclineMission>()
        .add_message::<AbandonMission>()
        .add_message::<MissionStarted>()
        .add_message::<MissionCompleted>()
        .add_message::<MissionFailed>()
        .add_message::<NpcMet>()
        .add_message::<NpcCaught>()
        .add_systems(
            Update,
            (
                // Only re-scan the catalog when something that can flip a
                // status actually changed — every other frame this is a no-op
                // poll. (Its own log writes re-trigger it once, letting the
                // Available/Auto cascade settle across frames.)
                progress::update_locked_to_available.run_if(
                    resource_changed::<MissionLog>
                        .or(resource_changed::<MissionCatalog>)
                        .or(resource_changed::<PlayerUnlocks>),
                ),
                progress::handle_ui_actions,
                progress::apply_start_effects,
                progress::roll_offers_on_land,
                // New offers can only appear when a status or the offer set
                // changed; don't scan the catalog every frame while landed.
                progress::roll_new_offers_while_landed.run_if(
                    resource_changed::<MissionLog>.or(resource_changed::<MissionOffers>),
                ),
                progress::advance_travel_objectives,
                progress::advance_land_objectives,
                progress::advance_collect_objectives,
                progress::advance_destroy_objectives,
                progress::advance_destroy_collect,
                progress::advance_meet_npc_objectives,
                progress::advance_catch_npc_objectives,
                progress::finalize_completions,
                progress::finalize_failures,
                progress::despawn_targets_on_failure,
            )
                .chain()
                // No missions logic in the main menu: there is no player, so an
                // Auto mission started there would silently drop its cargo
                // start-effects (MissionStarted consumed with no ship to load).
                .run_if(not(in_state(PlayState::MainMenu))),
        )
        // Spawn/force-target run every Flying frame, outside the event chain.
        .add_systems(
            Update,
            (
                progress::spawn_mission_targets,
                progress::force_target_player,
            )
                .run_if(in_state(PlayState::Flying)),
        );
}
