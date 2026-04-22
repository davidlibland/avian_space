//! Mission system. See `README.md` in this directory for how to add new
//! objective kinds, effects, templates, or preconditions.

use bevy::prelude::*;

use crate::PlayState;
use crate::session::SessionResourceExt;

#[cfg(test)]
mod tests;

mod events;
mod log;
mod progress;
pub mod types;
pub mod ui;

pub use events::*;
pub use log::{
    MissionCatalog, MissionCatalogSave, MissionLog, MissionLogSave, MissionOffers, PlayerUnlocks,
};
pub use types::{MissionDef, MissionTarget, MissionTemplate, NpcApproach, OfferKind};
pub use ui::{missions_ui_plugin, render_bar_tab, render_missions_tab};

pub fn missions_plugin(app: &mut App) {
    app.init_session_resource::<MissionLog>()
        .init_session_resource::<MissionCatalog>()
        .init_session_resource::<MissionOffers>()
        .init_session_resource::<PlayerUnlocks>()
        .add_message::<PlayerLandedOnPlanet>()
        .add_message::<PlayerEnteredSystem>()
        .add_message::<PickupCollected>()
        .add_message::<ShipDestroyed>()
        .add_message::<AcceptMission>()
        .add_message::<DeclineMission>()
        .add_message::<AbandonMission>()
        .add_message::<MissionStarted>()
        .add_message::<MissionCompleted>()
        .add_message::<MissionFailed>()
        .add_systems(
            Update,
            (
                progress::update_locked_to_available,
                progress::handle_ui_actions,
                progress::apply_start_effects,
                progress::roll_offers_on_land,
                progress::advance_travel_objectives,
                progress::advance_land_objectives,
                progress::advance_collect_objectives,
                progress::advance_destroy_objectives,
                progress::advance_destroy_collect,
                progress::finalize_completions,
                progress::finalize_failures,
                progress::despawn_targets_on_failure,
                ui::drain_completion_toasts,
            )
                .chain(),
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
