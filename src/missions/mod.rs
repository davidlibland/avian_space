//! Mission system. See `README.md` in this directory for how to add new
//! objective kinds, effects, templates, or preconditions.

use bevy::prelude::*;

mod events;
mod log;
mod progress;
pub mod types;
pub mod ui;

pub use events::*;
pub use log::{MissionCatalog, MissionLog, MissionOffers, PlayerUnlocks};
pub use types::{MissionDef, MissionTemplate};
pub use ui::{missions_ui_plugin, render_bar_tab, render_missions_tab, MissionToast};

pub fn missions_plugin(app: &mut App) {
    app.init_resource::<MissionLog>()
        .init_resource::<MissionCatalog>()
        .init_resource::<MissionOffers>()
        .init_resource::<MissionToast>()
        .init_resource::<PlayerUnlocks>()
        .add_message::<PlayerLandedOnPlanet>()
        .add_message::<PlayerEnteredSystem>()
        .add_message::<PickupCollected>()
        .add_message::<AcceptMission>()
        .add_message::<DeclineMission>()
        .add_message::<AbandonMission>()
        .add_message::<MissionStarted>()
        .add_message::<MissionCompleted>()
        .add_message::<MissionFailed>()
        .add_systems(Startup, progress::init_catalog)
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
                progress::finalize_completions,
                progress::finalize_failures,
                ui::drain_completion_toasts,
            )
                .chain(),
        );
}
