//! Player-facing game settings: master volume and fullscreen, persisted to
//! `settings.yaml` beside the pilots directory. The window is reachable from
//! the main-menu footer and from the in-game help (F1) window.

use bevy::audio::{GlobalVolume, Volume};
use bevy::prelude::*;
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use serde::{Deserialize, Serialize};

#[derive(Resource, Serialize, Deserialize, Clone, PartialEq)]
pub struct GameSettings {
    /// 0.0..=1.0 — multiplies every sound via Bevy's `GlobalVolume`.
    pub master_volume: f32,
    pub fullscreen: bool,
}

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            master_volume: 1.0,
            fullscreen: false,
        }
    }
}

/// Whether the settings window is showing (toggled from menu/help).
#[derive(Resource, Default)]
pub struct SettingsUiOpen(pub bool);

fn settings_path() -> std::path::PathBuf {
    // Sibling of the pilots directory, same per-platform logic.
    crate::game_save::user_data_dir().join("settings.yaml")
}

fn load_settings() -> GameSettings {
    std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|t| serde_yaml::from_str(&t).ok())
        .unwrap_or_default()
}

fn save_settings(s: &GameSettings) {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_yaml::to_string(s)
        && let Err(e) = std::fs::write(&path, text)
    {
        warn!("failed to write settings to {path:?}: {e}");
    }
}

/// Apply the settings to the engine whenever they change (and once at
/// startup): global volume + window mode.
fn apply_settings(
    settings: Res<GameSettings>,
    mut global_volume: ResMut<GlobalVolume>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    if !settings.is_changed() {
        return;
    }
    global_volume.volume = Volume::Linear(settings.master_volume.clamp(0.0, 1.0));
    if let Ok(mut window) = windows.single_mut() {
        let want = if settings.fullscreen {
            WindowMode::BorderlessFullscreen(MonitorSelection::Current)
        } else {
            WindowMode::Windowed
        };
        if window.mode != want {
            window.mode = want;
        }
    }
}

/// The settings window itself. Saves to disk on any change.
fn settings_ui(
    mut egui_contexts: EguiContexts,
    mut open: ResMut<SettingsUiOpen>,
    mut settings: ResMut<GameSettings>,
) {
    if !open.0 {
        return;
    }
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let mut stay_open = true;
    let mut edited = settings.clone();
    egui::Window::new("Settings")
        .open(&mut stay_open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Master volume");
                ui.add(egui::Slider::new(&mut edited.master_volume, 0.0..=1.0).show_value(false));
                ui.label(format!("{:.0}%", edited.master_volume * 100.0));
            });
            ui.checkbox(&mut edited.fullscreen, "Fullscreen (borderless)");
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new("Changes apply immediately and are saved.")
                    .small()
                    .color(egui::Color32::GRAY),
            );
        });
    if edited != *settings {
        *settings = edited;
        save_settings(&settings);
    }
    if !stay_open {
        open.0 = false;
    }
}

pub fn settings_plugin(app: &mut App) {
    app.insert_resource(load_settings())
        .init_resource::<SettingsUiOpen>()
        .add_systems(Update, apply_settings)
        .add_systems(EguiPrimaryContextPass, settings_ui);
}
