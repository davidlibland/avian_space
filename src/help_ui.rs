use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use egui::{Align2, Color32, FontId, RichText};

use crate::PlayState;
use crate::session::SessionResourceExt;

#[derive(Resource, Default)]
pub struct HelpUiOpen {
    pub open: bool,
}

impl crate::session::SessionResource for HelpUiOpen {
    type SaveData = ();
    fn new_session(_: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
    fn from_save(_: (), _: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
}

pub fn help_ui_plugin(app: &mut App) {
    app.init_session_resource::<HelpUiOpen>()
        .add_systems(
            Update,
            toggle_help.run_if(not(in_state(PlayState::MainMenu))),
        )
        .add_systems(
            Update,
            close_help_on_main_menu.run_if(state_changed::<PlayState>),
        )
        .add_systems(
            EguiPrimaryContextPass,
            (help_corner_button, help_window)
                .run_if(not(in_state(PlayState::MainMenu))),
        );
}

fn toggle_help(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<HelpUiOpen>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    let pressed_open = keyboard.just_pressed(KeyCode::F1);
    let pressed_close = state.open && keyboard.just_pressed(KeyCode::Escape);
    if pressed_open || pressed_close {
        let next = !state.open;
        set_open(&mut state, &mut virtual_time, next);
    }
}

fn close_help_on_main_menu(
    play_state: Res<State<PlayState>>,
    mut state: ResMut<HelpUiOpen>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if *play_state.get() == PlayState::MainMenu && state.open {
        set_open(&mut state, &mut virtual_time, false);
    }
}

fn set_open(state: &mut HelpUiOpen, virtual_time: &mut Time<Virtual>, open: bool) {
    if state.open == open {
        return;
    }
    state.open = open;
    if open {
        virtual_time.pause();
    } else {
        virtual_time.unpause();
    }
}

/// Small "?" button anchored to the top-left corner. Clicking toggles the help window.
fn help_corner_button(
    mut egui_contexts: EguiContexts,
    mut state: ResMut<HelpUiOpen>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    egui::Area::new(egui::Id::new("help_corner_button"))
        .anchor(Align2::LEFT_TOP, [12.0, 12.0])
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            let label = RichText::new("?").font(FontId::proportional(18.0)).strong();
            let button = egui::Button::new(label)
                .min_size(egui::Vec2::splat(28.0))
                .fill(Color32::from_rgba_unmultiplied(0, 0, 0, 160))
                .stroke(egui::Stroke::new(1.0, Color32::from_rgb(140, 180, 220)));
            let resp = ui.add(button).on_hover_text("Help (F1)");
            if resp.clicked() {
                let next = !state.open;
                set_open(&mut state, &mut virtual_time, next);
            }
        });
}

fn help_window(
    mut egui_contexts: EguiContexts,
    mut state: ResMut<HelpUiOpen>,
    play_state: Res<State<PlayState>>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if !state.open {
        return;
    }
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    let on_surface = matches!(*play_state.get(), PlayState::Exploring);
    let mut close = false;

    egui::Window::new("Controls")
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .default_width(640.0)
        .show(ctx, |ui| {
            ui.label(
                RichText::new(if on_surface {
                    "On the planet surface"
                } else {
                    "Flying in space"
                })
                .color(Color32::from_rgb(140, 210, 255))
                .small(),
            );
            ui.separator();

            // Two-column layout: current-context controls on the left, the
            // other context on the right. The order swaps based on play state
            // so the most relevant column always reads first.
            ui.columns(2, |cols| {
                if on_surface {
                    render_section(&mut cols[0], "On Planet", &surface_controls());
                    render_section(&mut cols[1], "In Space", &flying_controls());
                } else {
                    render_section(&mut cols[0], "In Space", &flying_controls());
                    render_section(&mut cols[1], "On Planet", &surface_controls());
                }
            });

            ui.add_space(6.0);
            ui.separator();
            render_section(ui, "Universal", &universal_controls());

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Close  [F1 / Esc]").clicked() {
                    close = true;
                }
            });
        });

    if close {
        set_open(&mut state, &mut virtual_time, false);
    }
}

struct Group {
    title: &'static str,
    bindings: &'static [(&'static str, &'static str)],
}

fn render_section(ui: &mut egui::Ui, header: &str, groups: &[Group]) {
    ui.label(
        RichText::new(header)
            .color(Color32::from_rgb(255, 220, 120))
            .strong(),
    );
    ui.add_space(2.0);
    for group in groups {
        ui.label(RichText::new(group.title).color(Color32::from_rgb(200, 200, 210)));
        egui::Grid::new(format!("help_grid_{}_{}", header, group.title))
            .num_columns(2)
            .spacing([12.0, 2.0])
            .min_col_width(70.0)
            .show(ui, |ui| {
                for (key, action) in group.bindings {
                    ui.label(RichText::new(*key).monospace().color(Color32::from_rgb(180, 230, 255)));
                    ui.label(*action);
                    ui.end_row();
                }
            });
        ui.add_space(4.0);
    }
}

fn flying_controls() -> Vec<Group> {
    vec![
        Group {
            title: "Movement",
            bindings: &[
                ("↑",      "Thrust forward"),
                ("↓",      "Reverse thrust"),
                ("← / →",  "Turn left / right"),
                ("A",      "Intercept autopilot (hold)"),
            ],
        },
        Group {
            title: "Targeting",
            bindings: &[
                ("Tab",    "Cycle ships"),
                ("R",      "Nearest mission / hostile ship"),
                ("Q",      "Nearest asteroid"),
                ("P",      "Nearest pickup"),
                ("[ / ]",  "Cycle planets"),
            ],
        },
        Group {
            title: "Weapons",
            bindings: &[
                ("Space",     "Fire primary (hold)"),
                ("W",         "Cycle secondary weapon"),
                ("L-Shift",   "Fire secondary (hold)"),
            ],
        },
        Group {
            title: "Escort orders",
            bindings: &[
                ("B", "Dock — return to mother ship"),
                ("N", "Escort — stay near, disengage"),
                ("M", "Attack current target"),
            ],
        },
        Group {
            title: "Navigation",
            bindings: &[
                ("J", "Star map / hyperspace jump"),
                ("L", "Land on nearby planet"),
                ("I", "Mission log"),
            ],
        },
    ]
}

fn surface_controls() -> Vec<Group> {
    vec![
        Group {
            title: "Movement",
            bindings: &[
                ("W / ↑",  "Move forward"),
                ("S / ↓",  "Move backward"),
                ("A / ←",  "Strafe left"),
                ("D / →",  "Strafe right"),
            ],
        },
        Group {
            title: "Interaction",
            bindings: &[
                ("E",      "Interact with NPC or building"),
                ("Esc",    "Close dialogue / UI"),
            ],
        },
    ]
}

fn universal_controls() -> Vec<Group> {
    vec![Group {
        title: "",
        bindings: &[
            ("F1",  "Open / close this help"),
            ("Esc", "Close open UI, or return to main menu"),
        ],
    }]
}
