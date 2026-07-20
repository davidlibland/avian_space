use crate::character_compositor::{AvatarSpec, CharacterLayers};
use crate::game_save::{Gender, PlayerGameState, list_saves, load_save};
use crate::item_universe::ItemUniverse;
use crate::session::PendingSessionLoad;
use crate::{CurrentStarSystem, PlayState};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass};

// ── Local state ───────────────────────────────────────────────────────────────

#[derive(Resource, Default)]
struct MainMenuState {
    /// Credits window open (LPC art attribution is a license obligation,
    /// so this must stay reachable from the shipped menu).
    show_credits: bool,
    new_pilot_name: String,
    new_pilot_gender: Gender,
    /// Avatar being built in the creator (None until first shown/edited).
    new_pilot_avatar: Option<AvatarSpec>,
    /// Composited preview sheet for the current avatar spec.
    preview: Option<(AvatarSpec, Handle<Image>)>,
    saves: Vec<String>,
}

/// One ◀ value ▶ cycler row. Returns true if the value changed.
fn cycle_row(
    ui: &mut egui::Ui,
    label: &str,
    current: &mut Option<String>,
    options: &[String],
    allow_none: bool,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized([64.0, 18.0], egui::Label::new(label));
        // Build the ring: [None?] + options.
        let mut ring: Vec<Option<String>> = Vec::new();
        if allow_none {
            ring.push(None);
        }
        ring.extend(options.iter().cloned().map(Some));
        if ring.is_empty() {
            return;
        }
        let idx = ring.iter().position(|o| o == current).unwrap_or(0);
        if ui.small_button("◀").clicked() {
            *current = ring[(idx + ring.len() - 1) % ring.len()].clone();
            changed = true;
        }
        let shown = current.as_deref().unwrap_or("none");
        ui.add_sized([110.0, 18.0], egui::Label::new(shown));
        if ui.small_button("▶").clicked() {
            *current = ring[(idx + 1) % ring.len()].clone();
            changed = true;
        }
    });
    changed
}

// ── Systems ───────────────────────────────────────────────────────────────────

fn refresh_saves(mut menu_state: ResMut<MainMenuState>) {
    menu_state.saves = list_saves();
}

fn main_menu_ui(
    mut commands: Commands,
    mut egui_contexts: EguiContexts,
    mut menu_state: ResMut<MainMenuState>,
    mut game_state: ResMut<PlayerGameState>,
    mut current_system: ResMut<CurrentStarSystem>,
    mut next_state: ResMut<NextState<PlayState>>,
    item_universe: Res<ItemUniverse>,
    mut layers: Option<ResMut<CharacterLayers>>,
    mut images: ResMut<Assets<Image>>,
) {
    // Keep the avatar spec in sync with the gender radios and composite the
    // 64px portrait preview (cached by spec) before borrowing the egui
    // context. Also warm the full walk-sheet cache so the surface walker
    // spawn finds it ready.
    let mut preview_tex: Option<egui::TextureId> = None;
    if let Some(layers) = layers.as_deref_mut() {
        let gender = menu_state.new_pilot_gender;
        let spec = menu_state
            .new_pilot_avatar
            .get_or_insert_with(|| AvatarSpec::for_gender(gender))
            .clone();
        let stale = menu_state
            .preview
            .as_ref()
            .map(|(s, _)| *s != spec)
            .unwrap_or(true);
        if stale {
            if let Some(handle) = layers.composite_portrait(&spec, &mut images) {
                menu_state.preview = Some((spec.clone(), handle));
            }
            layers.composite(&spec, &mut images);
        }
        if let Some((_, handle)) = &menu_state.preview {
            preview_tex =
                Some(egui_contexts.add_image(bevy_egui::EguiTextureHandle::Strong(handle.clone())));
        }
    }

    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(32.0);
            ui.heading(egui::RichText::new("AVIAN SPACE").size(48.0));
        });
        ui.add_space(24.0);

        // Credits: character art is LPC (CC-BY-SA/OGA-BY et al.) — visible
        // attribution from the shipped menu is a license obligation, not
        // just courtesy.
        egui::TopBottomPanel::bottom("menu_footer").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.small_button("Credits & Licenses").clicked() {
                    menu_state.show_credits = !menu_state.show_credits;
                }
            });
        });
        if menu_state.show_credits {
            let mut open = true;
            egui::Window::new("Credits & Licenses")
                .open(&mut open)
                .default_size([560.0, 480.0])
                .show(ui.ctx(), |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Avian Space");
                        ui.label(
                            "Code, ship/world/creature art, and weapon sound                              effects created for this game.",
                        );
                        ui.separator();
                        ui.heading("Character sprites");
                        ui.label(
                            "Composited from the Liberated Pixel Cup (LPC)                              collection — see the full per-artist list below.                              LPC art is used under its own licenses (CC0 /                              OGA-BY 3.0 / CC-BY / CC-BY-SA 3.0); those                              licenses cover the art only, not the game.",
                        );
                        ui.separator();
                        ui.heading("UI fonts");
                        ui.label(
                            "Embedded egui fonts under the SIL Open Font                              License 1.1 and the Ubuntu Font Licence 1.0.",
                        );
                        ui.separator();
                        ui.monospace(include_str!("../assets/CREDITS-SPRITES.md"));
                    });
                });
            if !open {
                menu_state.show_credits = false;
            }
        }

        // Two columns when there are saves to load; centered single panel
        // otherwise. Side-by-side keeps the avatar creator on screen without
        // pushing the load list below the fold.
        if menu_state.saves.is_empty() {
            ui.vertical_centered(|ui| {
                new_pilot_panel(
                    ui,
                    &mut menu_state,
                    &mut game_state,
                    &mut current_system,
                    &mut next_state,
                    &item_universe,
                    layers.as_deref_mut(),
                    preview_tex,
                );
            });
        } else {
            ui.columns(2, |cols| {
                cols[0].with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    new_pilot_panel(
                        ui,
                        &mut menu_state,
                        &mut game_state,
                        &mut current_system,
                        &mut next_state,
                        &item_universe,
                        layers.as_deref_mut(),
                        preview_tex,
                    );
                });
                cols[1].with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    load_pilot_panel(
                        ui,
                        &mut commands,
                        &menu_state,
                        &mut game_state,
                        &mut current_system,
                        &mut next_state,
                        &item_universe,
                    );
                });
            });
        }
    });
}

/// "New Pilot" group: name, gender, avatar creator, Create button.
#[allow(clippy::too_many_arguments)]
fn new_pilot_panel(
    ui: &mut egui::Ui,
    menu_state: &mut MainMenuState,
    game_state: &mut PlayerGameState,
    current_system: &mut CurrentStarSystem,
    next_state: &mut NextState<PlayState>,
    item_universe: &ItemUniverse,
    layers: Option<&mut CharacterLayers>,
    preview_tex: Option<egui::TextureId>,
) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_min_width(300.0);
        ui.label("New Pilot");
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut menu_state.new_pilot_name)
                    .hint_text("Pilot name…")
                    .desired_width(200.0),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Gender:");
            let before = menu_state.new_pilot_gender;
            ui.radio_value(&mut menu_state.new_pilot_gender, Gender::Boy, "Boy");
            ui.radio_value(&mut menu_state.new_pilot_gender, Gender::Girl, "Girl");
            if menu_state.new_pilot_gender != before {
                // Rebuild the avatar for the new body type.
                menu_state.new_pilot_avatar =
                    Some(AvatarSpec::for_gender(menu_state.new_pilot_gender));
            }
        });

        // ── Avatar creator ────────────────────────────────────────────────
        if let (Some(layers), Some(mut spec)) = (layers, menu_state.new_pilot_avatar.clone()) {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                // Live preview: the 64×64 native-resolution portrait
                // (down-facing idle frame), drawn at 2x.
                if let Some(tex) = preview_tex {
                    ui.add(egui::Image::new(egui::load::SizedTexture::new(
                        tex,
                        [128.0, 128.0],
                    )));
                }
                ui.vertical(|ui| {
                    let sex = spec.sex.clone();
                    let mut changed = false;
                    // Item slots (hair/beard/hat may be empty).
                    for (label, slot, allow_none) in [
                        ("Hair", "hair", true),
                        ("Beard", "beard", true),
                        ("Top", "shirt", false),
                        ("Over", "over", true),
                        ("Legs", "legs", false),
                        ("Shoes", "feet", false),
                        ("Hat", "hat", true),
                    ] {
                        if slot == "beard" && sex != "male" {
                            continue;
                        }
                        let options = layers.all_item_ids(slot, &sex);
                        if options.is_empty() {
                            continue;
                        }
                        let mut current = spec.slots.get(slot).cloned();
                        if cycle_row(ui, label, &mut current, &options, allow_none) {
                            match current {
                                Some(id) => spec.slots.insert(slot.into(), id),
                                None => spec.slots.remove(slot),
                            };
                            changed = true;
                        }
                    }
                    // Color ramps.
                    for (label, mat) in [
                        ("Skin", "body"),
                        ("Hair col", "hair"),
                        ("Eyes", "eye"),
                        ("Cloth", "cloth"),
                    ] {
                        let options: Vec<String> = layers
                            .ramp_names(mat, None)
                            .into_iter()
                            .map(String::from)
                            .collect();
                        if options.is_empty() {
                            continue;
                        }
                        let mut current = spec.colors.get(mat).cloned();
                        if cycle_row(ui, label, &mut current, &options, false) {
                            if let Some(ramp) = current {
                                spec.colors.insert(mat.into(), ramp);
                            }
                            changed = true;
                        }
                    }
                    ui.add_space(2.0);
                    if ui.button("🎲 Randomize").clicked() {
                        let mut rng = rand::thread_rng();
                        let mut random = layers.random_spec(&mut rng, "civilian");
                        // Keep the chosen gender's body type.
                        random.sex = sex.clone();
                        let head = if sex == "male" { "head_m" } else { "head_f" };
                        random.slots.insert("head".into(), head.into());
                        if sex != "male" {
                            random.slots.remove("beard");
                        }
                        spec = random;
                        changed = true;
                    }
                    if changed {
                        menu_state.new_pilot_avatar = Some(spec.clone());
                    }
                });
            });
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let ready = !menu_state.new_pilot_name.trim().is_empty();
            if ui.add_enabled(ready, egui::Button::new("Create")).clicked() {
                let name = menu_state.new_pilot_name.trim().to_string();
                let gender = menu_state.new_pilot_gender;
                *game_state = PlayerGameState::new_pilot(&name, gender, item_universe);
                if let Some(avatar) = menu_state.new_pilot_avatar.clone() {
                    game_state.avatar = avatar;
                }
                current_system.0 = game_state.current_star_system.clone();
                // No PendingSessionLoad — session resources start fresh
                // via their new_session() defaults.
                next_state.set(PlayState::Flying);
            }
        });
    });
}

/// "Load Pilot" group: scrollable save list.
fn load_pilot_panel(
    ui: &mut egui::Ui,
    commands: &mut Commands,
    menu_state: &MainMenuState,
    game_state: &mut PlayerGameState,
    current_system: &mut CurrentStarSystem,
    next_state: &mut NextState<PlayState>,
    item_universe: &ItemUniverse,
) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_min_width(300.0);
        ui.label("Load Pilot");
        ui.add_space(4.0);
        egui::ScrollArea::vertical()
            .max_height(ui.available_height() - 16.0)
            .show(ui, |ui| {
                for save_name in &menu_state.saves {
                    if ui
                        .add_sized([300.0, 28.0], egui::Button::new(save_name))
                        .clicked()
                        && let Some(save) = load_save(save_name)
                    {
                        current_system.0 = save.current_star_system.clone();
                        // Store the resources map for session resources to
                        // consume on entering Flying.
                        commands.insert_resource(PendingSessionLoad {
                            resources: save.resources.clone(),
                        });
                        *game_state = PlayerGameState::from_save(&save, item_universe);
                        // Saves must return to THE FILE we loaded: a
                        // hand-made file whose NAME differs from its
                        // pilot_name field would otherwise fork — the game
                        // writing "<pilot_name>.yaml" while the menu keeps
                        // loading the original, which never updates
                        // ("my pilot never saves").
                        game_state.pilot_name = save_name.clone();
                        next_state.set(PlayState::Flying);
                    }
                }
            });
    });
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub fn main_menu_plugin(app: &mut App) {
    app.init_resource::<MainMenuState>()
        // (Stuck-pause cleanup on menu entry is handled by sync_ui_pause +
        // close_uis_on_main_menu in main.rs — pause state is derived now.)
        .add_systems(OnEnter(PlayState::MainMenu), refresh_saves)
        .add_systems(
            EguiPrimaryContextPass,
            main_menu_ui.run_if(in_state(PlayState::MainMenu)),
        );
}
