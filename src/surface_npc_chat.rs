//! NPC dialogue / interaction system for the planet surface.
//!
//! When the player presses E near an NPC, this module opens a dialogue
//! window.  The content depends on the NPC's current behavior:
//!
//! - **OfferMission** → mission briefing with Accept/Decline buttons
//! - **AwaitPlayer** → objective-completion dialogue
//! - **SeekPlayer** → "I need to talk to you!" then pops to next behavior
//! - **Patrol** → generic flavour dialogue
//! - **FleePlayer / Despawn** → no interaction (ignored)
//!
//! This replaces the old `npc_interact` + `npc_mission_offer_ui` systems
//! in `surface_npc.rs`.

use bevy::prelude::*;
use bevy_egui::EguiContexts;

use crate::surface::{Walker, BuildingKind, ActiveBuildingUI, TILE_PX};
use crate::surface_npc::{Npc, NpcBehavior, Behavior};

// ── Chat state ───────────────────────────────────────────────────────────

/// What the dialogue window is currently showing.
#[derive(Default, Clone)]
enum ChatContent {
    #[default]
    None,
    /// Generic dialogue lines (cycle through with Continue).
    Dialogue {
        lines: Vec<String>,
        current: usize,
    },
    /// Mission offer with Accept/Decline.
    MissionOffer {
        mission_id: String,
    },
    /// Objective completed — brief acknowledgment.
    ObjectiveComplete {
        message: String,
    },
}

/// The current NPC chat state.  Replaces `ActiveNpcInteraction`.
#[derive(Resource, Default)]
pub struct NpcChatState {
    /// Entity being chatted with.
    pub entity: Option<Entity>,
    content: ChatContent,
}

// ── Constants ────────────────────────────────────────────────────────────

const ADJACENT_DIST: f32 = 1.5 * TILE_PX;

// ── Interaction system ───────────────────────────────────────────────────

/// Handle E-press near NPCs.  Opens dialogue based on front behavior.
/// NPC interaction takes priority over building interaction.
pub fn npc_chat_interact(
    keyboard: Res<ButtonInput<KeyCode>>,
    walker_q: Query<&Transform, With<Walker>>,
    mut npcs: Query<(Entity, &mut NpcBehavior, &Transform), (With<Npc>, Without<Walker>)>,
    mut chat: ResMut<NpcChatState>,
    mut active_building_ui: ResMut<ActiveBuildingUI>,
    game_state: Res<crate::game_save::PlayerGameState>,
    landed_context: Res<crate::planet_ui::LandedContext>,
    mut npc_met_writer: MessageWriter<crate::missions::NpcMet>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    // Escape closes chat.
    if keyboard.just_pressed(KeyCode::Escape) && chat.entity.is_some() {
        chat.entity = None;
        chat.content = ChatContent::None;
        sfx_writer.write(crate::sfx::SurfaceSfx::UiClose);
        return;
    }

    if !keyboard.just_pressed(KeyCode::KeyE) {
        return;
    }

    // If already chatting, close.
    if chat.entity.is_some() {
        chat.entity = None;
        chat.content = ChatContent::None;
        sfx_writer.write(crate::sfx::SurfaceSfx::UiClose);
        return;
    }

    let Ok(walker_tf) = walker_q.single() else { return };
    let wp = walker_tf.translation.truncate();
    let planet = landed_context.planet_name.clone().unwrap_or_default();
    let pilot = &game_state.pilot_name;

    // Find the nearest adjacent NPC.
    let mut best: Option<(Entity, f32)> = None;
    for (entity, _, tf) in &npcs {
        let dist = (tf.translation.truncate() - wp).length();
        if dist < ADJACENT_DIST {
            if best.map_or(true, |(_, d)| dist < d) {
                best = Some((entity, dist));
            }
        }
    }

    let Some((npc_entity, _)) = best else { return };
    let Ok((_, mut npc, _)) = npcs.get_mut(npc_entity) else { return };

    // Determine content from front behavior.
    let content = match npc.queue.front() {
        Some(Behavior::OfferMission { mission_id }) => {
            ChatContent::MissionOffer { mission_id: mission_id.clone() }
        }
        Some(Behavior::AwaitPlayer { mission_id }) => {
            // Complete the objective immediately.
            npc_met_writer.write(crate::missions::NpcMet {
                planet: planet.clone(),
                mission_id: mission_id.clone(),
            });
            let msg = format!("You must be {}. Glad I found you.", pilot);
            npc.queue.pop_front();
            ChatContent::ObjectiveComplete { message: msg }
        }
        Some(Behavior::SeekPlayer { .. }) => {
            // Pop SeekPlayer to reveal the next behavior (usually OfferMission).
            npc.queue.pop_front();
            // Check what's next.
            match npc.queue.front() {
                Some(Behavior::OfferMission { mission_id }) => {
                    ChatContent::MissionOffer { mission_id: mission_id.clone() }
                }
                Some(Behavior::AwaitPlayer { mission_id }) => {
                    npc_met_writer.write(crate::missions::NpcMet {
                        planet: planet.clone(),
                        mission_id: mission_id.clone(),
                    });
                    let msg = format!("Hey {}! Been looking all over for you.", pilot);
                    npc.queue.pop_front();
                    ChatContent::ObjectiveComplete { message: msg }
                }
                _ => {
                    ChatContent::Dialogue {
                        lines: vec!["Wait — I needed to tell you something... never mind.".into()],
                        current: 0,
                    }
                }
            }
        }
        Some(Behavior::Patrol { waypoints, .. }) => {
            let lines = generate_patrol_dialogue(pilot, waypoints);
            ChatContent::Dialogue { lines, current: 0 }
        }
        Some(Behavior::FleePlayer { .. }) | Some(Behavior::Despawn { .. }) | None => {
            // Can't chat with fleeing/despawning NPCs.
            return;
        }
    };

    // Close any open building UI — NPC chat takes priority.
    active_building_ui.0 = None;

    chat.entity = Some(npc_entity);
    chat.content = content;
    sfx_writer.write(crate::sfx::SurfaceSfx::UiOpen);
}

// ── Dialogue UI ──────────────────────────────────────────────────────────

/// Render the NPC chat egui window.
pub fn npc_chat_ui(
    mut egui_contexts: EguiContexts,
    mut chat: ResMut<NpcChatState>,
    catalog: Res<crate::missions::MissionCatalog>,
    player_q: Query<&crate::ship::Ship, With<Walker>>,
    mut accept_writer: MessageWriter<crate::missions::AcceptMission>,
    mut decline_writer: MessageWriter<crate::missions::DeclineMission>,
    mut npcs: Query<&mut NpcBehavior, With<Npc>>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    if chat.entity.is_none() { return }
    let Ok(ctx) = egui_contexts.ctx_mut() else { return };

    let mut close = false;

    bevy_egui::egui::Window::new("Conversation")
        .collapsible(false)
        .resizable(true)
        .anchor(bevy_egui::egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            match &mut chat.content {
                ChatContent::None => {
                    close = true;
                }

                ChatContent::Dialogue { lines, current } => {
                    if let Some(line) = lines.get(*current) {
                        ui.label(line.as_str());
                    }
                    ui.separator();
                    if *current + 1 < lines.len() {
                        if ui.button("Continue").clicked() {
                            *current += 1;
                        }
                    } else {
                        if ui.button("Goodbye").clicked() {
                            close = true;
                        }
                    }
                }

                ChatContent::MissionOffer { mission_id } => {
                    let mission_id = mission_id.clone();
                    if let Some(def) = catalog.defs.get(&mission_id) {
                        ui.label(&def.briefing);
                        let required = def.required_cargo_space();
                        let free = player_q.single().map(|s| s.remaining_cargo_space()).unwrap_or(0);
                        if required > 0 {
                            ui.label(format!(
                                "Cargo required: {} units (you have {} free)",
                                required, free
                            ));
                        }
                        let has_space = free >= required;
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.add_enabled_ui(has_space, |ui| {
                                let btn = ui.button("Accept");
                                if btn.clicked() {
                                    accept_writer.write(crate::missions::AcceptMission(mission_id.clone()));
                                    if let Some(npc_e) = chat.entity {
                                        if let Ok(mut npc) = npcs.get_mut(npc_e) {
                                            npc.queue.pop_front();
                                        }
                                    }
                                    close = true;
                                    sfx_writer.write(crate::sfx::SurfaceSfx::UiButton);
                                }
                                if !has_space {
                                    btn.on_hover_text("Not enough free cargo space.");
                                }
                            });
                            if ui.button("Decline").clicked() {
                                decline_writer.write(crate::missions::DeclineMission(mission_id.clone()));
                                if let Some(npc_e) = chat.entity {
                                    if let Ok(mut npc) = npcs.get_mut(npc_e) {
                                        npc.queue.pop_front();
                                    }
                                }
                                close = true;
                                sfx_writer.write(crate::sfx::SurfaceSfx::UiButton);
                            }
                        });
                    } else {
                        ui.label("(This person has nothing to say.)");
                        if ui.button("Goodbye").clicked() {
                            close = true;
                        }
                    }
                }

                ChatContent::ObjectiveComplete { message } => {
                    ui.label(message.as_str());
                    ui.separator();
                    if ui.button("Goodbye").clicked() {
                        close = true;
                    }
                }
            }
        });

    if close {
        chat.entity = None;
        chat.content = ChatContent::None;
    }
}

// ── Dialogue generation ──────────────────────────────────────────────────

fn generate_patrol_dialogue(pilot_name: &str, _waypoints: &[(u32, u32)]) -> Vec<String> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let greetings = [
        format!("Morning, {}!", pilot_name),
        format!("Hey there, {}.", pilot_name),
        "Hello!".to_string(),
        "Nice to see a new face around here.".to_string(),
    ];

    let remarks = [
        "Just heading to the next building over.",
        "Quiet day today. I like it.",
        "Watch your step out there — the terrain can be rough.",
        "I heard a ship landed earlier. Must be you!",
        "The views on this planet never get old.",
        "Business has been slow lately.",
        "If you're looking for work, check the mission board.",
        "Safe travels, pilot.",
    ];

    let greeting = greetings[rng.r#gen_range(0..greetings.len())].clone();
    let remark = remarks[rng.r#gen_range(0..remarks.len())].to_string();

    vec![greeting, remark]
}
