//! Walking, interaction (E), sounds, and the camera.
#[allow(unused_imports)]
use super::*;

use crate::PlayState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// WASD / arrow key movement for the walker. Frozen when a building UI is open.
/// Speed is divided by the current terrain's movement cost.
pub(crate) fn walker_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut walkers: Query<&mut LinearVelocity, With<Walker>>,
    active_ui: Res<ActiveBuildingUI>,
    terrain_speed: Res<TerrainSpeedModifier>,
) {
    let Ok(mut vel) = walkers.single_mut() else {
        return;
    };

    // Don't move while a building UI is open.
    if active_ui.0.is_some() {
        vel.0 = Vec2::ZERO;
        return;
    }

    let mut dir = Vec2::ZERO;
    if keyboard.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]) {
        dir.y += 1.0;
    }
    if keyboard.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]) {
        dir.y -= 1.0;
    }
    if keyboard.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]) {
        dir.x -= 1.0;
    }
    if keyboard.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]) {
        dir.x += 1.0;
    }

    // Hold Shift to run (uses the run animation cycle past 80 u/s).
    let base = if keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        RUN_SPEED
    } else {
        WALK_SPEED
    };
    let speed = base / terrain_speed.0;
    vel.0 = dir.normalize_or_zero() * speed;
}

// Walker animation is handled by the shared `animate_characters` system
// in surface_character.rs.  The walker just needs `CharacterAnim` +
// `LinearVelocity` + `Sprite` — same as civilians.

/// Play a footstep sound when the walker's animation frame advances.
/// Play door.ogg when the walker's depth crosses a door sprite's depth
/// (i.e. the walker visually passes in front of / behind the door).
/// Play ui_open/ui_close when ActiveBuildingUI changes, and ui_button
/// on mouse clicks while a building UI is open.
pub(crate) fn egui_button_click_sound(
    active_ui: Res<ActiveBuildingUI>,
    mut prev_ui: Local<Option<BuildingKind>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    let current = active_ui.0;

    // Detect open/close transitions.
    if active_ui.is_changed() {
        match (*prev_ui, current) {
            (None, Some(_)) => {
                sfx_writer.write(crate::sfx::SurfaceSfx::UiOpen);
            }
            (Some(_), None) => {
                sfx_writer.write(crate::sfx::SurfaceSfx::UiClose);
            }
            _ => {}
        }
        *prev_ui = current;
    }

    // Play button click sound on any mouse press while a UI is open.
    if current.is_some() && mouse.just_released(MouseButton::Left) {
        sfx_writer.write(crate::sfx::SurfaceSfx::UiButton);
    }
}

pub(crate) fn door_depth_sound(
    walker_q: Query<&Transform, With<Walker>>,
    mut doors: Query<(Entity, &Transform, &mut DoorSprite), Without<Walker>>,
    nearby: Res<NearbyBuilding>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    let Ok(walker_tf) = walker_q.single() else {
        return;
    };
    let walker_z = walker_tf.translation.z;
    let nearby_entity = nearby.current.map(|(e, _)| e);

    for (entity, door_tf, mut door) in &mut doors {
        let door_z = door_tf.translation.z;
        let walker_behind = walker_z > door_z; // higher z = behind

        if let Some(was_behind) = door.walker_was_behind {
            // Only play if depth flipped AND walker is colliding with this door.
            if was_behind != walker_behind && nearby_entity == Some(entity) {
                sfx_writer.write(crate::sfx::SurfaceSfx::Door);
            }
        }
        door.walker_was_behind = Some(walker_behind);
    }
}

pub(crate) fn play_footstep(
    walkers: Query<(&CharacterAnim, &Transform), With<Walker>>,
    footstep_data: Res<FootstepData>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    let Ok((anim, tf)) = walkers.single() else {
        return;
    };

    if !anim.just_stepped {
        return;
    }
    if footstep_data.terrains.is_empty() || footstep_data.map_w == 0 {
        return;
    }

    let tile_px = TILE_PX;
    let tx = ((tf.translation.x / tile_px) + footstep_data.map_w as f32 / 2.0) as u32;
    let ty = ((tf.translation.y / tile_px) + footstep_data.map_h as f32 / 2.0) as u32;
    let tx = tx.min(footstep_data.map_w.saturating_sub(1));
    let ty = ty.min(footstep_data.map_h.saturating_sub(1));
    let idx = (ty * footstep_data.map_w + tx) as usize;
    let terrain_idx = footstep_data.terrain_map.get(idx).copied().unwrap_or(0) as usize;
    let (surface, volume) = footstep_data
        .terrains
        .get(terrain_idx)
        .map(|(s, v)| (s.clone(), *v))
        .unwrap_or(("dull".into(), 0.3));

    sfx_writer.write(crate::sfx::SurfaceSfx::Footstep { surface, volume });
}

// ---------------------------------------------------------------------------
// Building Interaction
// ---------------------------------------------------------------------------

/// Track which building the walker overlaps.  Uses an overlap count so
/// that exiting one of two adjacent sensor tiles (e.g. mechanic garage
/// doors) doesn't prematurely clear the state.
pub(crate) fn track_nearby_building(
    mut collision_starts: MessageReader<CollisionStart>,
    mut collision_ends: MessageReader<CollisionEnd>,
    buildings: Query<&Building>,
    walkers: Query<(), With<Walker>>,
    mut nearby: ResMut<NearbyBuilding>,
) {
    for event in collision_starts.read() {
        let (a, b) = (event.collider1, event.collider2);
        if let Some((bldg_entity, bldg)) = match (
            buildings.get(a).ok(),
            buildings.get(b).ok(),
            walkers.get(a).ok(),
            walkers.get(b).ok(),
        ) {
            (Some(bldg), _, _, Some(_)) => Some((a, bldg)),
            (_, Some(bldg), Some(_), _) => Some((b, bldg)),
            _ => None,
        } {
            nearby.current = Some((bldg_entity, bldg.kind));
            nearby.overlap_count += 1;
        }
    }
    for event in collision_ends.read() {
        let (a, b) = (event.collider1, event.collider2);
        let involves_building = buildings.contains(a) || buildings.contains(b);
        let involves_walker = walkers.contains(a) || walkers.contains(b);
        if involves_building && involves_walker {
            nearby.overlap_count = nearby.overlap_count.saturating_sub(1);
            if nearby.overlap_count == 0 {
                nearby.current = None;
            }
        }
    }
}

/// Track the terrain movement cost under the walker.  When the walker
/// enters/exits `TerrainSensor` zones the speed modifier updates.
pub(crate) fn track_terrain_speed(
    mut collision_starts: MessageReader<CollisionStart>,
    mut collision_ends: MessageReader<CollisionEnd>,
    sensors: Query<&crate::world_assets::TerrainSensor>,
    walkers: Query<(), With<Walker>>,
    mut modifier: ResMut<TerrainSpeedModifier>,
) {
    for event in collision_starts.read() {
        let (a, b) = (event.collider1, event.collider2);
        let sensor = sensors.get(a).ok().or_else(|| sensors.get(b).ok());
        let is_walker = walkers.contains(a) || walkers.contains(b);
        if let (Some(ts), true) = (sensor, is_walker) {
            modifier.0 = ts.movement_cost;
        }
    }
    for event in collision_ends.read() {
        let (a, b) = (event.collider1, event.collider2);
        let is_sensor = sensors.contains(a) || sensors.contains(b);
        let is_walker = walkers.contains(a) || walkers.contains(b);
        if is_sensor && is_walker {
            modifier.0 = 1.0; // back to normal speed
        }
    }
}

/// Handle E key to interact with buildings, Escape to close UI or return
/// to main menu.
pub(crate) fn building_interact(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    nearby: Res<NearbyBuilding>,
    mut active_ui: ResMut<ActiveBuildingUI>,
    mut next_state: ResMut<NextState<PlayState>>,
    npc_chat: Res<crate::surface_npc_chat::NpcChatState>,
) {
    // Don't process building keys while NPC chat is open.
    if npc_chat.entity.is_some() {
        return;
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        if active_ui.0.is_some() {
            active_ui.0 = None;
        } else {
            next_state.set(PlayState::MainMenu);
        }
        return;
    }

    // Open building on E. Walk-in shops transition to their interior;
    // the rest open the full-screen window as before.
    if keyboard.just_pressed(KeyCode::KeyE)
        && let Some((_, kind)) = nearby.current
    {
        if crate::surface::interiors::has_interior(kind) {
            commands.insert_resource(crate::surface::interiors::InteriorContext { kind, level: 0 });
            next_state.set(PlayState::Inside);
        } else {
            active_ui.0 = Some(kind);
        }
    }
}

/// Show "Press E to enter X" in the comms ticker when near a building.
pub(crate) fn update_interact_prompt(
    nearby: Res<NearbyBuilding>,
    active_ui: Res<ActiveBuildingUI>,
    mut comms: ResMut<crate::hud::CommsChannel>,
) {
    if !nearby.is_changed() && !active_ui.is_changed() {
        return;
    }
    if let (Some((_, kind)), None) = (&nearby.current, &active_ui.0) {
        comms.send(format!("Press E to enter the {}", kind.label()));
    } else if nearby.current.is_none() {
        // Only clear if the message is a "Press E" prompt.
        if comms.message.starts_with("Press E") {
            comms.send("");
        }
    }
}

// ---------------------------------------------------------------------------
// Building egui UIs
// ---------------------------------------------------------------------------

/// Smoothly interpolate the camera zoom via `OrthographicProjection::scale`.
pub(crate) fn animate_camera_zoom(
    zoom: Res<CameraZoom>,
    time: Res<Time>,
    mut cameras: Query<&mut Projection, With<Camera2d>>,
) {
    let Ok(mut proj) = cameras.single_mut() else {
        return;
    };
    let Projection::Orthographic(ref mut ortho) = *proj else {
        return;
    };
    let dt = time.delta_secs();
    let speed = CAMERA_ZOOM_SPEED * dt;
    ortho.scale = ortho.scale + (zoom.target - ortho.scale) * speed.min(1.0);
}

/// Camera follows the walker during Exploring.
pub(crate) fn camera_follow_walker(
    walker_query: Query<&Transform, (With<Walker>, Without<Camera2d>)>,
    mut camera_query: Query<&mut Transform, (With<Camera2d>, Without<Walker>)>,
) {
    let Ok(walker_tf) = walker_query.single() else {
        return;
    };
    let Ok(mut cam_tf) = camera_query.single_mut() else {
        return;
    };
    cam_tf.translation = cam_tf.translation.lerp(walker_tf.translation, 0.1);
}
