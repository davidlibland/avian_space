use std::collections::HashSet;

use bevy::audio::{PlaybackMode, Volume};
use bevy::prelude::*;

use crate::explosions::{TriggerExplosion, TriggerJumpFlash};
use crate::item_universe::ItemUniverse;
use crate::missions::PickupCollected;
use crate::ship::{Ship, ShipCommand, Target};
use crate::weapons::WeaponFired;
use crate::{PlayState, Player};

// ── Surface sound effects (used by the surface module) ───────────────────

/// Fire-and-forget surface sound event.  Any system can send this
/// without needing Commands or AssetServer.
#[derive(Event, Message, Clone)]
pub enum SurfaceSfx {
    Door,
    UiOpen,
    UiClose,
    UiButton,
    Footstep { surface: String, volume: f32 },
}

/// Minimum gap between consecutive "you are being targeted" warnings.
const WARNING_COOLDOWN_SECS: f32 = 4.0;
/// How often to scan AI ships for new lock-ons.
const WARNING_SCAN_INTERVAL_SECS: f32 = 1.0;

pub fn sfx_plugin(app: &mut App) {
    app.init_resource::<ThrusterAudio>()
        .add_message::<SurfaceSfx>()
        .add_systems(Startup, load_sfx)
        .add_systems(OnEnter(PlayState::Traveling), play_player_jump_sfx)
        .add_systems(OnEnter(PlayState::Landed), play_landing_sfx)
        .add_systems(
            Update,
            (
                play_weapon_sfx,
                play_explosion_sfx,
                drain_jump_flash_events,
                update_thruster_sfx,
                play_pickup_sfx,
                play_target_change_sfx,
                play_warning_sfx,
            )
                .run_if(in_state(PlayState::Flying)),
        )
        .add_systems(Update, play_surface_sfx);
}

#[derive(Resource, Default)]
struct SfxHandles {
    explosion_small: Handle<AudioSource>,
    explosion_medium: Handle<AudioSource>,
    explosion_large: Handle<AudioSource>,
    thruster: Handle<AudioSource>,
    jump: Handle<AudioSource>,
    pickup: Handle<AudioSource>,
    landing: Handle<AudioSource>,
    target: Handle<AudioSource>,
    warning: Handle<AudioSource>,
    // Surface sounds
    door: Handle<AudioSource>,
    ui_open: Handle<AudioSource>,
    ui_close: Handle<AudioSource>,
    ui_button: Handle<AudioSource>,
}

impl SfxHandles {
    fn explosion(&self, size: f32) -> Handle<AudioSource> {
        if size < 15.0 {
            self.explosion_small.clone()
        } else if size < 35.0 {
            self.explosion_medium.clone()
        } else {
            self.explosion_large.clone()
        }
    }
}

#[derive(Resource, Default)]
struct ThrusterAudio {
    entity: Option<Entity>,
}

fn load_sfx(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(SfxHandles {
        explosion_small: asset_server.load("sounds/explosion_small.ogg"),
        explosion_medium: asset_server.load("sounds/explosion_medium.ogg"),
        explosion_large: asset_server.load("sounds/explosion_large.ogg"),
        thruster: asset_server.load("sounds/thruster_loop.ogg"),
        jump: asset_server.load("sounds/jump.ogg"),
        pickup: asset_server.load("sounds/pickup.ogg"),
        landing: asset_server.load("sounds/landing.ogg"),
        target: asset_server.load("sounds/target.ogg"),
        warning: asset_server.load("sounds/warning.ogg"),
        door: asset_server.load("sounds/world/door.ogg"),
        ui_open: asset_server.load("sounds/world/ui_open.ogg"),
        ui_close: asset_server.load("sounds/world/ui_close.ogg"),
        ui_button: asset_server.load("sounds/world/ui_button.ogg"),
    });
}

fn play_weapon_sfx(
    mut commands: Commands,
    mut reader: MessageReader<WeaponFired>,
    item_universe: Res<ItemUniverse>,
    ships: Query<(&Transform, Has<Player>)>,
) {
    for cmd in reader.read() {
        let Some(weapon) = item_universe.weapons.get(&cmd.weapon_type) else {
            continue;
        };
        let Some(handle) = weapon.sound_handle.clone() else {
            continue;
        };
        let Ok((transform, is_player)) = ships.get(cmd.ship) else {
            continue;
        };
        let volume = if is_player { 0.25 } else { 0.15 };
        let mut settings = PlaybackSettings {
            mode: PlaybackMode::Despawn,
            volume: Volume::Linear(volume),
            ..default()
        };
        if !is_player {
            settings = settings.with_spatial(true);
        }
        commands.spawn((
            AudioPlayer::<AudioSource>(handle),
            settings,
            Transform::from_translation(transform.translation),
            DespawnOnExit(PlayState::Flying),
        ));
    }
}

fn play_explosion_sfx(
    mut commands: Commands,
    mut reader: MessageReader<TriggerExplosion>,
    sfx: Option<Res<SfxHandles>>,
) {
    let Some(sfx) = sfx else {
        return;
    };
    for event in reader.read() {
        let handle = sfx.explosion(event.size);
        let volume = (0.15 + event.size * 0.005).min(0.4);
        commands.spawn((
            AudioPlayer::<AudioSource>(handle),
            PlaybackSettings {
                mode: PlaybackMode::Despawn,
                volume: Volume::Linear(volume),
                ..default()
            }
            .with_spatial(true),
            Transform::from_translation(event.location.extend(0.0)),
            DespawnOnExit(PlayState::Flying),
        ));
    }
}

fn drain_jump_flash_events(mut reader: MessageReader<TriggerJumpFlash>) {
    for _ in reader.read() {}
}

fn play_player_jump_sfx(mut commands: Commands, sfx: Option<Res<SfxHandles>>) {
    let Some(sfx) = sfx else {
        return;
    };
    commands.spawn((
        AudioPlayer::<AudioSource>(sfx.jump.clone()),
        PlaybackSettings {
            mode: PlaybackMode::Despawn,
            volume: Volume::Linear(0.4),
            ..default()
        },
    ));
}

fn update_thruster_sfx(
    mut commands: Commands,
    mut reader: MessageReader<ShipCommand>,
    mut audio: ResMut<ThrusterAudio>,
    sfx: Option<Res<SfxHandles>>,
    player: Query<Entity, With<Player>>,
    existing: Query<(), With<AudioPlayer<AudioSource>>>,
) {
    let Some(sfx) = sfx else {
        return;
    };
    let Ok(player_entity) = player.single() else {
        if let Some(e) = audio.entity.take() {
            if existing.get(e).is_ok() {
                commands.entity(e).despawn();
            }
        }
        return;
    };

    let mut thrust_active = false;
    for cmd in reader.read() {
        if cmd.entity == player_entity && cmd.thrust.abs() > f32::EPSILON {
            thrust_active = true;
        }
    }

    match (thrust_active, audio.entity) {
        (true, None) => {
            let e = commands
                .spawn((
                    AudioPlayer::<AudioSource>(sfx.thruster.clone()),
                    PlaybackSettings {
                        mode: PlaybackMode::Loop,
                        volume: Volume::Linear(0.12),
                        ..default()
                    },
                    DespawnOnExit(PlayState::Flying),
                ))
                .id();
            audio.entity = Some(e);
        }
        (false, Some(e)) => {
            if existing.get(e).is_ok() {
                commands.entity(e).despawn();
            }
            audio.entity = None;
        }
        _ => {}
    }
}

fn play_pickup_sfx(
    mut commands: Commands,
    mut reader: MessageReader<PickupCollected>,
    sfx: Option<Res<SfxHandles>>,
) {
    let Some(sfx) = sfx else {
        return;
    };
    for _ in reader.read() {
        commands.spawn((
            AudioPlayer::<AudioSource>(sfx.pickup.clone()),
            PlaybackSettings {
                mode: PlaybackMode::Despawn,
                volume: Volume::Linear(0.4),
                ..default()
            },
        ));
    }
}

fn play_landing_sfx(mut commands: Commands, sfx: Option<Res<SfxHandles>>) {
    let Some(sfx) = sfx else {
        return;
    };
    commands.spawn((
        AudioPlayer::<AudioSource>(sfx.landing.clone()),
        PlaybackSettings {
            mode: PlaybackMode::Despawn,
            volume: Volume::Linear(0.5),
            ..default()
        },
    ));
}

fn play_target_change_sfx(
    mut commands: Commands,
    mut last_target: Local<Option<Target>>,
    sfx: Option<Res<SfxHandles>>,
    player: Query<&Ship, With<Player>>,
) {
    let Some(sfx) = sfx else {
        return;
    };
    let Ok(ship) = player.single() else {
        *last_target = None;
        return;
    };
    if ship.nav_target != *last_target {
        *last_target = ship.nav_target.clone();
        // Don't chime when the target is cleared (Some → None or initial None).
        if ship.nav_target.is_some() {
            commands.spawn((
                AudioPlayer::<AudioSource>(sfx.target.clone()),
                PlaybackSettings {
                    mode: PlaybackMode::Despawn,
                    volume: Volume::Linear(0.4),
                    ..default()
                },
            ));
        }
    }
}

fn play_warning_sfx(
    mut commands: Commands,
    mut prev_lockers: Local<HashSet<Entity>>,
    mut cooldown: Local<f32>,
    mut scan_timer: Local<f32>,
    time: Res<Time>,
    sfx: Option<Res<SfxHandles>>,
    player: Query<Entity, With<Player>>,
    other_ships: Query<(Entity, &Ship), Without<Player>>,
) {
    let Some(sfx) = sfx else {
        return;
    };
    let dt = time.delta_secs();
    *cooldown = (*cooldown - dt).max(0.0);
    *scan_timer -= dt;
    if *scan_timer > 0.0 {
        return;
    }
    *scan_timer = WARNING_SCAN_INTERVAL_SECS;

    let Ok(player_entity) = player.single() else {
        prev_lockers.clear();
        return;
    };

    let mut current = HashSet::new();
    let mut new_lockers = false;
    for (entity, ship) in &other_ships {
        if matches!(ship.weapons_target, Some(Target::Ship(e)) if e == player_entity) {
            current.insert(entity);
            if !prev_lockers.contains(&entity) {
                new_lockers = true;
            }
        }
    }
    *prev_lockers = current;

    if new_lockers && *cooldown <= 0.0 {
        *cooldown = WARNING_COOLDOWN_SECS;
        commands.spawn((
            AudioPlayer::<AudioSource>(sfx.warning.clone()),
            PlaybackSettings {
                mode: PlaybackMode::Despawn,
                volume: Volume::Linear(0.5),
                ..default()
            },
        ));
    }
}

/// Play surface sound effects dispatched via [`SurfaceSfx`] events.
fn play_surface_sfx(
    mut commands: Commands,
    mut reader: MessageReader<SurfaceSfx>,
    sfx: Option<Res<SfxHandles>>,
    asset_server: Res<AssetServer>,
) {
    let Some(sfx) = sfx else { return };
    for event in reader.read() {
        let (handle, volume) = match event {
            SurfaceSfx::Door => (sfx.door.clone(), 0.4),
            SurfaceSfx::UiOpen => (sfx.ui_open.clone(), 0.5),
            SurfaceSfx::UiClose => (sfx.ui_close.clone(), 0.5),
            SurfaceSfx::UiButton => (sfx.ui_button.clone(), 0.5),
            SurfaceSfx::Footstep { surface, volume } => {
                use rand::Rng;
                let variant = rand::thread_rng().r#gen_range(0u32..5);
                let path = format!(
                    "sounds/people/walking/footstep_{surface}_{variant:03}.ogg"
                );
                let h = asset_server.load(path);
                (h, *volume)
            }
        };
        commands.spawn((
            AudioPlayer::<AudioSource>(handle),
            PlaybackSettings {
                mode: PlaybackMode::Despawn,
                volume: Volume::Linear(volume),
                ..default()
            },
        ));
    }
}
