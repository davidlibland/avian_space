use bevy::audio::{PlaybackMode, Volume};
use bevy::prelude::*;

use crate::explosions::{TriggerExplosion, TriggerJumpFlash};
use crate::item_universe::ItemUniverse;
use crate::missions::PickupCollected;
use crate::ship::ShipCommand;
use crate::weapons::WeaponFired;
use crate::{PlayState, Player};

pub fn sfx_plugin(app: &mut App) {
    app.init_resource::<ThrusterAudio>()
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
            )
                .run_if(in_state(PlayState::Flying)),
        );
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
        landing: asset_server.load("sounds/landing.wav"),
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
