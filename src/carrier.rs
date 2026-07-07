//! Carrier bay system — ships can launch and recover escort ships.
//!
//! A weapon with `carrier_bay: "fighter"` spawns an escort ship instead of a
//! projectile.  Escorts mirror the mother ship's targets.  When the mother has
//! no weapons target, escorts return and re-dock (despawn + ammo replenish).
//! If the mother is destroyed, escorts become independent.
//!
//! Launch animation: escort grows from a small sprite while accelerating
//! forward out of the mother ship.  Dock animation: escort shrinks as it
//! approaches the mother and disappears underneath.

use avian2d::prelude::*;
use bevy::prelude::*;

use crate::ai_ships::{AIShip, compute_ai_action};
use crate::rl_collection::{RLAgent, RLStepSet};
use crate::sfx::EscortSfx;
use crate::ship::{Ship, ShipCommand, Target};
use crate::ship_anim::{ANIM_MIN_SCALE, ScalingUp};
use crate::utils::safe_despawn;
use crate::weapons::FireCommand;
use crate::{GameLayer, PlayState, Player};

const DETECTION_RADIUS: f32 = 2000.0;

/// Duration of the launch grow animation (seconds).
const LAUNCH_DURATION: f32 = 1.2;

/// Distance at which the docking shrink animation begins.
const DOCK_START_RADIUS: f32 = 120.0;
/// Visual scale at which the escort is considered fully docked and despawned.
const DOCK_DESPAWN_SCALE: f32 = ANIM_MIN_SCALE;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marks a spawned escort, linking it to its mother ship. Escorts follow,
/// take orders (B/N/M), and fight; whether they can DOCK depends on the
/// separate [`CarriedBy`] component.
#[derive(Component)]
pub struct Escort {
    pub mother: Entity,
}

/// The escort lives in a carrier bay on the mother: it may dock to despawn
/// and replenish that bay's ammo. Squadron escorts (mission support wings)
/// have no `CarriedBy` — they fly with you but never dock.
#[derive(Component)]
pub struct CarriedBy {
    /// Weapon type key on the mother (for ammo replenishment on re-dock).
    pub weapon_type: String,
}

/// Tags a squadron escort spawned for a mission, so it despawns when the
/// mission ends (same cleanup pattern as MissionTarget).
#[derive(Component)]
pub struct MissionSquadron(pub String);

/// Escort is in the dock animation — shrinking as it returns to the mother.
#[derive(Component)]
pub struct DockingEscort {
    /// Distance to the mother when docking started, used to compute scale.
    start_distance: f32,
    full_size: Vec2,
}

/// Behavioral mode of an escort. Drives target assignment in
/// [`update_escort_modes`] and gates dock/firing systems.
#[derive(Component, Clone)]
pub enum EscortMode {
    /// Stay near the mother. No engagement.
    Escort,
    /// Pursue and engage a specific target until it is destroyed.
    /// Sticky: doesn't switch when the mother retargets.
    Attack { target: Target },
    /// Approach the mother and dock to replenish the carrier bay.
    Dock,
}

/// Spawn an escort ship. Emitted by `weapon_fire` for carrier bays
/// (`carried: Some(weapon_type)`) and by `spawn_mission_squadrons` for
/// mission support wings (`carried: None`, `mission: Some(id)`).
#[derive(Event, Message, Clone)]
pub struct SpawnEscort {
    pub mother: Entity,
    pub ship_type: String,
    /// Bay weapon key on the mother, when the escort is carried (dockable).
    pub carried: Option<String>,
    pub position: Vec2,
    /// Mission id for squadron escorts (cleanup on mission end).
    pub mission: Option<String>,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub fn carrier_plugin(app: &mut App) {
    // Register EscortSfx/TriggerJumpFlash here (rather than only in the
    // sfx/explosions plugins) so the producers in this module work in
    // headless mode, where those plugins aren't loaded.
    app.add_message::<EscortSfx>();
    app.add_message::<crate::explosions::TriggerJumpFlash>();
    app.add_message::<SpawnEscort>().add_systems(
        Update,
        (
            auto_launch_carrier_bays,
            spawn_mission_squadrons,
            spawn_escort_ships,
            escort_launch_movement,
            player_escort_input,
            update_escort_modes,
            escort_act,
            begin_escort_dock,
            cancel_escort_dock,
            animate_escort_dock,
            orphan_escorts,
        )
            .chain()
            .before(RLStepSet)
            .run_if(in_state(PlayState::Flying)),
    );
}

// ---------------------------------------------------------------------------
// Auto-launch
// ---------------------------------------------------------------------------

/// Auto-fire carrier bay weapons when a ship has a weapons target.
fn auto_launch_carrier_bays(
    ships: Query<(Entity, &Ship), Without<Player>>,
    mut fire_writer: MessageWriter<FireCommand>,
) {
    for (entity, ship) in &ships {
        let wep_entity = ship.weapons_target.as_ref().map(|t| t.get_entity());
        if wep_entity.is_none() {
            continue;
        }
        for (weapon_type, ws) in ship.weapon_systems.iter_all() {
            if ws.weapon.carrier_bay().is_none() {
                continue;
            }
            if ws.ammo_quantity.map(|n| n == 0).unwrap_or(false) {
                continue;
            }
            if !ws.cooldown.is_finished() {
                continue;
            }
            fire_writer.write(FireCommand {
                ship: entity,
                weapon_type: weapon_type.clone(),
                target: wep_entity,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn + launch animation
// ---------------------------------------------------------------------------

/// Spawn escort ships in response to [`SpawnEscort`] messages.
/// The escort starts with a tiny sprite on top of the mother with a
/// [`LaunchingEscort`] timer that drives the grow animation.
fn spawn_escort_ships(
    mut reader: MessageReader<SpawnEscort>,
    mut commands: Commands,
    item_universe: Res<crate::item_universe::ItemUniverse>,
    current_system: Res<crate::CurrentStarSystem>,
    mother_ships: Query<(&Position, &LinearVelocity, &Transform, &Ship)>,
    player: Query<Entity, With<Player>>,
    mut sfx_writer: MessageWriter<EscortSfx>,
    mut jump_flash_writer: MessageWriter<crate::explosions::TriggerJumpFlash>,
    images: Res<Assets<Image>>,
) {
    let player_entity = player.single().ok();
    for event in reader.read() {
        let Ok((_, mother_vel, mother_tf, mother_ship)) =
            mother_ships.get(event.mother)
        else {
            continue;
        };
        // Squadron wings start in formation; bay launches engage the
        // mother's current target immediately.
        let initial_mode = match (&event.carried, mother_ship.weapons_target.as_ref()) {
            (Some(_), Some(t)) => EscortMode::Attack { target: t.clone() },
            _ => EscortMode::Escort,
        };

        // Spawn where the event says, not at the mother: bay launches pass
        // the mother's position anyway, but squadron wings arrive in a ring
        // formation — stacking them all on one point left the physics solver
        // to pop the wing apart.
        let mut bundle = crate::ship::ship_bundle(
            &event.ship_type,
            &item_universe,
            &current_system.0,
            event.position,
        );
        let personality = bundle.get_personality();
        let is_bay_launch = event.carried.is_some();

        // Bay launches grow out of the mother's hull; squadron wings jump in
        // at their formation slots, full-size with a hyperspace flash — they
        // arrive WITH the player, they aren't built by her ship.
        let full_size = crate::ship::ship_display_size(bundle.radius());
        if is_bay_launch {
            bundle.sprite.custom_size = Some(full_size * ANIM_MIN_SCALE);
        } else {
            bundle.sprite.custom_size = Some(full_size);
            jump_flash_writer.write(crate::explosions::TriggerJumpFlash {
                location: event.position,
                size: 8.0,
            });
        }

        // Match the mother's heading via Transform rotation.
        // avian2d will initialize Rotation from this on the first physics step.
        let mother_heading = mother_tf.rotation;

        let escort_entity = commands
            .spawn((
                DespawnOnExit(PlayState::Flying),
                AIShip {
                    personality: personality.clone(),
                },
                Escort {
                    mother: event.mother,
                },
                initial_mode,
                bundle,
            ))
            .insert_if(
                ScalingUp {
                    timer: Timer::from_seconds(LAUNCH_DURATION, TimerMode::Once),
                    full_size,
                },
                || is_bay_launch,
            )
            .insert((
                Position(event.position),
                LinearVelocity(mother_vel.0),
            ))
            .insert_if(
                CarriedBy {
                    weapon_type: event.carried.clone().unwrap_or_default(),
                },
                || event.carried.is_some(),
            )
            .insert_if(
                MissionSquadron(event.mission.clone().unwrap_or_default()),
                || event.mission.is_some(),
            )
            .with_children(|parent| {
                parent.spawn((
                    Collider::circle(DETECTION_RADIUS),
                    Sensor,
                    CollisionLayers::new(
                        GameLayer::Radar,
                        [GameLayer::Planet, GameLayer::Asteroid, GameLayer::Ship],
                    ),
                ));
            })
            .id();

        if Some(event.mother) == player_entity {
            sfx_writer.write(EscortSfx::Launching);
        }
    }
}

/// During the launch animation, accelerate the escort forward out of the
/// mother ship.  The sprite scaling is handled by [`ship_anim::animate_scale_up`].
fn escort_launch_movement(
    time: Res<Time>,
    mut escorts: Query<
        (&ScalingUp, &mut LinearVelocity, &Escort),
        Without<DockingEscort>,
    >,
    mother_transforms: Query<&Transform, Without<Escort>>,
) {
    for (scaling, mut vel, escort) in &mut escorts {
        let t = scaling.timer.fraction();

        if let Ok(mother_tf) = mother_transforms.get(escort.mother) {
            let forward = (mother_tf.rotation * Vec3::Y).xy();
            let mother_vel_approx = vel.0.dot(forward).max(0.0);
            let target_forward_speed = mother_vel_approx + 40.0 * t;
            let current_forward = vel.0.dot(forward);
            if current_forward < target_forward_speed {
                vel.0 += forward
                    * (target_forward_speed - current_forward).min(200.0 * time.delta_secs());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Target sync
// ---------------------------------------------------------------------------

/// Handle mode transitions and apply nav/weapons targets accordingly.
///
/// Attack mode is sticky on its specific target. When that target is gone,
/// the escort either acquires the mother's current weapons target (NPC) or
/// returns to escort mode (player escorts always disengage on kill).
fn update_escort_modes(
    mother_ships: Query<(&Ship, Has<Player>), Without<Escort>>,
    mut escorts: Query<
        (&Escort, &mut EscortMode, &mut Ship),
        Without<DockingEscort>,
    >,
    target_alive: Query<(), With<Position>>,
    mut sfx_writer: MessageWriter<EscortSfx>,
) {
    for (escort, mut mode, mut ship) in &mut escorts {
        let Ok((mother, mother_is_player)) = mother_ships.get(escort.mother) else {
            continue;
        };

        if let EscortMode::Attack { ref target } = *mode {
            if target_alive.get(target.get_entity()).is_err() {
                if mother_is_player {
                    sfx_writer.write(EscortSfx::Neutralized);
                }
                *mode = if mother_is_player {
                    EscortMode::Escort
                } else {
                    match mother.weapons_target.as_ref() {
                        Some(new_target) => EscortMode::Attack {
                            target: new_target.clone(),
                        },
                        // Dock only means something for carried escorts;
                        // begin_escort_dock gates on CarriedBy, so a squadron
                        // wing in Dock mode just shadows its mother.
                        None => EscortMode::Dock,
                    }
                };
            }
        }

        match &*mode {
            EscortMode::Attack { target } => {
                ship.nav_target = Some(target.clone());
                ship.weapons_target = Some(target.clone());
            }
            EscortMode::Escort | EscortMode::Dock => {
                ship.nav_target = Some(Target::Ship(escort.mother));
                ship.weapons_target = None;
            }
        }
    }
}

/// Player keybinds for commanding their own escorts:
/// - `B`: dock — return to mother and replenish
/// - `N`: escort — disengage and stay near
/// - `M`: attack the player's current weapons target
fn player_escort_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    player: Query<(Entity, &Ship), With<Player>>,
    mut escorts: Query<(&Escort, Has<CarriedBy>, &mut EscortMode)>,
    mut sfx_writer: MessageWriter<EscortSfx>,
) {
    let Ok((player_entity, player_ship)) = player.single() else {
        return;
    };
    let (new_mode, sfx) = if keyboard.just_pressed(KeyCode::KeyB) {
        (Some(EscortMode::Dock), EscortSfx::Dock)
    } else if keyboard.just_pressed(KeyCode::KeyN) {
        (Some(EscortMode::Escort), EscortSfx::Escort)
    } else if keyboard.just_pressed(KeyCode::KeyM) {
        let mode = player_ship
            .weapons_target
            .as_ref()
            .map(|t| EscortMode::Attack { target: t.clone() });
        (mode, EscortSfx::Attack)
    } else {
        return;
    };
    let Some(new_mode) = new_mode else { return };
    // Only command (and only SOUND) if the player actually has escorts —
    // the cue used to play on every keypress, escorts or not.
    let mut commanded = false;
    let mut any_docked = false;
    for (escort, carried, mut mode) in &mut escorts {
        if escort.mother != player_entity {
            continue;
        }
        commanded = true;
        // Squadron wings (not carried) can't dock — B falls back to
        // holding formation.
        *mode = if matches!(new_mode, EscortMode::Dock) && !carried {
            EscortMode::Escort
        } else {
            any_docked |= carried;
            new_mode.clone()
        };
    }
    if commanded {
        // The dock cue only fits if something can actually dock; a
        // wings-only flight acknowledges with the formation cue instead.
        let sfx = match sfx {
            EscortSfx::Dock if !any_docked => EscortSfx::Escort,
            other => other,
        };
        sfx_writer.write(sfx);
    }
}

// ---------------------------------------------------------------------------
// Deterministic escort control
// ---------------------------------------------------------------------------

/// Drive escort thrust/turn/firing via the same rule-based policy used to
/// generate BC labels. Escorts are excluded from the RL pipeline (no `RLAgent`),
/// so this is the sole driver of their actions.
fn escort_act(
    mut escorts: Query<
        (
            Entity,
            &Position,
            &LinearVelocity,
            &AngularVelocity,
            &MaxLinearSpeed,
            &Transform,
            &mut Ship,
        ),
        (With<Escort>, Without<ScalingUp>),
    >,
    all_positions: Query<&Position>,
    all_velocities: Query<&LinearVelocity>,
    item_universe: Res<crate::item_universe::ItemUniverse>,
    mut ship_writer: MessageWriter<ShipCommand>,
    mut weapons_writer: MessageWriter<FireCommand>,
) {
    let mut rng = rand::thread_rng();
    for (entity, position, ship_vel, ang_vel, max_speed, ship_transform, mut ship) in &mut escorts
    {
        if ship.nav_target.is_none() {
            continue;
        }
        match compute_ai_action(
            &*ship,
            position.0,
            ship_vel.0,
            ang_vel.0,
            max_speed.0,
            ship_transform,
            &all_positions,
            &all_velocities,
            &item_universe,
            &mut rng,
        ) {
            Some(action) => {
                ship_writer.write(ShipCommand {
                    entity,
                    thrust: action.thrust,
                    turn: action.turn,
                    reverse: action.reverse,
                });
                for (weapon_type, fire_target) in action.weapons_to_fire {
                    weapons_writer.write(FireCommand {
                        ship: entity,
                        weapon_type,
                        target: fire_target,
                    });
                }
            }
            None => {
                ship.nav_target = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dock animation
// ---------------------------------------------------------------------------

/// When an escort in Dock mode is close enough to the mother, begin the
/// docking animation by adding [`DockingEscort`].
fn begin_escort_dock(
    mut commands: Commands,
    escorts: Query<
        (Entity, &Escort, &EscortMode, &Position, &Ship),
        (
            With<CarriedBy>,
            Without<ScalingUp>,
            Without<DockingEscort>,
        ),
    >,
    mother_ships: Query<(&Ship, &Position), Without<Escort>>,
) {
    for (entity, escort, mode, escort_pos, ship) in &escorts {
        if !matches!(mode, EscortMode::Dock) {
            continue;
        }
        let Ok((mother, mother_pos)) = mother_ships.get(escort.mother) else {
            continue;
        };
        let dist = (escort_pos.0 - mother_pos.0).length();
        if dist < DOCK_START_RADIUS + mother.data.radius {
            let full_size = crate::ship::ship_display_size(ship.data.radius);
            commands.entity(entity).insert(DockingEscort {
                start_distance: dist.max(1.0),
                full_size,
            });
        }
    }
}

/// If an escort leaves Dock mode mid-animation (e.g. player issued a new
/// command), cancel the dock and let it act normally again.
fn cancel_escort_dock(
    mut commands: Commands,
    mut escorts: Query<(Entity, &EscortMode, &DockingEscort, &mut Sprite)>,
) {
    for (entity, mode, dock, mut sprite) in &mut escorts {
        if !matches!(mode, EscortMode::Dock) {
            // Restore the full display size (atlas tiles are decoupled from it).
            sprite.custom_size = Some(dock.full_size);
            commands.entity(entity).remove::<DockingEscort>();
        }
    }
}

/// Animate the dock: shrink the escort's sprite as it approaches the mother.
/// When small enough, despawn and replenish ammo.
fn animate_escort_dock(
    mut commands: Commands,
    mut escorts: Query<(
        Entity,
        &Escort,
        &CarriedBy,
        &DockingEscort,
        &EscortMode,
        &Position,
        &mut Sprite,
        &mut Ship,
        &mut MaxLinearSpeed,
    )>,
    mut mother_ships: Query<(&mut Ship, &Position), Without<Escort>>,
    player: Query<Entity, With<Player>>,
    mut sfx_writer: MessageWriter<EscortSfx>,
) {
    let player_entity = player.single().ok();
    for (
        entity,
        escort,
        carried,
        dock,
        mode,
        escort_pos,
        mut sprite,
        mut escort_ship,
        mut max_speed,
    ) in &mut escorts
    {
        // Mode may have changed mid-dock (e.g. player issued a new command);
        // cancel_escort_dock will remove DockingEscort, but its `commands` flush
        // is deferred — guard here to avoid finishing the dock this same frame.
        if !matches!(mode, EscortMode::Dock) {
            continue;
        }
        let Ok((mut mother, mother_pos)) = mother_ships.get_mut(escort.mother) else {
            continue;
        };

        let dist = (escort_pos.0 - mother_pos.0).length();

        // Visual scale: 1.0 at start_distance → DOCK_DESPAWN_SCALE at 0
        let frac = (dist / dock.start_distance).clamp(DOCK_DESPAWN_SCALE, 1.0);
        sprite.custom_size = Some(dock.full_size * frac);

        // Slow down as we approach
        max_speed.0 = escort_ship.data.max_speed * frac.max(0.25);

        // Keep heading toward the mother
        escort_ship.nav_target = Some(Target::Ship(escort.mother));
        escort_ship.weapons_target = None;

        // Despawn when small enough
        if frac <= DOCK_DESPAWN_SCALE + 0.01 {
            if let Some(ws) = mother.weapon_systems.find_weapon(&carried.weapon_type) {
                ws.ammo_quantity = ws.ammo_quantity.map(|n| n + 1);
            }
            if Some(escort.mother) == player_entity {
                sfx_writer.write(EscortSfx::Docked);
            }
            safe_despawn(&mut commands, entity);
        }
    }
}

// ---------------------------------------------------------------------------
// Mission squadrons
// ---------------------------------------------------------------------------

/// Spawn support wings for active missions that declare a `squadron`, once
/// the player is in the mission's battle system. Idempotent per mission via
/// `ObjectiveProgress::squadron_spawned` — wings that die are NOT replaced
/// (losses are real), and the whole wing despawns when the mission ends
/// (see missions::despawn_targets_on_failure).
fn spawn_mission_squadrons(
    catalog: Res<crate::missions::MissionCatalog>,
    mut log: ResMut<crate::missions::MissionLog>,
    current_system: Res<crate::CurrentStarSystem>,
    player: Query<(Entity, &Position), With<Player>>,
    mut writer: MessageWriter<SpawnEscort>,
) {
    use crate::missions::{MissionStatus, Objective};
    let Ok((player_entity, player_pos)) = player.single() else {
        return;
    };
    for (id, def) in &catalog.defs {
        if def.squadron.is_empty() {
            continue;
        }
        let MissionStatus::Active(progress) = log.status(id) else {
            continue;
        };
        if progress.squadron_spawned {
            continue;
        }
        // The wing musters in the mission's battle system (or wherever the
        // player is, for objectives without a system).
        let battle_system = match &def.objective {
            Objective::DestroyShips { system, .. } | Objective::TravelToSystem { system } => {
                Some(system.as_str())
            }
            _ => None,
        };
        if battle_system.is_some_and(|s| s != current_system.0) {
            continue;
        }
        for (i, ship_type) in def.squadron.iter().enumerate() {
            let angle = i as f32 / def.squadron.len().max(1) as f32 * std::f32::consts::TAU;
            let offset = Vec2::new(angle.cos(), angle.sin()) * 80.0;
            writer.write(SpawnEscort {
                mother: player_entity,
                ship_type: ship_type.clone(),
                carried: None,
                position: player_pos.0 + offset,
                mission: Some(id.clone()),
            });
        }
        log.set(
            id,
            MissionStatus::Active(crate::missions::types::ObjectiveProgress {
                squadron_spawned: true,
                ..progress
            }),
        );
    }
}

// ---------------------------------------------------------------------------
// Orphan handling
// ---------------------------------------------------------------------------

/// When a mother ship is destroyed, strip carrier-related components and
/// attach an [`RLAgent`] so the escort becomes an independent AI ship driven
/// by the RL pipeline.
fn orphan_escorts(
    mut commands: Commands,
    mut escorts: Query<(Entity, &Escort, &AIShip, &Ship, &mut Sprite)>,
    mothers: Query<(), With<Ship>>,
) {
    for (entity, escort, ai_ship, ship, mut sprite) in &mut escorts {
        if mothers.get(escort.mother).is_err() {
            // Atlas tiles are a fixed 128 px — custom_size carries the real
            // display size, so clearing it ballooned orphans to tile size.
            sprite.custom_size = Some(crate::ship::ship_display_size(ship.data.radius));
            commands
                .entity(entity)
                .remove::<(Escort, EscortMode, ScalingUp, DockingEscort, CarriedBy, MissionSquadron)>()
                .insert(RLAgent::new(ai_ship.personality.clone()));
        }
    }
}

#[cfg(test)]
#[path = "tests/carrier_tests.rs"]
mod tests;
