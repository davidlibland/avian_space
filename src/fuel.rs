//! Fuel-rescue: never let a dry tank permanently strand the player.
//!
//! Two safety nets (see docs/fuel_rescue_plan.md):
//!  1. **Fuel asteroids** — hydrogen-ice rocks (tagged [`FuelShimmer`]) that
//!     shimmer and drop the `fuel` commodity, which [`crate::ship::Ship::
//!     receive_pickup`] routes into the jump tank. Mine your way home.
//!  2. **Distress call** — when Flying with an empty tank, a comms prompt and
//!     a button summon a Guild [`RescueTanker`] that hyperspaces in, tops the
//!     tank, bills a hefty fee, and leaves. Covers the weaponless / no-field
//!     case so there's no dead end.

use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use rand::Rng;

use crate::ship::Ship;
use crate::{PlayState, Player};

/// Multiple of the fuel-station rate a rescue costs (convenience has a price).
const RESCUE_FEE_MULT: i128 = 4;
/// Tanker gets this close (plus radii) before it services the ship.
const SERVICE_RANGE: f32 = 90.0;

// ── Fuel shimmer (shared by fuel rocks and fuel pickups) ──────────────────

/// Marks an entity that should sparkle cyan — hydrogen-ice asteroids and the
/// fuel pickups they drop both carry it, so the effect reads identically.
#[derive(Component)]
pub struct FuelShimmer;

/// Timer so the shimmer emits at a steady rate rather than every frame.
#[derive(Resource, Default)]
struct ShimmerClock(f32);

fn fuel_shimmer(
    mut commands: Commands,
    time: Res<Time>,
    mut clock: ResMut<ShimmerClock>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    shimmering: Query<&Transform, With<FuelShimmer>>,
) {
    clock.0 += time.delta_secs();
    if clock.0 < 0.12 {
        return;
    }
    clock.0 = 0.0;
    let mut rng = rand::thread_rng();
    for tf in &shimmering {
        // One drifting cyan spark per tick, from a random point on the rock.
        let a = rng.gen_range(0.0..std::f32::consts::TAU);
        let off = Vec2::new(a.cos(), a.sin()) * rng.gen_range(4.0..14.0);
        let vel = Vec2::new(a.cos(), a.sin()) * rng.gen_range(6.0..18.0);
        crate::explosions::spawn_spark(
            &mut commands,
            &mut meshes,
            &mut materials,
            tf.translation.truncate() + off,
            vel,
            rng.gen_range(1.0..2.2),
            Color::srgb(0.4, 0.95, 1.0),
            rng.gen_range(0.4..0.8),
        );
    }
}

// ── Distress call + rescue tanker ─────────────────────────────────────────

/// Whether a rescue is currently underway (don't offer or spawn another).
#[derive(Resource, Default)]
struct DistressState {
    active: bool,
}

/// The rescue tanker's lifecycle.
#[derive(Component)]
struct RescueTanker {
    stage: TankerStage,
    /// Countdown for the brief refueling pause.
    service_timer: f32,
}

#[derive(PartialEq)]
enum TankerStage {
    Approach,
    Service,
    Depart,
}

/// Offer a distress call whenever the player is Flying on an empty tank and
/// no rescue is already running. Renders a small prompt with a Send button.
fn distress_prompt(
    mut egui_contexts: EguiContexts,
    state: Res<State<PlayState>>,
    distress: Res<DistressState>,
    player: Query<&crate::ship::Ship, With<Player>>,
    mut summon: MessageWriter<SummonRescue>,
) {
    if *state.get() != PlayState::Flying || distress.active {
        return;
    }
    let Ok(ship) = player.single() else {
        return;
    };
    if ship.fuel > 0 {
        return;
    }
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let fee = ship.fuel_price_per_unit() * ship.data.fuel_capacity as i128 * RESCUE_FEE_MULT;
    egui::Window::new("Out of fuel")
        .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -80.0])
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label("Your jump tank is dry. Mine hydrogen-ice (the shimmering");
            ui.label("rocks) for fuel — or call a Guild tanker for a tow.");
            ui.add_space(4.0);
            if ui
                .button(format!("Send distress call  (tanker fee: {fee} cr)"))
                .clicked()
            {
                summon.write(SummonRescue);
            }
        });
}

/// Fired by the distress prompt button.
#[derive(Message)]
struct SummonRescue;

fn spawn_rescue_tanker(
    mut reader: MessageReader<SummonRescue>,
    mut commands: Commands,
    mut distress: ResMut<DistressState>,
    item_universe: Res<crate::item_universe::ItemUniverse>,
    current_system: Res<crate::CurrentStarSystem>,
    mut jump_flash: MessageWriter<crate::explosions::TriggerJumpFlash>,
    player: Query<&Position, With<Player>>,
) {
    if reader.read().count() == 0 || distress.active {
        return;
    }
    let Ok(player_pos) = player.single() else {
        return;
    };
    let mut rng = rand::thread_rng();
    // Arrive a short hop off the player's position.
    let (entry, _vel) = crate::ai_ships::jump_in_entry_at(&mut rng, 900.0);
    let pos = player_pos.0 + entry;
    let mut bundle =
        crate::ship::ship_bundle("relief_tanker", &item_universe, &current_system.0, pos);
    bundle.set_display_name("Relief Tanker");
    let radius = bundle.radius();
    jump_flash.write(crate::explosions::TriggerJumpFlash {
        location: pos,
        size: 10.0,
    });
    commands.spawn((
        DespawnOnExit(PlayState::Flying),
        bundle,
        crate::ai_ships::jump_in_markers(radius),
        RescueTanker {
            stage: TankerStage::Approach,
            service_timer: 2.5,
        },
    ));
    distress.active = true;
}

/// Drive the tanker: steer to the player, top the tank, jump out.
#[allow(clippy::type_complexity)]
fn run_rescue_tanker(
    mut commands: Commands,
    time: Res<Time>,
    mut comms: ResMut<crate::hud::CommsChannel>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut tankers: Query<
        (
            Entity,
            &mut RescueTanker,
            &Position,
            &mut LinearVelocity,
            &mut Transform,
            &Ship,
        ),
        (Without<Player>, Without<crate::ai_ships::JumpingIn>),
    >,
    mut player: Query<(&Position, &mut Ship), With<Player>>,
) {
    let Ok((player_pos, mut player_ship)) = player.single_mut() else {
        return;
    };
    let dt = time.delta_secs();
    let mut rng = rand::thread_rng();
    for (entity, mut tanker, pos, mut vel, mut tf, ship) in &mut tankers {
        let to_player = player_pos.0 - pos.0;
        let dist = to_player.length();
        match tanker.stage {
            TankerStage::Approach => {
                // Point at the player and cruise in.
                if dist > f32::EPSILON {
                    let dir = to_player / dist;
                    vel.0 = dir * ship.data.max_speed;
                    tf.rotation =
                        Quat::from_rotation_z(dir.y.atan2(dir.x) - std::f32::consts::FRAC_PI_2);
                }
                if dist <= SERVICE_RANGE + ship.data.radius + player_ship.data.radius {
                    tanker.stage = TankerStage::Service;
                    vel.0 = Vec2::ZERO;
                    comms.send("Relief Tanker: Hold still, topping you off.");
                }
            }
            TankerStage::Service => {
                vel.0 *= 0.85; // settle
                // Fuel-line sparks bridging the two hulls.
                let t = rng.gen_range(0.0..1.0);
                crate::explosions::spawn_spark(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    pos.0.lerp(player_pos.0, t),
                    Vec2::ZERO,
                    rng.gen_range(1.2..2.4),
                    Color::srgb(0.4, 0.95, 1.0),
                    0.4,
                );
                tanker.service_timer -= dt;
                if tanker.service_timer <= 0.0 {
                    // Fill the tank; bill the player (debt allowed).
                    let cap = player_ship.data.fuel_capacity;
                    let units = cap.saturating_sub(player_ship.fuel);
                    let fee = player_ship.fuel_price_per_unit() * units as i128 * RESCUE_FEE_MULT;
                    player_ship.fuel = cap;
                    player_ship.credits -= fee;
                    let owed = if player_ship.credits < 0 {
                        format!(" You owe {} cr.", -player_ship.credits)
                    } else {
                        String::new()
                    };
                    comms.send(format!("Relief Tanker: Tank's full. Fly safe.{owed}"));
                    tanker.stage = TankerStage::Depart;
                    // Head outward and jump.
                    let out = if dist > f32::EPSILON {
                        -to_player / dist
                    } else {
                        Vec2::Y
                    };
                    tf.rotation =
                        Quat::from_rotation_z(out.y.atan2(out.x) - std::f32::consts::FRAC_PI_2);
                    commands.entity(entity).insert(crate::ai_ships::JumpingOut);
                }
            }
            TankerStage::Depart => {
                // JumpingOut (in ai_ships) accelerates + despawns with a
                // flash; clear_distress_when_gone re-arms once it's gone.
            }
        }
    }
}

/// Clear the rescue flag when the tanker has jumped out (despawned).
fn clear_distress_when_gone(
    mut distress: ResMut<DistressState>,
    tankers: Query<(), With<RescueTanker>>,
) {
    if distress.active && tankers.is_empty() {
        distress.active = false;
    }
}

pub fn fuel_plugin(app: &mut App) {
    app.init_resource::<ShimmerClock>()
        .init_resource::<DistressState>()
        .add_message::<SummonRescue>()
        .add_systems(
            Update,
            (
                fuel_shimmer,
                spawn_rescue_tanker,
                run_rescue_tanker,
                clear_distress_when_gone,
            )
                .run_if(in_state(PlayState::Flying)),
        )
        .add_systems(EguiPrimaryContextPass, distress_prompt);
}

#[cfg(test)]
mod tests {
    use crate::item_universe::ItemUniverse;
    use crate::ship::{FUEL_COMMODITY, Ship};

    fn dry_shuttle(iu: &ItemUniverse) -> Ship {
        let mut s = Ship::from_ship_data(iu.ships.get("shuttle").unwrap(), "shuttle");
        s.fuel = 0;
        s
    }

    fn iu() -> ItemUniverse {
        let mut iu: ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        iu
    }

    #[test]
    fn scooped_fuel_fills_the_tank_not_the_hold() {
        let iu = iu();
        let mut ship = dry_shuttle(&iu);
        let cap = ship.data.fuel_capacity;
        let before_cargo = ship.current_cargo();
        let taken = ship.receive_pickup(FUEL_COMMODITY, 3);
        assert_eq!(taken, 3.min(cap));
        assert_eq!(ship.fuel, 3.min(cap));
        assert_eq!(ship.current_cargo(), before_cargo, "fuel is not cargo");
    }

    #[test]
    fn fuel_scoop_caps_at_tank_capacity() {
        let iu = iu();
        let mut ship = dry_shuttle(&iu);
        let cap = ship.data.fuel_capacity;
        let taken = ship.receive_pickup(FUEL_COMMODITY, cap + 5);
        assert_eq!(taken, cap);
        assert_eq!(ship.fuel, cap);
        // A full tank refuses more.
        assert_eq!(ship.receive_pickup(FUEL_COMMODITY, 1), 0);
    }

    #[test]
    fn non_fuel_pickups_still_go_to_cargo() {
        let iu = iu();
        let mut ship = dry_shuttle(&iu);
        let taken = ship.receive_pickup("iron", 4);
        assert_eq!(taken, 4);
        assert_eq!(ship.fuel, 0);
        assert_eq!(*ship.cargo.get("iron").unwrap(), 4);
    }
}

#[cfg(test)]
mod debt_tests {
    #[test]
    fn negative_credits_round_trip_through_the_save_format() {
        // A rescued-into-debt pilot: credits go negative. serde_yaml must
        // preserve i128 across the exact path PilotSave uses.
        let v: i128 = -48_000;
        let y = serde_yaml::to_string(&v).unwrap();
        let back: i128 = serde_yaml::from_str(&y).unwrap();
        assert_eq!(back, v);
    }
}
