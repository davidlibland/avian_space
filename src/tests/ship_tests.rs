//! Tests for ship handling / movement basics: the damage→handling curve and
//! the drive-flame (exhaust) flag driven by `ShipCommand`s.

use super::*;
use bevy::time::TimeUpdateStrategy;
use std::time::Duration;

// ── handling_factor ─────────────────────────────────────────────────────────

fn ship_with_health(health: i32, max_health: i32) -> Ship {
    let mut ship = Ship::from_ship_data(
        &ShipData {
            max_health,
            ..Default::default()
        },
        "test_ship",
    );
    ship.health = health;
    ship
}

#[test]
fn handling_factor_full_health_is_one() {
    assert!((ship_with_health(100, 100).handling_factor() - 1.0).abs() < 1e-6);
}

#[test]
fn handling_factor_sublinear_sqrt_rolloff() {
    // 0.5 + 0.5*sqrt(frac): quarter health → 0.75, not 0.625 (linear would).
    let hf = ship_with_health(25, 100).handling_factor();
    assert!((hf - 0.75).abs() < 1e-6, "expected 0.75, got {hf}");
}

#[test]
fn handling_factor_floors_at_half() {
    assert!((ship_with_health(0, 100).handling_factor() - 0.5).abs() < 1e-6);
    // Negative health (overkill damage) must clamp, not NaN.
    let hf = ship_with_health(-30, 100).handling_factor();
    assert!((hf - 0.5).abs() < 1e-6, "negative health must clamp: {hf}");
}

#[test]
fn handling_factor_zero_max_health_no_panic() {
    let hf = ship_with_health(0, 0).handling_factor();
    assert!(hf.is_finite(), "max_health 0 must not divide by zero: {hf}");
}

// ── DriveActive (exhaust plume) ──────────────────────────────────────────────

fn movement_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
            16,
        )))
        .add_message::<ShipCommand>()
        .add_systems(Update, ship_movement);
    // Ship::default() is the all-zero sentinel — give it real drive params so
    // the PD thrust controller actually produces force.
    let ship = Ship::from_ship_data(
        &ShipData {
            thrust: 200.0,
            max_speed: 300.0,
            torque: 20.0,
            max_health: 100,
            thrust_kp: 5.0,
            thrust_kd: 1.0,
            ..Default::default()
        },
        "test_ship",
    );
    let entity = app
        .world_mut()
        .spawn((
            ship,
            DriveActive(false),
            RigidBody::Dynamic,
            LinearVelocity(Vec2::ZERO),
            AngularVelocity(0.0),
            Transform::default(),
        ))
        .id();
    app.update(); // establish time baseline
    (app, entity)
}

fn send_command(app: &mut App, entity: Entity, thrust: f32) {
    app.world_mut().write_message(ShipCommand {
        entity,
        thrust,
        turn: 0.0,
        reverse: 0.0,
    });
}

/// Regression: the plume must track the LAST command — including an all-zero
/// one. (The player input layer once skipped sending zero-input commands, so
/// the flame stayed latched on after the throttle was released.)
#[test]
fn drive_flag_follows_thrust_commands() {
    let (mut app, entity) = movement_app();

    send_command(&mut app, entity, 1.0);
    app.update();
    assert!(
        app.world().get::<DriveActive>(entity).unwrap().0,
        "thrust command must light the drive flame"
    );

    send_command(&mut app, entity, 0.0);
    app.update();
    assert!(
        !app.world().get::<DriveActive>(entity).unwrap().0,
        "zero-thrust command must clear the drive flame"
    );
}

#[test]
fn thrust_accelerates_forward() {
    let (mut app, entity) = movement_app();
    send_command(&mut app, entity, 1.0);
    app.update();
    let vel = app.world().get::<LinearVelocity>(entity).unwrap().0;
    // Transform::default() faces +Y.
    assert!(
        vel.y > 0.0,
        "forward thrust must accelerate +Y, got {vel:?}"
    );
    assert!(vel.x.abs() < 1e-4);
}

// ── Ship mods ────────────────────────────────────────────────────────────────

mod mods {
    use super::*;
    use std::path::Path;

    fn universe() -> crate::item_universe::ItemUniverse {
        let mut iu: crate::item_universe::ItemUniverse =
            crate::item_universe::parse_dir(Path::new("assets")).expect("assets/ must parse");
        iu.finalize();
        iu
    }

    fn modded_ship() -> Ship {
        Ship::from_ship_data(
            &ShipData {
                max_speed: 200.0,
                thrust: 100.0,
                torque: 40.0,
                max_health: 100,
                item_space: 30,
                ..Default::default()
            },
            "test_ship",
        )
    }

    #[test]
    fn engine_mod_scales_speed_thrust_torque() {
        let iu = universe();
        let mut ship = modded_ship();
        ship.credits = 100_000;
        assert_eq!(ship.max_speed(), 200.0);
        ship.buy_mod("engine_mk2", &iu, 1.0);
        assert_eq!(ship.mods.get("engine_mk2"), Some(&1));
        assert!((ship.max_speed() - 200.0 * 1.2).abs() < 1e-3);
        assert!((ship.thrust() - 100.0 * 1.25).abs() < 1e-3);
        assert!((ship.torque() - 40.0 * 1.2).abs() < 1e-3);
        // Stacks multiplicatively.
        ship.buy_mod("engine_mk2", &iu, 1.0);
        assert!((ship.max_speed() - 200.0 * 1.2 * 1.2).abs() < 1e-3);
    }

    #[test]
    fn armor_raises_max_health_and_stabilizes_handling() {
        let iu = universe();
        let mut ship = modded_ship();
        ship.credits = 100_000;
        ship.buy_mod("armor_plating", &iu, 1.0);
        assert_eq!(ship.max_health(), 140);
        assert_eq!(ship.health, 100, "buying armor must not heal");
        // After a full repair, the same ABSOLUTE damage is a smaller fraction
        // of the armored hull, so handling degrades less. (Unrepaired, fresh
        // armor actually lowers the health fraction — repair to benefit.)
        ship.health = ship.max_health() - 50; // repaired, then took 50 damage
        let armored_hf = ship.handling_factor();
        let mut bare = modded_ship();
        bare.health = bare.max_health() - 50;
        let bare_hf = bare.handling_factor();
        assert!(
            armored_hf > bare_hf,
            "armor must stabilize handling after repair: {armored_hf} <= {bare_hf}"
        );
    }

    #[test]
    fn selling_armor_clamps_current_health() {
        let iu = universe();
        let mut ship = modded_ship();
        ship.credits = 100_000;
        ship.buy_mod("armor_plating", &iu, 1.0);
        ship.health = 140; // fully repaired with armor
        ship.sell_mod("armor_plating", &iu);
        assert_eq!(ship.health, 100, "health must clamp to the bare hull");
        assert!(ship.mods.is_empty());
    }

    #[test]
    fn buy_mod_respects_credits_and_item_space() {
        let iu = universe();
        let mut ship = modded_ship();
        ship.credits = 10; // can't afford anything
        ship.buy_mod("engine_mk2", &iu, 1.0);
        assert!(ship.mods.is_empty(), "must refuse without credits");

        ship.credits = 1_000_000;
        // Fill the hold so no item space remains.
        ship.mod_space = ship.data.item_space as i32;
        ship.buy_mod("engine_mk2", &iu, 1.0);
        assert!(ship.mods.is_empty(), "must refuse without item space");
    }

    #[test]
    fn mods_consume_item_space() {
        let iu = universe();
        let mut ship = modded_ship();
        ship.credits = 100_000;
        let before = ship.remaining_item_space();
        ship.buy_mod("armor_plating", &iu, 1.0); // space 6
        assert_eq!(ship.remaining_item_space(), before - 6);
        ship.sell_mod("armor_plating", &iu);
        assert_eq!(ship.remaining_item_space(), before);
    }
}

// ── Repair bot ───────────────────────────────────────────────────────────────

#[test]
fn repair_bot_heals_toward_effective_max() {
    use bevy::time::TimeUpdateStrategy;
    use std::time::Duration;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
            250,
        )))
        .add_systems(Update, repair_bot_tick);
    let mut ship = Ship::from_ship_data(
        &ShipData {
            max_health: 100,
            ..Default::default()
        },
        "test_ship",
    );
    ship.health = 90;
    ship.mod_stats.repair_per_sec = 2.0; // 0.5 hp per 250ms tick
    let entity = app.world_mut().spawn((ship, RepairBuffer::default())).id();
    app.update(); // time baseline
    for _ in 0..8 {
        app.update(); // 2s of repair = +4 hp
    }
    let healed = app.world().get::<Ship>(entity).unwrap().health;
    assert!(
        (93..=95).contains(&healed),
        "expected ~94 hp after 2s at 2 hp/s, got {healed}"
    );
    // Runs to cap, never beyond.
    for _ in 0..200 {
        app.update();
    }
    assert_eq!(app.world().get::<Ship>(entity).unwrap().health, 100);
}

#[test]
fn weapon_mounts_limit_guns_and_turrets() {
    let iu: crate::item_universe::ItemUniverse = {
        let mut iu: crate::item_universe::ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        iu
    };
    // A fighter: 3 gun mounts, no turret ring, ships with 2 lasers.
    let data = iu.ships.get("fighter").unwrap();
    assert_eq!((data.gun_mounts, data.turret_mounts), (3, 0));
    let mut ship = Ship::from_ship_data(data, "fighter");
    ship.weapon_systems = crate::weapons::WeaponSystems::build(&data.base_weapons, &iu);
    ship.credits = 1_000_000;
    assert_eq!(ship.mounts_used(), (2, 0));

    // One more laser fits; the fourth gun does not.
    ship.buy_weapon("laser", &iu, 1.0);
    assert_eq!(ship.mounts_used(), (3, 0));
    let credits = ship.credits;
    ship.buy_weapon("laser", &iu, 1.0);
    assert_eq!(
        ship.mounts_used(),
        (3, 0),
        "no fourth gun on a 3-mount hull"
    );
    assert_eq!(ship.credits, credits, "refused purchases charge nothing");

    // No turret ring on a fighter, whatever the credits.
    ship.buy_weapon("laser_turret", &iu, 1.0);
    assert_eq!(
        ship.mounts_used(),
        (3, 0),
        "small hulls can't mount turrets"
    );
    assert_eq!(ship.credits, credits);

    // Decoy pods use item space, not mounts.
    ship.buy_weapon("flare_pod", &iu, 1.0);
    assert!(ship.weapon_systems.find_weapon("flare_pod").is_some());
    assert_eq!(
        ship.mounts_used(),
        (3, 0),
        "countermeasures don't take mounts"
    );

    // A carrier has real turret rings.
    let carrier = iu.ships.get("fed_carrier").unwrap();
    let mut big = Ship::from_ship_data(carrier, "fed_carrier");
    big.weapon_systems = crate::weapons::WeaponSystems::build(&carrier.base_weapons, &iu);
    big.credits = 1_000_000;
    let (_, turrets) = big.mounts_used();
    assert_eq!(turrets, 2, "factory fit: two proton turrets");
    // Mount-wise there's room for more turrets (item space is the binding
    // constraint at factory fit, which is its own limit).
    let turret = iu.weapons.get("proton_beam_turret").unwrap();
    assert!(
        big.mount_free_for(turret),
        "carriers have spare turret rings"
    );
    let small = iu.ships.get("fighter").unwrap();
    let s2 = Ship::from_ship_data(small, "fighter");
    assert!(!s2.mount_free_for(turret) || small.turret_mounts > 0);
}
