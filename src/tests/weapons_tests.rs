//! Weapon-system timing tests against the real asset data.

use super::*;
use std::path::Path;

fn real_universe() -> ItemUniverse {
    crate::item_universe::parse_dir(Path::new("assets")).expect("assets/ must parse")
}

/// Regression: `from_type` used `cooldown / number` as the timer duration
/// while `weapon_system_cooldown` ALSO ticks the timer `number`× faster, so
/// N baked-in copies fired every cooldown/N² — and diverged from weapons
/// bought one at a time (which never rebuilt the timer). The duration must
/// be the full cooldown regardless of copy count.
#[test]
fn cooldown_duration_independent_of_copy_count() {
    let iu = real_universe();
    let laser = iu.weapons.get("laser").expect("laser in weapons.yaml");
    for n in [1u8, 2, 4] {
        let ws = WeaponSystem::from_type("laser", n, None, &iu).unwrap();
        assert!(
            (ws.cooldown.duration().as_secs_f32() - laser.cooldown).abs() < 1e-6,
            "{n} copies: duration {} != weapon cooldown {} — copy count must \
             only scale the tick rate, not the duration",
            ws.cooldown.duration().as_secs_f32(),
            laser.cooldown
        );
    }
}

/// A fresh weapon system starts ready to fire — no waiting out a full
/// cooldown (up to ~9 s for missiles) on a fresh ship or fresh purchase.
#[test]
fn fresh_weapon_starts_ready() {
    let iu = real_universe();
    for wt in ["laser", "javelin"] {
        let ws = WeaponSystem::from_type(wt, 1, Some(4), &iu).unwrap();
        assert!(
            ws.cooldown.is_finished(),
            "{wt}: newly built weapon system must start off cooldown"
        );
    }
}

// ── Guided-missile launch + drag model ───────────────────────────────────────

use avian2d::prelude::{LinearVelocity, Position};
use bevy::time::TimeUpdateStrategy;
use std::time::Duration;

/// Fire a javelin from a moving ship and inspect the spawned projectile:
/// launch velocity must be `forward·speed + ship_vel·MISSILE_LAUNCH_INHERIT`,
/// and the drag rate must satisfy the settle boundary condition
/// `r = ln(1/ε)/lifetime` (excess speed decays to ε by end of lifetime).
#[test]
fn guided_launch_inherits_partial_velocity_with_settling_drag() {
    let iu = real_universe();
    let javelin = iu
        .weapons
        .get("javelin")
        .expect("javelin in weapons.yaml")
        .clone();
    assert!(javelin.is_guided(), "test premise: javelin is guided");

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(iu)
        .add_message::<FireCommand>()
        .add_message::<WeaponFired>()
        .add_message::<DecoyDeployed>()
        .add_message::<crate::carrier::SpawnEscort>()
        .add_systems(Update, weapon_fire);

    let ship_vel = Vec2::new(120.0, 40.0);
    let mut ship = crate::ship::Ship::default();
    ship.weapon_systems = WeaponSystems::build(
        &HashMap::from([("javelin".to_string(), (1u8, Some(4u32)))]),
        app.world().resource::<ItemUniverse>(),
    );
    let shooter = app
        .world_mut()
        .spawn((
            ship,
            Transform::default(), // nose +Y
            Position(Vec2::ZERO),
            LinearVelocity(ship_vel),
        ))
        .id();

    app.world_mut().write_message(FireCommand {
        ship: shooter,
        weapon_type: "javelin".into(),
        target: None,
    });
    app.update();

    let world = app.world_mut();
    let mut q = world.query::<(&LinearVelocity, &GuidedMissile)>();
    let (vel, missile) = q.single(world).expect("one projectile spawned");

    let expected = Vec2::Y * javelin.speed + ship_vel * MISSILE_LAUNCH_INHERIT;
    assert!(
        (vel.0 - expected).length() < 1e-3,
        "launch velocity {:?} != forward·speed + ship_vel·{MISSILE_LAUNCH_INHERIT} = {expected:?}",
        vel.0
    );
    let expected_rate = (1.0 / MISSILE_SETTLE_FRACTION).ln() / javelin.lifetime;
    assert!(
        (missile.drag_rate - expected_rate).abs() < 1e-4,
        "drag rate {} != ln(1/ε)/lifetime = {expected_rate}",
        missile.drag_rate
    );
    assert!(
        (missile.cruise_speed - javelin.speed).abs() < 1e-6,
        "cruise speed must be the weapon's own speed"
    );
}

/// The drag model itself: an over-speed missile must decay monotonically
/// toward cruise, reaching (within ε) cruise speed by the end of its lifetime
/// — the "a fast ship can eventually outrun a missile" design invariant.
#[test]
fn missile_drag_settles_to_cruise_within_lifetime() {
    let cruise = 160.0_f32;
    let lifetime = 3.0_f32;
    let v0 = 400.0_f32; // big inherited boost
    let drag_rate = (1.0 / MISSILE_SETTLE_FRACTION).ln() / lifetime;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
            50,
        )))
        .add_systems(Update, missile_guidance);
    app.world_mut().spawn((
        Projectile {
            lifetime,
            owner: Entity::PLACEHOLDER,
            weapon_type: "test".into(),
        },
        GuidedMissile {
            target: None,
            turn_rate: 2.0,
            cruise_speed: cruise,
            drag_rate,
        },
        Transform::default(),
        LinearVelocity(Vec2::Y * v0),
    ));
    app.update(); // time baseline

    let mut last = v0;
    let steps = (lifetime / 0.05) as usize;
    for _ in 0..steps {
        app.update();
        let world = app.world_mut();
        let mut q = world.query::<(&LinearVelocity, &GuidedMissile)>();
        let (vel, _) = q.single(world).unwrap();
        let speed = vel.0.length();
        assert!(
            speed <= last + 1e-3,
            "speed must decay monotonically: {speed} > {last}"
        );
        assert!(
            speed >= cruise - 1e-3,
            "speed must never drop below cruise: {speed} < {cruise}"
        );
        last = speed;
    }
    // After a full lifetime the excess must have settled to ~ε of the boost.
    let allowed = cruise + (v0 - cruise) * MISSILE_SETTLE_FRACTION * 1.5;
    assert!(
        last <= allowed,
        "after {lifetime}s the speed ({last}) must be near cruise ({cruise}); \
         allowed {allowed}"
    );
}

/// Shift-click "sell all" must pay out every round at the listed ammo price
/// and leave the rack empty (but keep the launcher + ammo tracking).
#[test]
fn sell_all_ammo_empties_rack_and_pays() {
    let iu = real_universe();
    let ammo_price = match iu.outfitter_items.get("javelin").unwrap() {
        crate::item_universe::OutfitterItem::SecondaryWeapon { ammo_price, .. } => *ammo_price,
        _ => panic!("javelin must be a secondary weapon"),
    };
    let mut ship = crate::ship::Ship::default();
    ship.weapon_systems = WeaponSystems::build(
        &HashMap::from([("javelin".to_string(), (1u8, Some(4u32)))]),
        &iu,
    );
    let before = ship.credits;
    ship.sell_all_ammo("javelin", &iu);
    assert_eq!(ship.credits, before + 4 * ammo_price);
    assert_eq!(
        ship.weapon_systems.find_weapon("javelin").unwrap().ammo_quantity,
        Some(0),
        "rack empty, launcher retained"
    );
    // Selling again is a no-op, not a payout.
    let before = ship.credits;
    ship.sell_all_ammo("javelin", &iu);
    assert_eq!(ship.credits, before);
}

// ── Decoy flares ─────────────────────────────────────────────────────────────

/// Firing a flare pod spawns a Decoy and, at strength 1.0, every missile
/// homing on the launcher retargets to the flare; at 0.0 none do.
#[test]
fn decoy_retargets_inbound_missiles() {
    for (strength, expect_spoofed) in [(1.0_f32, true), (0.0_f32, false)] {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_message::<DecoyDeployed>()
            .add_systems(Update, decoy_missiles);
        let owner = app.world_mut().spawn_empty().id();
        let flare = app.world_mut().spawn(Decoy).id();
        let missile = app
            .world_mut()
            .spawn(GuidedMissile {
                target: Some(owner),
                turn_rate: 2.0,
                cruise_speed: 100.0,
                drag_rate: 1.0,
            })
            .id();
        // A missile chasing someone ELSE must never be affected.
        let other = app.world_mut().spawn_empty().id();
        let bystander = app
            .world_mut()
            .spawn(GuidedMissile {
                target: Some(other),
                turn_rate: 2.0,
                cruise_speed: 100.0,
                drag_rate: 1.0,
            })
            .id();

        app.world_mut().write_message(DecoyDeployed {
            owner,
            flare,
            strength,
        });
        app.update();

        let target = app.world().get::<GuidedMissile>(missile).unwrap().target;
        if expect_spoofed {
            assert_eq!(target, Some(flare), "strength 1.0 must always spoof");
        } else {
            assert_eq!(target, Some(owner), "strength 0.0 must never spoof");
        }
        assert_eq!(
            app.world().get::<GuidedMissile>(bystander).unwrap().target,
            Some(other),
            "missiles chasing other ships are unaffected"
        );
    }
}

/// Firing the flare_pod weapon (real assets) spawns a Decoy entity.
#[test]
fn flare_pod_fires_a_decoy() {
    let iu = real_universe();
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(iu)
        .add_message::<FireCommand>()
        .add_message::<WeaponFired>()
        .add_message::<DecoyDeployed>()
        .add_message::<crate::carrier::SpawnEscort>()
        .add_systems(Update, weapon_fire);
    let mut ship = crate::ship::Ship::default();
    ship.weapon_systems = WeaponSystems::build(
        &HashMap::from([("flare_pod".to_string(), (1u8, Some(4u32)))]),
        app.world().resource::<ItemUniverse>(),
    );
    let shooter = app
        .world_mut()
        .spawn((
            ship,
            Transform::default(),
            Position(Vec2::ZERO),
            LinearVelocity(Vec2::new(100.0, 0.0)),
        ))
        .id();
    app.world_mut().write_message(FireCommand {
        ship: shooter,
        weapon_type: "flare_pod".into(),
        target: None,
    });
    app.update();

    let world = app.world_mut();
    let mut q = world.query::<(&Decoy, &Projectile)>();
    let (_, proj) = q.single(world).expect("one decoy spawned");
    assert_eq!(proj.owner, shooter);
    // Ammo consumed.
    let ship = world.get::<crate::ship::Ship>(shooter).unwrap();
    assert_eq!(
        ship.weapon_systems
            .iter_all()
            .next()
            .unwrap()
            .1
            .ammo_quantity,
        Some(3)
    );
}
