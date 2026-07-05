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
