use super::*;
use crate::asteroids::AsteroidFieldData;
use crate::planets::PlanetData;
use crate::ship::ShipData;
use bevy::math::Vec2;
use std::collections::HashMap;

fn test_universe() -> ItemUniverse {
    let mut planets = HashMap::new();
    planets.insert(
        "earth".to_string(),
        PlanetData {
            display_name: String::new(),
            planet_type: String::new(),
            uncolonized: false,
            faction: String::new(),
            sprite_handle: Default::default(),
            location: Vec2::ZERO,
            description: String::new(),
            commodities: HashMap::from([
                ("iron".to_string(), 100_i128),
                ("gold".to_string(), 500_i128),
            ]),
            outfitter: vec!["missile_launcher".to_string()],
            shipyard: vec![],
            radius: 50.0,
            color: [0.0; 3],
        },
    );
    planets.insert(
        "mars".to_string(),
        PlanetData {
            display_name: String::new(),
            planet_type: String::new(),
            uncolonized: false,
            faction: String::new(),
            sprite_handle: Default::default(),
            location: Vec2::new(1000.0, 0.0),
            description: String::new(),
            commodities: HashMap::from([
                ("iron".to_string(), 200_i128),
                ("gold".to_string(), 300_i128),
            ]),
            outfitter: vec![],
            shipyard: vec![],
            radius: 40.0,
            color: [0.0; 3],
        },
    );

    let mut ships = HashMap::new();
    ships.insert(
        "corvette".to_string(),
        ShipData {
            display_name: String::new(),
            base_weapons: HashMap::from([("missile".to_string(), (1u8, Some(10u32)))]),
            ..Default::default()
        },
    );
    ships.insert(
        "shuttle".to_string(),
        ShipData {
            display_name: String::new(),
            base_weapons: HashMap::new(),
            ..Default::default()
        },
    );

    let system = StarSystem {
        display_name: String::new(),
        map_position: Vec2::ZERO,
        connections: vec![],
        planets,
        astroid_fields: vec![AsteroidFieldData {
            location: Vec2::new(500.0, 0.0),
            radius: 200.0,
            number: 10,
            commodities: HashMap::from([
                ("iron".to_string(), 0.8),
                ("gold".to_string(), 0.2),
            ]),
        }],
        ships: Default::default(),
    };

    let outfitter_items = HashMap::from([(
        "missile_launcher".to_string(),
        OutfitterItem::SecondaryWeapon {
            display_name: String::new(),
            required_unlocks: Vec::new(),
            price: 1000,
            space: 2,
            weapon_type: "missile".to_string(),
            ammo_price: 50,
            ammo_space: 1,
        },
    )]);

    let mut iu = ItemUniverse {
        weapons: HashMap::new(),
        ships,
        star_systems: HashMap::from([("sol".to_string(), system)]),
        outfitter_items,
        enemies: HashMap::new(),
        starting_ship: "shuttle".to_string(),
        starting_system: "sol".to_string(),
        commodities: HashMap::new(),
        missions: HashMap::new(),
        mission_templates: HashMap::new(),
        global_average_price: HashMap::new(),
        system_commodity_best_planet_to_sell: HashMap::new(),
        system_planet_best_commodity_to_buy: HashMap::new(),
        planet_best_margin: HashMap::new(),
        planet_has_ammo_for: HashMap::new(),
        asteroid_field_expected_value: HashMap::new(),
        ship_credit_scale: HashMap::new(),
        allies: HashMap::new(),
    };
    iu.compute_global_averages();
    iu.compute_trade_maps();
    iu.compute_planet_ammo();
    iu.compute_asteroid_values();
    iu.compute_ship_credit_scales();
    iu
}

#[test]
fn test_global_average_price() {
    let iu = test_universe();
    // iron: (100 + 200) / 2 = 150
    let iron_avg = iu.global_average_price.get("iron").copied().unwrap();
    assert!((iron_avg - 150.0).abs() < 1e-6, "iron avg={iron_avg}");
    // gold: (500 + 300) / 2 = 400
    let gold_avg = iu.global_average_price.get("gold").copied().unwrap();
    assert!((gold_avg - 400.0).abs() < 1e-6, "gold avg={gold_avg}");
}

#[test]
fn test_planet_best_margin() {
    let iu = test_universe();
    let margins = iu.planet_best_margin.get("sol").unwrap();

    // Earth: iron price=100, mars iron=200 → margin = 100-200 = -100
    //        gold price=500, mars gold=300 → margin = 500-300 = +200
    //        best (most negative) margin = -100
    let earth_margin = margins.get("earth").copied().unwrap();
    assert!(
        (earth_margin - (-100.0)).abs() < 1e-6,
        "earth margin={earth_margin}"
    );

    // Mars: iron price=200, earth iron=100 → margin = 200-100 = +100
    //       gold price=300, earth gold=500 → margin = 300-500 = -200
    //       best (most negative) margin = -200
    let mars_margin = margins.get("mars").copied().unwrap();
    assert!(
        (mars_margin - (-200.0)).abs() < 1e-6,
        "mars margin={mars_margin}"
    );
}

#[test]
fn test_planet_has_ammo_for() {
    let iu = test_universe();

    // Earth has outfitter "missile_launcher" which is SecondaryWeapon type "missile".
    // Corvette has base_weapon "missile" → earth has ammo for corvette.
    assert!(
        iu.planet_has_ammo_for
            .get("earth")
            .map(|s| s.contains("corvette"))
            .unwrap_or(false),
        "earth should have ammo for corvette"
    );

    // Shuttle has no secondary weapons → earth does NOT have ammo for shuttle.
    assert!(
        !iu.planet_has_ammo_for
            .get("earth")
            .map(|s| s.contains("shuttle"))
            .unwrap_or(false),
        "earth should not have ammo for shuttle"
    );

    // Mars has no outfitter → no ammo for anyone.
    assert!(
        !iu.planet_has_ammo_for
            .get("mars")
            .map(|s| s.contains("corvette"))
            .unwrap_or(false),
        "mars should not have ammo for corvette"
    );
}

#[test]
fn test_asteroid_field_expected_value() {
    let iu = test_universe();
    let values = iu.asteroid_field_expected_value.get("sol").unwrap();
    assert_eq!(values.len(), 1, "sol has one asteroid field");

    // EV = 0.8 * avg_price(iron) + 0.2 * avg_price(gold)
    //    = 0.8 * 150.0 + 0.2 * 400.0
    //    = 120.0 + 80.0 = 200.0
    let ev = values[0];
    assert!(
        (ev - 200.0).abs() < 0.01,
        "asteroid field EV={ev}, expected ~200.0"
    );
}
