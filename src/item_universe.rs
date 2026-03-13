// This file loads configs for the weapons, ships, planets, etc.
use bevy::prelude::*;
use std::collections::HashMap;

use crate::asteroids::AsteroidFieldData;
use crate::planets::PlanetData;
use crate::weapons::Weapon;

#[derive(Resource)]
pub struct ItemUniverse {
    pub weapons: HashMap<String, Weapon>,
    pub star_systems: HashMap<String, StarSystem>,
}

pub struct StarSystem {
    pub planets: Vec<PlanetData>,
    pub astroid_fields: Vec<AsteroidFieldData>,
    pub ships: u8,
}

impl Default for ItemUniverse {
    fn default() -> Self {
        ItemUniverse {
            weapons: HashMap::from([(
                "laser".to_string(),
                Weapon {
                    name: "laser".to_string(),
                    lifetime: 1.2,
                    speed: 500.0,
                    cooldown: 0.25,
                },
            )]),
            star_systems: HashMap::from([(
                "sol".to_string(),
                StarSystem {
                    planets: vec![PlanetData {
                        location: Vec2::new(500., 300.),
                    }],
                    astroid_fields: vec![AsteroidFieldData {
                        location: Vec2::new(500., 300.),
                        number: 10,
                        radius: 300.,
                    }],
                    ships: 3,
                },
            )]),
        }
    }
}

pub fn item_universe_plugin(app: &mut App) {
    app.init_resource::<ItemUniverse>();
}
