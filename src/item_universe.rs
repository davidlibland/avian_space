// This file loads configs for the weapons, ships, planets, etc.
use bevy::prelude::*;
use std::collections::HashMap;

use crate::planets::PlanetData;
use crate::weapons::Weapon;

#[derive(Resource)]
pub struct ItemUniverse {
    pub weapons: HashMap<String, Weapon>,
    pub star_systems: HashMap<String, StarSystem>,
}

pub struct StarSystem {
    pub planets: Vec<PlanetData>,
    pub astroid_density: f32,
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
                    astroid_density: 1.0,
                },
            )]),
        }
    }
}

pub fn item_universe_plugin(app: &mut App) {
    app.init_resource::<ItemUniverse>();
}
