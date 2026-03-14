use crate::item_universe::ItemUniverse;
use crate::ship::Ship;
use crate::utils::safe_despawn;
use crate::{GameLayer, GameState};
use avian2d::prelude::*;
use bevy::platform::collections::hash_map::OccupiedEntry;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub fn weapons_plugin(app: &mut App) {
    app.add_message::<FireCommand>().add_systems(
        Update,
        (weapon_fire, weapon_lifetime, weapon_system_cooldown),
    );
}

#[derive(Event, Message)]
pub struct FireCommand {
    pub ship: Entity,
    pub weapon_type: String,
}

/// A weapon fired by any ship.
///
/// `owner` is `None` for player weapons, or `Some(owner_entity)` for
/// enemy lasers so that hits can be credited to the correct trajectory.
#[derive(Component)]
pub struct Projectile {
    pub lifetime: f32,
    pub owner: Option<(Entity, usize)>,
    pub weapon_type: String,
}

#[derive(Component)]
pub struct WeaponSystems {
    pub primary: HashMap<String, WeaponSystem>,
}
impl WeaponSystems {
    fn find_weapon_entry(
        &mut self,
        weapon_type: &str,
    ) -> std::collections::hash_map::Entry<String, WeaponSystem> {
        self.primary.entry(weapon_type.to_string())
    }
    fn find_weapon(&mut self, weapon_type: &str) -> Option<&mut WeaponSystem> {
        match self.find_weapon_entry(weapon_type) {
            std::collections::hash_map::Entry::Vacant(_) => None,
            std::collections::hash_map::Entry::Occupied(mut view) => Some(view.into_mut()),
        }
    }
    // fn buy_weapon(
    //     &mut self,
    //     weapon_type: &str,
    //     ship: &mut Ship,
    //     item_universe: &Res<ItemUniverse>,
    // ) {
    //     let Some(weapon) = item_universe.weapons.get(weapon_type) else {
    //         return;
    //     };
    //     if weapon.price > ship.credits {
    //         return;
    //     }
    //     let maybe_index = self
    //         .primary
    //         .iter()
    //         .position(|w| w.weapon_type == weapon_type);
    //     if let Some(index) = maybe_index {
    //         self.primary[index].number += 1;
    //         ship.credits -= weapon.price;
    //     } else {
    //         if let Some(new_weapon) =
    //             WeaponSystem::from_type(weapon_type, 1, &item_universe.weapons)
    //         {
    //             self.primary.push(new_weapon);
    //             ship.credits -= weapon.price;
    //         }
    //     }
    // }
    // fn sell_weapon(
    //     &mut self,
    //     weapon_type: &str,
    //     ship: &mut Ship,
    //     item_universe: &Res<ItemUniverse>,
    // ) {
    //     let Some(weapon) = item_universe.weapons.get(weapon_type) else {
    //         return;
    //     };
    //     let maybe_index = self
    //         .primary
    //         .iter()
    //         .position(|w| w.weapon_type == weapon_type);
    //     if let Some(index) = maybe_index {
    //         ship.credits += weapon.price;
    //         self.primary[index].number -= 1;
    //         // If there are no more weapons, remove the weapon from the list
    //         if self.primary[index].number <= 0 {
    //             self.primary.remove(index);
    //         }
    //     }
    // }
}

impl Default for WeaponSystems {
    fn default() -> Self {
        WeaponSystems {
            primary: HashMap::new(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Weapon {
    pub name: String,
    pub lifetime: f32,
    pub speed: f32,
    pub cooldown: f32,
    pub price: i128,
}

impl Weapon {
    pub fn range(&self) -> f32 {
        self.speed * self.lifetime
    }
}
pub struct WeaponSystem {
    pub cooldown: Timer,
    pub weapon_type: String,
    pub number: u8,
}

impl WeaponSystem {
    pub fn from_type(
        weapon_type: &str,
        number: u8,
        weapons: &HashMap<String, Weapon>,
    ) -> Option<Self> {
        let Some(weapon) = weapons.get(weapon_type) else {
            return None;
        };
        return Some(WeaponSystem {
            weapon_type: weapon_type.to_string(),
            cooldown: Timer::from_seconds(weapon.cooldown / number as f32, TimerMode::Once),
            number: number,
        });
    }
}

pub fn weapon_fire(
    mut reader: MessageReader<FireCommand>,
    mut commands: Commands,
    mut ships: Query<(&Transform, &Ship, &mut WeaponSystems)>,
    item_universe: Res<ItemUniverse>,
) {
    for cmd in reader.read() {
        let Ok((ship_transform, ship, mut weapons_system)) = ships.get_mut(cmd.ship) else {
            continue;
        };
        let Some(specific) = weapons_system.find_weapon(&cmd.weapon_type) else {
            continue;
        };
        // Check that the weapon is ready to fire:
        if !specific.cooldown.is_finished() {
            continue;
        }
        specific.cooldown.reset();
        let Some(weapon) = item_universe.weapons.get(&cmd.weapon_type) else {
            continue;
        };
        let forward = ship_transform.rotation * Vec3::Y;
        let tip = ship_transform.translation + forward * 20.0;
        let vel = forward.truncate() * weapon.speed;
        commands.spawn((
            DespawnOnExit(GameState::Flying),
            Projectile {
                lifetime: weapon.lifetime,
                owner: None,
                weapon_type: cmd.weapon_type.clone(),
            },
            Collider::circle(10.),
            CollisionLayers::new(GameLayer::Weapon, [GameLayer::Asteroid, GameLayer::Ship]),
            RigidBody::Dynamic,
            LinearVelocity(vel),
            Transform::from_translation(tip).with_rotation(ship_transform.rotation),
            Sprite {
                color: Color::srgb(1.0, 0.8, 0.2),
                custom_size: Some(Vec2::new(3.0, 12.0)),
                ..default()
            },
        ));
    }
}

pub(crate) fn weapon_lifetime(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut Projectile)>,
) {
    let dt = time.delta_secs();
    for (entity, mut weapon) in &mut query {
        weapon.lifetime -= dt;
        if weapon.lifetime <= 0.0 {
            safe_despawn(&mut commands, entity);
        }
    }
}

pub fn weapon_system_cooldown(time: Res<Time>, mut query: Query<&mut WeaponSystems>) {
    for mut system in &mut query {
        for specific in system.primary.values_mut() {
            specific
                .cooldown
                .tick(time.delta() * (specific.number as u32));
        }
    }
}
