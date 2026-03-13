use crate::GameLayer;
use crate::item_universe::ItemUniverse;
use crate::ship::Ship;
use avian2d::prelude::*;
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
    pub primary: Vec<WeaponSystem>,
}

impl Default for WeaponSystems {
    fn default() -> Self {
        WeaponSystems { primary: vec![] }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Weapon {
    pub name: String,
    pub lifetime: f32,
    pub speed: f32,
    pub cooldown: f32,
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
            cooldown: Timer::from_seconds(weapon.cooldown / number as f32, TimerMode::Repeating),
            number: number,
        });
    }
}

pub fn weapon_fire(
    mut reader: MessageReader<FireCommand>,
    mut commands: Commands,
    ships: Query<(&Transform, &Ship)>,
    item_universe: Res<ItemUniverse>,
) {
    for cmd in reader.read() {
        let Ok((ship_transform, ship)) = ships.get(cmd.ship) else {
            continue;
        };
        let Some(weapon) = item_universe.weapons.get(&cmd.weapon_type) else {
            continue;
        };
        let forward = ship_transform.rotation * Vec3::Y;
        let tip = ship_transform.translation + forward * 20.0;
        let vel = forward.truncate() * weapon.speed;
        commands.spawn((
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

pub fn weapon_lifetime(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut Projectile)>,
) {
    let dt = time.delta_secs();
    for (entity, mut weapon) in &mut query {
        weapon.lifetime -= dt;
        if weapon.lifetime <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

pub fn weapon_system_cooldown(time: Res<Time>, mut query: Query<&mut WeaponSystems>) {
    for mut system in &mut query {
        for specific in system.primary.iter_mut() {
            specific
                .cooldown
                .tick(time.delta() * (specific.number as u32));
        }
    }
}
