use crate::item_universe::ItemUniverse;
use crate::ship::Ship;
use crate::utils::safe_despawn;
use crate::{GameLayer, PlayState};
use avian2d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, hash_map::Entry};
use std::f32::consts::FRAC_PI_2;

pub fn weapons_plugin(app: &mut App) {
    app.add_message::<FireCommand>().add_systems(
        Update,
        (
            weapon_fire,
            weapon_lifetime,
            weapon_system_cooldown,
            missile_guidance,
        ),
    );
}

#[derive(Event, Message)]
pub struct FireCommand {
    pub ship: Entity,
    pub weapon_type: String,
    pub target: Option<Entity>,
}

/// A weapon fired by any ship.
///
/// `owner` is `None` for player weapons, or `Some(owner_entity)` for
/// enemy lasers so that hits can be credited to the correct trajectory.
#[derive(Component)]
pub struct Projectile {
    pub lifetime: f32,
    pub owner: Option<Entity>,
    pub weapon_type: String,
}

#[derive(Component)]
pub struct WeaponSystems {
    pub primary: HashMap<String, WeaponSystem>,
}
impl WeaponSystems {
    fn find_weapon_entry(&mut self, weapon_type: &str) -> Entry<'_, String, WeaponSystem> {
        self.primary.entry(weapon_type.to_string())
    }
    fn find_weapon(&mut self, weapon_type: &str) -> Option<&mut WeaponSystem> {
        match self.find_weapon_entry(weapon_type) {
            Entry::Vacant(_) => None,
            Entry::Occupied(view) => Some(view.into_mut()),
        }
    }
    pub fn buy_weapon(
        &mut self,
        weapon_type: &str,
        ship: &mut Ship,
        item_universe: &Res<ItemUniverse>,
    ) {
        let Some(weapon) = item_universe.weapons.get(weapon_type) else {
            return;
        };
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        if outfitter_item.price() > ship.credits {
            return;
        }
        if outfitter_item.space() > ship.remaining_item_space() {
            return;
        }
        match self.find_weapon_entry(weapon_type) {
            Entry::Occupied(view) => {
                view.into_mut().number += 1;
                ship.credits -= outfitter_item.price();
                ship.consumed_item_space += outfitter_item.space();
            }
            Entry::Vacant(vacant) => {
                if let Some(new_weapon) =
                    WeaponSystem::from_type(weapon_type, 1, &item_universe.weapons)
                {
                    vacant.insert(new_weapon);
                    ship.credits -= outfitter_item.price();
                    ship.consumed_item_space += outfitter_item.space();
                }
            }
        }
    }
    pub fn sell_weapon(
        &mut self,
        weapon_type: &str,
        ship: &mut Ship,
        item_universe: &Res<ItemUniverse>,
    ) {
        let Some(weapon) = item_universe.weapons.get(weapon_type) else {
            return;
        };
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        if let Some(weapon) = self.find_weapon(weapon_type) {
            ship.credits += outfitter_item.price();
            ship.consumed_item_space = ship
                .consumed_item_space
                .saturating_sub(outfitter_item.space());
            weapon.number -= 1;
            // If there are no more weapons, remove the weapon from the list
            self.primary.remove(weapon_type);
        }
    }
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
    pub color: [f32; 3],
    pub damage: i16,
    #[serde(default)]
    pub guided: bool,
    #[serde(default)]
    pub turn_rate: f32,
}

#[derive(Component)]
pub struct GuidedMissile {
    pub target: Option<Entity>,
    pub turn_rate: f32,
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
        let [r, g, b] = weapon.color;
        let mut entity_cmd = commands.spawn((
            DespawnOnExit(PlayState::Flying),
            Projectile {
                lifetime: weapon.lifetime,
                owner: Some(cmd.ship),
                weapon_type: cmd.weapon_type.clone(),
            },
            Collider::circle(10.),
            CollisionLayers::new(GameLayer::Weapon, [GameLayer::Asteroid, GameLayer::Ship]),
            RigidBody::Dynamic,
            LinearVelocity(vel),
            Transform::from_translation(tip).with_rotation(ship_transform.rotation),
            Sprite {
                color: Color::srgb(r, g, b),
                custom_size: Some(Vec2::new(3.0, 12.0)),
                ..default()
            },
        ));
        if weapon.guided {
            entity_cmd.insert(GuidedMissile {
                target: cmd.target,
                turn_rate: weapon.turn_rate,
            });
        }
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

fn missile_guidance(
    time: Res<Time>,
    mut missiles: Query<(
        &mut Transform,
        &mut LinearVelocity,
        &mut GuidedMissile,
        &Projectile,
    )>,
    all_positions: Query<&Position>,
    ships: Query<Entity, With<Ship>>,
) {
    let dt = time.delta_secs();
    for (mut transform, mut vel, mut missile, projectile) in &mut missiles {
        // Auto-assign nearest non-owner ship if no target yet.
        if missile.target.is_none() {
            let pos = transform.translation.xy();
            missile.target = ships
                .iter()
                .filter(|&e| projectile.owner.map_or(true, |o| e != o))
                .filter_map(|e| {
                    all_positions
                        .get(e)
                        .ok()
                        .map(|p| (e, (p.0 - pos).length_squared()))
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(e, _)| e);
        }

        let Some(target) = missile.target else {
            continue;
        };
        let Ok(target_pos) = all_positions.get(target) else {
            missile.target = None;
            continue;
        };

        let speed = vel.0.length();
        if speed < f32::EPSILON {
            continue;
        }
        let current_dir = vel.0 / speed;
        let to_target = (target_pos.0 - transform.translation.xy()).normalize_or_zero();

        let angle = current_dir.angle_to(to_target);
        let max_turn = missile.turn_rate * dt;
        let turn = angle.clamp(-max_turn, max_turn);
        let (sin_t, cos_t) = turn.sin_cos();
        let new_dir = Vec2::new(
            current_dir.x * cos_t - current_dir.y * sin_t,
            current_dir.x * sin_t + current_dir.y * cos_t,
        );
        vel.0 = new_dir * speed;

        // Rotate sprite to match travel direction (+Y is forward).
        let visual_angle = new_dir.x.atan2(new_dir.y);
        transform.rotation = Quat::from_rotation_z(-visual_angle);
    }
}
