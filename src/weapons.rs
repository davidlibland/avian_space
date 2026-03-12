use crate::Layer;
use crate::ship::Ship;
use avian2d::prelude::*;
use bevy::prelude::*;

pub const LASER_SPEED: f32 = 500.0;
pub const LASER_LIFETIME: f32 = 1.2;
pub const LASER_COOLDOWN: f32 = 0.25;

pub fn weapons_plugin(app: &mut App) {
    app.add_message::<FireCommand>().add_systems(
        Update,
        (weapon_fire, weapon_lifetime, weapon_system_cooldown),
    );
}

#[derive(Event, Message)]
pub struct FireCommand {
    pub ship: Entity,
    pub weapon_type: WeaponType,
}

#[derive(Clone)]
pub enum WeaponType {
    Laser,
}

/// A weapon fired by any ship.
///
/// `owner` is `None` for player weapons, or `Some(owner_entity)` for
/// enemy lasers so that hits can be credited to the correct trajectory.
#[derive(Component)]
pub struct Weapon {
    pub lifetime: f32,
    pub owner: Option<(Entity, usize)>,
    pub weapon_type: WeaponType,
}
#[derive(Component)]
pub struct WeaponSystems {
    pub primary: WeaponSystem,
}

impl Default for WeaponSystems {
    fn default() -> Self {
        WeaponSystems {
            primary: WeaponSystem::from_type(WeaponType::Laser),
        }
    }
}

pub struct WeaponSystem {
    pub cooldown: Timer,
    pub weapon_type: WeaponType,
}

impl WeaponSystem {
    pub fn from_type(weapon_type: WeaponType) -> Self {
        match weapon_type {
            WeaponType::Laser => WeaponSystem {
                weapon_type: WeaponType::Laser,
                cooldown: Timer::from_seconds(LASER_COOLDOWN, TimerMode::Repeating),
            },
        }
    }
}

pub fn weapon_fire(
    mut reader: MessageReader<FireCommand>,
    mut commands: Commands,
    ships: Query<(&Transform, &Ship)>,
) {
    for cmd in reader.read() {
        let Ok((ship_transform, ship)) = ships.get(cmd.ship) else {
            continue;
        };
        let forward = ship_transform.rotation * Vec3::Y;
        let tip = ship_transform.translation + forward * 20.0;
        let vel = forward.truncate() * LASER_SPEED;
        commands.spawn((
            Weapon {
                lifetime: LASER_LIFETIME,
                owner: None,
                weapon_type: cmd.weapon_type.clone(),
            },
            Collider::circle(10.),
            CollisionLayers::new(Layer::Weapon, [Layer::Asteroid, Layer::Ship]),
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

// pub fn old_weapon_fire(
//     mut commands: Commands,
//     keyboard: Res<ButtonInput<KeyCode>>,
//     time: Res<Time>,
//     mut cooldown: ResMut<LaserCooldown>,
//     ship_query: Query<(&Transform, &ActivePowerups), With<PlayerShip>>,
//     state: Res<State<GameState>>,
// ) {
//     cooldown.0.tick(time.delta());
//     if *state.get() != GameState::Playing {
//         return;
//     }
//     let Ok((ship_transform, powerups)) = ship_query.get_single() else {
//         return;
//     };
//     let has_rapid_fire = (powerups.flags & POWERUP_BIT_RAPID_FIRE) != 0;
//     let can_fire = if has_rapid_fire {
//         cooldown.0.elapsed_secs() >= RAPID_FIRE_COOLDOWN
//     } else {
//         cooldown.0.finished()
//     };
//     if keyboard.pressed(KeyCode::Space) && can_fire {
//         cooldown.0.reset();
//         let forward = ship_transform.rotation * Vec3::Y;
//         let tip = ship_transform.translation + forward * 20.0;
//         let vel = forward.truncate() * LASER_SPEED;
//         commands.spawn((
//             Laser {
//                 lifetime: LASER_LIFETIME,
//                 owner: None,
//             },
//             Velocity(vel),
//             Transform::from_translation(tip).with_rotation(ship_transform.rotation),
//             Sprite {
//                 color: Color::srgb(1.0, 0.8, 0.2),
//                 custom_size: Some(Vec2::new(3.0, 12.0)),
//                 ..default()
//             },
//         ));
//     }
// }

pub fn weapon_lifetime(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut Weapon)>,
) {
    let dt = time.delta_secs();
    for (entity, mut weapon) in &mut query {
        weapon.lifetime -= dt;
        if weapon.lifetime <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

pub fn weapon_system_cooldown(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<&mut WeaponSystems>,
) {
    for mut system in &mut query {
        system.primary.cooldown.tick(time.delta());
    }
}
