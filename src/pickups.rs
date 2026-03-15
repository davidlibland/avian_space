use avian2d::prelude::*;
use bevy::prelude::*;
use std::collections::HashSet;

use crate::item_universe::ItemUniverse;
use crate::ship::Ship;
use crate::utils::{random_velocity, safe_despawn};
use crate::{GameLayer, PlayState};
use rand::Rng;

const PICKUP_RADIUS: f32 = 15.0;

#[derive(Component)]
pub struct Pickup {
    pub commodity: String,
    pub quantity: u16,
}

#[derive(Event, Message)]
pub struct PickupDrop {
    pub location: Vec2,
    pub commodity: String,
    pub quantity: u16,
}

pub fn pickup_plugin(app: &mut App) {
    app.add_message::<PickupDrop>().add_systems(
        Update,
        (spawn_pickups, collect_pickups).run_if(in_state(PlayState::Flying)),
    );
}

fn spawn_pickups(
    mut reader: MessageReader<PickupDrop>,
    mut commands: Commands,
    item_universe: Res<ItemUniverse>,
) {
    let mut rng = rand::thread_rng();
    for drop in reader.read() {
        let [r, g, b] = item_universe
            .commodities
            .get(&drop.commodity)
            .map(|c| c.color)
            .unwrap_or([1.0, 0.85, 0.1]);
        commands.spawn((
            DespawnOnExit(PlayState::Flying),
            Pickup {
                commodity: drop.commodity.clone(),
                quantity: drop.quantity,
            },
            Transform::from_translation(drop.location.extend(0.0)),
            RigidBody::Dynamic,
            MassPropertiesBundle::from_shape(&Collider::circle(PICKUP_RADIUS), 1.0),
            LinearVelocity(random_velocity(30.0)),
            AngularVelocity(rng.gen_range(-2.0..2.0)),
            Collider::circle(PICKUP_RADIUS),
            Sensor,
            CollisionEventsEnabled,
            CollisionLayers::new(GameLayer::Pickup, [GameLayer::Ship]),
            Sprite {
                color: Color::srgb(r, g, b),
                custom_size: Some(Vec2::splat(10.0)),
                ..default()
            },
        ));
    }
}

fn collect_pickups(
    mut collisions: MessageReader<CollisionStart>,
    pickups: Query<&Pickup>,
    mut ships: Query<&mut Ship>,
    mut commands: Commands,
) {
    let mut collected: HashSet<Entity> = HashSet::new();

    for event in collisions.read() {
        let (pickup_entity, ship_entity) =
            if pickups.contains(event.collider1) && ships.contains(event.collider2) {
                (event.collider1, event.collider2)
            } else if pickups.contains(event.collider2) && ships.contains(event.collider1) {
                (event.collider2, event.collider1)
            } else {
                continue;
            };

        if !collected.insert(pickup_entity) {
            continue;
        }

        let Ok(pickup) = pickups.get(pickup_entity) else {
            continue;
        };
        let Ok(mut ship) = ships.get_mut(ship_entity) else {
            continue;
        };

        let space = ship.remaining_cargo_space();
        let qty = pickup.quantity.min(space);
        if qty > 0 {
            *ship.cargo.entry(pickup.commodity.clone()).or_insert(0) += qty;
        }

        safe_despawn(&mut commands, pickup_entity);
    }
}
