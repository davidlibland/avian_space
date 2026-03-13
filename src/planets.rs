// The plugin to describe planets
// Planets are static avian2d objects
// but they have sensors which allow us to know when a ship is over them
// They live in the planet layer.
use avian2d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::item_universe::ItemUniverse;
use crate::planet_ui::LandedContext;
use crate::{CurrentStarSystem, GameLayer, GameState, Player};

const PLANET_RADIUS: f32 = 60.0;

#[derive(Deserialize, Serialize)]
pub struct PlanetData {
    pub location: Vec2,
    pub description: String,
}

#[derive(Component)]
pub struct Planet(pub String);

/// Tracks which planet (if any) the player is currently overlapping.
#[derive(Resource, Default)]
pub struct NearbyPlanet(pub Option<Entity>);

pub fn planets_plugin(app: &mut App) {
    app.init_resource::<NearbyPlanet>()
        .add_systems(Startup, spawn_planets)
        .add_systems(
            Update,
            (track_nearby_planet, landing_input).run_if(in_state(GameState::Flying)),
        );
}

fn spawn_planets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    item_universe: Res<ItemUniverse>,
    current_star_system: Res<CurrentStarSystem>,
) {
    let Some(star_system) = item_universe.star_systems.get(&current_star_system.0) else {
        return;
    };

    for (planet_name, planet_data) in &star_system.planets {
        commands.spawn((
            Planet(planet_name.clone()),
            RigidBody::Static,
            Collider::circle(PLANET_RADIUS),
            CollisionLayers::new(GameLayer::Planet, [GameLayer::Ship, GameLayer::Radar]),
            CollisionEventsEnabled,
            Sensor,
            Mesh2d(meshes.add(Circle::new(PLANET_RADIUS))),
            MeshMaterial2d(materials.add(Color::srgb(0.2, 0.6, 0.3))),
            Transform::from_xyz(planet_data.location.x, planet_data.location.y, -1.0),
        ));
    }
}

fn track_nearby_planet(
    mut collision_starts: MessageReader<CollisionStart>,
    mut collision_ends: MessageReader<CollisionEnd>,
    planets: Query<(), With<Planet>>,
    players: Query<(), With<Player>>,
    mut nearby: ResMut<NearbyPlanet>,
) {
    for event in collision_starts.read() {
        let (a, b) = (event.collider1, event.collider2);
        if planets.contains(a) && players.contains(b) {
            nearby.0 = Some(a);
        } else if planets.contains(b) && players.contains(a) {
            nearby.0 = Some(b);
        }
    }
    for event in collision_ends.read() {
        let (a, b) = (event.collider1, event.collider2);
        if (planets.contains(a) && players.contains(b))
            || (planets.contains(b) && players.contains(a))
        {
            nearby.0 = None;
        }
    }
}

fn landing_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    nearby: Res<NearbyPlanet>,
    mut state: ResMut<NextState<GameState>>,
    mut landed_context: ResMut<LandedContext>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyL) {
        if let Some(planet_entity) = nearby.0 {
            landed_context.planet = Some(planet_entity);
            state.set(GameState::Landed);
        }
    }
}
