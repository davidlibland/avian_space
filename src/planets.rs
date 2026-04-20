// The plugin to describe planets
// Planets are static avian2d objects
// but they have sensors which allow us to know when a ship is over them
// They live in the planet layer.
use avian2d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::item_universe::ItemUniverse;
use crate::missions::PlayerLandedOnPlanet;
use crate::planet_ui::LandedContext;
use crate::ship::{ShipHostility, Target};
use crate::ship_anim::{self, ANIM_MIN_SCALE, PLANET_ANIM_DURATION, ScalingDown, ScalingUp, ScaleDownFinished, image_size};
use crate::{CurrentStarSystem, GameLayer, PlayState, Player, Ship};

use std::collections::HashMap;

#[derive(Deserialize, Serialize)]
pub struct PlanetData {
    pub location: Vec2,
    pub description: String,
    pub commodities: HashMap<String, i128>, // Map of commodities to prices
    pub outfitter: Vec<String>,
    #[serde(default)]
    pub shipyard: Vec<String>, // Ship types available for purchase
    pub radius: f32,
    pub color: [f32; 3],
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub planet_type: String,
    /// If true, the planet has no colony and ships cannot land on it.
    #[serde(default)]
    pub uncolonized: bool,
    /// Controlling faction (e.g. "Federation", "Rebel", "Independent").
    #[serde(default)]
    pub faction: String,
    #[serde(skip)]
    pub sprite_handle: Handle<Image>,
}

#[derive(Component)]
pub struct Planet(pub String);

/// Tracks which planet (if any) the player is currently overlapping.
#[derive(Resource, Default)]
pub struct NearbyPlanet(pub Option<Entity>);

/// Marker: the player ship is in the landing scale-down animation.
#[derive(Component)]
pub struct PlayerLanding;

pub fn planets_plugin(app: &mut App) {
    app.init_resource::<NearbyPlanet>()
        .add_systems(OnEnter(PlayState::Flying), spawn_planets)
        .add_systems(
            Update,
            (track_nearby_planet, landing_input, finish_player_landing)
                .run_if(in_state(PlayState::Flying)),
        );
}

fn spawn_planets(
    mut commands: Commands,
    item_universe: Res<ItemUniverse>,
    current_star_system: Res<CurrentStarSystem>,
) {
    let Some(star_system) = item_universe.star_systems.get(&current_star_system.0) else {
        return;
    };

    for (planet_name, planet_data) in &star_system.planets {
        let r = planet_data.radius;
        let [cr, cg, cb] = planet_data.color;
        let has_sprite = planet_data.sprite_handle.id() != Handle::<Image>::default().id();
        commands.spawn((
            DespawnOnExit(PlayState::Flying),
            Planet(planet_name.clone()),
            RigidBody::Static,
            Collider::circle(r),
            CollisionLayers::new(
                GameLayer::Planet,
                [GameLayer::Ship, GameLayer::Asteroid, GameLayer::Radar],
            ),
            CollisionEventsEnabled,
            Sensor,
            if has_sprite {
                Sprite::from_image(planet_data.sprite_handle.clone())
            } else {
                Sprite::from_color(Color::srgb(cr, cg, cb), Vec2::splat(r * 2.0))
            },
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
    mut landed_context: ResMut<LandedContext>,
    planet_query: Query<&Planet>,
    item_universe: Res<ItemUniverse>,
    current_star_system: Res<CurrentStarSystem>,
    mut player_query: Query<
        (Entity, &mut Ship, &ShipHostility, &Sprite),
        (With<Player>, Without<PlayerLanding>),
    >,
    mut commands: Commands,
    images: Res<Assets<Image>>,
) {
    if !keyboard_input.just_pressed(KeyCode::KeyL) {
        return;
    }
    let Some(planet_entity) = nearby.0 else {
        return;
    };
    let Ok(planet) = planet_query.get(planet_entity) else {
        return;
    };
    let Ok((entity, mut ship, hostility, sprite)) = player_query.single_mut() else {
        return;
    };

    // Always set nav_target so the comms module can display the right message
    ship.nav_target = Some(Target::Planet(planet_entity));

    // Check landing eligibility
    if let Some(system) = item_universe.star_systems.get(&current_star_system.0) {
        if let Some(planet_data) = system.planets.get(&planet.0) {
            if planet_data.uncolonized {
                return;
            }
            if !planet_data.faction.is_empty()
                && hostility
                    .0
                    .get(&planet_data.faction)
                    .copied()
                    .unwrap_or(0.0)
                    > 0.0
            {
                return;
            }
        }
    }

    landed_context.planet_name = Some(planet.0.clone());

    // Start scale-down animation; actual state transition happens in
    // finish_player_landing when the animation completes.
    let full_size = image_size(sprite, &images);
    commands.entity(entity).insert((
        PlayerLanding,
        ScalingDown {
            timer: Timer::from_seconds(PLANET_ANIM_DURATION, TimerMode::Once),
            full_size,
        },
    ));
}

/// When the player's landing scale-down animation finishes, transition to Landed.
fn finish_player_landing(
    mut reader: MessageReader<ScaleDownFinished>,
    player_q: Query<(), (With<Player>, With<PlayerLanding>)>,
    mut commands: Commands,
    mut state: ResMut<NextState<PlayState>>,
    mut landed_writer: MessageWriter<PlayerLandedOnPlanet>,
    landed_context: Res<LandedContext>,
) {
    for event in reader.read() {
        if player_q.get(event.entity).is_ok() {
            commands.entity(event.entity).remove::<PlayerLanding>();
            if let Some(ref planet_name) = landed_context.planet_name {
                landed_writer.write(PlayerLandedOnPlanet {
                    planet: planet_name.clone(),
                });
            }
            state.set(PlayState::Exploring);
        }
    }
}
