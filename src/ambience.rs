//! Ambience director: the planet soundscape while the player is on foot.
//!
//! Two layers, both driven by what is actually near the player:
//!   * **beds** — looping, crossfaded: the biome's wind floor, water/lava
//!     that fades in with proximity (counted from terrain tiles), venue
//!     machinery inside mazes, and a crowd murmur in a staffed bar.
//!   * **one-shots** — positional critter calls (per fauna species, panned
//!     toward the animal with pitch jitter) and venue accents (mine drips,
//!     substation pipe pings).
//!
//! Sounds live in `assets/sounds/ambience/` and are synthesized in-house
//! (`scripts/synth_ambience.py`).

use bevy::audio::{AudioSink, AudioSource, PlaybackMode, PlaybackSettings, Volume};
use bevy::prelude::*;
use rand::Rng;

use crate::PlayState;
use crate::surface::{BuildingKind, FootstepData, TILE_PX, Walker};

/// Beds fade in/out over this long.
const BED_FADE_SECS: f32 = 1.8;
/// How often the bed mix is re-derived from the surroundings.
const SAMPLE_SECS: f32 = 0.8;
/// How often one-shot chances are rolled.
const SHOT_TICK_SECS: f32 = 0.5;
/// Tile radius scanned for water/lava beds.
const TERRAIN_RADIUS: i32 = 10;
/// Critters farther than this (in tiles) stay quiet.
const CALL_RADIUS_TILES: f32 = 8.0;

/// A looping ambience bed, fading toward `target` (0.0 → despawn).
#[derive(Component)]
struct AmbienceBed {
    key: &'static str,
    target: f32,
}

/// The desired bed mix for the current surroundings — pure policy, the
/// system wrapper only gathers the inputs.
pub(crate) fn bed_targets(
    state: &PlayState,
    interior_kind: Option<BuildingKind>,
    biome: &str,
    water_tiles: u32,
    lava_tiles: u32,
    bar_npcs: u32,
) -> Vec<(&'static str, f32)> {
    let mut beds: Vec<(&'static str, f32)> = Vec::new();
    match state {
        PlayState::Exploring => {
            beds.push(match biome {
                "garden" => ("wind_garden", 0.5),
                "ice" => ("wind_ice", 0.55),
                "desert" => ("wind_desert", 0.5),
                // Stations have no weather — life support instead of wind.
                "interior" => ("interior_hum", 0.45),
                _ => ("wind_rocky", 0.5),
            });
            if water_tiles > 0 {
                beds.push(("water_lap", 0.6 * (water_tiles as f32 / 30.0).min(1.0)));
            }
            if lava_tiles > 0 {
                beds.push(("lava_bubble", 0.65 * (lava_tiles as f32 / 30.0).min(1.0)));
            }
        }
        PlayState::Inside => match interior_kind {
            Some(BuildingKind::Mine) => beds.push(("mine_rumble", 0.6)),
            Some(BuildingKind::Substation) => beds.push(("substation_hum", 0.5)),
            Some(BuildingKind::Warehouse) => beds.push(("warehouse_fans", 0.5)),
            Some(BuildingKind::Bar) => {
                beds.push(("interior_hum", 0.3));
                if bar_npcs > 0 {
                    beds.push(("bar_murmur", (0.2 + 0.06 * bar_npcs as f32).min(0.5)));
                }
            }
            _ => beds.push(("interior_hum", 0.4)),
        },
        // In space / menus the surface soundscape fades out entirely.
        _ => {}
    }
    beds
}

/// The call sound for a fauna species: (file stem, volume, mean seconds
/// between calls). Silent species (butterflies, moths) return `None`.
fn species_call(key: &str) -> Option<(&'static str, f32, f32)> {
    Some(match key {
        "songbird" => ("bird_song", 0.4, 7.0),
        "deer" => ("deer_huff", 0.35, 15.0),
        "rabbit" | "snow_hare" => ("rustle", 0.3, 12.0),
        "fox" => ("fox_yip", 0.35, 18.0),
        "rock_monster" => ("stone_grumble", 0.45, 14.0),
        "lava_salamander" => ("lava_hiss", 0.4, 12.0),
        "ice_monster" => ("ice_groan", 0.45, 16.0),
        "petrel" => ("gull_cry", 0.4, 9.0),
        "sand_lizard" => ("skitter", 0.3, 10.0),
        "scarab" => ("skitter", 0.25, 8.0),
        "vulture" => ("vulture_caw", 0.4, 11.0),
        "rat" | "mine_rat" | "warehouse_rat" => ("rat_squeak", 0.35, 8.0),
        "cave_bat" => ("bat_flutter", 0.35, 7.0),
        "rock_crab" => ("crab_clack", 0.4, 9.0),
        "sweeper_bot" => ("bot_beep", 0.35, 10.0),
        "drone" | "inventory_drone" | "service_drone" => ("drone_whir", 0.3, 9.0),
        "pipe_gecko" => ("gecko_chirp", 0.35, 8.0),
        _ => return None,
    })
}

fn ambience_path(stem: &str) -> String {
    format!("sounds/ambience/{stem}.ogg")
}

/// Reconcile the bed entities with the desired mix for the surroundings.
fn direct_beds(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: Local<f32>,
    asset_server: Res<AssetServer>,
    state: Res<State<PlayState>>,
    interior: Option<Res<crate::surface::interiors::InteriorContext>>,
    landed: Res<crate::planet_ui::LandedContext>,
    current_system: Res<crate::CurrentStarSystem>,
    iu: Res<crate::item_universe::ItemUniverse>,
    footsteps: Res<FootstepData>,
    walker: Query<&Transform, With<Walker>>,
    bar_npcs: Query<
        (),
        Or<(
            With<crate::surface::interiors::Clerk>,
            With<crate::surface::interiors::HirePilot>,
            With<crate::surface_npc::MissionNpc>,
        )>,
    >,
    mut beds: Query<&mut AmbienceBed>,
) {
    *timer += time.delta_secs();
    if *timer < SAMPLE_SECS {
        return;
    }
    *timer = 0.0;

    let biome = landed
        .planet_name
        .as_deref()
        .and_then(|name| {
            iu.star_systems
                .get(&current_system.0)
                .and_then(|sys| sys.planets.get(name))
                .map(|pd| crate::world_assets::planet_type_to_biome(&pd.planet_type))
        })
        .unwrap_or("rocky");
    let (mut water, mut lava) = (0, 0);
    if *state.get() == PlayState::Exploring
        && let Ok(walker_tf) = walker.single()
    {
        let pos = walker_tf.translation.truncate();
        water = footsteps.count_terrain_near(pos, TERRAIN_RADIUS, "water");
        lava = footsteps.count_terrain_near(pos, TERRAIN_RADIUS, "lava");
    }
    let desired = bed_targets(
        state.get(),
        interior.map(|c| c.kind),
        biome,
        water,
        lava,
        bar_npcs.iter().count() as u32,
    );

    let mut missing: Vec<(&'static str, f32)> = desired.clone();
    for mut bed in &mut beds {
        if let Some(i) = missing.iter().position(|(key, _)| *key == bed.key) {
            let (_, target) = missing.swap_remove(i);
            if bed.target != target {
                bed.target = target;
            }
        } else if bed.target != 0.0 {
            bed.target = 0.0;
        }
    }
    for (key, target) in missing {
        commands.spawn((
            AudioPlayer::new(asset_server.load(ambience_path(key))),
            PlaybackSettings {
                mode: PlaybackMode::Loop,
                volume: Volume::Linear(0.0),
                ..default()
            },
            AmbienceBed { key, target },
        ));
    }
}

/// Walk bed volumes toward their targets; despawn fully-faded beds.
fn fade_beds(
    mut commands: Commands,
    time: Res<Time>,
    mut beds: Query<(Entity, &AmbienceBed, &mut AudioSink)>,
) {
    let step = time.delta_secs() / BED_FADE_SECS;
    for (entity, bed, mut sink) in &mut beds {
        let now = sink.volume().to_linear();
        let next = if now < bed.target {
            (now + step).min(bed.target)
        } else {
            (now - step).max(bed.target)
        };
        if bed.target == 0.0 && next <= 0.001 {
            commands.entity(entity).despawn();
        } else if next != now {
            sink.set_volume(Volume::Linear(next));
        }
    }
}

/// Spawn a positional one-shot with slight pitch jitter.
fn spawn_call(
    commands: &mut Commands,
    asset_server: &AssetServer,
    stem: &str,
    volume: f32,
    pos: Vec3,
) {
    let speed = 0.9 + rand::thread_rng().r#gen_range(0.0..0.2);
    commands.spawn((
        AudioPlayer::<AudioSource>(asset_server.load(ambience_path(stem))),
        PlaybackSettings {
            mode: PlaybackMode::Despawn,
            volume: Volume::Linear(volume),
            speed,
            ..default()
        }
        .with_spatial(true),
        Transform::from_translation(pos),
    ));
}

/// Roll critter calls for fauna near the walker, plus venue accents.
fn one_shots(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: Local<f32>,
    asset_server: Res<AssetServer>,
    state: Res<State<PlayState>>,
    interior: Option<Res<crate::surface::interiors::InteriorContext>>,
    fauna_world: Option<Res<crate::surface_fauna::FaunaWorld>>,
    fauna: Query<(&crate::surface_fauna::Fauna, &Transform)>,
    walker: Query<&Transform, With<Walker>>,
) {
    *timer += time.delta_secs();
    if *timer < SHOT_TICK_SECS {
        return;
    }
    *timer = 0.0;
    if !matches!(*state.get(), PlayState::Exploring | PlayState::Inside) {
        return;
    }
    let Ok(walker_tf) = walker.single() else {
        return;
    };
    let walker_pos = walker_tf.translation;
    let mut rng = rand::thread_rng();

    if let Some(world) = fauna_world {
        for (critter, tf) in &fauna {
            let dist = tf.translation.truncate().distance(walker_pos.truncate());
            if dist > CALL_RADIUS_TILES * TILE_PX {
                continue;
            }
            let Some((stem, volume, period)) = world
                .species_key(critter.species_idx())
                .and_then(species_call)
            else {
                continue;
            };
            if rng.r#gen_range(0.0..1.0) < SHOT_TICK_SECS / period {
                spawn_call(&mut commands, &asset_server, stem, volume, tf.translation);
            }
        }
    }

    // Venue accents: unseen drips and pipe knocks around the player.
    if *state.get() == PlayState::Inside {
        let accent = match interior.map(|c| c.kind) {
            Some(BuildingKind::Mine) => Some(("drip", 0.35, 6.0)),
            Some(BuildingKind::Substation) => Some(("pipe_ping", 0.3, 9.0)),
            _ => None,
        };
        if let Some((stem, volume, period)) = accent
            && rng.r#gen_range(0.0..1.0) < SHOT_TICK_SECS / period
        {
            let offset = Vec3::new(
                rng.r#gen_range(-5.0..5.0) * TILE_PX,
                rng.r#gen_range(-5.0..5.0) * TILE_PX,
                0.0,
            );
            spawn_call(
                &mut commands,
                &asset_server,
                stem,
                volume,
                walker_pos + offset,
            );
        }
    }
}

pub fn ambience_plugin(app: &mut App) {
    app.add_systems(Update, (direct_beds, fade_beds, one_shots).chain());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_biomes_pick_their_wind_floor() {
        for (biome, key) in [
            ("garden", "wind_garden"),
            ("ice", "wind_ice"),
            ("desert", "wind_desert"),
            ("rocky", "wind_rocky"),
            ("interior", "interior_hum"), // stations: life support, no wind
        ] {
            let beds = bed_targets(&PlayState::Exploring, None, biome, 0, 0, 0);
            assert_eq!(
                beds,
                vec![(
                    key,
                    if biome == "ice" {
                        0.55
                    } else if biome == "interior" {
                        0.45
                    } else {
                        0.5
                    }
                )]
            );
        }
    }

    #[test]
    fn water_and_lava_fade_in_with_tile_count() {
        let beds = bed_targets(&PlayState::Exploring, None, "garden", 15, 0, 0);
        let water = beds.iter().find(|(k, _)| *k == "water_lap").unwrap();
        assert!(water.1 > 0.0 && water.1 < 0.6);
        let beds = bed_targets(&PlayState::Exploring, None, "rocky", 0, 60, 0);
        let lava = beds.iter().find(|(k, _)| *k == "lava_bubble").unwrap();
        assert_eq!(lava.1, 0.65); // saturates
        // No water bed on a dry sample.
        assert!(!beds.iter().any(|(k, _)| *k == "water_lap"));
    }

    #[test]
    fn venues_pick_their_machinery_bed() {
        for (kind, key) in [
            (BuildingKind::Mine, "mine_rumble"),
            (BuildingKind::Substation, "substation_hum"),
            (BuildingKind::Warehouse, "warehouse_fans"),
            (BuildingKind::Market, "interior_hum"),
        ] {
            let beds = bed_targets(&PlayState::Inside, Some(kind), "garden", 0, 0, 0);
            assert_eq!(beds[0].0, key, "{kind:?}");
        }
    }

    #[test]
    fn bar_murmur_scales_with_patrons_and_needs_someone_present() {
        let empty = bed_targets(
            &PlayState::Inside,
            Some(BuildingKind::Bar),
            "garden",
            0,
            0,
            0,
        );
        assert!(!empty.iter().any(|(k, _)| *k == "bar_murmur"));
        let quiet = bed_targets(
            &PlayState::Inside,
            Some(BuildingKind::Bar),
            "garden",
            0,
            0,
            2,
        );
        let busy = bed_targets(
            &PlayState::Inside,
            Some(BuildingKind::Bar),
            "garden",
            0,
            0,
            8,
        );
        let vol = |beds: &[(&str, f32)]| beds.iter().find(|(k, _)| *k == "bar_murmur").unwrap().1;
        assert!(vol(&quiet) < vol(&busy));
        assert!(vol(&busy) <= 0.5);
    }

    #[test]
    fn space_and_menu_have_no_surface_beds() {
        assert!(bed_targets(&PlayState::Flying, None, "garden", 50, 50, 5).is_empty());
        assert!(bed_targets(&PlayState::MainMenu, None, "garden", 0, 0, 0).is_empty());
    }

    #[test]
    fn every_manifest_species_call_and_bed_file_exists() {
        // Species list mirrors assets/sprites/fauna/fauna_manifest.ron; the
        // silent ones are deliberate.
        for species in [
            "deer",
            "rabbit",
            "fox",
            "butterfly",
            "songbird",
            "rock_monster",
            "lava_salamander",
            "ember_moth",
            "ice_monster",
            "snow_hare",
            "petrel",
            "sand_lizard",
            "scarab",
            "vulture",
            "mine_rat",
            "rock_crab",
            "cave_bat",
            "warehouse_rat",
            "sweeper_bot",
            "inventory_drone",
            "pipe_gecko",
            "service_drone",
            "rat",
            "drone",
        ] {
            if let Some((stem, _, _)) = species_call(species) {
                let path = format!("assets/{}", ambience_path(stem));
                assert!(std::path::Path::new(&path).exists(), "missing {path}");
            }
        }
        for (key, _) in [
            bed_targets(&PlayState::Exploring, None, "garden", 99, 99, 0),
            bed_targets(&PlayState::Exploring, None, "ice", 0, 0, 0),
            bed_targets(&PlayState::Exploring, None, "desert", 0, 0, 0),
            bed_targets(&PlayState::Exploring, None, "rocky", 0, 99, 0),
            bed_targets(&PlayState::Exploring, None, "interior", 0, 0, 0),
            bed_targets(&PlayState::Inside, Some(BuildingKind::Mine), "", 0, 0, 0),
            bed_targets(
                &PlayState::Inside,
                Some(BuildingKind::Substation),
                "",
                0,
                0,
                0,
            ),
            bed_targets(
                &PlayState::Inside,
                Some(BuildingKind::Warehouse),
                "",
                0,
                0,
                0,
            ),
            bed_targets(&PlayState::Inside, Some(BuildingKind::Bar), "", 0, 0, 4),
        ]
        .concat()
        {
            let path = format!("assets/{}", ambience_path(key));
            assert!(std::path::Path::new(&path).exists(), "missing {path}");
        }
        // Venue accents.
        for stem in ["drip", "pipe_ping"] {
            let path = format!("assets/{}", ambience_path(stem));
            assert!(std::path::Path::new(&path).exists(), "missing {path}");
        }
    }
}
