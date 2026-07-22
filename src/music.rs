//! Music director: one looping track at a time, chosen from game context and
//! swapped with a crossfade. Context is *derived*, never scripted — the menu,
//! the faction controlling the current star system, the landed planet's
//! biome, or the interior venue (bar gets its own track; other interiors keep
//! the surface theme, ducked).
//!
//! Tracks live at `assets/music/<key>.ogg`. Slots with a finished
//! Band-in-a-Box render ALSO keep the original synthesized pad as
//! `<key>_pad.ogg`, and the director slowly alternates between the two —
//! song for a stretch, ambient chords for a stretch — with the usual
//! crossfade. See docs/music_and_ambience_plan.md and docs/biab_briefs/.

use bevy::audio::{AudioSink, PlaybackMode, PlaybackSettings, Volume};
use bevy::prelude::*;

use crate::PlayState;
use crate::surface::BuildingKind;

/// Crossfade duration between tracks (also fade-in from silence).
const FADE_SECS: f32 = 2.5;
/// Interiors other than the bar keep the surface theme at reduced level.
const INTERIOR_DUCK: f32 = 0.45;
/// How long each variant (render / pad) plays before rotating to the next.
const VARIANT_SECS: f32 = 160.0;

/// Slots that have BOTH a finished render (`<key>.ogg`) and the kept
/// synth-pad variant (`<key>_pad.ogg`). Update as renders arrive; a test
/// asserts this list matches the files on disk.
const RENDERED: &[&str] = &[
    "space_federation",
    "space_rebel",
    "space_freefrontier",
    "space_helios",
    "space_bastion",
    "space_order",
    "space_pirate",
];

/// File stems for a slot, in rotation order (render first).
fn variant_stems(slot: &str) -> Vec<String> {
    if RENDERED.contains(&slot) {
        vec![slot.to_string(), format!("{slot}_pad")]
    } else {
        vec![slot.to_string()]
    }
}

/// What should be playing: a track key (file stem) and a gain multiplier.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct MusicCue {
    pub key: &'static str,
    pub gain: f32,
}

/// A live (or fading-out) music sink. `target` is the gain this track is
/// fading toward; 0.0 means it is on its way out and is despawned on arrival.
#[derive(Component)]
struct MusicTrack {
    stem: String,
    target: f32,
}

/// Tracks how long the current slot has been playing, for variant rotation.
#[derive(Default)]
struct SlotClock {
    slot: String,
    elapsed: f32,
}

fn faction_track(controller: Option<&str>) -> &'static str {
    match controller {
        Some("Federation") => "space_federation",
        Some("Rebel") => "space_rebel",
        Some("Helios") => "space_helios",
        Some("Bastion") => "space_bastion",
        Some("Order") | Some("Precursor") => "space_order",
        Some("FreeFrontier") | Some("Merchant") | Some("Independent") => "space_freefrontier",
        // Pirate space and contested/unclaimed systems: lawless.
        _ => "space_pirate",
    }
}

fn surface_track(planet_type: Option<&str>) -> &'static str {
    match crate::world_assets::planet_type_to_biome(planet_type.unwrap_or("rocky")) {
        "garden" => "surface_garden",
        "ice" => "surface_ice",
        "desert" => "surface_desert",
        "interior" => "surface_interior",
        _ => "surface_rocky",
    }
}

/// Pure cue derivation — the whole "what plays when" policy in one place.
pub(crate) fn track_cue(
    state: &PlayState,
    controller: Option<&str>,
    planet_type: Option<&str>,
    interior_kind: Option<BuildingKind>,
) -> MusicCue {
    match state {
        PlayState::MainMenu => MusicCue {
            key: "menu",
            gain: 1.0,
        },
        PlayState::Flying | PlayState::Traveling => MusicCue {
            key: faction_track(controller),
            gain: 1.0,
        },
        PlayState::Landed | PlayState::Exploring => MusicCue {
            key: surface_track(planet_type),
            gain: 1.0,
        },
        PlayState::Inside => match interior_kind {
            Some(BuildingKind::Bar) => MusicCue {
                key: "bar",
                gain: 1.0,
            },
            _ => MusicCue {
                key: surface_track(planet_type),
                gain: INTERIOR_DUCK,
            },
        },
    }
}

/// Derive the cue from live game state and reconcile the track entities:
/// key change → fade the old out, spawn the new at zero volume; gain-only
/// change (entering/leaving a non-bar interior) → retarget in place.
fn direct_music(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    state: Res<State<PlayState>>,
    current_system: Res<crate::CurrentStarSystem>,
    galaxy: Res<crate::galaxy::GalaxyControl>,
    landed: Res<crate::planet_ui::LandedContext>,
    interior: Option<Res<crate::surface::interiors::InteriorContext>>,
    iu: Res<crate::item_universe::ItemUniverse>,
    time: Res<Time>,
    mut clock: Local<SlotClock>,
    mut tracks: Query<&mut MusicTrack>,
) {
    let controller = galaxy.controller(&current_system.0);
    let planet_type = landed.planet_name.as_deref().and_then(|name| {
        iu.star_systems
            .get(&current_system.0)
            .and_then(|sys| sys.planets.get(name))
            .map(|pd| pd.planet_type.as_str())
    });
    let cue = track_cue(
        state.get(),
        controller,
        planet_type,
        interior.map(|c| c.kind),
    );

    // Variant rotation: within one slot, alternate render ↔ synth pad on a
    // slow clock; entering a new slot always starts on the render.
    if clock.slot != cue.key {
        clock.slot = cue.key.to_string();
        clock.elapsed = 0.0;
    } else {
        clock.elapsed += time.delta_secs();
    }
    let variants = variant_stems(cue.key);
    let stem = &variants[(clock.elapsed / VARIANT_SECS) as usize % variants.len()];

    let mut already_playing = false;
    for mut track in &mut tracks {
        if &track.stem == stem {
            already_playing = true;
            if track.target != cue.gain {
                track.target = cue.gain;
            }
        } else if track.target != 0.0 {
            track.target = 0.0;
        }
    }
    if !already_playing {
        commands.spawn((
            AudioPlayer::new(asset_server.load(format!("music/{stem}.ogg"))),
            PlaybackSettings {
                mode: PlaybackMode::Loop,
                volume: Volume::Linear(0.0),
                ..default()
            },
            MusicTrack {
                stem: stem.clone(),
                target: cue.gain,
            },
        ));
    }
}

/// Walk each sink's volume toward `target * music_volume`; despawn tracks
/// that have fully faded out. The settings slider is applied here, so it
/// takes effect live.
fn fade_music(
    mut commands: Commands,
    time: Res<Time>,
    settings: Res<crate::settings::GameSettings>,
    mut tracks: Query<(Entity, &MusicTrack, &mut AudioSink)>,
) {
    let step = time.delta_secs() / FADE_SECS;
    for (entity, track, mut sink) in &mut tracks {
        let goal = track.target * settings.music_volume.clamp(0.0, 1.0);
        let now = sink.volume().to_linear();
        let next = if now < goal {
            (now + step).min(goal)
        } else {
            (now - step).max(goal)
        };
        if track.target == 0.0 && next <= 0.001 {
            commands.entity(entity).despawn();
        } else if next != now {
            sink.set_volume(Volume::Linear(next));
        }
    }
}

pub fn music_plugin(app: &mut App) {
    app.add_systems(Update, (direct_music, fade_music).chain());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_plays_menu_theme() {
        let cue = track_cue(&PlayState::MainMenu, Some("Federation"), None, None);
        assert_eq!(cue.key, "menu");
    }

    #[test]
    fn space_theme_follows_system_controller() {
        for (faction, key) in [
            (Some("Federation"), "space_federation"),
            (Some("Rebel"), "space_rebel"),
            (Some("Helios"), "space_helios"),
            (Some("Bastion"), "space_bastion"),
            (Some("Order"), "space_order"),
            (Some("Precursor"), "space_order"),
            (Some("FreeFrontier"), "space_freefrontier"),
            (Some("Merchant"), "space_freefrontier"),
            (Some("Pirate"), "space_pirate"),
            (None, "space_pirate"), // contested / unclaimed
        ] {
            let cue = track_cue(&PlayState::Flying, faction, None, None);
            assert_eq!(cue.key, key, "controller {faction:?}");
            assert_eq!(cue.gain, 1.0);
        }
        // Traveling (hyperspace flash) keeps space music going.
        let cue = track_cue(&PlayState::Traveling, Some("Rebel"), None, None);
        assert_eq!(cue.key, "space_rebel");
    }

    #[test]
    fn surface_theme_follows_planet_biome() {
        for (ptype, key) in [
            (Some("habitable"), "surface_garden"),
            (Some("icy_dwarf"), "surface_ice"),
            (Some("ice_giant"), "surface_ice"),
            (Some("desert"), "surface_desert"),
            (Some("rocky"), "surface_rocky"),
            (Some("station"), "surface_interior"),
            (Some("gas_giant"), "surface_interior"),
            (None, "surface_rocky"),
        ] {
            let cue = track_cue(&PlayState::Exploring, None, ptype, None);
            assert_eq!(cue.key, key, "planet_type {ptype:?}");
        }
        // Landed (pad UI) already plays the planet's theme.
        let cue = track_cue(&PlayState::Landed, None, Some("habitable"), None);
        assert_eq!(cue.key, "surface_garden");
    }

    #[test]
    fn bar_gets_its_own_track_other_interiors_duck_the_surface_theme() {
        let bar = track_cue(
            &PlayState::Inside,
            None,
            Some("habitable"),
            Some(BuildingKind::Bar),
        );
        assert_eq!((bar.key, bar.gain), ("bar", 1.0));

        let mine = track_cue(
            &PlayState::Inside,
            None,
            Some("rocky"),
            Some(BuildingKind::Mine),
        );
        assert_eq!(mine.key, "surface_rocky");
        assert!(mine.gain < 1.0);
    }

    #[test]
    fn every_cue_key_has_a_shipped_track_file() {
        let mut keys: Vec<&'static str> = vec!["menu", "bar"];
        for f in [
            Some("Federation"),
            Some("Rebel"),
            Some("Helios"),
            Some("Bastion"),
            Some("Order"),
            Some("FreeFrontier"),
            None,
        ] {
            keys.push(faction_track(f));
        }
        for p in [
            Some("habitable"),
            Some("icy_dwarf"),
            Some("desert"),
            Some("rocky"),
            Some("station"),
        ] {
            keys.push(surface_track(p));
        }
        for key in keys {
            for stem in variant_stems(key) {
                let path = format!("assets/music/{stem}.ogg");
                assert!(
                    std::path::Path::new(&path).exists(),
                    "missing music file {path}"
                );
            }
        }
    }

    #[test]
    fn rendered_list_matches_the_files_on_disk() {
        // Every `_pad.ogg` on disk must be listed in RENDERED and vice
        // versa — keeps the rotation list honest as renders arrive.
        let mut on_disk: Vec<String> = std::fs::read_dir("assets/music")
            .unwrap()
            .filter_map(|e| {
                let name = e.ok()?.file_name().into_string().ok()?;
                Some(name.strip_suffix("_pad.ogg")?.to_string())
            })
            .collect();
        on_disk.sort();
        let mut listed: Vec<String> = RENDERED.iter().map(|s| s.to_string()).collect();
        listed.sort();
        assert_eq!(on_disk, listed);
    }
}
