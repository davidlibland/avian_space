//! Ambient NPC dialogue, filled from live game data.
//!
//! The point of a "customer" NPC is that they are NOT advising you — they are
//! thinking out loud, and what they happen to mention is a real trade route on
//! a real planet. The player derives the run themselves instead of being handed
//! a tip, which teaches the economy without reading like a tutorial.
//!
//! Lines live in `assets/barks.yaml` (see that file for the placeholder list)
//! so adding dialogue never means touching Rust. A line whose placeholders
//! can't be resolved on this planet is simply skipped, so authors can write
//! freely without checking whether every world has a shipyard.

use bevy::prelude::*;
use rand::Rng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::collections::HashMap;

use crate::galaxy::GalaxyControl;
use crate::item_universe::ItemUniverse;
use crate::standing::FactionStandings;

/// Pre-rendered lines carried by an NPC, shown when you talk to them.
/// Rendered at SPAWN time (where the economy is in scope) rather than at chat
/// time, so the chat system stays ignorant of prices and hulls.
#[derive(Component, Clone, Debug, Default)]
pub struct Barks(pub Vec<String>);

#[derive(Deserialize, Debug, Default)]
struct BarkFile {
    roles: HashMap<String, Vec<String>>,
}

/// Everything a line may need to know about the here-and-now. Grouped so the
/// render signature stays sane as more of the world becomes quotable.
pub struct BarkContext<'a> {
    pub iu: &'a ItemUniverse,
    pub planet_key: &'a str,
    pub galaxy: Option<&'a GalaxyControl>,
    pub standings: Option<&'a FactionStandings>,
}

/// All authored lines, keyed by role.
#[derive(Resource, Debug, Default)]
pub struct BarkCatalog {
    roles: HashMap<String, Vec<String>>,
}

impl BarkCatalog {
    pub fn load() -> Self {
        match crate::embedded_assets::read_to_string("assets/barks.yaml")
            .ok()
            .and_then(|t| serde_yaml::from_str::<BarkFile>(&t).ok())
        {
            Some(f) => Self { roles: f.roles },
            None => {
                eprintln!("[barks] WARNING: could not load assets/barks.yaml");
                Self::default()
            }
        }
    }

    /// Two lines for `role` on this planet, with placeholders filled. Falls
    /// back to the `generic` role so an NPC is never mute.
    pub fn render(&self, role: &str, ctx: &BarkContext, rng: &mut impl Rng) -> Vec<String> {
        let facts = Facts::gather(ctx);
        let mut usable: Vec<String> = self
            .roles
            .get(role)
            .map(|lines| lines.iter().filter_map(|l| facts.fill(l)).collect())
            .unwrap_or_default();
        if usable.is_empty() {
            usable = self
                .roles
                .get("generic")
                .map(|lines| lines.iter().filter_map(|l| facts.fill(l)).collect())
                .unwrap_or_default();
        }
        usable.shuffle(rng);
        usable.truncate(2);
        usable
    }
}

/// The live values a line may reference on this planet. Anything we couldn't
/// work out stays `None`, and lines needing it are dropped.
struct Facts {
    here: Option<String>,
    commodity: Option<String>,
    dest: Option<String>,
    ship: Option<String>,
    item: Option<String>,
    faction: Option<String>,
    /// A faction at war with whoever holds this system.
    enemy: Option<String>,
    /// A live front: the contested system, who is pushing, against whom, and
    /// how hard. All read from current influence — no history is stored, so
    /// lines may describe the war NOW but never how it got here.
    front: Option<String>,
    front_sponsor: Option<String>,
    front_enemy: Option<String>,
    intensity: Option<String>,
    /// How the holding faction regards the player, in words.
    regard: Option<String>,
}

impl Facts {
    fn gather(ctx: &BarkContext) -> Self {
        let (iu, planet_key) = (ctx.iu, ctx.planet_key);
        let Some((system, pd)) = iu.find_gameplay_planet(planet_key) else {
            return Self {
                here: None,
                commodity: None,
                dest: None,
                ship: None,
                item: None,
                faction: None,
                enemy: None,
                front: None,
                front_sponsor: None,
                front_enemy: None,
                intensity: None,
                regard: None,
            };
        };
        let here = Some(pd.display_name.clone());
        let faction = (!pd.faction.is_empty()).then(|| pd.faction.clone());

        // Prefer a commodity this planet is genuinely CHEAP on, so the
        // overheard route is one worth flying; fall back to anything traded.
        let best_buy = iu
            .system_planet_best_commodity_to_buy
            .get(system)
            .and_then(|m| m.get(planet_key))
            .cloned();
        let commodity_key = best_buy.or_else(|| {
            let mut keys: Vec<&String> = pd.commodities.keys().collect();
            keys.sort();
            keys.first().map(|k| (*k).clone())
        });
        let commodity = commodity_key.as_ref().map(|k| {
            iu.commodities
                .get(k)
                .map(|c| c.display_name.clone())
                .unwrap_or_else(|| k.clone())
        });
        // Where that good actually fetches the most, in this system.
        let dest = commodity_key.as_ref().and_then(|k| {
            iu.system_commodity_best_planet_to_sell
                .get(system)
                .and_then(|m| m.get(k))
                .filter(|dest_key| dest_key.as_str() != planet_key)
                .and_then(|dest_key| {
                    iu.star_systems
                        .get(system)
                        .and_then(|s| s.planets.get(dest_key))
                        .map(|d| d.display_name.clone())
                })
        });
        let ship = pd.shipyard.first().map(|k| {
            iu.ships
                .get(k)
                .map(|s| s.display_name.clone())
                .unwrap_or_else(|| k.clone())
        });
        let item = pd.outfitter.first().map(|k| {
            iu.outfitter_items
                .get(k)
                .map(|i| i.display_name().to_string())
                .unwrap_or_else(|| k.clone())
        });
        // ── politics: who holds this system, and who wants it ──
        let holder = ctx
            .galaxy
            .and_then(|g| g.controller(system))
            .map(|s| s.to_string())
            .or_else(|| faction.clone());
        let enemy = holder.as_ref().and_then(|h| {
            iu.enemies
                .get(h)
                .and_then(|e| e.first())
                .filter(|e| iu.faction_takes_sides(e))
                .cloned()
        });
        // Fronts are derived from LIVE influence, so this is true right now.
        let (mut front, mut front_sponsor, mut front_enemy, mut intensity) =
            (None, None, None, None);
        if let Some(galaxy) = ctx.galaxy {
            let fronts = crate::war::detect_fronts(iu, galaxy);
            // Prefer a front touching this system, else any front at all.
            let pick = fronts
                .iter()
                .find(|f| f.target == system || f.home == system)
                .or_else(|| fronts.first());
            if let Some(f) = pick {
                front = iu
                    .star_systems
                    .get(&f.target)
                    .map(|s| s.display_name.clone())
                    .or_else(|| Some(f.target.clone()));
                front_sponsor = Some(f.sponsor.clone());
                front_enemy = Some(f.enemy.clone());
                intensity = Some(
                    match crate::war::front_tier(galaxy, f) {
                        1 => "raids and quiet work",
                        2 => "squadron fighting",
                        _ => "a decisive push",
                    }
                    .to_string(),
                );
            }
        }
        // What the holding faction makes of the player.
        let regard = holder.as_ref().and_then(|h| {
            let st = ctx.standings?.get(h);
            Some(
                if st <= crate::standing::ARREST_THRESHOLD {
                    "wanted"
                } else if st <= crate::standing::ENGAGE_THRESHOLD {
                    "not welcome"
                } else if st >= 25.0 {
                    "well regarded"
                } else {
                    "an unknown"
                }
                .to_string(),
            )
        });
        Self {
            here,
            commodity,
            dest,
            ship,
            item,
            faction: holder.or(faction),
            enemy,
            front,
            front_sponsor,
            front_enemy,
            intensity,
            regard,
        }
    }

    /// Substitute placeholders, or return `None` if the line needs a fact this
    /// planet can't supply (no shipyard, nowhere better to sell, ...).
    fn fill(&self, line: &str) -> Option<String> {
        let mut out = line.to_string();
        for (token, value) in [
            ("{here}", &self.here),
            ("{commodity}", &self.commodity),
            ("{dest}", &self.dest),
            ("{ship}", &self.ship),
            ("{item}", &self.item),
            ("{faction}", &self.faction),
            ("{enemy}", &self.enemy),
            ("{front}", &self.front),
            ("{front_sponsor}", &self.front_sponsor),
            ("{front_enemy}", &self.front_enemy),
            ("{intensity}", &self.intensity),
            ("{regard}", &self.regard),
        ] {
            if out.contains(token) {
                let v = value.as_ref()?;
                out = out.replace(token, v);
                // "a Asteroid Miner" — fix the article now that we know the
                // word, since authors can't know it when writing the line.
                if v.starts_with(['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']) {
                    out = out.replace(&format!(" a {v}"), &format!(" an {v}"));
                }
            }
        }
        Some(capitalise_sentences(&out))
    }
}

/// Uppercase the first letter of the line and of each sentence. A placeholder
/// can land at a sentence start ("{intensity} out there..."), and the author
/// can't know that when writing it, so fix it after substitution.
fn capitalise_sentences(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut start = true;
    for ch in text.chars() {
        if start && ch.is_alphabetic() {
            out.extend(ch.to_uppercase());
            start = false;
        } else {
            out.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                start = true;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iu() -> ItemUniverse {
        let mut iu: ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        iu
    }

    #[test]
    fn barks_load_and_render_with_real_data() {
        let cat = BarkCatalog::load();
        assert!(!cat.roles.is_empty(), "barks.yaml must parse");
        let iu = iu();
        let mut rng = rand::thread_rng();
        for role in [
            "customer_market",
            "customer_shipyard",
            "shopper",
            "regular",
            "generic",
        ] {
            let ctx = BarkContext {
                iu: &iu,
                planet_key: "earth",
                galaxy: None,
                standings: None,
            };
            let lines = cat.render(role, &ctx, &mut rng);
            assert!(!lines.is_empty(), "{role} produced no lines on earth");
            for l in &lines {
                assert!(!l.contains('{'), "{role}: unfilled placeholder in {l:?}");
            }
        }
    }

    /// A line needing a fact the planet lacks must be dropped, never shown raw.
    #[test]
    fn unresolvable_lines_are_skipped_not_shown() {
        let iu = iu();
        let ctx = BarkContext {
            iu: &iu,
            planet_key: "definitely_not_a_planet",
            galaxy: None,
            standings: None,
        };
        let facts = Facts::gather(&ctx);
        assert_eq!(facts.fill("Nice weather on {here}."), None);
        assert_eq!(
            facts.fill("Mind how you go."),
            Some("Mind how you go.".into())
        );
    }
}

#[cfg(test)]
mod sample {
    use super::*;
    /// Not an assertion so much as a window: prints what players will hear.
    #[test]
    #[ignore = "informational — run with --ignored to read sample barks"]
    fn print_samples() {
        let cat = BarkCatalog::load();
        let mut iu: ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        let mut rng = rand::thread_rng();
        // A real galaxy, so the war lines are exercised rather than skipped.
        let galaxy = crate::galaxy::GalaxyControl::seeded_from(&iu);
        let standings = crate::standing::FactionStandings::default();
        for planet in ["earth", "mars", "mercury"] {
            for role in [
                "customer_market",
                "customer_shipyard",
                "shopper",
                "old_pilot",
                "tipsy_pilot",
                "intel_officer",
                "veteran",
                "war_correspondent",
                "partisan",
            ] {
                let ctx = BarkContext {
                    iu: &iu,
                    planet_key: planet,
                    galaxy: Some(&galaxy),
                    standings: Some(&standings),
                };
                for l in cat.render(role, &ctx, &mut rng) {
                    println!("[{planet}/{role}] {l}");
                }
            }
        }
    }
}
