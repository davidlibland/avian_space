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

use crate::item_universe::ItemUniverse;

/// Pre-rendered lines carried by an NPC, shown when you talk to them.
/// Rendered at SPAWN time (where the economy is in scope) rather than at chat
/// time, so the chat system stays ignorant of prices and hulls.
#[derive(Component, Clone, Debug, Default)]
pub struct Barks(pub Vec<String>);

#[derive(Deserialize, Debug, Default)]
struct BarkFile {
    roles: HashMap<String, Vec<String>>,
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
    pub fn render(
        &self,
        role: &str,
        iu: &ItemUniverse,
        planet_key: &str,
        rng: &mut impl Rng,
    ) -> Vec<String> {
        let facts = Facts::gather(iu, planet_key);
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
}

impl Facts {
    fn gather(iu: &ItemUniverse, planet_key: &str) -> Self {
        let Some((system, pd)) = iu.find_gameplay_planet(planet_key) else {
            return Self {
                here: None,
                commodity: None,
                dest: None,
                ship: None,
                item: None,
                faction: None,
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
        Self {
            here,
            commodity,
            dest,
            ship,
            item,
            faction,
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
        Some(out)
    }
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
            let lines = cat.render(role, &iu, "earth", &mut rng);
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
        let facts = Facts::gather(&iu, "definitely_not_a_planet");
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
        for planet in ["earth", "mars", "mercury"] {
            for role in ["customer_market", "customer_shipyard", "shopper"] {
                for l in cat.render(role, &iu, planet, &mut rng) {
                    println!("[{planet}/{role}] {l}");
                }
            }
        }
    }
}
