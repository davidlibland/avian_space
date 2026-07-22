//! Player-facing storyline flow chart model.
//!
//! Builds a partially-obscured, faction-colored DAG of the authored story
//! missions for the Pilot Info "Story" tab. Fog-of-war rules (per design):
//!
//! - **Completed / Active / Failed** missions are fully revealed (name shown).
//! - The **next available** mission (offer requirements met but not yet done)
//!   shows as a faction-colored blank — color retained, no text.
//! - Missions whose requirements are **not yet met** are hidden entirely.
//! - A branch that has become **impossible** (a `Completed{X}` gate whose X
//!   failed, or a `Failed{X}` gate whose X was completed) is revealed only at
//!   its known branch point, as a locked blank.
//!
//! Declining a story mission is NOT permanent (it only lowers offer weight),
//! so there is no separate "declined" state — a declined mission stays Next.
//!
//! Layout is a left-to-right layered DAG: column = longest-path depth over
//! `Completed{...}` precondition edges; only revealed nodes are placed.

use std::collections::HashMap;

use super::log::{MissionLog, PlayerUnlocks};
use super::types::{MissionStatus, Precondition};
use crate::item_universe::ItemUniverse;

/// Visual state of a node in the player-facing chart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeUi {
    /// Finished — full color, name shown.
    Completed,
    /// Accepted, in progress — highlighted, name shown.
    Active,
    /// Offer requirements met, not yet taken — faction color, no text.
    Next,
    /// Attempted and failed — name shown, muted red.
    Failed,
    /// This branch can no longer be entered — locked blank at the branch point.
    Impossible,
}

impl NodeUi {
    /// Whether the mission's name should be legible to the player.
    pub fn shows_name(self) -> bool {
        matches!(self, NodeUi::Completed | NodeUi::Active | NodeUi::Failed)
    }
}

pub struct StoryNode {
    #[allow(dead_code)] // consumed by the chart exporter, not the in-game view
    pub id: String,
    pub label: String,
    #[allow(dead_code)] // consumed by the chart exporter, not the in-game view
    pub faction: Option<String>,
    /// Faction color (from factions.yaml), or a neutral grey.
    pub color: [u8; 3],
    pub ui: NodeUi,
    /// Unlocks granted on completion (shown on completed nodes).
    pub grants: Vec<String>,
    pub col: u32,
    pub row: u32,
}

pub struct StoryEdge {
    pub from: usize,
    pub to: usize,
    /// True if this prerequisite is satisfied (both ends known / met).
    pub satisfied: bool,
}

#[derive(Default)]
pub struct StoryGraph {
    pub nodes: Vec<StoryNode>,
    pub edges: Vec<StoryEdge>,
    pub cols: u32,
    pub rows: u32,
}

fn prettify(id: &str) -> String {
    id.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build the player-facing story graph. Only authored missions carrying a
/// `faction:` (i.e. the storyline chains) are considered.
pub fn build_story_graph(
    log: &MissionLog,
    unlocks: &PlayerUnlocks,
    universe: &ItemUniverse,
) -> StoryGraph {
    // Story mission set: ALL authored missions. Template-generated missions
    // can't appear — they live only in the runtime catalog, never in
    // `universe.missions` — and clutter control is the reveal logic below
    // (`ui_of` hides anything the player hasn't earned a look at), not this
    // set. Factionless arcs (lost_son, friend chains) draw in neutral grey.
    let ids: Vec<&String> = universe.missions.keys().collect();
    let in_set: std::collections::HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();

    let def = |id: &str| universe.missions.get(id);

    // `Completed{...}` precondition edges within the set.
    let mut preds: HashMap<&str, Vec<&str>> = ids.iter().map(|id| (id.as_str(), vec![])).collect();
    for id in &ids {
        for p in &def(id).unwrap().preconditions {
            if let Precondition::Completed { mission } = p
                && in_set.contains(mission.as_str())
            {
                preds.get_mut(id.as_str()).unwrap().push(mission.as_str());
            }
        }
    }

    // Is a precondition permanently unsatisfiable given current state?
    let precondition_impossible = |p: &Precondition| -> bool {
        match p {
            Precondition::Completed { mission } => {
                matches!(log.status(mission), MissionStatus::Failed)
            }
            Precondition::Failed { mission } => {
                matches!(log.status(mission), MissionStatus::Completed)
            }
            // Unlocks are treated as pending, not impossible.
            Precondition::HasUnlock { .. } => false,
        }
    };
    let precondition_terminal_known = |p: &Precondition| -> bool {
        matches!(
            p,
            Precondition::Completed { mission } | Precondition::Failed { mission }
            if matches!(
                log.status(mission),
                MissionStatus::Completed | MissionStatus::Failed
            )
        )
    };

    // Per-mission UI status (or None = hidden).
    let ui_of = |id: &str| -> Option<NodeUi> {
        let d = def(id)?;
        match log.status(id) {
            MissionStatus::Completed => Some(NodeUi::Completed),
            MissionStatus::Active(_) => Some(NodeUi::Active),
            MissionStatus::Failed => Some(NodeUi::Failed),
            MissionStatus::Available | MissionStatus::Locked => {
                if super::progress::preconditions_met(&d.preconditions, log, unlocks) {
                    Some(NodeUi::Next)
                } else if d.preconditions.iter().any(precondition_impossible)
                    && d.preconditions.iter().any(precondition_terminal_known)
                {
                    // A closed branch, revealed only at its known branch point.
                    Some(NodeUi::Impossible)
                } else {
                    None // requirements not yet met → hidden
                }
            }
        }
    };

    let revealed: HashMap<&str, NodeUi> = ids
        .iter()
        .filter_map(|id| ui_of(id).map(|ui| (id.as_str(), ui)))
        .collect();

    // Longest-path depth over completed-precondition edges (all story
    // missions, so columns stay stable as the fog lifts).
    let mut depth: HashMap<&str, u32> = HashMap::new();
    fn depth_of<'a>(
        id: &'a str,
        preds: &HashMap<&'a str, Vec<&'a str>>,
        depth: &mut HashMap<&'a str, u32>,
        stack: &mut std::collections::HashSet<&'a str>,
    ) -> u32 {
        if let Some(&d) = depth.get(id) {
            return d;
        }
        if !stack.insert(id) {
            return 0; // cycle guard
        }
        let d = preds
            .get(id)
            .map(|ps| {
                ps.iter()
                    .map(|p| 1 + depth_of(p, preds, depth, stack))
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        stack.remove(id);
        depth.insert(id, d);
        d
    }
    for id in &ids {
        let mut stack = std::collections::HashSet::new();
        depth_of(id.as_str(), &preds, &mut depth, &mut stack);
    }

    // Place revealed nodes: column = depth, rows packed by faction then id.
    let max_depth = revealed.keys().map(|id| depth[id]).max().unwrap_or(0);
    let mut columns: Vec<Vec<&str>> = vec![Vec::new(); (max_depth + 1) as usize];
    for id in &ids {
        if revealed.contains_key(id.as_str()) {
            columns[depth[id.as_str()] as usize].push(id.as_str());
        }
    }
    let mut node_index: HashMap<&str, usize> = HashMap::new();
    let mut nodes: Vec<StoryNode> = Vec::new();
    let mut rows_max = 0u32;
    for (col, column) in columns.iter_mut().enumerate() {
        column.sort_by_key(|id| {
            let f = def(id).and_then(|d| d.faction.clone()).unwrap_or_default();
            (f, id.to_string())
        });
        for (row, id) in column.iter().enumerate() {
            let d = def(id).unwrap();
            let faction = d.faction.clone();
            let color = faction
                .as_ref()
                .and_then(|f| universe.factions.get(f))
                .map(|f| f.color)
                .unwrap_or([120, 120, 120]);
            let grants = d
                .completion_effects
                .iter()
                .filter_map(|e| match e {
                    super::types::CompletionEffect::GrantUnlock { name } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            node_index.insert(id, nodes.len());
            nodes.push(StoryNode {
                id: id.to_string(),
                label: prettify(id),
                faction,
                color,
                ui: revealed[id],
                grants,
                col: col as u32,
                row: row as u32,
            });
            rows_max = rows_max.max(row as u32 + 1);
        }
    }

    // Edges between two revealed nodes.
    let mut edges = Vec::new();
    for id in &ids {
        let Some(&to) = node_index.get(id.as_str()) else {
            continue;
        };
        for p in preds[id.as_str()].iter() {
            if let Some(&from) = node_index.get(p) {
                let satisfied = matches!(log.status(p), MissionStatus::Completed);
                edges.push(StoryEdge {
                    from,
                    to,
                    satisfied,
                });
            }
        }
    }

    StoryGraph {
        nodes,
        edges,
        cols: if columns.is_empty() { 0 } else { max_depth + 1 },
        rows: rows_max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prettify_ids() {
        assert_eq!(prettify("bounty_story_1"), "Bounty Story 1");
        assert_eq!(prettify("rift_final"), "Rift Final");
    }

    #[test]
    fn shows_name_only_when_known() {
        assert!(NodeUi::Completed.shows_name());
        assert!(NodeUi::Failed.shows_name());
        assert!(!NodeUi::Next.shows_name());
        assert!(!NodeUi::Impossible.shows_name());
    }

    #[test]
    fn factionless_chains_are_in_the_story_set() {
        let mut iu: crate::item_universe::ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        // The lost_son arc (finding Jonah) has no faction tag but is a
        // Completed-precondition chain — it must chart. Completing part 1
        // reveals part 2 as reachable, so the graph includes both.
        let mut log = MissionLog::default();
        log.set("lost_son_1", MissionStatus::Completed);
        let graph = build_story_graph(&log, &PlayerUnlocks::default(), &iu);
        for id in ["lost_son_1", "lost_son_2"] {
            assert!(
                graph.nodes.iter().any(|n| n.id == id),
                "{id} missing from story graph"
            );
        }
    }
}
