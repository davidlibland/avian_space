# Unifying the on-foot paths (surface ↔ interiors)

The user-visible principle: walking a planet surface and walking a
building interior are the SAME activity — one avatar, one physics setup,
one HUD, one NPC stack — differing only in which scene got built around
the walker. The code should mirror that.

## What is already shared (by construction)

The interior implementation deliberately reuses the surface stack rather
than duplicating it. Identical systems run in both states: walker input,
character animation, footsteps, depth sorting, NPC behaviors/chat/markers,
companion avatars, camera follow, egui building windows. Shared data
structures: `SurfaceCostMap`/`SurfacePaths` (pathfinding), the blob47
autotiler + biome manifests (rendering + collision + footsteps),
`CommsChannel` (messaging), `SurfaceMiniMap` (the HUD map slot).

## What this pass unified

* **`surface::on_foot`** — one run condition (`Exploring | Inside`)
  replacing seven hand-written two-state `.or()` gates in the surface
  plugin, the starfield's `not_on_foot` mirror, and the HUD toggle's
  state check. New walking systems should gate on these, never on
  `Exploring` alone. (The interiors bugs of 2026-07-12 — ship visible,
  camera on the ship, flight HUD showing — were all systems still gated
  on `Exploring` alone.)
* **`surface::spawn_walker_at`** — one walker bundle. setup_surface and
  setup_interior both call it; physics/animation tuning can no longer
  drift between the scenes.
* **HUD** — `toggle_space_hud_visibility` treats both states as on-foot;
  the mini-map slot shows whatever `SurfaceMiniMap` resource the current
  scene inserted (surface terrain outdoors, the floor plan indoors), and
  the player dot works unchanged because it reads the same resource.
* **Stock filtering** — `purchasable_items`/`purchasable_ships` are the
  single source for "what can this player buy here", used by the plan
  sizing, the display bindings, and (indirectly matching) the classic
  windows.

## Remaining divergences, and the path to closing them

1. **Tile rendering.** The surface renders through `bevy_ecs_tilemap`;
   interiors spawn one `Sprite` per tile (64×64 = 4096 entities — fine,
   but a second code path). Unify by extracting the surface's tilemap
   construction into `fn spawn_tile_scene(terrain, biome, tint, scope)`
   used by both. Prerequisite: the interior's per-tile tint becomes a
   tilemap color, which bevy_ecs_tilemap supports per-tile.
2. **Scene setup shape.** `setup_surface` is one large system that
   generates terrain AND spawns everything; `setup_interior` builds an
   `InteriorPlan` first, then spawns from it. The plan-first shape is
   the better one: extract a `ScenePlan { terrain, solid, props, spawn,
   cost_map, minimap }` that setup_surface also produces (its terrain
   from fBm/station generators, its props from building placement), and
   ONE `spawn_scene(plan, scope)` consumes. That would collapse the
   collider spawning, cost-map insertion, and mini-map generation to
   single implementations (today: two each, deliberately same-shaped).
3. **Teardown.** `teardown_surface` manually re-reveals the ship and
   resets half a dozen resources; interiors rely on `InteriorScoped` +
   `DespawnOnExit`. When the ScenePlan refactor lands, move the
   ship-reveal to `OnEnter` of the flight states (idempotent) instead of
   `OnExit(Exploring)` — that asymmetry caused the ship-visible bug.
4. **States.** `Exploring` and `Inside` could become one `OnFoot` state
   with a `Venue(Option<BuildingKind>)` resource — but Bevy state-scoped
   entities (`DespawnOnExit`) are the main consumer, and interiors
   already need `InteriorScoped` for same-state level rebuilds. Verdict:
   not worth it; two states + the `on_foot` condition is simpler than
   one state + a venue resource that every scoped spawn must check.

Suggested order when touching this next: (1) tilemap extraction, then
(2) ScenePlan, then (3) teardown cleanup. Each is independently
shippable; (4) is explicitly declined.
