# Building interiors — walkable shops, showrooms, and mazes (design)

The player presses E at a door and *goes inside*: a walkable interior scene
where the outfitter's stock sits on display pedestals, the shipyard shows
hulls as glowing wireframes and hero shots with stat panels, mission NPCs
wait at the bar instead of loitering on the pad, and some interiors are
big enough to get lost in — which is the point, for hunt missions.

Three foundations ALREADY exist, which shapes everything below:

1. **`station_layout.rs`** — the corridor/room floor-plan generator for
   interior biomes (plazas, corridor spines, rooms, town squares,
   pedestals), honoring the blob47 autotiler contract. This *is* the maze
   generator; interiors reuse it directly.
2. **Interior tile atlases per style** — `buildings_manifest.ron` already
   carries `interior_atlas` for all five styles (colony/cryo/extraction/
   station/outpost). Interiors reskin by biome for free, like exteriors.
3. **Per-ship wireframes** (`assets/sprites/wireframes/*.png`, validator-
   enforced) and the ship3d Blender pipeline — the shipyard's holo-displays
   and "perspective shots" are one bake script away, not an art project.

The surface machinery (walker physics, foot-anchored colliders, NPC
behavior queue, SurfaceCostMap pathfinding, depth sorting, footsteps,
companion avatars) transfers unchanged: an interior is just another
tilemap.

---

## 1. Scene model: an interior is another surface

Stay in `PlayState::Exploring`. A `CurrentFloor` resource says where the
walker is:

```rust
enum CurrentFloor {
    Surface,
    Interior { building: BuildingKind, entrance: (u32, u32) },
}
```

Entering a door tears down the surface world and builds the interior
(same `DespawnOnExit`-style lifecycle, keyed on floor change rather than
state change); exiting rebuilds the surface and puts the walker back at
the door. This works because surface generation is **deterministic per
planet** (terrain seed, seeded building placement) — the world you return
to is exactly the one you left. A short fade covers the swap.

Why not keep both worlds alive: overlapping tilemaps/colliders need layer
partitioning for zero benefit — rebuild is <1 frame at these map sizes.

E-priority (buildings > mission NPCs > NPCs) already routes the doorway
press; the change is that E at a door *transitions* instead of opening the
full-screen window.

## 2. Two kinds of floor plan

**Authored shop templates** (market, outfitter, mechanic, bar, garrison):
small fixed rooms (≈14×10 tiles), defined in `assets/interiors/` RON —
tile grid + prop anchor list, reskinned by the style's interior atlas.
Authored because a shop's readability IS its layout: door at the south,
counter opposite, displays along walls. One template per kind, style does
the reskin (the exterior-template pattern exactly).

**Generated halls** (shipyard, and station/maze interiors):
`generate_station_map` with kind-specific parameters — the shipyard gets
one big hall + side rooms (cradle displays down the center line); hunt
interiors get the maze dial: narrower corridors (half=1), more rooms,
dead ends, 2–3× the shop footprint. The generator's pedestal feature
already produces display-plinth positions.

## 3. Displays: the stock is IN the room

`InteriorProp` entities placed by the template, each optionally bound to
a catalog slot:

```rust
struct DisplayProp {
    binding: DisplayBinding, // OutfitterSlot(n) | ShipSlot(n) | Static
}
```

Bindings resolve against the planet's DERIVED stock at build time (the
same lists the windows use), so displays inherit tech/faction/war
re-derivation for free — walk into the outfitter after a system flips and
the shelves have changed.

* **Outfitter**: pedestals along the walls, one item each — weapon sprite
  (they exist) hovering over the plinth with the weapon's `color` as a
  glow. Walking adjacent opens a *focused* egui panel: name, price, DPS/
  range/space, mount class, Buy — reusing `buy_weapon` and the mounts/
  markup logic verbatim. Ammo restock and the full-list view live at the
  **counter** (clerk NPC = today's window as fallback, so nothing
  regresses).
* **Shipyard**: hull cradles / holo-pads down the hall. Each shows the
  ship's **wireframe** (cyan, alpha-pulsing — the HUD asset, scaled up)
  or a **hero shot**: one new bake pass in the ship3d pipeline rendering
  each hull at a low-angle beauty view (~55 renders, one script run,
  `assets/sprites/ships/hero/<ship>.png`). Adjacent → stats panel with
  bar-chart rows (speed/thrust/hull/cargo/item space/gun+turret mounts vs.
  your current ship in a second color) + price/trade-in + Buy via the
  existing `BuyShip` flow.
* **Mechanic**: the hull bench (mods) as floor displays — engine block,
  armor plates; repair/refuel stay at the counter.
* **Market**: crates as ambience; trading stays a counter window (a
  pedestal per commodity would be noise, not clarity).
* **Garrison**: the war desk — duty officer at a desk, front map table
  prop showing the system's fronts (flavor), war offers via the officer.

## 4. People inside

* Building-bound **offer NPCs spawn inside** their building's interior
  (the `building:` field they already carry decides which). The bar
  finally means something: walk in, find the stranger at a corner table.
  Outside, the pad keeps only `seek`-approach givers who chase the player.
* **MeetNpc/CatchNpc with `building:`** become interior objectives — the
  NPC is somewhere in that building. With `hint:` (already built), a hunt
  composes to: *find the right planet → the right building → search the
  maze*. Flee behavior works unchanged on the interior cost map, and the
  corridor-aware flee goals (compass-ray sampling) were built for exactly
  this topology.
* **Companions follow you in** (same avatar system; spawn at the interior
  door). Bartender/clerk/foreman ambience NPCs come from the npcs.yaml
  role pools, one or two per shop, patrolling between anchors.

## 5. Maze hunts (new mission flavor, no new machinery)

A `catch_npc` objective with `building: station_depths` on a
station-biome planet: the interior generator (maze parameters, seeded by
planet + mission id so re-entry doesn't reshuffle) hides the target in a
far room. Mission log shows the hint; the fun is navigation. Later:
locked doors + a key prop for light dungeon structure, and the arrest
flow's penal missions get "extract someone from the cell block" variants
inside garrisons.

## 6. Build order

1. **The loop** — CurrentFloor, door transition + fade, one authored
   interior (the BAR) with counter + patrons + relocated offer NPCs,
   exit door. Proves lifecycle, pathfinding, companions inside.
   *Everything else is content on this rail.*
2. **Shops** — outfitter/mechanic/garrison templates; DisplayProp +
   focused buy panels; prop sprite bakes (pedestal, counter, rack,
   holo-pad — buildings3d-style Blender pass).
3. **Shipyard showroom** — hero-shot bake pass, wireframe holo-pads,
   comparative stat panels, BuyShip from the cradle.
4. **Mazes & hunts** — station_layout maze parameters, interior-scoped
   meet/catch objectives, one flagship hunt mission per faction space,
   seeded-per-mission layouts.
5. **Polish** — interior minimap, ambient sound beds, locked doors/keys,
   garrison cells, market crate ambience.

Tests along the way, per the usual pattern: door round-trip determinism
(surface identical after exit), display bindings match the derived
catalogs (validator), interior cost maps are connected (generator
contract already tested), hunt NPC always reachable, buy-from-display
equivalence with the window paths.

## 7. Open questions

* Does the full-screen window stay as a fallback (counter interaction)
  everywhere, or do interiors fully replace it once built? (Lean: keep
  counters as the power-user path; displays are the browse path.)
* Hero shots vs. wireframes in the shipyard — both? (Lean: wireframe on
  the pad, hero shot + stats in the focused panel.)
* Should big stations be their own landable "planets" with interior-only
  surfaces? `station_layout` + the "interior" biome already support it —
  a natural place for maze content and the Marches freeport flavor.
