# Maze-layout buildings — mines & covert-ops venues (plan)

Phase-1 interiors made the bar/outfitter/shipyard walkable single rooms.
This plans the opposite kind of interior: **buildings you can get lost
in**, built for `meet_npc`/`catch_npc` hunts — especially covert-ops
targets ("find the defector, we don't know more than the building") and
the pirate/miner economy.

The maze generator already exists: `station_layout.rs` produces
corridor-spine + room floor plans honoring the blob47 contract, with
pedestals, plazas, and dead ends — it's what interior-biome planets use.
Maze buildings are that generator with different knobs, behind a door.

## 1. The venues

Three new `BuildingKind`s, each a natural home for a mission archetype:

| Kind | Where it spawns | Layout feel | Who you find there |
|---|---|---|---|
| **Mine** | extraction-style worlds; pirate & miner economies | rough-cut tunnels, irregular rooms, ore-cart spurs, dead-end galleries | miners with grievances, pirate quartermasters, "lost survey team" hunts, ore-theft covert ops |
| **Warehouse row** (freight depot) | trade hubs, free ports | grid of container canyons — long sightlines, right-angle ambushes | smugglers, fences, dead-drop covert missions, catch-the-courier chases |
| **Undercity** (service level) | high-tech / station worlds | narrow service corridors, pump rooms, maintenance crawls | defectors in hiding, saboteurs, informant meets for the war's espionage arcs |

Why these three: they cover the three world flavors (frontier rock, trade,
high-tech), all justify "the person you want is *somewhere inside*", and
each gives covert missions a distinct texture — galleries (occlusion),
canyons (sightlines), corridors (topology).

## 2. Generation

* `station_layout::generate_station_map` gains a small param struct
  (corridor half-width, room count/size range, dead-end bias, plaza
  allowance) with presets per venue: **mine** = half 1, many small rooms,
  high dead-end bias, no plazas; **warehouse** = wide straight corridors
  in a grid, rooms replaced by container blocks; **undercity** = half 1,
  long winding spines, sparse rooms, extra loops.
* Seeded by `(planet seed, building kind, mission id?)` — stable across
  re-entry, but hunt missions can request a fresh shuffle per mission so
  repeat hunts in the same mine differ.
* Footprint ~2–3× a shop (they're the *point*, not an errand), still far
  below the full-planet interior maps, so the per-tile sprite path from
  Phase 1 holds.
* The hunt target spawns in the room **farthest from the door** (cost-map
  distance), patrol/flee behaviors unchanged — corridor-aware flee was
  built for exactly this topology. `hint:` fields compose: *find the
  planet → the building → search the maze*.

## 3. Tilesets (per the sourcing research: generate, one atlas per venue)

Construction differs per venue, so each gets its own `terrain_gen.py`
bake, sharing the six-tier semantics (floor/plating/grate/wall/conduit/
void) so no Rust changes:

* **mine_atlas** — dirt/gravel floor, timber-shored walls, rock face top
  tier; conduit tier becomes an ore-vein glitter. Warm lamp lighting.
* **warehouse_atlas** — painted concrete with lane markings, container
  walls (corrugated, stenciled), girder tops. Cool sodium lighting.
* **undercity_atlas** — deck grating, cable-run walls, pipe-bundle tops;
  conduit tier is live steam/coolant. Dim cyan-green lighting.

Per-world flavor via tint (the extraction mine vs. cryo mine differ by
palette, not geometry). Props (ore carts, crates, valve wheels, cage
lamps) come from a `buildings3d.py`-style prop bake, placed at the
generator's pedestal anchors.

## 4. Exteriors & placement

Each venue needs an exterior sprite pass in `buildings3d.py` (5 styles ×
3 kinds): mine = headframe + adit into a rock face, warehouse = long
gabled shed + container stacks, undercity = a stair-kiosk/blast-door
entrance (small footprint — the building is *below*). Spawn rules join
the existing table: mines on extraction/outpost worlds, warehouses where
`market` tech is high or the port is free, undercity on station/high-tech
worlds. Buildings stay rare (0–1 per world) so a hunt's "which building?"
step stays meaningful.

## 5. Missions that use them

* Covert war arcs (`espionage`): meet/catch objectives gain an optional
  `venue: mine|warehouse|undercity`; the offer text and `hint:` reference
  it. Existing missions untouched.
* Pirate/miner arcs: "collect the mine's protection ledger",
  "extract the assayer before the collapse", warehouse fence hand-offs.
* One flagship hunt per faction space at launch (reuses the
  loop-until-found pacing the flee AI already provides).

## 6. Build order

1. Generator presets + `BuildingKind::{Mine, Warehouse, Undercity}` +
   door transition reusing Phase-1 `InteriorContext` (maze plans instead
   of room plans). Tests: connectivity door→every room, contract, seed
   stability.
2. `terrain_gen.py` venue atlases + manifest entries; prop bake.
3. Exterior sprites + spawn rules + validators.
4. Mission `venue:` field + 3–4 authored covert/pirate missions using it.
5. Polish: minimap fog for mazes, ambient audio beds (drips, fans,
   distant machinery), locked side rooms + key props.

Phases 1–2 are playable; 3 makes them discoverable; 4 makes them matter.
