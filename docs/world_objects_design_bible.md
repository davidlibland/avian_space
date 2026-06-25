# World Objects Design Bible

Companion to [`ship_design_bible.md`](ship_design_bible.md) and
[`building_design_bible.md`](building_design_bible.md). Defines the **living and
scattered things** of each planet surface — rocks, plants, animals, wildlife,
machines, obstacles, water/lava life — what belongs where, how dense, and on
which terrain tiles. Drives a re-art of `scripts/objects.py` to match the rest
of the world.

---

## Where we are (and the mandate)

The placement system (`src/surface_objects.rs` + `objects_manifest.ron`) is good
and stays: each object declares the **terrains** it spawns on, a base **density**
modulated by an fBm field (so things clump into groves/fields rather than spread
evenly), `min_distance`, `max_per_tile`, an animation frame count, and a `shy`
flag for creatures that freeze/flee when the player approaches. Objects are
Y-sorted like everything else.

What must change is the **art and the roster**:

1. **Match the 3D toon pipeline.** Today's objects are flat PIL primitives (a
   tree is a blob on a stick; a rock reads as a pagoda; lava is a plus-sign).
   They should be offline 3D-rendered (Blender, toon + ink, world-fixed light)
   like ships/asteroids/buildings — small, readable, with a gentle idle
   animation (sway / bob / bubble / blink).
2. **Make each thing native to its biome.** No green leafy bush on a lava world.
   Every object's silhouette, palette, and material should say *this world*.
3. **Richer, legible wildlife.** Real creature silhouettes (not blobs), plus
   ambient life that sells the place — birds, insects, herds, fliers — and
   signature shy creatures per biome.

### Density & placement principles

- **Density follows the terrain's productivity.** Lush tiles (forest, snow-free
  grass, dunes-with-water) are busy; hostile tiles (lava, crevasse, quicksand,
  deep water) are sparse and what's there is *dangerous or special*.
- **Clump, don't sprinkle.** The fBm field makes forests, reed beds, ore veins,
  cactus stands — groves not confetti. Tune base density so the *peak* of a clump
  reads as full, the troughs as empty.
- **Layer by size.** Ground cover (grass, lichen, ripples) high-density + many
  per tile; mid props (bushes, rocks, cacti) medium; landmarks (tall trees,
  spires, boulders, rigs) low-density, one-per-several-tiles.
- **Reserve motion for meaning.** Most props idle-sway subtly; *animals move
  more* and the shy ones react — that contrast is what makes a world feel alive.

---

## Garden — temperate living world
*Terrains: water · sand · grass · forest · mountain*

**Story.** The kind world: rain, deep soil, breathable air. Life is everywhere
and layered — meadows thick with flowers and insects, dense mixed forest, a
shoreline of reeds and fish, bare rocky heights. The biome should feel *busy and
soft*, the densest of all.

| Category | Object | Tiles | Density | Notes |
|---|---|---|---|---|
| Ground flora | grass tufts, wildflowers, clover | sand·grass | **high**, 3/tile | sway; flowers add color clumps |
| | ferns, mushrooms | forest | med | damp undergrowth |
| Shrubs | leafy bush, berry shrub | grass·forest | med | |
| Trees | broadleaf tree, fruit tree | grass·forest | med, 1/tile | canopy landmarks |
| | conifer | forest·mountain | med | |
| | tall tree / old-growth | forest | **low** | towering, sparse |
| Water life | reeds, cattails, lily pads | water (shallows)·sand | med | shoreline band |
| | jumping fish, ripples, frogs | water | low | motion on open water |
| Rocks | mossy stones | grass·mountain | low | |
| | mossy boulder (obstacle) | mountain·grass | **low** | landmark/blocker |
| Wildlife (ambient) | butterflies, fireflies, bees | grass·forest | med | tiny drifting motion |
| | songbirds (flying) | over grass·forest | low | |
| Wildlife (**shy**) | rabbit / small deer | grass·forest | low | flees the player |
| | burrowing critter | grass·sand | low | pops into a hole |
| | curious alien peeker | forest | rare | |
| **Empty** | mountain bare rock, deep water | mountain·water | very low | breathing room |

**Read:** forest = dense (trees + ferns + mushrooms + birds); grass = the most
*alive* (flowers, insects, bushes, the odd tree, shy mammals); shoreline = a band
of reeds + fish; mountain = sparse rock + conifer; open water = quiet with the
occasional fish.

---

## Ice — frozen world
*Terrains: deep_ice · ice · snow · ice_rock · crevasse*

**Story.** Wind-scoured, half the year dark, life clinging on. Beauty is
crystalline and cold, not green. The biome should feel *sparse, hushed, hard* —
long empty stretches punctuated by ice formations and the rare hardy creature.

| Category | Object | Tiles | Density | Notes |
|---|---|---|---|---|
| Ice forms | ice floes / chunks | deep_ice·ice·snow | med | drift/bob on deep_ice |
| | ice spires, pressure crystals | ice·ice_rock | low | translucent landmarks |
| | frozen bubbles, blue glints | deep_ice | low | under-ice shimmer |
| Rocks | frosted stones, rimed boulder | ice_rock·snow | low | |
| Flora | lichen patches, frost-fern | snow·ice_rock | low | the only "plants", clinging |
| | frozen shrub (dead) | snow | rare | |
| Special | crevasse steam vent, geyser | crevasse edge | low | hazard + motion |
| | aurora shimmer (ambient sky) | over all | — | slow color drift |
| Wildlife (ambient) | snow-petrels (flying) | over snow·ice | low | |
| Wildlife (**shy**) | white fox / snow-hare | snow | low | nearly invisible till it bolts |
| | seal-thing by the water | crevasse·deep_ice edge | rare | |
| | penguin-ish rookery | ice | rare | small cluster |
| **Empty / hazard** | crevasse | crevasse | — | barren, impassable, dangerous |

**Read:** mostly *empty white*; snow carries the little life (lichen, fox,
petrels); deep_ice has drifting floes + glints; ice_rock has spires + boulders;
crevasse is a dangerous void with steam.

---

## Rocky — volcanic mining world
*Terrains: lava · basalt · rock · dust · cliff*

**Story.** You don't live here, you dig here. Molten, fuming, mineral-rich —
almost no biology, lots of geology and *industry*. The biome should feel
*dangerous and worked*: glowing lava, jagged basalt, ore veins, and the
prospectors' machines clattering away.

| Category | Object | Tiles | Density | Notes |
|---|---|---|---|---|
| Lava life | lava bubbles, pops | lava | **high** | the signature motion |
| | lava spurts / fountains | lava | med | bigger eruptions |
| | cooling-crust plates, ember drift | lava·basalt edge | med | |
| Geology | jagged rocks, basalt shards | basalt·rock | high | angular, dark |
| | basalt columns (obstacle) | basalt·cliff | low | hexagonal landmarks |
| | **ore/crystal deposits** (glowing) | basalt·rock | med | gold/iron veins — ties to mining |
| | large boulder | rock·cliff | low | |
| Machines | mining drill / pump-jack | basalt·rock·dust | med | animated, the "industry" |
| | ore hopper, conveyor, pipes | rock·dust | low | scattered equipment |
| | fumarole / steam vent | basalt·dust | med | smoke plume |
| Flora | sulfur crust, extremophile fungus | dust | low | sickly yellow, rare |
| | ash-scrub (dead) | dust | rare | |
| Wildlife (**shy**) | rock-dweller / silicon critter | basalt·dust | low | skitters into cracks |
| | lava-salamander | lava edge | rare | heat-loving |
| **Hazard** | lava (damaging), fumarole | lava·basalt | — | |

**Read:** lava = alive with bubbles/spurts (and deadly); basalt/rock = jagged
stone + glowing ore + drills + steam; dust = the only (barely) living tiles +
shy critters; cliff = bare boulders and columns.

---

## Desert — arid frontier world
*Terrains: quicksand · dunes · hard_sand · sandstone · mesa*

**Story.** Hot, dry, wind-carved, dust-storm country. Hardy life, slow geology,
and the bones of prospectors who came for water. The biome should feel *spare,
sun-baked, and quietly alive* — cacti stands, scrub, skittering things, circling
birds, and the odd relic.

| Category | Object | Tiles | Density | Notes |
|---|---|---|---|---|
| Flora | cactus (saguaro, barrel) | dunes·hard_sand | med | spiny landmarks, flower variants |
| | desert scrub / sagebrush | hard_sand | med | |
| | dry grass tufts | hard_sand·dunes | high, 3/tile | wind-sway |
| | tumbleweed (rolling) | hard_sand | low | ambient motion across tiles |
| | dead/petrified tree | sandstone | low | bleached landmark |
| Geology | wind-worn rocks | hard_sand·sandstone | med | smooth, rounded |
| | sandstone arch / pillar (obstacle) | sandstone·mesa | low | hero landmark |
| | mesa boulder, hoodoo | mesa | low | |
| | salt crust, cracked pan | quicksand edge | low | |
| Relics | prospector wreck, old drill, bones | hard_sand·sandstone | rare | story props |
| | water condenser (machine) | hard_sand | rare | |
| Wildlife (ambient) | vultures / dust-hawks (circling) | over mesa·sandstone | low | |
| Wildlife (**shy**) | sand-lizard / scarab | dunes·hard_sand | low | freezes, then darts |
| | burrowing sand-fish / snake | dunes | low | swims under the sand |
| **Hazard** | quicksand | quicksand | — | slow/sinking, barren |

**Read:** dunes = cacti + critters + tumbleweed; hard_sand = scrub + tufts +
rocks (the busiest); sandstone = arches + dead trees + relics; mesa = boulders +
circling birds; quicksand = dangerous and bare.

---

## Station / Interior — built world
*Terrains: floor · plating · grate · wall · conduit*

**Story.** No nature — the "wildlife" is machinery, signage, and vermin. The
biome should feel *engineered and lived-in*: humming kiosks, dripping pipes,
maintenance bots, and rats in the grates.

| Category | Object | Tiles | Density | Notes |
|---|---|---|---|---|
| Machines | terminal / kiosk / vending | floor·plating | med | screen-flicker animation |
| | maintenance machine, generator | plating·grate | med | |
| | floor vent (steam), wall pipe | floor·plating·grate | med | |
| | holographic ad / sign | floor·plating | low | bright loop |
| Clutter | cargo crates, barrels, canisters | floor·plating | med | stackable |
| | cabling, junction box | conduit·wall | med | |
| Fluids | coolant/sewage flow, drip | grate·conduit | med | the only "water" |
| | steam plume, sparks | grate·plating | low | |
| Greenery | hydroponic planter, potted plant | floor | low | the *only* nature — prized |
| Life (ambient) | maintenance drone (flying) | over floor·plating | low | not shy; patrols |
| Life (**shy**) | rat / vermin | floor·plating·grate | low | scurries into the grate |
| | cyber-roach swarm | grate·conduit | rare | |

**Read:** floor/plating = kiosks, crates, ads, the odd planter; grate = vents,
sewage, vermin; conduit = fluids + cabling; wall = mounted pipes/panels; drones
patrol overhead.

---

## Fauna — roaming vs peeking vs flying

Three kinds of "animal", handled by **two** systems. The split is by *motion*,
not by species.

| Kind | Moves? | System | Art | Sort |
|---|---|---|---|---|
| **Peeker** | no — fixed tile, pops up in place | `surface_objects` (existing) | idle frames (emerge/blink), `shy` flips to frame 0 when player nears | static, by spawn `y` |
| **Roamer** | yes — wanders tile-to-tile, flees | **`surface_fauna`** (new) | **directional walk sheet** (facing rows × walk-frame cols, like the people sheets) | dynamic, `depth_z(y-8)` each frame |
| **Flier** | yes — drifts/circles above ground | `surface_fauna`, flier variant | small flap loop; ignores terrain | high z (drawn over ground & player) |

**Why two systems:** peekers are cheap static props (hundreds, no AI); roamers
need a per-frame brain + directional animation + dynamic Y-sort, reusing
`CharacterAnim`/`Facing`/`sprite_index` + `depth_z` from the walker code. See the
architecture note that prompted this section.

### Roamer behavior (the `surface_fauna` state machine)

A roamer carries `Fauna { home_terrains, speed, flee_speed, group }` and cycles:

- **Graze / idle** — pause a few seconds at a tile (nibble/look-around frames).
- **Wander** — pick a random nearby tile **whose terrain ∈ `home_terrains`**
  (never water/lava/quicksand/wall unless that's home), walk there at `speed`,
  then graze. Off-path; terrain-constrained via the surface collision/terrain map.
- **Flee** — when the player comes within `flee_radius`, switch to `flee_speed`
  and move *away* to a far valid tile; once `calm_distance` away, resume wander.
  This is the mobile evolution of the `shy` peeker's freeze.
- **Group (optional)** — herd animals (deer) spawn in clusters of 2–4 and bias
  their wander target toward the group centroid, so they drift together and flee
  together.

**Counts & speed.** Few per surface (a handful, maintained at a target like
civilians — respawn off-screen when one despawns). Most roamers are **slower than
the player** but flee *before* they can be caught — they're meant to be seen and
chased off, not captured. Tune per species: rabbit = fast skittish dart, deer =
medium grazer, rock/ice monster = slow lumber, lizard = freeze-then-sprint.

**Fliers** wander like roamers but ignore terrain (fly over anything), sort above
the player, don't flee (they scatter/circle), and have no ground collision —
butterflies drifting low, birds/vultures circling high.

### Per-biome fauna roster

| Biome | Roamers (`surface_fauna`) | Fliers | Peekers (`surface_objects`) |
|---|---|---|---|
| **Garden** | rabbit (grass — fast, skittish); small **deer** (grass·forest — grazes, **herds** 2–4); fox (grass·forest — rare, lopes) | butterflies (low, grass), songbirds (high) | burrowing critter, curious alien |
| **Ice** | snow-fox / hare (snow — rare, near-invisible, **bolts**); small **ice-monster** (snow·ice_rock — slow lumber) | snow-petrels (high, over snow·ice) | seal-thing (crevasse edge), alien |
| **Rocky** | rock-dweller / silicon critter (basalt·dust — skitters); small **rock-monster** (basalt — slow, looks fierce/harmless); lava-salamander (lava edge — rare) | ember-moths (low, near lava) | hole-creature |
| **Desert** | sand-lizard (dunes·hard_sand — **freeze-then-dart**); sand-snake (dunes — surfaces & dives); small alien (hard_sand — rare) | vultures / dust-hawks (high, circling over mesa·sandstone) | static sand-critter, alien |
| **Station** | rat / vermin (floor·plating·grate — **scurries to a grate**); | maintenance drone (over floor·plating — patrols, **not shy**) | cyber-roach swarm (grate) |

The **signature roamer** per biome (deer / ice-monster / rock-monster /
sand-lizard / rat) gets the most art + behavior attention — it's what players
notice and chase.

## Critique methodology (run on every object + every populated scene)

An object passes only if it clears all of these. Judge **both** a single sprite
*and* a field of them composited on real terrain at game scale — a sprite can
look fine alone and a stand of them read as noise.

1. **Style coherence.** Matches the world's 3D toon + ink language (ships,
   buildings, terrain, asteroids): consistent ink-outline weight, banded toon
   shading, and the **same world-fixed light direction** as the buildings. No
   flat primitive blobs, no lighting that disagrees with its neighbours.
2. **Storyline adherence.** It is the *right thing for this biome* per this bible
   — native material, palette, and silhouette; placed on the **right terrains**;
   at the **right density tier**. A green leafy bush on lava, or a desert cactus
   in the forest, is an automatic fail. Score each object: *does its existence,
   look, and placement tell this world's story?*
3. **Readability at game scale.** Legible silhouette at 16–24 px on its terrain;
   reads as *what it is* in a glance and is **distinct from its neighbours**
   (an oak vs a birch vs a conifer must differ in silhouette, not just hue).
4. **Aesthetic appeal.** Pleasing form, harmonious palette, one clear focal
   highlight; not muddy, not busy.
5. **Grounding & coherence.** Sits on the ground with a base/contact shadow; no
   floating parts; correct anchor for Y-sort; lighting consistent across frames.
6. **Animation.** Idle life (sway / bob / flicker / blink) is subtle, looped,
   and minimal-frame — sells "alive" without flicker or distraction. Animals
   move more than plants; the shy ones react.
7. **Scene density & variety read.** At target density the biome reads as the
   bible says — *dense lush forest, sparse meadow, barren lava* — with a visible
   density **gradient** between terrains. Variety is legible; repetition is only
   where intended (grass, lichen, lava bubbles).
8. **Palette discipline.** Biome-native and harmonised with the terrain tile it
   sits on; accent/focal colour is earned and rare.
9. **Performance sanity.** Frame counts and per-tile counts stay modest; a dense
   forest must not mean thousands of many-frame sprites.

## Cross-biome notes for the re-art

- **One generator, biome modules.** Keep `objects.py`'s manifest contract
  (name, terrains, density, frames, shy, y_offset). Re-render the art in 3D
  (toon + ink, world-fixed light) at 16–24px, idle animation baked as frames.
- **Signature creature per biome** (garden mammal, ice fox, rocky rock-dweller,
  desert lizard, station rat) gets the most love — it's the thing players
  remember and chase.
- **Density budget:** garden densest → desert/rocky medium (clumped) → ice
  sparsest. Always leave *negative space* (bare mountain, open water, crevasse,
  quicksand, lava) so the busy tiles read.
- **Tie objects to systems where possible:** rocky **ore deposits** echo the
  mining economy; station kiosks/ads echo the market; desert relics hint at
  missions.
