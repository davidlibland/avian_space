# Building Design Bible

Companion to [`ship_design_bible.md`](ship_design_bible.md). Defines what every
planet-surface building **is for**, and how each of the five frontier
**styles** had to adapt that building to its world. The 3/4 building sprites
(`scripts/ship3d/buildings3d.py`) are designed against this document — function
must read at a glance, and biome must read in the silhouette, not just the paint.

---

## How buildings are used in the game

When the player lands, the planet/station surface becomes a small walkable town.
You set down on the **Ship Pad** and walk between buildings to use their
services; each is entered through a door and opens an interface. The surface
generator only spawns the services a given planet actually offers (a sleepy
farm world may have just a Bar; a capital has everything), so the town's
composition already tells a story before you read a single sign.

Six civic functions exist (`BuildingKind` in `src/surface.rs`):

| Function | In-game purpose | Spawn rule |
|---|---|---|
| **Ship Pad** | Where you land, refuel, repair-on-departure, and take off. The town's anchor. | Always (the landing field itself). |
| **Market** | Buy/sell trade commodities — the core of the economy loop. | If the planet lists commodities. |
| **Outfitter** | Buy/sell weapons, equipment, ship upgrades. | If the planet stocks outfits. |
| **Shipyard** | Buy whole ships (the local shipyard roster). | If the planet sells hulls. |
| **Mechanic Shop** | Hull/system repair bay. Sits beside the pad (you limp in damaged). | Near the pad. |
| **Bar** | Cantina + **mission board**: pick up jobs, rumors, escort/haul/bounty contracts. | Always — every town has a watering hole. |

**Design rule — function reads first.** Across all five biomes, a given
function keeps the same *silhouette grammar* so a returning player recognizes it
anywhere. Biome only changes **material, roof logic, and palette** — never the
functional cue.

### The functional silhouette grammar

- **Ship Pad** — a flat, open field, not a building: a ringed circular apron with
  edge lights and painted approach chevrons. Lowest thing in town; everything
  else rises around it.
- **Market** — low, wide, and *open*: a market hall with a deep awning/portico,
  stacked cargo crates and shutters out front. Welcoming, horizontal.
- **Outfitter** — a compact armored workshop: a blocky hardpoint-studded shed
  with an external weapon rack / test mount on the roof. Reads "arms dealer."
- **Shipyard** — the **tallest working structure**: an open hull-bay with an
  overhead **gantry crane** and a half-built frame inside. Industrial, skeletal.
- **Mechanic Shop** — a wide repair garage with a big **roll-up door**, a small
  jib crane, and tool clutter. Squat, practical, scorched.
- **Bar** — mid-size and the most *inviting*: a glowing sign over the door, warm
  lit windows, a roof vent/chimney. The one building that looks lived-in.

Footprints (from `buildings.py`) give each a natural mass: Shipyard → the
`large_building` (8×6) or `tower`; Market/Bar → `medium_house` (6×5); Outfitter
→ `small_house` (4×4); Mechanic → the wide 4×3 garage; Station services →
`station_room` (5×5). A `tower` per town serves as the civic landmark
(comms mast / control spire).

---

## The five frontier styles — biome adaptation

Each style is a *building tradition* shaped by two pressures: **what the world
gives you cheaply** (you build the bulk of every structure from the local
abundant material, and ration scarce/imported material to the parts that truly
need it) and **what the world does to you** (every building carries features that
exist only to protect its occupants from that biome's specific hazard). Same six
functions, five material economies, five threat models, five answers.

### Colony — garden worlds (`biome: garden`)
**Story.** The easy worlds — breathable air, soft ground, rain. Colonies here
are agrarian and unhurried: the first wave terraformed nothing, they just built.
**Materials economy.** *Abundant & structural:* timber, fieldstone, clay, plant
fiber, water — the bulk of every wall. *Scarce & imported:* refined metal and
composites — rationed to what must be machined (door gear, the shipyard gantry,
the outfitter's weapon mount). Colony builds read as **wood and stone with
sparing metal accents.**
**Hazards (protect against):** rain & damp (rot), seasonal wind-storms,
biological overgrowth (mold, pests), occasional flood.
**Adaptation.** Pitched shingle roofs + deep eaves throw rain clear of the
walls; a raised fieldstone plinth lifts the timber frame above damp and flood;
shutters close over windows for storms. Lowest, most spread-out footprints of
any style — land is cheap and the weather is survivable.
**Palette / glow.** Olive→sage panel `(82,98,72)`→`(155,168,148)`, warm wood
trim, **soft green** window-glow `(180,230,175)` (hydroponic grow-light).
**Roof.** Gabled, shingled, solar slats to the sun.

### Cryo — ice worlds (`biome: ice`)
**Story.** Frozen, wind-scoured, dark half the year. Everything is about heat
retention and not being buried by drift snow.
**Materials economy.** *Abundant & structural:* water-ice and packed snow, cast
and foamed into **cryocrete** ice-brick and thick insulating slab — you build out
of the planet itself. *Scarce & precious:* timber (none at all), metal (imported),
and above all **heat/fuel**. Every gram of warmth is hoarded, so the architecture
is shaped to keep it in.
**Hazards (protect against):** lethal cold (heat loss), blizzard wind, snow-load
& drift burial, ice-heave cracking the ground, months of darkness.
**Adaptation.** Walls are thick double-insulated cryocrete (the silhouette looks
*padded*); buildings hunker and berm half-buried into the ice to duck the wind
and borrow its thermal mass; entries are raised **storm-porch airlocks** above the
drift line (no cold blast straight in); roofs are steep smooth domes/wedges so
snow slides off before it loads them. Openings are few and deep. A heated apron
ring melts the pad clear.
**Palette / glow.** Pale ice-blue panel `(195,218,238)`→`(218,232,245)` rimed
white, frost on the wind-side faces, **cyan** heat-trace glow `(145,220,255)`
leaking from every seam — the hoarded warmth inside, escaping as light.
**Roof.** Steep, smooth, snow-shedding domes/wedges with heat-trace lines.

### Extraction — rocky & volcanic worlds (`biome: rocky`)
**Story.** You don't *live* here, you *work* here. Mining and refining camps
clamped onto bad ground near heat and ore.
**Materials economy.** *Abundant & cheap:* stone, slag, and **metal** — they mine
and smelt it on-site, so raw steel/iron plate is the one place in the setting
where metal is lavished structurally, not rationed. *Scarce & imported:* organics,
soft goods, comfort, and **clean air/water** (filtered or shipped in). Builds read
as **riveted plate and cut stone, everywhere.**
**Hazards (protect against):** rockfall & landslide, ashfall, extreme heat / lava
proximity, seismic shock, toxic gas & abrasive dust.
**Adaptation.** Heavy armored stone-and-steel shells with **sloped roofs to shed
rockfall and ash**; structures stand on **stilts/piers** over unstable scree (and
to clear ash drift / radiant ground heat); interiors are **sealed and filtered**
against toxic air, so light only escapes from pressure-sealed seams; external
scrubbers, ducting, ore hoppers and stacks bolt to every face. Soot-stained,
blunt, function over comfort.
**Palette / glow.** Rust-brown→tan plate `(88,72,55)`→`(118,95,68)`, exposed
riveted steel, **molten-orange** glow `(255,135,55)` (forge/refinery light and
hazard strips).
**Roof.** Flat/sloped industrial decks crowded with tanks, vents, scrubbers, stacks.

### Station — city & orbital worlds (`biome: city`)
**Story.** The dense, wealthy core — arcology blocks and station decks where
land is vertical and everything is engineered.
**Materials economy.** *Abundant & cheap (via manufacture/trade):* engineered
composites, alloys, glass, **prefab modules** — nothing is "natural," everything is
fabricated, and it's all cheap here at the hub. *Scarce & precious:* **floor area
itself** — land is the constraint, so you build *up*, not out; genuine wood or
stone is a luxury affectation. Builds read as **flush manufactured panel and glass.**
**Hazards (protect against):** overcrowding/density, thin or vacuum atmosphere
(stations are pressurized), radiation, and engineered failure modes (fire,
decompression) rather than weather.
**Adaptation.** Tall **stacked modular prefab** (space is the scarce resource →
go vertical), **flush sealed curtain walls** rated for pressure differential,
redundant systems, holographic signage layered on because manufactured glow is
free. The most *finished*-looking tradition — and the most anonymous, until you
read the glowing logos.
**Palette / glow.** Charcoal→slate composite `(48,55,65)`→`(68,78,92)`,
brushed-metal trim, **electric-cyan** signage glow `(55,195,235)`.
**Roof.** Flat tech decks with antenna farms, beacons, and rooftop signage.

### Outpost — desert worlds (`biome: desert`)
**Story.** Hot, dry, dust-storm frontier. Trade waystations and prospector
hubs clinging to a water claim.
**Materials economy.** *Abundant & structural:* regolith and sand, rammed and
cast on-site into **adobe/sandcrete**, plus stone — cheap thermal mass straight
out of the ground. *Scarce & precious:* timber (none), metal (imported), and
above all **water** — the whole reason the outpost exists, displayed and guarded.
Builds read as **earthen mass with a prized metal water tank on top.**
**Hazards (protect against):** extreme heat and huge day↔night temperature
swings, abrasive dust storms (burial, infiltration), glare/UV, dehydration.
**Adaptation.** Thick thermal-mass adobe walls ride the temperature swing (cool
by day, warm by night); roofs are **domed/vaulted** because there's no timber to
span a flat one — and the curve sheds dust; windows are **small, deep-set and
shaded** against glare; **wind-baffles and dust-skirts** flare the base against
storms and burial; the building is sealed against dust. A rooftop **water tank +
condenser fins** crowns each one — the most valuable thing in town, worn proudly.
**Palette / glow.** Sand→pale-clay `(158,138,105)`→`(192,172,138)`, sun-bleached,
**amber** glow `(225,190,105)` (low warm interior light, and the lit water tank).
**Roof.** Domed/vaulted adobe with parapets and a tank.

---

## Reading the matrix (function × biome)

The function tells you the **shape**; the biome tells you the **skin and roof**.

| | Colony (garden) | Cryo (ice) | Extraction (rocky) | Station (city) | Outpost (desert) |
|---|---|---|---|---|---|
| **Market** | open timber hall, awning, produce crates | insulated trade vault, storm porch | ore-broker shed, hopper out front | clean modular arcade, holo-prices | shaded souk, dust-screen awning |
| **Outfitter** | panelled workshop, rooftop rack | domed armory, frosted | armored arms-shed, hazard stripes | prefab arms block, holo-sign | adobe gunsmith, shade screen |
| **Shipyard** | timber gantry, half-framed hull | heated hull-dome, steep roof | the biggest rig — crane, stacks, glow | tall clean hull-tower, beacons | open sun-bay, water tank, gantry |
| **Mechanic** | wood-and-panel garage | heated bay, melt-ring | scorched repair rig, jib crane | flush service block | dusty repair shed, dust-skirt |
| **Bar** | cozy gabled inn, lit windows | warm half-buried lodge, cyan seams | grimy miners' canteen, orange glow | neon arcade bar, holo-sign | adobe cantina, amber lamp |
| **Ship Pad** | grass-edged apron | heated melt-ring apron | blast-scorched ferro-pad | lit deck pad, painted | dust-swept apron, wind socks |

---

## Rendering notes (for `buildings3d.py`)

- **Projection:** 3/4 oblique (high-ish elevation so it sits on the top-down
  terrain without fighting it), orthographic, world-fixed light, toon shading +
  Freestyle ink — same family as the ships/asteroids.
- **Anchor:** sprite pivots at the *front-center of the ground footprint*; the
  building rises up-screen. Engine Y-sorts by that base so towns occlude
  naturally and the player can walk "behind" a building.
- **Glow:** the per-style accent color is an emissive material on windows / door
  sign / hazard strips — it's the cheapest, strongest "this place is alive" cue
  and the fastest biome tell at a distance.
- **Material & hazard read:** every sprite should *show its economy* — bulk built
  from the local abundant material (timber/ice-brick/plate/prefab/adobe), with the
  scarce imported metal visible only where it's earned (the shipyard gantry, the
  outfitter mount, door gear, the desert water tank) — and carry at least one
  visible **hazard-protection feature** for its biome (eaves, storm-porch, stilts,
  pressure seam, dust-skirt). Function silhouette stays constant across biomes;
  material, roof and protective feature change per the sections above.
- **Consistency:** one bespoke sprite per (style × function/template).

## Critique methodology (run on every building before it ships)

Each building is checked against all of the following; a fail sends it back.

1. **Multi-angle coherence** *(critique only — these alt-angle renders are never
   used in game)*: render from the game angle (elev 50, azim 0) **plus** two side
   3/4s and a low "eye" angle. Nothing may float or detach — every support must
   reach what it holds, every cantilever (awning, sign, gantry beam, crane arm)
   must visibly attach to the building. Thin vertical supports vanish near
   top-down, so cantilevers need stout/gusseted/sloped connections that read at
   50°, not just from the side. (`buildings3d.py angles`)
2. **Tile-true footprint:** base/collision box is an integer W×D tiles, verified
   on a grid overlay (roofs/eaves/props may overhang; the *base* may not).
3. **Doorway alignment:** the painted door sits over the walkable door tile(s);
   even-width buildings get a centred 2-tile entrance.
4. **Material economy via texture:** bulk uses the biome's abundant material as a
   real texture — timber plank-grain (colony), cut-stone/brick (extraction +
   every stone plinth), brushed metal, cryocrete, adobe — and scarce metal shows
   only where earned. No flat untextured fills on large faces.
5. **Specularity:** metal is shiny (sharp toon spec); timber/stone/adobe are
   matte (spec ≈ 0); glass/signs/windows are emissive glow.
6. **Detail budget — interesting, not cluttered:** one clear functional prop +
   texture per building, not a pile of greebles. If two angles read as "busy,"
   cut detail. (e.g. the shipyard rear windows were thinned to one row.)
7. **No "stripe" artifacts:** thin horizontal bands compress into black stripes
   near top-down — carry the wall with texture + corner posts, not dark rails.
