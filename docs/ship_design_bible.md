# Ship Design Bible

The guiding principle: **every ship's silhouette must be legible from its
purpose alone.** A glance from top-down should tell you faction, role, and
rough size — without color. Color is the last 10% of identity, not the first.

Ships are rendered top-down (Blender, cel-shaded, ink outline) for a slightly
cartoonish, friendly-not-scary look.

---

## Factions

### Federation — *"A hull is a fortress that happens to move."*
The old core-world government and its standing navy. Centralized, bureaucratic,
industrially dominant. Doctrine is **attrition through overwhelming force**:
out-armor and out-gun the enemy, accept being slow. Standardized state
shipyards stamp out conservative, over-engineered hulls from reused armored
blocks.

- **Design philosophy:** flat armored citadels, slab sides, layered belt armor,
  ram prows, sponson weapon batteries. Symmetry and mass. Intimidation by bulk.
- **Constraints:** bureaucratic conservatism → blocky, repeated hull sections;
  doctrine hides the crew → enclosed hulls, armored slit cockpits, few windows.
- **Silhouette cues:** wide, angular, **broad-shouldered wedges**; squared
  tails; visible turrets and armor plates. Heaviest aspect ratios in the game.
- **Palette/material:** charcoal/gunmetal, **red** command striping, matte
  heavy plate with rivets and seams. Blue ion drives.

### Rebels — *"Speed is our armor."*
A coalition of frontier worlds and Federation defectors. Resource-poor but
clever and motivated; they cannot out-build the Federation, so they
out-maneuver it. Lightweight composite hulls, oversized engines and control
surfaces, hit-and-run doctrine.

- **Design philosophy:** slender aerodynamic-*looking* hulls, large
  wings/canards, exposed glowing drives, bubble canopies (they trust and
  prize their pilots).
- **Constraints:** scarce materials → minimal armor, every gram counts; limited
  industry → small/medium craft dominate, capital ships are converted civilian
  hulls.
- **Silhouette cues:** **narrow body + big wingspan**, forward canards, swept
  curves, twin booms. Looks fast standing still.
- **Palette/material:** deep blue with **green** running lights/tips, clean
  composite skin. Green proton drives.

### Pirates — *"If it flies and bites, it sails."*
Clans of raiders, ex-miners, and deserters with **no shipyards**. Everything is
salvaged, stolen, and welded together from the wrecks of all three other
factions. Cheap, fragile, menacing; they win by numbers and ambush.

- **Design philosophy:** bolt-on everything; exposed fuel tanks and machinery;
  ramming prows and boarding spikes; trophy parts from kills.
- **Constraints:** salvage-only → **mandatory asymmetry**, mismatched plating
  and engines, one-off silhouettes; no native capitals (only captured hulks).
- **Silhouette cues:** lopsided, lumpy, **asymmetric**; mismatched left/right
  wings; off-center weapons; antennae and spikes.
- **Palette/material:** rust browns and tans, **orange** hazard stripes,
  scavenged faction parts in the wrong colors. Sputtering orange fire drives.

### Merchant / Independent — *"Every cubic meter pays rent."*
The economic backbone: the Merchant Guild plus independent frontier operators.
Pragmatic, cost-driven, function-first. **The cargo defines the ship** — the
hull is built around the hold. Independents also fly aftermarket *surplus*
military hulls (the unaffiliated fighter/corvette/frigate/surplus_carrier):
older-generation, rugged, lived-in mongrels with no allegiance.

- **Design philosophy:** the hold is the hull; standardized detachable
  containers; cheap, repairable, endurance over performance. Surplus combat
  hulls are "used-spaceship" practical — greebly, lived-in, asymmetric-but-
  functional (Firefly/Falcon energy).
- **Constraints:** profit margins → no wasted mass on armor; civilian licensing
  → defensive armament only.
- **Silhouette cues:** **exposed cargo modules/containers** dominate trade
  ships; tiny cockpit vs. huge hold; industrial framework. Surplus warships look
  patched and practical, never sleek.
- **Palette/material:** warm utilitarian tan/grey, **amber/hi-vis** markings,
  exposed paneling. Amber fusion drives.

---

## Ships, by purpose and silhouette

### Independent / Merchant civilian line
| ship | radius | purpose | distinct silhouette |
|---|---|---|---|
| **shuttle** | 10 | the everyman starter; short-hop passenger/light freight | small rounded **teardrop pod**, bubble canopy, stubby fins — friendly egg |
| **prospector** | 14 | cheap nimble miner; tiny hold forces the land-and-sell loop | compact dart with a single **forward drill spike** + swept agility fins |
| **asteroid_miner** | 20 | heavy industrial digger; slow, tough, big hold | squat **crab** with twin forward **drill-arm mandibles** + side ore pods |
| **courier** | 12 | fast small-parcel runner; speed over capacity | thin **needle/cigar** fuselage, long shallow swept fins — all axis |
| **fighter** | 12 | surplus militia interceptor | sleek **delta + canards** (the look you approved) |
| **corvette** | 9 | smallest surplus attack craft; fastest hull in game | tiny **needle blade**, near-no wings, tail tabs |
| **frigate** | 32 | surplus patrol warship; jack-of-all-trades escort | chunky practical combat hull, side weapon nacelles, greebly |
| **freighter** | 32 | medium hauler; balanced cargo runner | boxy hull with **paired side cargo modules** + center hold |
| **hauler** | 40 | dedicated freight spine; "all hold, no hull" | central spine carrying **rows of detachable containers** |
| **bulk_carrier** | 55 | the giant; bulk ore/goods over long routes | enormous **brick of stacked pod columns** — widest cargo mass |

### Federation
| ship | radius | purpose | distinct silhouette |
|---|---|---|---|
| **fed_patrol** | 15 | fast border interdiction; the navy's light hand | angular armored **dart**, short swept wings, red wingtips |
| **fed_destroyer** | 42 | line battlewagon; absorb and deliver punishment | wide **slab fortress**, armor flanks, prow ram, dorsal turrets |
| **fed_missile_cruiser** | 35 | standoff bombardment + small fighter complement | narrow spine flanked by huge **missile-tube pod wings** ("all tubes") |
| **fed_carrier** | 60 | fleet flagship; project airpower | very **wide flat-top flight deck**, offset dorsal island, bow ram |

### Rebels
| ship | radius | purpose | distinct silhouette |
|---|---|---|---|
| **rebel_fighter** | 12 | fast hit-and-run interceptor | slim **winged dart**, big swept wings, green tips, bubble canopy |
| **rebel_gunboat** | 22 | weapon-forward strike craft | stockier body with **twin forward gun booms** + wing plates |
| **rebel_frigate** | 28 | fast escort carrier-frigate | angular wedge, **forward-swept nacelles**, small fighter bay |
| **rebel_carrier** | 55 | mobile base from a converted hull | sleeker than Fed; **open lattice flight deck**, long and narrow |

### Pirates
| ship | radius | purpose | distinct silhouette |
|---|---|---|---|
| **pirate_corvette** | 11 | scrappy ambush raider | asymmetric **scrap blade**, mismatched wings, rust patch |
| **pirate_missile_boat** | 18 | converted hull bristling with stolen ordnance | boxy hull with crude **bolt-on missile racks** welded to the sides |
| **pirate_carrier** | 50 | captured hulk turned mothership | cobbled, **mismatched welded flight deck**, trophy parts, spikes |

### Carriers (cross-faction)
| ship | radius | purpose | distinct silhouette |
|---|---|---|---|
| **surplus_carrier** | 45 | decommissioned light carrier on the open market | streamlined neutral **wedge** with a compact rear deck, single bridge fin |

---

## What renders differently (and why it avoids same-y output)
1. **Hull form is bespoke per ship**, derived from purpose — not a scaled
   template. A miner is a crab; a hauler is a container rack; a carrier is a
   flat-top. These cannot be confused even in shadow.
2. **Aspect ratio carries role:** fighters are slim, freighters are blocky,
   carriers are wide-and-flat, destroyers are broad wedges.
3. **Faction shape language** layers on top (Fed angular/armored, Rebel
   winged/slim, Pirate asymmetric, Civilian cargo-defined).
4. **Drive exhaust** color+shape encodes faction at a glance (ion/proton/fire/
   fusion).
5. **Color is last** — identity must survive grayscale.

---

## Runtime integration (Bevy)

Ships render from a **pre-baked sprite atlas** per ship instead of one static
PNG. Baked by `scripts/ship3d/bake_atlases.py`, loaded via `assets/ships.yaml`
`sprite_path: sprites/ships/atlas/<ship>.png`.

**Atlas format** — 8×8 grid of 128 px tiles (64 frames):
- frames `0..N` (N = `SHIP_HEADINGS` = 32): idle, no drive flame
- frames `N..2N`: identical but with the drive flame
- within each block, frame `i` is the ship **nose-up** lit as if its heading
  were `i·(2π/N)` (the bake spins the *light rig* by `-i·step`, not the ship).

**Heading count N = 32.** Rotation stays smooth (physics drives the Transform),
so N only sets *lighting* granularity — 11.25° steps, fine even on the big slow
capitals whose highlights are largest. Power of two → cheap `round(heading/step)`
indexing and a tidy 8×8 sheet.

**Runtime** ([src/ship.rs](../src/ship.rs)):
- `ship_sprite()` builds an atlas `Sprite`, sized to `radius × 2.2` via
  `custom_size` (tile resolution is decoupled from on-screen size).
- `update_ship_sprite_frame` (Update) sets `index = round(heading/step) mod N
  (+N if thrusting)`. The entity keeps rotating the sprite smoothly; picking the
  matching baked-light frame makes the light land **world-fixed** → highlights
  glide across the hull as the ship turns.
- `DriveActive` is set by `ship_movement` from the thrust command; drives the
  `+N` flame-frame offset.
- Scale animations (dock/launch/land) use `ship_display_size(radius)`, not the
  texture size, since tiles are a fixed 128 px.

Camera MUST remain straight-down (the engine rotates the sprite per heading).
To re-bake at a different N or tile size, edit `bake_atlases.py` (`N`, `TILE`)
and the matching consts in `src/ship.rs` (`SHIP_HEADINGS`, `SHIP_ATLAS_*`).
