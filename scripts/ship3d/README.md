# Sprite generators

All in-game sprites are **procedurally generated** by the scripts here (3D
rendered offline with Blender's `bpy`, or PIL/numpy for flat textures) and
committed as PNGs under `assets/sprites/`. Re-run a generator after editing it
to regenerate its assets — nothing is hand-drawn.

Design rationale & per-ship concepts: [`docs/ship_design_bible.md`](../../docs/ship_design_bible.md).

## Virtualenvs (created with `uv`, one-time)
| venv | contents | used by |
|---|---|---|
| `scripts/.blender_venv` | Python 3.13 + `bpy` (~390 MB) + pillow | all Blender (3D) generators below |
| `scripts/.sprite3d_venv` | numpy + pillow | flat-texture / montage scripts |
| `scripts/.planets_venv` | numpy, pyyaml, trimesh, pyrender, noise, pillow | planet generator |

Run a script with its venv's python, e.g.
`scripts/.blender_venv/bin/python fleet_gen.py`.

## Generators
| script | venv | produces → output |
|---|---|---|
| `blender_gen.py` | — | **shared library** (no output): mesh primitives, toon/glow materials, top-down ortho camera + light rig + Freestyle ink outline + Standard view transform. Imported by the others. |
| `fleet_gen.py` | blender | bespoke builders for all 22 ships + previews/montages (`out/`). The geometry source of truth for ships. |
| `bake_atlases.py` | blender | **ship sprites** → `assets/sprites/ships/atlas/<ship>.png` (8×8 = 32 heading × idle/thrust frames). |
| `weapons_gen.py` | blender | **weapon projectiles** → `assets/sprites/weapons/{ir_missile,javelin,goose,space_mine}.png`. |
| `asteroid_proto.py` | blender | asteroid geometry/materials (rock mesh, deposits, cracks, tumble) + a single-shape preview. Imported by `asteroid_gen.py`. |
| `asteroid_gen.py` | blender | **asteroid library** → `assets/sprites/asteroids/rock_<i>.png` + `dep_<i>.png` (12 shapes; 8×8 = 64-frame tumble atlases; rock + tintable deposit mask). |
| `pickup_gen.py` | blender | **pickup crystal** → `assets/sprites/pickups/crystal.png` (4×4 = 16-frame tumble atlas, tintable). |
| `effects_gen.py` | sprite3d | **VFX textures** → `assets/sprites/effects/smoke.png` (damage-smoke puff). |
| `buildings3d.py` | blender | **3/4 building sprites** → `assets/sprites/worlds/buildings3d/<style>_<func>{,_front,_back}.png` + `buildings3d_manifest.ron`. 6 functions (market/outfitter/shipyard/mechanic/bar/pad) × 5 biome styles, each forward-facing 3/4 and **depth-split** into `_back` (behind the player) + `_front` (over the player, doorway cut open) so the walker stands framed in a doorway. Designed per [`docs/building_design_bible.md`](../../docs/building_design_bible.md). `bake` mode writes the committed sprites; `angles <style>` renders the multi-angle coherence sheet. Footprints/door-funcs must match `kind_template`/`spawn_building_3d` in `src/surface.rs`. |
| `terrain_gen.py` | sprite3d | **world terrain atlases** → `assets/sprites/worlds/{biome}_atlas.png` (5 biomes; 3D-lit heightfield, real inter-terrain elevation, organic blob47 transitions, 4 seamless interior variants). Bumps `atlas_cols` in `blob47_lut.ron`/`world_manifest.ron`. Replaces only the *visuals* — `tilegen.py` still produces the collision/terrain/manifest `.ron` metadata. Interior-variant count must match `INTERIOR_VARIANTS` in `src/world_assets.rs`. |
| `../generate_planet_sprites.py` | planets | **planet sprites** → `assets/sprites/planets/<planet>.png` (reads every `assets/*system*.yaml`; skips existing, `--force` to rebake). |

Frame counts in the bakers must match the matching Rust constants:
- ships: `SHIP_HEADINGS` / `SHIP_ATLAS_*` in `src/ship.rs`
- asteroids: `ASTEROID_TUMBLE_FRAMES` / layout grid in `src/asteroids.rs`
- pickups: `PICKUP_TUMBLE_FRAMES` / layout grid in `src/pickups.rs`

## Legacy / superseded
- `render3d.py`, `ships3d.py`, `generate_3d_ships.py` — the original pure-numpy
  software renderer (pre-Blender prototype). Kept for reference.
- `../generate_sprites.py` — the original flat 2D PIL ship/weapon generator,
  superseded by the 3D pipeline. Don't run it (it would re-clutter `ships/`).
- `_asteroid_hero.py`, `_*.py` and `out/_*.png` — scratch previews/showcases.
