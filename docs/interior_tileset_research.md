# Interior tilesets — source online or generate? (research)

Question: for building interiors (and the coming maze interiors), should we
buy/download tilesets, and is that safe for a game sold on Steam — or keep
generating them? Should tilesets vary by world type and building type?

## TL;DR recommendation

**Generate them ourselves with `terrain_gen.py`, one interior atlas per
world style, plus per-building-kind palettes and prop sheets.** Online
tilesets are legally fine (CC0 or standard commercial licenses) but
technically and stylistically a poor fit: nothing sold online ships in our
blob47 autotiler layout, and hand-pixeled art would clash with the game's
3D-rendered toon look. Use good CC0 packs as *reference material*, not as
shipped assets.

## 1. Licensing landscape (the Steam question)

Selling the game on Steam is a commercial use. The safe tiers, best → worst:

* **CC0 / public domain** — no attribution, no restrictions, safe to ship,
  modify, and even re-render. Best sources: [Kenney](https://kenney.nl)
  (thousands of CC0 sprites/tiles, including sci-fi),
  [itch.io's CC0 tileset listings](https://itch.io/game-assets/assets-cc0/tag-tileset)
  ([sci-fi tag](https://itch.io/game-assets/assets-cc0/tag-science-fiction),
  [curated CC0 collections](https://itch.io/c/3621170/cc0-tilesets)), and
  [OpenGameArt's CC0/OGA-BY section](https://opengameart.org/content/cc0oga-by-pixel-art).
* **Paid packs with a commercial license** — the norm on itch.io: shipping
  in a paid game is allowed; *redistributing the assets themselves* (even
  modified) is not. itch.io now offers a
  [General Paid Asset License](https://itch.io/blog/929708/general-paid-asset-license)
  saying exactly this, but licensing is **per-creator, not per-platform**
  ([itch.io support confirms](https://itch.io/t/1402928/commercial-use)) —
  every pack's terms must be read individually. Fine for Steam; a bookkeeping
  burden. Browsable via the
  [commercial-license + royalty-free tags](https://itch.io/game-assets/tag-commercial-license/tag-royalty-free).
* **CC-BY** — commercial use OK with attribution (credits screen). Mild
  burden, easy to satisfy.
* **Avoid**: CC-BY-NC (no commercial use — disqualifying), CC-BY-SA for art
  (share-alike scope around games is murky), anything with unclear
  provenance (asset-flip packs containing traced/ripped art exist; the
  *seller's* license can't cure stolen art — buy only from established
  authors).

So: **legality is not the blocker.** CC0 from a reputable author is
completely safe for Steam.

## 2. Why generating still wins for THIS codebase

1. **Atlas format.** The renderer consumes blob47 autotile atlases —
   47 transition shapes × N terrain tiers × 4 seamless interior variants,
   laid out per `blob47_lut.ron`, with the ±1 tier-gradient contract.
   No commercial tileset ships in this layout; every purchased tile would
   need manual re-cutting into ~200 sub-tiles per biome. That's more work
   than a generator run, and repeats for every reskin.
2. **Style coherence.** Every sprite in the game comes from the same
   pipeline (3D toon renders, Freestyle ink, Standard view transform).
   Interiors already use `terrain_gen.py`'s 3D-lit heightfield atlas —
   a hand-pixeled 16×16 pack would visibly clash with the buildings the
   player just walked past, the character sprites, and the props baked by
   `buildings3d.py`/`objects3d.py`.
3. **Reskin cost is near zero.** The generator already parameterizes
   palette, material, and tier structure — a new interior style is a
   config entry + a bake, not a purchase + recut.

Where online packs DO help: **reference and inspiration** (what makes a
mine read as a mine, palette studies), which CC0 explicitly allows,
including redrawing in our style.

## 3. Per world type, per building type?

* **Per world style: yes — this is the existing pattern.** Exterior
  buildings already reskin across the five styles (colony / cryo /
  extraction / station / outpost). Interiors should follow: one interior
  atlas per style (warm colony timber-and-plaster, cryo insulated panels +
  frost, extraction rough-cut rock + girders, station clean plating,
  outpost scavenged plate). That's five `terrain_gen.py` bakes sharing
  tier semantics (floor / plating / grate / wall / conduit / void), so the
  interior code needs zero changes — `setup_interior` just picks the
  biome by planet style instead of always `"interior"`.
* **Per building type: palette + props, not new atlases.** A bar and an
  outfitter on the same world should share construction (same walls) and
  differ in furnishing: floor-tint per kind plus baked prop sheets
  (tables/bottles vs. racks/plinths vs. cradles/gantries). Full per-kind
  atlases would multiply bake targets (5 styles × 8 kinds) for detail the
  camera barely shows; props carry the identity much more cheaply — and
  we already have a prop pipeline (`buildings3d.py`, `objects3d.py`).
* **Per maze venue (mine, undercity, archive): one atlas each**, because
  their *construction* differs (rock tunnels vs. service corridors vs.
  stacks), shared across worlds with per-world tint. See
  `docs/maze_interiors_plan.md`.
