# Improving interior graphics — options

## Diagnosis: it's the floor tiles, not the props

Everything in this game is 3D-baked in Blender with toon shading + ink
outlines and real lighting — the exterior terrain, the buildings, the
ships, and even the interior **props** (counters, barrels, jail bars).
The one exception is the interior **floor/wall tiles**, which are drawn
flat in Python with PIL `ImageDraw` (`terrain_gen.py` `_interior_base`
/ `_venue_base`): flat fills, a seam line or two, and a whisper of
noise grain. No height field, no shadow, no relief.

The tell: the interior atlases are ~45 KB while the exterior biome
atlases are ~465 KB at the *same* pixel dimensions — 10× less
information, because there's almost nothing to compress. On screen the
floors read as low-contrast grey/tan repeats under nicely-shaded props,
and the mismatch is what looks "not great."

So the props are fine. The win is concentrated on **tiles** (the big
one) and **room layout** (empty rooms feel unfinished).

## The four options

### A. Improved building layouts — cheap, real, do regardless
Pure Rust/data, no art. Bigger rooms, more prop density, structural
variety (support pillars, floor mats/rugs under counters, rope dividers,
hanging lamps, crates against walls), and per-venue dressing so a
warehouse doesn't feel like a bar with different signage. Doesn't fix
flat tiles but makes rooms feel *designed*. **Low effort, complements
everything below.**

### B. Existing licensed tilesets — I'd pass
Off-the-shelf CC0/permissive sci-fi interior tilesets exist (OpenGameArt,
Kenney, itch.io). But two problems make them a poor fit here:
- **Perspective + style clash.** Our world is a 50° oblique toon-ink
  bake. Almost every tileset out there is either flat top-down or
  side-scroller pixel art. Dropping a pixel-art floor under our baked
  buildings/props/ships would look *more* incoherent, not less.
- **Autotiling.** We use blob47 (47-mask) autotiling; a downloaded
  tileset rarely ships a matching 47-tile connectivity set, so it needs
  re-slicing anyway.
- Plus per-set license homework (and the CC-BY-SA-vs-Steam-DRM wrinkle
  we already worked through for the character art).
Verdict: nominally "easy," actually a style regression. Skip.

### C. Regenerate interior tiles through the pipeline we ALREADY own — recommended
The exterior terrain generator (`terrain_gen.py` `render_tile` /
`shade_field`) is a full height-field renderer: normal-mapped Phong
lighting, ambient occlusion in the crevices, per-tile relief. The
interior tiles simply **don't call it** — they take the flat PIL path.

The fix is to give each interior tier a height profile and route it
through the existing shader: recessed floor panels with raised bevels,
grates with real hole-depth and shadow, plating that catches the key
light, tall walls with cast shadow at their base, glowing conduits with
bloom. Same art identity as the buildings and props (because it's the
same renderer), **zero new assets, zero licensing risk**, and it fixes
the actual root cause. Medium effort, all in code we control.
**This is the highest-value, lowest-risk path.**

### D. Generative image model — time-box an experiment at most, eyes open
Tempting, and modern tile-specific models (Retro Diffusion, Tiled
Diffusion/CVPR'25) are decent. But for *this* project:
- **Style match is a gamble** — matching our specific oblique toon-ink
  bake is exactly what diffusion models are worst at (they impose their
  own look; known edge-vignetting and cross-tile inconsistency).
- **It doesn't solve autotiling** — a model gives you a texture, not a
  blob47 connectivity set; you still slice/mask it (which the current
  pipeline already does).
- **Licensing/provenance is the real cost.** We deliberately kept the
  asset base clean for Steam (in-house + CC0/permissive, audited).
  AI-generated art reintroduces the questions we avoided: training-data
  provenance, model-license terms, and the unsettled copyrightability of
  purely-generated images (the US Copyright Office won't register them).
  Not disqualifying, but it trades away a property we spent effort to
  secure.
Verdict: worth a *time-boxed* experiment for texture *bases* only if C
underdelivers — not a first move, and a decision you'd want to make
deliberately given the Steam-cleanliness tradeoff.

## Recommendation

**Do C + A.** Regenerate the interior tiles with real relief/lighting
through the renderer we already own (fixes the flat floors, keeps the
art identity, no licensing risk), and enrich the room layouts (kills the
empty-room feel). Leave the props alone — they're already fine. Hold B
and D in reserve; D only as a deliberate, time-boxed experiment if C
somehow isn't enough.

Suggested order: C first (biggest visual delta), then A dressing on top.
