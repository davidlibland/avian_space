# Music & planet ambience — plan

Two systems: a **music director** (faction-flavored space themes, biome
surface themes, crossfaded) and an **ambience director** (positional
planet soundscapes driven by what's actually near the player: fauna,
lava, water, machinery). Music comes from Band-in-a-Box sessions you
drive with charts I author, plus in-house synthesized ambient beds;
ambience SFX are synthesized in-house like the weapon sounds — keeping
the whole audio layer license-clean.

---

## 1. Music: what plays when

Context is derived, never scripted: the director watches the game state
every frame and crossfades (~2.5s) when the context key changes.

| Context | Key | Flavor brief |
|---|---|---|
| Main menu | `menu` | The game's title theme — big, hopeful, slow build |
| Space | `space_<controller>` | Faction controlling the CURRENT system (live `GalaxyControl`, so a front flipping mid-war changes the music on your next jump — free drama) |
| Surface | `surface_<biome>` | The landed planet's biome (garden/ice/rocky/desert/interior) |
| Interior: bar | `bar` | Jazz/lounge — the jukebox. BIAB's home turf; this one writes itself |
| Interior: other | inherit surface theme at −6 dB | No new assets needed |
| Combat (later) | intensity layer | Phase 4 — additive percussion stem when hostiles engage |

### Faction space themes (7 to write, 3 reuses)
| Faction | Brief |
|---|---|
| Federation | Noble brass over steady strings — navy in space |
| Rebel | Driving low strings + irregular percussion — gritty, urgent |
| FreeFrontier | Sparse slide guitar + pads — americana against the void |
| Helios | Clean minimal electronica — automated, precise, a little cold |
| Bastion | Anvil-heavy industrial percussion, low male drone |
| Order | Choral pads, bells, open fifths — monastery in vacuum |
| Pirate / contested | Tense ostinato, muted — lawless space |
| Merchant / Independent | reuse FreeFrontier |
| Precursor systems | reuse Order at half tempo (or a dedicated eerie one later) |

### Surface biome themes (5)
garden (pastoral, woodwinds), ice (glassy, high sustained), rocky
(dark percussive), desert (sparse, bent notes), interior/station (warm
synth hum — also covers station "planets").

**Total to produce: ~14 loops** (2–3 min each) + the bar track + title.

## 2. Producing it: Band-in-a-Box workflow

BIAB has no reliable headless/CLI mode, so the honest division of labor:

* **I author, you render.** For each track I produce a one-page brief:
  chord chart (BIAB text format you can paste/import), suggested style
  (e.g. "_ELECPOP.STY", tempo, key), RealTracks picks, form (intro / A /
  B / A / tag, with bar counts chosen so the loop point lands clean),
  and a target mood sentence. You open BIAB, load style, paste chart,
  generate, tweak to taste, **export WAV at 44.1k**.
* **I post-process everything**: trim to the loop point, micro-crossfade
  the seam so it loops seamlessly, normalize to a consistent LUFS,
  encode ogg (`ffmpeg -c:a libvorbis -q:a 5`), drop into
  `assets/music/<key>.ogg`, and the director picks it up by filename.
* **License**: PG Music's terms grant royalty-free commercial use of
  music you create with BIAB/RealTracks (you can't redistribute the
  RealTracks themselves — rendered songs are fine). I'll record it in
  CREDITS-SOUNDS.md and the credits screen.
* **Interim placeholders**: I can synthesize ambient drone-pad
  placeholders for every slot in one script run, so the director ships
  and is testable BEFORE any BIAB session — you replace tracks at your
  own pace and immediately hear them in place.

## 3. Ambience: the planet soundscape

An `AmbienceDirector` (surface + interiors) samples the player's
surroundings every ~2s and maintains two kinds of sound:

* **Beds** (looping, crossfaded, max 2–3 concurrent): chosen by what's
  within ~10 tiles — counts of terrain tiles and venue kind.
  * wind per biome (garden breeze / ice howl / desert hiss / thin rocky
    whistle) — always-on floor for the biome
  * water lapping (garden water tiles), lava bubbling (lava tiles),
    machinery hum (substation), shelf-fan drone (warehouse), deep rumble
    + drips (mine), crowd murmur (bar with NPCs)
* **One-shots** (randomized timer per nearby source, panned toward it,
  pitch-jittered ±10%): fauna calls — bird song & deer huff (garden),
  rat squeaks, bat flutter, gecko chirps, rock-crab clacks, sweeper-bot
  beeps, drone whir; plus door creaks in mazes and coolant-pipe pings.

All synthesized in-house (`scripts/synth_ambience.py`, sibling of the
weapon-SFX script): wind/water/lava/drips/hums are classic procedural
synthesis (filtered noise + LFOs), animal chirps are FM blips — the
toon aesthetic actually WANTS stylized calls, not field recordings.

## 4. Engine work (all phases small)

1. **Music director** — `src/music.rs`: context key derivation,
   two-slot crossfader (two looping `AudioPlayer`s, volume-lerped),
   `music_volume` slider added to Settings (music should duck under a
   separate control from SFX). Ships with synthesized placeholders.
2. **Ambience synth pack + director** — `synth_ambience.py` (~20
   sounds), `AmbienceDirector` with the bed/one-shot logic above,
   `ambience_volume` folded into the SFX slider.
3. **BIAB rounds** — briefs for the 7 faction themes first (space is
   where the game spends most time), then biomes, bar, title. You
   render at your pace; each finished WAV replaces its placeholder.
4. **Polish** — combat intensity stem, jump stinger (the flash wants a
   hit), landing resolve chord, per-storyline stingers (the Pirate
   King deserves leitmotif treatment), Precursor eerie theme.

Suggested start: phase 1 + 2 now (fully in-house, immediately audible),
first BIAB brief pack alongside so you can start rendering whenever.
