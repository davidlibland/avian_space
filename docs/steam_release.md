# Steam release — license audit, gaps, and process

Audited 2026-07-20. Verdict up front: **nothing blocks a commercial Steam
release**, but there are obligations to honor (LPC art attribution — now
built into the main menu) and **one open item: 51 sound files with
unrecorded origins** that must be documented or replaced first.

---

## 1. License audit

### Code dependencies — CLEAR
`cargo metadata` across all ~900 crates: everything is permissive
(MIT/Apache-2.0/BSD/Zlib/ISC/Unlicense/CC0/MPL-2.0/Unicode/BSL). Notes:

* `self_cell` is `Apache-2.0 OR GPL-2.0-only` — dual-licensed; we take
  Apache. No GPL obligations.
* Two `MPL-2.0` crates: file-level copyleft only — fine for a closed
  binary; obligation is to make the *MPL files'* source available (they
  are unmodified upstream crates, so linking to crates.io satisfies it).
* egui embeds fonts under `OFL-1.1` and `Ubuntu-font-1.0` — both permit
  embedding in applications; OFL only forbids selling the *font itself*.
  Credited in the new in-game Credits screen.
* Bevy and the whole ecosystem: MIT/Apache dual. Nothing viral.

Recommended (not required): add `cargo-deny` to CI with an allow-list so
a future dependency can't silently introduce a GPL/NC license.

### Visual assets — CLEAR (one attribution obligation)
Every sprite except the character sheets is **generated in-house** by
the scripts under `scripts/` (ships, buildings, terrain atlases, fauna,
props, planets, stations, wireframes, effects) — no third-party rights.

**Character sprites are LPC** (Liberated Pixel Cup), documented
per-artist in `assets/CREDITS-SPRITES.md`, under mixes of CC0 /
OGA-BY 3.0 / CC-BY / CC-BY-SA 3.0 / GPL 3.0. This is **compatible with
commercial Steam release** and done by many shipped games, with these
obligations:

1. **Attribution must ship with the game** — DONE: the main menu now has
   a "Credits & Licenses" screen embedding the full credits file.
2. Where entries are multi-licensed, we elect the most permissive
   listed license (OGA-BY over CC-BY-SA over GPL). For SA-only entries,
   ShareAlike applies to the **art**, not the game: our derived layer
   sheets under `assets/sprites/people/layers/` remain under CC-BY-SA
   and the credits say so. (They ship inside the game's assets, which
   satisfies availability; if ever asked, we can also post the sheets.)
3. Do NOT claim the character art as proprietary in the Steam EULA —
   use Valve's default Subscriber Agreement or carve out third-party
   art credits.

### Audio — ONE OPEN ITEM
* All **weapon** SFX: synthesized in-house
  (`scripts/synth_weapon_sounds.py`) — clear.
* **51 files have no recorded origin** (see
  `assets/sounds/CREDITS-SOUNDS.md`): explosions, jump/landing,
  thruster, UI, door, footsteps, escort voice lines.
  * The footstep sets (`footstep_<surface>_000..004`) match the naming
    of Kenney's CC0 audio packs — if that's where they came from,
    they're fine; record it.
  * The escort voice lines (`escorts/*.mp3`) look like TTS output —
    TTS **terms of service vary**; some (e.g. certain online services)
    restrict commercial redistribution. Needs confirmation.
  * **Action:** for each file, either (a) record source + license in
    CREDITS-SOUNDS.md, or (b) replace it — extend
    `synth_weapon_sounds.py` to cover explosions/UI/thruster, or pull
    CC0 equivalents from kenney.nl and note it.

---

## 2. Release-readiness gaps

| Gap | Status | Priority |
|---|---|---|
| In-game credits (license obligation) | **DONE this pass** | — |
| Sound provenance (51 files) | **OPEN — user input or replacement** | Blocking |
| Settings: master volume / music / SFX sliders | Missing | High — Steam players expect it |
| Settings: fullscreen/windowed toggle, resolution | Missing (fixed window) | High |
| Window icon + app icon | Missing | Medium |
| Windows + Linux builds | Only macOS bundle script exists | Blocking for those depots |
| macOS: codesign + notarization | Not set up (Gatekeeper will block unsigned) | Blocking for macOS depot |
| Rebindable keys / controller | Missing | Nice-to-have (note keyboard-only on store page) |
| Steam overlay compatibility | Untested (usually fine with winit/wgpu) | Test during beta |
| Crash telemetry / panic handler | Panics print to console only | Nice-to-have |
| Save compatibility policy | Saves are YAML keyed by struct fields | Adopt "don't rename fields; serde defaults" rule |
| Pause on focus loss | Unknown | Polish |

Deliberate non-issues: no network play (no GDPR/privacy surface), no
IAP, procedural content only (no trademark exposure beyond the title —
**do search "Avian Space" on Steam/trademark DBs before committing to
the name**).

---

## 3. The Steam process, step by step

1. **Steamworks account** — partner.steamgames.com → "Join Steamworks".
   Company or individual; W-8/W-9 tax interview + bank details.
   One-time **$100 app fee** per app (recoupable after $1,000 revenue).
2. **Create the app** — you receive an App ID. Fill the basic package/
   depot layout: one depot per OS (Windows/macOS/Linux).
3. **Store page** — required assets: capsule images (multiple sizes:
   616×353, 231×87, 467×181, 300×450, 660×420 header), 5+ screenshots
   (1920×1080), a trailer is strongly recommended, short + long
   description, tags, system requirements. The page must be live and
   pass review ≥ **2 weeks before launch** ("Coming Soon" period).
4. **Builds** — install `steamcmd`; write an `app_build.vdf` +
   `depot_build.vdf` per OS; `steamcmd +login <user> +run_app_build
   app_build.vdf` uploads to a branch. Set the default branch live from
   the Steamworks UI. (I can write these scripts + a CI job.)
   * Windows: cross-compile with `cargo build --release --target
     x86_64-pc-windows-msvc` (or GNU) + bundle feature; test on real
     Windows for wgpu/DX12.
   * Linux: build on an older glibc (or use `cargo zigbuild`) for
     compatibility; Steam Runtime container ("sniper") is the clean way.
   * macOS: existing `build_macos_app.sh` + `codesign` with a Developer
     ID certificate + `notarytool` submission ($99/yr Apple developer
     account) — otherwise Gatekeeper blocks it for players.
5. **Steamworks SDK (optional at launch)** — achievements, cloud saves,
   overlay hooks via the `steamworks` Rust crate. **Steam Cloud can be
   enabled without code** using Auto-Cloud (point it at the pilots/
   directory path per OS) — recommended, zero code.
6. **Review** — Valve reviews the store page and a build (runs it
   briefly). Typical turnaround: a few days each. Plan: store page up
   ~a month out, build review ~2 weeks out.
7. **Pricing & launch** — set price, regional matrix auto-fills,
   pick launch date, press the button. Post-launch patches are just new
   builds pushed to the default branch.

### Suggested order from today
1. Resolve the 51 sound files (confirm or replace). ← only blocker
2. Settings menu (volume + fullscreen) and window icon.
3. Windows/Linux build scripts + macOS signing; test on hardware.
4. Register Steamworks, reserve the name, start the store page art.
5. Beta on Steam via a private branch (family/friends keys).
