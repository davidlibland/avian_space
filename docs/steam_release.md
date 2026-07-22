# Steam release — license audit, gaps, and process

Audited 2026-07-20. Verdict up front: **nothing blocks a commercial Steam
release**. All licenses are cleared and recorded: dependencies
permissive, LPC character-art attribution shipped in the main menu's
Credits screen, and every sound file's source is documented (in-house
synthesis, Kenney CC0, TTSMaker commercial terms).

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

### Audio — CLEAR (confirmed 2026-07-20)
* All **weapon** SFX: synthesized in-house
  (`scripts/synth_weapon_sounds.py`).
* Ambient/UI/footsteps/explosions: **Kenney** packs — CC0 1.0.
* Escort voice lines: **TTSMaker** — their Commercial License Terms
  grant 100% commercial usage rights to generated audio, attribution
  optional. (Minor diligence note: TTSMaker marks a usage scope per
  VOICE in their UI — worth a one-time glance that the voices used
  were commercial-scope, which nearly all are.)
* Per-file record: `assets/sounds/CREDITS-SOUNDS.md`; all three
  sources credited in the in-game Credits screen.

---

## 2. Release-readiness gaps

| Gap | Status | Priority |
|---|---|---|
| In-game credits (license obligation) | **DONE this pass** | — |
| Sound provenance (51 files) | **DONE — Kenney CC0 + TTSMaker, recorded** | — |
| Settings: master volume slider | **DONE** (settings.yaml, live-applied) | — |
| Settings: fullscreen/windowed toggle | **DONE** (borderless fullscreen) | — |
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

## 2b. Steam + Mac App Store dual distribution

**No store-to-store conflict.** Valve's distribution agreement is
non-exclusive; Apple's likewise. Shipping the same game on both is
common. The practical notes:

* **Steam keys are the only parity obligation.** Valve's key rules
  require you not to give OTHER stores a better deal when using
  Steam-generated keys. Selling an independent MAS build is outside
  that; still, keep pricing roughly consistent for goodwill.
* **Two build variants.** MAS requires the App Sandbox entitlement and
  an "Apple Distribution" certificate + provisioning profile;
  Steam/mac uses Developer ID + notarization. Same code, two signing
  paths. (Saves in `~/Library/Application Support/AvianSpace` map into
  the sandbox container automatically.)
* **Anti-steering:** the MAS build must not link out to "buy on
  Steam" or reference other stores' purchasing.
* **No Steamworks SDK inside the MAS build** (and no MAS receipt
  checking in the Steam build) — keep store integrations per-variant.

**The one real wrinkle: CC art vs App Store DRM.** CC-BY 3.0 and
CC-BY-SA 3.0 forbid distributing the licensed work behind "effective
technological measures" that restrict recipients' rights — and Mac App
Store delivery wraps apps in FairPlay DRM. This is the same class of
issue that got GPL apps (VLC) pulled from the App Store.

* **On Steam this is a non-issue IF we don't enable Valve's optional
  DRM wrapper** — ship DRM-free on Steam (recommended; it's opt-in and
  most indies skip it).
* **OGA-BY 3.0 entries are explicitly fine** — OGA-BY exists precisely
  to remove CC's anti-DRM clause. CC0 entries are fine anywhere.
* For LPC entries whose ONLY licenses are CC-BY/CC-BY-SA/GPL, MAS
  distribution is legally gray (the community's "parallel
  distribution" workaround — also offering the art DRM-free — is
  accepted by many artists but not settled law).
* **Options if MAS matters:** (1) audit `CREDITS-SPRITES.md` and
  restrict the compositor to layers from OGA-BY/CC0 entries for the
  MAS build; (2) contact the handful of SA-only artists for a waiver;
  (3) treat MAS as lower priority and ship Steam (DRM-free) +
  optionally itch.io first. Enforcement risk is low, but "low risk"
  is a business decision, not a legal clearance.

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
1. ~~Resolve the 51 sound files~~ DONE (Kenney CC0 + TTSMaker).
2. Settings menu (volume + fullscreen) and window icon.
3. Windows/Linux build scripts + macOS signing; test on hardware.
4. Register Steamworks, reserve the name, start the store page art.
5. Beta on Steam via a private branch (family/friends keys).
