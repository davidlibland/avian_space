# Avian Space

An open-world space RPG built with [Bevy](https://bevyengine.org/) and [Avian2D](https://github.com/Jondolf/avian) physics, where AI ships are trained using reinforcement learning.

![Screenshot](screenshot.png)

## Overview

Explore a galaxy of interconnected star systems as a pilot forging your own path. Trade commodities between planets, hunt bounties on wanted ships, mine asteroids for resources, or follow hand-crafted storylines that unlock as you progress. The universe is configured entirely through YAML files, making it straightforward to add new missions, ships, weapons, and star systems without touching code.

What sets Avian Space apart is its AI: other ships in the world are controlled by neural networks trained via behavioral cloning and proximal policy optimization (PPO), producing emergent, adaptive behavior rather than scripted routines.

## Features

- **Multiple career paths** -- Take on delivery contracts as a merchant, destroy targets as a bounty hunter, harvest asteroid fields as a miner, or mix and match as you see fit.
- **Story-driven and procedural missions** -- Hand-crafted mission chains with preconditions sit alongside parameterized templates that generate varied objectives dynamically.
- **Inter-system travel** -- Jump between star systems through hyperspace, each with its own planets, stations, and hazards.
- **Combat and weapons** -- Lasers, missiles, proton beams, space mines, and more, each with distinct behavior and sound effects.
- **Planet landing and trade** -- Dock at planets and stations to buy and sell commodities, pick up missions, and outfit your ship.
- **Data-driven configuration** -- Ships, weapons, star systems, missions, and items are all defined in YAML files under `assets/`.
- **RL-trained AI ships** -- NPC ships use neural network policies trained with behavioral cloning and PPO, powered by the [Burn](https://burn.dev/) ML framework.

## Running

Requires Rust. All commands should include `--features dev` for fast incremental builds during development.

```bash
# Play the game (default: behavioral cloning training mode)
cargo run --features dev

# Play with RL-trained AI (inference only, no training)
cargo run --features dev -- --inference

# Classic mode (rule-based AI, no neural networks)
cargo run --features dev -- --classic

# RL training mode
cargo run --features dev -- --rl-training

# Headless training (no renderer, faster)
cargo run --features dev -- --bc-training --headless

# Start fresh, ignoring saved checkpoints
cargo run --features dev -- --inference --fresh
```

### Training backend

The training thread uses Burn's `wgpu` backend by default — Metal on macOS,
Vulkan/DX12 elsewhere. On a Linux/Windows box with the CUDA toolkit installed,
add `cuda` to the feature list to swap in Burn's native CUDA backend:

```bash
cargo run --features "dev cuda" -- --rl-training
```

This requires CUDA 12.x and `nvcc` on `PATH` (`cubecl-cuda` builds GPU kernels
at runtime). On Ubuntu the recommended install is the `cuda-toolkit-12-4`
package from NVIDIA's repo.

### Linux build dependencies

Bevy needs the standard Linux audio / windowing dev packages. On Ubuntu:

```bash
sudo apt-get install -y libwayland-dev libxkbcommon-dev libudev-dev \
    libasound2-dev libx11-dev libxcursor-dev libxi-dev libxrandr-dev
```

## Adding Content

All game content lives in YAML files under [`assets/`](assets/):

| File | Purpose |
|------|---------|
| `star_systems.yaml` | Star systems, planets, and connections |
| `ships.yaml` | Ship definitions (stats, sprites, factions) |
| `weapons.yaml` | Weapon types and parameters |
| `missions.yaml` | Hand-crafted story missions |
| `mission_templates.yaml` | Parameterized templates for procedural missions |
| `outfitter_items.yaml` | Purchasable ship upgrades |
| `npc.yaml` | Recurring storyline NPC characters (names + composited appearances) |

New storylines can be added by writing mission entries in `missions.yaml` with briefing text, objectives, rewards, and optional preconditions that gate progression.

## Testing

```bash
cargo test --features dev
```

## Building a Standalone Bundle

The `bundle` cargo feature bakes the entire `assets/` directory into the
binary (via `bevy_embedded_assets` + `include_dir`), producing a fully
self-contained executable — no assets folder needs to ship alongside it:

```bash
cargo build --release --features bundle
```

Note: build **without** the `dev` feature — `dev` enables Bevy
`dynamic_linking`, which doesn't relocate cleanly into a distributable.

On macOS, `scripts/build_macos_app.sh` wraps this into a double-clickable
`.app` (output: `target/macos/AvianSpace.app`; when launched from Finder it
defaults to `--inference` mode):

```bash
scripts/build_macos_app.sh                # release build + package
SKIP_BUILD=1 scripts/build_macos_app.sh   # repackage an existing binary
```

## Regenerating Character Sprites

Character layer sheets under `assets/sprites/people/layers/` are generated
(and committed) by `scripts/people_lpc_gen.py` from the LPC asset collection.
The LPC source repo is large (~800 MB) and gitignored, so clone it once before
regenerating:

```bash
git clone --depth 1 \
    https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator.git \
    scripts/lpc

.venv/bin/python scripts/people_lpc_gen.py            # regenerate layers + manifest + credits
.venv/bin/python scripts/people_lpc_gen.py --preview  # also render a contact sheet to /tmp
.venv/bin/python scripts/people_lpc_gen.py --strict   # only CC0/OGA-BY/CC-BY assets (DRM-safe)
```

Which items/colors ship is curated in `scripts/people_lpc_config.py`. Output
is deterministic — re-running with an unchanged config and LPC checkout is
byte-identical. The committed sheets were generated from LPC commit
`d57c8424` (2026-07-06).

## Art Credits & Licensing

NPC and player character sprites are composited at runtime from layers derived
from the [Liberated Pixel Cup (LPC)](https://lpc.opengameart.org/) asset
collection (via the [Universal LPC Spritesheet Character Generator](https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator)).
Those art assets — including our derived layer sheets under
`assets/sprites/people/layers/` — are licensed CC-BY-SA 3.0 / GPL 3.0 (some
CC0 / OGA-BY / CC-BY); per-artist attributions are auto-generated into
[`assets/CREDITS-SPRITES.md`](assets/CREDITS-SPRITES.md) by
`scripts/people_lpc_gen.py`. These licenses cover the art only, not the game
code. When distributing builds, ship the credits file and avoid wrapping the
game in DRM (or regenerate with `--strict` to restrict to CC0/OGA-BY/CC-BY
assets).
