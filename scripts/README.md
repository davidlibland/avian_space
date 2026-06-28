# scripts/ — tooling & environments

All standalone Python scripts here carry [PEP 723](https://peps.python.org/pep-0723/)
inline metadata (the `# /// script ... ///` header), so the simplest way to run
any of them is to let **uv** build a throwaway environment on the fly — no
virtualenv to create or activate:

```bash
uv run scripts/mission_flowchart.py
uv run scripts/economy.py --tune --write
uv run scripts/generate_planet_sprites.py
```

uv reads each script's declared dependencies and `requires-python`, provisions
them (cached between runs), and runs.

## Shared, reusable environments

To keep a persistent venv (e.g. to run many scripts back to back), two
`uv`-installable requirement sets are provided:

| Env | Requirements | Covers |
|-----|--------------|--------|
| **common** (no GPU) | [requirements/common.txt](requirements/common.txt) | every script except the two 3-D ones |
| **sprites3d** (GPU/EGL) | [requirements/sprites3d.txt](requirements/sprites3d.txt) | `generate_planet_sprites.py`, `generate_app_icon.py` |

```bash
# general-purpose env
uv venv .venv
uv pip install --python .venv -r scripts/requirements/common.txt

# 3-D planet-sprite env (needs a GPU + EGL; pin Python for the `noise` C-ext)
uv venv .venv-sprites --python 3.10
uv pip install --python .venv-sprites -r scripts/requirements/sprites3d.txt
```

## Which script needs what

| Script(s) | Env | Notes |
|-----------|-----|-------|
| `economy.py` | common | supply/demand price autotuner (`--tune --write`) |
| `mission_flowchart.py` | common | renders `docs/mission_flowchart.png` |
| `gen_wireframes.py` | common | HUD target wireframes from ship atlases |
| `generate_sprites.py`, `combine_spritesheet.py`, `objects.py` | common | misc sprite tooling |
| `buildings.py`, `buildings_scifi.py`, `tilegen.py`, `test_blob47.py` | common | terrain / building tile tooling |
| `analyze_progress.py`, `analyze_rewards.py`, `experiment_summary.py`, `monitor_training.py`, `overnight_monitor.py`, `hourly_backup.py` | common | RL training/log analysis (TensorBoard via `tbparse`) |
| `generate_planet_sprites.py` | **sprites3d** | headless pyrender; auto-sets `PYOPENGL_PLATFORM=egl` on Linux |
| `generate_app_icon.py` | **sprites3d** | builds on `generate_planet_sprites` |

## scripts/ship3d/ — Blender pipeline (separate)

The ship/asset 3-D generators in [ship3d/](ship3d/) (`fleet_gen.py`,
`bake_atlases.py`, `buildings3d.py`, …) `import bpy` and run **inside Blender**,
not under uv. Launch them with Blender's bundled Python, e.g.:

```bash
blender --background --python scripts/ship3d/bake_atlases.py
```

## Legacy environment files

The pre-uv specs still work and are kept for reference:
`sprites_environment.yml` (conda, equivalent to **sprites3d**) and
`analysis_requirements.txt` (pip, equivalent to the analysis subset of
**common**).
