# Scripts

Utility scripts for the avian_space project. Each script has its own
environment specification — see below.

## `generate_sprites.py`

Procedurally generates all ship and weapon sprite PNGs and writes them to
`assets/sprites/ships/`.

**Environment:**

```bash
conda env create -f scripts/sprites_environment.yml
conda activate avian-sprites
python scripts/generate_sprites.py
```

Dependencies are pinned in `sprites_environment.yml` (Python 3.9,
Pillow 11.3.0).

## `generate_planet_sprites.py`

Procedurally generates planet sprite PNGs for every planet listed in
`assets/star_systems.yaml` and writes them to `assets/sprites/planets/`.

Each planet's `planet_type` field in the YAML selects a factory function
(habitable, rocky, cloud, desert, gas_giant, gas_giant_rings, ice_giant,
icy_dwarf) and the existing `color` field drives the palette. Sprites are RGBA PNGs sized to match the game radius (`radius * 2.2`, same
convention as ship sprites), rendered from 3D simplex-noise geometry via
pyrender.

**Environment:**

```bash
conda env update -f scripts/sprites_environment.yml
conda activate avian-sprites
python scripts/generate_planet_sprites.py
```

Additional dependencies beyond the ship sprite env: PyYAML, trimesh, pyrender,
noise (all listed in `sprites_environment.yml`).

## `analyze_rewards.py`

Reads TensorBoard event files from an experiment run and prints
per-personality × reward-type tables showing reward totals, per-step
averages, value-head explained variance, and training diagnostics.

**Environment:**

```bash
pip install -r scripts/analysis_requirements.txt
python scripts/analyze_rewards.py                        # latest run
python scripts/analyze_rewards.py experiments/run_3/tb   # specific run
```

Dependencies are pinned in `analysis_requirements.txt` (tbparse 0.0.9,
pandas ≥2.0, numpy ≥1.22, tensorboard ≥2.12).

