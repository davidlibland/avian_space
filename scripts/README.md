# Scripts

Utility scripts for the avian_space project. Each script has its own
environment specification — see below.

## `generate_sprites.py`

Procedurally generates all ship sprite PNGs and writes them to
`assets/ship_sprites/`.

**Environment:**

```bash
conda env create -f scripts/sprites_environment.yml
conda activate avian-sprites
python scripts/generate_sprites.py
```

Dependencies are pinned in `sprites_environment.yml` (Python 3.9,
Pillow 11.3.0).

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
