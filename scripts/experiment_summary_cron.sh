#!/usr/bin/env bash
# Recurring distilled summary of the SFT->RL handoff experiment. Appends a
# timestamped block to <latest run>/experiment_summary.md. Cron-safe: absolute
# repo path, venv interpreter, auto-detects the highest-numbered run.
set -euo pipefail

REPO="/home/dlibland/dev/avian_space"
PY="$REPO/.venv/bin/python"
cd "$REPO"
[ -x "$PY" ] || { echo "[exp-summary] venv missing at $PY" >&2; exit 0; }

latest="$(ls -d experiments/run_* 2>/dev/null | sed 's#.*/run_##' | sort -n | tail -1)"
[ -n "${latest}" ] || { echo "[exp-summary] no runs" >&2; exit 0; }
md="experiments/run_${latest}/experiment_summary.md"
mkdir -p "experiments/run_${latest}"

{
  printf '\n## %s\n\n```\n' "$(date '+%Y-%m-%d %H:%M %Z')"
  "$PY" scripts/experiment_summary.py 2>&1
  printf '```\n'
} >> "$md"

echo "[exp-summary] appended to $md"
