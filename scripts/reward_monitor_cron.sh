#!/usr/bin/env bash
# Hourly reward-breakdown monitor (per-personality × reward-type) for the
# latest training run. Appends a timestamped snapshot to <run>/monitor.md.
#
# Cron-safe: absolute repo path, isolated venv interpreter, auto-detects the
# highest-numbered experiments/run_N. Reuses scripts/monitor_training.py so the
# snapshot includes the weighted reward breakdown + trend + throughput.
#
# Install (hourly):
#   crontab -e  →  0 * * * * /home/dlibland/dev/avian_space/scripts/reward_monitor_cron.sh
set -euo pipefail

REPO="/home/dlibland/dev/avian_space"
PY="$REPO/.venv/bin/python"
cd "$REPO"

if [ ! -x "$PY" ]; then
  echo "[reward_monitor] venv python missing at $PY" >&2
  exit 0
fi

# Latest experiments/run_N by numeric suffix.
latest="$(ls -d experiments/run_* 2>/dev/null | sed 's#.*/run_##' | sort -n | tail -1)"
if [ -z "${latest}" ]; then
  echo "[reward_monitor] no experiments/run_* found" >&2
  exit 0
fi
run_dir="experiments/run_${latest}"
tb_dir="${run_dir}/tb"
md="${run_dir}/monitor.md"

if [ ! -d "$tb_dir" ]; then
  echo "[reward_monitor] no tb dir at $tb_dir" >&2
  exit 0
fi

{
  printf '\n## %s\n\n```\n' "$(date '+%Y-%m-%d %H:%M %Z')"
  "$PY" scripts/monitor_training.py "$tb_dir" 2>&1
  printf '```\n'
} >> "$md"

echo "[reward_monitor] appended snapshot to $md"
