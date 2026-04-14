#!/bin/bash
# Append a timestamped monitor snapshot for the named run to its monitor.md.
# Usage: scripts/monitor_snapshot.sh <run_dir>
#   e.g. scripts/monitor_snapshot.sh experiments/run_19

set -e

run_dir="${1:-experiments/run_19}"
md="$run_dir/monitor.md"
mkdir -p "$run_dir"

{
  printf "\n## %s (hourly)\n\n\`\`\`\n" "$(date '+%Y-%m-%d %H:%M')"
  python3 scripts/monitor_training.py "$run_dir/tb" 2>&1
  printf "\`\`\`\n"
} >> "$md"
