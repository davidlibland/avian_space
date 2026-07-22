#!/usr/bin/env bash
# CI status for a commit (default: HEAD). Public repo — no auth needed.
# Usage: scripts/ci_status.sh [sha]
set -euo pipefail
sha="${1:-$(git rev-parse HEAD)}"
curl -s --max-time 15 \
  "https://api.github.com/repos/davidlibland/avian_space/actions/runs?head_sha=$sha" \
  | python3 -c '
import json, sys
runs = json.load(sys.stdin).get("workflow_runs", [])
if not runs:
    print("no CI runs found for this commit (yet)")
for r in runs:
    concl = r["conclusion"] or r["status"]
    print(f"{concl:12} {r[\"name\"]}  {r[\"head_sha\"][:8]}  {r[\"html_url\"]}")
'
