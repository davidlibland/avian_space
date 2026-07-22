#!/usr/bin/env bash
# Turn in-game bug-report bundles into GitHub issues (one per bundle).
#
# The `bugreport` tester builds write bundles (report.md, screenshot.png,
# pilot.yaml, log.txt) to <user data dir>/bug_reports/<timestamp>/.
# This script files each un-synced bundle as a GitHub issue via `gh` and
# marks it with a `.issue` file containing the issue URL.
#
# Usage: scripts/sync_bug_reports.sh [reports-dir]
set -euo pipefail

if [[ $# -ge 1 ]]; then
    DIR="$1"
elif [[ "$(uname)" == "Darwin" ]]; then
    DIR="$HOME/Library/Application Support/AvianSpace/bug_reports"
else
    DIR="./pilots/../bug_reports"  # dev builds: beside ./pilots
    [[ -d "$DIR" ]] || DIR="./bug_reports"
fi

if [[ ! -d "$DIR" ]]; then
    echo "No bug reports directory at: $DIR"
    exit 0
fi

# Make sure the label exists (no-op if it already does).
gh label create bug --description "Player-filed bug report" --color d73a4a 2>/dev/null || true

filed=0 skipped=0
for d in "$DIR"/*/; do
    [[ -f "$d/report.md" ]] || continue
    if [[ -f "$d/.issue" ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    stamp="$(basename "$d")"
    # Title: first line of the player's note if present, else the timestamp.
    note="$(sed -n '/^## Player note/,$p' "$d/report.md" | sed '1,2d' | sed '/^$/d' | head -n1 || true)"
    if [[ -n "$note" ]]; then
        title="Bug: ${note:0:70}"
    else
        title="Bug report $stamp"
    fi

    body="$(cat "$d/report.md")"
    if [[ -f "$d/log.txt" ]]; then
        body+=$'\n\n## Log tail\n\n```\n'
        body+="$(tail -n 60 "$d/log.txt")"
        body+=$'\n```'
    fi
    body+=$'\n\nBundle (screenshot + pilot save): `'"$d"'`'

    url="$(gh issue create --title "$title" --body "$body" --label bug)"
    echo "$url" > "$d/.issue"
    echo "Filed: $title -> $url"
    filed=$((filed + 1))
done

echo "Done: $filed filed, $skipped already synced."
