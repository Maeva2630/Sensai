#!/usr/bin/env bash
#
# Delete every saved conversation and artifact (they can hold personal data).
#
# Usage: scripts/clean_sessions.sh [sessions_directory]
# Without an argument, the project's sessions/ folder is cleaned, wherever the script is
# started from. Only portable find options are used, so it works on macOS and Linux.

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_DIR="${1:-$PROJECT_ROOT/sessions}"

count_entries() {
    find "$SESSION_DIR" -mindepth 1 -maxdepth 1 "$@" | wc -l | tr -d ' '
}

if [[ ! -d "$SESSION_DIR" ]]; then
    echo "Session directory does not exist: $SESSION_DIR"
    echo "Nothing to delete."
    exit 0
fi

total=$(count_entries)
if [[ "$total" -eq 0 ]]; then
    echo "No session or artifact files to delete in $SESSION_DIR"
    exit 0
fi

artifacts=$(count_entries -type f \( -name '*_artifact.md' -o -name '*_artifact.json' \))
sessions=$(count_entries -type f -name '*.json' ! -name '*_artifact.json')
others=$((total - artifacts - sessions))

echo "Removing everything in $SESSION_DIR:"
echo "  $sessions conversation file(s), $artifacts artifact file(s), $others other entr(ies)"

if ! find "$SESSION_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; then
    echo "Failed to delete session files"
    exit 1
fi

remaining=$(count_entries)
if [[ "$remaining" -ne 0 ]]; then
    echo "Failed to delete session files: $remaining entr(ies) remain in $SESSION_DIR"
    exit 1
fi
echo "All session and artifact files successfully deleted"
