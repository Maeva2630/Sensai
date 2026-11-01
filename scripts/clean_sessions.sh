#!/usr/bin/env bash

SESSION_DIR="./sessions"

count_session_entries() {
    find "$SESSION_DIR" -mindepth 1 -maxdepth 1 -printf '.' | wc -c
}

if [[ ! -d "$SESSION_DIR" ]]; then
    echo "Session directory does not exist: $SESSION_DIR"
    echo "There are 0 session files/directories in $SESSION_DIR"
    exit 0
fi

files=$(count_session_entries)
if [[ "$files" -eq 0 ]]; then
    echo "No session files or directories to delete in $SESSION_DIR"
    exit 0
fi

echo "Removing all session files from $SESSION_DIR"
echo "There are $files session files/directories in $SESSION_DIR"

if find "$SESSION_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; then
    echo "All session files successfully deleted"
else
    echo "Failed to delete session files"
    exit 1
fi

files=$(count_session_entries)
echo "There are $files session files/directories in $SESSION_DIR"
