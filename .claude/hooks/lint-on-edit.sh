#!/usr/bin/env bash
# Non-blocking lint after Edit/Write to .py files. Exit 0 always.
# Surfaces flake8 warnings to stderr so Claude sees them; never blocks the edit.
set -u

FILE="${1:-}"
[[ -z "$FILE" ]] && exit 0
[[ "$FILE" != *.py ]] && exit 0
[[ ! -f "$FILE" ]] && exit 0

if command -v flake8 >/dev/null 2>&1; then
  flake8 "$FILE" --max-line-length=100 --extend-ignore=E203,W503 2>&1 | head -20 >&2
fi

exit 0
