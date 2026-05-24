#!/usr/bin/env bash
# PostToolUse: auto-run dynamics / wind tests when the canonical math
# modules are edited. CLAUDE.md requires this before commit, so we wire
# it as a hook to make the rule unforgettable.
#
# Non-blocking: failures surface to stderr (Claude sees them) but the
# edit itself is never reverted.
set -u

FILE="${1:-}"
[[ -z "$FILE" ]] && exit 0

# Normalize backslashes (Windows) and strip leading ./.
NORM_FILE="${FILE//\\//}"
NORM_FILE="${NORM_FILE#./}"

case "$NORM_FILE" in
  *src/common/dynamics.py|*src/common/wind_field.py|*src/common/aircraft.py|*src/common/cost.py)
    ;;
  *)
    exit 0
    ;;
esac

echo "[auto-test-dynamics] $NORM_FILE edited; running tests/test_dynamics.py tests/test_wind_field.py" >&2

if ! command -v pytest >/dev/null 2>&1; then
  echo "[auto-test-dynamics] pytest not on PATH; skipping" >&2
  exit 0
fi

pytest tests/test_dynamics.py tests/test_wind_field.py -q --no-header 2>&1 | tail -30 >&2 || true
exit 0
