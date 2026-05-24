#!/usr/bin/env bash
# PreToolUse: block Edit/Write inside results/<method>/flight_X/ to
# prevent the agent from overwriting solver artefacts by hand. These
# files (solution.npz, meta.json, the 8 standard PNGs) must only be
# (re-)written by the solvers themselves -- editing them manually has
# silently corrupted comparisons in the past.
#
# Exit 0 -> allow the operation.
# Exit 2 -> block the operation; stderr is shown to Claude.
set -u

FILE="${1:-}"
[[ -z "$FILE" ]] && exit 0

NORM_FILE="${FILE//\\//}"
NORM_FILE="${NORM_FILE#./}"

case "$NORM_FILE" in
  results/direct/flight_*|results/rl/flight_*|results/compare/flight_*)
    echo "[block-result-overwrites] BLOCKED: $NORM_FILE is a solver artefact." >&2
    echo "  Regenerate it via the responsible pipeline instead:" >&2
    echo "    python -m src.direct.solve --flight <id>" >&2
    echo "    python -m src.rl.evaluate  --flight <id>" >&2
    echo "    python -m src.compare.make_report_figs" >&2
    exit 2
    ;;
esac

exit 0
