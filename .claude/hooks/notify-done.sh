#!/usr/bin/env bash
# Fired on Stop. Console banner + bell so you can see/hear when Claude finishes.
TS=$(date '+%Y-%m-%d %H:%M:%S')
printf '\a\n=========================================\n  Claude finished at %s\n=========================================\n' "$TS"
exit 0
