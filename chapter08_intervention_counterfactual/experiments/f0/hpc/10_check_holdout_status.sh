#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; F0="$(dirname "$HERE")"
for pair in pair1 pair2; do
  d="$F0/runs/holdout_${pair}"
  if [ -f "$d/_run_metadata.json" ]; then echo "holdout_${pair}: COMPLETE"; cat "$d/_run_metadata.json"
  else echo "holdout_${pair}: NOT_STARTED_OR_INCOMPLETE"; fi
done
