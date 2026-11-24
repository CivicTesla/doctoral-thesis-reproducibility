#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; F0="$(dirname "$HERE")"
for pair in pair1 pair2; do
  d="$F0/runs/main_${pair}"
  if [ -f "$d/_run_metadata.json" ]; then echo "main_${pair}: COMPLETE"; cat "$d/_run_metadata.json"
  elif [ -d "${d}.tmp" ]; then echo "main_${pair}: IN_PROGRESS (tmp dir present)"
  else echo "main_${pair}: NOT_STARTED"; fi
done
