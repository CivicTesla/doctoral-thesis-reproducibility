#!/usr/bin/env bash
# Runs one manifest row (task_id = $SLURM_ARRAY_TASK_ID or $1). Atomic
# output: writes to a .tmp dir then renames, so a killed task never leaves
# a partially-written output directory that looks complete.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F0="$(dirname "$HERE")"
MANIFEST="${2:-$F0/06_main_run_manifest.csv}"
TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:?task_id required}}"

ROW=$(awk -F, -v id="$TASK_ID" 'NR==1{for(i=1;i<=NF;i++)h[$i]=i} NR>1 && $h["task_id"]==id{print}' "$MANIFEST")
[ -z "$ROW" ] && { echo "no manifest row for task_id=$TASK_ID"; exit 1; }

IFS=, read -r task_id phase pair mechanisms actions seed_start seed_end n_seeds driver config_hash expected <<< "$ROW"
OUT_FINAL="$F0/runs/${phase}_${pair}"
OUT_TMP="${OUT_FINAL}.tmp"

if [ -f "$OUT_FINAL/_run_metadata.json" ]; then
  echo "already complete: $OUT_FINAL (skip -- idempotent rerun)"
  exit 0
fi
rm -rf "$OUT_TMP"
mkdir -p "$OUT_TMP"
python3 "$F0/$pair/$driver" "$phase" "$seed_start" "$seed_end" "$OUT_TMP"
mv "$OUT_TMP" "$OUT_FINAL"
echo "completed task_id=$task_id -> $OUT_FINAL"
