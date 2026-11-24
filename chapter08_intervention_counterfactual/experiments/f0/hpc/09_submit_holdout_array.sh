#!/usr/bin/env bash
# Submits ONLY the holdout-phase tasks, and only after 08_freeze_pre_holdout
# has produced a commit. Holdout runs exactly once (Section 6).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; F0="$(dirname "$HERE")"
MANIFEST="$F0/06_main_run_manifest.csv"

if [ -f "$F0/runs/holdout_pair1/_run_metadata.json" ] && [ -f "$F0/runs/holdout_pair2/_run_metadata.json" ]; then
  echo "REFUSING: holdout already run once (Section 6: single-shot only)"
  exit 1
fi
cd "$F0/../.."
if ! git log --oneline -1 --grep="Freeze B1 final claims before holdout evaluation" | grep -q .; then
  echo "REFUSING: no pre-holdout freeze commit found -- run 08_freeze_pre_holdout.sh first"
  exit 1
fi

HOLDOUT_IDS=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i} NR>1 && $h["phase"]=="holdout"{print $h["task_id"]}' "$MANIFEST" | paste -sd, -)
if command -v sbatch >/dev/null 2>&1; then
  sbatch --array="$HOLDOUT_IDS" --cpus-per-task=4 --mem=4G --time=02:00:00 \
    --job-name=f0_holdout -o "$F0/hpc/logs/holdout_%a.out" \
    "$HERE/02_run_array_task.sh" '$SLURM_ARRAY_TASK_ID' "$MANIFEST"
else
  echo "no sbatch -- local fallback:"
  for id in ${HOLDOUT_IDS//,/ }; do bash "$HERE/02_run_array_task.sh" "$id" "$MANIFEST"; done
fi
