#!/usr/bin/env bash
# Submits ONLY the main-phase tasks (task_id 1-2 in the manifest: pair1,
# pair2 for phase=main). Never submits holdout here (Section 17: main and
# holdout must never be submitted together).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F0="$(dirname "$HERE")"
MANIFEST="$F0/06_main_run_manifest.csv"

MAIN_IDS=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i} NR>1 && $h["phase"]=="main"{print $h["task_id"]}' "$MANIFEST" | paste -sd, -)

if command -v sbatch >/dev/null 2>&1; then
  sbatch --array="$MAIN_IDS" \
    --cpus-per-task=4 --mem=4G --time=02:00:00 \
    --job-name=f0_main \
    -o "$F0/hpc/logs/main_%a.out" \
    "$HERE/02_run_array_task.sh" '$SLURM_ARRAY_TASK_ID' "$MANIFEST"
else
  echo "no sbatch on this host -- copy this repo to a Slurm login node and run:"
  echo "  sbatch --array=$MAIN_IDS --cpus-per-task=4 --mem=4G --time=02:00:00 \\"
  echo "    --job-name=f0_main -o $F0/hpc/logs/main_%a.out \\"
  echo "    $HERE/02_run_array_task.sh '\$SLURM_ARRAY_TASK_ID' $MANIFEST"
  echo
  echo "Local fallback (used for this F0 run, see 03_preregistered_protocol.md):"
  for id in ${MAIN_IDS//,/ }; do
    echo "  bash $HERE/02_run_array_task.sh $id $MANIFEST"
  done
fi
