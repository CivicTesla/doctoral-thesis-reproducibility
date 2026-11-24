#!/usr/bin/env bash
# Resubmits only manifest rows whose expected_output is missing. Safe to
# rerun repeatedly (02_run_array_task.sh skips already-complete outputs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; F0="$(dirname "$HERE")"
MANIFEST="${1:-$F0/06_main_run_manifest.csv}"
FAILED=""
while IFS=, read -r task_id phase pair mechanisms actions seed_start seed_end n_seeds driver config_hash expected; do
  [ "$task_id" = "task_id" ] && continue
  [ -f "$F0/$expected" ] || FAILED="$FAILED,$task_id"
done < "$MANIFEST"
FAILED="${FAILED#,}"
[ -z "$FAILED" ] && { echo "nothing to resubmit"; exit 0; }
echo "resubmitting task_ids: $FAILED"
if command -v sbatch >/dev/null 2>&1; then
  sbatch --array="$FAILED" --cpus-per-task=4 --mem=4G --time=02:00:00 \
    --job-name=f0_resubmit -o "$F0/hpc/logs/resubmit_%a.out" \
    "$HERE/02_run_array_task.sh" '$SLURM_ARRAY_TASK_ID' "$MANIFEST"
else
  for id in ${FAILED//,/ }; do bash "$HERE/02_run_array_task.sh" "$id" "$MANIFEST"; done
fi
