#!/usr/bin/env bash
# Section 6/23: commit main results + frozen claim wording BEFORE holdout
# is ever run. Refuses to proceed if the holdout has already been run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; F0="$(dirname "$HERE")"
if [ -f "$F0/runs/holdout_pair1/_run_metadata.json" ] || [ -f "$F0/runs/holdout_pair2/_run_metadata.json" ]; then
  echo "REFUSING: holdout output already exists -- pre-holdout freeze must happen first"
  exit 1
fi
[ -f "$F0/04_pre_holdout_claim_freeze.md" ] || { echo "missing 04_pre_holdout_claim_freeze.md -- write it before running this"; exit 1; }
cd "$F0/../.."
git add -A B1/F0_final_hpc_validation
git commit -m "Freeze B1 final claims before holdout evaluation"
echo "pre-holdout commit created: $(git rev-parse HEAD)"
