#!/usr/bin/env bash
# F0 environment audit (Section 7). Run once before any submission.
set -euo pipefail
echo "=== hostname ==="; hostname
echo "=== slurm ==="
for c in sbatch squeue sacct; do command -v "$c" >/dev/null 2>&1 && echo "$c: $(command -v "$c")" || echo "$c: NOT FOUND"; done
echo "=== python ==="; command -v python3; python3 --version
echo "=== cpu ==="; lscpu | head -30
echo "=== mem ==="; free -h
echo "=== disk ==="; df -h .
