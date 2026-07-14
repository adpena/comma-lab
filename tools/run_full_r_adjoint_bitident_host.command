#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/full_r_adjoint_n600.json"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/full_r_adjoint_n600.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/probe_pythagorean_exact_arithmetic_bitident.py \
  --scope full-r-n600 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pair-start 0 \
  --pair-count 600 \
  --n 10 \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
