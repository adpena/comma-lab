#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/integer_r_backend_n600.json"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/integer_r_backend_n600.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/bench_integer_r_adjoint_backend.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --full-r-receipt experiments/results/throughput_authority_ladder_20260714/full_r_adjoint_n600.json \
  --pair-start 0 \
  --pair-count 600 \
  --warmup-frames 4 \
  --determinism-repeats 10 \
  --output "$OUT" \
  >>"$LOG" 2>&1
