#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/metal_dynamic_fixedpoint_segnet_n600.json"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/metal_dynamic_fixedpoint_segnet_n600.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/bench_fixedpoint_authority_kernels.py \
  --pair-start 0 \
  --pair-count 600 \
  --n-processes 10 \
  --calibration-receipt experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_n600.json \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
