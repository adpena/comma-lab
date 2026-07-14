#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/mixed_int64_fixedpoint_scorer_forward_n600.json"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/mixed_int64_fixedpoint_scorer_forward_n600.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/probe_mixed_int64_scorer_forward_n600.py \
  --pair-start 0 \
  --pair-count 600 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --qdq-precursor experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json \
  --uniform-predecessor experiments/results/throughput_authority_ladder_20260714/exact_int64_fixedpoint_scorer_forward_n600.json \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1

