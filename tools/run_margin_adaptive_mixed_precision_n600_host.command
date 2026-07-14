#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/margin_adaptive_mixed_precision_20260714/margin_adaptive_mixed_precision_n600.json"
LOG="$ROOT/experiments/results/margin_adaptive_mixed_precision_20260714/margin_adaptive_mixed_precision_n600.host.log"

cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/probe_margin_adaptive_mixed_precision_n600.py \
  --pair-start 0 \
  --pair-stop 600 \
  --profile-caps 8,10,12,14,16,18,20,22,24,26,27,28,29,30,31 \
  --n-processes 10 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --uniform-nogo-receipt experiments/results/throughput_authority_ladder_20260714/fixedpoint_scorer_forward_n600_fresh_89b970ff60.json \
  --calibration-receipt experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json \
  --integer-precursor-receipt experiments/results/throughput_authority_ladder_20260714/weight_l1_class_pair_tie_snap_scorer_forward_n600.json \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
