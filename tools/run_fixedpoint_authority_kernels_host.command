#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/metal_weight_l1_class_pair_tie_snap_w27_w31_exact_int64_segnet_n600.json"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/metal_weight_l1_class_pair_tie_snap_w27_w31_exact_int64_segnet_n600.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/bench_fixedpoint_authority_kernels.py \
  --pair-start 0 \
  --pair-count 600 \
  --n-processes 10 \
  --bits 26 \
  --weight-l1-safe \
  --calibration-receipt experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json \
  --integer-precursor-receipt experiments/results/throughput_authority_ladder_20260714/weight_l1_class_pair_tie_snap_scorer_forward_n600.json \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
