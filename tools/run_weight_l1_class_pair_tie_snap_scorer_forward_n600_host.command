#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/weight_l1_class_pair_tie_snap_scorer_forward_n600.json"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/weight_l1_class_pair_tie_snap_scorer_forward_n600.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/probe_weight_l1_class_pair_tie_snap_scorer_forward_n600.py \
  --pair-start 0 \
  --pair-count 600 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --weight-predecessor experiments/results/throughput_authority_ladder_20260714/weight_l1_int64_fixedpoint_scorer_forward_n600.json \
  --design-receipt experiments/results/throughput_authority_ladder_20260714/weight_l1_tie_conflict_diagnostic_design_0_263.json \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
