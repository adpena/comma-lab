#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_int64_ceiling_n600.json"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_int64_ceiling_n600.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

# W26 is the last uniform precision whose real-SegNet maximum-fan-in bound
# fits a single exact signed-int64 accumulator. This is a finite ceiling test.
exec .venv/bin/python tools/probe_fixedpoint_scorer_forward_n600.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pair-start 0 \
  --pair-count 600 \
  --bits 25,26 \
  --activation-scale-mode dynamic_exact_absmax \
  --no-include-pose \
  --checkpoint-every 1 \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
