#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/fixedpoint_scorer_forward_n600_v2.json"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/fixedpoint_scorer_forward_n600_v2.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/probe_fixedpoint_scorer_forward_n600.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pair-start 0 \
  --pair-count 600 \
  --bits 8,10,12,14,16,18,20,22,24 \
  --include-pose \
  --checkpoint-every 1 \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
