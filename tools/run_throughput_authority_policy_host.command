#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/throughput_authority_policy.json"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/compile_throughput_authority_policy.py \
  --qdq experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json \
  --integer-scorer experiments/results/throughput_authority_ladder_20260714/weight_l1_class_pair_tie_snap_scorer_forward_n600.json \
  --metal experiments/results/throughput_authority_ladder_20260714/metal_weight_l1_class_pair_tie_snap_w27_w31_exact_int64_segnet_n600.json \
  --integer-r experiments/results/throughput_authority_ladder_20260714/integer_r_backend_n600.json \
  --no-pose-gate \
  --pose-canary-every 8 \
  --unselected-r1-advisory-dpose 0.001610 \
  --require-receipts \
  --output "$OUT"
