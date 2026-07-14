#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/throughput_authority_policy.json"
cd "$ROOT"
mkdir -p "${OUT:h}"

exec .venv/bin/python tools/compile_throughput_authority_policy.py \
  --qdq experiments/results/throughput_authority_ladder_20260714/dynamic_fixedpoint_scorer_forward_n600.json \
  --metal experiments/results/throughput_authority_ladder_20260714/metal_dynamic_fixedpoint_segnet_n600.json \
  --integer-r experiments/results/throughput_authority_ladder_20260714/integer_r_backend_n600.json \
  --pose-gate \
  --pose-canary-every 8 \
  --banked-r1-dpose 0.001610 \
  --require-receipts \
  --output "$OUT"
