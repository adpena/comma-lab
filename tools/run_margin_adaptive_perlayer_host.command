#!/bin/zsh
# SPDX-License-Identifier: MIT
# MAIN/operator Metal handoff.  MODE=plan-only stays local/MLX-free; MODE=metal
# is the controlled native-width receipt path and may fail closed without Metal.
set -euo pipefail

ROOT=${0:A:h:h}
MODE=${MODE:-metal}

if [[ "$MODE" != "plan-only" && "$MODE" != "metal" ]]; then
  print -u2 "usage: MODE=plan-only|metal $0"
  exit 64
fi

if [[ "$MODE" == "plan-only" ]]; then
  DEFAULT_OUT="$ROOT/experiments/results/margin_adaptive_perlayer_followon_20260714/plan_only.json"
else
  DEFAULT_OUT="$ROOT/experiments/results/margin_adaptive_perlayer_followon_20260714/metal_n600.json"
fi
OUT=${OUT:-"$DEFAULT_OUT"}
LOG=${LOG:-"$ROOT/experiments/results/margin_adaptive_perlayer_followon_20260714/margin_adaptive_perlayer_${MODE}.host.log"}

cd "$ROOT"
mkdir -p "${OUT:h}"

if [[ "$MODE" == "plan-only" ]]; then
  exec .venv/bin/python tools/probe_margin_adaptive_perlayer_n600.py \
    --execution-mode plan-only \
    --pair-start 0 \
    --pair-stop 600 \
    --design-stop 264 \
    --logical-widths 8,16,28 \
    --channel-prefix-fractions 0.125,0.25,0.5,0.75,1.0 \
    --n-processes 10 \
    --checkpoint-every-pairs 1 \
    --se-control full-fp \
    --predecessor-receipt experiments/results/margin_adaptive_mixed_precision_20260714/margin_adaptive_mixed_precision_n600.json \
    --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
    --segnet-weights upstream/models/segnet.safetensors \
    --resume \
    --output "$OUT" \
    >>"$LOG" 2>&1
fi

exec .venv/bin/python tools/probe_margin_adaptive_perlayer_metal_n600.py \
  --plan experiments/results/margin_adaptive_perlayer_followon_20260714/plan_only.json \
  --resume \
  --output "$OUT" \
  >>"$LOG" 2>&1
