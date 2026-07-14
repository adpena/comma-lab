#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
OUT="$ROOT/experiments/results/throughput_authority_ladder_20260714/pose_verdict_gate_n96_dry_start"
LOG="$ROOT/experiments/results/throughput_authority_ladder_20260714/pose_verdict_gate_n96_dry_start.host.log"
cd "$ROOT"
mkdir -p "${OUT:h}"

# MAIN/operator execution only: governed, bounded, resumable n96 dry-start. The typed probe
# cadence exercises live -> banked -> live PoseNet verdicts and the resume round-trip.
exec .venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 96 \
  --config crucible_v752 \
  --dsl-lever PoseVerdictGateDryStart \
  --out-dir "$OUT" \
  --purpose "task-494 n96 pose-verdict gate drift canary and resume dry-start" \
  --dry-start 3 \
  --dry-start-boot-budget-s 600 \
  --dry-start-per-ep-budget-s 240 \
  --no-dashboard \
  >>"$LOG" 2>&1
