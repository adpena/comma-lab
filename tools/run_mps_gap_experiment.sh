#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Local-side harness for the Phase B MPS-train CUDA-score gap diagnostic.
#
# SPLIT-DEVICE ARCHITECTURE (per predecessor verdict
# `mps_phase_b_gap_experiment_verdict_20260519T053530Z` Option A reactivation):
# the gap measurement requires a LOCAL MPS forward (captured on Apple Silicon)
# and a REMOTE Modal A10G CUDA forward (no MPS hardware exists on Modal), then
# a LOCAL diff. Both forwards MUST use the SAME EMA checkpoint + frame_cache.pt.
#
# Flow:
#   1. Train tiny renderer on local MPS for 100 epochs (~5-10min wallclock);
#      train_on_mps_real_frames also captures local_mps_components.json +
#      local_mps_forward_outputs.pt as a post-train step (split-device REFERENCE).
#   2. Verify local checkpoint + frame cache + local_mps_components.json exist.
#   3. Fire Modal A10G dispatch via tools/operator_authorize.py (REQUIRES
#      paired-env operator approval per Catalog #199).
#   4. Harvest the Modal target_cuda_components.json via tools/harvest_modal_calls.py.
#   5. Compute the local gap manifest via
#      tac.mps_gap_experiment.harvest_and_verdict_cli (diff + verdict).
#
# Pre-condition: the recipe at
# .omx/operator_authorize_recipes/mps_gap_experiment_tiny_renderer_modal_a10g_dispatch.yaml
# must have `dispatch_enabled: true` (predecessor's commit bf6a2ecea already flipped it).

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-experiments/results/mps_gap_experiment_local}
EPOCHS=${MPS_GAP_EPOCHS:-100}
NUM_PAIRS=${MPS_GAP_NUM_PAIRS:-10}
SKIP_TRAIN=${MPS_GAP_SKIP_TRAIN:-0}

if [ "$SKIP_TRAIN" = "1" ] && [ -f "$LOCAL_OUTPUT_DIR/checkpoint_ema.pt" ]; then
    echo "[mps_gap_experiment] Phase 1: SKIPPING re-training; reusing existing checkpoint at $LOCAL_OUTPUT_DIR"
else
    echo "[mps_gap_experiment] Phase 1: MPS local training (${EPOCHS} epochs, ${NUM_PAIRS} pairs)"
    echo "[mps_gap_experiment] Output: $LOCAL_OUTPUT_DIR"

    .venv/bin/python -m tac.mps_gap_experiment.train_on_mps_cli \
        --output-dir "$LOCAL_OUTPUT_DIR" \
        --upstream-dir upstream \
        --epochs "$EPOCHS" \
        --num-pairs "$NUM_PAIRS" \
        --device mps
fi

if [ ! -f "$LOCAL_OUTPUT_DIR/checkpoint_ema.pt" ]; then
    echo "[FATAL] MPS training did not produce checkpoint_ema.pt; aborting before dispatch"
    exit 2
fi
if [ ! -f "$LOCAL_OUTPUT_DIR/frame_cache.pt" ]; then
    echo "[FATAL] MPS training did not produce frame_cache.pt; aborting before dispatch"
    exit 2
fi

# Phase 1.5: split-device LOCAL MPS REFERENCE — captured automatically by
# train_on_mps_real_frames as a post-train step. If it failed silently OR the
# operator skipped training, recompute via the canonical CLI here so the
# REMOTE dispatch has a paired reference to diff against.
if [ ! -f "$LOCAL_OUTPUT_DIR/local_mps_components.json" ]; then
    echo "[mps_gap_experiment] Phase 1.5: re-capturing LOCAL MPS reference (training did not emit it)"
    .venv/bin/python -m tac.mps_gap_experiment.harvest_and_verdict_cli reference \
        --checkpoint "$LOCAL_OUTPUT_DIR/checkpoint_ema.pt" \
        --frame-cache "$LOCAL_OUTPUT_DIR/frame_cache.pt" \
        --output-dir "$LOCAL_OUTPUT_DIR" \
        --device mps \
        --upstream-dir upstream
fi

if [ ! -f "$LOCAL_OUTPUT_DIR/local_mps_components.json" ]; then
    echo "[FATAL] local_mps_components.json missing; cannot proceed to split-device diff"
    exit 3
fi

echo "[mps_gap_experiment] Phase 2: local checkpoint + LOCAL MPS reference ready for split-device diff"

# Catalog #199 paired-env discipline: operator MUST set BOTH env vars
# explicitly OR operator_authorize.py refuses with SystemExit(11).
if [ -z "${OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE:-}" ]; then
    echo ""
    echo "==================================================================="
    echo "OPERATOR ACTION REQUIRED before dispatch fires:"
    echo ""
    echo "  Step A: confirm .omx/operator_authorize_recipes/mps_gap_experiment_tiny_renderer_modal_a10g_dispatch.yaml"
    echo "          has dispatch_enabled: true (predecessor flipped per OPERATOR_FRONTIER_OVERRIDE 2026-05-19)"
    echo ""
    echo "  Step B: re-run with paired env vars set:"
    echo "    OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \\"
    echo "    OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.50 \\"
    echo "    bash tools/run_mps_gap_experiment.sh"
    echo ""
    echo "  (Catalog #199: both env vars are mandatory for non-interactive dispatch)"
    echo "==================================================================="
    exit 0
fi

if [ -z "${OPERATOR_AUTHORIZE_SESSION_BUDGET_USD:-}" ]; then
    echo "[FATAL] OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE set without paired"
    echo "        OPERATOR_AUTHORIZE_SESSION_BUDGET_USD; refusing per Catalog #199"
    exit 11
fi

echo "[mps_gap_experiment] Phase 3: firing Modal A10G dispatch (\$${OPERATOR_AUTHORIZE_SESSION_BUDGET_USD} envelope)"

.venv/bin/python tools/operator_authorize.py \
    --recipe mps_gap_experiment_tiny_renderer_modal_a10g_dispatch \
    --target modal

echo "[mps_gap_experiment] Phase 4: harvest happens via tools/harvest_modal_calls.py within 24h"
echo "[mps_gap_experiment] Phase 5: local split-device diff via"
echo "    .venv/bin/python -m tac.mps_gap_experiment.harvest_and_verdict_cli diff \\"
echo "        --local $LOCAL_OUTPUT_DIR/local_mps_components.json \\"
echo "        --target <modal-output-dir>/target_cuda_components.json \\"
echo "        --output $LOCAL_OUTPUT_DIR/gap_results.json"
