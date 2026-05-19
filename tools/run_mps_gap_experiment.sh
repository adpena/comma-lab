#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Local-side harness for the Phase B MPS-train CUDA-score gap diagnostic.
#
# Flow:
#   1. Train tiny renderer on local MPS for 100 epochs (~5-10min wallclock).
#   2. Verify local checkpoint + frame cache exist.
#   3. Fire Modal A10G dispatch via tools/operator_authorize.py (REQUIRES
#      paired-env operator approval per Catalog #199).
#   4. Harvest the Modal target-device components.
#   5. Compute the local gap manifest via tac.mps_gap_experiment.harvest_and_verdict.
#
# Pre-condition: the recipe at
# .omx/operator_authorize_recipes/mps_gap_experiment_tiny_renderer_modal_a10g_dispatch.yaml
# must have `dispatch_enabled: true` (default is false per the build-then-await pattern).
# Operator flips that BEFORE running this harness.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-experiments/results/mps_gap_experiment_local}
EPOCHS=${MPS_GAP_EPOCHS:-100}
NUM_PAIRS=${MPS_GAP_NUM_PAIRS:-10}

echo "[mps_gap_experiment] Phase 1: MPS local training (${EPOCHS} epochs, ${NUM_PAIRS} pairs)"
echo "[mps_gap_experiment] Output: $LOCAL_OUTPUT_DIR"

.venv/bin/python -m tac.mps_gap_experiment.train_on_mps_cli \
    --output-dir "$LOCAL_OUTPUT_DIR" \
    --upstream-dir upstream \
    --epochs "$EPOCHS" \
    --num-pairs "$NUM_PAIRS" \
    --device mps

if [ ! -f "$LOCAL_OUTPUT_DIR/checkpoint_ema.pt" ]; then
    echo "[FATAL] MPS training did not produce checkpoint_ema.pt; aborting before dispatch"
    exit 2
fi
if [ ! -f "$LOCAL_OUTPUT_DIR/frame_cache.pt" ]; then
    echo "[FATAL] MPS training did not produce frame_cache.pt; aborting before dispatch"
    exit 2
fi

echo "[mps_gap_experiment] Phase 2: local checkpoint produced — local-MPS reference baseline ready"

# Catalog #199 paired-env discipline: operator MUST set BOTH env vars
# explicitly OR operator_authorize.py refuses with SystemExit(11).
if [ -z "${OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE:-}" ]; then
    echo ""
    echo "==================================================================="
    echo "OPERATOR ACTION REQUIRED before dispatch fires:"
    echo ""
    echo "  Step A: flip .omx/operator_authorize_recipes/mps_gap_experiment_tiny_renderer_modal_a10g_dispatch.yaml"
    echo "          'dispatch_enabled: false' -> 'dispatch_enabled: true'"
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
echo "[mps_gap_experiment] Phase 5: local gap manifest via"
echo "    .venv/bin/python -m tac.mps_gap_experiment.harvest_and_verdict_cli ..."
echo "[mps_gap_experiment] (Phase 4/5 helpers are out of scope for the build-then-await infrastructure landing;"
echo "    operator runs them after harvest based on the canonical Modal call_id ledger.)"
