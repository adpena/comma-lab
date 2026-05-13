#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml``
#
# Per Catalog #167, this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100  (default A100 per operator directive 2026-05-12)
#   GRAYSCALE_LUT_EPOCHS=2000           (council default; full training)
#   GRAYSCALE_LUT_BATCH_SIZE=16         (council default for grayscale_lut substrate)
#   MODAL_TIMEOUT_HOURS=4.0             (full Modal hard-kill wall-clock)
#   GRAYSCALE_LUT_SMOKE_EPOCHS=100      (smoke epoch override)
#   GRAYSCALE_LUT_SMOKE_GPU=T4          (smoke GPU class)
#
# Lane: lane_wave4_grayscale_lut_trainer_build_20260512
# Cross-ref: feedback_wave4_grayscale_lut_trainer_build_LANDED_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${GRAYSCALE_LUT_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${GRAYSCALE_LUT_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_grayscale_lut_modal_a100_dispatch \
    --smoke-epochs "${GRAYSCALE_LUT_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${GRAYSCALE_LUT_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${GRAYSCALE_LUT_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_grayscale_lut_modal_a100_dispatch" \
    "${SMOKE_ARGS[@]}" \
    "$@"
