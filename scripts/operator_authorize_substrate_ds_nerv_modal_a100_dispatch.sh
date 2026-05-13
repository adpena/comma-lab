#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_ds_nerv_modal_a100_dispatch.yaml``
#
# Per Catalog #167, this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100  (default A100 per operator directive 2026-05-12)
#   DS_NERV_EPOCHS=2000          (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0      (full Modal hard-kill wall-clock)
#   DS_NERV_SMOKE_EPOCHS=100     (smoke epoch override)
#   DS_NERV_SMOKE_GPU=T4         (smoke GPU class)
#
# Lane: lane_substrate_ds_nerv_20260512
# Cross-ref: feedback_wave3_hnerv_c_2_trainers_LANDED_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${DS_NERV_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${DS_NERV_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_ds_nerv_modal_a100_dispatch \
    --smoke-epochs "${DS_NERV_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${DS_NERV_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${DS_NERV_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_ds_nerv_modal_a100_dispatch" \
    "${SMOKE_ARGS[@]}" \
    "$@"
