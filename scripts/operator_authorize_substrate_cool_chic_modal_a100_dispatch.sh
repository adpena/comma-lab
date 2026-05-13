#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_cool_chic_modal_a100_dispatch.yaml``
#
# Per Catalog #167, this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100  (default A100 per operator directive 2026-05-12)
#   COOL_CHIC_EPOCHS=2000        (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0      (full Modal hard-kill wall-clock)
#   COOL_CHIC_SMOKE_EPOCHS=100   (smoke epoch override)
#   COOL_CHIC_SMOKE_GPU=T4       (smoke GPU class)
#
# Lane: lane_substrate_cool_chic_20260512
# Cross-ref: feedback_phase_b2_build_3_high_target_trainers_LANDED_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${COOL_CHIC_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${COOL_CHIC_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_cool_chic_modal_a100_dispatch \
    --smoke-epochs "${COOL_CHIC_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${COOL_CHIC_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${COOL_CHIC_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_cool_chic_modal_a100_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
