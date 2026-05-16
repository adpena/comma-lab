#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: .omx/operator_authorize_recipes/substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch.yaml
#
# Per Catalog #167 this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary.
# Per Catalog #189 we use ${ARR[@]+"${ARR[@]}"} for empty-array safety.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100   (default T4 — Carmack-Hotz needs only one forward pass)
#   NSCS06_EPOCHS=1               (no training loop; epochs is just dispatch sentinel)
#   MODAL_TIMEOUT_HOURS=1.0       (full Modal hard-kill wall-clock)
#   NSCS06_SMOKE_EPOCHS=1         (smoke epoch override)
#   NSCS06_SMOKE_GPU=T4           (smoke GPU class)
#
# Lane: lane_nscs06_carmack_hotz_strip_everything_20260515
# Cross-ref: feedback_grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515.md#composite-4

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${NSCS06_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${NSCS06_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch \
    --smoke-epochs "${NSCS06_SMOKE_EPOCHS:-1}" \
    --smoke-gpu "${NSCS06_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${NSCS06_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:operator_authorize_substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
