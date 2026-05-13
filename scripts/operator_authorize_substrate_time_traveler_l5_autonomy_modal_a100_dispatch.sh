#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml``
#
# Per Catalog #167, this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100  (default A100 per operator directive 2026-05-13)
#   TT5L_EPOCHS=3000             (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0      (full Modal hard-kill wall-clock)
#   TT5L_SMOKE_EPOCHS=100        (smoke epoch override)
#   TT5L_SMOKE_GPU=T4            (smoke GPU class)
#
# Lane: lane_time_traveler_l5_autonomy_substrate_20260513
# Cross-ref: .omx/research/time_traveler_architecture_reverse_engineered_20260513.md
#
# Per Catalog #189 the empty-array expansion uses the
# ${ARR[@]+"${ARR[@]}"} pattern so macOS bash 3.2 under `set -u` does not
# trip "unbound variable" on dry-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${TT5L_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${TT5L_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_time_traveler_l5_autonomy_modal_a100_dispatch \
    --smoke-epochs "${TT5L_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${TT5L_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${TT5L_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_time_traveler_l5_autonomy_modal_a100_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
