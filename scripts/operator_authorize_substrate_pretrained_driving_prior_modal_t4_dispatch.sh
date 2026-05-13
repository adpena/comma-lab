#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml``
#
# Per Catalog #167 + #176, this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary. Scaffold L0:
# the full path is intentionally a NotImplementedError until Phase 2 council
# review approves the real training design.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100  (default T4 per operator directive 2026-05-13)
#   DPP_EPOCHS=2000              (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0      (full Modal hard-kill wall-clock)
#   DPP_SMOKE_EPOCHS=100         (smoke epoch override)
#   DPP_SMOKE_GPU=T4             (smoke GPU class)
#
# Lane: lane_pretrained_driving_prior_lane_scaffold_20260513
# Cross-ref: .omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${DPP_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${DPP_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_pretrained_driving_prior_modal_t4_dispatch \
    --smoke-epochs "${DPP_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${DPP_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${DPP_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_pretrained_driving_prior_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
