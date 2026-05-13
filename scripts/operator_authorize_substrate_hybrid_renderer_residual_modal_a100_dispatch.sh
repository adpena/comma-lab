#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_hybrid_renderer_residual_modal_a100_dispatch.yaml``
#
# Per Catalog #167, this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary. The recipe still
# fails closed while its pre_promotion_blockers remain declared.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100   (default A100 per operator directive 2026-05-12)
#   HYBRID_RES_EPOCHS=2000        (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.5       (full Modal hard-kill wall-clock)
#   HYBRID_RES_SMOKE_EPOCHS=100   (smoke epoch override)
#   HYBRID_RES_SMOKE_GPU=T4       (smoke GPU class)
#   HYBRID_RES_FREEZE_ALPHA=1     (set to freeze alpha; default trains the full
#                                  composite end-to-end)
#   HYBRID_RES_ALPHA_PRETRAINED_CHECKPOINT=<path>  (warm-start from alpha anchor)
#
# COMPOSITION-RISK PRE-PROMOTION DISCIPLINE (NON-NEGOTIABLE):
#   This recipe carries `pre_promotion_blockers` declaring that BOTH alpha
#   (lane_substrate_sane_hnerv_20260512) AND beta
#   (lane_substrate_balle_renderer_20260512) MUST have a verified
#   [contest-CUDA] anchor BEFORE this dispatch fires. The canonical
#   tools/operator_authorize.py reads the recipe's pre_promotion_blockers
#   field and refuses dispatch if any listed blocker is unmet. That gate is
#   the structural enforcement of CLAUDE.md "Substrate vs codec composition
#   meta-pattern" (composition without verified single-axis substrate first
#   is the kitchen_sink anti-pattern).
#
#   To smoke-test the trainer locally (no auth eval) prior to clearance, run
#   the trainer directly with --smoke (CPU; no scorer load); do NOT run this
#   wrapper. The wrapper is the dispatch gate.
#
# Lane: lane_wave1_hybrid_renderer_residual_trainer_build_20260512
# Cross-ref: feedback_wave1_hybrid_renderer_residual_trainer_build_LANDED_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${HYBRID_RES_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${HYBRID_RES_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_hybrid_renderer_residual_modal_a100_dispatch \
    --smoke-epochs "${HYBRID_RES_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${HYBRID_RES_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${HYBRID_RES_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_hybrid_renderer_residual_modal_a100_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
