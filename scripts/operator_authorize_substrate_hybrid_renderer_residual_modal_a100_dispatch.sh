#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_hybrid_renderer_residual_modal_a100_dispatch.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100   (default A100 per operator directive 2026-05-12)
#   HYBRID_RES_EPOCHS=2000        (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.5       (Modal hard-kill wall-clock)
#   HYBRID_RES_FREEZE_ALPHA=1     (set to freeze α; default trains the full
#                                  composite end-to-end)
#   HYBRID_RES_ALPHA_PRETRAINED_CHECKPOINT=<path>  (warm-start from α anchor)
#
# COMPOSITION-RISK PRE-PROMOTION DISCIPLINE (NON-NEGOTIABLE):
#   This recipe carries `pre_promotion_blockers` declaring that BOTH α
#   (lane_substrate_sane_hnerv_20260512) AND β
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

exec .venv/bin/python tools/operator_authorize.py \
    --recipe substrate_hybrid_renderer_residual_modal_a100_dispatch \
    --agent "claude:operator_authorize_substrate_hybrid_renderer_residual_modal_a100_dispatch" \
    "$@"
