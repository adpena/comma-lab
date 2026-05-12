#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_sane_hnerv_modal_a100_dispatch.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100  (default A100 per operator directive 2026-05-12)
#   SANE_HNERV_EPOCHS=2000       (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0      (Modal hard-kill wall-clock)
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec .venv/bin/python tools/operator_authorize.py \
    --recipe substrate_sane_hnerv_modal_a100_dispatch \
    --agent "claude:operator_authorize_substrate_sane_hnerv_modal_a100_dispatch" \
    "$@"
