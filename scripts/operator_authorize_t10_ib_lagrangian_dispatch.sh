#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/t10_ib_lagrangian_dispatch.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``: every operator-
# authorize .sh wrapper MUST be a thin shim delegating to
# ``tools/operator_authorize.py --recipe <name>``.
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec .venv/bin/python tools/operator_authorize.py \
    --recipe t10_ib_lagrangian_dispatch \
    --agent "claude:operator_authorize_t10_ib_lagrangian_dispatch" \
    "$@"
