#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12),
# the 10 ``scripts/operator_authorize_*.sh`` wrappers (~1,497 LOC, ~70%
# structurally duplicate) collapse to one canonical entry point
# (``tools/operator_authorize.py``) + N YAML recipes.
#
# Recipe: ``.omx/operator_authorize_recipes/scpp_stage1_anchor_dispatch.yaml``
#
# Per Catalog #162 ``check_operator_authorize_canonical_use`` STRICT preflight
# gate: every operator-authorize .sh wrapper MUST be a thin shim delegating to
# ``tools/operator_authorize.py --recipe <name>``. Same-line waiver
# ``# OPERATOR_AUTHORIZE_LEGACY_OK:<reason>`` for explicit legacy preservation.
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec .venv/bin/python tools/operator_authorize.py \
    --recipe scpp_stage1_anchor_dispatch \
    --agent "claude:operator_authorize_scpp_stage1_anchor_dispatch" \
    "$@"
