#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/phase1_t1_balle_cheap_config_dispatch.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Env-var overrides honored by the canonical entry point:
#   PHASE1_PLATFORM=modal|vastai (default modal)
#   MODAL_GPU=T4|A10G|A100|H100  (default T4)
#   MODAL_TIMEOUT_HOURS=4.0      (default 4.0)
#   VASTAI_TIMEOUT_HOURS=4.0     (default 4.0)
#
# The PHASE1_PLATFORM env-var is consumed via the recipe's `platform` field
# (modal vs vastai). Cost-band predictions use the same recipe but the
# canonical entry point's _dispatch_modal / _dispatch_vastai branches.
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Translate the legacy PHASE1_PLATFORM env-var into the recipe's platform field
# by passing through to the canonical entry point. The recipe declares
# platform=modal as default; vastai override requires a separate Vast.ai
# recipe (kept in YAML for future, but for now the legacy shim's vastai path
# stays available via direct invocation of the canonical Vast.ai launcher).
exec .venv/bin/python tools/operator_authorize.py \
    --recipe phase1_t1_balle_cheap_config_dispatch \
    --agent "claude:operator_authorize_phase1_t1_balle_cheap_config_dispatch" \
    "$@"
