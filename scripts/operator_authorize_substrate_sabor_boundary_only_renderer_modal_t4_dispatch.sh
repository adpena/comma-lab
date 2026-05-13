#!/bin/bash
# Operator-authorize wrapper for the SABOR boundary-only renderer substrate.
#
# Recipe:  .omx/operator_authorize_recipes/substrate_sabor_boundary_only_renderer_modal_t4_dispatch.yaml
# Council: Council F O1 first-principles (Shannon LEAD); φ1 capacity audit
#          .omx/research/sabor_boundary_audit_20260513.md
# Lane:    lane_sabor_boundary_only_renderer_substrate_20260513
#
# Smoke-before-full chain (Catalog #167):
#   1. Modal T4 SMOKE ($0.20): 100-epoch tiny dispatch with auth eval gate.
#      Score must land in plausible band [0.10, 0.30] or chain refuses full.
#   2. Modal T4 FULL ($1.50): 2000-epoch full training + auth eval on the
#      trained SBO1 archive.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full dispatch
# ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored:
#   MODAL_GPU=T4|A10G|A100|H100              (default T4 per recipe)
#   SABOR_EPOCHS=2000                        (council default; full training)
#   SABOR_SMOKE_EPOCHS=100                   (smoke epoch override)
#   SABOR_SMOKE_GPU=T4                       (smoke GPU class)
#   SABOR_SMOKE_ONLY=1                       (skip full even on smoke-green)
#   SABOR_FULL_ONLY=1                        (skip smoke; ≥3 successful anchors)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke-before-full chain (Catalog #167) — empty array guarded under set -u
# per Catalog #189.
SMOKE_ARGS=()
if [ "${SABOR_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${SABOR_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_sabor_boundary_only_renderer_modal_t4_dispatch \
    --smoke-epochs "${SABOR_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${SABOR_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${SABOR_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:operator_authorize_substrate_sabor_boundary_only_renderer_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
