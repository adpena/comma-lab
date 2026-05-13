#!/bin/bash
# Operator-authorize wrapper for the S2SBS stride-2-stem byte-stuffing substrate.
#
# Recipe:  .omx/operator_authorize_recipes/substrate_s2sbs_byte_stuffing_modal_t4_dispatch.yaml
# Audit:   .omx/research/s2sbs_blindspot_audit_20260513.md
# Lane:    lane_s2sbs_stride2_byte_stuffing_substrate_20260513
#
# Council F O3 PAIR T+OPT3. Per CLAUDE.md "Operator gates must be wired
# and used" + Catalog #162 (canonical use) + Catalog #167 (smoke-before-
# full) + Catalog #189 (empty array guard).
#
# Env-var overrides honored:
#   MODAL_GPU=T4|A10G|A100|H100  (default T4 per recipe)
#   S2SBS_EPOCHS=2000            (full training epochs)
#   S2SBS_SMOKE_EPOCHS=100       (smoke epoch override)
#   S2SBS_SMOKE_GPU=T4           (smoke GPU class)
#   S2SBS_SMOKE_ONLY=1           (skip full even on smoke-green)
#   S2SBS_FULL_ONLY=1            (skip smoke; >=3 successful anchors)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Stage 1: Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: empty array expansion guarded under set -u.
SMOKE_ARGS=()
if [ "${S2SBS_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${S2SBS_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_s2sbs_byte_stuffing_modal_t4_dispatch \
    --smoke-epochs "${S2SBS_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${S2SBS_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${S2SBS_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:operator_authorize_substrate_s2sbs_byte_stuffing_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
