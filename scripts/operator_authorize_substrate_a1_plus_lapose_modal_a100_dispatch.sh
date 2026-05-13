#!/bin/bash
# Operator-authorize wrapper for the A1 + LAPose D1.D HIERARCHICAL composition substrate.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_a1_plus_lapose_modal_a100_dispatch.yaml``
# Council: ``.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md``
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# A 100-epoch ~$0.30 Modal T4 smoke fires FIRST, validates rc=0 + auth-eval
# JSON present + score in plausible band [0.15, 0.25], and only proceeds to
# the full 3000-epoch A100 dispatch on smoke-green.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full
# dispatch ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored by the smoke wrapper:
#   MODAL_GPU=T4|A10G|A100|H100              (default A100 per recipe)
#   A1_PLUS_LAPOSE_EPOCHS=3000               (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0                  (Modal hard-kill wall-clock)
#   A1_PLUS_LAPOSE_SMOKE_EPOCHS=100          (smoke epoch override)
#   A1_PLUS_LAPOSE_SMOKE_GPU=T4              (smoke GPU class)
#   A1_PLUS_LAPOSE_SMOKE_ONLY=1              (skip full even on smoke-green)
#   A1_PLUS_LAPOSE_FULL_ONLY=1               (skip smoke; operator override
#                                             after >=3 successful anchors)
#
# Lane: lane_a1_plus_lapose_composition_20260513
# Cross-ref: feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: empty array expansion guarded under set -u.
SMOKE_ARGS=()
if [ "${A1_PLUS_LAPOSE_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${A1_PLUS_LAPOSE_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_a1_plus_lapose_modal_a100_dispatch \
    --smoke-epochs "${A1_PLUS_LAPOSE_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${A1_PLUS_LAPOSE_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${A1_PLUS_LAPOSE_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_a1_plus_lapose_modal_a100_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
