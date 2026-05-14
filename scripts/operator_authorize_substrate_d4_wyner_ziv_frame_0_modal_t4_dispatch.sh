#!/bin/bash
# Operator-authorize wrapper for the D4 Wyner-Ziv frame-0 substrate.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml``
# Deep-math memo D4 entry: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §6 D4
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# A 100-epoch ~$0.30 Modal T4 smoke fires FIRST, validates rc=0 + auth-eval
# JSON present + score in plausible band [0.05, 0.50], and only proceeds to
# the full 2000-epoch T4 dispatch on smoke-green.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full
# dispatch ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored by the smoke wrapper:
#   MODAL_GPU=T4|A10G|A100|H100              (default T4 per recipe)
#   D4_WYNER_ZIV_FRAME_0_EPOCHS=2000         (council default; full training)
#   MODAL_TIMEOUT_HOURS=3.0                  (Modal hard-kill wall-clock)
#   D4_WYNER_ZIV_FRAME_0_SMOKE_EPOCHS=100    (smoke epoch override)
#   D4_WYNER_ZIV_FRAME_0_SMOKE_GPU=T4        (smoke GPU class)
#   D4_WYNER_ZIV_FRAME_0_SMOKE_ONLY=1        (skip full even on smoke-green)
#   D4_WYNER_ZIV_FRAME_0_FULL_ONLY=1         (skip smoke; operator override
#                                             after >=3 successful anchors)
#
# Lane: lane_d4_wyner_ziv_frame_0_substrate_20260514
# Cross-ref:
#   .omx/research/deep_math_geometry_manifolds_synthesis_20260514.md §3.5 + §6 D4

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: empty array expansion guarded under set -u (macOS bash 3.2).
SMOKE_ARGS=()
if [ "${D4_WYNER_ZIV_FRAME_0_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${D4_WYNER_ZIV_FRAME_0_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch \
    --smoke-epochs "${D4_WYNER_ZIV_FRAME_0_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${D4_WYNER_ZIV_FRAME_0_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${D4_WYNER_ZIV_FRAME_0_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
