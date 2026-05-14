#!/bin/bash
# Operator-authorize wrapper for the C6 MDL-IBPS substrate.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml``
# Campaign ledger: ``.omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md``
# Floor v3: ``.omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md``
#   (zen-Z1 LARGEST single bet)
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# A smoke fires FIRST via the canonical helper. The recipe declares
# min_smoke_gpu=A10G after the 2026-05-14 100ep T4 timeout, so stale T4 CLI
# defaults are upgraded before any dispatch. Smoke validates rc=0 + auth-eval
# JSON present + score in plausible band [0.10, 0.30], and only proceeds to
# the full dispatch on smoke-green.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full
# dispatch ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored by the smoke wrapper:
#   MODAL_GPU=T4|A10G|A100|H100              (default T4 per recipe)
#   C6_E4_MDL_IBPS_EPOCHS=200                (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0                  (Modal hard-kill wall-clock)
#   C6_E4_MDL_IBPS_SMOKE_EPOCHS=100          (smoke epoch override)
#   C6_E4_MDL_IBPS_SMOKE_GPU=T4              (helper upgrades to A10G per recipe min)
#   C6_E4_MDL_IBPS_SMOKE_ONLY=1              (skip full even on smoke-green)
#   C6_E4_MDL_IBPS_FULL_ONLY=1               (skip smoke; operator override
#                                             after >=3 successful anchors)
#
# Lane: lane_c6_e4_mdl_ibps_substrate_20260514
# Cross-ref:
#   .omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md
#   .omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: empty array expansion guarded under set -u (macOS bash 3.2).
SMOKE_ARGS=()
if [ "${C6_E4_MDL_IBPS_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${C6_E4_MDL_IBPS_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_c6_e4_mdl_ibps_modal_t4_dispatch \
    --smoke-epochs "${C6_E4_MDL_IBPS_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${C6_E4_MDL_IBPS_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${C6_E4_MDL_IBPS_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_c6_e4_mdl_ibps_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
