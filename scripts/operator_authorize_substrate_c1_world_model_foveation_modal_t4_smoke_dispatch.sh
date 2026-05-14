#!/bin/bash
# Operator-authorize wrapper for the C1 world-model + foveation substrate (smoke).
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml``
# Campaign ledger: ``.omx/research/campaign_c1_world_model_foveation_20260514.md``
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# A 100-epoch ~$1 Modal T4 smoke fires FIRST, validates rc=0 + substrate
# plumbing end-to-end. The trainer's `_full_main` raises NotImplementedError
# pending Phase 3 council approval, so this wrapper is SMOKE-ONLY at L1.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the dispatch
# ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored by the smoke wrapper:
#   MODAL_GPU=T4|A10G|A100|H100                    (default T4 per recipe)
#   C1_WORLD_MODEL_FOVEATION_EPOCHS=100            (smoke default; 2000 for full once council unlocks)
#   MODAL_TIMEOUT_HOURS=1.0                        (Modal hard-kill wall-clock)
#   C1_WORLD_MODEL_FOVEATION_SMOKE_EPOCHS=100      (smoke epoch override)
#   C1_WORLD_MODEL_FOVEATION_SMOKE_GPU=T4          (smoke GPU class)
#   C1_WORLD_MODEL_FOVEATION_SMOKE_ONLY=1          (skip full; default at L1)
#   C1_WORLD_MODEL_FOVEATION_FULL_ONLY=1           (skip smoke; FORBIDDEN at L1
#                                                   pending Phase 3 council)
#
# Lane: lane_c1_world_model_foveation_campaign_l1_scaffold_20260514
# Cross-ref:
#   .omx/research/campaign_c1_world_model_foveation_20260514.md
#   feedback_long_term_multi_year_campaigns_landed_20260514.md (C1 campaign)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke-only by default at L1 -- full main raises NotImplementedError until
# Phase 3 council approves the multi-stage schedule. Operator can override
# C1_WORLD_MODEL_FOVEATION_SMOKE_ONLY=0 once council unlocks _full_main.
# Catalog #189: empty array expansion guarded under set -u (macOS bash 3.2).
SMOKE_ARGS=()
if [ "${C1_WORLD_MODEL_FOVEATION_SMOKE_ONLY:-1}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${C1_WORLD_MODEL_FOVEATION_FULL_ONLY:-0}" = "1" ]; then
    echo "ERROR: --full-only is FORBIDDEN at L1 scaffold. The trainer's" >&2
    echo "       _full_main raises NotImplementedError pending Phase 3" >&2
    echo "       council approval. See:" >&2
    echo "         .omx/research/campaign_c1_world_model_foveation_20260514.md" >&2
    echo "         feedback_c1_world_model_foveation_campaign_l1_scaffold_landed_20260514.md" >&2
    exit 30
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_c1_world_model_foveation_modal_t4_smoke_dispatch \
    --smoke-epochs "${C1_WORLD_MODEL_FOVEATION_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${C1_WORLD_MODEL_FOVEATION_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${C1_WORLD_MODEL_FOVEATION_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_c1_world_model_foveation_modal_t4_smoke_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
