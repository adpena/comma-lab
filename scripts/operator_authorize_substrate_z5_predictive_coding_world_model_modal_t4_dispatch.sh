#!/bin/bash
# Operator-authorize wrapper for the Z5 predictive-coding world-model
# substrate (Time-Traveler L5 staircase Step 3).
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_z5_predictive_coding_world_model_modal_t4_dispatch.yaml``
# Campaign ledger: ``.omx/research/campaign_z5_predictive_coding_world_model_20260514.md``
# Council ledger: ``feedback_grand_council_maximize_value_landed_20260514.md`` STAIRCASE end-state
# Long-term roadmap: ``feedback_long_term_multi_year_campaigns_landed_20260514.md`` C5
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# The Z5 recipe is gated on Z4's contest-CUDA anchor per Catalog #173
# (`canary_status: post_canary_dependent`, `canary_dependency: lane_z4_cooperative_receiver_loss_step2_20260514`).
# Time-Traveler staircase order Z3 -> Z4 -> Z5 cannot be skipped.
# The 100-epoch smoke validates the hierarchical-predictor grammar +
# autoregressive recurrence at score_claim=false; the full 300-epoch
# dispatch is the Rao-Ballard probe-disambiguator (identity_predictor
# ablation at matched parameter budget).
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full
# dispatch ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Per Catalog #189 (`check_shell_empty_arrays_guarded_under_set_u`) every
# optional-array expansion is guarded as ``${ARR[@]+"${ARR[@]}"}`` so the
# `--dry-run` path works under macOS bash 3.2 + `set -u`.
#
# Per Catalog #199 (paired-env-var bypass) operators MAY set both:
#   OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
#   OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.50
# to skip the interactive y/n prompt under non-interactive subprocess
# invocation. The wrapper does NOT set either by default.
#
# Per Catalog #202 (clean-bypass paired-env-var) operators MAY set both:
#   OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1
#   OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1
# to dispatch from a dirty shared worktree (sister-subagent in-flight work
# expected). Default refuses dirty worktree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: empty array expansion guarded under set -u (macOS bash 3.2).
SMOKE_ARGS=()
if [ "${Z5_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${Z5_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_z5_predictive_coding_world_model_modal_t4_dispatch \
    --smoke-epochs "${Z5_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${Z5_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${Z5_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:operator_authorize_substrate_z5_predictive_coding_world_model_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
