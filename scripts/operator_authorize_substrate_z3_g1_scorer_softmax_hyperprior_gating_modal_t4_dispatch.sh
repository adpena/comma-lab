#!/bin/bash
# Operator-authorize wrapper for the Z3-G1 scorer-class-conditional gating substrate.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch.yaml``
# Source memo: ``feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md``
# Council ledger: ``feedback_grand_council_evidence_review_modal_failures_no_compromise_optimal_lowest_score_landed_20260515.md``
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# Cheap T4 smoke validates the integration BEFORE the $5-10 full T4 dispatch.
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
#   OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=15.00
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
if [ "${Z3_G1_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${Z3_G1_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch \
    --smoke-epochs "${Z3_G1_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${Z3_G1_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${Z3_G1_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:operator_authorize_substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
