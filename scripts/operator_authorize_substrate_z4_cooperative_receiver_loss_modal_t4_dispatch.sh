#!/bin/bash
# Operator-authorize wrapper for the Z4 cooperative-receiver loss substrate
# (Time-Traveler L5 staircase Step 2).
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.yaml``
# Campaign ledger: ``.omx/research/campaign_z4_cooperative_receiver_loss_20260514.md``
# Council ledger: ``feedback_grand_council_maximize_value_landed_20260514.md`` STAIRCASE 10/11
# Long-term roadmap: ``feedback_long_term_multi_year_campaigns_landed_20260514.md`` C5
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# The Z4 recipe is gated on Z3's contest-CUDA anchor per Catalog #173
# (`canary_status: post_canary_dependent`, `canary_dependency: lane_z3_balle_hyperprior_bolton_recover_20260514`).
# The 100-epoch smoke validates the cooperative-receiver loss grammar at
# score_claim=false; the full 200-epoch dispatch is the Atick-Redlich
# probe-disambiguator (lambda_pixel sweep over [0.0, 0.5, 1.0]).
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
if [ "${Z4_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${Z4_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_z4_cooperative_receiver_loss_modal_t4_dispatch \
    --smoke-epochs "${Z4_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${Z4_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${Z4_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:operator_authorize_substrate_z4_cooperative_receiver_loss_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
