#!/bin/bash
# Operator-authorize wrapper for the PR95++ meta-stack-of-stacks enhanced
# curriculum substrate (11 enhancements over PR95's 8-stage protocol).
#
# Recipe: .omx/operator_authorize_recipes/substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch.yaml
#
# This wrapper routes through tools/run_modal_smoke_before_full.py per
# Catalog #167 (check_substrate_dispatch_uses_smoke_before_full_pattern).
# A 100-epoch ~$0.30 Modal T4 smoke fires FIRST, validates rc=0 + auth-eval
# JSON present + score in plausible band, and only proceeds to the full
# 29,650-epoch A100 dispatch on smoke-green.
#
# Per Catalog #162 (check_operator_authorize_canonical_use) the full
# dispatch ultimately delegates to tools/operator_authorize.py --recipe.
#
# Env-var overrides:
#   MODAL_GPU=T4|A10G|A100|H100              (default A100 per recipe)
#   PR95PLUS_EPOCHS_MULTIPLIER=1.0           (full protocol; 0.01 for $0.30 smoke)
#   PR95PLUS_CURRICULUM=pr95_enhanced        (recommended) | pr95_faithful (A/B)
#   PR95PLUS_CODEBOOK_PATH=""                (path to pre-distilled DP1 codebook)
#   MODAL_TIMEOUT_HOURS=8.0                  (Modal hard-kill wall-clock)
#   PR95PLUS_SMOKE_EPOCHS=100                (smoke epoch override)
#   PR95PLUS_SMOKE_GPU=T4                    (smoke GPU class)
#   PR95PLUS_SMOKE_ONLY=1                    (skip full even on smoke-green)
#   PR95PLUS_FULL_ONLY=1                     (skip smoke; operator override)
#
# Lane: lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: guard the optional array expansion for macOS bash 3.2 + set -u.
SMOKE_ARGS=()
if [ "${PR95PLUS_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${PR95PLUS_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch \
    --smoke-epochs "${PR95PLUS_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${PR95PLUS_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${PR95PLUS_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
