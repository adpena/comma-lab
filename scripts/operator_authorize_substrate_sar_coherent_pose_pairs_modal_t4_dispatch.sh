#!/bin/bash
# Operator-authorize wrapper for the L2 SAR coherent pose-pair substrate.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_sar_coherent_pose_pairs_modal_t4_dispatch.yaml``
# Lincoln Lab L2 ledger: ``.omx/research/expert_team_signal_processing_lincoln_lab_20260513.md`` §2
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# A 100-epoch ~$0.30 Modal T4 smoke fires FIRST, validates rc=0 + auth-eval
# JSON present + score in plausible band [0.05, 0.40], and only proceeds to
# the full 2000-epoch T4 dispatch on smoke-green.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full
# dispatch ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored by the smoke wrapper:
#   MODAL_GPU=T4|A10G|A100|H100              (default T4 per recipe; cheap dispatch)
#   SARC_EPOCHS=2000                         (council default; full training)
#   MODAL_TIMEOUT_HOURS=2.0                  (Modal hard-kill wall-clock)
#   SARC_SMOKE_EPOCHS=100                    (smoke epoch override)
#   SARC_SMOKE_GPU=T4                        (smoke GPU class)
#   SARC_SMOKE_ONLY=1                        (skip full even on smoke-green)
#   SARC_FULL_ONLY=1                         (skip smoke; operator override
#                                             after >=3 successful anchors)
#
# Lane: lane_sar_coherent_pose_pairs_substrate_20260513
# Cross-ref:
#   feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md
#   feedback_expert_team_signal_processing_alien_tech_landed_20260513.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: empty array expansion guarded under set -u.
SMOKE_ARGS=()
if [ "${SARC_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${SARC_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_sar_coherent_pose_pairs_modal_t4_dispatch \
    --smoke-epochs "${SARC_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${SARC_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${SARC_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_sar_coherent_pose_pairs_modal_t4_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
