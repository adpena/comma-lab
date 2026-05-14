#!/bin/bash
# Operator-authorize wrapper for the Wyner-Ziv cooperative-receiver substrate.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_wyner_ziv_cooperative_receiver_modal_a100_dispatch.yaml``
# Alien-tech N3 entry: ``feedback_expert_team_signal_processing_alien_tech_landed_20260513.md``
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# A 100-epoch ~$0.30 Modal T4 smoke fires FIRST, validates rc=0 + auth-eval
# JSON present + score in plausible band [0.05, 0.50], and only proceeds to
# the full 3000-epoch A100 dispatch on smoke-green.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full
# dispatch ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored by the smoke wrapper:
#   MODAL_GPU=T4|A10G|A100|H100              (default A100 per recipe)
#   WYNER_ZIV_EPOCHS=3000                    (council default; full training)
#   MODAL_TIMEOUT_HOURS=3.0                  (Modal hard-kill wall-clock)
#   WYNER_ZIV_SMOKE_EPOCHS=100               (smoke epoch override)
#   WYNER_ZIV_SMOKE_GPU=T4                   (smoke GPU class)
#   WYNER_ZIV_SMOKE_ONLY=1                   (skip full even on smoke-green)
#   WYNER_ZIV_FULL_ONLY=1                    (skip smoke; operator override
#                                             after >=3 successful anchors)
#
# Lane: lane_wyner_ziv_cooperative_receiver_substrate_20260513
# Cross-ref:
#   feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md
#   feedback_expert_team_signal_processing_alien_tech_landed_20260513.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke + Full chain via the canonical Catalog #167 helper.
# Catalog #189: empty array expansion guarded under set -u (macOS bash 3.2).
SMOKE_ARGS=()
if [ "${WYNER_ZIV_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${WYNER_ZIV_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_wyner_ziv_cooperative_receiver_modal_a100_dispatch \
    --smoke-epochs "${WYNER_ZIV_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${WYNER_ZIV_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${WYNER_ZIV_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_wyner_ziv_cooperative_receiver_modal_a100_dispatch" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
