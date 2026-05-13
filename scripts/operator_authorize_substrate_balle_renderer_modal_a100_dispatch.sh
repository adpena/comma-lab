#!/bin/bash
# Operator-authorize wrapper for the Ballé hyperprior renderer (β) substrate.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_balle_renderer_modal_a100_dispatch.yaml``
#
# This wrapper routes through `tools/run_modal_smoke_before_full.py` per
# Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`).
# A 100-epoch ~$0.30 Modal T4 smoke fires FIRST, validates rc=0 + auth-eval
# JSON present + score in plausible band, and only proceeds to the full
# 2000-epoch A100 dispatch on smoke-green.
#
# Per Catalog #162 (`check_operator_authorize_canonical_use`) the full
# dispatch ultimately delegates to `tools/operator_authorize.py --recipe`.
#
# Env-var overrides honored by the smoke wrapper:
#   MODAL_GPU=T4|A10G|A100|H100              (default A100 per recipe)
#   BALLE_RENDERER_EPOCHS=2000               (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0                  (Modal hard-kill wall-clock)
#   BALLE_RENDERER_SMOKE_EPOCHS=100          (smoke epoch override)
#   BALLE_RENDERER_SMOKE_GPU=T4              (smoke GPU class)
#   BALLE_RENDERER_SMOKE_ONLY=1              (skip full even on smoke-green)
#   BALLE_RENDERER_FULL_ONLY=1               (skip smoke; operator override
#                                             after >=3 successful anchors)
#
# Lane: lane_substrate_balle_renderer_20260512
# Cross-ref: feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Smoke + Full chain via the canonical Catalog #167 helper.
exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_balle_renderer_modal_a100_dispatch \
    --smoke-epochs "${BALLE_RENDERER_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${BALLE_RENDERER_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${BALLE_RENDERER_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_balle_renderer_modal_a100_dispatch" \
    "$@"
