#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/scpp_stage1_modal_a100_dispatch.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Wave 5 ITEM 2: SC++ Stage 1 production-hardened Modal A100 dispatch
# infrastructure (sister of ``operator_authorize_scpp_stage1_anchor_dispatch.sh``
# which targets Modal T4 3-epoch smoke; this wrapper targets Modal A100 100-epoch
# full Stage 1 anchor for ~$8).
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100  (default A100 per Wave 5 production target)
#   SCPP_STAGE_1_EPOCHS=100      (production default; Stage 1 anchor full)
#   SCPP_TARGET_ARCHIVE_BYTES=180000  (PR106 r2 frontier band)
#   MODAL_TIMEOUT_HOURS=4.0      (Modal hard-kill wall-clock)
#
# Lane: lane_wave5_scpp_stage1_build_20260512
# Cross-ref: feedback_wave5_scpp_stage1_build_LANDED_20260512.md
#
# Per Catalog #146 / #151 / #152 / #153 / #163: required-input pre-dispatch
# validation + TIER_1 manifest threading + Modal mount manifest + remote-lane
# sentinel — ALL routed through tools/operator_authorize.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec .venv/bin/python tools/operator_authorize.py \
    --recipe scpp_stage1_modal_a100_dispatch \
    --agent "claude:operator_authorize_scpp_stage1_modal_a100_dispatch" \
    "$@"
