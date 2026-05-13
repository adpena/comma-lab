#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_self_compress_nn_modal_a100_dispatch.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100              (default A100 per recipe default)
#   SELF_COMPRESS_NN_EPOCHS=2000             (council default; full training)
#   SELF_COMPRESS_NN_CODEBOOK_K=256          (cluster count; default 256)
#   SELF_COMPRESS_NN_CODEBOOK_DV=8           (per-cluster vector dim; default 8)
#   MODAL_TIMEOUT_HOURS=4.0                  (Modal hard-kill wall-clock)
#
# Lane: lane_wave1_self_compress_nn_trainer_build_20260512
# Cross-ref: feedback_wave1_self_compress_nn_trainer_build_LANDED_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec .venv/bin/python tools/operator_authorize.py \
    --recipe substrate_self_compress_nn_modal_a100_dispatch \
    --agent "claude:operator_authorize_substrate_self_compress_nn_modal_a100_dispatch" \
    "$@"
