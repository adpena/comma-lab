#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/substrate_self_compress_nn_modal_a100_dispatch.yaml``
#
# Per Catalog #167, this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100              (default A100 per recipe default)
#   SELF_COMPRESS_NN_EPOCHS=2000             (council default; full training)
#   SELF_COMPRESS_NN_CODEBOOK_K=256          (cluster count; default 256)
#   SELF_COMPRESS_NN_CODEBOOK_DV=8           (per-cluster vector dim; default 8)
#   MODAL_TIMEOUT_HOURS=4.0                  (full Modal hard-kill wall-clock)
#   SELF_COMPRESS_NN_SMOKE_EPOCHS=100        (smoke epoch override)
#   SELF_COMPRESS_NN_SMOKE_GPU=T4            (smoke GPU class)
#
# Lane: lane_wave1_self_compress_nn_trainer_build_20260512
# Cross-ref: feedback_wave1_self_compress_nn_trainer_build_LANDED_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${SELF_COMPRESS_NN_SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${SELF_COMPRESS_NN_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_self_compress_nn_modal_a100_dispatch \
    --smoke-epochs "${SELF_COMPRESS_NN_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${SELF_COMPRESS_NN_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${SELF_COMPRESS_NN_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_self_compress_nn_modal_a100_dispatch" \
    "${SMOKE_ARGS[@]}" \
    "$@"
