#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: .omx/operator_authorize_recipes/substrate_nscs02_downsampled_renderer_modal_t4_dispatch.yaml
#
# Per Catalog #167 this wrapper fires a short Modal smoke through
# tools/run_modal_smoke_before_full.py before any full canary.
# Per Catalog #189 we use ${ARR[@]+"${ARR[@]}"} for empty-array safety.
# Per Catalog #244 + #224 the canonical NVML 3-export block is auto-emitted
# by tac.substrate_registry.driver_generator into the remote_lane_*.sh
# (DALI_DISABLE_NVML=1 + CUBLAS_WORKSPACE_CONFIG + PYTORCH_CUDA_ALLOC_CONF).
# Per Catalog #191 + #201 the sentinel files (declared in the recipe) stay
# inside the canonical Modal mount-set; the dispatcher honors them.
# Per Catalog #202 the clean-bypass paired-env discipline applies via
# tools/operator_authorize.py if invoked.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=T4|A10G|A100|H100   (default T4 per recipe gpu: ${MODAL_GPU:-T4})
#   NSCS02_EPOCHS=100             (full epochs per recipe cost_band.epochs)
#   MODAL_TIMEOUT_HOURS=1.5       (full Modal hard-kill wall-clock)
#   NSCS02_SMOKE_EPOCHS=1         (smoke epoch override)
#   NSCS02_SMOKE_GPU=T4           (smoke GPU class; recipe min_smoke_gpu=T4)
#   NSCS02_SMOKE_TIMEOUT_HOURS=0.5
#   NSCS02_SMOKE_ONLY=1           (toggle smoke-only via SMOKE_ONLY mode)
#   NSCS02_FULL_ONLY=1            (operator override; defeats the gate)
#   NSCS02_UPSAMPLE_MODE=bicubic  (inflate-time upsample method)
#   NSCS02_SEG_WEIGHT=100.0
#   NSCS02_POSE_WEIGHT=1.0
#
# Lane: lane_nscs02_downsampled_renderer_inflate_upsample_20260515
# Substrate: nscs02 downsampled-renderer + inflate-upsample (192,256 → 384,512)
# Cross-ref:
#   .omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json#NSCS02
#   feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md
#   feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md
#
# Status at landing: recipe dispatch_enabled=false (smoke_only=true,
# research_only=true) per Catalog #240 sister-protection — trainer's
# _full_main pending the resizing-chain ablation + paired-axis auth-eval
# custody per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${NSCS02_SMOKE_ONLY:-0}" = "1" ] || [ "${SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${NSCS02_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs02_downsampled_renderer_modal_t4_dispatch \
    --smoke-epochs "${NSCS02_SMOKE_EPOCHS:-1}" \
    --smoke-gpu "${NSCS02_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${NSCS02_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:nscs02_downsampled_renderer_first_anchor_full" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
