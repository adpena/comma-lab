#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: .omx/operator_authorize_recipes/substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.yaml
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
#   NSCS01_EPOCHS=100             (full epochs per recipe cost_band.epochs)
#   MODAL_TIMEOUT_HOURS=1.0       (full Modal hard-kill wall-clock)
#   NSCS01_SMOKE_EPOCHS=1         (smoke epoch override)
#   NSCS01_SMOKE_GPU=T4           (smoke GPU class; recipe min_smoke_gpu=T4)
#   NSCS01_SMOKE_TIMEOUT_HOURS=0.5
#   NSCS01_SMOKE_ONLY=1           (toggle smoke-only via SMOKE_ONLY mode)
#   NSCS01_FULL_ONLY=1            (operator override; defeats the gate)
#   NSCS01_HEAD0_BITS=4           (split-head per-pair latent renderer)
#   NSCS01_HEAD1_BITS=8
#   NSCS01_LATENT_DIM=16
#
# Lane: lane_nscs01_nullspace_split_renderer_20260515
# Substrate: nscs01 nullspace-split-renderer (SegNet last-frame nullspace exploit)
# Cross-ref:
#   .omx/research/nscs01_nullspace_split_renderer_design_20260515.md
#   feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md (SA02)
#   src/tac/substrates/nscs01_nullspace_split_renderer/__init__.py (Catalog #124 8-field decl)
#   feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md
#
# Status at landing: recipe dispatch_enabled=false (smoke_only=true,
# research_only=true) per Catalog #240 sister-protection — trainer's
# _full_main has the PR95-paradigm implementation but recipe stays gated
# pending Phase 2 council green-up + paired CPU+CUDA Tier C anchor.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${NSCS01_SMOKE_ONLY:-0}" = "1" ] || [ "${SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${NSCS01_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch \
    --smoke-epochs "${NSCS01_SMOKE_EPOCHS:-1}" \
    --smoke-gpu "${NSCS01_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${NSCS01_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:nscs01_nullspace_split_renderer_first_anchor_full" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
