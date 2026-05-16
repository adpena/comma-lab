#!/bin/bash
# Thin shim: delegates to the canonical smoke-before-full entry point.
#
# Recipe: .omx/operator_authorize_recipes/substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch.yaml
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
# Per Catalog #215 (FIX-HARDEN-OPT 2026-05-14 P0): A100 full-run requires
# matching smoke-phase compute class. End-to-end joint codec at 384x512
# resolution + 64-channel main latent + entropy bottleneck is too memory-
# intensive for T4; A100 smoke is required to avoid 1h timeouts.
#
# Env-var overrides honored by the canonical entry point:
#   MODAL_GPU=A100|H100           (default A100 per recipe gpu: ${MODAL_GPU:-A100})
#   NSCS03_EPOCHS=100             (full epochs per recipe cost_band.epochs)
#   MODAL_TIMEOUT_HOURS=12.0      (full Modal hard-kill wall-clock)
#   NSCS03_SMOKE_EPOCHS=1         (smoke epoch override)
#   NSCS03_SMOKE_GPU=A100         (smoke GPU class; recipe min_smoke_gpu=A100)
#   NSCS03_SMOKE_TIMEOUT_HOURS=0.5
#   NSCS03_SMOKE_ONLY=1           (toggle smoke-only via SMOKE_ONLY mode)
#   NSCS03_FULL_ONLY=1            (operator override; defeats the gate)
#   NSCS03_MAIN_LATENT_CHANNELS=64
#   NSCS03_HYPER_LATENT_CHANNELS=32
#   NSCS03_LAMBDA_R=0.5           (rate-distortion Lagrange multiplier target)
#   NSCS03_GDN_EPS=1e-6
#   NSCS03_SIGMA_FLOOR=1e-4
#
# Lane: lane_nscs03_end_to_end_balle_joint_codec_20260515
# Substrate: nscs03 end-to-end Ballé 2018 joint codec
#            (g_a + entropy bottleneck + scale hyperprior + g_s; jointly trained)
# Cross-ref:
#   .omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json#NSCS03
#   src/tac/substrates/nscs03_end_to_end_balle_joint_codec/__init__.py (HNeRV parity 13-lessons)
#   feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md
#   Ballé/Minnen/Singh/Hwang/Johnston — "Variational Image Compression with a
#   Scale Hyperprior" ICLR 2018
#
# Status at landing: recipe dispatch_enabled=false (smoke_only=true,
# research_only=true) per Catalog #240 + #220 sister-protection +
# CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" —
# trainer's _full_main has the END-TO-END Ballé joint codec implemented
# per UNIQUE-AND-COMPLETE-PER-METHOD operating mode (per
# feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md)
# but recipe stays council-gated pending Phase 2 λ_R sweep +
# σ-floor calibration + first smoke anchor.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ARGS=()
if [ "${NSCS03_SMOKE_ONLY:-0}" = "1" ] || [ "${SMOKE_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--smoke-only)
fi
if [ "${NSCS03_FULL_ONLY:-0}" = "1" ]; then
    SMOKE_ARGS+=(--full-only)
fi

exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch \
    --smoke-epochs "${NSCS03_SMOKE_EPOCHS:-1}" \
    --smoke-gpu "${NSCS03_SMOKE_GPU:-A100}" \
    --smoke-timeout-hours "${NSCS03_SMOKE_TIMEOUT_HOURS:-0.5}" \
    --operator-handle "claude:nscs03_end_to_end_balle_joint_codec_first_anchor_full" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    "$@"
