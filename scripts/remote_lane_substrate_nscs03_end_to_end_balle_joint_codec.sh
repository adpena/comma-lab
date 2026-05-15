#!/bin/bash
# Remote lane script: substrate NSCS03 end-to-end Ballé joint codec.
#
# Trainer: experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py
# Lane: lane_nscs03_end_to_end_balle_joint_codec_20260515
#
# Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
# non-negotiable + assumptions-challenge-audit NSCS03: this lane is
# RESEARCH_ONLY at the trainer level. _full_main raises NotImplementedError;
# only the SMOKE phase is dispatchable for integration validation. The
# operator must approve the Phase 2 follow-up subagent (λ_R sweep + σ-floor
# calibration) before the FULL trainer can be wired.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Council memo refs:
#   - .omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json#NSCS03
#     (NSCS03 design memo: end-to-end Ballé joint codec; predicted ΔS [-0.020, -0.050])
#   - feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md
#     (this lane's dispatch chain runs under the smoke-before-full pattern
#     per Catalog #167)
#
# Score-tagging: SMOKE PHASE produces a `[smoke_no_auth_eval]` tag (no auth
# eval custody); the trainer's _smoke_main writes
# `auth_eval_score_axis: smoke_no_auth_eval` + `promotion_eligible: false`
# per CLAUDE.md "Apples-to-apples evidence discipline".
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# === Catalog #244 / D1 incident anchor (commit 611495f26): canonical Modal/CUDA env hygiene ===
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing directive
# 2026-05-15 ("all possible should be pulled into the decorator or similar reusable
# and shareable tools and helpers and such"). Sister substrates D1/D4/Z3/Z4/Z5 carry
# this block; backfilled to all 31 sister drivers via Catalog #244 strict-flip wave.
# Future drivers should be auto-generated via tac.substrate_registry.driver_generator
# (which AUTO-EMITS the block from canonical constants in tac.deploy.modal.runtime).
# D1 dispatch 2026-05-15 (Modal T4 smoke) crashed at NVML 999 because the lane script
# did not export DALI_DISABLE_NVML=1 before DALI imported NVML.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_nscs03_end_to_end_balle_joint_codec_20260515"
TAG="${TAG:-substrate_nscs03_end_to_end_balle_joint_codec}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_nscs03_end_to_end_balle_joint_codec_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
NSCS03_VIDEO_PATH="${NSCS03_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
NSCS03_OUTPUT_DIR="${NSCS03_OUTPUT_DIR:-$OUTPUT_DIR}"
NSCS03_EPOCHS="${NSCS03_EPOCHS:-2000}"
NSCS03_UPSTREAM_DIR="${NSCS03_UPSTREAM_DIR:-$WORKSPACE/upstream}"
NSCS03_DEVICE="${NSCS03_DEVICE:-cuda}"
NSCS03_MAIN_LATENT_CHANNELS="${NSCS03_MAIN_LATENT_CHANNELS:-64}"
NSCS03_HYPER_LATENT_CHANNELS="${NSCS03_HYPER_LATENT_CHANNELS:-32}"
NSCS03_LAMBDA_R="${NSCS03_LAMBDA_R:-0.5}"
NSCS03_GDN_EPS="${NSCS03_GDN_EPS:-1e-6}"
NSCS03_SIGMA_FLOOR="${NSCS03_SIGMA_FLOOR:-1e-4}"

# Smoke-phase mode is determined by NSCS03_SMOKE_PHASE env var
# (set by tools/run_modal_smoke_before_full.py).
NSCS03_SMOKE_PHASE="${NSCS03_SMOKE_PHASE:-0}"

DISPATCH_INSTANCE_JOB_ID="${NSCS03_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${NSCS03_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-nscs03] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: NSCS03_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py"
    exit 21
fi
if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
    log "FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH"
    exit 21
fi

# Stage 0b: NVDEC probe (per CLAUDE.md `feedback_vastai_nvdec_host_variation`).
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "FATAL: NVDEC probe failed; refusing dispatch"
        exit 2
    }
    log "stage_0b_nvdec_probe_done"
else
    log "WARN: scripts/probe_nvdec.sh missing - skipping NVDEC early-fail probe"
fi

# Stage 1: bootstrap runtime deps (canonical, per CLAUDE.md).
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# shellcheck source=/dev/null
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 prevents the sourced script's main
# flow from running; we only need bootstrap_runtime_deps() (Catalog #163).
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"

if declare -f bootstrap_runtime_deps > /dev/null 2>&1; then
    log "stage_1_bootstrap_runtime_deps_begin"
    bootstrap_runtime_deps
    log "stage_1_bootstrap_runtime_deps_done PYBIN=${PYBIN:-unset}"
else
    log "FATAL: bootstrap_runtime_deps function not found after sourcing scripts/remote_archive_only_eval.sh"
    exit 23
fi

if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    log "FATAL: PYBIN not set or not executable after bootstrap (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Stage 2: provenance.
log "stage_2_provenance_begin"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch, pathlib
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'lane_id': '$LANE_ID',
    'dispatch_instance_job_id': '$DISPATCH_INSTANCE_JOB_ID',
    'dispatch_platform': '$DISPATCH_PLATFORM',
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'video_path': '$NSCS03_VIDEO_PATH',
    'upstream_dir': '$NSCS03_UPSTREAM_DIR',
    'epochs': $NSCS03_EPOCHS,
    'device': '$NSCS03_DEVICE',
    'main_latent_channels': $NSCS03_MAIN_LATENT_CHANNELS,
    'hyper_latent_channels': $NSCS03_HYPER_LATENT_CHANNELS,
    'lambda_R': $NSCS03_LAMBDA_R,
    'gdn_eps': $NSCS03_GDN_EPS,
    'sigma_floor': $NSCS03_SIGMA_FLOOR,
    'smoke_phase': '$NSCS03_SMOKE_PHASE',
    'predicted_band': [0.18, 0.21],
    'predicted_basis': 'assumptions_challenge_audit_NSCS03_end_to_end_differentiable_codec',
}
pathlib.Path('$PROVENANCE').write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
"
log "stage_2_provenance_done"

# Stage 3: heartbeat watchdog (every 5 min per CLAUDE.md).
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

# Stage 4: invoke trainer.
#
# SMOKE phase: --smoke flag → _smoke_main runs (synthetic batches; CPU OK;
#   no auth eval; integration validation only).
# FULL phase: NO --smoke flag → _full_main raises NotImplementedError per
#   CLAUDE.md substrate-engineering exception until Phase 2 council
#   adjudicates λ_R sweep + σ-floor calibration.
SMOKE_FLAG=""
if [ "$NSCS03_SMOKE_PHASE" = "1" ]; then
    SMOKE_FLAG="--smoke"
    log "stage_4_smoke_phase_invoke epochs=$NSCS03_EPOCHS"
else
    log "stage_4_full_phase_invoke epochs=$NSCS03_EPOCHS (will raise NotImplementedError per substrate-engineering exception)"
fi

TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py \
    --video-path "$NSCS03_VIDEO_PATH" \
    --output-dir "$NSCS03_OUTPUT_DIR" \
    --epochs "$NSCS03_EPOCHS" \
    --upstream-dir "$NSCS03_UPSTREAM_DIR" \
    --device "$NSCS03_DEVICE" \
    --main-latent-channels "$NSCS03_MAIN_LATENT_CHANNELS" \
    --hyper-latent-channels "$NSCS03_HYPER_LATENT_CHANNELS" \
    --lambda-R "$NSCS03_LAMBDA_R" \
    --gdn-eps "$NSCS03_GDN_EPS" \
    --sigma-floor "$NSCS03_SIGMA_FLOOR" \
    $SMOKE_FLAG \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
STATS_JSON="$OUTPUT_DIR/stats.json"
if [ -f "$STATS_JSON" ]; then
    log "stats_artifact_present path=$STATS_JSON"
    if [ "$NSCS03_SMOKE_PHASE" = "1" ]; then
        log "LANE_NSCS03_SMOKE_DONE [smoke_no_auth_eval] stats=$STATS_JSON rc=$TRAIN_RC"
    else
        log "LANE_NSCS03_FULL_BLOCKED [research_only] stats=$STATS_JSON rc=$TRAIN_RC (full trainer is council-gated; reactivation pending Phase 2)"
    fi
else
    log "stats_artifact_missing path=$STATS_JSON (trainer may have failed before writing stats)"
fi

exit "$TRAIN_RC"
