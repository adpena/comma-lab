#!/bin/bash
# Remote lane script: substrate vq_vae first-anchor dispatch.
#
# Trainer: experiments/train_substrate_vq_vae.py (WAVE-1-A).
# Lane: lane_substrate_vq_vae_20260512
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# MUST be set before sourcing the bootstrap to prevent the sourced main flow
# (which expects a pre-built archive.zip) from running.
#
# Council memo refs:
#   - feedback_wave1_vq_vae_trainer_build_LANDED_20260512.md (this landing)
#   - .omx/research/grand_council_fields_medal_substrate_design_20260512.md
#     (council Phase 5 prediction: 0.17 [contest-CUDA])
#   - van den Oord, Vinyals, Kavukcuoglu - "Neural Discrete Representation
#     Learning" NeurIPS 2017 (architectural anchor)
#
# Score-tagging: the Modal training wrapper usually sets AUTH_EVAL_DEVICE=cpu
# and MODAL_AUTH_EVAL_ADVISORY_ONLY=1, so inline auth-eval is diagnostic and
# non-promotable unless the actual auth-eval artifact is CUDA-axis.
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
LANE_ID="lane_substrate_vq_vae_20260512"
TAG="${TAG:-substrate_vq_vae}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_vq_vae_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${VQ_VAE_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$VQ_VAE_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
VQ_VAE_VIDEO_PATH="${VQ_VAE_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
VQ_VAE_OUTPUT_DIR="${VQ_VAE_OUTPUT_DIR:-$OUTPUT_DIR}"
VQ_VAE_EPOCHS="${VQ_VAE_EPOCHS:-2000}"
VQ_VAE_BATCH_SIZE="${VQ_VAE_BATCH_SIZE:-16}"
VQ_VAE_UPSTREAM_DIR="${VQ_VAE_UPSTREAM_DIR:-$WORKSPACE/upstream}"
VQ_VAE_DEVICE="${VQ_VAE_DEVICE:-cuda}"
VQ_VAE_CODEBOOK_SIZE="${VQ_VAE_CODEBOOK_SIZE:-}"
VQ_VAE_ALPHA_RATE="${VQ_VAE_ALPHA_RATE:-}"
VQ_VAE_ENABLE_PROCEDURAL_INDICES_RESIDUAL="${VQ_VAE_ENABLE_PROCEDURAL_INDICES_RESIDUAL:-0}"
VQ_VAE_PROCEDURAL_INDICES_SEED_BYTES="${VQ_VAE_PROCEDURAL_INDICES_SEED_BYTES:-32}"
VQ_VAE_PROCEDURAL_INDICES_GENERATOR_KIND="${VQ_VAE_PROCEDURAL_INDICES_GENERATOR_KIND:-pcg64}"

DISPATCH_INSTANCE_JOB_ID="${VQ_VAE_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${VQ_VAE_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-vq-vae] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

if [ -z "$VQ_VAE_CODEBOOK_SIZE" ]; then
    log "FATAL: VQ_VAE_CODEBOOK_SIZE is required by the K-sweep recipe; refusing phantom default K"
    exit 64
fi
if [ -z "$VQ_VAE_ALPHA_RATE" ]; then
    log "FATAL: VQ_VAE_ALPHA_RATE is required by the K-sweep recipe; refusing phantom default alpha"
    exit 64
fi
if ! [[ "$VQ_VAE_CODEBOOK_SIZE" =~ ^[0-9]+$ ]]; then
    log "FATAL: VQ_VAE_CODEBOOK_SIZE must be an integer, got '$VQ_VAE_CODEBOOK_SIZE'"
    exit 64
fi
if ! [[ "$VQ_VAE_ALPHA_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    log "FATAL: VQ_VAE_ALPHA_RATE must be a non-negative decimal, got '$VQ_VAE_ALPHA_RATE'"
    exit 64
fi

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: VQ_VAE_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

# Stage 0b: NVDEC probe (per CLAUDE.md `feedback_vastai_nvdec_host_variation`).
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "FATAL: NVDEC probe failed (host hardware/driver/DALI mismatch); refusing dispatch"
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
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel per Catalog #163.
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

# Stage 2: provenance + remote code parity (per CLAUDE.md "Remote code parity").
log "stage_2_provenance_begin"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch
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
    'video_path': '$VQ_VAE_VIDEO_PATH',
    'upstream_dir': '$VQ_VAE_UPSTREAM_DIR',
    'epochs': $VQ_VAE_EPOCHS,
    'batch_size': $VQ_VAE_BATCH_SIZE,
    'device': '$VQ_VAE_DEVICE',
    'codebook_size': $VQ_VAE_CODEBOOK_SIZE,
    'alpha_rate': $VQ_VAE_ALPHA_RATE,
    'diagnostic_contract': 'fixed_int16_vqv1_quality_probe_no_frontier_authority',
    'predicted_band': None,
    'predicted_basis': 'none_for_fixed_int16_diagnostic',
    'class_shift_followup': 'vq_v1_k_dependent_entropy_packed_archive',
    'score_claim': False,
    'promotion_eligible': False,
    'rank_or_kill_eligible': False,
    'literature_anchor': 'van den Oord et al. NeurIPS 2017 - Neural Discrete Representation Learning',
}
import pathlib
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
# All 6 TIER_1_OPERATOR_REQUIRED_FLAGS are threaded explicitly per Catalog #151.
log "stage_4_trainer_invoke_begin video=$VQ_VAE_VIDEO_PATH epochs=$VQ_VAE_EPOCHS device=$VQ_VAE_DEVICE batch_size=$VQ_VAE_BATCH_SIZE codebook_size=$VQ_VAE_CODEBOOK_SIZE alpha_rate=$VQ_VAE_ALPHA_RATE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
EXTRA_TRAINER_FLAGS=()
if [ "$VQ_VAE_ENABLE_PROCEDURAL_INDICES_RESIDUAL" = "1" ]; then
    EXTRA_TRAINER_FLAGS+=(
        "--enable-procedural-indices-residual"
        "--procedural-indices-seed-bytes" "$VQ_VAE_PROCEDURAL_INDICES_SEED_BYTES"
        "--procedural-indices-generator-kind" "$VQ_VAE_PROCEDURAL_INDICES_GENERATOR_KIND"
    )
fi
set +e
"$PYBIN" experiments/train_substrate_vq_vae.py \
    --video-path "$VQ_VAE_VIDEO_PATH" \
    --output-dir "$VQ_VAE_OUTPUT_DIR" \
    --epochs "$VQ_VAE_EPOCHS" \
    --batch-size "$VQ_VAE_BATCH_SIZE" \
    --upstream-dir "$VQ_VAE_UPSTREAM_DIR" \
    --device "$VQ_VAE_DEVICE" \
    --codebook-size "$VQ_VAE_CODEBOOK_SIZE" \
    --alpha-rate "$VQ_VAE_ALPHA_RATE" \
    "${EXTRA_TRAINER_FLAGS[@]}" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
AUTH_EVAL_DEVICE_RESOLVED="${AUTH_EVAL_DEVICE:-$VQ_VAE_DEVICE}"
AUTH_EVAL_DEVICE_RESOLVED="${AUTH_EVAL_DEVICE_RESOLVED%%:*}"
case "$AUTH_EVAL_DEVICE_RESOLVED" in
    cuda|cpu) ;;
    *) AUTH_EVAL_DEVICE_RESOLVED="cuda" ;;
esac
AUTH_EVAL_JSON="$OUTPUT_DIR/contest_auth_eval_${AUTH_EVAL_DEVICE_RESOLVED}.json"
ARCHIVE_PATH="$OUTPUT_DIR/archive.zip"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present device=$AUTH_EVAL_DEVICE_RESOLVED path=$AUTH_EVAL_JSON"
    if [ "$AUTH_EVAL_DEVICE_RESOLVED" = "cuda" ]; then
        log "LANE_VQ_VAE_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
    else
        log "LANE_VQ_VAE_DONE [diagnostic-auth-eval] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC score_claim=false promotion_eligible=false"
    fi
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
