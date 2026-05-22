#!/bin/bash
# Remote lane script: substrate grayscale_lut first-anchor dispatch.
#
# Trainer: experiments/train_substrate_grayscale_lut.py (WAVE-4-GRAYSCALE-LUT).
# Lane: lane_substrate_grayscale_lut_20260512
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
#   - feedback_wave4_grayscale_lut_trainer_build_LANDED_20260512.md (this landing)
#   - .omx/research/grand_council_fields_medal_substrate_design_20260512.md
#     (council Phase 5 prediction: 0.18 [contest-CUDA])
#   - Selfcomp / szabolcs-cs PR #56 (architectural anchor: grayscale-LUT
#     analog mask paradigm)
#   - Quantizr 0.33 anchor 2026-04-21 (88-94K param decoder reference)
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_GRAYSCALE_LUT_DONE marker) per the CLAUDE.md
# score-tag rule and preflight completion-tag check.
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
LANE_ID="lane_substrate_grayscale_lut_20260512"
TAG="${TAG:-substrate_grayscale_lut}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_grayscale_lut_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${GRAYSCALE_LUT_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$GRAYSCALE_LUT_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
GRAYSCALE_LUT_VIDEO_PATH="${GRAYSCALE_LUT_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
GRAYSCALE_LUT_OUTPUT_DIR="${GRAYSCALE_LUT_OUTPUT_DIR:-$OUTPUT_DIR}"
GRAYSCALE_LUT_EPOCHS="${GRAYSCALE_LUT_EPOCHS:-2000}"
GRAYSCALE_LUT_BATCH_SIZE="${GRAYSCALE_LUT_BATCH_SIZE:-16}"
GRAYSCALE_LUT_UPSTREAM_DIR="${GRAYSCALE_LUT_UPSTREAM_DIR:-$WORKSPACE/upstream}"
GRAYSCALE_LUT_DEVICE="${GRAYSCALE_LUT_DEVICE:-cuda}"
# Catalog #151 env-var ladder extension 2026-05-21 (OVERNIGHT-XX lane_overnight_xx_selfcomp_tier_2_paid_modal_a100_first_anchor_dispatch_20260521):
# lut_bits parameterization landed via OVERNIGHT-TT Phase 2 BUILD 2026-05-21 (commit 92a77da47);
# canonical Modal A100 dispatch now consumes GRAYSCALE_LUT_LUT_BITS env override per AA HIGH
# verdict (lut_bits=5 = 32-level cargo-cult unwind from PR #56 lut_bits=4 default 16-level).
# Default 8 preserves byte-stable backward-compat per trainer argparse default.
GRAYSCALE_LUT_LUT_BITS="${GRAYSCALE_LUT_LUT_BITS:-8}"
GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT="${GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT:-}"
GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS="${GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS:-}"

DISPATCH_INSTANCE_JOB_ID="${GRAYSCALE_LUT_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${GRAYSCALE_LUT_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-grayscale-lut] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: GRAYSCALE_LUT_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
    'video_path': '$GRAYSCALE_LUT_VIDEO_PATH',
    'upstream_dir': '$GRAYSCALE_LUT_UPSTREAM_DIR',
    'epochs': $GRAYSCALE_LUT_EPOCHS,
    'batch_size': $GRAYSCALE_LUT_BATCH_SIZE,
    'device': '$GRAYSCALE_LUT_DEVICE',
    'lut_bits': '$GRAYSCALE_LUT_LUT_BITS',
    'export_only_checkpoint': '$GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT',
    'soft_train_deadline_seconds': '$GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS',
    # Council Phase 5 prediction: 0.18 [contest-CUDA].
    # Source: .omx/research/grand_council_fields_medal_substrate_design_20260512.md
    # + recipe substrate_grayscale_lut_modal_a100_dispatch.yaml::predicted_score_target.
    # Band convention: [LOW, HIGH] in score-space (lower = better contest score).
    # Predicted target = 0.18; council 95% CI band [0.165, 0.195].
    'predicted_band': [0.165, 0.195],
    'predicted_basis': 'grand_council_fields_medal_substrate_design_20260512',
    'literature_anchor': 'Selfcomp PR #56 grayscale-LUT analog mask + Quantizr 0.33 anchor',
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
# All 5 TIER_1_OPERATOR_REQUIRED_FLAGS are threaded explicitly per Catalog #151.
TRAINER_EXTRA_ARGS=()
if [ -n "$GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT" ]; then
    TRAINER_EXTRA_ARGS+=(--export-only-checkpoint "$GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT")
fi
if [ -n "$GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS" ]; then
    TRAINER_EXTRA_ARGS+=(--soft-train-deadline-seconds "$GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS")
fi
log "stage_4_trainer_invoke_begin video=$GRAYSCALE_LUT_VIDEO_PATH epochs=$GRAYSCALE_LUT_EPOCHS device=$GRAYSCALE_LUT_DEVICE batch_size=$GRAYSCALE_LUT_BATCH_SIZE lut_bits=$GRAYSCALE_LUT_LUT_BITS export_only=${GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT:-none} soft_deadline=${GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS:-none}"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_grayscale_lut.py \
    --video-path "$GRAYSCALE_LUT_VIDEO_PATH" \
    --output-dir "$GRAYSCALE_LUT_OUTPUT_DIR" \
    --epochs "$GRAYSCALE_LUT_EPOCHS" \
    --batch-size "$GRAYSCALE_LUT_BATCH_SIZE" \
    --upstream-dir "$GRAYSCALE_LUT_UPSTREAM_DIR" \
    --device "$GRAYSCALE_LUT_DEVICE" \
    --lut-bits "$GRAYSCALE_LUT_LUT_BITS" \
    ${TRAINER_EXTRA_ARGS[@]+"${TRAINER_EXTRA_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
AUTH_EVAL_CUDA_JSON="$OUTPUT_DIR/contest_auth_eval_cuda.json"
AUTH_EVAL_CPU_JSON="$OUTPUT_DIR/contest_auth_eval_cpu.json"
ARCHIVE_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_CUDA_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_CUDA_JSON"
    log "LANE_GRAYSCALE_LUT_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_CUDA_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
elif [ -f "$AUTH_EVAL_CPU_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_CPU_JSON"
    log "LANE_GRAYSCALE_LUT_DONE [diagnostic-CPU] auth_eval=$AUTH_EVAL_CPU_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing paths=$AUTH_EVAL_CUDA_JSON,$AUTH_EVAL_CPU_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
