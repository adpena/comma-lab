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

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_grayscale_lut_20260512"
TAG="${TAG:-substrate_grayscale_lut}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_grayscale_lut_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
GRAYSCALE_LUT_VIDEO_PATH="${GRAYSCALE_LUT_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
GRAYSCALE_LUT_OUTPUT_DIR="${GRAYSCALE_LUT_OUTPUT_DIR:-$OUTPUT_DIR}"
GRAYSCALE_LUT_EPOCHS="${GRAYSCALE_LUT_EPOCHS:-2000}"
GRAYSCALE_LUT_BATCH_SIZE="${GRAYSCALE_LUT_BATCH_SIZE:-16}"
GRAYSCALE_LUT_UPSTREAM_DIR="${GRAYSCALE_LUT_UPSTREAM_DIR:-$WORKSPACE/upstream}"
GRAYSCALE_LUT_DEVICE="${GRAYSCALE_LUT_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${GRAYSCALE_LUT_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${GRAYSCALE_LUT_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-grayscale-lut] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

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
log "stage_4_trainer_invoke_begin video=$GRAYSCALE_LUT_VIDEO_PATH epochs=$GRAYSCALE_LUT_EPOCHS device=$GRAYSCALE_LUT_DEVICE batch_size=$GRAYSCALE_LUT_BATCH_SIZE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_grayscale_lut.py \
    --video-path "$GRAYSCALE_LUT_VIDEO_PATH" \
    --output-dir "$GRAYSCALE_LUT_OUTPUT_DIR" \
    --epochs "$GRAYSCALE_LUT_EPOCHS" \
    --batch-size "$GRAYSCALE_LUT_BATCH_SIZE" \
    --upstream-dir "$GRAYSCALE_LUT_UPSTREAM_DIR" \
    --device "$GRAYSCALE_LUT_DEVICE" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
AUTH_EVAL_JSON="$OUTPUT_DIR/contest_auth_eval_cuda.json"
ARCHIVE_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    log "LANE_GRAYSCALE_LUT_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
