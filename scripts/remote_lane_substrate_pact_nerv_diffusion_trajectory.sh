#!/bin/bash
# Remote lane script: substrate pact_nerv_diffusion_trajectory L0 SCAFFOLD smoke.
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_pact_nerv_diffusion_trajectory_l0_scaffold_20260520"
TAG="${TAG:-substrate_pact_nerv_diffusion_trajectory_l0_scaffold}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_pact_nerv_diffusion_trajectory_l0_scaffold_results}"

if [ -n "${PACT_NERV_DIFFUSION_TRAJECTORY_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$PACT_NERV_DIFFUSION_TRAJECTORY_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

PACT_NERV_DIFFUSION_TRAJECTORY_VIDEO_PATH="${PACT_NERV_DIFFUSION_TRAJECTORY_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
PACT_NERV_DIFFUSION_TRAJECTORY_OUTPUT_DIR="${PACT_NERV_DIFFUSION_TRAJECTORY_OUTPUT_DIR:-$OUTPUT_DIR}"
PACT_NERV_DIFFUSION_TRAJECTORY_EPOCHS="${PACT_NERV_DIFFUSION_TRAJECTORY_EPOCHS:-2}"
PACT_NERV_DIFFUSION_TRAJECTORY_UPSTREAM_DIR="${PACT_NERV_DIFFUSION_TRAJECTORY_UPSTREAM_DIR:-$WORKSPACE/upstream}"
PACT_NERV_DIFFUSION_TRAJECTORY_DEVICE="${PACT_NERV_DIFFUSION_TRAJECTORY_DEVICE:-cpu}"

PACT_NERV_DIFFUSION_TRAJECTORY_TRAINER_MODE="${PACT_NERV_DIFFUSION_TRAJECTORY_TRAINER_MODE:-}"
PACT_NERV_DIFFUSION_TRAJECTORY_SMOKE="${PACT_NERV_DIFFUSION_TRAJECTORY_SMOKE:-${SMOKE_ONLY:-1}}"
if [ "$PACT_NERV_DIFFUSION_TRAJECTORY_TRAINER_MODE" = "full" ]; then
    echo "[lane-pact-nerv-diffusion-trajectory-l0] WARNING: TRAINER_MODE=full but _full_main raises NotImplementedError. Forcing smoke." >&2
    PACT_NERV_DIFFUSION_TRAJECTORY_SMOKE="1"
elif [ "$PACT_NERV_DIFFUSION_TRAJECTORY_SMOKE" != "1" ]; then
    echo "[lane-pact-nerv-diffusion-trajectory-l0] WARNING: SMOKE=$PACT_NERV_DIFFUSION_TRAJECTORY_SMOKE; forcing smoke." >&2
    PACT_NERV_DIFFUSION_TRAJECTORY_SMOKE="1"
fi

DISPATCH_INSTANCE_JOB_ID="${PACT_NERV_DIFFUSION_TRAJECTORY_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"

log() { echo "[lane-pact-nerv-diffusion-trajectory-l0] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

rm -f upstream/videos/._*.mkv

if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: DISPATCH_INSTANCE_JOB_ID required"
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing"
    exit 21
fi

if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "WARN: NVDEC probe failed; continuing because L0 SCAFFOLD smoke is CPU-only"
    }
    log "stage_0b_nvdec_probe_done"
fi

if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing"
    exit 22
fi
# shellcheck source=/dev/null
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"

if declare -f bootstrap_runtime_deps > /dev/null 2>&1; then
    log "stage_1_bootstrap_runtime_deps_begin"
    bootstrap_runtime_deps
    log "stage_1_bootstrap_runtime_deps_done PYBIN=${PYBIN:-unset}"
else
    log "FATAL: bootstrap_runtime_deps function not found"
    exit 23
fi

if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    log "FATAL: PYBIN not set or not executable"
    exit 24
fi

if [ ! -f "$PACT_NERV_DIFFUSION_TRAJECTORY_VIDEO_PATH" ]; then
    log "FATAL: required input --video-path missing: $PACT_NERV_DIFFUSION_TRAJECTORY_VIDEO_PATH"
    exit 25
fi

log "stage_2_provenance_begin"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
"$PYBIN" -c "
import json, time
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'lane_id': '$LANE_ID',
    'dispatch_instance_job_id': '$DISPATCH_INSTANCE_JOB_ID',
    'dispatch_platform': '$DISPATCH_PLATFORM',
    'git_hash': '$GIT_HASH',
    'video_path': '$PACT_NERV_DIFFUSION_TRAJECTORY_VIDEO_PATH',
    'epochs': $PACT_NERV_DIFFUSION_TRAJECTORY_EPOCHS,
    'device': '$PACT_NERV_DIFFUSION_TRAJECTORY_DEVICE',
    'smoke': True,
    'posture': 'L0_SCAFFOLD',
    'literature_anchor': 'Rombach_2022_LDM_arXiv_2112.10752_Blattmann_2023_video_LDM',
    'score_claim': False,
    'promotion_eligible': False,
}
import pathlib
pathlib.Path('$PROVENANCE').write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
"
log "stage_2_provenance_done"

HEARTBEAT_PID=""
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

log "stage_4_trainer_invoke_begin video=$PACT_NERV_DIFFUSION_TRAJECTORY_VIDEO_PATH epochs=$PACT_NERV_DIFFUSION_TRAJECTORY_EPOCHS device=$PACT_NERV_DIFFUSION_TRAJECTORY_DEVICE smoke=1"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_pact_nerv_diffusion_trajectory.py \
    --video-path "$PACT_NERV_DIFFUSION_TRAJECTORY_VIDEO_PATH" \
    --output-dir "$PACT_NERV_DIFFUSION_TRAJECTORY_OUTPUT_DIR" \
    --epochs "$PACT_NERV_DIFFUSION_TRAJECTORY_EPOCHS" \
    --upstream-dir "$PACT_NERV_DIFFUSION_TRAJECTORY_UPSTREAM_DIR" \
    --device "$PACT_NERV_DIFFUSION_TRAJECTORY_DEVICE" \
    --smoke \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

SMOKE_PROVENANCE="$PACT_NERV_DIFFUSION_TRAJECTORY_OUTPUT_DIR/provenance.json"
if [ -f "$SMOKE_PROVENANCE" ]; then
    log "smoke_provenance_present path=$SMOKE_PROVENANCE"
    log "LANE_PACT_NERV_DIFFUSION_TRAJECTORY_L0_SCAFFOLD_DONE [scaffold-smoke-no-score-axis] provenance=$SMOKE_PROVENANCE rc=$TRAIN_RC"
else
    log "smoke_provenance_missing path=$SMOKE_PROVENANCE (trainer may have failed)"
fi

exit "$TRAIN_RC"
