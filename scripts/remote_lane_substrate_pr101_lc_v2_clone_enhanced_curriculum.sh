#!/bin/bash
# Remote lane script: PR95++ enhanced-curriculum first-anchor dispatch.
#
# Trainer: experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py
# Lane: lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513
#
# This script is intentionally a thin actuator: shared bootstrap remains in
# scripts/remote_archive_only_eval.sh, lane coordination remains in
# tools/claim_lane_dispatch.py, and the model/curriculum logic remains in the
# trainer + tac.substrates.pr101_lc_v2_clone.curriculum_enhanced.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_pr95plus_results}"
OUTPUT_DIR="${PR95PLUS_OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

PR95PLUS_VIDEO_PATH="${PR95PLUS_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
PR95PLUS_UPSTREAM_DIR="${PR95PLUS_UPSTREAM_DIR:-$WORKSPACE/upstream}"
PR95PLUS_DEVICE="${PR95PLUS_DEVICE:-cuda}"
PR95PLUS_CURRICULUM="${PR95PLUS_CURRICULUM:-pr95_enhanced}"
PR95PLUS_EPOCHS_MULTIPLIER="${PR95PLUS_EPOCHS_MULTIPLIER:-1.0}"
PR95PLUS_SEED="${PR95PLUS_SEED:-0}"
PR95PLUS_GPU="${PR95PLUS_GPU:-${MODAL_GPU:-A100}}"
PR95PLUS_CODEBOOK_PATH="${PR95PLUS_CODEBOOK_PATH:-}"
PR95PLUS_BATCH_SIZE="${PR95PLUS_BATCH_SIZE:-8}"

DISPATCH_INSTANCE_JOB_ID="${PR95PLUS_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${PR95PLUS_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-pr95plus] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"
rm -f upstream/videos/._*.mkv

if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: PR95PLUS_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

CLAIM_SUMMARY_JSON="$LOG_DIR/dispatch_claim_summary.json"
"$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
    --claims-path "$DISPATCH_CLAIMS_PATH" \
    --format json > "$CLAIM_SUMMARY_JSON"
"$CLAIM_PYTHON" -c 'import json, sys
summary_path, lane_id, job_id = sys.argv[1:4]
payload = json.loads(open(summary_path, encoding="utf-8").read())
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(f"missing active claim lane={lane_id} job={job_id}", file=sys.stderr)
raise SystemExit(1)' "$CLAIM_SUMMARY_JSON" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID"
log "stage_0_dispatch_claim_verified lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"

if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh"
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
    log "FATAL: PYBIN not set or not executable after bootstrap"
    exit 24
fi

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" - "$PROVENANCE" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" "$DISPATCH_PLATFORM" "$GIT_HASH" "$GPU_NAME" "$DRIVER_VER" "$PR95PLUS_VIDEO_PATH" "$PR95PLUS_UPSTREAM_DIR" "$PR95PLUS_DEVICE" "$PR95PLUS_CURRICULUM" "$PR95PLUS_EPOCHS_MULTIPLIER" "$PR95PLUS_GPU" "$PR95PLUS_CODEBOOK_PATH" <<'PY'
import json, pathlib, sys, time
import torch

(
    provenance_path,
    lane_id,
    dispatch_instance_job_id,
    dispatch_platform,
    git_hash,
    gpu_name,
    driver_version,
    video_path,
    upstream_dir,
    device,
    curriculum,
    epochs_multiplier,
    gpu,
    codebook_path,
) = sys.argv[1:15]
payload = {
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "lane_id": lane_id,
    "dispatch_instance_job_id": dispatch_instance_job_id,
    "dispatch_platform": dispatch_platform,
    "git_hash": git_hash,
    "gpu_name": gpu_name,
    "driver_version": driver_version,
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "video_path": video_path,
    "upstream_dir": upstream_dir,
    "device": device,
    "curriculum": curriculum,
    "epochs_multiplier": float(epochs_multiplier),
    "gpu": gpu,
    "codebook_path": codebook_path,
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
pathlib.Path(provenance_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
print("[provenance]", json.dumps(payload, indent=2, sort_keys=True))
PY

(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

log "stage_4_trainer_invoke_begin curriculum=$PR95PLUS_CURRICULUM multiplier=$PR95PLUS_EPOCHS_MULTIPLIER device=$PR95PLUS_DEVICE"
set +e
"$PYBIN" experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py \
    --video-path "$PR95PLUS_VIDEO_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --upstream-dir "$PR95PLUS_UPSTREAM_DIR" \
    --device "$PR95PLUS_DEVICE" \
    --curriculum "$PR95PLUS_CURRICULUM" \
    --epochs-multiplier "$PR95PLUS_EPOCHS_MULTIPLIER" \
    --batch-size "$PR95PLUS_BATCH_SIZE" \
    --seed "$PR95PLUS_SEED" \
    --gpu "$PR95PLUS_GPU" \
    --codebook-path "$PR95PLUS_CODEBOOK_PATH" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
log "stage_4_trainer_invoke_done rc=$TRAIN_RC"

if [ "$TRAIN_RC" -ne 0 ]; then
    "$PYBIN" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --lane-id "$LANE_ID" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --agent codex \
        --status failed_remote_trainer \
        --notes "PR95++ enhanced trainer failed rc=$TRAIN_RC" \
        --force || true
    exit "$TRAIN_RC"
fi

"$PYBIN" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
    --claims-path "$DISPATCH_CLAIMS_PATH" \
    --lane-id "$LANE_ID" \
    --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
    --platform "$DISPATCH_PLATFORM" \
    --agent codex \
    --status completed_remote_trainer \
    --notes "PR95++ enhanced trainer completed; score_claim=false until auth eval harvest" \
    --force || true
log "completed"
