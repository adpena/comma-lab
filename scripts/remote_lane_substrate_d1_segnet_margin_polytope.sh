#!/bin/bash
# Remote lane script: D1 SegNet margin polytope encoder dispatch.
#
# Trainer: experiments/train_substrate_d1_segnet_margin_polytope.py
# Lane:    lane_d1_segnet_margin_polytope_encoder_20260514
# Memo:    .omx/research/deep_math_geometry_manifolds_synthesis_20260514.md §3.6 + §10 D1
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + Catalog
# #163: this script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function via the SOURCE_ONLY sentinel.
#
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable: this
# is a FIRST-ANCHOR research dispatch on CUDA only; the resulting tag is
# [contest-CUDA] single-axis (CPU axis required separately before promotion).
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_d1_segnet_margin_polytope_encoder_20260514"
TAG="${TAG:-substrate_d1_segnet_margin_polytope}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_d1_polytope_results}"

# Catalog #224 sister fix (parity with D4/Z3/Z4/Z5 lane scripts): set
# deterministic CUDA/runtime environment before bootstrap or trainer code can
# import torch and hit cuBLAS or DALI/NVDEC. Empirical anchor: D1 dispatch
# 2026-05-15T08:26:38Z (substrate_d1_segnet_margin_polytope_modal_t4_dispatch_
# 20260515T082638Z__smoke__50ep) reached upstream/evaluate.py and crashed at
# `nvml error (999): A nvml internal driver error occurred` inside DALI fn.
# experimental.inputs.video pipeline. The 4 sister lane scripts (D4/Z3/Z4/Z5)
# already export DALI_DISABLE_NVML; D1 was the missing wire-in.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Catalog #204 (PR95++ Modal smoke score-custody harden 2026-05-14):
# contest_auth_eval.py refuses score-grade evidence under /tmp. Modal workers
# run from /tmp/pact, so when MODAL_RUNTIME=1 + /modal_results exists, write
# archive/runtime/auth-eval work under the mounted result volume by default
# and let modal_train_lane.py harvest it. Local non-Modal runs keep the
# legacy $LOG_DIR/output path.
DISPATCH_INSTANCE_JOB_ID_TEMP="${D1_POLYTOPE_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-d1_polytope_unknown_job}}"
if [ -n "${D1_POLYTOPE_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$D1_POLYTOPE_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID_TEMP}/output"
else
    OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
fi
unset DISPATCH_INSTANCE_JOB_ID_TEMP

PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
D1_POLYTOPE_A1_ARCHIVE="${D1_POLYTOPE_A1_ARCHIVE:-$WORKSPACE/experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip}"
D1_POLYTOPE_VIDEO_PATH="${D1_POLYTOPE_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
D1_POLYTOPE_OUTPUT_DIR="$OUTPUT_DIR"
D1_POLYTOPE_EPOCHS="${D1_POLYTOPE_EPOCHS:-1000}"
D1_POLYTOPE_UPSTREAM_DIR="${D1_POLYTOPE_UPSTREAM_DIR:-$WORKSPACE/upstream}"
D1_POLYTOPE_DEVICE="${D1_POLYTOPE_DEVICE:-cuda}"
D1_POLYTOPE_PAYLOAD_BITS="${D1_POLYTOPE_PAYLOAD_BITS:-8000}"
D1_POLYTOPE_JACOBIAN_LIPSCHITZ="${D1_POLYTOPE_JACOBIAN_LIPSCHITZ:-20.0}"
D1_POLYTOPE_MARGIN_H="${D1_POLYTOPE_MARGIN_H:-96}"
D1_POLYTOPE_MARGIN_W="${D1_POLYTOPE_MARGIN_W:-128}"

DISPATCH_INSTANCE_JOB_ID="${D1_POLYTOPE_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${D1_POLYTOPE_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-d1-polytope] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: D1_POLYTOPE_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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
CLAIM_SUMMARY_JSON="$LOG_DIR/dispatch_claim_summary.json"
"$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
    --claims-path "$DISPATCH_CLAIMS_PATH" \
    --format json > "$CLAIM_SUMMARY_JSON" || {
    log "FATAL: claim summary failed for $DISPATCH_CLAIMS_PATH"
    exit 21
}
"$CLAIM_PYTHON" - "$CLAIM_SUMMARY_JSON" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" <<'PY'
import json
import sys
summary_path, lane_id, job_id = sys.argv[1:4]
payload = json.loads(open(summary_path, encoding="utf-8").read())
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(f"missing active claim lane={lane_id} job={job_id}", file=sys.stderr)
raise SystemExit(1)
PY
CLAIM_RC=$?
if [ "$CLAIM_RC" -ne 0 ]; then
    log "FATAL: no active dispatch claim for lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"
    exit 21
fi
log "stage_0_dispatch_claim_verified lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"

# Stage 0b: NVDEC probe (best-effort).
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

# Stage 1: bootstrap runtime deps (canonical, per Catalog #163).
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# shellcheck source=/dev/null
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

# Stage 1b: validate required input files PRE-dispatch (Catalog #152).
log "stage_1b_required_input_files_validate_begin"
"$PYBIN" "$WORKSPACE/tools/validate_dispatch_required_inputs.py" \
    --trainer "$WORKSPACE/experiments/train_substrate_d1_segnet_margin_polytope.py" \
    || {
    log "FATAL: Catalog #152 required input file validation failed; refusing dispatch"
    exit 25
}
log "stage_1b_required_input_files_validate_done"

# Stage 2: provenance + remote code parity.
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
    'a1_archive': '$D1_POLYTOPE_A1_ARCHIVE',
    'video_path': '$D1_POLYTOPE_VIDEO_PATH',
    'upstream_dir': '$D1_POLYTOPE_UPSTREAM_DIR',
    'epochs': $D1_POLYTOPE_EPOCHS,
    'device': '$D1_POLYTOPE_DEVICE',
    'polytope_payload_bits': $D1_POLYTOPE_PAYLOAD_BITS,
    'jacobian_lipschitz': $D1_POLYTOPE_JACOBIAN_LIPSCHITZ,
    'margin_h': $D1_POLYTOPE_MARGIN_H,
    'margin_w': $D1_POLYTOPE_MARGIN_W,
    'predicted_band': [0.181, 0.188],
    'predicted_basis': 'deep_math_geometry_manifolds_synthesis_20260514_d1_central_band',
}
import pathlib
pathlib.Path('$PROVENANCE').write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
"
log "stage_2_provenance_done"

# Stage 3: heartbeat watchdog.
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

# Stage 4: invoke trainer.
log "stage_4_trainer_invoke_begin epochs=$D1_POLYTOPE_EPOCHS device=$D1_POLYTOPE_DEVICE payload_bits=$D1_POLYTOPE_PAYLOAD_BITS L=$D1_POLYTOPE_JACOBIAN_LIPSCHITZ margin=${D1_POLYTOPE_MARGIN_H}x${D1_POLYTOPE_MARGIN_W}"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
# TIER_REQUIRED_FLAG_WAIVED_OK:--overlay-channel-policy:metadata_only_runtime_policy_sweep_default_rgb_preserved_per_comprehensive_bug_audit_cascade_20260526
# TIER_REQUIRED_FLAG_WAIVED_OK:--overlay-amplitude-scale:metadata_only_runtime_attenuation_sweep_default_1_0_no_attenuation_per_comprehensive_bug_audit_cascade_20260526
# TIER_REQUIRED_FLAG_WAIVED_OK:--overlay-sign-policy:metadata_only_runtime_sign_schedule_sweep_default_encoded_preserved_per_comprehensive_bug_audit_cascade_20260526
"$PYBIN" experiments/train_substrate_d1_segnet_margin_polytope.py \
    --a1-archive "$D1_POLYTOPE_A1_ARCHIVE" \
    --video-path "$D1_POLYTOPE_VIDEO_PATH" \
    --output-dir "$D1_POLYTOPE_OUTPUT_DIR" \
    --epochs "$D1_POLYTOPE_EPOCHS" \
    --upstream-dir "$D1_POLYTOPE_UPSTREAM_DIR" \
    --device "$D1_POLYTOPE_DEVICE" \
    --polytope-payload-bits "$D1_POLYTOPE_PAYLOAD_BITS" \
    --jacobian-lipschitz "$D1_POLYTOPE_JACOBIAN_LIPSCHITZ" \
    --margin-h "$D1_POLYTOPE_MARGIN_H" \
    --margin-w "$D1_POLYTOPE_MARGIN_W" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: optional CPU-axis auth eval (BOTH-AXES emission per CLAUDE.md).
AUTH_EVAL_JSON="$D1_POLYTOPE_OUTPUT_DIR/auth_eval.json"
ARCHIVE_PATH="$D1_POLYTOPE_OUTPUT_DIR/submission_dir/archive.zip"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    # Optional CPU eval if this is a Linux x86_64 host (otherwise the operator
    # wrapper schedules CPU separately).
    if [ -f "$ARCHIVE_PATH" ] && [ "${D1_POLYTOPE_ALSO_RUN_CPU:-0}" = "1" ]; then
        CPU_AUTH_EVAL_JSON="$D1_POLYTOPE_OUTPUT_DIR/auth_eval_cpu.json"
        log "stage_5_cpu_auth_eval_begin"
        "$PYBIN" "$WORKSPACE/experiments/contest_auth_eval.py" \
            --archive "$ARCHIVE_PATH" \
            --inflate-sh "$D1_POLYTOPE_OUTPUT_DIR/submission_dir/inflate.sh" \
            --upstream-dir "$D1_POLYTOPE_UPSTREAM_DIR" \
            --device cpu \
            --json-out "$CPU_AUTH_EVAL_JSON" \
            2>&1 | tee -a "$LOG_DIR/run.log" || {
            log "WARN: CPU auth eval failed; CUDA artifact still present at $AUTH_EVAL_JSON"
        }
        log "stage_5_cpu_auth_eval_done path=$CPU_AUTH_EVAL_JSON"
    fi
    log "LANE_D1_POLYTOPE_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before auth eval stage)"
fi

exit "$TRAIN_RC"
