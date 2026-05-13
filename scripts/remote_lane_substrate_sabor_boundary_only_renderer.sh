#!/bin/bash
# Remote lane script: SABOR boundary-only renderer first-anchor dispatch.
#
# Trainer: experiments/train_substrate_sabor_boundary_only_renderer.py
# Lane:    lane_sabor_boundary_only_renderer_substrate_20260513
# Audit:   .omx/research/sabor_boundary_audit_20260513.md
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
LANE_ID="lane_sabor_boundary_only_renderer_substrate_20260513"
TAG="${TAG:-substrate_sabor_boundary_only_renderer}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_sabor_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
SABOR_VIDEO_PATH="${SABOR_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
SABOR_OUTPUT_DIR="${SABOR_OUTPUT_DIR:-$OUTPUT_DIR}"
SABOR_EPOCHS="${SABOR_EPOCHS:-2000}"
SABOR_UPSTREAM_DIR="${SABOR_UPSTREAM_DIR:-$WORKSPACE/upstream}"
SABOR_DEVICE="${SABOR_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${SABOR_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${SABOR_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-sabor] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: SABOR_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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
    --trainer "$WORKSPACE/experiments/train_substrate_sabor_boundary_only_renderer.py" \
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
    'video_path': '$SABOR_VIDEO_PATH',
    'upstream_dir': '$SABOR_UPSTREAM_DIR',
    'epochs': $SABOR_EPOCHS,
    'device': '$SABOR_DEVICE',
    'predicted_band': [0.165, 0.185],
    'predicted_basis': 'sabor_boundary_audit_20260513_council_F_O1',
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
log "stage_4_trainer_invoke_begin epochs=$SABOR_EPOCHS device=$SABOR_DEVICE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_sabor_boundary_only_renderer.py \
    --video-path "$SABOR_VIDEO_PATH" \
    --output-dir "$SABOR_OUTPUT_DIR" \
    --epochs "$SABOR_EPOCHS" \
    --upstream-dir "$SABOR_UPSTREAM_DIR" \
    --device "$SABOR_DEVICE" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: optional CPU-axis auth eval (BOTH-AXES emission per CLAUDE.md).
AUTH_EVAL_JSON="$SABOR_OUTPUT_DIR/contest_auth_eval_cuda.json"
ARCHIVE_PATH="$SABOR_OUTPUT_DIR/archive.zip"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    if [ -f "$ARCHIVE_PATH" ] && [ "${SABOR_ALSO_RUN_CPU:-0}" = "1" ]; then
        CPU_AUTH_EVAL_JSON="$SABOR_OUTPUT_DIR/contest_auth_eval_cpu.json"
        log "stage_5_cpu_auth_eval_begin"
        "$PYBIN" "$WORKSPACE/experiments/contest_auth_eval.py" \
            --archive "$ARCHIVE_PATH" \
            --inflate-sh "$SABOR_OUTPUT_DIR/submission/inflate.sh" \
            --upstream-dir "$SABOR_UPSTREAM_DIR" \
            --device cpu \
            --json-out "$CPU_AUTH_EVAL_JSON" \
            2>&1 | tee -a "$LOG_DIR/run.log" || {
            log "WARN: CPU auth eval failed; CUDA artifact still present at $AUTH_EVAL_JSON"
        }
        log "stage_5_cpu_auth_eval_done path=$CPU_AUTH_EVAL_JSON"
    fi
    log "LANE_SABOR_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before archive emission)"
fi

exit "$TRAIN_RC"
