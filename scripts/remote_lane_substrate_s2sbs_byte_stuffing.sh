#!/bin/bash
# Remote lane script: S2SBS stride-2-stem byte-stuffing substrate.
#
# Trainer: experiments/train_substrate_s2sbs_byte_stuffing.py
# Lane:    lane_s2sbs_stride2_byte_stuffing_substrate_20260513
# Audit:   .omx/research/s2sbs_blindspot_audit_20260513.md
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" +
# Catalog #163: bootstrap is DELEGATED to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function via the SOURCE_ONLY sentinel.
#
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": this is a
# FIRST-ANCHOR research dispatch on CUDA only; the CPU axis is a separate
# operator step.

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
LANE_ID="lane_s2sbs_stride2_byte_stuffing_substrate_20260513"
TAG="${TAG:-substrate_s2sbs_byte_stuffing}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_s2sbs_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${S2SBS_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$S2SBS_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

S2SBS_VIDEO_PATH="${S2SBS_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
S2SBS_OUTPUT_DIR="${S2SBS_OUTPUT_DIR:-$OUTPUT_DIR}"
S2SBS_EPOCHS="${S2SBS_EPOCHS:-2000}"
S2SBS_UPSTREAM_DIR="${S2SBS_UPSTREAM_DIR:-$WORKSPACE/upstream}"
S2SBS_DEVICE="${S2SBS_DEVICE:-cuda}"
S2SBS_PAYLOAD_BYTES_PER_PAIR="${S2SBS_PAYLOAD_BYTES_PER_PAIR:-32}"
S2SBS_DELTA_AMP_UINT8="${S2SBS_DELTA_AMP_UINT8:-0.75}"
S2SBS_SMOKE="${S2SBS_SMOKE:-1}"

DISPATCH_INSTANCE_JOB_ID="${S2SBS_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${S2SBS_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-s2sbs] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: S2SBS_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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
import json, sys
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
        log "FATAL: NVDEC probe failed; refusing dispatch"
        exit 2
    }
    log "stage_0b_nvdec_probe_done"
else
    log "WARN: scripts/probe_nvdec.sh missing - skipping NVDEC early-fail probe"
fi

# Stage 1: bootstrap runtime deps (canonical, Catalog #163).
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
    log "FATAL: bootstrap_runtime_deps not found after sourcing"
    exit 23
fi
if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    log "FATAL: PYBIN not set or not executable after bootstrap"
    exit 24
fi

# Stage 1b: validate required input files PRE-dispatch (Catalog #152).
log "stage_1b_required_input_files_validate_begin"
"$PYBIN" "$WORKSPACE/tools/validate_dispatch_required_inputs.py" \
    --trainer "$WORKSPACE/experiments/train_substrate_s2sbs_byte_stuffing.py" \
    || {
    log "FATAL: Catalog #152 required input file validation failed"
    exit 25
}
log "stage_1b_required_input_files_validate_done"

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
    'video_path': '$S2SBS_VIDEO_PATH',
    'upstream_dir': '$S2SBS_UPSTREAM_DIR',
    'epochs': $S2SBS_EPOCHS,
    'device': '$S2SBS_DEVICE',
    'payload_bytes_per_pair': $S2SBS_PAYLOAD_BYTES_PER_PAIR,
	    'delta_amp_uint8': $S2SBS_DELTA_AMP_UINT8,
	    'smoke_only': '$S2SBS_SMOKE' in ('1', 'true', 'TRUE', 'yes', 'YES'),
	    'predicted_band': [0.168, 0.188],
	    'predicted_basis': 's2sbs_blindspot_audit_20260513',
	    'research_only': True,
	    'score_claim': False,
	    'score_claim_valid': False,
	    'promotion_eligible': False,
	    'rank_or_kill_eligible': False,
	    'ready_for_exact_eval_dispatch': False,
	}
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
log "stage_4_trainer_invoke_begin epochs=$S2SBS_EPOCHS device=$S2SBS_DEVICE payload=$S2SBS_PAYLOAD_BYTES_PER_PAIR delta=$S2SBS_DELTA_AMP_UINT8"
TRAIN_START_UTC=$(date -u +%FT%TZ)
TRAINER_SMOKE_ARGS=()
case "$S2SBS_SMOKE" in
    1|true|TRUE|yes|YES)
        TRAINER_SMOKE_ARGS+=(--smoke)
        ;;
    *)
        log "FATAL: S2SBS full training is not implemented; set S2SBS_SMOKE=1 until the full trainer lands"
        exit 26
        ;;
esac
set +e
"$PYBIN" experiments/train_substrate_s2sbs_byte_stuffing.py \
    --video-path "$S2SBS_VIDEO_PATH" \
    --output-dir "$S2SBS_OUTPUT_DIR" \
    --epochs "$S2SBS_EPOCHS" \
    --upstream-dir "$S2SBS_UPSTREAM_DIR" \
    --device "$S2SBS_DEVICE" \
    --payload-bytes-per-pair "$S2SBS_PAYLOAD_BYTES_PER_PAIR" \
    --delta-amp-uint8 "$S2SBS_DELTA_AMP_UINT8" \
    "${TRAINER_SMOKE_ARGS[@]}" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

log "LANE_S2SBS_DONE [smoke-only-no-score-claim] rc=$TRAIN_RC output_dir=$S2SBS_OUTPUT_DIR"
exit "$TRAIN_RC"
