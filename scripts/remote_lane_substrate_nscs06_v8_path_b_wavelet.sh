#!/bin/bash
# Remote lane script: substrate nscs06 v8 Path B wavelet-residual first-anchor dispatch.
#
# Trainer: experiments/train_substrate_nscs06_v8_path_b_wavelet.py
# Lane: lane_nscs06_v8_path_b_wavelet_residual_substrate_build_20260516
# Design memo: .omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function. Per Catalog #163 the canonical
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel is set before sourcing.
#
# Score-tagging: completion logs derive any contest-axis marker from the auth
# eval JSON's score_claim_valid + score_axis fields. Modal CPU advisory runs
# are training artifacts and must never be logged as contest-CUDA.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# === Catalog #244 canonical Modal/CUDA env hygiene (auto-emitted block) ===
# D1 dispatch 2026-05-15 (Modal T4 smoke) crashed at NVML 999 because the
# lane script did not export DALI_DISABLE_NVML=1 before DALI imported NVML.
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing
# directive 2026-05-15 ("all possible should be pulled into the decorator or
# similar reusable and shareable tools and helpers and such"). Future drivers
# should be auto-generated via tac.substrate_registry.driver_generator (which
# AUTO-EMITS the block from canonical constants in tac.deploy.modal.runtime).
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_nscs06_v8_path_b_wavelet_residual_substrate_build_20260516"
TAG="${TAG:-substrate_nscs06_v8_path_b_wavelet}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_nscs06_v8_path_b_wavelet_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${NSCS06_V8_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$NSCS06_V8_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
NSCS06_V8_VIDEO_PATH="${NSCS06_V8_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
NSCS06_V8_OUTPUT_DIR="${NSCS06_V8_OUTPUT_DIR:-$OUTPUT_DIR}"
NSCS06_V8_EPOCHS="${NSCS06_V8_EPOCHS:-1}"
NSCS06_V8_UPSTREAM_DIR="${NSCS06_V8_UPSTREAM_DIR:-$WORKSPACE/upstream}"
NSCS06_V8_DEVICE="${NSCS06_V8_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${NSCS06_V8_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${NSCS06_V8_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-nscs06-v8-path-b] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: NSCS06_V8_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID required"
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

# Stage 0b: NVDEC probe.
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

# Stage 1: bootstrap runtime deps (canonical).
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing"
    exit 22
fi
# Catalog #163 sentinel: prevent the sourced main flow from executing
# shellcheck source=/dev/null
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"

if declare -f bootstrap_runtime_deps > /dev/null 2>&1; then
    log "stage_1_bootstrap_runtime_deps_begin"
    bootstrap_runtime_deps
    log "stage_1_bootstrap_runtime_deps_done PYBIN=${PYBIN:-unset}"
else
    log "FATAL: bootstrap_runtime_deps function not found after sourcing"
    exit 23
fi

if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    log "FATAL: PYBIN not set or not executable after bootstrap (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Ensure pywavelets is installed (v8 runtime dep per HNeRV L4 + substrate
# package design memo Section 6.3). The canonical bootstrap installs torch
# + numpy + Pillow but does not include pywt by default; pip install if needed.
if ! "$PYBIN" -c "import pywt" 2>/dev/null; then
    log "stage_1b_pywt_install_begin"
    "$PYBIN" -m pip install --quiet "pywavelets>=1.4,<2.0" || {
        log "FATAL: pywavelets install failed"
        exit 25
    }
    log "stage_1b_pywt_install_done"
fi

# Stage 2: provenance + remote code parity.
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
    'video_path': '$NSCS06_V8_VIDEO_PATH',
    'upstream_dir': '$NSCS06_V8_UPSTREAM_DIR',
    'epochs': $NSCS06_V8_EPOCHS,
    'device': '$NSCS06_V8_DEVICE',
    'predicted_band': [15.0, 25.0],
    'predicted_band_variance': 'MEDIUM',
    'predicted_basis': 'nscs06_v8_path_b_wavelet_residual_design_memo_section_18_dykstra_feasible',
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

# Stage 4: invoke trainer (compress-only pass; no training loop).
log "stage_4_trainer_invoke_begin video=$NSCS06_V8_VIDEO_PATH epochs=$NSCS06_V8_EPOCHS device=$NSCS06_V8_DEVICE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_nscs06_v8_path_b_wavelet.py \
    --video-path "$NSCS06_V8_VIDEO_PATH" \
    --output-dir "$NSCS06_V8_OUTPUT_DIR" \
    --epochs "$NSCS06_V8_EPOCHS" \
    --upstream-dir "$NSCS06_V8_UPSTREAM_DIR" \
    --device "$NSCS06_V8_DEVICE" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
AUTH_EVAL_DEVICE_EFFECTIVE="${AUTH_EVAL_DEVICE:-$NSCS06_V8_DEVICE}"
AUTH_EVAL_DEVICE_NORMALIZED=$(printf '%s' "$AUTH_EVAL_DEVICE_EFFECTIVE" | tr '[:upper:]' '[:lower:]')
case "$AUTH_EVAL_DEVICE_NORMALIZED" in
    cuda|gpu)
        AUTH_EVAL_AXIS="cuda"
        ;;
    *)
        AUTH_EVAL_AXIS="cpu"
        ;;
esac
AUTH_EVAL_JSON="$OUTPUT_DIR/contest_auth_eval_${AUTH_EVAL_AXIS}.json"
ARCHIVE_ZIP_PATH="$OUTPUT_DIR/archive.zip"
PAYLOAD_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    AUTH_EVAL_SUMMARY=$("$PYBIN" - "$AUTH_EVAL_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.loads(open(path, encoding="utf-8").read())
axis = str(payload.get("score_axis") or "unknown")
device = str(payload.get("device") or payload.get("auth_eval_device") or "unknown")
score = payload.get("score")
score_claim = payload.get("score_claim") is True
score_claim_valid = payload.get("score_claim_valid") is True
promotion_eligible = payload.get("promotion_eligible") is True
if score_claim_valid and axis == "contest_cuda":
    marker = "[contest-CUDA]"
elif score_claim_valid and axis == "contest_cpu":
    marker = "[contest-CPU]"
elif score_claim_valid:
    marker = f"[{axis}]"
else:
    marker = "[training-artifact-no-score-claim]"
print(
    f"{marker} score={score} axis={axis} device={device} "
    f"score_claim={str(score_claim).lower()} "
    f"score_claim_valid={str(score_claim_valid).lower()} "
    f"promotion_eligible={str(promotion_eligible).lower()}"
)
PY
)
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON $AUTH_EVAL_SUMMARY"
    log "LANE_NSCS06_V8_PATH_B_DONE $AUTH_EVAL_SUMMARY auth_eval=$AUTH_EVAL_JSON archive_zip=$ARCHIVE_ZIP_PATH payload=$PAYLOAD_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 11)"
fi

exit "$TRAIN_RC"
