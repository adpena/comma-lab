#!/bin/bash
# Remote lane script: substrate ATW codec V2 full-stack cooperative-receiver
# dispatch (Atick-Tishby-Wyner; V2 design memo 2026-05-16).
#
# Trainer: experiments/train_substrate_atw_codec_v2.py
# Lane: lane_atw_codec_v2_substrate_build_20260516
# Design memo: .omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" this script
# DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function. Per Catalog #163 the canonical
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel is set before sourcing.
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in the
# completion-log line (LANE_ATW_CODEC_V2_DONE marker) per CLAUDE.md
# "Apples-to-apples evidence discipline" non-negotiable.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# === Catalog #244 canonical Modal/CUDA env hygiene (auto-emitted block) ===
# D1 dispatch 2026-05-15 (Modal T4 smoke) crashed at NVML 999 because the lane
# script did not export DALI_DISABLE_NVML=1 before DALI imported NVML. Per
# CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing
# directive 2026-05-15 ("all possible should be pulled into the decorator or
# similar reusable and shareable tools and helpers and such"). Future drivers
# should be auto-generated via tac.substrate_registry.driver_generator (which
# AUTO-EMITS the block from canonical constants in tac.deploy.modal.runtime).
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_atw_codec_v2_substrate_build_20260516"
TAG="${TAG:-substrate_atw_codec_v2}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_atw_codec_v2_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${ATW_V2_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$ATW_V2_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
ATW_V2_VIDEO_PATH="${ATW_V2_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
ATW_V2_OUTPUT_DIR="${ATW_V2_OUTPUT_DIR:-$OUTPUT_DIR}"
ATW_V2_EPOCHS="${ATW_V2_EPOCHS:-200}"
ATW_V2_BATCH_SIZE="${ATW_V2_BATCH_SIZE:-4}"
ATW_V2_LR="${ATW_V2_LR:-5e-4}"
ATW_V2_UPSTREAM_DIR="${ATW_V2_UPSTREAM_DIR:-$WORKSPACE/upstream}"
ATW_V2_DEVICE="${ATW_V2_DEVICE:-cuda}"
ATW_V2_VARIANT="${ATW_V2_VARIANT:-B}"
ATW_V2_KAPPA_IB="${ATW_V2_KAPPA_IB:-0.0}"
ATW_V2_LAMBDA_WZ="${ATW_V2_LAMBDA_WZ:-1.0}"
ATW_V2_LAMBDA_PIXEL="${ATW_V2_LAMBDA_PIXEL:-0.0}"
ATW_V2_LAMBDA_DISTILL="${ATW_V2_LAMBDA_DISTILL:-0.1}"

DISPATCH_INSTANCE_JOB_ID="${ATW_V2_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${ATW_V2_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-atw-codec-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: ATW_V2_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID required"
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

CLAIM_VERIFIED=0
append_terminal_claim() {
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        status="completed_atw_codec_v2_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_atw_codec_v2_claim_verification_rc_${rc}"
    else
        status="failed_atw_codec_v2_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_atw_codec_v2" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$ATW_V2_OUTPUT_DIR" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: failed to append terminal dispatch claim status=$status"
    }
}

cleanup() {
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT

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
CLAIM_VERIFIED=1
log "stage_0_dispatch_claim_verified lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"

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
    'video_path': '$ATW_V2_VIDEO_PATH',
    'upstream_dir': '$ATW_V2_UPSTREAM_DIR',
    'epochs': $ATW_V2_EPOCHS,
    'batch_size': $ATW_V2_BATCH_SIZE,
    'lr': float('$ATW_V2_LR'),
    'variant': '$ATW_V2_VARIANT',
    'kappa_ib': float('$ATW_V2_KAPPA_IB'),
    'lambda_wz': float('$ATW_V2_LAMBDA_WZ'),
    'lambda_pixel': float('$ATW_V2_LAMBDA_PIXEL'),
    'lambda_distill': float('$ATW_V2_LAMBDA_DISTILL'),
    'device': '$ATW_V2_DEVICE',
    'predicted_band_status': 'NULL_pending_d4_probe_verdict_plus_dykstra_feasibility',
    'predicted_basis': 'atw_codec_v2_design_memo_section_18_d4_probe_conditional',
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

# Stage 4: invoke trainer (Catalog #172/#179 engineering primitives wired).
log "stage_4_trainer_invoke_begin video=$ATW_V2_VIDEO_PATH epochs=$ATW_V2_EPOCHS variant=$ATW_V2_VARIANT device=$ATW_V2_DEVICE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_atw_codec_v2.py \
    --video-path "$ATW_V2_VIDEO_PATH" \
    --output-dir "$ATW_V2_OUTPUT_DIR" \
    --epochs "$ATW_V2_EPOCHS" \
    --batch-size "$ATW_V2_BATCH_SIZE" \
    --lr "$ATW_V2_LR" \
    --upstream-dir "$ATW_V2_UPSTREAM_DIR" \
    --device "$ATW_V2_DEVICE" \
    --variant "$ATW_V2_VARIANT" \
    --kappa-ib "$ATW_V2_KAPPA_IB" \
    --lambda-wz "$ATW_V2_LAMBDA_WZ" \
    --lambda-pixel "$ATW_V2_LAMBDA_PIXEL" \
    --lambda-distill "$ATW_V2_LAMBDA_DISTILL" \
    --enable-autocast-fp16 \
    --enable-torch-compile \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
case "$ATW_V2_DEVICE" in
    cuda|gpu)
        AUTH_EVAL_AXIS="cuda"
        AUTH_EVAL_AXIS_LABEL="CUDA"
        ;;
    *)
        AUTH_EVAL_AXIS="cpu"
        AUTH_EVAL_AXIS_LABEL="CPU"
        ;;
esac
AUTH_EVAL_JSON="$OUTPUT_DIR/contest_auth_eval_${AUTH_EVAL_AXIS}.json"
ARCHIVE_ZIP_PATH="$OUTPUT_DIR/archive.zip"
PAYLOAD_0BIN_PATH="$OUTPUT_DIR/0.bin"
if [ "$TRAIN_RC" -eq 0 ] && [ -f "$AUTH_EVAL_JSON" ]; then
    if AUTH_EVAL_SCORE="$(PYTHONPATH="$WORKSPACE/src${PYTHONPATH:+:$PYTHONPATH}" "$PYBIN" - "$AUTH_EVAL_JSON" "contest_${AUTH_EVAL_AXIS}" <<'PY'
import json
import sys
from pathlib import Path

from tac.auth_eval_result import parse_auth_eval_score_claim

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
claim = parse_auth_eval_score_claim(
    payload,
    required_score_axis=sys.argv[2],
    require_component_recompute=True,
)
if claim is None:
    raise SystemExit("auth eval JSON is not a custody-valid score claim")
print(f"{claim.score:.12g}")
PY
)"; then
        log "LANE_ATW_CODEC_V2_DONE [contest-$AUTH_EVAL_AXIS_LABEL] score=$AUTH_EVAL_SCORE auth_eval=$AUTH_EVAL_JSON archive_zip=$ARCHIVE_ZIP_PATH payload_0bin=$PAYLOAD_0BIN_PATH rc=$TRAIN_RC variant=$ATW_V2_VARIANT"
    else
        log "LANE_ATW_CODEC_V2_DONE [training-artifact] auth_eval_not_custody_valid=$AUTH_EVAL_JSON archive_zip=$ARCHIVE_ZIP_PATH payload_0bin=$PAYLOAD_0BIN_PATH rc=$TRAIN_RC variant=$ATW_V2_VARIANT"
    fi
else
    log "LANE_ATW_CODEC_V2_DONE [training-artifact] auth_eval_missing_or_trainer_failed=$AUTH_EVAL_JSON archive_zip=$ARCHIVE_ZIP_PATH payload_0bin=$PAYLOAD_0BIN_PATH rc=$TRAIN_RC variant=$ATW_V2_VARIANT"
fi

exit "$TRAIN_RC"
