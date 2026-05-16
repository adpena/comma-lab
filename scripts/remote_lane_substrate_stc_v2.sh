#!/bin/bash
# Remote lane script: substrate stc_v2 CUDA disambiguator dispatch.
#
# Trainer: experiments/train_substrate_stc_v2.py
# Lane: lane_stc_clean_source_v2_substrate_build_20260516
# Design memo: .omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md
# Resurrection audit: .omx/research/resurrection_audit_20260516.md (Tier 1 #2)
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function. Per Catalog #163 the canonical
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel is set before sourcing.
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_STC_V2_DONE marker) per CLAUDE.md
# "Apples-to-apples evidence discipline" non-negotiable.
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
LANE_ID="lane_stc_clean_source_v2_substrate_build_20260516"
TAG="${TAG:-substrate_stc_v2}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_stc_clean_source_v2_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
STC_V2_VIDEO_PATH="${STC_V2_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
STC_V2_ANCHOR_ARCHIVE="${STC_V2_ANCHOR_ARCHIVE:-$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip}"
STC_V2_OUTPUT_DIR="${STC_V2_OUTPUT_DIR:-$OUTPUT_DIR}"
STC_V2_EPOCHS="${STC_V2_EPOCHS:-1}"
STC_V2_UPSTREAM_DIR="${STC_V2_UPSTREAM_DIR:-$WORKSPACE/upstream}"
STC_V2_DEVICE="${STC_V2_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${STC_V2_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${STC_V2_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-stc-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: STC_V2_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID required"
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

# Stage 1b: verify anchor archive is present (Catalog #152 required-input
# pre-dispatch validation sister check; the canonical operator_authorize.py
# routes through tools/validate_dispatch_required_inputs.py before reaching
# this script, so this is defense-in-depth).
if [ ! -f "$STC_V2_ANCHOR_ARCHIVE" ]; then
    log "FATAL: Lane A anchor archive missing at $STC_V2_ANCHOR_ARCHIVE"
    log "       STC v2 swap-archive requires renderer.bin + optimized_poses.pt"
    log "       from Lane A; the lane has no fallback path."
    exit 25
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
    'video_path': '$STC_V2_VIDEO_PATH',
    'anchor_archive': '$STC_V2_ANCHOR_ARCHIVE',
    'upstream_dir': '$STC_V2_UPSTREAM_DIR',
    'epochs': $STC_V2_EPOCHS,
    'device': '$STC_V2_DEVICE',
    'disambiguator_thresholds': {'reactivated_lt_kb': 200, 'competitive_lt_mb': 1, 'research_only_lt_mb': 5, 'falsification_ge_mb': 5},
    'predicted_basis': 'stc_clean_source_v2_substrate_build_20260516_section_2_3_dykstra_feasibility_check',
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
log "stage_4_trainer_invoke_begin video=$STC_V2_VIDEO_PATH epochs=$STC_V2_EPOCHS device=$STC_V2_DEVICE anchor=$STC_V2_ANCHOR_ARCHIVE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_stc_v2.py \
    --video-path "$STC_V2_VIDEO_PATH" \
    --anchor-archive "$STC_V2_ANCHOR_ARCHIVE" \
    --output-dir "$STC_V2_OUTPUT_DIR" \
    --epochs "$STC_V2_EPOCHS" \
    --upstream-dir "$STC_V2_UPSTREAM_DIR" \
    --device "$STC_V2_DEVICE" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
case "$STC_V2_DEVICE" in
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
ARCHIVE_PATH="$OUTPUT_DIR/archive.zip"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    log "LANE_STC_V2_DONE [contest-$AUTH_EVAL_AXIS_LABEL] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 9)"
fi

exit "$TRAIN_RC"
