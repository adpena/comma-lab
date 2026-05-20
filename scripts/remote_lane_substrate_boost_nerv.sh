#!/bin/bash
# Remote lane script: substrate boost_nerv L0 SCAFFOLD smoke dispatch.
#
# Trainer: experiments/train_substrate_boost_nerv.py (WAVE-3-NERV-LITERATURE-L0-RESCOPED
#          2026-05-20).
# Lane: lane_boost_nerv_l0_scaffold_20260520
# Recipe: .omx/operator_authorize_recipes/substrate_boost_nerv_modal_t4_dispatch.yaml
#
# This is the L0 SCAFFOLD smoke driver. The matching recipe declares
# dispatch_enabled:false + research_only:true so this driver should ONLY be
# invoked via the canonical operator-authorize harness for diagnostic smoke;
# the trainer's _full_main path raises NotImplementedError per Catalog #240.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# MUST be set before sourcing the bootstrap.
#
# Per Catalog #326 driver-mode-hardcode-fix: this driver consumes BOOST_NERV_SMOKE
# env var. Recipe MUST set BOOST_NERV_SMOKE=1. Full mode is unreachable from
# this driver because the trainer's _full_main raises NotImplementedError.
set -euo pipefail

# === Catalog #244 / D1 incident anchor: canonical Modal/CUDA env hygiene ===
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_boost_nerv_l0_scaffold_20260520"
TAG="${TAG:-substrate_boost_nerv_l0_scaffold}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_boost_nerv_l0_scaffold_results}"

# Catalog #204 cross-driver expansion: Modal-aware OUTPUT_DIR resolution.
if [ -n "${BOOST_NERV_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$BOOST_NERV_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
BOOST_NERV_VIDEO_PATH="${BOOST_NERV_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
BOOST_NERV_OUTPUT_DIR="${BOOST_NERV_OUTPUT_DIR:-$OUTPUT_DIR}"
BOOST_NERV_EPOCHS="${BOOST_NERV_EPOCHS:-2}"
BOOST_NERV_UPSTREAM_DIR="${BOOST_NERV_UPSTREAM_DIR:-$WORKSPACE/upstream}"
BOOST_NERV_DEVICE="${BOOST_NERV_DEVICE:-cpu}"

# Catalog #326: multi-key mode env-var consumption. Recipe MUST set BOOST_NERV_SMOKE=1.
BOOST_NERV_SMOKE="${BOOST_NERV_SMOKE:-${SMOKE_ONLY:-1}}"
if [ "$BOOST_NERV_SMOKE" != "1" ]; then
    echo "[lane-boost-nerv-l0] WARNING: BOOST_NERV_SMOKE=$BOOST_NERV_SMOKE; trainer _full_main raises NotImplementedError per Catalog #240. Forcing smoke." >&2
    BOOST_NERV_SMOKE="1"
fi

DISPATCH_INSTANCE_JOB_ID="${BOOST_NERV_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${BOOST_NERV_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"

log() { echo "[lane-boost-nerv-l0] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: BOOST_NERV_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID required for active lane-claim verification"
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py"
    exit 21
fi

# Stage 0b: NVDEC probe (per CLAUDE.md `feedback_vastai_nvdec_host_variation`).
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "WARN: NVDEC probe failed; continuing because L0 SCAFFOLD smoke is CPU-only"
    }
    log "stage_0b_nvdec_probe_done"
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

# Stage 1b: required-input file existence pre-dispatch validation (Catalog #152).
if [ ! -f "$BOOST_NERV_VIDEO_PATH" ]; then
    log "FATAL: required input --video-path does not exist on remote host: $BOOST_NERV_VIDEO_PATH"
    exit 25
fi

# Stage 2: provenance + remote code parity.
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
    'video_path': '$BOOST_NERV_VIDEO_PATH',
    'upstream_dir': '$BOOST_NERV_UPSTREAM_DIR',
    'epochs': $BOOST_NERV_EPOCHS,
    'device': '$BOOST_NERV_DEVICE',
    'smoke': True,
    'posture': 'L0_SCAFFOLD',
    'literature_anchor': 'Liu_ECCV_2024_BoostNeRV',
    'fit_ranking': 'HIGH_FIT_5_of_5',
    'predicted_band': None,
    'predicted_basis': 'L0_SCAFFOLD_no_empirical_anchor_pending_phase_2_council_per_catalog_325',
    'score_claim': False,
    'promotion_eligible': False,
}
import pathlib
pathlib.Path('$PROVENANCE').write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
"
log "stage_2_provenance_done"

# Stage 3: heartbeat (every 5 min per CLAUDE.md).
HEARTBEAT_PID=""
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

# Stage 4: invoke trainer in --smoke mode (full-mode unreachable).
log "stage_4_trainer_invoke_begin video=$BOOST_NERV_VIDEO_PATH epochs=$BOOST_NERV_EPOCHS device=$BOOST_NERV_DEVICE smoke=1"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_boost_nerv.py \
    --video-path "$BOOST_NERV_VIDEO_PATH" \
    --output-dir "$BOOST_NERV_OUTPUT_DIR" \
    --epochs "$BOOST_NERV_EPOCHS" \
    --upstream-dir "$BOOST_NERV_UPSTREAM_DIR" \
    --device "$BOOST_NERV_DEVICE" \
    --smoke \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record (no auth eval; L0 SCAFFOLD smoke).
SMOKE_PROVENANCE="$BOOST_NERV_OUTPUT_DIR/provenance.json"
if [ -f "$SMOKE_PROVENANCE" ]; then
    log "smoke_provenance_present path=$SMOKE_PROVENANCE"
    log "LANE_BOOST_NERV_L0_SCAFFOLD_DONE [scaffold-smoke-no-score-axis] provenance=$SMOKE_PROVENANCE rc=$TRAIN_RC"
else
    log "smoke_provenance_missing path=$SMOKE_PROVENANCE (trainer may have failed)"
fi

exit "$TRAIN_RC"
