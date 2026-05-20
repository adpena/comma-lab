#!/bin/bash
# Remote lane script: substrate pact_nerv_ia3 L0 SCAFFOLD smoke dispatch.
#
# Trainer: experiments/train_substrate_pact_nerv_ia3.py (WAVE-3-PACT-NERV-IA3-L0-BUILD-STAGE-1
#          2026-05-20).
# Lane: lane_pact_nerv_ia3_l0_scaffold_20260520
# Recipe: .omx/operator_authorize_recipes/substrate_pact_nerv_ia3_modal_t4_dispatch.yaml
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
# Per Catalog #326 driver-mode-hardcode-fix: this driver consumes
# PACT_NERV_IA3_TRAINER_MODE > PACT_NERV_IA3_SMOKE > SMOKE_ONLY env var
# precedence chain. Recipe MUST set PACT_NERV_IA3_SMOKE=1 OR SMOKE_ONLY=1.
# Full mode is unreachable from this driver because the trainer's _full_main
# raises NotImplementedError.
set -euo pipefail

# === Catalog #244 / D1 incident anchor: canonical Modal/CUDA env hygiene ===
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_pact_nerv_ia3_l0_scaffold_20260520"
TAG="${TAG:-substrate_pact_nerv_ia3_l0_scaffold}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_pact_nerv_ia3_l0_scaffold_results}"

# Catalog #204 cross-driver expansion: Modal-aware OUTPUT_DIR resolution.
if [ -n "${PACT_NERV_IA3_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$PACT_NERV_IA3_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
PACT_NERV_IA3_VIDEO_PATH="${PACT_NERV_IA3_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
PACT_NERV_IA3_OUTPUT_DIR="${PACT_NERV_IA3_OUTPUT_DIR:-$OUTPUT_DIR}"
PACT_NERV_IA3_EPOCHS="${PACT_NERV_IA3_EPOCHS:-2}"
PACT_NERV_IA3_UPSTREAM_DIR="${PACT_NERV_IA3_UPSTREAM_DIR:-$WORKSPACE/upstream}"
PACT_NERV_IA3_DEVICE="${PACT_NERV_IA3_DEVICE:-cpu}"

# Catalog #326: multi-key mode env-var consumption with explicit precedence
# (PACT_NERV_IA3_TRAINER_MODE > PACT_NERV_IA3_SMOKE > SMOKE_ONLY > default).
# Recipe MUST set PACT_NERV_IA3_SMOKE=1 OR SMOKE_ONLY=1 OR
# PACT_NERV_IA3_TRAINER_MODE=smoke.
PACT_NERV_IA3_TRAINER_MODE="${PACT_NERV_IA3_TRAINER_MODE:-}"
PACT_NERV_IA3_SMOKE="${PACT_NERV_IA3_SMOKE:-${SMOKE_ONLY:-1}}"
if [ "$PACT_NERV_IA3_TRAINER_MODE" = "full" ]; then
    echo "[lane-pact-nerv-ia3-l0] WARNING: PACT_NERV_IA3_TRAINER_MODE=full; trainer _full_main raises NotImplementedError per Catalog #240. Forcing smoke." >&2
    PACT_NERV_IA3_SMOKE="1"
elif [ "$PACT_NERV_IA3_SMOKE" != "1" ]; then
    echo "[lane-pact-nerv-ia3-l0] WARNING: PACT_NERV_IA3_SMOKE=$PACT_NERV_IA3_SMOKE; trainer _full_main raises NotImplementedError per Catalog #240. Forcing smoke." >&2
    PACT_NERV_IA3_SMOKE="1"
fi

DISPATCH_INSTANCE_JOB_ID="${PACT_NERV_IA3_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${PACT_NERV_IA3_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"

log() { echo "[lane-pact-nerv-ia3-l0] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: PACT_NERV_IA3_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID required for active lane-claim verification"
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
if [ ! -f "$PACT_NERV_IA3_VIDEO_PATH" ]; then
    log "FATAL: required input --video-path does not exist on remote host: $PACT_NERV_IA3_VIDEO_PATH"
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
    'video_path': '$PACT_NERV_IA3_VIDEO_PATH',
    'upstream_dir': '$PACT_NERV_IA3_UPSTREAM_DIR',
    'epochs': $PACT_NERV_IA3_EPOCHS,
    'device': '$PACT_NERV_IA3_DEVICE',
    'smoke': True,
    'posture': 'L0_SCAFFOLD',
    'literature_anchor': 'Liu_2022_IA3_arXiv_2205.05638',
    'pact_nerv_stage': 'Stage_1_HYBRID_per_PACT_NERV_DESIGN_SYMPOSIUM',
    'predicted_band': None,
    'predicted_basis': 'L0_SCAFFOLD_no_empirical_anchor_pending_stage_1_dispatch_per_catalog_325',
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
log "stage_4_trainer_invoke_begin video=$PACT_NERV_IA3_VIDEO_PATH epochs=$PACT_NERV_IA3_EPOCHS device=$PACT_NERV_IA3_DEVICE smoke=1"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_pact_nerv_ia3.py \
    --video-path "$PACT_NERV_IA3_VIDEO_PATH" \
    --output-dir "$PACT_NERV_IA3_OUTPUT_DIR" \
    --epochs "$PACT_NERV_IA3_EPOCHS" \
    --upstream-dir "$PACT_NERV_IA3_UPSTREAM_DIR" \
    --device "$PACT_NERV_IA3_DEVICE" \
    --smoke \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record (no auth eval; L0 SCAFFOLD smoke).
SMOKE_PROVENANCE="$PACT_NERV_IA3_OUTPUT_DIR/provenance.json"
if [ -f "$SMOKE_PROVENANCE" ]; then
    log "smoke_provenance_present path=$SMOKE_PROVENANCE"
    log "LANE_PACT_NERV_IA3_L0_SCAFFOLD_DONE [scaffold-smoke-no-score-axis] provenance=$SMOKE_PROVENANCE rc=$TRAIN_RC"
else
    log "smoke_provenance_missing path=$SMOKE_PROVENANCE (trainer may have failed)"
fi

exit "$TRAIN_RC"
