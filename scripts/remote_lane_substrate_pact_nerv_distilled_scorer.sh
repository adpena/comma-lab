#!/bin/bash
# Remote lane script: substrate pact_nerv_distilled_scorer L0 SCAFFOLD smoke.
#
# Trainer: experiments/train_substrate_pact_nerv_distilled_scorer.py
# Lane: lane_pact_nerv_distilled_scorer_l0_scaffold_20260520
# Recipe: .omx/operator_authorize_recipes/substrate_pact_nerv_distilled_scorer_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + Catalog
# #163 sentinel + Catalog #326 driver-mode-hardcode-fix + Catalog #244 NVML.
set -euo pipefail

# === Catalog #244: canonical Modal/CUDA env hygiene ===
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_pact_nerv_distilled_scorer_l0_scaffold_20260520"
TAG="${TAG:-substrate_pact_nerv_distilled_scorer_l0_scaffold}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_pact_nerv_distilled_scorer_l0_scaffold_results}"

# Catalog #204 cross-driver expansion: Modal-aware OUTPUT_DIR resolution.
if [ -n "${PACT_NERV_DISTILLED_SCORER_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$PACT_NERV_DISTILLED_SCORER_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
PACT_NERV_DISTILLED_SCORER_VIDEO_PATH="${PACT_NERV_DISTILLED_SCORER_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
PACT_NERV_DISTILLED_SCORER_OUTPUT_DIR="${PACT_NERV_DISTILLED_SCORER_OUTPUT_DIR:-$OUTPUT_DIR}"
PACT_NERV_DISTILLED_SCORER_EPOCHS="${PACT_NERV_DISTILLED_SCORER_EPOCHS:-2}"
PACT_NERV_DISTILLED_SCORER_UPSTREAM_DIR="${PACT_NERV_DISTILLED_SCORER_UPSTREAM_DIR:-$WORKSPACE/upstream}"
PACT_NERV_DISTILLED_SCORER_DEVICE="${PACT_NERV_DISTILLED_SCORER_DEVICE:-cpu}"

# Catalog #326: multi-key mode env-var consumption with explicit precedence.
PACT_NERV_DISTILLED_SCORER_TRAINER_MODE="${PACT_NERV_DISTILLED_SCORER_TRAINER_MODE:-}"
PACT_NERV_DISTILLED_SCORER_SMOKE="${PACT_NERV_DISTILLED_SCORER_SMOKE:-${SMOKE_ONLY:-1}}"
if [ "$PACT_NERV_DISTILLED_SCORER_TRAINER_MODE" = "full" ]; then
    echo "[lane-pact-nerv-distilled-scorer-l0] WARNING: TRAINER_MODE=full but trainer _full_main raises NotImplementedError per Catalog #240. Forcing smoke." >&2
    PACT_NERV_DISTILLED_SCORER_SMOKE="1"
elif [ "$PACT_NERV_DISTILLED_SCORER_SMOKE" != "1" ]; then
    echo "[lane-pact-nerv-distilled-scorer-l0] WARNING: SMOKE=$PACT_NERV_DISTILLED_SCORER_SMOKE; trainer _full_main raises NotImplementedError. Forcing smoke." >&2
    PACT_NERV_DISTILLED_SCORER_SMOKE="1"
fi

DISPATCH_INSTANCE_JOB_ID="${PACT_NERV_DISTILLED_SCORER_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"

log() { echo "[lane-pact-nerv-distilled-scorer-l0] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: DISPATCH_INSTANCE_JOB_ID required for active lane-claim verification"
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing"
    exit 21
fi

# Stage 0b: NVDEC probe.
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "WARN: NVDEC probe failed; continuing because L0 SCAFFOLD smoke is CPU-only"
    }
    log "stage_0b_nvdec_probe_done"
fi

# Stage 1: bootstrap runtime deps (canonical).
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel per Catalog #163.
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
    log "FATAL: PYBIN not set or not executable (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Stage 1b: required-input file existence pre-dispatch validation (Catalog #152).
if [ ! -f "$PACT_NERV_DISTILLED_SCORER_VIDEO_PATH" ]; then
    log "FATAL: required input --video-path does not exist: $PACT_NERV_DISTILLED_SCORER_VIDEO_PATH"
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
    'video_path': '$PACT_NERV_DISTILLED_SCORER_VIDEO_PATH',
    'upstream_dir': '$PACT_NERV_DISTILLED_SCORER_UPSTREAM_DIR',
    'epochs': $PACT_NERV_DISTILLED_SCORER_EPOCHS,
    'device': '$PACT_NERV_DISTILLED_SCORER_DEVICE',
    'smoke': True,
    'posture': 'L0_SCAFFOLD',
    'literature_anchor': 'Hinton_Vinyals_Dean_2015_arXiv_1503.02531_KL_T2_distillation',
    'score_claim': False,
    'promotion_eligible': False,
}
import pathlib
pathlib.Path('$PROVENANCE').write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
"
log "stage_2_provenance_done"

# Stage 3: heartbeat (every 5 min).
HEARTBEAT_PID=""
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

# Stage 4: invoke trainer in --smoke mode.
log "stage_4_trainer_invoke_begin video=$PACT_NERV_DISTILLED_SCORER_VIDEO_PATH epochs=$PACT_NERV_DISTILLED_SCORER_EPOCHS device=$PACT_NERV_DISTILLED_SCORER_DEVICE smoke=1"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_pact_nerv_distilled_scorer.py \
    --video-path "$PACT_NERV_DISTILLED_SCORER_VIDEO_PATH" \
    --output-dir "$PACT_NERV_DISTILLED_SCORER_OUTPUT_DIR" \
    --epochs "$PACT_NERV_DISTILLED_SCORER_EPOCHS" \
    --upstream-dir "$PACT_NERV_DISTILLED_SCORER_UPSTREAM_DIR" \
    --device "$PACT_NERV_DISTILLED_SCORER_DEVICE" \
    --smoke \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record (no auth eval; L0 SCAFFOLD smoke).
SMOKE_PROVENANCE="$PACT_NERV_DISTILLED_SCORER_OUTPUT_DIR/provenance.json"
if [ -f "$SMOKE_PROVENANCE" ]; then
    log "smoke_provenance_present path=$SMOKE_PROVENANCE"
    log "LANE_PACT_NERV_DISTILLED_SCORER_L0_SCAFFOLD_DONE [scaffold-smoke-no-score-axis] provenance=$SMOKE_PROVENANCE rc=$TRAIN_RC"
else
    log "smoke_provenance_missing path=$SMOKE_PROVENANCE (trainer may have failed)"
fi

exit "$TRAIN_RC"
