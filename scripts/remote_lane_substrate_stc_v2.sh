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
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${STC_V2_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$STC_V2_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# === Catalog #152 driver-path-expectation discipline (2026-05-16 extension) ===
# Bug-class anchor: STC v2 Modal T4 dispatch fc-01KRSVKF9VEESQY2FS33FF4WDM
# (2026-05-17T02:17:51Z) rc=25 at 1.56s with "FATAL: Lane A anchor archive
# missing at /tmp/pact/experiments/results/lane_a_landed/archive_lane_a.zip".
# Root cause: experiments/modal_train_lane.py calls build_training_image with
# trainer_module_path=None (line 154), so the trainer's TIER_1_EXTRA_MOUNT_PATHS
# (Wave 1 fix at experiments/train_substrate_stc_v2.py:111) is NEVER consumed
# by the canonical mount builder. The driver must defensively probe multiple
# candidate locations (Modal mount → Modal workspace copy → Vast.ai workspace)
# and explicitly fail with diagnostic context if all are missing.
# Per CLAUDE.md "Apples-to-apples evidence discipline" + operator-prescribed
# Option 1 (driver-side defensive resolution + STRICT preflight Catalog #152
# extension).
resolve_required_input_modal_aware() {
    # Usage: resolve_required_input_modal_aware <env_var_name> <relative_path_under_repo>
    # Echoes the resolved absolute path or empty string if not found.
    local env_var="$1"
    local rel_path="$2"
    local override="${!env_var:-}"
    if [ -n "$override" ] && [ -f "$override" ]; then
        echo "$override"
        return 0
    fi
    # Probe in order of likelihood:
    #   1. $WORKSPACE/<rel> (canonical Vast.ai layout; also Modal post-copytree)
    #   2. /workspace/pact/<rel> (Modal read-only mount root; survives if
    #      shutil.copytree at modal_train_lane.py:195 didn't pick up an
    #      add_local_file-mounted file under an add_local_dir-ignored parent)
    #   3. /tmp/pact/<rel> (Modal writable workspace; explicit fallback if
    #      WORKSPACE env wasn't injected by env.sh source)
    local candidates=(
        "$WORKSPACE/$rel_path"
        "/workspace/pact/$rel_path"
        "/tmp/pact/$rel_path"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    echo ""
    return 1
}

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
STC_V2_VIDEO_PATH="${STC_V2_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
# Modal-aware anchor archive resolution per Catalog #152 driver-path-expectation
# discipline. Default still points at $WORKSPACE for Vast.ai layout; the
# Modal-aware probe happens in Stage 1b (defensive resolution).
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
#
# Modal-aware resolution per Catalog #152 driver-path-expectation extension
# (2026-05-16). The previous hardcoded $STC_V2_ANCHOR_ARCHIVE check failed
# on Modal because modal_train_lane.py:154 passes trainer_module_path=None
# to build_training_image, so the trainer's TIER_1_EXTRA_MOUNT_PATHS
# (Wave 1 fix) is INERT for generic Modal dispatches. The resolver
# probes $WORKSPACE → /workspace/pact → /tmp/pact in order and updates
# STC_V2_ANCHOR_ARCHIVE to the actual found path so downstream stages
# (trainer subprocess at Stage 4) see the right value.
log "stage_1b_anchor_resolve_begin requested=$STC_V2_ANCHOR_ARCHIVE workspace=$WORKSPACE modal_runtime=${MODAL_RUNTIME:-0}"
RESOLVED_ANCHOR="$(resolve_required_input_modal_aware STC_V2_ANCHOR_ARCHIVE "experiments/results/lane_a_landed/archive_lane_a.zip")"
if [ -z "$RESOLVED_ANCHOR" ]; then
    log "FATAL: Lane A anchor archive missing in every candidate location"
    log "       Probed:"
    log "         1. \$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip = $WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip"
    log "         2. /workspace/pact/experiments/results/lane_a_landed/archive_lane_a.zip (Modal read-only mount)"
    log "         3. /tmp/pact/experiments/results/lane_a_landed/archive_lane_a.zip (Modal writable workspace)"
    log "       (MODAL_RUNTIME=${MODAL_RUNTIME:-0}, override env STC_V2_ANCHOR_ARCHIVE=${STC_V2_ANCHOR_ARCHIVE:-unset})"
    log "       Diagnostic: STC v2 swap-archive requires renderer.bin + optimized_poses.pt from Lane A;"
    log "       the lane has no fallback path. Under Modal, this means the canonical"
    log "       mount manifest at tac.deploy.modal.mount_manifest.build_training_image did NOT stage"
    log "       the anchor archive — modal_train_lane.py passes trainer_module_path=None so"
    log "       TIER_1_EXTRA_MOUNT_PATHS is inert. Operator action: declare the file via"
    log "       a Modal-aware mount mechanism OR pre-stage via volume."
    log "       Bug-class anchor: fc-01KRSVKF9VEESQY2FS33FF4WDM rc=25 (2026-05-17T02:17:51Z)."
    exit 25
fi
if [ "$RESOLVED_ANCHOR" != "$STC_V2_ANCHOR_ARCHIVE" ]; then
    log "stage_1b_anchor_resolved_via_fallback original=$STC_V2_ANCHOR_ARCHIVE resolved=$RESOLVED_ANCHOR"
    STC_V2_ANCHOR_ARCHIVE="$RESOLVED_ANCHOR"
else
    log "stage_1b_anchor_resolved_at_expected_path path=$RESOLVED_ANCHOR"
fi
ANCHOR_BYTES=$(stat -c '%s' "$STC_V2_ANCHOR_ARCHIVE" 2>/dev/null || stat -f '%z' "$STC_V2_ANCHOR_ARCHIVE" 2>/dev/null || echo 0)
log "stage_1b_anchor_resolve_done path=$STC_V2_ANCHOR_ARCHIVE bytes=$ANCHOR_BYTES"

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
