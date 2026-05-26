#!/bin/bash
# Remote lane script: A1 + wavelet residual sidecar retarget dispatch.
#
# Trainer: experiments/train_substrate_a1_plus_wavelet_residual.py
# Lane:    lane_a1_plus_wavelet_residual_retarget_20260513
# Memo:    .omx/research/meta_council_decision_attribution_audit_20260513.md
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
LANE_ID="lane_a1_plus_wavelet_residual_retarget_20260513"
TAG="${TAG:-substrate_a1_plus_wavelet_residual}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_a1_plus_wavelet_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${A1_PLUS_WAVELET_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$A1_PLUS_WAVELET_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# === Catalog #152 driver-path-expectation discipline (Wave 2, 2026-05-16) ===
# Sister of STC v2 driver fix (lane lane_stc_v2_driver_fix_catalog_152_
# driver_path_extension_20260516). modal_train_lane.py passes
# trainer_module_path=None to build_training_image, so a trainer's
# TIER_1_EXTRA_MOUNT_PATHS is INERT for generic Modal dispatchers. Driver
# scripts that consume required-input files under the Modal-IGNORED
# `experiments/results/**` subtree MUST defensively resolve paths.
resolve_required_input_modal_aware() {
    local env_var="$1"
    local rel_path="$2"
    local override="${!env_var:-}"
    if [ -n "$override" ] && [ -f "$override" ]; then
        echo "$override"
        return 0
    fi
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
A1_PLUS_WAVELET_A1_ARCHIVE="${A1_PLUS_WAVELET_A1_ARCHIVE:-$WORKSPACE/experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip}"
A1_PLUS_WAVELET_PAIR_MANIFEST="${A1_PLUS_WAVELET_PAIR_MANIFEST:-$WORKSPACE/.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lapose_motion_atom_manifest_fixture.json}"
A1_PLUS_WAVELET_VIDEO_PATH="${A1_PLUS_WAVELET_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
A1_PLUS_WAVELET_OUTPUT_DIR="${A1_PLUS_WAVELET_OUTPUT_DIR:-$OUTPUT_DIR}"
A1_PLUS_WAVELET_EPOCHS="${A1_PLUS_WAVELET_EPOCHS:-2000}"
A1_PLUS_WAVELET_UPSTREAM_DIR="${A1_PLUS_WAVELET_UPSTREAM_DIR:-$WORKSPACE/upstream}"
A1_PLUS_WAVELET_DEVICE="${A1_PLUS_WAVELET_DEVICE:-cuda}"
A1_PLUS_WAVELET_COEFF_RANK="${A1_PLUS_WAVELET_COEFF_RANK:-1}"
A1_PLUS_WAVELET_MAX_PAIRS="${A1_PLUS_WAVELET_MAX_PAIRS:-16}"

DISPATCH_INSTANCE_JOB_ID="${A1_PLUS_WAVELET_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${A1_PLUS_WAVELET_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-a1-plus-wavelet] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: A1_PLUS_WAVELET_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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

# Stage 1b: resolve + validate required input files PRE-dispatch (Catalog #152).
log "stage_1b_required_input_files_resolve_begin requested_a1_archive=$A1_PLUS_WAVELET_A1_ARCHIVE"
RESOLVED_A1_ARCHIVE="$(resolve_required_input_modal_aware A1_PLUS_WAVELET_A1_ARCHIVE "experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip")"
if [ -z "$RESOLVED_A1_ARCHIVE" ]; then
    log "FATAL: A1 archive missing in Modal-aware candidate locations"
    log "       Probed: $WORKSPACE/experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip"
    log "               /workspace/pact/experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip"
    log "               /tmp/pact/experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip"
    exit 25
fi
A1_PLUS_WAVELET_A1_ARCHIVE="$RESOLVED_A1_ARCHIVE"
export A1_PLUS_WAVELET_A1_ARCHIVE
log "stage_1b_required_input_files_resolve_done a1_archive=$A1_PLUS_WAVELET_A1_ARCHIVE"
log "stage_1b_required_input_files_validate_begin"
"$PYBIN" "$WORKSPACE/tools/validate_dispatch_required_inputs.py" \
    --trainer "$WORKSPACE/experiments/train_substrate_a1_plus_wavelet_residual.py" \
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
    'a1_archive': '$A1_PLUS_WAVELET_A1_ARCHIVE',
    'pair_manifest': '$A1_PLUS_WAVELET_PAIR_MANIFEST',
    'video_path': '$A1_PLUS_WAVELET_VIDEO_PATH',
    'upstream_dir': '$A1_PLUS_WAVELET_UPSTREAM_DIR',
    'epochs': $A1_PLUS_WAVELET_EPOCHS,
    'device': '$A1_PLUS_WAVELET_DEVICE',
    'coeff_rank': $A1_PLUS_WAVELET_COEFF_RANK,
    'max_pairs': $A1_PLUS_WAVELET_MAX_PAIRS,
    'predicted_band': [0.187, 0.194],
    'predicted_basis': 'meta_council_decision_attribution_audit_20260513_central_band',
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
log "stage_4_trainer_invoke_begin epochs=$A1_PLUS_WAVELET_EPOCHS device=$A1_PLUS_WAVELET_DEVICE coeff_rank=$A1_PLUS_WAVELET_COEFF_RANK max_pairs=$A1_PLUS_WAVELET_MAX_PAIRS"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
# TIER_REQUIRED_FLAG_WAIVED_OK:--enable-gt-scorer-cache:F3_GTScorerCache_optimization_pending_subagent_wave_per_canonical_decision_NOT_required_for_smoke_or_first_full_run_per_comprehensive_bug_audit_cascade_20260526
"$PYBIN" experiments/train_substrate_a1_plus_wavelet_residual.py \
    --a1-archive "$A1_PLUS_WAVELET_A1_ARCHIVE" \
    --pair-manifest "$A1_PLUS_WAVELET_PAIR_MANIFEST" \
    --video-path "$A1_PLUS_WAVELET_VIDEO_PATH" \
    --output-dir "$A1_PLUS_WAVELET_OUTPUT_DIR" \
    --epochs "$A1_PLUS_WAVELET_EPOCHS" \
    --upstream-dir "$A1_PLUS_WAVELET_UPSTREAM_DIR" \
    --device "$A1_PLUS_WAVELET_DEVICE" \
    --coeff-rank "$A1_PLUS_WAVELET_COEFF_RANK" \
    --max-pairs "$A1_PLUS_WAVELET_MAX_PAIRS" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: optional CPU-axis auth eval (BOTH-AXES emission per CLAUDE.md).
AUTH_EVAL_JSON="$A1_PLUS_WAVELET_OUTPUT_DIR/auth_eval.json"
ARCHIVE_PATH="$A1_PLUS_WAVELET_OUTPUT_DIR/submission_dir/archive.zip"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    # Optional CPU eval if this is a Linux x86_64 host (otherwise the operator
    # wrapper schedules CPU separately).
    if [ -f "$ARCHIVE_PATH" ] && [ "${A1_PLUS_WAVELET_ALSO_RUN_CPU:-0}" = "1" ]; then
        CPU_AUTH_EVAL_JSON="$A1_PLUS_WAVELET_OUTPUT_DIR/auth_eval_cpu.json"
        log "stage_5_cpu_auth_eval_begin"
        "$PYBIN" "$WORKSPACE/experiments/contest_auth_eval.py" \
            --archive "$ARCHIVE_PATH" \
            --inflate-sh "$A1_PLUS_WAVELET_OUTPUT_DIR/submission_dir/inflate.sh" \
            --upstream-dir "$A1_PLUS_WAVELET_UPSTREAM_DIR" \
            --device cpu \
            --json-out "$CPU_AUTH_EVAL_JSON" \
            2>&1 | tee -a "$LOG_DIR/run.log" || {
            log "WARN: CPU auth eval failed; CUDA artifact still present at $AUTH_EVAL_JSON"
        }
        log "stage_5_cpu_auth_eval_done path=$CPU_AUTH_EVAL_JSON"
    fi
    log "LANE_A1_PLUS_WAVELET_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 13)"
fi

exit "$TRAIN_RC"
