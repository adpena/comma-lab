#!/bin/bash
# Remote lane script: substrate cascade_c_prime_frame_1_segnet_waterfill smoke-anchor dispatch.
#
# Trainer: experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py
# Lane: lane_cascade_c_prime_option_a_build_scaffold_20260526
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing
# directive 2026-05-15 ("all possible should be pulled into the decorator or
# similar reusable and shareable tools and helpers"), this script DELEGATES
# bootstrap to scripts/remote_archive_only_eval.sh's bootstrap_runtime_deps()
# function rather than re-implementing uv install, ffmpeg install, or torch
# CUDA pinning.
#
# Per Catalog #163 the canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# MUST be set before sourcing the bootstrap to prevent the sourced main flow
# (which expects a pre-built archive.zip) from running.
#
# Per Catalog #244 / D1 incident anchor: canonical Modal/CUDA env hygiene block
# IMMEDIATELY after `set -euo pipefail` so DALI / CUDA see the env vars before
# any Python import.
#
# Per Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under
# /modal_results so contest_auth_eval.py does not refuse score-grade evidence
# under /tmp per CLAUDE.md "Forbidden /tmp paths in any persisted artifact".
#
# Per Catalog #326 (sister of Z6/STC anti-pattern): mode env var consumption
# defaults SAFE; multi-key precedence CASCADE_C_PRIME_TRAINER_MODE > SMOKE_ONLY
# > default. Recipe `env_overrides` block sets CASCADE_C_PRIME_TRAINER_MODE
# explicitly to avoid driver-mode-mismatch dispatch bug class.
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] /
# [contest-CPU] per axis in the completion-log marker per CLAUDE.md
# "Apples-to-apples evidence discipline" non-negotiable.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# === Catalog #244 / D1 incident anchor (commit 611495f26): canonical Modal/CUDA env hygiene ===
# D1 dispatch 2026-05-15 (Modal T4 smoke) crashed at NVML 999 because the lane
# script did not export DALI_DISABLE_NVML=1 before DALI imported NVML.
# Future drivers should be auto-generated via tac.substrate_registry.driver_generator
# (which AUTO-EMITS the block from canonical constants in tac.deploy.modal.runtime).
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_cascade_c_prime_option_a_build_scaffold_20260526"
TAG="${TAG:-substrate_cascade_c_prime_frame_1_segnet_waterfill}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_cascade_c_prime_frame_1_segnet_waterfill_results}"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
CASCADE_C_PRIME_VIDEO_PATH="${CASCADE_C_PRIME_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
CASCADE_C_PRIME_UPSTREAM_DIR="${CASCADE_C_PRIME_UPSTREAM_DIR:-$WORKSPACE/upstream}"
CASCADE_C_PRIME_DEVICE="${CASCADE_C_PRIME_DEVICE:-cuda}"
CASCADE_C_PRIME_EPOCHS="${CASCADE_C_PRIME_EPOCHS:-1}"
CASCADE_C_PRIME_SEED="${CASCADE_C_PRIME_SEED:-20260526}"

# Catalog #326 mode env var consumption with multi-key precedence.
# Recipe env_overrides sets CASCADE_C_PRIME_TRAINER_MODE=full explicitly to
# avoid the Z6 Wave 2 4c dispatch bug class (driver default of "smoke" while
# recipe intent is "full"). FAIL-LOUD warning if neither key is set so the
# operator sees the implicit default.
if [ -n "${CASCADE_C_PRIME_TRAINER_MODE:-}" ]; then
    CASCADE_C_PRIME_MODE_RESOLVED="$CASCADE_C_PRIME_TRAINER_MODE"
elif [ -n "${SMOKE_ONLY:-}" ]; then
    if [ "$SMOKE_ONLY" = "1" ] || [ "$SMOKE_ONLY" = "true" ] || [ "$SMOKE_ONLY" = "yes" ]; then
        CASCADE_C_PRIME_MODE_RESOLVED="smoke"
    else
        CASCADE_C_PRIME_MODE_RESOLVED="full"
    fi
else
    CASCADE_C_PRIME_MODE_RESOLVED="full"
    echo "[lane-cascade-c-prime] WARN $(date -u +%FT%TZ) neither CASCADE_C_PRIME_TRAINER_MODE nor SMOKE_ONLY set; defaulting to 'full' per Catalog #326 sister anti-pattern fail-loud" >&2
fi

# Catalog #204 canonical 3-branch Modal-aware OUTPUT_DIR resolution:
# (a) explicit override via CASCADE_C_PRIME_OUTPUT_DIR; (b) Modal worker via
# /modal_results/${INSTANCE_JOB_ID}/output; (c) local/Vast.ai via $LOG_DIR/output.
if [ -n "${CASCADE_C_PRIME_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$CASCADE_C_PRIME_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

DISPATCH_INSTANCE_JOB_ID="${CASCADE_C_PRIME_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${CASCADE_C_PRIME_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-cascade-c-prime] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: CASCADE_C_PRIME_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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

# Stage 0b: NVDEC probe (per CLAUDE.md feedback_vastai_nvdec_host_variation).
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

# Stage 1: bootstrap runtime deps (canonical, per CLAUDE.md).
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
    log "FATAL: PYBIN not set or not executable after bootstrap (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Stage 1b: Catalog #152 + #204 sister required-input file multi-candidate probe.
# Per the canonical resolve_required_input_modal_aware pattern: probe Vast.ai
# layout, Modal read-only mount, and Modal writable workspace copy. Fail with
# diagnostic context if all candidates are missing.
if [ ! -f "$CASCADE_C_PRIME_VIDEO_PATH" ]; then
    log "stage_1b_video_probe_begin"
    for CANDIDATE in \
        "/workspace/pact/upstream/videos/0.mkv" \
        "/tmp/pact/upstream/videos/0.mkv" \
        "$WORKSPACE/upstream/videos/0.mkv"; do
        if [ -f "$CANDIDATE" ]; then
            CASCADE_C_PRIME_VIDEO_PATH="$CANDIDATE"
            log "stage_1b_video_resolved path=$CASCADE_C_PRIME_VIDEO_PATH"
            break
        fi
    done
    if [ ! -f "$CASCADE_C_PRIME_VIDEO_PATH" ]; then
        log "FATAL: video file missing at all candidate paths (Vast.ai/Modal-ro/Modal-rw)"
        exit 25
    fi
fi

# Stage 2: provenance + remote code parity (per CLAUDE.md "Remote code parity").
log "stage_2_provenance_begin"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" - "$PROVENANCE" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" "$DISPATCH_PLATFORM" "$GIT_HASH" "$GPU_NAME" "$DRIVER_VER" "$CASCADE_C_PRIME_VIDEO_PATH" "$CASCADE_C_PRIME_UPSTREAM_DIR" "$CASCADE_C_PRIME_DEVICE" "$CASCADE_C_PRIME_EPOCHS" "$CASCADE_C_PRIME_MODE_RESOLVED" "$CASCADE_C_PRIME_SEED" <<'PY'
import json, pathlib, sys, time
import torch

(
    provenance_path,
    lane_id,
    dispatch_instance_job_id,
    dispatch_platform,
    git_hash,
    gpu_name,
    driver_version,
    video_path,
    upstream_dir,
    device,
    epochs,
    mode,
    seed,
) = sys.argv[1:14]
payload = {
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "lane_id": lane_id,
    "dispatch_instance_job_id": dispatch_instance_job_id,
    "dispatch_platform": dispatch_platform,
    "git_hash": git_hash,
    "gpu_name": gpu_name,
    "driver_version": driver_version,
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "video_path": video_path,
    "upstream_dir": upstream_dir,
    "device": device,
    "epochs": int(epochs),
    "trainer_mode_resolved": mode,
    "seed": int(seed),
    # Per-substrate symposium PROCEED_WITH_REVISIONS predicted band:
    "predicted_band": [-0.058820, -0.006],
    "predicted_band_variance": "HIGH",
    "predicted_basis": "cascade_c_prime_synthesis_48_cell_sweep_atick_redlich_asymmetric_channel",
    "predicted_band_validation_status": "pending_post_training",
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
pathlib.Path(provenance_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
print("[provenance]", json.dumps(payload, indent=2, sort_keys=True))
PY
log "stage_2_provenance_done"

# Stage 3: heartbeat watchdog (every 5 min per CLAUDE.md).
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

# Stage 4: invoke trainer (MLX-first compress-only pass + paired-CUDA bridge).
# Trainer wrapper at experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py
# is sister subagent C's scope. THIS lane script invokes the wrapper canonically;
# if the wrapper is missing the dispatch fails-fast with a diagnostic exit.
TRAINER_PATH="$WORKSPACE/experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py"
if [ ! -f "$TRAINER_PATH" ]; then
    log "FATAL: trainer wrapper missing at $TRAINER_PATH (sister subagent C scope per per-substrate symposium revision #3)"
    exit 26
fi

log "stage_4_trainer_invoke_begin video=$CASCADE_C_PRIME_VIDEO_PATH epochs=$CASCADE_C_PRIME_EPOCHS device=$CASCADE_C_PRIME_DEVICE mode=$CASCADE_C_PRIME_MODE_RESOLVED seed=$CASCADE_C_PRIME_SEED"
TRAIN_START_UTC=$(date -u +%FT%TZ)
TRAIN_ARGS=(
    "$TRAINER_PATH"
    --video-path "$CASCADE_C_PRIME_VIDEO_PATH"
    --output-dir "$OUTPUT_DIR"
    --upstream-dir "$CASCADE_C_PRIME_UPSTREAM_DIR"
    --device "$CASCADE_C_PRIME_DEVICE"
    --epochs "$CASCADE_C_PRIME_EPOCHS"
    --seed "$CASCADE_C_PRIME_SEED"
)
if [ "$CASCADE_C_PRIME_MODE_RESOLVED" = "smoke" ]; then
    TRAIN_ARGS+=(--smoke)
fi

set +e
"$PYBIN" "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record per CLAUDE.md "Apples-to-apples evidence discipline".
case "$CASCADE_C_PRIME_DEVICE" in
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
ARCHIVE_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    log "LANE_CASCADE_C_PRIME_DONE [contest-$AUTH_EVAL_AXIS_LABEL] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before auth eval stage)"
fi

exit "$TRAIN_RC"
