#!/bin/bash
# Remote lane script: master-gradient extractor for fec6 archive on Modal CPU.
#
# Lane: lane_op_routable_1_master_gradient_extractor_20260517
# Recipe: .omx/operator_authorize_recipes/master_gradient_fec6_modal_cpu_dispatch.yaml
# Trainer: tools/extract_master_gradient.py
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" (Catalog #243):
# delegates bootstrap to scripts/remote_archive_only_eval.sh's bootstrap_runtime_deps()
# rather than re-implementing uv install / ffmpeg install / torch CPU pinning.
#
# This dispatch produces:
#   - .omx/state/master_gradient_anchors.jsonl  (canonical fcntl-locked anchor ledger)
#   - .omx/state/master_gradient_fec6_contest_cpu_20260517.npy  (sidecar; ~534 KB
#     for ~178,517 bytes × 3 cols × float32)
#
# Smoke-before-full pattern (Catalog #167): this lane is short enough (~10-15 min CPU)
# that smoke ≡ full. The smoke phase is identical to the full phase; we declare
# min_smoke_gpu = CPU in the recipe so the wrapper does NOT downgrade to a cheaper
# class.
#
# Score-tagging: this dispatch does NOT produce a score claim. It produces a
# MasterGradient anchor at axis [contest-CPU] per the operating point
# (d_seg, d_pose, R) measured on fec6's known frontier configuration.
set -euo pipefail

# === Catalog #244 / D1 incident anchor: canonical Modal/CUDA env hygiene ===
# This is a CPU-only dispatch; the env exports are still required by sister
# tools (the canonical mount manifest expects them). DALI_DISABLE_NVML is the
# critical one for sister GPU lanes; setting it to 1 here is a no-op for CPU.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
# Catalog #152 WAVE-2 driver-path-expectation extension 2026-05-16: defensive
# multi-candidate resolution for Modal vs Vast.ai workspace layout.
resolve_required_input_modal_aware() {
    local var_name="$1"
    local current="${!var_name}"
    if [ -f "$current" ]; then
        return 0
    fi
    if [ -n "${MODAL_RUNTIME:-}" ]; then
        for candidate_root in /workspace/pact /tmp/pact; do
            local rel
            rel="${current#$WORKSPACE/}"
            if [ -f "$candidate_root/$rel" ]; then
                eval "$var_name=\"$candidate_root/$rel\""
                return 0
            fi
        done
    fi
    return 1
}

PYBIN="${PYBIN:-}"
LANE_ID="lane_op_routable_1_master_gradient_extractor_20260517"
TAG="${TAG:-master_gradient_fec6}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_master_gradient_fec6_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
MASTER_GRADIENT_ARCHIVE_PATH="${MASTER_GRADIENT_ARCHIVE_PATH:-$WORKSPACE/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip}"
MASTER_GRADIENT_INFLATE_PY_PATH="${MASTER_GRADIENT_INFLATE_PY_PATH:-$WORKSPACE/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py}"
MASTER_GRADIENT_UPSTREAM_DIR="${MASTER_GRADIENT_UPSTREAM_DIR:-$WORKSPACE/upstream}"
MASTER_GRADIENT_VIDEO_PATH="${MASTER_GRADIENT_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
MASTER_GRADIENT_OUTPUT_NPY="${MASTER_GRADIENT_OUTPUT_NPY:-$WORKSPACE/.omx/state/master_gradient_fec6_contest_cpu_20260517.npy}"
MASTER_GRADIENT_AXIS="${MASTER_GRADIENT_AXIS:-[contest-CPU]}"
MASTER_GRADIENT_DEVICE="${MASTER_GRADIENT_DEVICE:-cpu}"
MASTER_GRADIENT_N_PAIRS_USED="${MASTER_GRADIENT_N_PAIRS_USED:-8}"
MASTER_GRADIENT_HARDWARE_SUBSTRATE="${MASTER_GRADIENT_HARDWARE_SUBSTRATE:-linux_x86_64_modal_cpu}"

DISPATCH_INSTANCE_JOB_ID="${MASTER_GRADIENT_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${MASTER_GRADIENT_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"

# === Catalog #204 cross-driver expansion (2026-05-19) + Catalog #220 transient-evidence trap fix (2026-05-20) + WAVE-3-HARDEN-1 META extension ===
# When running on Modal (MODAL_RUNTIME=1), the WORKSPACE-anchored path
# $WORKSPACE/.omx/state/... resolves to /tmp/pact/.omx/state/... on the worker
# (Modal mounts working tree under /tmp/pact/, NOT /workspace/pact/).
# tools/extract_master_gradient.py:2369-2373 REFUSES /tmp paths per CLAUDE.md
# "Forbidden /tmp paths in any persisted artifact" (Catalog #220 transient-
# evidence trap). Sister bug-class anchor: WAVE-3-OP3 dispatch
# fc-01KS2Z2WJQW532A9226JAVQM8Y (2026-05-20T15:11:22Z) failed rc=1 at 9.74s on
# the CUDA T4 sister recipe (commit 75d39f32e landed the canonical fix there).
# This L2 sister CPU recipe inherits the identical bug class by construction
# because both recipes emit the same `/workspace/pact/.omx/state/...` path
# pattern through `env_overrides`.
# Fix: redirect to /modal_results/<DISPATCH_INSTANCE_JOB_ID>/output/ (durable
# Modal volume; modal_train_lane.py harvests it back to local repo at completion).
# Sister of stack_of_stacks / stc_v2 / a1_plus_lapose / master_gradient_t4_cuda
# driver fixes per the canonical Catalog #204 3-branch pattern.
# Override placed AFTER DISPATCH_INSTANCE_JOB_ID resolution to ensure non-empty
# under set -u + ${VAR:-} expansion semantics.
if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    # Always override on Modal to extinct the /tmp refusal class structurally,
    # regardless of whether the env_overrides block specified a /workspace/... default.
    MASTER_GRADIENT_OUTPUT_NPY="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output/master_gradient_fec6_contest_cpu_20260517.npy"
fi
HEARTBEAT_PID=""

log() { echo "[lane-master-gradient] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$(dirname "$MASTER_GRADIENT_OUTPUT_NPY")"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (mirrors sister substrate drivers).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: MASTER_GRADIENT_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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

# Stage 1: bootstrap runtime deps via canonical helper.
# CPU-only dispatch: torch CPU wheel (no CUDA), pyav for video decode, brotli for codec.
log "stage_1_bootstrap_begin"
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    # Source-only: get bootstrap_runtime_deps in scope without running its main flow.
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
    bootstrap_runtime_deps || {
        log "FATAL: bootstrap_runtime_deps failed"
        exit 22
    }
else
    log "WARN: canonical bootstrap helper missing; assuming environment is pre-provisioned"
fi
log "stage_1_bootstrap_done"

# Stage 1b: ensure CPU-side torch + pyav + brotli are importable.
log "stage_1b_pythondeps_check_begin"
"$CLAIM_PYTHON" -c "import torch, av, brotli, numpy; print('torch', torch.__version__, 'cpu_only=', not torch.cuda.is_available())" 2>&1 | tee -a "$LOG_DIR/run.log" || {
    log "FATAL: required python deps (torch, av, brotli, numpy) missing"
    exit 23
}
log "stage_1b_pythondeps_check_done"

# Stage 2: required-input file validation per Catalog #152 + #244 Modal-IGNORED extension.
log "stage_2_required_inputs_begin"
if ! resolve_required_input_modal_aware MASTER_GRADIENT_ARCHIVE_PATH; then
    log "FATAL: archive missing at $MASTER_GRADIENT_ARCHIVE_PATH (Modal-IGNORED required input; declare in TIER_1_EXTRA_MOUNT_PATHS or REQUIRED_INPUT_MODAL_STAGED_OK waiver)"
    exit 25
fi
if ! resolve_required_input_modal_aware MASTER_GRADIENT_INFLATE_PY_PATH; then
    log "FATAL: inflate.py missing at $MASTER_GRADIENT_INFLATE_PY_PATH"
    exit 25
fi
if ! resolve_required_input_modal_aware MASTER_GRADIENT_VIDEO_PATH; then
    log "FATAL: video missing at $MASTER_GRADIENT_VIDEO_PATH"
    exit 25
fi
if [ ! -d "$MASTER_GRADIENT_UPSTREAM_DIR" ]; then
    log "FATAL: upstream dir missing at $MASTER_GRADIENT_UPSTREAM_DIR"
    exit 25
fi
if [ ! -f "$MASTER_GRADIENT_UPSTREAM_DIR/models/posenet.safetensors" ]; then
    log "FATAL: posenet.safetensors missing at $MASTER_GRADIENT_UPSTREAM_DIR/models/"
    exit 25
fi
if [ ! -f "$MASTER_GRADIENT_UPSTREAM_DIR/models/segnet.safetensors" ]; then
    log "FATAL: segnet.safetensors missing at $MASTER_GRADIENT_UPSTREAM_DIR/models/"
    exit 25
fi
log "stage_2_required_inputs_done"

# Stage 3: heartbeat (per CLAUDE.md "Remote code parity - non-negotiable" #3).
heartbeat_loop() {
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ)" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
}
heartbeat_loop &
HEARTBEAT_PID=$!
trap "kill $HEARTBEAT_PID 2>/dev/null || true" EXIT

# Stage 4: invoke the extractor.
log "stage_4_extractor_begin"
log "archive=$MASTER_GRADIENT_ARCHIVE_PATH inflate=$MASTER_GRADIENT_INFLATE_PY_PATH"
log "axis=$MASTER_GRADIENT_AXIS device=$MASTER_GRADIENT_DEVICE n_pairs_used=$MASTER_GRADIENT_N_PAIRS_USED"

set +e
PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE/tools:${PYTHONPATH:-}" \
    "$CLAIM_PYTHON" -u "$WORKSPACE/tools/extract_master_gradient.py" \
    --archive "$MASTER_GRADIENT_ARCHIVE_PATH" \
    --inflate-py "$MASTER_GRADIENT_INFLATE_PY_PATH" \
    --upstream-dir "$MASTER_GRADIENT_UPSTREAM_DIR" \
    --video-path "$MASTER_GRADIENT_VIDEO_PATH" \
    --axis "$MASTER_GRADIENT_AXIS" \
    --output-npy "$MASTER_GRADIENT_OUTPUT_NPY" \
    --device "$MASTER_GRADIENT_DEVICE" \
    --n-pairs-used "$MASTER_GRADIENT_N_PAIRS_USED" \
    --hardware-substrate "$MASTER_GRADIENT_HARDWARE_SUBSTRATE" \
    --call-id "$DISPATCH_INSTANCE_JOB_ID" \
    --verbose \
    2>&1 | tee -a "$LOG_DIR/extractor.log"
EXTRACT_RC=${PIPESTATUS[0]}
set -e

log "stage_4_extractor_done rc=$EXTRACT_RC"

# Stage 5: completion + provenance.
if [ $EXTRACT_RC -ne 0 ]; then
    log "FATAL: extractor failed rc=$EXTRACT_RC"
    exit "$EXTRACT_RC"
fi

if [ ! -f "$MASTER_GRADIENT_OUTPUT_NPY" ]; then
    log "FATAL: sidecar .npy missing at $MASTER_GRADIENT_OUTPUT_NPY"
    exit 30
fi

SIDECAR_SIZE=$(stat -c %s "$MASTER_GRADIENT_OUTPUT_NPY" 2>/dev/null || stat -f %z "$MASTER_GRADIENT_OUTPUT_NPY")
log "sidecar emitted: $MASTER_GRADIENT_OUTPUT_NPY ($SIDECAR_SIZE bytes)"

# Write provenance JSON
cat > "$PROVENANCE" <<JSON
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "axis": "$MASTER_GRADIENT_AXIS",
  "device": "$MASTER_GRADIENT_DEVICE",
  "archive_path": "$MASTER_GRADIENT_ARCHIVE_PATH",
  "sidecar_path": "$MASTER_GRADIENT_OUTPUT_NPY",
  "sidecar_size_bytes": $SIDECAR_SIZE,
  "n_pairs_used": $MASTER_GRADIENT_N_PAIRS_USED,
  "hardware_substrate": "$MASTER_GRADIENT_HARDWARE_SUBSTRATE",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "completed_at_utc": "$(date -u +%FT%TZ)",
  "extract_rc": $EXTRACT_RC
}
JSON
log "provenance written: $PROVENANCE"

# Completion marker per the canonical pattern (see sister substrate drivers).
log "LANE_MASTER_GRADIENT_FEC6_DONE [contest-CPU] sidecar=$MASTER_GRADIENT_OUTPUT_NPY size=$SIDECAR_SIZE"
exit 0
