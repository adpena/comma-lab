#!/bin/bash
# Remote lane script: master-gradient extractor for fec6 archive on Modal T4
# (Linux x86_64 + Tesla T4 = contest-CUDA axis per CLAUDE.md "Submission auth
# eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable).
#
# Lane: lane_wave_3_op3_paid_master_gradient_anchor_20260520
# Recipe: .omx/operator_authorize_recipes/master_gradient_fec6_modal_t4_cuda_anchor_dispatch.yaml
# Trainer: tools/extract_master_gradient.py
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" (Catalog #243):
# delegates bootstrap to scripts/remote_archive_only_eval.sh's bootstrap_runtime_deps()
# which auto-pins INFLATE_TORCH_SPEC=torch==2.5.1+cu124 for CUDA 12.x drivers
# (Modal T4 instances run NVIDIA driver 535+ = CUDA 12.x), preventing the cu13
# silent-CPU-fallback trap per CLAUDE.md "Forbidden uv torch install without
# driver-version pin (the cu13-vs-cu124 trap)".
#
# This dispatch produces:
#   - .omx/state/master_gradient_anchors.jsonl  (canonical fcntl-locked anchor ledger;
#     adds the FIRST authoritative [contest-CUDA T4] row to the existing 10-row baseline)
#   - .omx/state/master_gradient_fec6_contest_cuda_t4_20260520.npy  (sidecar; aggregate
#     path = ~534 KB for ~178,517 bytes x 3 cols x float32)
#
# Smoke-before-full pattern (Catalog #167): single-shot extractor; smoke == full.
# min_smoke_gpu = T4 in the recipe pins the wrapper's smoke phase to T4 (not a
# cheaper class).
#
# Score-tagging: this dispatch does NOT produce a score claim. It produces a
# MasterGradient anchor at axis [contest-CUDA T4] per the operating point
# (d_seg, d_pose, R) measured on fec6's known CUDA-axis configuration.
set -euo pipefail

# === Catalog #244 / D1 incident anchor: canonical Modal/CUDA env hygiene ===
# DALI_DISABLE_NVML extincts the NVML 999 bug class that bit D1 6x in 24h
# (2026-05-15). CUBLAS_WORKSPACE_CONFIG enforces deterministic CUDA matmul
# (sister to CLAUDE.md "MPS auth eval is NOISE" / CUDA determinism). PYTORCH_
# CUDA_ALLOC_CONF=expandable_segments:True prevents the T4 allocator
# fragmentation class.
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
LANE_ID="lane_wave_3_op3_paid_master_gradient_anchor_20260520"
TAG="${TAG:-master_gradient_fec6_t4_cuda}"
# Catalog #204 CROSS-DRIVER 3-branch Modal-aware OUTPUT_DIR resolution:
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_master_gradient_fec6_t4_cuda_results}"
if [ -n "${MODAL_RUNTIME:-}" ] && [ -n "${MASTER_GRADIENT_DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-/modal_results/${MASTER_GRADIENT_DISPATCH_INSTANCE_JOB_ID}/output}"
elif [ -n "${OUTPUT_DIR:-}" ]; then
    : # operator-provided
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
MASTER_GRADIENT_ARCHIVE_PATH="${MASTER_GRADIENT_ARCHIVE_PATH:-$WORKSPACE/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip}"
MASTER_GRADIENT_INFLATE_PY_PATH="${MASTER_GRADIENT_INFLATE_PY_PATH:-$WORKSPACE/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py}"
MASTER_GRADIENT_UPSTREAM_DIR="${MASTER_GRADIENT_UPSTREAM_DIR:-$WORKSPACE/upstream}"
MASTER_GRADIENT_VIDEO_PATH="${MASTER_GRADIENT_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
MASTER_GRADIENT_OUTPUT_NPY="${MASTER_GRADIENT_OUTPUT_NPY:-$WORKSPACE/.omx/state/master_gradient_fec6_contest_cuda_t4_20260520.npy}"
MASTER_GRADIENT_AXIS="${MASTER_GRADIENT_AXIS:-[contest-CUDA]}"
MASTER_GRADIENT_DEVICE="${MASTER_GRADIENT_DEVICE:-cuda}"
# CUDA authoritative-axis gate per tools/extract_master_gradient.py:2177 requires
# n_pairs_used == n_pairs_total (full 600 pair contest set).
MASTER_GRADIENT_N_PAIRS_USED="${MASTER_GRADIENT_N_PAIRS_USED:-600}"
# Catalog #218 sister mini-batch chunk size for decoder forward + scorer forward + backward loop.
# Default 20 for T4 (14.56 GB): 600/20 = 30 chunks. Activation memory per chunk dominated by the
# eval_roundtrip 384x512→874x1164 bicubic upsample = 20 × 2 × 3 × 874 × 1164 × 4 bytes ≈ 488 MiB
# for the upsampled tensor (doubled for autograd grad storage). Plus FastViT-T12 PoseNet +
# EfficientNet-B2 SegNet forward activations. With the per-chunk GT transfer fix (avoids holding
# all 600 GT pairs on GPU), 20-chunk should fit T4 capacity. Recipe can override via
# env_overrides (e.g. 50-100 for A10G/A100, 0 for full-batch CPU smoke).
# 2026-05-20 anchor 2: chunk=100 fc-01KS36941EMJBZT0PYEADWYYW7 STILL OOM'd at eval_roundtrip
# bicubic upsample (450 MiB needed; 295 MiB free; baseline 14.27 GiB) because all 600 GT pairs
# were transferred to GPU upfront. Fix at tools/extract_master_gradient.py: per-chunk GT transfer
# + smaller default chunk size.
MASTER_GRADIENT_DECODER_FORWARD_BATCH_SIZE="${MASTER_GRADIENT_DECODER_FORWARD_BATCH_SIZE:-10}"
MASTER_GRADIENT_HARDWARE_SUBSTRATE="${MASTER_GRADIENT_HARDWARE_SUBSTRATE:-linux_x86_64_t4_modal}"

DISPATCH_INSTANCE_JOB_ID="${MASTER_GRADIENT_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${MASTER_GRADIENT_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"

# === Catalog #204 cross-driver expansion (2026-05-19) + Catalog #220 transient-evidence trap fix (2026-05-20) ===
# When running on Modal (MODAL_RUNTIME=1), the WORKSPACE-anchored path
# $WORKSPACE/.omx/state/... resolves to /tmp/pact/.omx/state/... on the worker
# (Modal mounts working tree under /tmp/pact/, NOT /workspace/pact/).
# tools/extract_master_gradient.py:2369-2373 REFUSES /tmp paths per CLAUDE.md
# "Forbidden /tmp paths in any persisted artifact" (Catalog #220 transient-
# evidence trap). Bug-class anchor: WAVE-3-OP3 dispatch fc-01KS2Z2WJQW532A9226JAVQM8Y
# (2026-05-20T15:11:22Z) failed rc=1 at 9.74s because the recipe-supplied
# MASTER_GRADIENT_OUTPUT_NPY=/workspace/pact/.omx/state/... resolved to
# /tmp/pact/.omx/state/... and triggered the extractor's /tmp refusal.
# Fix: redirect to /modal_results/<DISPATCH_INSTANCE_JOB_ID>/output/ (durable
# Modal volume; modal_train_lane.py harvests it back to local repo at completion).
# Sister of stack_of_stacks / stc_v2 / a1_plus_lapose driver fixes per the
# canonical Catalog #204 3-branch pattern.
# Override placed AFTER DISPATCH_INSTANCE_JOB_ID resolution to ensure non-empty
# under set -u + ${VAR:-} expansion semantics.
if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    # Always override on Modal to extinct the /tmp refusal class structurally,
    # regardless of whether the env_overrides block specified a /workspace/... default.
    MASTER_GRADIENT_OUTPUT_NPY="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output/master_gradient_fec6_contest_cuda_t4_20260520.npy"
fi
HEARTBEAT_PID=""

log() { echo "[lane-master-gradient-t4-cuda] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
# CUDA dispatch: the canonical bootstrap auto-pins INFLATE_TORCH_SPEC=torch==2.5.1+cu124
# for driver < 580 (CUDA 12.x; matches Modal T4) per CLAUDE.md "Forbidden uv torch
# install without driver-version pin (the cu13-vs-cu124 trap)".
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

# Stage 1b: ensure CUDA-side torch + pyav + brotli are importable AND
# torch.cuda.is_available() returns True (catches the cu13 silent-CPU-fallback
# class if the bootstrap auto-pin didn't fire).
log "stage_1b_pythondeps_check_begin"
"$CLAIM_PYTHON" -c "
import torch, av, brotli, numpy
cuda_available = torch.cuda.is_available()
print('torch', torch.__version__, 'cuda_available=', cuda_available)
if cuda_available:
    print('cuda_device_count=', torch.cuda.device_count())
    print('cuda_device_name=', torch.cuda.get_device_name(0))
    print('cuda_capability=', torch.cuda.get_device_capability(0))
import sys
sys.exit(0 if cuda_available else 23)
" 2>&1 | tee -a "$LOG_DIR/run.log" || {
    log "FATAL: required python deps (torch with CUDA, av, brotli, numpy) missing OR torch.cuda.is_available() returned False (silent CPU fallback class per CLAUDE.md 'MPS auth eval is NOISE' family)"
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
log "axis=$MASTER_GRADIENT_AXIS device=$MASTER_GRADIENT_DEVICE n_pairs_used=$MASTER_GRADIENT_N_PAIRS_USED hardware_substrate=$MASTER_GRADIENT_HARDWARE_SUBSTRATE decoder_forward_batch_size=$MASTER_GRADIENT_DECODER_FORWARD_BATCH_SIZE"

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
    --decoder-forward-batch-size "$MASTER_GRADIENT_DECODER_FORWARD_BATCH_SIZE" \
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
log "LANE_MASTER_GRADIENT_FEC6_T4_CUDA_DONE [contest-CUDA T4] sidecar=$MASTER_GRADIENT_OUTPUT_NPY size=$SIDECAR_SIZE"
exit 0
