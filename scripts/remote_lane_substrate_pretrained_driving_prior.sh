#!/bin/bash
# Remote lane script: substrate pretrained_driving_prior first-anchor dispatch.
#
# Trainer: experiments/train_substrate_pretrained_driving_prior.py (L0 scaffold)
# Lane: lane_pretrained_driving_prior_lane_scaffold_20260513
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# MUST be set before sourcing the bootstrap to prevent the sourced main flow
# from running.
#
# Memo refs:
#   - .omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md
#   - feedback_pretrained_driving_prior_lane_scaffold_LANDED_20260513.md (post-landing)
#
# Score-tagging: completion logs report the auth-eval JSON's own score_axis /
# lane_tag. Modal T4 training is diagnostic unless the auth-eval contract
# itself says score_axis=contest_cuda and score_claim_valid=true.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
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
LANE_ID="${DPP_LANE_ID:-lane_pretrained_driving_prior_lane_scaffold_20260513}"
TAG="${TAG:-substrate_pretrained_driving_prior}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_pretrained_driving_prior_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${DPP_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$DPP_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
DPP_VIDEO_PATH="${DPP_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
DPP_OUTPUT_DIR="${DPP_OUTPUT_DIR:-$OUTPUT_DIR}"
DPP_EPOCHS="${DPP_EPOCHS:-2000}"
DPP_BATCH_SIZE="${DPP_BATCH_SIZE:-1}"
DPP_UPSTREAM_DIR="${DPP_UPSTREAM_DIR:-$WORKSPACE/upstream}"
DPP_DEVICE="${DPP_DEVICE:-cuda}"
DPP_DATASET_NAME="${DPP_DATASET_NAME:-comma2k19}"
DPP_ENABLE_AUTOCAST_FP16="${DPP_ENABLE_AUTOCAST_FP16:-1}"
DPP_ENABLE_GT_SCORER_CACHE="${DPP_ENABLE_GT_SCORER_CACHE:-0}"
DPP_ENABLE_TORCH_COMPILE="${DPP_ENABLE_TORCH_COMPILE:-0}"
DPP_SKIP_AUTH_EVAL="${DPP_SKIP_AUTH_EVAL:-0}"
DPP_FULL_CPU="${DPP_FULL_CPU:-0}"
DPP_ADVISORY_CPU_EXPLICITLY_WAIVED="${DPP_ADVISORY_CPU_EXPLICITLY_WAIVED:-0}"
DPP_CACHE_DIR="${DPP_CACHE_DIR:-}"
DPP_MAX_DISK_GB="${DPP_MAX_DISK_GB:-100.0}"
DPP_LOG_INCREMENTAL_BASE="${DPP_LOG_INCREMENTAL_BASE:-2}"
DPP_LOG_INCREMENTAL_MAX_CHUNKS="${DPP_LOG_INCREMENTAL_MAX_CHUNKS:-80}"
DPP_LOG_INCREMENTAL_QUALITY_THRESHOLD="${DPP_LOG_INCREMENTAL_QUALITY_THRESHOLD:-0.005}"
DPP_DISABLE_LOG_INCREMENTAL="${DPP_DISABLE_LOG_INCREMENTAL:-0}"
DPP_USE_STREAMER="${DPP_USE_STREAMER:-0}"
DPP_STREAM_LOG_DIR="${DPP_STREAM_LOG_DIR:-$LOG_DIR/stream_logs}"
DPP_RAM_BUFFER_GB="${DPP_RAM_BUFFER_GB:-2.0}"
DPP_STREAMER_FRAMES_PER_CHUNK="${DPP_STREAMER_FRAMES_PER_CHUNK:-256}"
DPP_STREAM_CHUNKING_MODE="${DPP_STREAM_CHUNKING_MODE:-frame_range}"
DPP_STREAM_FRAME_RANGE_SIZE="${DPP_STREAM_FRAME_RANGE_SIZE:-256}"
DPP_STREAM_BYTE_SIZE_TARGET="${DPP_STREAM_BYTE_SIZE_TARGET:-0}"
DPP_STREAM_TEMPORAL_WINDOW_SEC="${DPP_STREAM_TEMPORAL_WINDOW_SEC:-0}"
DPP_STREAM_MOTION_THRESHOLD="${DPP_STREAM_MOTION_THRESHOLD:-}"
DPP_STREAM_ENTROPY_THRESHOLD="${DPP_STREAM_ENTROPY_THRESHOLD:-}"
DPP_STREAM_SALIENCY_TOPK="${DPP_STREAM_SALIENCY_TOPK:-}"
DPP_MAX_DISTILLATION_FRAMES="${DPP_MAX_DISTILLATION_FRAMES:-4096}"
DPP_MAX_DISTILLATION_CHUNKS="${DPP_MAX_DISTILLATION_CHUNKS:-8}"
DPP_MAX_PAIRS="${DPP_MAX_PAIRS:-600}"
DPP_VAL_PAIR_COUNT="${DPP_VAL_PAIR_COUNT:-64}"
DPP_VAL_EVERY_EPOCHS="${DPP_VAL_EVERY_EPOCHS:-50}"
DPP_PROCEDURAL_CODEBOOK_REPLACEMENT="${DPP_PROCEDURAL_CODEBOOK_REPLACEMENT:-0}"
DPP_PROCEDURAL_CODEBOOK_SEED_HEX="${DPP_PROCEDURAL_CODEBOOK_SEED_HEX:-}"
DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND="${DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND:-pcg64}"
DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL="${DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL:-0}"
DPP_PROCEDURAL_CODEBOOK_VALIDATE_DOMAIN="${DPP_PROCEDURAL_CODEBOOK_VALIDATE_DOMAIN:-1}"
DPP_PROCEDURAL_VARIANT_PROVENANCE_PATH="${DPP_PROCEDURAL_VARIANT_PROVENANCE_PATH:-}"
DPP_PROCEDURAL_VARIANT_DISTILLATION_SKIP="${DPP_PROCEDURAL_VARIANT_DISTILLATION_SKIP:-0}"
# Phase 2 Comma2k19 chunk path (Catalog #209-routed via Comma2k19FrameIterator).
# Smoke path uses synthetic stub regardless; full path with --dataset-name=comma2k19
# requires this to be set by the operator-authorize wrapper. Default empty so
# the smoke path doesn't accidentally point at a real chunk dir on the worker.
DPP_COMMA2K19_CHUNKS_DIR="${DPP_COMMA2K19_CHUNKS_DIR:-}"

DISPATCH_INSTANCE_JOB_ID="${DPP_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${DPP_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-dpp] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

DPP_PROCEDURAL_ARGS=()
if [ "$DPP_PROCEDURAL_CODEBOOK_REPLACEMENT" = "1" ]; then
    DPP_PROCEDURAL_ARGS+=(--enable-procedural-codebook-replacement)
fi
if [ -n "$DPP_PROCEDURAL_CODEBOOK_SEED_HEX" ]; then
    DPP_PROCEDURAL_ARGS+=(--procedural-codebook-seed-hex "$DPP_PROCEDURAL_CODEBOOK_SEED_HEX")
fi
if [ -n "$DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND" ]; then
    DPP_PROCEDURAL_ARGS+=(--procedural-codebook-generator-kind "$DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND")
fi
if [ "$DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL" = "1" ]; then
    DPP_PROCEDURAL_ARGS+=(--procedural-codebook-null-exploit-control)
fi
if [ "$DPP_PROCEDURAL_CODEBOOK_VALIDATE_DOMAIN" = "0" ]; then
    DPP_PROCEDURAL_ARGS+=(--no-procedural-codebook-validate-domain)
fi
if [ -n "$DPP_PROCEDURAL_VARIANT_PROVENANCE_PATH" ]; then
    DPP_PROCEDURAL_ARGS+=(--procedural-variant-provenance-path "$DPP_PROCEDURAL_VARIANT_PROVENANCE_PATH")
fi
if [ "$DPP_PROCEDURAL_VARIANT_DISTILLATION_SKIP" = "1" ]; then
    DPP_PROCEDURAL_ARGS+=(--procedural-variant-distillation-skip)
fi

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: DPP_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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

# Stage 0b: verify the Level-2 active dispatch claim before any bootstrap/training.
CLAIM_SUMMARY_JSON="$LOG_DIR/dispatch_claim_summary.json"
"$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
    --claims-path "$DISPATCH_CLAIMS_PATH" \
    --format json > "$CLAIM_SUMMARY_JSON" || {
    log "FATAL: claim summary failed for $DISPATCH_CLAIMS_PATH"
    exit 21
}
"$CLAIM_PYTHON" - "$CLAIM_SUMMARY_JSON" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" <<'PY'
import json, sys
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
log "stage_0b_dispatch_claim_verified lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"

# Stage 1: bootstrap runtime deps via canonical helper (Catalog #163).
log "Stage 1: bootstrap runtime deps"
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# shellcheck source=/dev/null
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
bootstrap_runtime_deps
if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    log "FATAL: PYBIN not set or not executable after bootstrap (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Stage 1b: validate required input files PRE-dispatch (Catalog #152).
log "Stage 1b: validate required input files"
"$PYBIN" "$WORKSPACE/tools/validate_dispatch_required_inputs.py" \
    --trainer "$WORKSPACE/experiments/train_substrate_pretrained_driving_prior.py" \
    || {
    log "FATAL: Catalog #152 required input file validation failed"
    exit 25
}
log "Stage 1b: required input files validated"

# Stage 2: write provenance.json (per Catalog #L wire-in).
log "Stage 2: write provenance.json"
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "started_at_utc": "$(date -u +%FT%TZ)",
  "trainer": "experiments/train_substrate_pretrained_driving_prior.py",
  "video_path": "$DPP_VIDEO_PATH",
  "epochs": $DPP_EPOCHS,
  "batch_size": $DPP_BATCH_SIZE,
  "dataset_name": "$DPP_DATASET_NAME",
  "comma2k19_chunks_dir_set": $(if [ -n "$DPP_COMMA2K19_CHUNKS_DIR" ]; then echo true; else echo false; fi),
  "cache_dir_set": $(if [ -n "$DPP_CACHE_DIR" ]; then echo true; else echo false; fi),
  "use_streamer": $(if [ "$DPP_USE_STREAMER" = "1" ]; then echo true; else echo false; fi),
  "stream_chunking_mode": "$DPP_STREAM_CHUNKING_MODE",
  "max_distillation_frames": $DPP_MAX_DISTILLATION_FRAMES,
  "max_distillation_chunks": $DPP_MAX_DISTILLATION_CHUNKS,
  "max_pairs": $DPP_MAX_PAIRS,
  "val_pair_count": $DPP_VAL_PAIR_COUNT,
  "enable_gt_scorer_cache": $(if [ "$DPP_ENABLE_GT_SCORER_CACHE" = "1" ]; then echo true; else echo false; fi),
  "enable_torch_compile": $(if [ "$DPP_ENABLE_TORCH_COMPILE" = "1" ]; then echo true; else echo false; fi),
  "device": "$DPP_DEVICE",
  "platform": "$DISPATCH_PLATFORM",
  "instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "evidence_grade": "[scaffold-only-no-score-claim]",
  "score_claim": false,
  "score_claim_valid": false,
  "promotion_eligible": false,
  "rank_or_kill_eligible": false,
  "ready_for_exact_eval_dispatch": false
}
EOF
log "wrote provenance: $PROVENANCE"

# Stage 3: smoke path FIRST (always runs — exercises codebook distill + pack + parse).
log "Stage 3: smoke distill + pack + parse"
"$PYBIN" "$WORKSPACE/experiments/train_substrate_pretrained_driving_prior.py" \
    --smoke \
    --video-path "$DPP_VIDEO_PATH" \
    --output-dir "$DPP_OUTPUT_DIR" \
    --upstream-dir "$DPP_UPSTREAM_DIR" \
    --device "$DPP_DEVICE" \
    --epochs "$DPP_EPOCHS" \
    --batch-size "$DPP_BATCH_SIZE" \
    --dataset-name synthetic_test \
    "${DPP_PROCEDURAL_ARGS[@]}"

# Stage 4: Phase 2 full training (lane_pretrained_driving_prior_phase_2_20260514).
# Only fires when DPP_RUN_FULL=1 is explicitly set by the operator-authorize
# wrapper; otherwise we exit at scaffold-smoke completion to preserve $0
# default behavior. Per CLAUDE.md "Race-mode rigor inversion": the full
# dispatch is operator-gated; the smoke MUST PASS (above) before the full
# path is attempted.
DPP_RUN_FULL="${DPP_RUN_FULL:-0}"
if [ "$DPP_RUN_FULL" = "1" ]; then
    log "Stage 4: Phase 2 full training (DPP_RUN_FULL=1)"
    if [ "$DPP_DATASET_NAME" = "comma2k19" ] \
        && [ -z "$DPP_COMMA2K19_CHUNKS_DIR" ] \
        && [ -z "$DPP_CACHE_DIR" ] \
        && [ "$DPP_USE_STREAMER" != "1" ]; then
        log "FATAL: DPP_RUN_FULL=1 with DPP_DATASET_NAME=comma2k19 requires one explicit dataset source: DPP_COMMA2K19_CHUNKS_DIR, DPP_CACHE_DIR, or DPP_USE_STREAMER=1"
        exit 26
    fi
    DPP_FULL_ARGS=(
        --video-path "$DPP_VIDEO_PATH"
        --output-dir "$DPP_OUTPUT_DIR"
        --upstream-dir "$DPP_UPSTREAM_DIR"
        --device "$DPP_DEVICE"
        --epochs "$DPP_EPOCHS"
        --batch-size "$DPP_BATCH_SIZE"
        --dataset-name "$DPP_DATASET_NAME"
        --max-disk-gb "$DPP_MAX_DISK_GB"
        --log-incremental-base "$DPP_LOG_INCREMENTAL_BASE"
        --log-incremental-max-chunks "$DPP_LOG_INCREMENTAL_MAX_CHUNKS"
        --log-incremental-quality-threshold "$DPP_LOG_INCREMENTAL_QUALITY_THRESHOLD"
        --max-distillation-frames "$DPP_MAX_DISTILLATION_FRAMES"
        --max-distillation-chunks "$DPP_MAX_DISTILLATION_CHUNKS"
        --max-pairs "$DPP_MAX_PAIRS"
        --val-pair-count "$DPP_VAL_PAIR_COUNT"
        --val-every-epochs "$DPP_VAL_EVERY_EPOCHS"
    )
    if [ -n "$DPP_COMMA2K19_CHUNKS_DIR" ]; then
        DPP_FULL_ARGS+=(--comma2k19-chunks-dir "$DPP_COMMA2K19_CHUNKS_DIR")
    fi
    if [ -n "$DPP_CACHE_DIR" ]; then
        DPP_FULL_ARGS+=(--cache-dir "$DPP_CACHE_DIR")
    fi
    if [ "$DPP_DISABLE_LOG_INCREMENTAL" = "1" ]; then
        DPP_FULL_ARGS+=(--disable-log-incremental)
    fi
    if [ "$DPP_USE_STREAMER" = "1" ]; then
        DPP_FULL_ARGS+=(
            --use-streamer
            --stream-log-dir "$DPP_STREAM_LOG_DIR"
            --ram-buffer-gb "$DPP_RAM_BUFFER_GB"
            --streamer-frames-per-chunk "$DPP_STREAMER_FRAMES_PER_CHUNK"
            --stream-chunking-mode "$DPP_STREAM_CHUNKING_MODE"
            --stream-frame-range-size "$DPP_STREAM_FRAME_RANGE_SIZE"
            --stream-byte-size-target "$DPP_STREAM_BYTE_SIZE_TARGET"
            --stream-temporal-window-sec "$DPP_STREAM_TEMPORAL_WINDOW_SEC"
        )
        if [ -n "$DPP_STREAM_MOTION_THRESHOLD" ]; then
            DPP_FULL_ARGS+=(--stream-motion-threshold "$DPP_STREAM_MOTION_THRESHOLD")
        fi
        if [ -n "$DPP_STREAM_ENTROPY_THRESHOLD" ]; then
            DPP_FULL_ARGS+=(--stream-entropy-threshold "$DPP_STREAM_ENTROPY_THRESHOLD")
        fi
        if [ -n "$DPP_STREAM_SALIENCY_TOPK" ]; then
            DPP_FULL_ARGS+=(--stream-saliency-topk "$DPP_STREAM_SALIENCY_TOPK")
        fi
    fi
    if [ "$DPP_ENABLE_AUTOCAST_FP16" = "1" ]; then
        DPP_FULL_ARGS+=(--enable-autocast-fp16)
    fi
    if [ "$DPP_ENABLE_GT_SCORER_CACHE" = "1" ]; then
        DPP_FULL_ARGS+=(--enable-gt-scorer-cache)
    fi
    if [ "$DPP_ENABLE_TORCH_COMPILE" = "1" ]; then
        DPP_FULL_ARGS+=(--enable-torch-compile)
    fi
    if [ "$DPP_SKIP_AUTH_EVAL" = "1" ]; then
        DPP_FULL_ARGS+=(--skip-auth-eval)
    fi
    if [ "$DPP_FULL_CPU" = "1" ]; then
        DPP_FULL_ARGS+=(--full-cpu)
    fi
    if [ "$DPP_ADVISORY_CPU_EXPLICITLY_WAIVED" = "1" ]; then
        DPP_FULL_ARGS+=(--advisory-cpu-explicitly-waived)
    fi
    DPP_FULL_ARGS+=("${DPP_PROCEDURAL_ARGS[@]}")
    "$PYBIN" "$WORKSPACE/experiments/train_substrate_pretrained_driving_prior.py" \
        "${DPP_FULL_ARGS[@]}"
    log "DPP Phase 2 full training complete"
else
    log "DPP scaffold smoke complete; DPP_RUN_FULL=0 — skipping Phase 2 full training"
    log "To run Phase 2 full training, the operator-authorize wrapper sets DPP_RUN_FULL=1 + DPP_COMMA2K19_CHUNKS_DIR"
fi
log "Exiting cleanly (rc=0)."
exit 0
