#!/bin/bash
# Remote lane script: substrate NSCS02 downsampled-renderer + inflate-upsample.
#
# Trainer: experiments/train_substrate_nscs02_downsampled_renderer.py
# Lane: lane_nscs02_downsampled_renderer_inflate_upsample_20260515
# Recipe: .omx/operator_authorize_recipes/substrate_nscs02_downsampled_renderer_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" the
# bootstrap is delegated to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps function via the Catalog #163 sentinel.
#
# Per Catalog #244 the canonical NVML/CUDA env hygiene block is set
# IMMEDIATELY after `set -euo pipefail` (DALI_DISABLE_NVML +
# CUBLAS_WORKSPACE_CONFIG + PYTORCH_CUDA_ALLOC_CONF).
#
# Score-tagging: [contest-CUDA] only after parse_auth_eval_score_claim
# certifies the JSON as a contest-CUDA custody-valid score claim
# (Catalog #221 fail-closed semantics).
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity".
set -euo pipefail

# Catalog #244 canonical NVML/CUDA env hygiene block - MUST appear
# immediately after `set -euo pipefail` per the per-substrate-driver
# self-protection. Anchor: 2026-05-15 D1 nvml error 999 incident
# (commit 611495f26 patched D1 directly; this propagates the fix).
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_nscs02_downsampled_renderer_inflate_upsample_20260515"
TAG="${TAG:-substrate_nscs02_downsampled_renderer}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_nscs02_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${NSCS02_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$NSCS02_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
NSCS02_VIDEO_PATH="${NSCS02_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
NSCS02_OUTPUT_DIR="${NSCS02_OUTPUT_DIR:-$OUTPUT_DIR}"
NSCS02_EPOCHS="${NSCS02_EPOCHS:-200}"
NSCS02_BATCH_SIZE="${NSCS02_BATCH_SIZE:-8}"
NSCS02_LR="${NSCS02_LR:-5e-4}"
NSCS02_UPSAMPLE_MODE="${NSCS02_UPSAMPLE_MODE:-bicubic}"
NSCS02_SEG_WEIGHT="${NSCS02_SEG_WEIGHT:-100.0}"
NSCS02_POSE_WEIGHT="${NSCS02_POSE_WEIGHT:-1.0}"
NSCS02_UPSTREAM_DIR="${NSCS02_UPSTREAM_DIR:-$WORKSPACE/upstream}"
NSCS02_DEVICE="${NSCS02_DEVICE:-cuda}"
ENABLE_AUTOCAST_FP16="${ENABLE_AUTOCAST_FP16:-true}"

DISPATCH_INSTANCE_JOB_ID="${NSCS02_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${NSCS02_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-nscs02] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (Catalog #245 gate).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: NSCS02_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID required"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

append_terminal_claim() {
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        status="completed_nscs02_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_nscs02_claim_verification_rc_${rc}"
    else
        status="failed_nscs02_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_nscs02_downsampled_renderer" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$NSCS02_OUTPUT_DIR" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: failed to append terminal dispatch claim status=$status"
    }
}

cleanup() {
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT

# Stage 1: bootstrap remote runtime deps via canonical sourced helper
# (Catalog #163 sentinel + canonical helper per CLAUDE.md "Forbidden
# re-implementing remote bootstrap inline").
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "stage_1_bootstrap_via_canonical_sourced_helper"
    # shellcheck disable=SC1091
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
    bootstrap_runtime_deps || {
        log "FATAL: bootstrap_runtime_deps failed; refusing dispatch"
        exit 22
    }
else
    log "WARN: canonical bootstrap script missing; assuming runtime deps present"
fi

# Stage 2: heartbeat (every 5 min).
(
    while true; do
        date -u +%FT%TZ > "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!

# Stage 3: write provenance.
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_nscs02_downsampled_renderer.py",
  "recipe": ".omx/operator_authorize_recipes/substrate_nscs02_downsampled_renderer_modal_t4_dispatch.yaml",
  "video_path": "$NSCS02_VIDEO_PATH",
  "output_dir": "$NSCS02_OUTPUT_DIR",
  "epochs": "$NSCS02_EPOCHS",
  "batch_size": "$NSCS02_BATCH_SIZE",
  "lr": "$NSCS02_LR",
  "upsample_mode": "$NSCS02_UPSAMPLE_MODE",
  "device": "$NSCS02_DEVICE",
  "enable_autocast_fp16": "$ENABLE_AUTOCAST_FP16",
  "cublas_workspace_config": "$CUBLAS_WORKSPACE_CONFIG",
  "dali_disable_nvml": "$DALI_DISABLE_NVML",
  "pytorch_cuda_alloc_conf": "$PYTORCH_CUDA_ALLOC_CONF",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin epochs=$NSCS02_EPOCHS batch_size=$NSCS02_BATCH_SIZE upsample_mode=$NSCS02_UPSAMPLE_MODE"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_nscs02_downsampled_renderer.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

AUTOCAST_ARGS=()
if [ "$ENABLE_AUTOCAST_FP16" = "true" ]; then
    AUTOCAST_ARGS+=(--enable-autocast-fp16)
fi

"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --video-path "$NSCS02_VIDEO_PATH" \
    --output-dir "$NSCS02_OUTPUT_DIR" \
    --epochs "$NSCS02_EPOCHS" \
    --batch-size "$NSCS02_BATCH_SIZE" \
    --lr "$NSCS02_LR" \
    --upsample-mode "$NSCS02_UPSAMPLE_MODE" \
    --seg-weight "$NSCS02_SEG_WEIGHT" \
    --pose-weight "$NSCS02_POSE_WEIGHT" \
    --upstream-dir "$NSCS02_UPSTREAM_DIR" \
    --device "$NSCS02_DEVICE" \
    ${AUTOCAST_ARGS[@]+"${AUTOCAST_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker. [contest-CUDA] tag only after
# parse_auth_eval_score_claim certifies the JSON (Catalog #221).
AUTH_EVAL_JSON=""
for candidate in \
    "$NSCS02_OUTPUT_DIR/contest_auth_eval.json" \
    "$NSCS02_OUTPUT_DIR/contest_auth_eval_cuda.json" \
    "$NSCS02_OUTPUT_DIR/auth_eval.json"; do
    if [ -f "$candidate" ]; then
        AUTH_EVAL_JSON="$candidate"
        break
    fi
done

AUTH_EVAL_SCORE=""
if [ -n "$AUTH_EVAL_JSON" ]; then
    if AUTH_EVAL_SCORE="$(PYTHONPATH="$WORKSPACE/src${PYTHONPATH:+:$PYTHONPATH}" "$PYBIN_RESOLVED" - "$AUTH_EVAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

from tac.auth_eval_result import parse_auth_eval_score_claim

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
claim = parse_auth_eval_score_claim(
    payload,
    required_score_axis="contest_cuda",
    require_component_recompute=True,
)
if claim is None:
    raise SystemExit("auth eval JSON is not a custody-valid contest-CUDA score claim")
print(f"{claim.score:.12g}")
PY
)"; then
        log "LANE_NSCS02_DONE [contest-CUDA] score=$AUTH_EVAL_SCORE auth_eval=$AUTH_EVAL_JSON output_dir=$NSCS02_OUTPUT_DIR"
        echo "LANE_NSCS02_DONE [contest-CUDA] $LANE_ID score=$AUTH_EVAL_SCORE auth_eval=$AUTH_EVAL_JSON $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
    else
        log "LANE_NSCS02_DONE [training-artifact] auth_eval_not_custody_valid=$AUTH_EVAL_JSON output_dir=$NSCS02_OUTPUT_DIR"
        echo "LANE_NSCS02_DONE [training-artifact] $LANE_ID auth_eval_not_custody_valid=$AUTH_EVAL_JSON $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
    fi
else
    log "LANE_NSCS02_DONE [training-artifact] auth_eval_missing output_dir=$NSCS02_OUTPUT_DIR"
    echo "LANE_NSCS02_DONE [training-artifact] $LANE_ID auth_eval_missing $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
fi
