#!/bin/bash
# Remote lane script: NSCS01 nullspace-split-renderer L1 SCAFFOLD smoke dispatch.
#
# Trainer: experiments/train_substrate_nscs01_nullspace_split_renderer.py
# Lane: lane_nscs01_nullspace_split_renderer_20260515
# Recipe: .omx/operator_authorize_recipes/substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the sentinel ``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1`` is
# prepended to the source line so the sourced script's main flow does NOT run.
#
# Per Catalog #244 the canonical NVML block (DALI_DISABLE_NVML +
# CUBLAS_WORKSPACE_CONFIG + PYTORCH_CUDA_ALLOC_CONF) is exported BEFORE
# bootstrap or any torch import.
#
# Design refs:
#   - .omx/research/nscs01_nullspace_split_renderer_design_20260515.md
#   - .omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json (SA02)
#
# L1 SCAFFOLD smoke-only: NO auth-eval (research_only=true per recipe).
# Score-tagging: completion-log line is `LANE_NSCS01_DONE [smoke-no-scorer]`.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_nscs01_nullspace_split_renderer_20260515"
TAG="${TAG:-substrate_nscs01_nullspace_split_renderer}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_nscs01_nullspace_split_renderer_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${NSCS01_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$NSCS01_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Catalog #244 NVML/CUDA env block — set BEFORE bootstrap or torch import.
# Anchor: D1 Modal T4 NVML 999 crash fix (commit 611495f26 sister-class).
# Constants source: tac.deploy.modal.runtime.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
NSCS01_VIDEO_PATH="${NSCS01_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
NSCS01_OUTPUT_DIR="${NSCS01_OUTPUT_DIR:-$OUTPUT_DIR}"
NSCS01_EPOCHS="${NSCS01_EPOCHS:-100}"
NSCS01_UPSTREAM_DIR="${NSCS01_UPSTREAM_DIR:-$WORKSPACE/upstream}"
NSCS01_DEVICE="${NSCS01_DEVICE:-cuda}"
NSCS01_HEAD0_BITS="${NSCS01_HEAD0_BITS:-4}"
NSCS01_HEAD1_BITS="${NSCS01_HEAD1_BITS:-8}"
NSCS01_LATENT_DIM="${NSCS01_LATENT_DIM:-16}"
NSCS01_ENABLE_GT_SCORER_CACHE="${NSCS01_ENABLE_GT_SCORER_CACHE:-true}"

DISPATCH_INSTANCE_JOB_ID="${NSCS01_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${NSCS01_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-nscs01] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: NSCS01_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

verify_active_dispatch_claim() {
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "FATAL: claim helper missing; cannot verify active dispatch claim"
        exit 26
    fi
    local claim_summary_json="$LOG_DIR/dispatch_claim_summary.json"
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --live-only \
        --format json \
        > "$claim_summary_json" || {
        log "FATAL: claim summary failed; refusing remote driver startup"
        exit 26
    }
    "$CLAIM_PYTHON" - "$claim_summary_json" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" <<'PY' || {
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
lane_id = sys.argv[2]
job_id = sys.argv[3]
payload = json.loads(summary_path.read_text(encoding="utf-8"))
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(
    f"no active dispatch claim for lane_id={lane_id} instance_job_id={job_id}",
    file=sys.stderr,
)
raise SystemExit(1)
PY
        log "FATAL: no active dispatch claim for lane=$LANE_ID instance/job=$DISPATCH_INSTANCE_JOB_ID"
        exit 27
    }
    CLAIM_VERIFIED=1
    log "Stage 0 DONE: active dispatch claim verified"
}

append_terminal_claim() {
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        status="completed_nscs01_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_nscs01_claim_verification_rc_${rc}"
    else
        status="failed_nscs01_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --status "$status" \
        --lane-id "$LANE_ID" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --agent "remote_lane_substrate_nscs01_nullspace_split_renderer" \
        --notes "remote_driver terminal append (rc=$rc)" >> "$LOG_DIR/run.log" 2>&1 || \
        log "WARN: terminal claim append failed (rc=$?)"
}

cleanup() {
    local rc=$?
    if [ -n "$HEARTBEAT_PID" ] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT

log "Stage 0: verifying lane dispatch claim for instance=$DISPATCH_INSTANCE_JOB_ID"
verify_active_dispatch_claim

# Stage 1: bootstrap runtime deps via the canonical helper.
log "Stage 1: bootstrap runtime deps (delegated)"
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing"
    exit 23
fi
# Catalog #163 sentinel: source-only (no main flow).
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
bootstrap_runtime_deps

if [ -z "$PYBIN" ]; then
    if [ -x "$WORKSPACE/.venv/bin/python" ]; then
        PYBIN="$WORKSPACE/.venv/bin/python"
    else
        PYBIN="python3"
    fi
fi

# Stage 2: heartbeat per CLAUDE.md "Remote code parity".
log "Stage 2: starting heartbeat (5-min interval)"
(
    while true; do
        sleep 300
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID instance=$DISPATCH_INSTANCE_JOB_ID alive" >> "$LOG_DIR/heartbeat.log"
    done
) &
HEARTBEAT_PID=$!

# Stage 3: emit provenance.
log "Stage 3: writing provenance"
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "platform": "$DISPATCH_PLATFORM",
  "started_at_utc": "$(date -u +%FT%TZ)",
  "trainer": "experiments/train_substrate_nscs01_nullspace_split_renderer.py",
  "research_only": true,
  "smoke_only": true,
  "council_phase_2_required_before_full_dispatch": true,
  "video_path": "$NSCS01_VIDEO_PATH",
  "epochs": $NSCS01_EPOCHS,
  "head0_bits": $NSCS01_HEAD0_BITS,
  "head1_bits": $NSCS01_HEAD1_BITS,
  "latent_dim": $NSCS01_LATENT_DIM,
  "device": "$NSCS01_DEVICE",
  "git_head": "$(git -C $WORKSPACE rev-parse HEAD 2>/dev/null || echo unknown)",
  "env": {
    "CUBLAS_WORKSPACE_CONFIG": "$CUBLAS_WORKSPACE_CONFIG",
    "DALI_DISABLE_NVML": "$DALI_DISABLE_NVML",
    "PYTORCH_CUDA_ALLOC_CONF": "$PYTORCH_CUDA_ALLOC_CONF"
  }
}
EOF

# Stage 4: train.
log "Stage 4: running NSCS01 trainer (--smoke per recipe research_only=true)"
set +e
"$PYBIN" -u experiments/train_substrate_nscs01_nullspace_split_renderer.py \
    --smoke \
    --output-dir "$NSCS01_OUTPUT_DIR" \
    --device "$NSCS01_DEVICE" \
    --epochs "$NSCS01_EPOCHS" \
    --head0-bits "$NSCS01_HEAD0_BITS" \
    --head1-bits "$NSCS01_HEAD1_BITS" \
    --latent-dim "$NSCS01_LATENT_DIM" \
    --upstream-dir "$NSCS01_UPSTREAM_DIR" \
    2>&1 | tee -a "$LOG_DIR/train.log"

TRAIN_RC=${PIPESTATUS[0]}
set -e

# Stage 5: completion log line — research_only smoke; no contest-CUDA score claim.
if [ "$TRAIN_RC" -eq 0 ]; then
    log "LANE_NSCS01_DONE [smoke-no-scorer] research_only=true output_dir=$NSCS01_OUTPUT_DIR"
else
    log "LANE_NSCS01_FAILED rc=$TRAIN_RC"
    exit "$TRAIN_RC"
fi

exit 0
