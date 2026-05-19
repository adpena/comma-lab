#!/bin/bash
# Remote lane script: substrate Z3-G1 scorer-class-conditional gating dispatch.
#
# Trainer: experiments/train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py
# Lane: lane_z3_g1_scorer_softmax_hyperprior_gating_20260515
# Recipe: .omx/operator_authorize_recipes/substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the sentinel ``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1`` is
# prepended to the source line so the sourced script's main flow does NOT run.
#
# Design refs:
#   - feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md
#   - feedback_grand_council_evidence_review_modal_failures_no_compromise_optimal_lowest_score_landed_20260515.md
#   - Ballé et al. (2018) ICLR scale-hyperprior arXiv:1802.01436
#
# Score-tagging: smoke/no-scorer artifacts are explicitly logged as
# score_claim=false and never as [contest-CUDA]. A [contest-CUDA] marker is
# allowed only when stats.json proves a valid contest_cuda score claim.
# Per Catalog #204 the output is written to
# /modal_results/${DISPATCH_INSTANCE_JOB_ID}/output for durable provider
# custody when MODAL_RUNTIME=1.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_z3_g1_scorer_softmax_hyperprior_gating_20260515"
TAG="${TAG:-substrate_z3_g1_scorer_softmax_hyperprior_gating}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_z3_g1_scorer_softmax_hyperprior_gating_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${Z3_G1_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$Z3_G1_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Catalog #224: CUBLAS deterministic + DALI NVML disable for stable inflate.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
Z3_G1_A1_ARCHIVE_PATH="${Z3_G1_A1_ARCHIVE_PATH:-$WORKSPACE/submissions/a1/archive.zip}"
Z3_G1_VIDEO_PATH="${Z3_G1_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
Z3_G1_OUTPUT_DIR="${Z3_G1_OUTPUT_DIR:-$OUTPUT_DIR}"
Z3_G1_EPOCHS="${Z3_G1_EPOCHS:-1000}"
Z3_G1_UPSTREAM_DIR="${Z3_G1_UPSTREAM_DIR:-$WORKSPACE/upstream}"
Z3_G1_DEVICE="${Z3_G1_DEVICE:-cuda}"

# Smoke vs full ladder.
SMOKE_ONLY="${SMOKE_ONLY:-0}"

DISPATCH_INSTANCE_JOB_ID="${Z3_G1_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${Z3_G1_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-z3-g1] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: Z3_G1_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
        status="completed_z3_g1_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_z3_g1_claim_verification_rc_${rc}"
    else
        status="failed_z3_g1_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_z3_g1_scorer_softmax_hyperprior_gating" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$Z3_G1_OUTPUT_DIR smoke=$SMOKE_ONLY" \
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

# Stage 1: bootstrap remote runtime deps via canonical sourced helper.
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
  "trainer": "experiments/train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py",
  "recipe": ".omx/operator_authorize_recipes/substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch.yaml",
  "a1_archive_path": "$Z3_G1_A1_ARCHIVE_PATH",
  "video_path": "$Z3_G1_VIDEO_PATH",
  "output_dir": "$Z3_G1_OUTPUT_DIR",
  "epochs": "$Z3_G1_EPOCHS",
  "device": "$Z3_G1_DEVICE",
  "smoke_only": "$SMOKE_ONLY",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin epochs=$Z3_G1_EPOCHS smoke=$SMOKE_ONLY"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

SMOKE_FLAG_ARGS=()
if [ "$SMOKE_ONLY" = "1" ]; then
    SMOKE_FLAG_ARGS+=(--smoke)
fi

"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --a1-archive-path "$Z3_G1_A1_ARCHIVE_PATH" \
    --video-path "$Z3_G1_VIDEO_PATH" \
    --output-dir "$Z3_G1_OUTPUT_DIR" \
    --epochs "$Z3_G1_EPOCHS" \
    --upstream-dir "$Z3_G1_UPSTREAM_DIR" \
    --device "$Z3_G1_DEVICE" \
    ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker (operator + autopilot consume).
EVIDENCE_STATUS="$("$PYBIN_RESOLVED" - "$Z3_G1_OUTPUT_DIR/stats.json" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
marker = "[training-artifact-no-score-claim]"
score_claim = "score_claim=false"
if stats_path.is_file():
    try:
        stats = json.loads(stats_path.read_text())
    except json.JSONDecodeError:
        stats = {}
    if (
        stats.get("auth_eval_score_claim_valid") is True
        and stats.get("auth_eval_score_axis") == "contest_cuda"
    ):
        marker = "[contest-CUDA]"
        score_claim = "score_claim=true"
    elif stats.get("evidence_grade"):
        marker = f"[{stats['evidence_grade']}]"
print(f"{marker} {score_claim}")
PY
)"
EVIDENCE_MARKER="${EVIDENCE_STATUS%% *}"
SCORE_CLAIM_FLAG="${EVIDENCE_STATUS#* }"
log "LANE_Z3_G1_DONE ${EVIDENCE_MARKER} output_dir=$Z3_G1_OUTPUT_DIR smoke=$SMOKE_ONLY ${SCORE_CLAIM_FLAG}"
echo "LANE_Z3_G1_DONE ${EVIDENCE_MARKER} $LANE_ID ${SCORE_CLAIM_FLAG} $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
