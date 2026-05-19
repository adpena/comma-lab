#!/bin/bash
# Remote lane script: substrate C6 MDL-IBPS (IBPS1) first-anchor dispatch.
#
# Trainer: experiments/train_substrate_c6_e4_mdl_ibps.py
# Lane: lane_c6_e4_mdl_ibps_substrate_20260514
# Recipe: .omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the sentinel ``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1`` is
# prepended to the source line so the sourced script's main flow does NOT
# run.
#
# Design refs:
#   - .omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md
#   - .omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md
#   - Tishby & Zaslavsky 2015 IB; Rissanen 1978 MDL; Alemi et al. 2017 VIB
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_C6_MDL_IBPS_DONE marker).
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
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
DEFAULT_LANE_ID="lane_c6_e4_mdl_ibps_substrate_20260514"
LANE_ID="${C6_E4_MDL_IBPS_LANE_ID:-${PACT_DISPATCH_LANE_ID:-$DEFAULT_LANE_ID}}"
TAG="${TAG:-substrate_c6_e4_mdl_ibps}"
RECIPE_PATH="${C6_E4_MDL_IBPS_RECIPE_PATH:-.omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_c6_e4_mdl_ibps_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${C6_E4_MDL_IBPS_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$C6_E4_MDL_IBPS_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
C6_E4_MDL_IBPS_VIDEO_PATH="${C6_E4_MDL_IBPS_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
C6_E4_MDL_IBPS_OUTPUT_DIR="${C6_E4_MDL_IBPS_OUTPUT_DIR:-$OUTPUT_DIR}"
C6_E4_MDL_IBPS_EPOCHS="${C6_E4_MDL_IBPS_EPOCHS:-200}"
C6_E4_MDL_IBPS_BATCH_SIZE="${C6_E4_MDL_IBPS_BATCH_SIZE:-4}"
C6_E4_MDL_IBPS_LR="${C6_E4_MDL_IBPS_LR:-5e-4}"
C6_E4_MDL_IBPS_LATENT_DIM="${C6_E4_MDL_IBPS_LATENT_DIM:-24}"
C6_E4_MDL_IBPS_BETA_IB="${C6_E4_MDL_IBPS_BETA_IB:-0.01}"
C6_E4_MDL_IBPS_ENABLE_AUTOCAST_FP16="${C6_E4_MDL_IBPS_ENABLE_AUTOCAST_FP16:-false}"
C6_E4_MDL_IBPS_UPSTREAM_DIR="${C6_E4_MDL_IBPS_UPSTREAM_DIR:-$WORKSPACE/upstream}"
C6_E4_MDL_IBPS_DEVICE="${C6_E4_MDL_IBPS_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${C6_E4_MDL_IBPS_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${C6_E4_MDL_IBPS_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-c6-mdl-ibps] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
    log "FATAL: C6_E4_MDL_IBPS_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi

if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    case "$C6_E4_MDL_IBPS_OUTPUT_DIR" in
        "$WORKSPACE"/*|/tmp/*|/workspace/*)
            LOG_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}"
            OUTPUT_DIR="$LOG_DIR/output"
            C6_E4_MDL_IBPS_OUTPUT_DIR="$OUTPUT_DIR"
            ;;
    esac
fi
PROVENANCE="$LOG_DIR/provenance.json"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

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
        status="completed_c6_mdl_ibps_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_c6_mdl_ibps_claim_verification_rc_${rc}"
    else
        status="failed_c6_mdl_ibps_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_c6_e4_mdl_ibps" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$C6_E4_MDL_IBPS_OUTPUT_DIR latent_dim=$C6_E4_MDL_IBPS_LATENT_DIM beta_ib=$C6_E4_MDL_IBPS_BETA_IB recipe=$RECIPE_PATH" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: failed to append terminal dispatch claim status=$status"
    }
}

cleanup() {
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT
verify_active_dispatch_claim

# Stage 1: bootstrap remote runtime deps via canonical sourced helper.
# Per Catalog #163 prepend REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 so the
# sourced script's main flow does NOT run.
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
    HEARTBEAT_SLEEP_PID=""
    trap 'if [ -n "$HEARTBEAT_SLEEP_PID" ]; then kill "$HEARTBEAT_SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT EXIT
    while true; do
        date -u +%FT%TZ > "$LOG_DIR/heartbeat.log"
        sleep 300 &
        HEARTBEAT_SLEEP_PID="$!"
        wait "$HEARTBEAT_SLEEP_PID" || true
        HEARTBEAT_SLEEP_PID=""
    done
) &
HEARTBEAT_PID=$!

# Stage 3: write provenance.
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_c6_e4_mdl_ibps.py",
  "recipe": "$RECIPE_PATH",
  "video_path": "$C6_E4_MDL_IBPS_VIDEO_PATH",
  "output_dir": "$C6_E4_MDL_IBPS_OUTPUT_DIR",
  "epochs": "$C6_E4_MDL_IBPS_EPOCHS",
  "batch_size": "$C6_E4_MDL_IBPS_BATCH_SIZE",
  "lr": "$C6_E4_MDL_IBPS_LR",
  "latent_dim": "$C6_E4_MDL_IBPS_LATENT_DIM",
  "beta_ib": "$C6_E4_MDL_IBPS_BETA_IB",
  "enable_autocast_fp16": "$C6_E4_MDL_IBPS_ENABLE_AUTOCAST_FP16",
  "device": "$C6_E4_MDL_IBPS_DEVICE",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin epochs=$C6_E4_MDL_IBPS_EPOCHS latent_dim=$C6_E4_MDL_IBPS_LATENT_DIM beta_ib=$C6_E4_MDL_IBPS_BETA_IB"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_c6_e4_mdl_ibps.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

C6_TRAINER_ARGS=()
case "$C6_E4_MDL_IBPS_ENABLE_AUTOCAST_FP16" in
    1|true|TRUE|True|yes|YES|Yes)
        C6_TRAINER_ARGS+=(--enable-autocast-fp16)
        ;;
esac

"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --video-path "$C6_E4_MDL_IBPS_VIDEO_PATH" \
    --output-dir "$C6_E4_MDL_IBPS_OUTPUT_DIR" \
    --epochs "$C6_E4_MDL_IBPS_EPOCHS" \
    --batch-size "$C6_E4_MDL_IBPS_BATCH_SIZE" \
    --lr "$C6_E4_MDL_IBPS_LR" \
    --latent-dim "$C6_E4_MDL_IBPS_LATENT_DIM" \
    --beta-ib "$C6_E4_MDL_IBPS_BETA_IB" \
    --upstream-dir "$C6_E4_MDL_IBPS_UPSTREAM_DIR" \
    --device "$C6_E4_MDL_IBPS_DEVICE" \
    ${C6_TRAINER_ARGS[@]+"${C6_TRAINER_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# The trainer is allowed to write non-authoritative stats, but this remote
# driver may only emit the [contest-CUDA] completion marker after a validated
# CUDA auth-eval claim. This prevents stale auth-eval flags plus trainer rc=0
# from looking like a successful contest-CUDA lane.
"$PYBIN_RESOLVED" - "$C6_E4_MDL_IBPS_OUTPUT_DIR/stats.json" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
if not stats_path.is_file():
    raise SystemExit(f"missing C6 stats.json: {stats_path}")
stats = json.loads(stats_path.read_text(encoding="utf-8"))
if stats.get("auth_eval_score_claim_valid") is not True:
    raise SystemExit(
        "C6 stats missing valid auth_eval_score_claim_valid=true; "
        f"blockers={stats.get('result_review_blockers')!r}"
    )
if stats.get("auth_eval_score_axis") != "contest_cuda":
    raise SystemExit(
        f"C6 stats not on contest_cuda axis: {stats.get('auth_eval_score_axis')!r}"
    )
if stats.get("auth_eval_exact_cuda_complete") is not True:
    raise SystemExit("C6 stats missing auth_eval_exact_cuda_complete=true")
print(
    "C6_AUTH_EVAL_VALIDATED "
    f"score={stats.get('auth_eval_score')} "
    f"path={stats.get('auth_eval_result_path')}"
)
PY

# Stage 5: emit completion marker (operator + autopilot consume).
log "LANE_C6_MDL_IBPS_DONE [contest-CUDA] output_dir=$C6_E4_MDL_IBPS_OUTPUT_DIR"
cat >> "$LOG_DIR/completion.log" <<EOF
LANE_C6_MDL_IBPS_DONE [contest-CUDA] $LANE_ID $(date -u +%FT%TZ)
EOF
