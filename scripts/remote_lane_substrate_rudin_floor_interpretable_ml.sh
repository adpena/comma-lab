#!/bin/bash
# Remote lane script: substrate rudin_floor interpretable-ML L1 SCAFFOLD dispatch.
#
# Trainer: experiments/train_substrate_rudin_floor_interpretable_ml.py
# Lane: lane_rudin_floor_l1_scaffold_substrate_build_20260516
# Design memo: .omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md
#
# Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" +
# Catalog #220 cascade: this driver is the design-time placeholder for the
# Phase 2 council-lift activation. The recipe (substrate_rudin_floor_*) has
# dispatch_enabled:false at landing; the trainer's _full_main raises
# NotImplementedError; the driver CAN be invoked but the trainer aborts
# before any Modal/GPU spend occurs.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" +
# Catalog #163 (canonical bootstrap sentinel): this script DELEGATES bootstrap
# to scripts/remote_archive_only_eval.sh's bootstrap_runtime_deps() via the
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel.
#
# Score-tagging: completion logs derive contest-axis marker from auth eval
# JSON's score_claim_valid + score_axis fields. Modal CPU advisory runs are
# training artifacts and MUST never be logged as contest-CUDA.
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
LANE_ID="lane_rudin_floor_l1_scaffold_substrate_build_20260516"
TAG="${TAG:-substrate_rudin_floor_interpretable_ml}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_rudin_floor_interpretable_ml_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${RUDIN_FLOOR_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$RUDIN_FLOOR_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
RUDIN_FLOOR_VIDEO_PATH="${RUDIN_FLOOR_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
RUDIN_FLOOR_OUTPUT_DIR="${RUDIN_FLOOR_OUTPUT_DIR:-$OUTPUT_DIR}"
RUDIN_FLOOR_UPSTREAM_DIR="${RUDIN_FLOOR_UPSTREAM_DIR:-$WORKSPACE/upstream}"
RUDIN_FLOOR_EPOCHS="${RUDIN_FLOOR_EPOCHS:-8}"
RUDIN_FLOOR_SEED="${RUDIN_FLOOR_SEED:-0}"
RUDIN_FLOOR_DEVICE="${RUDIN_FLOOR_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${RUDIN_FLOOR_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${RUDIN_FLOOR_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-rudin-floor] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv 2>/dev/null || true

# Stage 0: dispatch claim verification.
log "Stage 0: dispatch claim verification (lane=$LANE_ID; instance/job=$DISPATCH_INSTANCE_JOB_ID; platform=$DISPATCH_PLATFORM)"
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: RUDIN_FLOOR_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
        status="completed_rudin_floor_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_rudin_floor_claim_verification_rc_${rc}"
    else
        status="failed_rudin_floor_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_rudin_floor_interpretable_ml" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$RUDIN_FLOOR_OUTPUT_DIR smoke=${SMOKE_ONLY:-0} device=$RUDIN_FLOOR_DEVICE" \
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
verify_active_dispatch_claim

# Stage 1: canonical bootstrap (delegated to scripts/remote_archive_only_eval.sh).
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel per Catalog #163 prevents the
# sourced script's main flow from running to completion in this caller shell.
log "Stage 1: canonical bootstrap (delegated)"
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
    if declare -f bootstrap_runtime_deps >/dev/null 2>&1; then
        bootstrap_runtime_deps
    else
        log "WARNING: bootstrap_runtime_deps function not found after sourcing remote_archive_only_eval.sh"
    fi
fi

# Resolve PYBIN if not set.
if [ -z "$PYBIN" ]; then
    if [ -x "$WORKSPACE/.venv/bin/python" ]; then
        PYBIN="$WORKSPACE/.venv/bin/python"
    else
        PYBIN="$(command -v python3 || command -v python || echo python)"
    fi
fi
log "Stage 1 DONE: PYBIN=$PYBIN"

# Stage 2: trainer invocation (smoke or full).
# At landing: dispatch_enabled:false + _full_main raises NotImplementedError.
# The trainer aborts cleanly if --smoke is NOT passed and the council gate
# is unlifted. The driver is the design-time placeholder for Phase 2 lift.
log "Stage 2: invoke trainer (substrate=$TAG; lane=$LANE_ID)"
TRAINER_RC=0
if [ "${SMOKE_ONLY:-0}" = "1" ]; then
    log "Stage 2: SMOKE mode (canonical rule-list + archive roundtrip + inflate)"
    "$PYBIN" "$WORKSPACE/experiments/train_substrate_rudin_floor_interpretable_ml.py" \
        --video-path "$RUDIN_FLOOR_VIDEO_PATH" \
        --output-dir "$RUDIN_FLOOR_OUTPUT_DIR" \
        --upstream-dir "$RUDIN_FLOOR_UPSTREAM_DIR" \
        --device "$RUDIN_FLOOR_DEVICE" \
        --epochs "$RUDIN_FLOOR_EPOCHS" \
        --seed "$RUDIN_FLOOR_SEED" \
        --smoke 2>&1 | tee -a "$LOG_DIR/trainer.log" || TRAINER_RC=$?
else
    log "Stage 2: FULL mode (will raise NotImplementedError at landing per Catalog #240)"
    "$PYBIN" "$WORKSPACE/experiments/train_substrate_rudin_floor_interpretable_ml.py" \
        --video-path "$RUDIN_FLOOR_VIDEO_PATH" \
        --output-dir "$RUDIN_FLOOR_OUTPUT_DIR" \
        --upstream-dir "$RUDIN_FLOOR_UPSTREAM_DIR" \
        --device "$RUDIN_FLOOR_DEVICE" \
        --epochs "$RUDIN_FLOOR_EPOCHS" \
        --seed "$RUDIN_FLOOR_SEED" 2>&1 | tee -a "$LOG_DIR/trainer.log" || TRAINER_RC=$?
fi
log "Stage 2 DONE: trainer rc=$TRAINER_RC"

# Stage 3: provenance.json emission (per Catalog L `check_remote_scripts_write_provenance`).
log "Stage 3: write provenance.json"
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "substrate_tag": "$TAG",
  "design_memo": ".omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md",
  "horizon_class": "asymptotic_pursuit",
  "research_only": true,
  "dispatch_enabled_at_landing": false,
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "dispatch_platform": "$DISPATCH_PLATFORM",
  "completed_at_utc": "$(date -u +%FT%TZ)",
  "trainer_rc": $TRAINER_RC,
  "trainer_path": "experiments/train_substrate_rudin_floor_interpretable_ml.py",
  "smoke_only_env": "${SMOKE_ONLY:-0}",
  "video_path": "$RUDIN_FLOOR_VIDEO_PATH",
  "output_dir": "$RUDIN_FLOOR_OUTPUT_DIR",
  "epochs": $RUDIN_FLOOR_EPOCHS,
  "seed": $RUDIN_FLOOR_SEED,
  "device": "$RUDIN_FLOOR_DEVICE",
  "score_claim_valid": false,
  "score_axis": "diagnostic_cpu",
  "promotion_eligible": false,
  "rank_or_kill_eligible": false,
  "result_review_blockers": [
    "l1_scaffold_no_empirical_anchor",
    "_full_main_raises_NotImplementedError_pending_phase_2_council",
    "recipe_dispatch_enabled_false_at_landing"
  ]
}
EOF
log "Stage 3 DONE: provenance.json written to $PROVENANCE"

# Stage 4: completion marker (NEVER tagged contest-CUDA at L1 SCAFFOLD).
if [ "$TRAINER_RC" = "0" ]; then
    log "LANE_RUDIN_FLOOR_DONE [diagnostic-CPU] (L1 SCAFFOLD smoke; no contest-axis claim)"
else
    log "LANE_RUDIN_FLOOR_FAILED rc=$TRAINER_RC (L1 SCAFFOLD; expected at landing per Catalog #240)"
fi

exit "$TRAINER_RC"
