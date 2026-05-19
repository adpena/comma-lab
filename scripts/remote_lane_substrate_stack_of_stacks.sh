#!/bin/bash
# Remote lane script: stack-of-stacks single-arm A1 passthrough canary.
#
# Trainer: experiments/train_substrate_stack_of_stacks.py
# Lane: lane_stack_of_stacks_composition_implementation_20260513
#
# This script is intentionally a custody/runtime canary, not a score-lowering
# full-stack dispatch. It builds the byte-closed SOS1-wrapped A1 passthrough
# packet, then runs canonical [contest-CUDA] auth eval through
# scripts/remote_archive_only_eval.sh. Full multi-arm score lowering stays
# blocked until score-aware selectors and frame-level stitching land.
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
LANE_ID="${LANE_ID:-lane_stack_of_stacks_composition_implementation_20260513}"
TAG="${TAG:-substrate_stack_of_stacks_canary}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_stack_of_stacks_results}"
DISPATCH_INSTANCE_JOB_ID="${STACK_OF_STACKS_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"

# === Catalog #204 CONTEST_AUTH_EVAL_DURABLE_OUTPUT discipline ===
# Bug-class anchor: E.8 SGLD #2 dispatch fc-01KRZCSQ7FPVMSAXZQDSZJCTN4 (2026-05-19)
# trainer rc=0 archive built sha=110cfaa3 size=179008, then auth_eval rc=1
# because evidence path was under /tmp/pact (Modal worker workspace) and
# experiments/contest_auth_eval.py correctly refuses temp-storage evidence
# per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" non-negotiable.
# Sister regression to STC v2 2026-05-14 anchor; canonical fix lives in
# scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh.
# On Modal workers (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts
# under the /modal_results volume so modal_train_lane.py harvests durable
# provider output and contest_auth_eval can produce promotable score evidence.
if [ -n "${STACK_OF_STACKS_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$STACK_OF_STACKS_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
    LOG_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/lane_stack_of_stacks_results"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
DISPATCH_CLAIMS_PATH="${STACK_OF_STACKS_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
PROVENANCE="$LOG_DIR/provenance.json"
RUN_RECORD="$LOG_DIR/run_record.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
HEARTBEAT_PID=""
TERMINAL_CLAIM_WRITTEN=0

STACK_OF_STACKS_BASE_ARCHIVE="${STACK_OF_STACKS_BASE_ARCHIVE:-$WORKSPACE/submissions/a1/archive.zip}"
STACK_OF_STACKS_BASE_RUNTIME_DIR="${STACK_OF_STACKS_BASE_RUNTIME_DIR:-$WORKSPACE/submissions/a1}"
STACK_OF_STACKS_VIDEO_PATH="${STACK_OF_STACKS_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
STACK_OF_STACKS_UPSTREAM_DIR="${STACK_OF_STACKS_UPSTREAM_DIR:-$WORKSPACE/upstream}"
STACK_OF_STACKS_DEVICE="${STACK_OF_STACKS_DEVICE:-cuda}"
STACK_OF_STACKS_EPOCHS="${STACK_OF_STACKS_EPOCHS:-0}"
STACK_OF_STACKS_MIDDLE_ARM_SUBSTRATE_IDS="${STACK_OF_STACKS_MIDDLE_ARM_SUBSTRATE_IDS:-a1}"
STACK_OF_STACKS_OUTER_K="${STACK_OF_STACKS_OUTER_K:-1}"
STACK_OF_STACKS_MAX_TOTAL_ARCHIVE_BYTES="${STACK_OF_STACKS_MAX_TOTAL_ARCHIVE_BYTES:-250000}"
STACK_OF_STACKS_LANGEVIN_T_INIT_CAP="${STACK_OF_STACKS_LANGEVIN_T_INIT_CAP:-1.0}"
STACK_OF_STACKS_LANGEVIN_POLISH_EPOCHS="${STACK_OF_STACKS_LANGEVIN_POLISH_EPOCHS:-100}"

log() { echo "[lane-stack-of-stacks] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

write_remote_record() {
    local status="$1"
    local rc="${2:-}"
    local py="${PYBIN:-}"
    if [ -z "$py" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
        py="$WORKSPACE/.venv/bin/python"
    fi
    if [ -z "$py" ]; then
        py="python3"
    fi
    local git_head
    git_head="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    local git_dirty
    git_dirty="$(git status --short 2>/dev/null || true)"
    "$py" - "$PROVENANCE" "$RUN_RECORD" "$status" "$rc" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

provenance_path, run_record_path, status, rc = sys.argv[1:5]
now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
payload = {
    "schema_version": "remote_stack_of_stacks_run.v1",
    "updated_at_utc": now,
    "status": status,
    "returncode": int(rc) if rc not in ("", None) else None,
    "lane_id": os.environ.get("LANE_ID", "lane_stack_of_stacks_composition_implementation_20260513"),
    "tag": os.environ.get("TAG", "substrate_stack_of_stacks_canary"),
    "dispatch_instance_job_id": os.environ.get("DISPATCH_INSTANCE_JOB_ID", ""),
    "dispatch_platform": os.environ.get("DISPATCH_PLATFORM", "modal"),
    "workspace": os.environ.get("WORKSPACE", ""),
    "output_dir": os.environ.get("OUTPUT_DIR", ""),
    "archive_path": os.environ.get("ARCHIVE_PATH", ""),
    "inflate_sh": os.environ.get("INFLATE_SH", ""),
    "auth_eval_dir": os.environ.get("AUTH_EVAL_LOG_DIR", ""),
    "base_archive": os.environ.get("STACK_OF_STACKS_BASE_ARCHIVE", ""),
    "base_runtime_dir": os.environ.get("STACK_OF_STACKS_BASE_RUNTIME_DIR", ""),
    "video_path": os.environ.get("STACK_OF_STACKS_VIDEO_PATH", ""),
    "upstream_dir": os.environ.get("STACK_OF_STACKS_UPSTREAM_DIR", ""),
    "device": os.environ.get("STACK_OF_STACKS_DEVICE", ""),
    "epochs": os.environ.get("STACK_OF_STACKS_EPOCHS", ""),
    "middle_arm_substrate_ids": os.environ.get("STACK_OF_STACKS_MIDDLE_ARM_SUBSTRATE_IDS", ""),
    "outer_stack_k": os.environ.get("STACK_OF_STACKS_OUTER_K", ""),
    "max_total_archive_bytes": os.environ.get("STACK_OF_STACKS_MAX_TOTAL_ARCHIVE_BYTES", ""),
    "langevin_t_init_cap": os.environ.get("STACK_OF_STACKS_LANGEVIN_T_INIT_CAP", ""),
    "langevin_polish_epochs": os.environ.get("STACK_OF_STACKS_LANGEVIN_POLISH_EPOCHS", ""),
    "predicted_band": [0.190, 0.210],
    "predicted_band_basis": "single-arm A1 passthrough stack-of-stacks canary; score-neutral or slight rate penalty before multi-arm selector work",
    "git_head": os.environ.get("STACK_OF_STACKS_GIT_HEAD", ""),
    "git_dirty_status": os.environ.get("STACK_OF_STACKS_GIT_DIRTY_STATUS", ""),
    "score_claim": False,
    "promotion_eligible": False,
}
for path in (provenance_path, run_record_path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
PY
}

STACK_OF_STACKS_GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STACK_OF_STACKS_GIT_DIRTY_STATUS="$(git status --short 2>/dev/null || true)"
export WORKSPACE OUTPUT_DIR TAG LANE_ID DISPATCH_INSTANCE_JOB_ID DISPATCH_PLATFORM
export STACK_OF_STACKS_GIT_HEAD STACK_OF_STACKS_GIT_DIRTY_STATUS
write_remote_record "started" ""

if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: STACK_OF_STACKS_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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
close_dispatch_claim() {
    local rc="$1"
    local claim_status
    local claim_notes
    if [ "$TERMINAL_CLAIM_WRITTEN" = "1" ]; then
        return 0
    fi
    if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
        return 0
    fi
    if [ "$rc" -eq 0 ]; then
        claim_status="completed_stack_of_stacks_auth_eval"
        claim_notes="terminal stack-of-stacks exact-eval canary claim; score_claim=false; artifact=$RUN_RECORD"
    else
        claim_status="failed_stack_of_stacks_rc_${rc}"
        claim_notes="terminal stack-of-stacks exact-eval canary claim; score_claim=false; artifact=$RUN_RECORD"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "codex:gpt-5.5" \
        --status "$claim_status" \
        --notes "$claim_notes" \
        --force >> "$LOG_DIR/run.log" 2>&1 || true
    TERMINAL_CLAIM_WRITTEN=1
}

finalize_remote_run() {
    local rc=$?
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    write_remote_record "exited" "$rc" || true
    close_dispatch_claim "$rc" || true
}

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
trap finalize_remote_run EXIT

log "stage_0_nvdec_probe_begin"
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed; refusing to spend GPU on this host"
    exit 2
}
log "stage_0_nvdec_probe_done"

if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap/eval script missing at scripts/remote_archive_only_eval.sh"
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

log "stage_1b_required_input_files_validate_begin"
"$PYBIN" "$WORKSPACE/tools/validate_dispatch_required_inputs.py" \
    --trainer "$WORKSPACE/experiments/train_substrate_stack_of_stacks.py" \
    --flag-value="--base-archive=$STACK_OF_STACKS_BASE_ARCHIVE" \
    --flag-value="--video-path=$STACK_OF_STACKS_VIDEO_PATH" || {
    log "FATAL: Catalog #152 required input file validation failed; refusing dispatch"
    exit 25
}
log "stage_1b_required_input_files_validate_done"

(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$HEARTBEAT"
        sleep 300
    done
) &
HEARTBEAT_PID=$!

log "stage_2_trainer_invoke_begin mode=single_arm_a1_passthrough epochs=$STACK_OF_STACKS_EPOCHS"
set +e
"$PYBIN" experiments/train_substrate_stack_of_stacks.py \
    --base-archive "$STACK_OF_STACKS_BASE_ARCHIVE" \
    --base-runtime-dir "$STACK_OF_STACKS_BASE_RUNTIME_DIR" \
    --video-path "$STACK_OF_STACKS_VIDEO_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$STACK_OF_STACKS_EPOCHS" \
    --upstream-dir "$STACK_OF_STACKS_UPSTREAM_DIR" \
    --device "$STACK_OF_STACKS_DEVICE" \
    --middle-arm-substrate-ids "$STACK_OF_STACKS_MIDDLE_ARM_SUBSTRATE_IDS" \
    --outer-stack-k "$STACK_OF_STACKS_OUTER_K" \
    --max-total-archive-bytes "$STACK_OF_STACKS_MAX_TOTAL_ARCHIVE_BYTES" \
    --langevin-t-init "$STACK_OF_STACKS_LANGEVIN_T_INIT_CAP" \
    --langevin-polish-epochs "$STACK_OF_STACKS_LANGEVIN_POLISH_EPOCHS" \
    --lane-id "$LANE_ID" \
    --dispatch-instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
    --dispatch-platform "$DISPATCH_PLATFORM" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
log "stage_2_trainer_invoke_done rc=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    exit "$TRAIN_RC"
fi

ARCHIVE_PATH="$OUTPUT_DIR/submission_dir/archive.zip"
INFLATE_SH="$OUTPUT_DIR/submission_dir/inflate.sh"
export ARCHIVE_PATH
export INFLATE_SH
if [ ! -f "$ARCHIVE_PATH" ]; then
    log "FATAL: trainer did not emit archive at $ARCHIVE_PATH"
    exit 26
fi
if [ ! -x "$INFLATE_SH" ]; then
    log "FATAL: trainer did not emit executable inflate.sh at $INFLATE_SH"
    exit 26
fi

log "stage_3_auth_eval_begin archive=$ARCHIVE_PATH inflate_sh=$INFLATE_SH"
export ARCHIVE_LABEL="$TAG"
export PREDICTED_LOW="0.190"
export PREDICTED_HIGH="0.210"
export CONTROLLED_BASELINE="A1 single-arm passthrough canary; no score claim before exact eval"
export LOG_DIR="$LOG_DIR/auth_eval"
export AUTH_EVAL_LOG_DIR="$LOG_DIR"
export KEEP_EVAL_WORK="${KEEP_EVAL_WORK:-1}"
if [ "${STACK_OF_STACKS_AUTH_EVAL_REQUIRE_CONTEST_CUDA:-0}" = "1" ]; then
    unset MODAL_AUTH_EVAL_ADVISORY_ONLY
fi
set +e
bash "$WORKSPACE/scripts/remote_archive_only_eval.sh"
AUTH_RC=$?
set -e
log "LANE_STACK_OF_STACKS_CANARY_DONE [contest-CUDA] auth_eval=$LOG_DIR/contest_auth_eval.json archive=$ARCHIVE_PATH rc=$AUTH_RC"
write_remote_record "auth_eval_completed" "$AUTH_RC"
exit "$AUTH_RC"
