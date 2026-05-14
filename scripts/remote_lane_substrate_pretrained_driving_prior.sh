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

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_pretrained_driving_prior_lane_scaffold_20260513"
TAG="${TAG:-substrate_pretrained_driving_prior}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_pretrained_driving_prior_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
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

DISPATCH_INSTANCE_JOB_ID="${DPP_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${DPP_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-dpp] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

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

# Stage 3: smoke path FIRST (scaffold L0 — only smoke is enabled).
log "Stage 3: smoke distill + pack + parse"
"$PYBIN" "$WORKSPACE/experiments/train_substrate_pretrained_driving_prior.py" \
    --smoke \
    --video-path "$DPP_VIDEO_PATH" \
    --output-dir "$DPP_OUTPUT_DIR" \
    --upstream-dir "$DPP_UPSTREAM_DIR" \
    --device "$DPP_DEVICE" \
    --epochs "$DPP_EPOCHS" \
    --batch-size "$DPP_BATCH_SIZE" \
    --dataset-name synthetic_test

log "DPP scaffold smoke complete; full training path raises NotImplementedError"
log "Full training requires Phase 2 council approval + real Comma2k19 distillation"
log "Exiting cleanly (rc=0) to signal scaffold-only success."
exit 0
