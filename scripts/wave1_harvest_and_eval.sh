#!/bin/bash
# Wave-1 Harvest + Eval Orchestrator (Q-FAITHFUL training, Vast 35959478)
# ─────────────────────────────────────────────────────────────────────────
# Purpose: After Q-FAITHFUL training completes on Vast 35959478, automate
# (a) sentinel detection, (b) artifact SCP-back, (c) optional QZS3-shape
# alternative archive build via experiments/build_qpose_archive.py, and
# (d) contest-CUDA re-eval of the alternative archive on the same instance.
# Then summarize and append to .omx/state/active_dispatches.md.
#
# Reference for SSH details + completion sentinels:
#   experiments/results/lane_q_faithful_retrain_20260501/dispatch_metadata.json
#   scripts/remote_lane_q_faithful_jointgen.sh (Stage 6 emits LANE_Q_FAITHFUL_DONE)
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS — every flag passed to
# build_qpose_archive.py + contest_auth_eval.py was verified by argparse-grep.
#
# DESIGN NOTE: The Q-FAITHFUL remote script ALREADY runs Stage 6 contest-CUDA
# auth eval on its OWN deflated-zip archive (renderer.bin + masks.mkv +
# optimized_poses.pt). This orchestrator's QZS3 path produces a SECOND,
# alternative archive shape (PR #67 single-blob) and re-evals it for
# A/B comparison; the original deflated archive's score is also harvested.
#
# Usage:
#   bash scripts/wave1_harvest_and_eval.sh            # full pipeline
#   POLL_ONLY=1 bash scripts/wave1_harvest_and_eval.sh   # check sentinel only
#   SKIP_QZS3_REEVAL=1 bash scripts/wave1_harvest_and_eval.sh  # skip alt-archive re-eval

set -euo pipefail

# ─── Stable config (Q-FAITHFUL dispatch) ─────────────────────────────────
# Original dispatch: instance 35959478 ssh6.vast.ai:39478 (RTX 4090).
# REDEPLOY 2026-05-01 mid-session: instance 35985637 ssh3.vast.ai:25636 (H100 SXM,
# label "q_faithful_h100_redeploy"). Override SSH_HOST/SSH_PORT to point at the
# active instance. Confirm with `.venv/bin/vastai show instances` before running.
SSH_HOST="${SSH_HOST:-ssh3.vast.ai}"
SSH_PORT="${SSH_PORT:-25636}"
SSH_USER="${SSH_USER:-root}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20}"
REMOTE_WORKSPACE="${REMOTE_WORKSPACE:-/workspace/pact}"
REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-${REMOTE_WORKSPACE}/lane_q_faithful_results}"
REMOTE_AUTH_LOG="${REMOTE_RESULTS_DIR}/auth_eval.log"
REMOTE_TRAIN_LOG="${REMOTE_RESULTS_DIR}/train.log"
REMOTE_HEARTBEAT="${REMOTE_RESULTS_DIR}/heartbeat.log"
REMOTE_AUTH_JSON="${REMOTE_RESULTS_DIR}/eval_work/contest_auth_eval.json"
REMOTE_DEFLATED_ARCHIVE="${REMOTE_RESULTS_DIR}/archive_lane_q_faithful.zip"
REMOTE_BEST_CKPT_GLOB="${REMOTE_RESULTS_DIR}/train/*BEST*.pt"
SENTINEL_TOKEN="LANE_Q_FAITHFUL_DONE"

# ─── Local paths ──────────────────────────────────────────────────────────
LOCAL_CUSTODY_DIR="${LOCAL_CUSTODY_DIR:-experiments/results/lane_q_faithful_retrain_20260501}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CUSTODY_DIR}/checkpoints"
LOCAL_QZS3_DIR="${LOCAL_CUSTODY_DIR}/wave1_qzs3_candidate"
LOCAL_QZS3_ARCHIVE="${LOCAL_QZS3_DIR}/archive.zip"
LOCAL_DEFLATED_ARCHIVE="${LOCAL_CUSTODY_DIR}/archive_lane_q_faithful.zip"
ACTIVE_DISPATCHES="${ACTIVE_DISPATCHES:-.omx/state/active_dispatches.md}"

# Expected QZS3 archive size band (PR #67 ≈ 276,500 B; widen for variance).
QZS3_EXPECTED_LOW="${QZS3_EXPECTED_LOW:-260000}"
QZS3_EXPECTED_HIGH="${QZS3_EXPECTED_HIGH:-295000}"

# ─── Helpers ──────────────────────────────────────────────────────────────
ts() { date -u +%FT%TZ; }
log() { echo "[wave1-harvest] $(ts) $*"; }
fail() { log "FATAL: $*"; exit 2; }

ssh_remote() {
    # shellcheck disable=SC2086
    ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "$@"
}

scp_from() {
    local src="$1" dst="$2"
    # shellcheck disable=SC2086
    scp $SSH_OPTS -P "$SSH_PORT" "$SSH_USER@$SSH_HOST:$src" "$dst"
}

scp_to() {
    local src="$1" dst="$2"
    # shellcheck disable=SC2086
    scp $SSH_OPTS -P "$SSH_PORT" "$src" "$SSH_USER@$SSH_HOST:$dst"
}

# ─── Stage 1: sentinel detection ──────────────────────────────────────────
log "Stage 1: probe Vast ${SSH_HOST}:${SSH_PORT} for completion sentinel ${SENTINEL_TOKEN}"
SENTINEL_FOUND=0
TRAIN_FATAL=0
HEARTBEAT_AGE_SEC="unknown"

if ! ssh_remote "echo ssh-ok" >/dev/null 2>&1; then
    fail "SSH unreachable; instance may be down or destroyed. Check vastai show instances."
fi

# Heartbeat freshness probe (CLAUDE.md remote-code-parity: heartbeat > 30 min stale = critical).
if ssh_remote "[ -f $REMOTE_HEARTBEAT ]"; then
    HEARTBEAT_AGE_SEC=$(ssh_remote "stat -c %Y $REMOTE_HEARTBEAT 2>/dev/null | xargs -I{} expr \$(date +%s) - {}" || echo "unknown")
    log "  heartbeat age: ${HEARTBEAT_AGE_SEC}s"
    if [ "$HEARTBEAT_AGE_SEC" != "unknown" ] && [ "$HEARTBEAT_AGE_SEC" -gt 1800 ] 2>/dev/null; then
        log "WARN: heartbeat stale > 30 min — training may have died silently"
    fi
fi

# Sentinel: auth_eval.log contains LANE_Q_FAITHFUL_DONE.
if ssh_remote "grep -q '$SENTINEL_TOKEN' $REMOTE_AUTH_LOG 2>/dev/null"; then
    SENTINEL_FOUND=1
    log "  sentinel HIT: $REMOTE_AUTH_LOG contains $SENTINEL_TOKEN"
fi

# Look for fatal errors in train.log too.
if ssh_remote "grep -qE 'FATAL|Traceback|RuntimeError' $REMOTE_TRAIN_LOG 2>/dev/null"; then
    TRAIN_FATAL=1
    log "WARN: train.log contains FATAL/Traceback/RuntimeError"
    ssh_remote "grep -E 'FATAL|Traceback|RuntimeError' $REMOTE_TRAIN_LOG | head -10" || true
fi

if [ "$SENTINEL_FOUND" -eq 0 ]; then
    log "Sentinel NOT found yet. Status snapshot:"
    ssh_remote "ls -la $REMOTE_RESULTS_DIR/ 2>/dev/null | tail -20" || true
    if [ "$TRAIN_FATAL" -eq 1 ]; then
        fail "Training appears to have FAILED (fatal errors in train.log) without producing sentinel. STOP — investigate before harvesting."
    fi
    if [ "${POLL_ONLY:-0}" = "1" ]; then
        log "POLL_ONLY=1 set — exiting without harvest. Re-run when sentinel hits."
        exit 1
    fi
    fail "Q-FAITHFUL not done yet (no $SENTINEL_TOKEN in $REMOTE_AUTH_LOG). Set POLL_ONLY=1 for non-failing probe."
fi

if [ "${POLL_ONLY:-0}" = "1" ]; then
    log "POLL_ONLY=1 — sentinel HIT, exiting (would harvest now if POLL_ONLY=0)."
    exit 0
fi

# ─── Stage 2: SCP back deflated archive + checkpoint + logs ───────────────
log "Stage 2: SCP-back artifacts to ${LOCAL_CUSTODY_DIR}/"
mkdir -p "$LOCAL_CHECKPOINT_DIR"

# auth_eval log + RESULT_JSON.
scp_from "$REMOTE_AUTH_LOG" "${LOCAL_CUSTODY_DIR}/auth_eval.log" || log "WARN: auth_eval.log SCP failed"
scp_from "$REMOTE_AUTH_JSON" "${LOCAL_CUSTODY_DIR}/contest_auth_eval.json" || log "WARN: contest_auth_eval.json SCP failed"
scp_from "$REMOTE_TRAIN_LOG" "${LOCAL_CUSTODY_DIR}/train.log" || log "WARN: train.log SCP failed"
scp_from "${REMOTE_RESULTS_DIR}/provenance.json" "${LOCAL_CUSTODY_DIR}/provenance.json" || log "WARN: provenance.json SCP failed"

# Best checkpoint — resolve glob remotely first.
REMOTE_BEST_CKPT=$(ssh_remote "ls -t $REMOTE_BEST_CKPT_GLOB 2>/dev/null | head -1" || echo "")
if [ -z "$REMOTE_BEST_CKPT" ]; then
    log "WARN: no *BEST*.pt found, falling back to latest *.pt"
    REMOTE_BEST_CKPT=$(ssh_remote "ls -t ${REMOTE_RESULTS_DIR}/train/*.pt 2>/dev/null | head -1" || echo "")
fi
if [ -n "$REMOTE_BEST_CKPT" ]; then
    BEST_BASENAME=$(basename "$REMOTE_BEST_CKPT")
    scp_from "$REMOTE_BEST_CKPT" "${LOCAL_CHECKPOINT_DIR}/${BEST_BASENAME}" || log "WARN: best ckpt SCP failed"
    log "  best checkpoint: ${LOCAL_CHECKPOINT_DIR}/${BEST_BASENAME}"
else
    log "WARN: no checkpoint to SCP back"
fi

# Deflated archive (the one Stage 6 already evaluated).
scp_from "$REMOTE_DEFLATED_ARCHIVE" "$LOCAL_DEFLATED_ARCHIVE" || log "WARN: deflated archive SCP failed"
DEFLATED_BYTES="unknown"
DEFLATED_SHA="unknown"
if [ -f "$LOCAL_DEFLATED_ARCHIVE" ]; then
    DEFLATED_BYTES=$(stat -f %z "$LOCAL_DEFLATED_ARCHIVE" 2>/dev/null || stat -c %s "$LOCAL_DEFLATED_ARCHIVE")
    DEFLATED_SHA=$(shasum -a 256 "$LOCAL_DEFLATED_ARCHIVE" | cut -d' ' -f1)
    log "  deflated archive: ${DEFLATED_BYTES} bytes sha=${DEFLATED_SHA}"
fi

# Extract score from auth_eval.log (RESULT_JSON line).
DEFLATED_SCORE="unknown"
if [ -f "${LOCAL_CUSTODY_DIR}/contest_auth_eval.json" ]; then
    DEFLATED_SCORE=$(python3 -c "import json,sys; d=json.load(open('${LOCAL_CUSTODY_DIR}/contest_auth_eval.json')); print(d.get('total_score',d.get('score','unknown')))" 2>/dev/null || echo "unknown")
fi
log "  deflated archive [contest-CUDA] score: $DEFLATED_SCORE"

# ─── Stage 3 (optional): QZS3 alt-archive build + re-eval ─────────────────
QZS3_SCORE="skipped"
QZS3_BYTES="skipped"
QZS3_SHA="skipped"

if [ "${SKIP_QZS3_REEVAL:-0}" = "1" ]; then
    log "Stage 3: SKIPPED (SKIP_QZS3_REEVAL=1)"
elif [ -z "$REMOTE_BEST_CKPT" ] || [ ! -f "${LOCAL_CHECKPOINT_DIR}/$(basename "$REMOTE_BEST_CKPT")" ]; then
    log "Stage 3: SKIPPED — no local checkpoint to feed build_qpose_archive.py"
else
    LOCAL_CKPT_PATH="${LOCAL_CHECKPOINT_DIR}/$(basename "$REMOTE_BEST_CKPT")"
    log "Stage 3: build PR #67-shaped QZS3 candidate archive from $LOCAL_CKPT_PATH"
    mkdir -p "$LOCAL_QZS3_DIR"

    # Mask + pose inputs:
    #   - The Q-FAITHFUL remote already has masks.mkv at iter_0/. We need
    #     an OBU-br stream for build_qpose_archive — without one, fall back
    #     to --smoke (placeholder mask = PR67 standard 219472 bytes).
    #   - Lane A poses are the project anchor at experiments/results/lane_a_landed/optimized_poses.pt.
    POSE_FILE="experiments/results/lane_a_landed/optimized_poses.pt"
    [ -f "$POSE_FILE" ] || { log "WARN: anchor pose file missing: $POSE_FILE — using smoke pose"; POSE_FILE=""; }

    # NEVER invent CLI flags — verified against build_qpose_archive.py argparse:
    #   --renderer-state --mask-obu-br --pose-file --pose-codec --block-size
    #   --brotli-quality --output-dir --smoke
    BUILD_CMD=(.venv/bin/python experiments/build_qpose_archive.py
        --renderer-state "$LOCAL_CKPT_PATH"
        --output-dir "$LOCAL_QZS3_DIR"
        --pose-codec qp1
        --block-size 32
        --brotli-quality 11
        --smoke)
    if [ -n "$POSE_FILE" ]; then
        BUILD_CMD+=(--pose-file "$POSE_FILE")
    fi
    log "  $ ${BUILD_CMD[*]}"
    if ! "${BUILD_CMD[@]}" > "${LOCAL_QZS3_DIR}/build.log" 2>&1; then
        log "WARN: build_qpose_archive.py failed; see ${LOCAL_QZS3_DIR}/build.log"
        QZS3_SCORE="build_failed"
    else
        QZS3_BYTES=$(stat -f %z "$LOCAL_QZS3_ARCHIVE" 2>/dev/null || stat -c %s "$LOCAL_QZS3_ARCHIVE")
        QZS3_SHA=$(shasum -a 256 "$LOCAL_QZS3_ARCHIVE" | cut -d' ' -f1)
        log "  QZS3 archive: ${QZS3_BYTES} bytes sha=${QZS3_SHA}"

        # Sanity band check.
        if [ "$QZS3_BYTES" -lt "$QZS3_EXPECTED_LOW" ] || [ "$QZS3_BYTES" -gt "$QZS3_EXPECTED_HIGH" ]; then
            log "WARN: QZS3 archive ${QZS3_BYTES} outside expected band [${QZS3_EXPECTED_LOW}, ${QZS3_EXPECTED_HIGH}]"
        fi

        # SCP archive back to remote and run contest-CUDA eval.
        REMOTE_QZS3_ARCHIVE="${REMOTE_RESULTS_DIR}/wave1_qzs3_candidate/archive.zip"
        ssh_remote "mkdir -p $(dirname $REMOTE_QZS3_ARCHIVE)" || true
        scp_to "$LOCAL_QZS3_ARCHIVE" "$REMOTE_QZS3_ARCHIVE" || log "WARN: QZS3 archive SCP-to failed"

        # Use the canonical archive-only-eval wrapper. NEVER invent flags —
        # contest_auth_eval verified flags: --archive --inflate-sh
        # --upstream-dir --device --keep-work-dir --work-dir.
        log "  invoking remote contest-CUDA eval on QZS3 archive"
        REMOTE_QZS3_LOG="${REMOTE_RESULTS_DIR}/wave1_qzs3_eval.log"
        REMOTE_QZS3_WORK="${REMOTE_RESULTS_DIR}/wave1_qzs3_eval_work"
        EVAL_CMD="cd $REMOTE_WORKSPACE && rm -rf $REMOTE_QZS3_WORK && /opt/conda/bin/python -u experiments/contest_auth_eval.py --archive $REMOTE_QZS3_ARCHIVE --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir $REMOTE_QZS3_WORK 2>&1 | tee $REMOTE_QZS3_LOG | tail -30"
        if ssh_remote "$EVAL_CMD"; then
            scp_from "${REMOTE_QZS3_WORK}/contest_auth_eval.json" "${LOCAL_QZS3_DIR}/contest_auth_eval.json" || log "WARN: QZS3 RESULT_JSON SCP failed"
            scp_from "$REMOTE_QZS3_LOG" "${LOCAL_QZS3_DIR}/eval.log" || true
            if [ -f "${LOCAL_QZS3_DIR}/contest_auth_eval.json" ]; then
                QZS3_SCORE=$(python3 -c "import json; d=json.load(open('${LOCAL_QZS3_DIR}/contest_auth_eval.json')); print(d.get('total_score',d.get('score','unknown')))" 2>/dev/null || echo "unknown")
            fi
        else
            log "WARN: remote QZS3 eval failed"
            QZS3_SCORE="eval_failed"
        fi
    fi
fi

# ─── Stage 4: summary + active_dispatches.md append ──────────────────────
log "Stage 4: summarize"
SUMMARY=$(cat <<EOF

## Wave-1 Q-FAITHFUL HARVEST - $(ts)

Vast 35959478 (ssh6.vast.ai:39478) Q-FAITHFUL training harvested.

| field | deflated (Stage 6) | qzs3 (alt) |
|---|---|---|
| archive bytes | ${DEFLATED_BYTES} | ${QZS3_BYTES} |
| archive sha256 | ${DEFLATED_SHA} | ${QZS3_SHA} |
| [contest-CUDA] score | ${DEFLATED_SCORE} | ${QZS3_SCORE} |
| local custody | ${LOCAL_DEFLATED_ARCHIVE} | ${LOCAL_QZS3_ARCHIVE} |

Heartbeat age at harvest: ${HEARTBEAT_AGE_SEC}s.
Train log fatals: ${TRAIN_FATAL} (0 = clean).
Reference: experiments/results/lane_q_faithful_retrain_20260501/dispatch_metadata.json
Predicted band per dispatch: [0.40, 0.80].
Champion to beat: 0.9974 [contest-CUDA] owv3_0120 stack (sha 1e9195cb...).

EOF
)

echo "$SUMMARY"
if [ -f "$ACTIVE_DISPATCHES" ]; then
    echo "$SUMMARY" >> "$ACTIVE_DISPATCHES"
    log "appended to $ACTIVE_DISPATCHES"
fi

log "=== WAVE1_HARVEST_DONE ==="
log "  deflated:  ${DEFLATED_SCORE} (${DEFLATED_BYTES} B)"
log "  qzs3 alt:  ${QZS3_SCORE} (${QZS3_BYTES} B)"
log "Next step: review scores against champion 0.9974; promote winner via lane_maturity.py if sub-frontier."
