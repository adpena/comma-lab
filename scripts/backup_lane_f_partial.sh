#!/bin/bash
# Periodic backup of Lane F's partial QAT artifacts from a Vast.ai instance
# to a local persistent path. Runs every 60s. If the remote instance dies,
# the latest QAT checkpoint + log are preserved locally and a future Lane F
# run can resume from the partial.
#
# Usage:
#   bash scripts/backup_lane_f_partial.sh <ssh_host> <ssh_port> <local_dir>
#
# Per CLAUDE.md (feedback_zip_dep_bootstrap_trap): -e is required to surface
# programming bugs (typos, missing dirs). Transient SSH errors handled per-cmd.
set -euo pipefail

SSH_HOST="${1:?ssh host required}"
SSH_PORT="${2:?ssh port required}"
LOCAL_DIR="${3:?local backup dir required}"

mkdir -p "$LOCAL_DIR"

while true; do
    # QAT artifacts: best float checkpoint + final FP4 binary + run log
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_b_results/qat/" \
        "$LOCAL_DIR/qat/" 2>/dev/null || echo "[$(date -u +%FT%TZ)] rsync qat/ skipped"
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_b_results/qat.log" \
        "$LOCAL_DIR/qat.log" 2>/dev/null || true
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_b_results/run.log" \
        "$LOCAL_DIR/run.log" 2>/dev/null || true
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_b_results/auth_eval.log" \
        "$LOCAL_DIR/auth_eval.log" 2>/dev/null || true
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_b_results/archive_lane_b.zip" \
        "$LOCAL_DIR/archive_lane_b.zip" 2>/dev/null || true
    if [ -f "$LOCAL_DIR/qat.log" ]; then
        EPOCH=$(grep -oE 'epoch [0-9]+' "$LOCAL_DIR/qat.log" 2>/dev/null | tail -1 || true)
        echo "[$(date -u +%FT%TZ)] backed up: ${EPOCH:-no-epoch-yet}"
    fi
    sleep 60
done
