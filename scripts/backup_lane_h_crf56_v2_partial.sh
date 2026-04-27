#!/bin/bash
# Periodic backup of Lane H v2 partial artifacts from a Vast.ai instance to
# a local persistent path. Lane H has no pose TTO, so the only "partial"
# state worth backing up is the build.log + auth_eval.log streams.
#
# Usage:
#   bash scripts/backup_lane_h_crf56_v2_partial.sh <ssh_host> <ssh_port> <local_dir>
set -euo pipefail

SSH_HOST="${1:?ssh host required}"
SSH_PORT="${2:?ssh port required}"
LOCAL_DIR="${3:?local backup dir required}"

mkdir -p "$LOCAL_DIR"

while true; do
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_h_crf56_v2_results/run.log" \
        "$LOCAL_DIR/run.log" 2>/dev/null || echo "[$(date -u +%FT%TZ)] rsync skipped (instance unreachable)"
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_h_crf56_v2_results/build.log" \
        "$LOCAL_DIR/build.log" 2>/dev/null || true
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_h_crf56_v2_results/auth_eval.log" \
        "$LOCAL_DIR/auth_eval.log" 2>/dev/null || true
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_h_crf56_v2_results/archive_crf56.zip" \
        "$LOCAL_DIR/archive_crf56.zip" 2>/dev/null || true
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_h_crf56_v2_results/eval_work/contest_auth_eval.json" \
        "$LOCAL_DIR/contest_auth_eval.json" 2>/dev/null || true
    sleep 60
done
