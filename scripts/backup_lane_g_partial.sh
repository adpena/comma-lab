#!/bin/bash
# Periodic backup of Lane A's partial pose checkpoint from a Vast.ai instance
# to a local persistent path. Runs every 60s. If the remote instance dies,
# the latest partial is preserved locally and a future Lane A run can resume.
#
# Usage:
#   bash scripts/backup_lane_a_partial.sh <ssh_host> <ssh_port> <local_dir>
# Per CLAUDE.md non-negotiable (feedback_zip_dep_bootstrap_trap): `-uo` without
# `-e` allows silent error cascades. Transient SSH errors are handled per-command
# below via `|| echo skipped` / `|| true`, so `-e` is safe and catches programming
# errors (missing dirs, typoed paths, broken pipes) loudly instead of looping
# forever on a broken backup.
set -euo pipefail

SSH_HOST="${1:?ssh host required}"
SSH_PORT="${2:?ssh port required}"
LOCAL_DIR="${3:?local backup dir required}"

mkdir -p "$LOCAL_DIR"

while true; do
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_g_v2_results/optimized_poses_partial.pt" \
        "$LOCAL_DIR/optimized_poses_partial.pt" 2>/dev/null || echo "[$(date -u +%FT%TZ)] rsync skipped (instance unreachable)"
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_g_v2_results/optimized_poses_partial.meta" \
        "$LOCAL_DIR/optimized_poses_partial.meta" 2>/dev/null || true
    rsync -az --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT" \
        "root@$SSH_HOST:/workspace/pact/lane_g_v2_results/optimize_poses.log" \
        "$LOCAL_DIR/optimize_poses.log" 2>/dev/null || true
    if [ -f "$LOCAL_DIR/optimized_poses_partial.meta" ]; then
        # `|| true`: meta may be partially-written by remote rsync; non-match
        # under `set -e` would abort the polling loop. Empty N is acceptable
        # (we just skip the print), but a missing-file is a programming bug.
        N=$(grep -oE '"n_pairs_complete":\s*[0-9]+' "$LOCAL_DIR/optimized_poses_partial.meta" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
        echo "[$(date -u +%FT%TZ)] backed up: n_pairs_complete=${N:-unknown}"
    fi
    sleep 60
done
