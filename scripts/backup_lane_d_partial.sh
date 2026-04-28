#!/bin/bash
# Periodic backup of Lane D's partial training artifacts from a Vast.ai instance
# to a local persistent path. Per CLAUDE.md "Vast.ai launch protocol" rule 4:
# spot interruptions happen, save partial results to a persistent location.
#
# Usage:
#   bash scripts/backup_lane_d_partial.sh <ssh_host> <ssh_port> <local_dir>
#
# Artifacts backed up (whichever exist):
#   - lane_d_results/run.log (main bootstrap log)
#   - lane_d_results/train.log (train_renderer.py output)
#   - lane_d_results/train/renderer_lane_d_halfframe_best_fp32.pt (best float)
#   - lane_d_results/train/renderer_lane_d_halfframe_best_fp4.pt (best FP4-packed)
#   - lane_d_results/train/*.pt (any other epoch checkpoints)
#   - lane_d_results/iter_0/renderer.bin (Stage 1b export)
#   - lane_d_results/iter_0/zoom_scalars.pt (zoom warm-start)
#   - lane_d_results/auth_eval_*.json (Stage final auth eval)
#   - lane_d_results/kill_targets.json (smoke-kill metadata)
set -euo pipefail

SSH_HOST="${1:?ssh host required}"
SSH_PORT="${2:?ssh port required}"
LOCAL_DIR="${3:?local backup dir required}"

mkdir -p "$LOCAL_DIR"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p $SSH_PORT"

while true; do
    # Recursive rsync of the whole lane_d_results dir — captures all
    # partial+final artifacts in one go. --partial keeps in-progress files.
    rsync -az --timeout=30 --partial \
        -e "ssh $SSH_OPTS" \
        "root@$SSH_HOST:/workspace/pact/lane_d_results/" \
        "$LOCAL_DIR/" 2>/dev/null || echo "[$(date -u +%FT%TZ)] rsync skipped (instance unreachable)"

    # Surface progress: most-recent epoch line from train.log
    if [ -f "$LOCAL_DIR/train.log" ]; then
        LAST_EP=$(grep -oE '\[ep[ ]+[0-9]+/[0-9]+ P[0-9]+\]' "$LOCAL_DIR/train.log" 2>/dev/null | tail -1 || true)
        echo "[$(date -u +%FT%TZ)] backed up: ${LAST_EP:-no-epoch-yet}"
    fi
    sleep 60
done
