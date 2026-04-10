#!/bin/bash
# Lightning AI training monitor — pull logs and checkpoints
# Usage: bash tools/lightning_monitor.sh [check|log|download|status]
set -euo pipefail

STUDIO="scratch-studio-devbox"
TS="adpena/default-project"
LIGHTNING=".venv/bin/lightning"
REMOTE="lit://${TS}/studios/${STUDIO}/tac"
LOCAL_DIR="lightning_results"

case "${1:-status}" in
  log)
    echo "=== Lightning train.log ==="
    $LIGHTNING cp "${REMOTE}/train.log" /tmp/lightning_train.log 2>/dev/null
    tail -20 /tmp/lightning_train.log
    ;;
  check)
    echo "=== Lightning checkpoint ==="
    $LIGHTNING cp "${REMOTE}/weights/" "${LOCAL_DIR}/weights/" 2>/dev/null
    ls -la "${LOCAL_DIR}/weights/"*best* 2>/dev/null || echo "No checkpoints yet"
    cat "${LOCAL_DIR}/weights/"*best_meta.json 2>/dev/null
    ;;
  download)
    echo "=== Downloading all results ==="
    mkdir -p "${LOCAL_DIR}"
    $LIGHTNING cp -r "${REMOTE}/weights/" "${LOCAL_DIR}/weights/" 2>/dev/null
    $LIGHTNING cp "${REMOTE}/train.log" "${LOCAL_DIR}/train.log" 2>/dev/null
    echo "Downloaded to ${LOCAL_DIR}/"
    ls -la "${LOCAL_DIR}/"
    ;;
  status)
    echo "=== Lightning Studio Status ==="
    $LIGHTNING list studios --teamspace "$TS" 2>/dev/null
    echo ""
    echo "=== Latest log ==="
    $LIGHTNING cp "${REMOTE}/train.log" /tmp/lightning_train.log 2>/dev/null
    tail -5 /tmp/lightning_train.log
    ;;
  restart)
    echo "=== Restarting training ==="
    # Upload fresh train.sh (in case we updated it)
    $LIGHTNING cp /tmp/tac_bundle/train.sh "${REMOTE}/train.sh" 2>/dev/null
    echo "Use the Lightning terminal to run: cd /teamspace/studios/this_studio/tac && nohup bash train.sh > train.log 2>&1 &"
    ;;
  *)
    echo "Usage: $0 [log|check|download|status|restart]"
    ;;
esac
