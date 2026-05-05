#!/bin/bash
# Lightning AI training monitor through SSH.
# Usage: bash tools/lightning_monitor.sh [check|log|download|status]
set -euo pipefail

TARGET="${LIGHTNING_SSH_TARGET:-}"
REMOTE="${LIGHTNING_REMOTE_TAC:-/teamspace/studios/this_studio/tac}"
LOCAL_DIR="lightning_results"
if [ -z "$TARGET" ]; then
  echo "FATAL: set LIGHTNING_SSH_TARGET to a user-qualified Studio SSH target or SSH config alias" >&2
  exit 2
fi
case "$TARGET" in
  ssh.lightning.ai)
    echo "FATAL: use a user-qualified target or SSH config alias, not bare ssh.lightning.ai" >&2
    exit 2
    ;;
esac
SSH_OPTS=(
  -o BatchMode=yes
  -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no
  -o ConnectTimeout=15
  -o ConnectionAttempts=3
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  -o TCPKeepAlive=yes
)

_ssh() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "$@"
}

_scp() {
  scp -q "${SSH_OPTS[@]}" "$@"
}

case "${1:-status}" in
  log)
    echo "=== Lightning train.log ==="
    _scp "${TARGET}:${REMOTE}/train.log" /tmp/lightning_train.log 2>/dev/null
    tail -20 /tmp/lightning_train.log
    ;;
  check)
    echo "=== Lightning checkpoint ==="
    mkdir -p "${LOCAL_DIR}/weights"
    _scp -r "${TARGET}:${REMOTE}/weights/" "${LOCAL_DIR}/" 2>/dev/null
    ls -la "${LOCAL_DIR}/weights/"*best* 2>/dev/null || echo "No checkpoints yet"
    cat "${LOCAL_DIR}/weights/"*best_meta.json 2>/dev/null
    ;;
  download)
    echo "=== Downloading all results ==="
    mkdir -p "${LOCAL_DIR}"
    _scp -r "${TARGET}:${REMOTE}/weights/" "${LOCAL_DIR}/" 2>/dev/null
    _scp "${TARGET}:${REMOTE}/train.log" "${LOCAL_DIR}/train.log" 2>/dev/null
    echo "Downloaded to ${LOCAL_DIR}/"
    ls -la "${LOCAL_DIR}/"
    ;;
  status)
    echo "=== Lightning Studio Status ==="
    _ssh "hostname && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true"
    echo ""
    echo "=== Latest log ==="
    _scp "${TARGET}:${REMOTE}/train.log" /tmp/lightning_train.log 2>/dev/null
    tail -5 /tmp/lightning_train.log
    ;;
  restart)
    echo "=== Restarting training ==="
    # Upload fresh train.sh (in case we updated it)
    _scp /tmp/tac_bundle/train.sh "${TARGET}:${REMOTE}/train.sh" 2>/dev/null
    echo "Run remotely: cd ${REMOTE} && nohup bash train.sh > train.log 2>&1 &"
    ;;
  *)
    echo "Usage: $0 [log|check|download|status|restart]"
    ;;
esac
