#!/bin/bash
# Run a command on Lightning AI Studio through SSH.
#
# Configure ~/.ssh/config with a Host such as "lightning-pact", then export
# LIGHTNING_SSH_TARGET to that alias.
set -euo pipefail

CMD="${1:?Usage: $0 \"command\"}"
TARGET="${LIGHTNING_SSH_TARGET:-lightning-pact}"
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

ssh "${SSH_OPTS[@]}" "$TARGET" "bash -lc $(printf '%q' "$CMD")"
