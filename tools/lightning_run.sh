#!/bin/bash
# Run a command on Lightning AI studio headlessly via SSH pipe.
# Usage: bash tools/lightning_run.sh "command to run"
# Examples:
#   bash tools/lightning_run.sh "tail -5 /teamspace/studios/this_studio/tac/train.log"
#   bash tools/lightning_run.sh "nvidia-smi"
#   bash tools/lightning_run.sh "cat /teamspace/studios/this_studio/tac/weights/*best_meta.json"
set -euo pipefail

CMD="${1:?Usage: $0 \"command\"}"
echo "$CMD" | .venv/bin/lightning connect studio \
  --name scratch-studio-devbox \
  --teamspace adpena/default-project 2>&1 | grep -v "SSH\|Warning\|Pseudo\|key already"
