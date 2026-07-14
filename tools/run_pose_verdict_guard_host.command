#!/bin/zsh
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="/Users/adpena/Projects/pact"
cd "$ROOT"

# $0 structural verification only. The old dry-start launcher was retired because
# it substituted an unbound R1 advisory scalar into the current score path.
exec .venv/bin/python -m pytest -q \
  src/tac/witness_control/tests/test_pose_verdict_gate.py \
  src/tac/tests/test_pose_verdict_gate_trainer_wirein.py
