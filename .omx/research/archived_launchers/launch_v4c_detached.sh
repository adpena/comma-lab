#!/usr/bin/env bash
# HISTORICAL_RECIPE_ONLY: frozen launch-command record for the completed 2026-07-30 lane; not a live deploy path (Catalog codebase-drift relocation 2026-08-25).
# ddm_v4c detached launcher — runs ddm_v4c_resolve.py fully detached (ppid-1)
# so the SIGURG-orphan class (tracked-bg death at ~3 min) cannot kill it.
# PATH + PYTHONPATH exported INSIDE (bare-python death + tac-hijack guards).
# Usage: bash experiments/launch_v4c_detached.sh <label> -- <resolve args...>
set -euo pipefail
cd /Users/adpena/Projects/pact
export PATH="/Users/adpena/Projects/pact/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/upstream:$PWD/experiments"
LABEL="$1"; shift
[ "$1" = "--" ] && shift
LOG="/Volumes/VertigoDataTier/pact/ddm_v4c_20260730/${LABEL}.log"
mkdir -p "$(dirname "$LOG")"
# tac-hijack guard: assert main src is on the resolve path
python -c "import sys; sys.path.insert(0,'src'); import tac; assert '/src/tac/' in tac.__file__, tac.__file__; print('tac OK', tac.__file__)"
nohup python experiments/ddm_v4c_resolve.py "$@" >>"$LOG" 2>&1 < /dev/null &
disown || true
echo "launched: $LABEL -> $LOG (pid $!)"
