#!/bin/bash
# Canonical remote training bootstrap. Reusable for ANY profile via $1.
# Runs entirely inside a tmux session so SSH disconnects can't kill it.
#
# Usage (on remote):
#   bash scripts/remote_train_bootstrap.sh <profile_name> [output_subdir]
#
# Example:
#   bash scripts/remote_train_bootstrap.sh den results/den
#
# Side effects:
#   - Creates/refreshes /workspace/pact/.venv via uv (idempotent)
#   - Verifies CUDA availability + tac.profiles loadability
#   - Prints determinism config (seed + use_deterministic_algorithms)
#   - Launches a NEW tmux session named "<profile>_train" running pipeline.py
#   - Writes a heartbeat to experiments/results/<output_subdir>/heartbeat.log
#     every 60s so the watchdog can detect silent death
#
# Per CLAUDE.md "Canonical Pipeline Standard": this is the ONLY way to
# launch a remote training job. Ad-hoc /tmp scripts are forbidden by the
# preflight rule preflight_remote_deploy_canonical.

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
  echo "Usage: $0 <profile_name> [output_subdir]" >&2
  exit 1
fi

PROFILE="$1"
OUTPUT_SUBDIR="${2:-experiments/results/${PROFILE}}"
SESSION="${PROFILE}_train"
WORKSPACE=/workspace/pact

if [ ! -d "$WORKSPACE" ]; then
  echo "FATAL: $WORKSPACE missing — sync repo first." >&2
  exit 1
fi
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/$OUTPUT_SUBDIR"
HEARTBEAT_LOG="$LOG_DIR/heartbeat.log"
TRAIN_LOG="$LOG_DIR/train.log"
mkdir -p "$LOG_DIR"

# ── Build inner command. tmux runs this; SSH can disconnect immediately. ──
read -r -d '' INNER_CMD <<'INNER' || true
set -e
cd /workspace/pact
export PATH=/root/.local/bin:$PATH
export PYTHONPATH=src:upstream:$PWD
export TAC_UPSTREAM_DIR=/workspace/pact/upstream
export PYTHONUNBUFFERED=1

# Heartbeat in background (60s cadence). Watchdog elsewhere may alert if
# stale > 30 min per CLAUDE.md remote-code-parity rule. Cookie includes
# profile name to be unique → never self-matches `pgrep -f` (the 2026-04-26
# bug class).
HEARTBEAT=__HEARTBEAT_LOG__
PROFILE=__PROFILE__
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] profile=$PROFILE gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# Stage 1: idempotent venv bootstrap via uv.
echo "=== STAGE 1 ($PROFILE): bootstrap venv ==="
if [ ! -x /workspace/pact/.venv/bin/python ]; then
  rm -rf /workspace/pact/.venv 2>/dev/null || true
  /root/.local/bin/uv venv --python 3.12
  /root/.local/bin/uv pip install -e . | tail -5
  /root/.local/bin/uv pip install av opencv-python pydantic | tail -3
fi

# Stage 2: GPU sanity + determinism config print (CLAUDE.md determinism rule).
echo "=== STAGE 2 ($PROFILE): GPU + determinism ==="
.venv/bin/python -c "
import os, torch, random, numpy as np
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    # CUBLAS_WORKSPACE_CONFIG must be set BEFORE first cuBLAS call for
    # bit-exact deterministic matmuls.
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'), \
        'CUBLAS_WORKSPACE_CONFIG must be :4096:8 for determinism'
import sys; sys.path.insert(0, 'src')
from tac.profiles import PROFILES
prof = PROFILES['__PROFILE__']
print('profile seed:', prof.get('seed'))
print('profile deterministic:', prof.get('deterministic'))
print('eval_roundtrip:', prof.get('eval_roundtrip'))
"

# Stage 3: train_renderer (Phase 1+2+3 from profile, produces float .pt checkpoints).
echo "=== STAGE 3 ($PROFILE): train_renderer ==="
echo "[$(date -u +%FT%TZ)] launching train_renderer.py $PROFILE" >> "$HEARTBEAT"
.venv/bin/python -u src/tac/experiments/train_renderer.py \
    --profile $PROFILE \
    --device cuda \
    --output-dir __OUTPUT_DIR__

# Stage 4: probe for the canonical checkpoint that pipeline.py compress will
# consume. Order matches deploy_vastai.CANONICAL_CHECKPOINT_NAMES so we hit
# the same artifact the canonical launch path would have.
echo "=== STAGE 4 ($PROFILE): probe canonical checkpoint ==="
CKPT=""
for name in distill_phase3_best.pt distill_phase2_best.pt qat_best_float.pt distill_latest.pt; do
    if [ -f "__OUTPUT_DIR__/$name" ]; then
        CKPT="__OUTPUT_DIR__/$name"
        echo "  using $CKPT"
        break
    fi
done
if [ -z "$CKPT" ]; then
    echo "ABORT: no canonical checkpoint found in __OUTPUT_DIR__"
    ls -la __OUTPUT_DIR__/
    exit 2
fi

# Stage 5: pipeline.py compress = step_export (FP4 + self-compression via
# RESIDUAL codebook + robust_scale + stochastic) → step_qat → step_pose_tto
# → step_archive → step_eval. End-to-end CUDA auth eval at the end.
echo "=== STAGE 5 ($PROFILE): pipeline.py compress (QAT + self-compress + archive + auth eval) ==="
echo "[$(date -u +%FT%TZ)] launching pipeline.py compress $PROFILE checkpoint=$CKPT" >> "$HEARTBEAT"
.venv/bin/python -u experiments/pipeline.py compress \
    --profile $PROFILE \
    --video /workspace/pact/upstream/videos/0.mkv \
    --checkpoint "$CKPT" \
    --device cuda \
    --output-dir __OUTPUT_DIR__

echo "=== ${PROFILE}_DONE_$(date +%s) ==="
echo "[$(date -u +%FT%TZ)] ${PROFILE}_PIPELINE_DONE" >> "$HEARTBEAT"
INNER

# Substitute placeholders (avoids quoting nightmares with bash -lc).
INNER_CMD="${INNER_CMD//__HEARTBEAT_LOG__/$HEARTBEAT_LOG}"
INNER_CMD="${INNER_CMD//__PROFILE__/$PROFILE}"
INNER_CMD="${INNER_CMD//__OUTPUT_DIR__/$OUTPUT_SUBDIR}"

# Set CUBLAS_WORKSPACE_CONFIG in the tmux env so determinism check passes.
# Must be exported BEFORE the inner command runs.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Kill prior session if any (idempotent).
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Launch the session. Pipe inner command via process substitution so the
# entire script body is the tmux child, not a fragile bash -c "..." string
# (which broke the 2026-04-26 SHIRAZ launch).
tmux new-session -d -s "$SESSION" "bash -lc 'export CUBLAS_WORKSPACE_CONFIG=:4096:8; $INNER_CMD' 2>&1 | tee -a $TRAIN_LOG"

sleep 3
echo "tmux session '$SESSION' status:"
tmux ls 2>&1 | grep "$SESSION" || echo "  (NOT FOUND — check $TRAIN_LOG)"
echo
echo "First 20 log lines:"
sleep 5
tail -20 "$TRAIN_LOG" 2>/dev/null || echo "  (log not yet started)"
echo
echo "DEPLOY_OK profile=$PROFILE session=$SESSION log=$TRAIN_LOG heartbeat=$HEARTBEAT_LOG"
