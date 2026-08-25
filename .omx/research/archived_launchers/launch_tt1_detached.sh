#!/usr/bin/env bash
# HISTORICAL_RECIPE_ONLY: frozen launch-command record for the completed 2026-07-30 lane; not a live deploy path (Catalog codebase-drift relocation 2026-08-25).
# ddm_tt1 QA71 — detached (ppid-1) joint payload gradient-TTO, all 3 ablation
# modes on the ~50-pair stratified subset (worst-joint pairs + controls),
# resumable.  tac-HIJACK guard + PATH export (the bare-python death).
# Axis [macOS advisory]; bounded scorers only (NOT n600); pointer UNMOVED.
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
export PATH="${REPO}/.venv/bin:$PATH"
export PYTHONPATH="${REPO}/src:${REPO}/experiments"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
       VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
       PYTORCH_ENABLE_MPS_FALLBACK=1

# tac-hijack guard (the shared-venv editable-install hijack from arm worktrees)
TACF="$(python -c 'import tac; print(tac.__file__)')"
if [ "$TACF" != "${REPO}/src/tac/__init__.py" ]; then
  echo "tac HIJACK: $TACF — refusing (export PYTHONPATH=${REPO}/src)" >&2
  exit 3
fi
echo "[tt1 launch] tac OK: $TACF"

DEV="${1:-mps}"
STEPS="${2:-24}"
LOG="/Volumes/VertigoDataTier/pact/ddm_tt1_20260731/run.log"
cd "$REPO"

for MODE in pose_only ab_only joint; do
  echo "[tt1 launch] === mode=$MODE device=$DEV steps=$STEPS $(date -u +%H:%M:%S) ==="
  python experiments/ddm_tt1_joint_tto.py \
    --mode "$MODE" --device "$DEV" \
    --n-hard 35 --n-ctrl 15 --steps "$STEPS" \
    --lrs 3e-4,1e-3,3e-3,1e-2 \
    --per-pair-seconds 120 --max-seconds 5400 --resume
done
echo "[tt1 launch] ALL MODES DONE $(date -u +%H:%M:%S)"
