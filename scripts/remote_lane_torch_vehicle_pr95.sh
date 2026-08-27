#!/usr/bin/env bash
# Remote driver for the P2 torch-vehicle (vendored PR95 HNeRV-Muon) resumable run.
#
# The ONLY Modal/CUDA-runnable, basin-proven capstone vehicle. Drives the
# faithful PR95 8-stage curriculum at a SMALLER basis (base_ch=20, the rate-win
# config) with COMPLETE resume: a SIGKILL/OOM/Modal-preempt loses <= one
# checkpoint interval, and a re-launch of THIS script resumes the EXACT
# trajectory from the durable checkpoint in OUT_DIR (the "Durable detached
# daemons" + "LONG RESUMABLE SATURATION SWEEPS" non-negotiables).
#
# Usage (re-launch-safe; resumes if OUT_DIR has a checkpoint):
#   OUT_DIR=experiments/results/torch_vehicle_n600_bc20 \
#   BASE_CHANNELS=20 TOTAL_EPOCH_BUDGET=12000 DEVICE=cuda \
#   scripts/remote_lane_torch_vehicle_pr95.sh
#
# Authority: in-loop d_seg/d_pose are [contest-CPU advisory] NON-PROMOTABLE; the
# leaderboard score is authoritative ONLY after upstream/evaluate.py on the
# byte-closed OUT_DIR/best/best_archive.bin. NO MPS (DEVICE must be cpu|cuda).
#
# Heartbeat + done-marker: the driver writes torch_vehicle_summary.json every
# epoch and torch_vehicle_run.DONE on completion. A watchdog can tail those.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:?set OUT_DIR (the resumable run dir)}"
BASE_CHANNELS="${BASE_CHANNELS:-20}"
LATENT_DIM="${LATENT_DIM:-28}"
TOTAL_EPOCH_BUDGET="${TOTAL_EPOCH_BUDGET:-}"  # empty = full 29,650 PR95 curriculum
EMA_DECAY="${EMA_DECAY:-0.999}"               # PR95-faithful
EVAL_EVERY="${EVAL_EVERY:-25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
PYBIN="${PYBIN:-.venv/bin/python}"

# NO MPS — fail closed on a misconfigured device (sister of the driver guard).
if [[ "$DEVICE" == mps* ]]; then
  echo "ERROR: MPS is NEVER trusted (CLAUDE.md). DEVICE must be cpu or cuda." >&2
  exit 2
fi

# Modal/CUDA NVML hygiene (sister of Catalog #244 canonical 3-export block).
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="src:upstream:${PYTHONPATH:-}"

# Stage 0: NVDEC/CUDA preflight (fail-closed before any GPU-cost stage).
echo "=== Stage 0: NVDEC and CUDA preflight ==="
_nvdec_probe="${WORKSPACE:-$PWD}/scripts/probe_nvdec.sh"
[ -f "$_nvdec_probe" ] || _nvdec_probe="$(cd "$(dirname "$0")/.." && pwd)/scripts/probe_nvdec.sh"
bash "$_nvdec_probe" || { echo "FATAL: NVDEC probe failed" >&2; exit 2; }
# The vendored data.py resolves the frozen SegNet/PoseNet here.
export COMMA_CHALLENGE_ROOT="${COMMA_CHALLENGE_ROOT:-$REPO_ROOT/upstream}"

mkdir -p "$OUT_DIR"

# If a previous run already completed, this is a no-op (the DONE marker idempotency).
if [[ -f "$OUT_DIR/torch_vehicle_run.DONE" ]]; then
  echo "[torch-vehicle] $OUT_DIR already DONE; nothing to do."
  exit 0
fi

BUDGET_FLAG=()
[[ -n "$TOTAL_EPOCH_BUDGET" ]] && BUDGET_FLAG=(--total-epoch-budget "$TOTAL_EPOCH_BUDGET")

# Provenance (feedback_canonical_remote_bootstraps): every remote run emits
# provenance.json so a fresh agent can reconstruct the experiment. Written at
# each (re)launch; the trainer's per-epoch torch_vehicle_summary.json is the
# run_record and torch_vehicle_run.DONE is the completion marker.
GIT_HASH=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo no-git)
cat > "$OUT_DIR/provenance.json" <<EOF
{
  "predicted_band": ${PREDICTED_BAND:-null},
  "schema": "remote_run_provenance.v1",
  "started_at_utc": "$(date -u +%FT%TZ)",
  "git_hash": "$GIT_HASH",
  "lane_script": "scripts/remote_lane_torch_vehicle_pr95.sh",
  "base_channels": "$BASE_CHANNELS",
  "latent_dim": "$LATENT_DIM",
  "total_epoch_budget": "${TOTAL_EPOCH_BUDGET:-full}",
  "ema_decay": "$EMA_DECAY",
  "device": "$DEVICE",
  "seed": "$SEED",
  "out_dir": "$OUT_DIR"
}
EOF

echo "[torch-vehicle] launching/resuming run in $OUT_DIR"
echo "  base_channels=$BASE_CHANNELS ema_decay=$EMA_DECAY device=$DEVICE budget=${TOTAL_EPOCH_BUDGET:-full}"

exec "$PYBIN" -u -m tac.torch_vehicle.run \
  --base-channels "$BASE_CHANNELS" \
  --latent-dim "$LATENT_DIM" \
  --ema-decay "$EMA_DECAY" \
  --eval-every "$EVAL_EVERY" \
  --checkpoint-every-epochs "$CHECKPOINT_EVERY" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  "${BUDGET_FLAG[@]}"
