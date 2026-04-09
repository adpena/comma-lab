#!/usr/bin/env bash
# Karpathy-style principled experimental sweep.
#
# Based on the SVD finding (rank ~1) and the CNN residual analysis (dense
# mid-frequency corrections), we run a minimal set of informative
# experiments rather than a brute-force grid.
#
# Each experiment is 1000 epochs on MPS, using the winning QAT+EMA recipe
# unless noted otherwise. Results go to experiments/postfilter_weights/
# and are auto-proxied once complete.
#
# The sweep focuses on three axes Karpathy would actually test:
#
# 1. Width scaling curve:       h=48, h=64 (confirm LeCun's curve)
# 2. Alternative averaging:     Kalman vs EMA (user's idea, live)
# 3. Activation-aware training: uint8 STE (audit YELLOW fix, live)
# 4. Scorer-faithful loss:      SegNet attack (Tao's leverage, live)
# 5. DCT-basis parametrization: explicit mid-frequency prior (live)
# 6. Alpha sweep:               α=30, α=40 (does more saliency help?)
# 7. Long-epoch:                2000 epochs (does the scaling hold?)
#
# Items 2-5 are already running. This script launches 1, 6, 7 as
# compute frees up.
#
# Usage: bash experiments/run_karpathy_sweep.sh

set -euo pipefail

cd /tmp/pact-mine

UV_ARGS="--with av --with torch --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy"

wait_for_free_slot() {
  while true; do
    running=$(pgrep -f "train_postfilter" | wc -l)
    if [ "$running" -lt 4 ]; then
      echo "[sweep] $running trainers running, slot free"
      return 0
    fi
    echo "[sweep] $running trainers running, waiting..."
    sleep 300
  done
}

launch() {
  local tag=$1; shift
  local script=$1; shift
  local log="experiments/postfilter_weights/train_${tag}.log"
  if [ -f "experiments/postfilter_weights/postfilter_${tag}_int8.pt" ]; then
    echo "[sweep] $tag already done, skipping"
    return 0
  fi
  wait_for_free_slot
  echo "[sweep] launching $tag"
  PYTHONUNBUFFERED=1 uv run $UV_ARGS python -u "experiments/$script" "$@" --tag "$tag" > "$log" 2>&1 &
  disown
  sleep 30
}

# Width scaling
launch long1000_h48 train_postfilter_qat_ema.py --alpha 20 --hidden 48 --epochs 1000
launch long1000_h64 train_postfilter_qat_ema.py --alpha 20 --hidden 64 --epochs 1000

# Alpha sweep on the winning h=32 recipe
launch long1000_h32_a30 train_postfilter_qat_ema.py --alpha 30 --hidden 32 --epochs 1000
launch long1000_h32_a40 train_postfilter_qat_ema.py --alpha 40 --hidden 32 --epochs 1000

# Extended training (the scaling law extrapolation)
launch long2000_h32 train_postfilter_qat_ema.py --alpha 20 --hidden 32 --epochs 2000

echo "[sweep] all launches queued"
