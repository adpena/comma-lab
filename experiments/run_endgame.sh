#!/usr/bin/env bash
# Fire-and-forget endgame orchestrator.
# Runs phases 1-4 of the endgame checklist sequentially,
# waiting for free compute slots between experiments.
# Recoverable: each experiment saves best checkpoints + final weights.
# Results go to experiments/postfilter_weights/ with _best_meta.json.
#
# Usage: nohup bash experiments/run_endgame.sh > experiments/endgame.log 2>&1 &
#
# To monitor: tail -f experiments/endgame.log
# To cancel: kill $(cat /tmp/endgame.pid)
set -euo pipefail

cd /tmp/pact-mine
echo $$ > /tmp/endgame.pid

UV_ARGS="--with av --with torch --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy"
MAX_CONCURRENT=3

wait_for_slot() {
  while true; do
    running=$(pgrep -f "train_postfilter" 2>/dev/null | wc -l)
    if [ "$running" -lt "$MAX_CONCURRENT" ]; then
      echo "[endgame $(date +%H:%M)] slot free ($running running)"
      return 0
    fi
    sleep 300
  done
}

run_experiment() {
  local tag=$1; shift
  local script=$1; shift
  local logfile="experiments/postfilter_weights/train_${tag}.log"

  if [ -f "experiments/postfilter_weights/postfilter_${tag}_best_int8.pt" ] || \
     [ -f "experiments/postfilter_weights/postfilter_${tag}_int8.pt" ]; then
    echo "[endgame $(date +%H:%M)] $tag already done, skipping"
    return 0
  fi

  wait_for_slot
  echo "[endgame $(date +%H:%M)] launching $tag"
  PYTHONUNBUFFERED=1 uv run $UV_ARGS python -u "experiments/$script" "$@" --tag "$tag" > "$logfile" 2>&1 &
  local pid=$!
  echo "[endgame] $tag PID=$pid"
  # Don't wait — let it run in parallel and move to next experiment
  sleep 60
}

echo "=========================================="
echo "ENDGAME ORCHESTRATOR STARTED $(date)"
echo "=========================================="

# Phase 1: Architecture scaling
echo "--- Phase 1: Architecture scaling ---"
run_experiment psd_h48_long1000 train_postfilter_pixelshuffle_dilated.py --alpha 20 --hidden 48 --epochs 1000
run_experiment psd_h96_long1500 train_postfilter_pixelshuffle_dilated.py --alpha 20 --hidden 96 --epochs 1500

# Phase 2: Alpha sweep on PSD h=64
echo "--- Phase 2: Alpha sweep ---"
run_experiment psd_h64_a5 train_postfilter_pixelshuffle_dilated.py --alpha 5 --hidden 64 --epochs 1000
run_experiment psd_h64_a10 train_postfilter_pixelshuffle_dilated.py --alpha 10 --hidden 64 --epochs 1000

# Phase 2b: EMA decay sweep
echo "--- Phase 2b: EMA decay sweep ---"
run_experiment psd_h64_ema999 train_postfilter_pixelshuffle_dilated.py --alpha 20 --hidden 64 --epochs 1000 --ema-decay 0.999
run_experiment psd_h64_ema995 train_postfilter_pixelshuffle_dilated.py --alpha 20 --hidden 64 --epochs 1000 --ema-decay 0.995

# Phase 3: Extended training
echo "--- Phase 3: Extended training ---"
run_experiment psd_h64_long2000 train_postfilter_pixelshuffle_dilated.py --alpha 20 --hidden 64 --epochs 2000

# Wait for everything to finish
echo "[endgame $(date +%H:%M)] all experiments launched, waiting for completion..."
while pgrep -f "train_postfilter" > /dev/null 2>&1; do
  running=$(pgrep -f "train_postfilter" 2>/dev/null | wc -l)
  echo "[endgame $(date +%H:%M)] $running experiments still running..."
  sleep 600
done

echo "=========================================="
echo "ENDGAME COMPLETE $(date)"
echo "=========================================="

# Auto-triage: rank all best checkpoints
echo "--- Final triage ---"
for f in experiments/postfilter_weights/*_best_meta.json; do
  python3 -c "
import json
d = json.load(open('$f'))
tag = '$f'.split('/')[-1].replace('_best_meta.json','').replace('postfilter_','')
print(f'{tag:40s} ep={d[\"epoch\"]:4d} scorer={d[\"scorer\"]:.4f} int8={d[\"int8_size\"]:6d}')
" 2>/dev/null
done | sort -k3 -t= -n

rm -f /tmp/endgame.pid
echo "Done. Review results and proxy-score the best candidates."
