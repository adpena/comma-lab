#!/bin/bash
# Training watchdog — keeps experiments alive by restarting on crash
#
# Usage:
#   ./tools/training_watchdog.sh
#
# Checks every 60 seconds. Restarts dead trainers automatically.
# Runs ONE trainer at a time to avoid MPS memory pressure.
# Logs all restarts to /tmp/watchdog.log

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VENV=".venv/bin/python"
LOG="/tmp/watchdog.log"
CHECK_INTERVAL=60

# Primary experiment — runs first, always
# Check for resumable training state
RESUME_PATH="experiments/postfilter_weights/training_state_standard_h64_long2500.pt"
if [ -f "$RESUME_PATH" ]; then
    PRIMARY_CMD="$VENV -u experiments/train_postfilter_qat_ema.py --hidden 64 --epochs 2500 --alpha 20 --tag standard_h64_long2500"
    # TODO: wire --resume-from when the partner's trainer supports it
else
    PRIMARY_CMD="$VENV -u experiments/train_postfilter_qat_ema.py --hidden 64 --epochs 2500 --alpha 20 --tag standard_h64_long2500"
fi
PRIMARY_LOG="/tmp/standard_h64.log"
PRIMARY_TAG="standard_h64"

log() { echo "$(date '+%H:%M:%S') $1" | tee -a "$LOG"; }

log "Watchdog started. Checking every ${CHECK_INTERVAL}s."
log "Primary: $PRIMARY_TAG"

while true; do
    # Count active training processes
    ACTIVE=$(ps aux | grep python | grep -E 'train_postfilter|segnet_boundary' | grep -v grep | grep -v dashboard | grep -v watchdog | wc -l | tr -d ' ')

    if [ "$ACTIVE" -eq 0 ]; then
        log "⚠️ No trainers running. Restarting primary..."
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        nohup $PRIMARY_CMD > "$PRIMARY_LOG" 2>&1 &
        PID=$!
        log "✅ Restarted $PRIMARY_TAG (PID $PID)"

        # Wait a bit to confirm it stays alive
        sleep 15
        if ps -p $PID > /dev/null 2>&1; then
            log "✅ $PRIMARY_TAG confirmed alive"
        else
            log "❌ $PRIMARY_TAG died immediately — check $PRIMARY_LOG"
            # Wait longer before retry to avoid tight crash loop
            sleep 120
        fi
    else
        # Just log status quietly
        BEST=$(python3 -c "
import json, os
path = 'experiments/postfilter_weights/postfilter_standard_h64_long2500_best_meta.json'
if os.path.exists(path):
    d = json.load(open(path))
    print(f'ep {d[\"epoch\"]} scorer {d[\"scorer\"]:.4f}')
else:
    print('no checkpoint')
" 2>/dev/null || echo "?")
        log "✓ $ACTIVE trainer(s) alive. Best: $BEST"
    fi

    sleep "$CHECK_INTERVAL"
done
