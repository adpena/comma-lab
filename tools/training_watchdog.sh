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

# Primary experiment — runs first, always.
# train_postfilter_qat_ema.py is a partner-owned trainer; --resume-from is
# not yet supported on its CLI (upstream API stable). When upstream adds
# --resume-from FILE, the if-block here can be enabled. Until then the
# both branches must produce the same command — keeping the if-else as
# a placeholder breeds confusion. Removed; resume-detection will return
# when partner adds support.
PRIMARY_CMD="$VENV -u experiments/train_postfilter_qat_ema.py --hidden 64 --epochs 2500 --alpha 20 --tag standard_h64_long2500"
PRIMARY_LOG="/tmp/standard_h64.log"
PRIMARY_TAG="standard_h64"

log() { echo "$(date '+%H:%M:%S') $1" | tee -a "$LOG"; }

log "Watchdog started. Checking every ${CHECK_INTERVAL}s."
log "Primary: $PRIMARY_TAG"

while true; do
    # Count active training processes
    ACTIVE=$(ps aux | grep python | grep -E 'train_postfilter|segnet_boundary' | grep -v grep | grep -v dashboard | grep -v watchdog | wc -l | tr -d ' ')

    if [ "$ACTIVE" -eq 0 ]; then
        log "⚠️ No trainers running. Restarting primary in tmux..."
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        # Per CLAUDE.md "Always tmux": NEVER use nohup&. All 3 watchers using
        # nohup died on 2026-04-25, wasting hours of A100 time. Use tmux so
        # the session survives + we can attach for debugging.
        SESSION="watchdog_${PRIMARY_TAG}"
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        tmux new-session -d -s "$SESSION" \
            "$PRIMARY_CMD > $PRIMARY_LOG 2>&1; echo 'WATCHDOG_DONE_'\$(date +%s) >> $PRIMARY_LOG"
        log "✅ Restarted $PRIMARY_TAG in tmux session '$SESSION'"

        # Wait a bit to confirm it stays alive (process AND session both must exist)
        sleep 15
        if tmux has-session -t "$SESSION" 2>/dev/null && \
           pgrep -f "train_postfilter_qat_ema.*$PRIMARY_TAG" > /dev/null; then
            log "✅ $PRIMARY_TAG confirmed alive (tmux session + process both running)"
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
