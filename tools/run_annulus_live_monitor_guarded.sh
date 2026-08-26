#!/usr/bin/env bash
# Guarded LIVE annulus-convergence monitor. Mirrors run_probe_guarded.sh: a memory
# watchdog kills ONLY the monitor (and its render child via the process group) if free
# RAM drops below the floor -- it can NEVER push the box over the P0 machine-crash edge,
# and it NEVER touches the live trainer (pid unrelated).
#
# NON-INVASIVE + READ-ONLY on the run's checkpoints/log. numpy-fp32 advisory telemetry;
# pointer 0.19110 UNMOVED. The monitor renders through-R at a small --pairs (default 16),
# ~5-6GB transient in the child, released on child exit.
#
# USAGE:
#   ONE bounded smoke:   tools/run_annulus_live_monitor_guarded.sh --once
#   Continuous loop:     tools/run_annulus_live_monitor_guarded.sh          (ticks every INTERVAL_SEC)
#   env overrides: RUN_DIR, GT_CACHE, PAIRS, FLOOR_MB, INTERVAL_SEC, THREADS
set -euo pipefail
cd /Users/adpena/Projects/pact

RUN_DIR="${RUN_DIR:-experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z}"
GT_CACHE="${GT_CACHE:-experiments/results/mlx_fleet_gt_cache/gt_n600.npz}"
PAIRS="${PAIRS:-16}"
FLOOR_MB="${FLOOR_MB:-6144}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
THREADS="${THREADS:-4}"
GUARD_LOG="$RUN_DIR/annulus_live_monitor_guard.log"

ONCE=0
if [ "${1:-}" = "--once" ] || [ "${ONCE_MODE:-0}" = "1" ]; then ONCE=1; fi

run_one_guarded() {
  # Launch the monitor as a NEW SESSION (process-group leader) so the watchdog can kill
  # the whole group (monitor + its render subprocess) without reaching any other process.
  setsid .venv/bin/python tools/witness_annulus_live_monitor.py \
    --run-dir "$RUN_DIR" \
    --gt-cache "$GT_CACHE" \
    --pairs "$PAIRS" \
    --threads "$THREADS" \
    --once >> "$GUARD_LOG" 2>&1 &
  local MON=$!
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) monitor pgid $MON (floor ${FLOOR_MB}MB, pairs ${PAIRS})" | tee -a "$GUARD_LOG"
  while kill -0 "$MON" 2>/dev/null; do
    AVAIL=$(.venv/bin/python -c "from tools.mem_basis import conservative_free_gib as f; g=f(default=float('inf'));print(99999 if g==float('inf') else int(g*1024))" 2>/dev/null || echo 99999)
    if [ "$AVAIL" -lt "$FLOOR_MB" ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WATCHDOG: avail ${AVAIL}MB < ${FLOOR_MB}MB — killing monitor group -$MON to protect the box" | tee -a "$GUARD_LOG"
      kill -9 -- "-$MON" 2>/dev/null || true
      break
    fi
    sleep 4
  done
  wait "$MON" 2>/dev/null || true
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) monitor tick done; avail now $(.venv/bin/python -c 'from tools.mem_basis import conservative_free_gib as f; g=f(default=float('inf'));print(99999 if g==float('inf') else int(g*1024))')MB" | tee -a "$GUARD_LOG"
}

if [ "$ONCE" = "1" ]; then
  run_one_guarded
else
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous guarded monitor loop (every ${INTERVAL_SEC}s)" | tee -a "$GUARD_LOG"
  while true; do
    run_one_guarded
    sleep "$INTERVAL_SEC"
  done
fi
