#!/bin/bash
# Detached waiter: fire ng4's cell only after (a) ng4's owed B=16 smoke receipt exists and (b) cell_admission ADMITS a 45 GiB cell
# on THREE consecutive polls a minute apart (a single poll fires on the oscillation trough — ng4's rule). Logs every poll.
cd /Users/adpena/Projects/pact
S=/Volumes/APDataStore/pact/ddm_ng4_continuous_objective; LOG=$S/fire/wait_then_fire.log; HOLD=0; N=0
while :; do N=$((N+1)); T=$(date -u +%H:%M:%SZ)
  if [ ! -e .omx/tmp/codex_runs/NG4_SMOKE_WAITER4_DONE.json.done ]; then echo "$T poll $N: smoke receipt absent; hold=0" >> $LOG; HOLD=0; sleep 60; continue; fi
  if .venv/bin/python tools/cell_admission.py admit --candidate-peak-gib 45 >/dev/null 2>&1; then HOLD=$((HOLD+1)); echo "$T poll $N: ADMIT hold=$HOLD/3" >> $LOG; else HOLD=0; echo "$T poll $N: refuse; hold=0" >> $LOG; fi
  [ "$HOLD" -ge 3 ] && break; sleep 60
done
echo "$(date -u +%H:%M:%SZ) firing" >> $LOG
bash $S/fire/fire_ng4_continuous_cell.sh >> $LOG 2>&1; RC=$?; echo "$(date -u +%H:%M:%SZ) fire rc=$RC" >> $LOG; exit $RC
