#!/bin/bash
set -uo pipefail
R=".omx/research/ddm_m1_20260808"
TK="$R/launch_ticket_v5_event_driven.json"
run_key() {  # execute the ticket's argv for KEY verbatim (no drift)
  .venv/bin/python -c "
import json, subprocess, sys
t = json.load(open('$TK'))
sys.exit(subprocess.run(t['$1']).returncode)"
}
echo "=== fp32 mem-probe ==="
.venv/bin/python -c "
import json, subprocess, sys
t = json.load(open('$TK'))
sys.exit(subprocess.run(t['mem_probe_command_fp32']).returncode)" || { echo FP32_PROBE_FAILED; exit 4; }
for i in 1 2 3 4 5; do
  K="argv_sigma_fp16_run${i}"
  .venv/bin/python tools/mx1_fire_guard.py --ticket "$TK" --argv-key "$K" \
    --out "$R/sigma/run_${i}/fire_guard_verdict.json" || { echo "GUARD_REFUSED_$K"; exit 5; }
  run_key "$K" || { echo "SIGMA_${K}_FAILED"; exit 6; }
done
K="argv_sigma_fp32_ref"
.venv/bin/python tools/mx1_fire_guard.py --ticket "$TK" --argv-key "$K" \
  --out "$R/sigma/run_fp32/fire_guard_verdict.json" || { echo "GUARD_REFUSED_$K"; exit 5; }
run_key "$K" || { echo "SIGMA_FP32_FAILED"; exit 7; }
echo "SIGMA_LEG_V2_COMPLETE"
