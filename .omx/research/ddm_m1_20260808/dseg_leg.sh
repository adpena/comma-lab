#!/bin/bash
set -uo pipefail
TK=".omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json"
for K in argv_dseg_verdict_fp16 argv_dseg_verdict_fp32; do
  .venv/bin/python -c "
import json,subprocess,sys
sys.exit(subprocess.run(json.load(open('$TK'))['$K']).returncode)" || { echo "FAILED_$K"; exit 5; }
done
echo DSEG_LEG_COMPLETE
